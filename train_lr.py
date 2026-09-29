"""Standalone trainer with configurable maximum learning rate.

This keeps the original train.py untouched and otherwise follows its data,
optimizer, warmup/cosine schedule, validation, and checkpoint conventions.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", default="combined_model")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--max-lr", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.warmup_steps < 1:
        parser.error("steps, batch-size, and warmup-steps must be positive")
    if args.max_lr <= 0 or args.weight_decay < 0:
        parser.error("max-lr must be positive and weight-decay cannot be negative")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("run directory already contains results; choose a new --run-dir")

    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr,
                                  weight_decay=args.weight_decay)
    tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared

    started = time.perf_counter()
    history, validation_history = [], []
    intermediate_validation_seconds = 0.0
    for step in range(args.steps):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        warmup = min(1.0, (step + 1) / args.warmup_steps)
        cosine = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / args.steps))
        learning_rate = args.max_lr * warmup * cosine
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1).float(),
                                   batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {"step": step + 1, "loss": loss.item(),
                   "seconds": time.perf_counter() - started - intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            validation = score(model, *data["validation"], device, "fp32")
            validation.pop("window_nll_nats")
            intermediate_validation_seconds += validation["seconds"]
            validation_history.append({"step": step + 1, **validation})
            print(json.dumps({"validation": validation_history[-1]}), flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - intermediate_validation_seconds
    validation = score(model, *data["validation"], device, "fp32")
    validation.pop("window_nll_nats")
    checkpoint = args.run_dir / "checkpoint.pt"
    torch.save({"protocol": PROTOCOL, "implementation": args.implementation,
                "config": config, "model": model.cpu().state_dict(), "seed": args.seed,
                "train_tokens": args.steps * args.batch_size * 256}, checkpoint)
    recipe = {"max_lr": args.max_lr, "warmup_steps": args.warmup_steps,
              "weight_decay": args.weight_decay, "steps": args.steps,
              "batch_size": args.batch_size, "minimum_lr_ratio": 0.1}
    result = {"protocol": PROTOCOL, "implementation": args.implementation,
              "config": config, "seed": args.seed, "parameters": sum(p.numel() for p in model.parameters()),
              "precision": precision, "train_tokens": args.steps * args.batch_size * 256,
              "preparation_seconds": preparation_seconds, "train_seconds": train_seconds,
              "training_recipe": recipe, "validation": validation, "history": history,
              "validation_history": validation_history,
              "intermediate_validation_seconds": intermediate_validation_seconds,
              "process_seconds": time.perf_counter() - total_started,
              "torch_version": str(torch.__version__), "threads": args.threads,
              "checkpoint_sha256": sha(checkpoint),
              "implementation_sha256": implementation_sha, **device_metrics(device)}
    (args.run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result | {"history": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
