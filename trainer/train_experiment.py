import argparse
import json
import math
from pathlib import Path
import re
import sys
import time

import torch
from torch.nn import functional as F

# Make imports independent of whether this is launched as a script or module.
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common import (PROTOCOL, ROOT, autocast, device_metrics, load_data,
                    make_model, setup, sha)
from evaluate import score


def safe_part(value, label):
    value = str(value).replace("\\", "/").strip("/")
    parts = value.split("/")
    if not value or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Invalid {label}: {value!r}")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts):
        raise ValueError(f"Invalid {label}: {value!r}")
    return Path(*parts)


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def checkpoint_payload(implementation, config, model, seed, train_tokens,
                       best_step=None):
    payload = {
        "protocol": PROTOCOL,
        "implementation": implementation,
        "config": config,
        "model": {name: value.detach().cpu().clone()
                  for name, value in model.state_dict().items()},
        "seed": seed,
        "train_tokens": train_tokens,
    }
    if best_step is not None:
        payload["best_step"] = best_step
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="Experiment profile under code/configs/experiments")
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"])
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--max-lr", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--min-lr-ratio", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--run-name", help="Optional name for a repeated run")
    parser.add_argument("--runs-root", type=Path,
                        help="Optional output root; defaults to code/runs")
    args = parser.parse_args()

    profile_path = args.config.resolve()
    profile = load_json(profile_path)
    training = dict(profile.get("training", {}))
    runtime = dict(profile.get("runtime", {}))
    overrides = {
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_every": args.eval_every,
        "max_lr": args.max_lr,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "weight_decay": args.weight_decay,
    }
    for key, value in overrides.items():
        if value is not None:
            training[key] = value
    seed = int(training.get("seed", 17))
    steps = int(training.get("steps", 1200))
    batch_size = int(training.get("batch_size", 32))
    eval_every = int(training.get("eval_every", 300))
    max_lr = float(training.get("max_lr", 0.001))
    warmup_steps = int(training.get("warmup_steps", 100))
    min_lr_ratio = float(training.get("min_lr_ratio", 0.01))
    weight_decay = float(training.get("weight_decay", 0.1))
    if steps < 1 or batch_size < 1 or warmup_steps < 1 or eval_every < 0:
        parser.error("steps, batch-size, and warmup-steps must be positive; eval-every cannot be negative")
    if max_lr <= 0 or weight_decay < 0 or not 0 <= min_lr_ratio <= 1:
        parser.error("Require max-lr > 0, weight-decay >= 0, and min-lr-ratio in [0, 1]")

    architecture_path = (ROOT / profile["architecture"]).resolve()
    try:
        architecture_path.relative_to(ROOT.resolve())
    except ValueError:
        parser.error("architecture path must remain inside the code directory")
    model_config = load_json(architecture_path)
    implementation = profile["implementation"]
    run_group = safe_part(profile.get("run_group", "experiments"), "run_group")
    experiment_name = safe_part(args.run_name or profile["name"], "run_name")
    run_dir = Path(args.runs_root) if args.runs_root else ROOT / "runs"
    run_dir = run_dir.resolve() / run_group / experiment_name / f"seed-{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"Run directory already contains files: {run_dir}; choose --run-name")

    device_name = args.device or runtime.get("device", "cuda")
    precision_name = args.precision or runtime.get("precision", "auto")
    threads = args.threads or int(runtime.get("threads", 4))
    device, precision = setup(device_name, precision_name, threads)
    torch.manual_seed(seed)

    prepared = time.perf_counter()
    data = load_data()
    model, implementation_sha = make_model(implementation, model_config, device)
    run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr,
                                  weight_decay=weight_decay)
    tokens = data["train"][0].to(device)
    rng = torch.Generator().manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared

    started = time.perf_counter()
    history, validation_history = [], []
    validation_seconds = 0.0
    best_bpb, best_validation, best_step, best_state = float("inf"), None, 0, None
    train_tokens = steps * batch_size * 256

    for step in range(steps):
        starts = torch.randint(len(tokens) - 257, (batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        warmup = min(1.0, (step + 1) / warmup_steps)
        progress = step / steps
        cosine = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        learning_rate = max_lr * warmup * cosine
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(
                model(batch[:, :-1]).flatten(0, 1).float(),
                batch[:, 1:].flatten(),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 100 == 0 or step + 1 == steps:
            row = {"step": step + 1, "loss": loss.item(),
                   "seconds": time.perf_counter() - started - validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)

        if eval_every and (step + 1) % eval_every == 0:
            val = score(model, *data["validation"], device, "fp32")
            val.pop("window_nll_nats")
            validation_seconds += val["seconds"]
            row = {"step": step + 1, **val}
            validation_history.append(row)
            if val["bpb"] < best_bpb:
                best_bpb, best_validation, best_step = val["bpb"], row, step + 1
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            print(json.dumps({"validation": row}), flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - validation_seconds
    final_val = score(model, *data["validation"], device, "fp32")
    final_val.pop("window_nll_nats")
    if final_val["bpb"] < best_bpb:
        best_bpb = final_val["bpb"]
        best_validation = {"step": steps, **final_val}
        best_step = steps
        best_state = {name: value.detach().cpu().clone()
                      for name, value in model.state_dict().items()}

    torch.save(checkpoint_payload(implementation, model_config, model, seed,
                                  train_tokens), run_dir / "checkpoint.pt")
    best_payload = checkpoint_payload(implementation, model_config, model, seed,
                                      train_tokens, best_step)
    best_payload["model"] = best_state
    torch.save(best_payload, run_dir / "best_checkpoint.pt")
    (run_dir / "experiment.json").write_text(
        json.dumps({"profile": profile, "training": training,
                    "runtime": {"device": str(device), "precision": precision,
                                "threads": threads}}, indent=2) + "\n",
        encoding="utf-8",
    )

    result = {
        "protocol": PROTOCOL,
        "implementation": implementation,
        "config": model_config,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "precision": precision,
        "train_tokens": train_tokens,
        "preparation_seconds": preparation_seconds,
        "train_seconds": train_seconds,
        "training_recipe": {"max_lr": max_lr, "warmup_steps": warmup_steps,
                            "minimum_lr_ratio": min_lr_ratio,
                            "weight_decay": weight_decay, "steps": steps,
                            "batch_size": batch_size},
        "validation": final_val,
        "best_validation": best_validation,
        "best_step": best_step,
        "history": history,
        "validation_history": validation_history,
        "intermediate_validation_seconds": validation_seconds,
        "process_seconds": time.perf_counter() - prepared,
        "torch_version": str(torch.__version__),
        "threads": threads,
        "checkpoint_sha256": sha(run_dir / "checkpoint.pt"),
        "best_checkpoint_sha256": sha(run_dir / "best_checkpoint.pt"),
        "implementation_sha256": implementation_sha,
        "run_dir": str(run_dir),
        **device_metrics(device),
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(result | {"history": []}, indent=2), flush=True)


if __name__ == "__main__":
    main()
