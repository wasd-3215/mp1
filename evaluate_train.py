"""Score a saved checkpoint on the official training split using the course BPB metric.

This is a separate diagnostic utility. It does not modify the trainer or the
official evaluator, and it never loads or scores validation/test text.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from MP1_student_starter.code.common import PROTOCOL, ROOT, autocast, device_metrics, make_model, setup, sha, windows


def load_train_data():
    """Load only the fixed tokenizer and verified train text."""
    manifest_path = ROOT / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Dataset manifest belongs to a different course protocol.")

    tokenizer_path = ROOT / "data" / "tokenizer.json"
    train_path = ROOT / "data" / "wikitext_train.txt"
    expected = manifest["sha256"]
    for name, path in (("tokenizer.json", tokenizer_path),
                       ("wikitext_train.txt", train_path)):
        if sha(path) != expected[name]:
            raise ValueError(f"Changed benchmark file: {name}")

    raw = train_path.read_bytes()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = tokenizer.encode(raw.decode("utf-8")).ids
    return torch.tensor(ids, dtype=torch.long), len(raw), sha(tokenizer_path)


@torch.no_grad()
def score_train(model, tokens, byte_count, device, precision, batch_size):
    """Use the same fixed non-overlapping windows and byte-normalized NLL as evaluate.py."""
    previous_mode = model.training
    model.eval()
    started = time.perf_counter()
    nll = 0.0
    count = 0
    window_nll = []

    for x, y in windows(tokens, batch_size):
        x, y = x.to(device), y.to(device)
        with autocast(device, precision):
            logp = model.predict_log_probs(x).float()
        if logp.shape != (*x.shape, 2048) or not torch.isfinite(logp).all():
            raise ValueError("Model must return finite log probabilities shaped [batch, time, 2048].")
        if torch.logsumexp(logp, dim=-1).abs().max().item() > 1e-3:
            raise ValueError("Model output is not a normalized log-probability distribution.")

        losses = -logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        losses.masked_fill_(y == -100, 0)
        per_window = losses.double().sum(-1).cpu().tolist()
        window_nll.extend(per_window)
        nll += sum(per_window)
        count += (y != -100).sum().item()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model.train(previous_mode)
    return {
        "bpb": nll / math.log(2) / byte_count,
        "token_ppl": math.exp(nll / count),
        "nll_nats": nll,
        "targets": count,
        "utf8_bytes": byte_count,
        "seconds": time.perf_counter() - started,
        "window_nll_nats": window_nll,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "auto"], default="fp32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    device, precision = setup(args.device, args.precision, args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["protocol"] != PROTOCOL:
        raise ValueError("Checkpoint belongs to a different course protocol.")
    model, implementation_sha = make_model(
        checkpoint["implementation"], checkpoint["config"], device
    )
    model.load_state_dict(checkpoint["model"])

    tokens, byte_count, tokenizer_sha = load_train_data()
    result = score_train(model, tokens, byte_count, device, precision, args.batch_size)
    window_losses = result.pop("window_nll_nats")
    output = args.output or args.checkpoint.parent / f"train_{device.type}_{precision}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output.with_suffix(".window-nll.npy"), np.asarray(window_losses))
    result.update(
        protocol=PROTOCOL,
        split="train",
        precision=precision,
        implementation=checkpoint["implementation"],
        config=checkpoint["config"],
        checkpoint_best_step=checkpoint.get("best_step"),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        checkpoint_sha256=sha(args.checkpoint),
        implementation_sha256=implementation_sha,
        evaluator_sha256=sha(Path(__file__)),
        tokenizer_sha256=tokenizer_sha,
        **device_metrics(device),
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
