# Attention-mechanism experiments

These variants follow the existing `model.py` and `context_models` interfaces.
They keep the benchmark's configured context/vocabulary and expose
`build_model(config)`, `forward(ids)`, and `predict_log_probs(ids)`. Run the
commands from `code/`; use a fresh output directory for each training run.

| Module | Attention change | Position representation |
|---|---|---|
| `attention_models.relative_bias` | Full causal attention with a learned per-head bias by query-key distance. | Learned absolute embeddings, as in the baseline. |
| `attention_models.gated_local_global` | Computes local and full-prefix attention, then learns a per-token, per-head blend. | Learned absolute embeddings. |
| `attention_models.rotary_local_global` | Alternates local and full-prefix blocks. | RoPE, matching the strongest current validation candidate. |

Example full-budget run and explicit validation evaluation:

```bash
python train.py --implementation attention_models.relative_bias --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/myruns/attention/relative-bias
python evaluate.py --checkpoint runs/myruns/attention/relative-bias/checkpoint.pt --device cpu --precision fp32 --split validation --output runs/myruns/attention/relative-bias/validation.json
```
gated_local_global:
```bash
python train.py --implementation attention_models.gated_local_global --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/myruns/attention/gated_local_global
python evaluate.py --checkpoint runs/myruns/attention/gated_local_global/checkpoint.pt --device cpu --precision fp32 --split validation --output runs/myruns/attention/gated_local_global/validation.json
```

rotary_local_global:
```bash
python train.py --implementation attention_models.rotary_local_global --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/myruns/attention/rotary_local_global
python evaluate.py --checkpoint runs/myruns/attention/rotary_local_global/checkpoint.pt --device cpu --precision fp32 --split validation --output runs/myruns/attention/rotary_local_global/validation.json
```

Replace the implementation and run-directory suffix with `attention_models.gated_local_global` / `gated-local-global` or `attention_models.rotary_local_global` / `rotary-local-global` to run the other variants. The local window defaults to 64; it can be overridden by `local_window` in a separate config JSON.

Choose the method and checkpoint using validation BPB. After freezing the final
method, evaluate that checkpoint on test once:

```bash
python evaluate.py --checkpoint runs/myruns/attention/rotary-local-global/checkpoint.pt --device cpu --precision fp32 --split test --output runs/myruns/attention/rotary-local-global/test.json
```

The local/global gate evaluates both attention paths, so it is an accuracy
experiment and may cost more CPU time than the baseline. The local mask is
applied through SDPA; do not assume it reduces measured time on every backend.
These are hypotheses to measure, not guaranteed BPB improvements. For a clean
comparison, keep the seed, training-target count, optimizer, and evaluation
protocol fixed, then ablate the component responsible for any gain. Do not use
test BPB to select settings.
