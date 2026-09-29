# Context-use model variants

These are standalone model implementations for the fixed MP1 benchmark. They
keep `context=256`, use the configured vocabulary, and implement both `forward`
and `predict_log_probs`. The baseline in `model.py` is unchanged.

Run commands from `code/`. The trainer resolves implementation names as Python
modules; each module below provides `build_model(config)`.

| Module | Change from baseline |
|---|---|
| `context_models.no_position` | Ablates learned absolute position embeddings. |
| `context_models.rotary` | Replaces learned absolute positions with rotary position features in Q/K. |
| `context_models.local` | Uses learned absolute positions and a causal local window in every block. |
| `context_models.local_global` | Alternates local and full-prefix causal attention blocks. |

## Train and evaluate

Run these commands from `code/`. Use a new `--run-dir` for every training run;
the trainer creates the directory and saves `checkpoint.pt` and `metrics.json`
there. `--eval-every 300` records validation scores during training, and the
trainer also evaluates validation after the final step.

Example for the no-position ablation:

```bash
python train.py --implementation context_models.no_position --eval-every 300 --run-dir runs/myruns/context/no-position
python evaluate.py --checkpoint runs/myruns/context/no-position/checkpoint.pt --device cpu --precision fp32 --split validation --output runs/myruns/context/no-position/validation.json
```

After choosing and freezing the final method using validation results, run its
test evaluation:

```bash
python evaluate.py --checkpoint runs/myruns/context/no-position/checkpoint.pt --device cpu --precision fp32 --split test --output runs/myruns/context/no-position/test.json
```

Use the `bpb` field from the complete-test JSON as the test score. Do not use
test results to choose the variant, checkpoint, or settings. The explicit flags
above select reproducible FP32 CPU evaluation.

To train another variant, replace `context_models.no_position` and the output
directory suffix with `context_models.rotary` / `rotary`,
`context_models.local` / `local`, or `context_models.local_global` /
`local-global`. Evaluate the matching checkpoint from that run directory.

All variants use the baseline configuration by default. Set `local_window` in
a separate JSON config to change the local window size; its default is 64.
For a controlled comparison, keep seed, training targets, and optimizer setup
the same, select variants using validation BPB, and reserve test evaluation for
the frozen final method. Perform mechanism ablations before choosing the final
variant. These implementations are experimental candidates, not claims of
improved BPB; measure their validation quality and resource costs.

```bash
python train.py --implementation context_models.rotary --eval-every 300 --run-dir runs/myruns/context/rotary
python evaluate.py --checkpoint runs/myruns/context/rotary/checkpoint.pt --device cpu --precision fp32 --split validation --output runs/myruns/context/rotary/validation.json
```

local:
```bash
python train.py --implementation context_models.local --eval-every 300 --run-dir runs/myruns/context/local
python evaluate.py --checkpoint runs/myruns/context/local/checkpoint.pt --device cpu --precision fp32 --split validation --output runs/myruns/context/local/validation.json
```

local_global:
```bash
python train.py --implementation context_models.local_global --eval-every 300 --run-dir runs/myruns/context/local_global
python evaluate.py --checkpoint runs/myruns/context/local_global/checkpoint.pt --device cpu --precision fp32 --split validation --output runs/myruns/context/local_global/validation.json
```