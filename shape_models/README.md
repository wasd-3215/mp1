# Width/depth screening

The three variants share the same Pre-RMSNorm, RoPE, GQA, SwiGLU, and bias-free
linear design from `combined_model.py`. Only width and depth differ.

| Module | Config | Approx. parameters |
|---|---|---:|
| `shape_models.balanced` | width 128, depth 4 | 988,288 |
| `shape_models.deep_narrow` | width 96, depth 8 | 1,009,248 |
| `shape_models.shallow_wide` | width 168, depth 2 | 965,832 |

Run the commands from `code/`. Every run uses seed 17, 1,200 steps, batch size
32, and the same optimizer schedule. Validation is evaluated every 300 steps
and again after training.

```bash
python train.py --implementation shape_models.balanced --config shape_models/configs/balanced.json --device cpu --threads 4 --seed 17 --steps 1200 --eval-every 300 --run-dir runs/myruns/shapes/balanced
python train.py --implementation shape_models.deep_narrow --config shape_models/configs/deep_narrow.json --device cpu --threads 4 --seed 17 --steps 1200 --eval-every 300 --run-dir runs/myruns/shapes/deep-narrow
python train.py --implementation shape_models.shallow_wide --config shape_models/configs/shallow_wide.json --device cpu --threads 4 --seed 17 --steps 1200 --eval-every 300 --run-dir runs/myruns/shapes/shallow-wide
```

The near-matched parameter counts make this a shape comparison, not an exact
compute-matched comparison. Record the actual training times as well as
validation BPB. The trainer evaluates validation; test evaluation should wait
until the final shape is selected and frozen.
