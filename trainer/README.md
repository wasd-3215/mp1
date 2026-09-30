# Custom experiment trainer

Use `train_experiment.py` for new training runs. The supplied course trainer
`train.py` remains available for the assignment's baseline workflow.

Run commands from the `code/` directory. Each experiment profile in
`configs/experiments/` selects an implementation, an architecture JSON, runtime
settings, and training settings. Width, depth, head counts, dropout, and other
model choices belong in the architecture JSON; optimizer and schedule choices
belong in the experiment profile. CLI flags can override training settings
for a one-off run.

```bash
python trainer/train_experiment.py --config configs/experiments/width288-reference-19200.json
```

Results are written below `runs/<run_group>/<profile-name>/seed-<seed>/`,
including `checkpoint.pt`, `best_checkpoint.pt`, `experiment.json`, and
`metrics.json`. Use the supplied `evaluate.py` to score saved checkpoints.

The primary implementation is `models.full_attention_dropout`: full causal
self-attention with RoPE, pre-RMSNorm, SwiGLU, tied embeddings, and bias-free
linear layers. Its dimensions are selected by the architecture JSON. Other
model modules are retained only for explicit mechanism experiments, such as
hybrid mixing or parameter sharing.
