# Context-use model variants

These are standalone model implementations for the fixed MP1 benchmark. They
keep `context=256`, use the configured vocabulary, and implement both `forward`
and `predict_log_probs`. The baseline in `model.py` is unchanged.

These are optional context/position mechanism variants. Run commands from
`code/` using `trainer/train_experiment.py` and an experiment profile in
`configs/experiments/`; the profile selects the implementation and points to
an architecture JSON.

| Module | Change from baseline |
|---|---|
| `context_models.no_position` | Ablates learned absolute position embeddings. |
| `context_models.rotary` | Replaces learned absolute positions with rotary position features in Q/K. |
| `context_models.local` | Uses learned absolute positions and a causal local window in every block. |
| `context_models.local_global` | Alternates local and full-prefix causal attention blocks. |

To run a variant, copy a profile from `configs/experiments/`, set its
`implementation` to the selected module, and point `architecture` at an
architecture JSON. Launch it with
`python trainer/train_experiment.py --config configs/experiments/<your-profile>.json`.
The trainer stores checkpoints and metrics under the profile's hierarchical
run path. Score checkpoints with the supplied `evaluate.py`; use validation
for selection and reserve test evaluation for the frozen final method.
