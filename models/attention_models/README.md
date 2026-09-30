# Attention-mechanism experiments

These are optional attention variants. The primary model lives in
`models.full_attention_dropout`; these modules are retained for controlled
mechanism comparisons. Run commands from `code/` and use
`trainer/train_experiment.py` with an experiment profile that names the chosen
module and its architecture JSON. Keep the training profile fixed when
comparing mechanisms.

| Module | Attention change | Position representation |
|---|---|---|
| `attention_models.relative_bias` | Full causal attention with a learned per-head bias by query-key distance. | Learned absolute embeddings, as in the baseline. |
| `attention_models.gated_local_global` | Computes local and full-prefix attention, then learns a per-token, per-head blend. | Learned absolute embeddings. |
| `attention_models.rotary_local_global` | Alternates local and full-prefix blocks. | RoPE, matching the strongest current validation candidate. |

To run one, copy an existing profile from `configs/experiments/`, set its
`implementation` to the desired module, and point `architecture` at an
architecture JSON. Then run `python trainer/train_experiment.py --config
configs/experiments/<your-profile>.json`; evaluate the resulting checkpoint
with the supplied `evaluate.py`.

The local/global gate evaluates both attention paths, so it is an accuracy
experiment and may cost more CPU time than the baseline. The local mask is
applied through SDPA; do not assume it reduces measured time on every backend.
These are hypotheses to measure, not guaranteed BPB improvements. Select
settings using validation BPB; reserve test BPB for the frozen final model.
