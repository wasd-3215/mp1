# Lecture 2 attention variants

This directory contains new, independent implementations inspired by Lecture
2's frontier-attention section. Existing project files and earlier experiment
directories are unchanged. All modules keep the benchmark context and
`build_model(config)`, `forward(ids)`, and `predict_log_probs(ids)` interfaces.

| Module | Mechanism | Practical note |
|---|---|---|
| `lecture_attention_models.gqa` | Four query heads share two KV heads. | Mild KV projection/cache reduction; query heads remain distinct. |
| `lecture_attention_models.mqa` | All query heads share one KV head. | Stronger compression, with greater risk of losing key/value diversity. |
| `lecture_attention_models.gated` | Output gate, Q/K normalization, and partial RoPE. | Follows the lecture's gated-attention block. |
| `lecture_attention_models.dsa` | Learned low-dimensional top-k indexer followed by causal attention. | Dense reference implementation for the 256-token task; top-k does not currently save runtime. |
| `lecture_attention_models.gated_delta_hybrid` | Gated-delta recurrence in all but the final exact-attention block. | Resets recurrent state for every independent evaluation window. |

Run from `code/` and use a different run directory for every model. Example:

```bash
python train.py --implementation lecture_attention_models.gqa --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/myruns/lecture/gqa
python evaluate.py --checkpoint runs/myruns/lecture/gqa/checkpoint.pt --device cpu --precision fp32 --split validation --output runs/myruns/lecture/gqa/validation.json
```

Replace the module and path suffix with `mqa`, `gated`, `dsa`, or
`gated_delta_hybrid` to compare the other candidates. For the hybrid, the
output directory suffix can be `gated-delta-hybrid`.

Select settings and checkpoints using validation BPB. After freezing the final
method, evaluate its test score in FP32 on CPU:

```bash
python evaluate.py --checkpoint runs/myruns/lecture/gated-delta-hybrid/checkpoint.pt --device cpu --precision fp32 --split test --output runs/myruns/lecture/gated-delta-hybrid/test.json
```

The DSA variant uses a dense score/index matrix because the benchmark window is
short and PyTorch's standard SDPA interface does not provide the lecture's
specialized sparse kernel. It is useful to study learned selection, but should
not be presented as a speed optimization. The DeltaNet implementation is a
simple recurrent reference and may be slower on CPU due to its token loop. GQA,
MQA, and DeltaNet chiefly reduce KV-cache costs; the supplied evaluator resets
state for each 256-token window and does not currently expose an incremental
KV-cache API, so cache savings may not improve measured evaluation time here.

These models combine some lecture mechanisms (the gated variant uses all three
gated-attention changes). Treat them as hypotheses, not expected improvements.
Use matched seeds and processed-target counts, compare validation BPB and
measured costs, and ablate components before attributing any gains. Do not use
test BPB to choose a model or setting.
