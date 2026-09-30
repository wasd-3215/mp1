
train:
```bash
python train_lr_flex.py --implementation combined_model_dropout \
  --config shape_models/configs/width288_mha4_4heads.json \
  --resid-dropout 0.15 \
  --device cuda --precision auto --seed 71 \
  --steps 19200 --batch-size 32 --eval-every 300 \
  --max-lr 0.0015 --warmup-steps 100 --min-lr-ratio 0.01 \
  --weight-decay 0.4 \
  --run-dir runs/gpu-width256-depth4-mha4-dropout015-wd04-minlr001-19200-seed71
```