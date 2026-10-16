#!/bin/bash
# Doi chung KHOP cua w5 (crudetta = cung sieu tham so, chi khac cach lay mau khung)
# hien chi co n=1 -> chay seed 1,2 de co n=3 vs n=3. 20 epoch cho khop ngan sach w5.
cd ~/p2/code
PY=~/p2/venv/bin/python
rm -f ~/p2/runs/CTRL_SEEDS_DONE
COMMON="--epochs 20 --lr 1e-4 --warmup 2 --layer_decay 1.0 --smoothing 0.0 --reprob 0.0 --jitter 0.0"
for s in 1 2; do
  nice -n 5 $PY p2_train_v6.py --seed $s --out ~/p2/runs/crudetta_s$s $COMMON \
      > ~/p2/runs/crudetta_s$s.log 2>&1
done
{
echo "===== DOI CHUNG KHOP 16-khung, 3 seed, 20 epoch ====="
for r in crudetta_s0 crudetta_s1 crudetta_s2; do
  for ck in best_acc last; do
    $PY p2_eval_auc23.py --ckpt ~/p2/runs/$r/$ck.pth --data faces 2>/dev/null | grep AUC23 \
      || echo "$r/$ck FAILED"
  done
done
echo "DONE $(date -u)"
} > ~/p2/runs/ctrl_seeds.log 2>&1
touch ~/p2/runs/CTRL_SEEDS_DONE
