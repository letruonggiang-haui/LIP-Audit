#!/bin/bash
# Phep so KHOP THUOC: AUC23 tren best_acc.pth cua CA HAI the he, cung Validation.
cd ~/p2/code
L=~/p2/runs/match23.log
PY=~/p2/venv/bin/python
rm -f ~/p2/runs/MATCH23_DONE
{
echo "===== BAN CU (16 khung, 1 view) — checkpoint chon theo best_acc ====="
for r in crudetta_s0 tuned_s0 frozen_s0 base_s2; do
  $PY p2_eval_auc23.py --ckpt ~/p2/runs/$r/best_acc.pth --data faces 2>&1 | grep -v '^ckpt ep' || echo "$r FAILED"
done
echo "===== BAN MOI (5 cua so, gop 5 view) — checkpoint chon theo best_acc ====="
for r in w5_s0 w5_s1 w5_s2; do
  $PY p2_eval_auc23.py --ckpt ~/p2/runs/$r/best_acc.pth --data faces5 2>&1 | grep -v '^ckpt ep' || echo "$r FAILED"
done
echo "===== THAM CHIEU: w5 best_auc23 (chon tren chinh auc23 — VE LECH THUOC) ====="
for r in w5_s1 w5_s2; do
  $PY p2_eval_auc23.py --ckpt ~/p2/runs/$r/best_auc23.pth --data faces5 2>&1 | grep -v '^ckpt ep' || echo "$r FAILED"
done
echo "DONE $(date -u)"
} > $L 2>&1
touch ~/p2/runs/MATCH23_DONE
