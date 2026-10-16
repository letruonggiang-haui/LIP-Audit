#!/bin/bash
# Bound seed noise cua AUC23 trong ho 16-khung: quy tac 'last.pth' ap chung.
cd ~/p2/code
PY=~/p2/venv/bin/python
rm -f ~/p2/runs/NOISE23_DONE
{
echo "===== 16-khung, quy tac last.pth (3 seed vanilla + 2 recipe tuned) ====="
for r in base_s0 base_s1 base_s2 crudetta_s0 tuned_s0; do
  $PY p2_eval_auc23.py --ckpt ~/p2/runs/$r/last.pth --data faces 2>/dev/null | grep AUC23 || echo "$r FAILED"
done
echo "===== 5-cua-so, quy tac last.pth ====="
for r in w5_s0 w5_s1 w5_s2; do
  $PY p2_eval_auc23.py --ckpt ~/p2/runs/$r/last.pth --data faces5 2>/dev/null | grep AUC23 || echo "$r FAILED"
done
echo "DONE $(date -u)"
} > ~/p2/runs/noise23.log 2>&1
touch ~/p2/runs/NOISE23_DONE
