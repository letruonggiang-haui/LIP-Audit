#!/bin/bash
# 4a: 3 seeds w5 + merge_val, sequential. Official checkpoint = last.pth (pre-registered).
cd ~/p2/code
rm -f ~/p2/runs/W5M_SEEDS_DONE
for s in 0 1 2; do
  nice -n 5 ~/p2/venv/bin/python p2_train_v10.py --seed $s --merge_val \
      --out ~/p2/runs/w5m_s$s > ~/p2/runs/w5m_s$s.log 2>&1
done
touch ~/p2/runs/W5M_SEEDS_DONE
