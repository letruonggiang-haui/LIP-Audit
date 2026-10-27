#!/bin/bash
# R1 registration 4G: prepare frames (Train/Validation only), then start the three arms.
cd ~/r1/g/code || exit 1
~/p2/venv/bin/python r1g_frames_prep.py > ~/r1/g/prep.log 2>&1 || exit 1
for a in 16c 16u 50u; do
  tmux new-session -d -s r1_g$a "~/p2/venv/bin/python ~/r1/g/code/r1g_frames_train.py --arm $a > ~/r1/g/train_$a.log 2>&1"
done
