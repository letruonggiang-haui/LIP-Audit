"""Measure a 69-CLASS prototype classifier (nearest-class-mean, leave-one-out) on the
embedding cache.

Why: `proto_top1` recorded in the training log is a 4-CLASS problem (P=4 prototypes per
batch), not directly comparable with the adversary top-1 (69 classes) or the identity-probe
(69/88 classes). The paper's backbone table must use the same number of classes. This script
gives the 69-class number for the VERY quantity that the penalty optimises -> the 3rd row of
the table becomes comparable with the first two rows.

Each dimension is standardised with pool statistics (exactly as the penalty standardises with
batch statistics).
CPU-only, reads the existing z cache.

Run: ~/p2/venv/bin/python p2_proto69.py --z ~/p2/probe/proto2_s0_best_acc_late_z.npz
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DAISEE = Path(os.path.expanduser("~/daisee"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True)
    args = ap.parse_args()
    d = np.load(os.path.expanduser(args.z), allow_pickle=True)
    z, subj = d["z"], d["subj"]

    val_lab = pd.read_csv(DAISEE / "DAiSEE" / "Labels" / "ValidationLabels.csv")
    val_lab.columns = [c.strip() for c in val_lab.columns]
    val_subjects = set(val_lab["ClipID"].astype(str).str[:6])
    keep = np.array([s not in val_subjects for s in subj])
    z, subj = z[keep].astype(np.float64), subj[keep]

    z = (z - z.mean(0)) / (z.std(0) + 1e-5)  # same as the standardisation inside the penalty
    subs = np.unique(subj)
    idx = {s: np.where(subj == s)[0] for s in subs}
    sums = np.stack([z[idx[s]].sum(0) for s in subs])
    cnts = np.array([len(idx[s]) for s in subs], dtype=np.float64)
    lab = np.array([np.where(subs == s)[0][0] for s in subj])

    # leave-one-out: remove the anchor itself from its own class prototype
    hits = 0
    for i in range(len(z)):
        S, C = sums.copy(), cnts.copy()
        S[lab[i]] -= z[i]
        C[lab[i]] -= 1.0
        mu = S / np.maximum(C, 1.0)[:, None]
        hits += int(np.argmin(((mu - z[i]) ** 2).sum(1)) == lab[i])
    acc, n = hits / len(z), len(subs)
    print(f"{Path(args.z).stem}: prototype-{n}lop (nearest-class-mean, LOO) top-1 {acc:.4f} "
          f"(chance {1/n:.4f}) = {acc*n:.1f}x chance | n_clip={len(z)}", flush=True)


if __name__ == "__main__":
    main()
