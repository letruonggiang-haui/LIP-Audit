"""Do prototype classifier 69 LOP (nearest-class-mean, leave-one-out) tren embedding cache.

Ly do: `proto_top1` ghi trong log train la bai toan 4 LOP (P=4 prototype moi batch), khong
so truc tiep duoc voi adversary top-1 (69 lop) hay identity-probe (69/88 lop). Bang xuong
song cua bai phai dung chung so lop. Script nay cho ra con so 69 lop cua CHINH dai luong
ma phat toi uu hoa -> hang thu 3 cua bang so sanh duoc voi hai hang dau.

Chuan hoa tung chieu bang thong ke pool (giong het ham phat chuan hoa bang thong ke batch).
CPU-only, doc cache z co san.

Chay: ~/p2/venv/bin/python p2_proto69.py --z ~/p2/probe/proto2_s0_best_acc_late_z.npz
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

    z = (z - z.mean(0)) / (z.std(0) + 1e-5)  # giong chuan hoa trong ham phat
    subs = np.unique(subj)
    idx = {s: np.where(subj == s)[0] for s in subs}
    sums = np.stack([z[idx[s]].sum(0) for s in subs])
    cnts = np.array([len(idx[s]) for s in subs], dtype=np.float64)
    lab = np.array([np.where(subs == s)[0][0] for s in subj])

    # leave-one-out: tru chinh anchor khoi prototype cua lop no
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
