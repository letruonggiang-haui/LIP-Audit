"""Pack the released MARLIN tensors into one array per split and write the
label file as CSV, so that the server receives two files per split instead of
eight thousand and does not need openpyxl.

The test split is packed only when --include-test is passed. Until E0 returns
an answer from the dataset authors on the status of the test labels, the test
split stays on the workstation.

Usage:  python p4_pack_marlin.py [--include-test] [--out DIR]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4_labels as L  # noqa: E402

SRC = L.ENGAGENET_ROOT
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "staging_p4")
SPLITS = {"train": ("MARLIN_Train", "Train"),
          "validation": ("MARLIN_Validation", "Validation"),
          "test": ("MARLIN_Test", "Test")}


T_FIXED = 9  # the modal sequence length of the released tensors


def resample(a, T=T_FIXED):
    """Uniform temporal resampling to a fixed length.

    The released tensors are not one shape. Most are 9 x 1024, but 1,012 of the
    11,311 are 312 x 1024 and a long tail runs from 1 to 292, so the archive
    mixes at least two extraction settings. Everything is brought to the modal
    length by nearest-index resampling, which is the identity on the 9 x 1024
    majority, and the original length travels with the array so that the mix can
    be conditioned on later.
    """
    n = a.shape[0]
    if n == T:
        return a
    idx = np.rint(np.linspace(0, n - 1, T)).astype(int)
    return a[idx]


def pack(split, out_dir):
    marlin_dir, _ = SPLITS[split]
    lab = L.load_engagenet(split)
    lab = lab[["chunk", "label", "subj", "sess", "vid", "t"]].copy()
    lab["y"] = lab["label"].map(L.EN_LEVELS)

    feats, lens, keep_rows, missing = [], [], [], []
    for i, chunk in enumerate(lab["chunk"].values):
        path = os.path.join(SRC, marlin_dir, chunk + ".pt")
        if not os.path.exists(path):
            missing.append(chunk)
            continue
        a = np.asarray(torch.load(path, map_location="cpu", weights_only=False),
                       dtype=np.float32)
        lens.append(a.shape[0])
        feats.append(resample(a))
        keep_rows.append(i)
        if (len(feats) % 2000) == 0:
            print("  %d / %d" % (len(feats), len(lab)))
    X = np.stack(feats)
    lens = np.asarray(lens, dtype=np.int32)
    lab = lab.iloc[keep_rows].reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "marlin_%s.npz" % split),
                        X=X, seq_len=lens, chunk=lab["chunk"].values.astype(str))
    lab["marlin_seq_len"] = lens
    lab.to_csv(os.path.join(out_dir, "labels_%s.csv" % split), index=False,
               encoding="utf-8")
    vals, cnt = np.unique(lens, return_counts=True)
    print("%s: X %s, %d rows, %d missing tensors" % (split, X.shape, len(lab), len(missing)))
    print("  original lengths: " +
          ", ".join("%d:%d" % (v, c) for v, c in zip(vals, cnt) if c >= 20)
          + " (+%d more values)" % int((cnt < 20).sum()))
    if missing:
        with open(os.path.join(out_dir, "missing_%s.txt" % split), "w") as fh:
            fh.write("\n".join(missing))
    return X.shape, len(missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true",
                    help="pack the test split too; only after E0 returns")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    splits = ["train", "validation"] + (["test"] if args.include_test else [])
    for s in splits:
        pack(s, args.out)
    if not args.include_test:
        print("test split not packed: E0 has not returned on the status of its labels")


if __name__ == "__main__":
    main()
