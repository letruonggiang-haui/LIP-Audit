"""Identity-probe RESTRICTED to the 69 Train subjects - to share the adversary's label space.

Why: the adversary classifies 69 Train subjects (chance .0145) on Train clips; the identity-probe
by default runs on the Train+Val pool = 88 subjects (chance .0114). Comparing the two numbers
directly compares mismatched label spaces. This script runs the probe on EXACTLY the 69 Train
subjects, using Train clips, split 75/25 within each subject -> a legitimate comparison with
the adversary's top-1.

CPU-only, reads the existing z cache. Run:
  ~/p2/venv/bin/python p2_id_restricted.py --z ~/p2/probe/mlp025_s0_last_z.npz
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from p2_probe import center, identity_probe

DAISEE = Path(os.path.expanduser("~/daisee"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True, help="cache <run>_z.npz tu p2_probe.py")
    args = ap.parse_args()

    d = np.load(os.path.expanduser(args.z), allow_pickle=True)
    z, subj = d["z"], d["subj"]

    val_lab = pd.read_csv(DAISEE / "DAiSEE" / "Labels" / "ValidationLabels.csv")
    val_lab.columns = [c.strip() for c in val_lab.columns]
    val_subjects = set(val_lab["ClipID"].astype(str).str[:6])
    keep = np.array([s not in val_subjects for s in subj])
    zt, st = z[keep], subj[keep]
    print(f"{Path(args.z).stem}: giu {keep.sum()}/{len(keep)} clip Train, "
          f"{len(np.unique(st))} doi tuong (bo doi tuong Validation)")

    for mode, zz in (("none", zt), ("trans", center(zt, st))):
        acc, n = identity_probe(zz, st)
        print(f"identity-probe-69[{mode:5s}]: top-1 {acc:.4f} "
              f"(chance {1/n:.4f}, {n} lop) = {acc*n:.1f}x chance", flush=True)


if __name__ == "__main__":
    main()
