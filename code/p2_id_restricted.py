"""Identity-probe GIOI HAN tren 69 doi tuong Train — de cung khong gian nhan voi adversary.

Ly do: adversary phan loai 69 doi tuong Train (chance .0145) tren clip Train; identity-probe
mac dinh chay tren pool Train+Val = 88 doi tuong (chance .0114). So truc tiep hai con so la
so lech khong gian nhan. Script nay chay probe tren DUNG 69 doi tuong Train, dung clip Train,
split 75/25 trong tung doi tuong -> so sanh chinh danh voi top-1 cua adversary.

CPU-only, doc cache z co san. Chay:
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
