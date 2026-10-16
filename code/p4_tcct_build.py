"""Build TCCT-Net's inference inputs from full OpenFace CSVs (E4).

TCCT-Net (Vedernikov et al., CVPR-W 2024) reads, per clip, a CSV with the
OpenFace columns " pose_Rx" and " pose_Ry" (leading spaces as OpenFace writes
them), every clip exactly 280 rows long, at a nominal 30 Hz; its README says
shorter signals were repeated to reach 280 and says nothing about longer ones,
so longer ones are cut to the first 280 rows here. Its released weights give
68.91 % on the 1,071 validation clips under its own preprocessing.

Two things its loader forces on us and are handled explicitly:

  * it matches a CSV to its label with pandas `str.contains`, a substring test,
    and takes the first hit, so "vid_0_1" would take the label of "vid_0_10".
    Every clip is therefore renamed to a fixed-width id, v0001..v1071, with the
    mapping written beside the outputs;
  * about a fifth of the corpus is stored with a 1000 Hz container timebase and
    decodes to ~10,000 frames for a ten-second clip (verified 07/09/2026 on six
    clips: 9,999 ms each). OpenFace's CSV for such a clip has ~10,000 rows, so
    its first 280 rows would be a quarter of a second of video. Those clips are
    resampled to 30 Hz by nearest timestamp before cutting, which makes 280 rows
    the same 9.3 s it is for every other clip. The choice is recorded per clip.

Usage:  python p4_tcct_build.py Validation ~/TCCT_Net
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

P4 = os.path.expanduser("~/p4")
FEATS = [" pose_Rx", " pose_Ry"]
L = 280
LONG_ROWS = 2000


def resample_30hz(df):
    t = df[" timestamp"].values
    ticks = np.arange(0, t[-1] + 1e-9, 1.0 / 30)
    idx = np.searchsorted(t, ticks)
    idx = np.clip(idx, 0, len(t) - 1)
    return df.iloc[np.unique(idx)].reset_index(drop=True)


def fit_280(df):
    n = len(df)
    if n >= L:
        return df.iloc[:L], "cut"
    reps = int(np.ceil(L / n))
    return pd.concat([df] * reps, ignore_index=True).iloc[:L], "tiled x%d" % reps


def main(split, repo):
    src = os.path.join(P4, "openface_full", split)
    lab = pd.read_csv(os.path.join(P4, "data", "labels_%s.csv" % split.lower()))
    lab = lab[lab["y"].notna()].reset_index(drop=True)
    out_dir = os.path.join(repo, "inference_files")
    for f in glob.glob(os.path.join(out_dir, "*.csv")):
        os.remove(f)
    os.makedirs(out_dir, exist_ok=True)

    rows, manifest, missing = [], [], []
    for i, r in lab.iterrows():
        base = r["chunk"][:-4] if r["chunk"].endswith(".mp4") else r["chunk"]
        path = os.path.join(src, base + ".csv")
        if not os.path.exists(path):
            missing.append(r["chunk"])
            continue
        df = pd.read_csv(path)
        n_raw = len(df)
        how = "as is"
        if n_raw > LONG_ROWS:
            df = resample_30hz(df)
            how = "resampled 1000Hz->30Hz (%d->%d rows)" % (n_raw, len(df))
        df, fit = fit_280(df[[" frame"] + FEATS] if " frame" in df.columns else df[["frame"] + FEATS])
        vid = "v%04d" % (i + 1)
        df.columns = ["frame"] + FEATS
        df.to_csv(os.path.join(out_dir, vid + ".csv"), index=False)
        rows.append({"Unnamed: 0": i, "chunk": vid + ".mp4", "label": int(r["y"])})
        manifest.append({"id": vid, "chunk": r["chunk"], "subj": int(r["subj"]),
                         "rows_raw": int(n_raw), "handling": how + ", " + fit,
                         "y": int(r["y"])})
    pd.DataFrame(rows).to_csv(os.path.join(repo, "labels_inference.csv"), index=False)
    json.dump(manifest, open(os.path.join(P4, "analysis", "E4_tcct_manifest_%s.json" % split), "w"),
              indent=2)
    n_long = sum(1 for m in manifest if m["handling"].startswith("resampled"))
    n_tiled = sum(1 for m in manifest if "tiled" in m["handling"])
    print("%s: %d inputs written, %d missing CSV, %d resampled from 1000 Hz, %d tiled to 280"
          % (split, len(rows), len(missing), n_long, n_tiled))
    if missing:
        print("MISSING:", missing[:10])
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1], os.path.expanduser(sys.argv[2]))
