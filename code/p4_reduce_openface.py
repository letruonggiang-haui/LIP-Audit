"""E3a - reduce OpenFace per-frame CSV to the 98-dimensional per-chunk vector.

The layout follows Singh et al. (2023): gaze 16, head pose 12, action units 70,
each block being the per-frame mean and standard deviation of its channels.

  gaze       8 channels  gaze_0_{x,y,z}, gaze_1_{x,y,z}, gaze_angle_{x,y}
  head pose  6 channels  pose_T{x,y,z}, pose_R{x,y,z}
  AU        35 channels  17 intensities AU*_r, 18 presences AU*_c
  total     49 channels x {mean, sd} = 98

Frames the tracker marks unsuccessful are dropped before the statistics; a clip
with no successful frame gets a zero vector and is written to the failure log,
which is the convention P2 used and reported. The CSV of a reduced clip is
deleted when --delete-csv is passed, because the server runs at 92 per cent.

Usage:  python p4_reduce_openface.py Validation [--delete-csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

P4 = os.path.expanduser("~/p4")

GAZE = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z",
        "gaze_angle_x", "gaze_angle_y"]
POSE = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]


def au_columns(cols):
    r = sorted([c for c in cols if c.startswith("AU") and c.endswith("_r")])
    c = sorted([c for c in cols if c.startswith("AU") and c.endswith("_c")])
    return r + c


def _mean_sd(block, d):
    if len(block) == 0:
        return np.zeros(2 * d, dtype=np.float32)
    v = np.concatenate([block.mean(axis=0), block.std(axis=0, ddof=0)])
    return np.nan_to_num(v.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def reduce_one(path, n_win):
    """Whole-chunk 98-d vector plus an n_win x 98 sequence of sub-window vectors.

    The dataset paper describes the descriptor as a per-chunk mean and standard
    deviation, which leaves nothing for a recurrent or convolutional model to run
    over. The sub-window sequence is our reading of what those baselines consume;
    it is recorded as such rather than presented as the published construction.
    Sub-windows split the frame axis into n_win contiguous parts of equal length.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    au = au_columns(df.columns)
    chans = GAZE + POSE + au
    missing = [c for c in chans if c not in df.columns]
    if missing:
        raise ValueError("%s missing %s" % (path, missing[:5]))
    ok = (df["success"] == 1).values if "success" in df.columns else np.ones(len(df), bool)
    arr = df[chans].values.astype(np.float64)
    d = len(chans)
    good = arr[ok]
    n_ok = int(len(good))
    whole = _mean_sd(good, d)
    seq = np.zeros((n_win, 2 * d), dtype=np.float32)
    if n_ok > 0:
        edges = np.linspace(0, len(arr), n_win + 1).astype(int)
        for w in range(n_win):
            block = arr[edges[w]:edges[w + 1]]
            mask = ok[edges[w]:edges[w + 1]]
            block = block[mask]
            seq[w] = _mean_sd(block, d) if len(block) else whole
    return whole, seq, chans, n_ok, len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["Train", "Validation", "Test"])
    ap.add_argument("--delete-csv", action="store_true")
    ap.add_argument("--windows", type=int, default=9,
                    help="sub-windows per chunk; 9 matches the modal MARLIN length")
    args = ap.parse_args()

    csv_dir = os.path.join(P4, "openface", args.split)
    files = sorted(f for f in os.listdir(csv_dir) if f.endswith(".csv"))
    if not files:
        sys.exit("no CSV in " + csv_dir)

    X, S, ids, n_ok_all, n_all, failed, names = [], [], [], [], [], [], None
    for i, f in enumerate(files, 1):
        v, seq, chans, n_ok, n_tot = reduce_one(os.path.join(csv_dir, f), args.windows)
        names = names or ([c + "_mean" for c in chans] + [c + "_sd" for c in chans])
        X.append(v)
        S.append(seq)
        ids.append(f[:-4] + ".mp4")
        n_ok_all.append(n_ok)
        n_all.append(n_tot)
        if n_ok == 0:
            failed.append(f)
        if args.delete_csv:
            os.remove(os.path.join(csv_dir, f))
        if i % 500 == 0:
            print("  %d / %d" % (i, len(files)), flush=True)

    X = np.stack(X)
    S = np.stack(S)
    out = os.path.join(P4, "feats", "openface98_%s.npz" % args.split.lower())
    np.savez_compressed(out, X=X, X_seq=S, n_windows=args.windows,
                        chunk=np.asarray(ids, dtype=str),
                        frames_ok=np.asarray(n_ok_all, dtype=np.int32),
                        frames_total=np.asarray(n_all, dtype=np.int32),
                        feature_names=np.asarray(names, dtype=str))
    print("%s: X %s  X_seq %s -> %s" % (args.split, X.shape, S.shape, out))
    print("clips with no successful frame: %d of %d" % (len(failed), len(files)))
    print("median tracked frames per clip: %.0f" % float(np.median(n_ok_all)))
    if failed:
        with open(os.path.join(P4, "logs", "noface_%s.txt" % args.split), "w") as fh:
            fh.write("\n".join(failed))


if __name__ == "__main__":
    main()
