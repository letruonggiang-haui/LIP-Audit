"""E3a - reduce OpenFace CSV to per-clip vectors as they appear, and delete them.

The server runs at 94 per cent of its disk. About one EngageNet clip in five is
read by OpenFace as roughly 8,900 frames rather than the roughly 300 of a
ten-second chunk, and those produce CSV files of 18 to 20 MB. Holding a whole
split's CSV would need over 20 GB, which is not available, so this runs beside
the extraction and empties the directory behind it.

A CSV is treated as finished when it has not been written to for --age seconds.
FeatureExtraction appends continuously while it works, so an idle file is done.

Two modes:
  --watch   loop until the extraction marker appears and no CSV is left
  --pack    assemble the per-clip vectors of a split into one npz

Usage:
  python p4_reduce_stream.py Validation --watch
  python p4_reduce_stream.py Validation --pack
"""
import argparse
import os
import time

import numpy as np

from p4_reduce_openface import reduce_one

P4 = os.path.expanduser("~/p4")


def per_clip_dir(split):
    d = os.path.join(P4, "feats", "percli", split)
    os.makedirs(d, exist_ok=True)
    return d


def sweep(split, age, n_win, delete=True):
    csv_dir = os.path.join(P4, "openface", split)
    out_dir = per_clip_dir(split)
    if not os.path.isdir(csv_dir):
        return 0, 0
    now = time.time()
    done = err = 0
    for f in sorted(os.listdir(csv_dir)):
        if not f.endswith(".csv"):
            continue
        p = os.path.join(csv_dir, f)
        try:
            if now - os.path.getmtime(p) < age:
                continue
            v, seq, chans, n_ok, n_tot = reduce_one(p, n_win)
        except Exception as ex:                      # a truncated or odd file
            print("  SKIP %s: %s" % (f, ex), flush=True)
            err += 1
            continue
        np.savez(os.path.join(out_dir, f[:-4] + ".npz"), x=v, seq=seq,
                 frames_ok=n_ok, frames_total=n_tot)
        if delete:
            os.remove(p)
        done += 1
    return done, err


def expected_chunks(split):
    """Chunk names the label file lists for this split, SNP included.

    The pack step must cover all of them. A partial pack would be joined away
    silently by the training script, and the clips are deleted right after the
    pack, so a missed clip could not be recovered without reshipping 22 GB.
    """
    import csv
    path = os.path.join(P4, "data", "labels_%s.csv" % split.lower())
    with open(path, newline="", encoding="utf-8") as fh:
        return set(row["chunk"].strip() for row in csv.DictReader(fh))


def pack(split, n_win):
    out_dir = per_clip_dir(split)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".npz"))
    if not files:
        raise SystemExit("nothing reduced for " + split)
    want = expected_chunks(split)
    have = set(f[:-4] + ".mp4" for f in files)
    if have != want:
        missing = sorted(want - have)
        extra = sorted(have - want)
        raise SystemExit(
            "PACK REFUSED for %s: %d of %d labelled chunks reduced; "
            "%d missing (first: %s), %d unexpected (first: %s)"
            % (split, len(have & want), len(want), len(missing),
               missing[:3], len(extra), extra[:3]))
    X, S, ids, ok, tot = [], [], [], [], []
    for f in files:
        z = np.load(os.path.join(out_dir, f))
        X.append(z["x"])
        S.append(z["seq"])
        ids.append(f[:-4] + ".mp4")
        ok.append(int(z["frames_ok"]))
        tot.append(int(z["frames_total"]))
    X, S = np.stack(X), np.stack(S)
    out = os.path.join(P4, "feats", "openface98_%s.npz" % split.lower())
    np.savez_compressed(out, X=X, X_seq=S, n_windows=n_win,
                        chunk=np.asarray(ids, dtype=str),
                        frames_ok=np.asarray(ok, dtype=np.int32),
                        frames_total=np.asarray(tot, dtype=np.int32))
    ok = np.asarray(ok)
    print("%s: X %s  X_seq %s -> %s" % (split, X.shape, S.shape, out))
    print("clips with no tracked frame: %d of %d" % (int((ok == 0).sum()), len(ok)))
    print("frames per clip: median %.0f, 90th pct %.0f, max %d"
          % (np.median(tot), np.percentile(tot, 90), max(tot)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["Train", "Validation", "Test"])
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--age", type=int, default=180)
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--windows", type=int, default=9)
    ap.add_argument("--keep-csv", action="store_true")
    args = ap.parse_args()

    if args.pack:
        pack(args.split, args.windows)
        return

    marker = os.path.join(P4, "runs", "EXTRACT_DONE_%s" % args.split)
    total = 0
    while True:
        d, e = sweep(args.split, args.age, args.windows, delete=not args.keep_csv)
        total += d
        left = len([f for f in os.listdir(os.path.join(P4, "openface", args.split))
                    if f.endswith(".csv")]) if os.path.isdir(
                        os.path.join(P4, "openface", args.split)) else 0
        print("[%s] reduced %d this pass, %d total, %d csv left, %d errors"
              % (time.strftime("%H:%M:%S"), d, total, left, e), flush=True)
        if os.path.exists(marker) and left == 0:
            break
        if not args.watch:
            break
        time.sleep(args.every)
    print("reduction finished for", args.split)


if __name__ == "__main__":
    main()
