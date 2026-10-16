"""E3a integrity gate, second layer: the source clip's own frame count.

One clip in the Train split was reduced from a CSV that OpenFace was still
writing: 134 frames stored where a fresh extraction returns 300. The stream
reducer decides a CSV is finished by its modification time, and the container
died six times during the campaign, so every clip in flight at each death went
quiet and looked finished. That failure is invisible to any statistic computed
on the descriptor itself.

The source video knows the answer. OpenCV reads the frame count out of the
container header without decoding, so the whole split is checked in a couple of
minutes, and any clip whose stored count falls short of its source is listed for
re-extraction.

Usage:
  python p4_verify_frames.py Train                 # report
  python p4_verify_frames.py Train --write-list    # also write the redo list
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

P4 = os.path.expanduser("~/p4")
TOLERANCE = 2   # OpenFace and the container header can disagree by a frame or two


def known_partial(split):
    """Clips OpenFace cannot finish, declared in the manifest.

    These are not pipeline failures and re-running does not help: OpenFace
    aborts on an out-of-bounds face ROI at the same frame every time. They are
    listed so the gate stops flagging them, and they are printed on every run so
    that they never become invisible.
    """
    path = os.path.join(P4, "logs", "partial_%s.json" % split)
    if not os.path.exists(path):
        return {}
    return {m["chunk"]: m for m in json.load(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--write-list", action="store_true")
    args = ap.parse_args()

    clip_dir = os.path.join(P4, "clips", args.split)
    red_dir = os.path.join(P4, "feats", "percli", args.split)
    clips = sorted(f for f in os.listdir(clip_dir) if f.endswith(".mp4"))
    print("%s: %d clips" % (args.split, len(clips)))

    partial = known_partial(args.split)
    if partial:
        cov = sorted(m["coverage"] for m in partial.values())
        print("  declared partial (OpenFace aborts): %d clips, coverage %.3f to %.3f"
              % (len(partial), cov[0], cov[-1]))

    bad, missing, checked = [], [], 0
    src_counts = {}
    for i, f in enumerate(clips, 1):
        npz = os.path.join(red_dir, f[:-4] + ".npz")
        if not os.path.exists(npz):
            missing.append(f)
            continue
        if f in partial:
            checked += 1
            continue
        cap = cv2.VideoCapture(os.path.join(clip_dir, f))
        n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        stored = int(np.load(npz)["frames_total"])
        src_counts[f] = (n_src, stored)
        checked += 1
        if n_src > 0 and stored < n_src - TOLERANCE:
            bad.append((f, n_src, stored))
        if i % 2000 == 0:
            print("  %d / %d" % (i, len(clips)), flush=True)

    print("checked %d, missing a descriptor %d, short against source %d"
          % (checked, len(missing), len(bad)))
    for f, n, s in bad[:15]:
        print("  %-46s source %6d, stored %6d" % (f, n, s))
    if len(bad) > 15:
        print("  ... and %d more" % (len(bad) - 15))

    if args.write_list:
        path = os.path.join(P4, "logs", "redo_%s.txt" % args.split)
        with open(path, "w") as fh:
            for f, _, _ in bad:
                fh.write(os.path.join(clip_dir, f) + "\n")
            for f in missing:
                fh.write(os.path.join(clip_dir, f) + "\n")
        print("wrote", path)

    sys.exit(1 if (bad or missing) else 0)


if __name__ == "__main__":
    main()
