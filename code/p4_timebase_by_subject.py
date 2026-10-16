"""Is the 1000 Hz container timebase a per-participant property?

Reads the header of every local clip of a split and tabulates, per participant,
how many clips carry a 1000 fps header. If it is equipment- or session-bound it
will be all-or-nothing per participant, and then it is one more participant
signature that any frame-derived statistic carries.

Usage:  python p4_timebase_by_subject.py Validation
"""
import glob
import os
import re
import sys

import cv2
import pandas as pd

ROOT = r"E:\datasets\EmotiW_2023_EngageNet"


def main(split):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, split, "*.mp4"))):
        c = cv2.VideoCapture(f)
        rows.append({"subj": int(re.match(r"subject_(\d+)_", os.path.basename(f)).group(1)),
                     "sess": os.path.basename(f).split("_")[2],
                     "fps": round(c.get(cv2.CAP_PROP_FPS), 1),
                     "frames": int(c.get(cv2.CAP_PROP_FRAME_COUNT))})
        c.release()
    d = pd.DataFrame(rows)
    d["tb1000"] = d["fps"] >= 999
    g = d.groupby("subj").agg(clips=("tb1000", "size"), tb1000=("tb1000", "sum"),
                              sessions=("sess", "nunique"))
    g["share"] = (g["tb1000"] / g["clips"]).round(3)
    print("%s: %d clips, %d participants, %d clips with a 1000 fps header (%.1f %%)"
          % (split, len(d), d["subj"].nunique(), d["tb1000"].sum(), 100 * d["tb1000"].mean()))
    print(g.to_string())
    mixed = g[(g["share"] > 0) & (g["share"] < 1)]
    print("\nparticipants all-1000Hz: %d, none: %d, mixed: %d"
          % ((g["share"] == 1).sum(), (g["share"] == 0).sum(), len(mixed)))
    if len(mixed):
        gs = d.groupby(["subj", "sess"])["tb1000"].mean().round(3)
        print("mixed participants by session:\n", gs[gs.index.get_level_values(0).isin(mixed.index)].to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Validation")
