"""R1 registration, Section 4G: frames for the ResNet-TCN frame ablation.

Train and Validation of DAiSEE only; the Test split is never listed, read or decoded.

  16c  the first 16 consecutive frames (released configuration): the frames
       already extracted for the re-run, ~/p2/repro/frames/<split>/<stem>/00..15.jpg
  16u  16 frames spread uniformly over the clip, index floor(i*n/16)
  50u  50 frames spread uniformly over the clip, index floor(i*n/50)
       (the article: "down-sampled, temporally and spatially, to get
       50 x 3 x 224 x 224 tensors")

n is the decoded frame count of the clip. Frames are stored as JPEG quality 92
at native resolution, as the re-run's were, named 00.jpg.. so that sorted()
keeps the order. Writes one CSV per arm and split in the format of the
repository's VideoDataset (path,label).

Run: ~/p2/venv/bin/python r1g_frames_prep.py   (cwd anywhere)
"""
import os
from multiprocessing import Pool

import cv2
import pandas as pd

SRC = os.path.expanduser("~/daisee/DAiSEE/DataSet")
LABELS = os.path.expanduser("~/daisee/DAiSEE/Labels")
OLD = os.path.expanduser("~/p2/repro/frames")
R1 = os.path.expanduser("~/r1/g")
SPLITS = ("Train", "Validation")
ARMS = {"16u": 16, "50u": 50}


def find_video(split, stem):
    d = os.path.join(SRC, split, stem[:6], stem)
    if not os.path.isdir(d):
        return None
    for f in os.listdir(d):
        if f.lower().endswith((".avi", ".mp4")):
            return os.path.join(d, f)
    return None


def extract_one(args):
    split, stem = args
    assert split in SPLITS
    outs = {a: os.path.join(R1, "frames" + a, split, stem) for a in ARMS}
    if all(os.path.exists(os.path.join(o, "%02d.jpg" % (ARMS[a] - 1))) for a, o in outs.items()):
        return stem, True, -1
    vp = find_video(split, stem)
    if vp is None:
        return stem, False, 0
    cap = cv2.VideoCapture(vp)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)
    if n == 0:
        return stem, False, 0
    for a, k in ARMS.items():
        os.makedirs(outs[a], exist_ok=True)
        for i in range(k):
            cv2.imwrite(os.path.join(outs[a], "%02d.jpg" % i), frames[(i * n) // k],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    return stem, True, n


def labeled(split):
    lab = pd.read_csv(os.path.join(LABELS, "%sLabels.csv" % split))
    lab.columns = [c.strip() for c in lab.columns]
    lab["stem"] = lab["ClipID"].astype(str).str.replace(r"\.(avi|mp4)$", "", regex=True)
    return list(zip(lab["stem"], lab["Engagement"].astype(int)))


if __name__ == "__main__":
    os.makedirs(R1, exist_ok=True)
    for split in SPLITS:
        items = labeled(split)
        with Pool(24) as pool:
            res = pool.map(extract_one, [(split, s) for s, _ in items])
        ok = {s for s, good, _ in res if good}
        ns = [n for _, good, n in res if good and n > 0]
        print("%s: %d/%d decoded; frame count min %s max %s"
              % (split, len(ok), len(items), min(ns) if ns else "-", max(ns) if ns else "-"), flush=True)
        for arm in ("16c", "16u", "50u"):
            root = OLD if arm == "16c" else os.path.join(R1, "frames" + arm)
            rows = []
            for s, y in items:
                p = os.path.join(root, split, s)
                last = os.path.join(p, "%02d.jpg" % ((16 if arm != "50u" else 50) - 1))
                if os.path.exists(last):
                    rows.append((p, y))
            pd.DataFrame(rows, columns=["path", "label"]).to_csv(
                os.path.join(R1, "%s_%s.csv" % (arm, split.lower())), index=False)
            print("  %s %s: %d clips" % (arm, split, len(rows)), flush=True)
    open(os.path.join(R1, "PREP_DONE"), "w").close()
