"""Prepare the ResNet-TCN reproduction: extract the FIRST 16 frames of each clip + build the csv.

Faithful to the repo's transforms.py (EXTRACT_FREQUENCY=1, num_time_steps=16, read in sorted
filename order): saves 00.jpg..15.jpg (zero-padded so that sorted() gives the right order).
Train+Val -> train.csv (Abedi protocol), Test -> test.csv.
"""
import os
from multiprocessing import Pool

import cv2
import pandas as pd

SRC = os.path.expanduser("~/daisee/DAiSEE/DataSet")
LABELS = os.path.expanduser("~/daisee/DAiSEE/Labels")
OUT = os.path.expanduser("~/p2/repro/frames")
CSV_DIR = os.path.expanduser("~/p2/repro")
N_FRAMES = 16


def find_video(split, stem):
    subj = stem[:6]
    d = os.path.join(SRC, split, subj, stem)
    if not os.path.isdir(d):
        return None
    for f in os.listdir(d):
        if f.lower().endswith((".avi", ".mp4")):
            return os.path.join(d, f)
    return None


def extract_one(args):
    split, stem = args
    outdir = os.path.join(OUT, split, stem)
    done_flag = os.path.join(outdir, "15.jpg")
    if os.path.exists(done_flag):
        return (stem, True)
    vp = find_video(split, stem)
    if vp is None:
        return (stem, False)
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(vp)
    i = 0
    last = None
    while i < N_FRAMES:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        cv2.imwrite(os.path.join(outdir, f"{i:02d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        last = frame
        i += 1
    cap.release()
    # pad with the last frame if the video is short/broken (keep 16 so the transform does not crash)
    while 0 < i < N_FRAMES and last is not None:
        cv2.imwrite(os.path.join(outdir, f"{i:02d}.jpg"), last,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        i += 1
    return (stem, i == N_FRAMES)


def labeled(split):
    lab = pd.read_csv(os.path.join(LABELS, f"{split}Labels.csv"))
    lab.columns = [c.strip() for c in lab.columns]
    lab["stem"] = lab["ClipID"].astype(str).str.replace(r"\.(avi|mp4)$", "", regex=True)
    return list(zip(lab["stem"], lab["Engagement"].astype(int)))


if __name__ == "__main__":
    rows_train, rows_test = [], []
    for split in ("Train", "Validation", "Test"):
        items = labeled(split)
        with Pool(16) as pool:
            res = pool.map(extract_one, [(split, s) for s, _ in items])
        ok = {s for s, good in res if good}
        miss = [s for s, good in res if not good]
        print(f"{split}: {len(ok)}/{len(items)} ok, thieu {len(miss)}", flush=True)
        if miss[:5]:
            print("  vi du thieu:", miss[:5], flush=True)
        for s, y in items:
            if s not in ok:
                continue
            rel = os.path.join(OUT, split, s)
            if split == "Test":
                rows_test.append((rel, y))
            else:
                rows_train.append((rel, y))
    pd.DataFrame(rows_train, columns=["path", "label"]).to_csv(
        os.path.join(CSV_DIR, "train_repro.csv"), index=False)
    pd.DataFrame(rows_test, columns=["path", "label"]).to_csv(
        os.path.join(CSV_DIR, "test_repro.csv"), index=False)
    print("csv:", len(rows_train), "train(+val),", len(rows_test), "test", flush=True)
