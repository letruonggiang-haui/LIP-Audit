"""P2 stage 3 - extract 5 WINDOWS x 16 frames at sampling rate 4, MATCHING the pretraining domain.

Diagnosis 15/08: the AUC separating level 2 from level 3 is only .60 (~.79 needed to reach
SOTA). One of the causes not yet ruled out is the FRAME SAMPLING METHOD. The old version took
16 frames SPREAD EVENLY over 10 seconds (~0,63s/frame) - MAE-DFER was pretrained on 16
CONSECUTIVE frames at sampling rate 4 (~2,1s). This mismatch both shifts the domain and
throws away 95% of the video (16/300 frames).

This version fixes FOUR recorded shortcomings AT ONCE:
  1. Matches the pretraining domain: each window = 16 frames x sr4 = 2,1s (exactly as DFEW).
  2. Uses 80/300 frames instead of 16/300 - 5 times more temporal information.
  3. Allows TEMPORAL AUGMENTATION during training (each epoch picks 1 of the 5 windows at
     random) - until now the frames were fixed, so there was NO temporal aug, recorded as a
     limitation since Phase 0.
  4. Allows MULTI-WINDOW POOLING at test time (like --test_num_segment of the original recipe).

Output: ~/p2/faces5/<Split>_X.npy [N,5,16,160,160,3] uint8, <Split>_ids.npy.
Size: Train 33,7GB + Val 10,6GB + Test 11,5GB = 55,8GB (112GB still free).

Run: ~/p2/venv/bin/python p2_extract_multi.py <Split> [--limit N]
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_extract_faces import center_square, clip_bbox  # reuse exactly the same bbox logic

ROOT = Path(os.environ.get("DAISEE_ROOT", os.path.expanduser("~/daisee")))
OUT = Path(os.path.expanduser("~/p2/faces5"))
N_WIN, N_FRAMES, SR, SIZE = 5, 16, 4, 160


def window_starts(n):
    """5 start points spread evenly over the clip; each window spans span = 16*sr frames."""
    span = N_FRAMES * SR
    if n >= span:
        return np.linspace(0, n - span, N_WIN).astype(int), SR
    # clip too short: reduce sr so that the window still fits inside the clip
    sr = max(1, n // N_FRAMES)
    return np.zeros(N_WIN, dtype=int), sr


def decode(args):
    path_str, csv_str = args
    cap = cv2.VideoCapture(path_str)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n <= 0:
        n = 300
    box = clip_bbox(csv_str, w, h) if w > 0 and h > 0 else None

    starts, sr = window_starts(n)
    wins = [[min(int(s) + k * sr, n - 1) for k in range(N_FRAMES)] for s in starts]
    need = sorted({i for wn in wins for i in wn})

    got, j = {}, 0
    for i in range(n):
        if not cap.grab():
            break
        if j < len(need) and i == need[j]:
            ok, fr = cap.retrieve()
            if not ok:
                break
            if box is None:
                box = center_square(fr.shape[1], fr.shape[0])
            x0, y0, x1, y1 = box
            fr = cv2.resize(fr[y0:y1, x0:x1], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            j += 1
        while j < len(need) and need[j] <= i:
            j += 1
    cap.release()
    if not got:
        return None

    # tolerant padding: missing frame -> take the nearest decoded frame
    keys = np.array(sorted(got))
    out = np.empty((N_WIN, N_FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
    for a, wn in enumerate(wins):
        for b, i in enumerate(wn):
            out[a, b] = got.get(i, got[int(keys[np.abs(keys - i).argmin()])])
    return out


def main():
    split = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    clips = sorted((ROOT / "DAiSEE" / "DataSet" / split).rglob("*.avi")) + \
        sorted((ROOT / "DAiSEE" / "DataSet" / split).rglob("*.mp4"))
    if limit:
        clips = clips[:limit]
    print(f"{split}: {len(clips)} clip, {N_WIN} cua so x {N_FRAMES} khung x sr{SR}", flush=True)
    jobs = [(str(c), str(ROOT / "OpenFace" / split / f"{c.stem}.csv")) for c in clips]

    OUT.mkdir(parents=True, exist_ok=True)
    mm, ids = None, []
    with Pool(16) as pool:
        for c, arr in zip(clips, pool.imap(decode, jobs, chunksize=4)):
            if arr is None:
                continue
            if mm is None:  # write straight to the memmap, do not hold 34GB in RAM
                mm = np.lib.format.open_memmap(
                    OUT / f"{split}_X.npy", mode="w+", dtype=np.uint8,
                    shape=(len(clips), N_WIN, N_FRAMES, SIZE, SIZE, 3))
            mm[len(ids)] = arr
            ids.append(c.stem)
            if len(ids) % 500 == 0:
                print(f"  {len(ids)}/{len(clips)}", flush=True)
    mm.flush()
    del mm
    # trim the excess if any clip was broken
    if len(ids) < len(clips):
        full = np.load(OUT / f"{split}_X.npy", mmap_mode="r")
        np.save(OUT / f"{split}_X_tmp.npy", np.asarray(full[:len(ids)]))
        os.replace(OUT / f"{split}_X_tmp.npy", OUT / f"{split}_X.npy")
    np.save(OUT / f"{split}_ids.npy", np.array(ids))
    print(f"{split}: xong {len(ids)}/{len(clips)} clip -> {OUT}/{split}_X.npy", flush=True)


if __name__ == "__main__":
    main()
