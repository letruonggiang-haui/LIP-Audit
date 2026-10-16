"""P2 Phase 0 — trich 16 khung mat 160x160/clip cho pipeline MAE-DFER.

Bbox mat tinh tu eye landmarks OpenFace (median toan clip, khong can landmark 68 diem):
  - eye1 = trung binh eye_lmk_{x,y}_0..27, eye2 = trung binh 28..55
  - d = khoang cach hai mat; canh vuong = 3.2*d, tam = (mid_x, mid_y + 0.55*d)
  - fallback (thieu CSV / <10 dong hop le / d<10px): crop vuong giua khung 480x480

Dau ra: ~/p2/faces/<Split>_X.npy [N,16,160,160,3] uint8 (RGB), <Split>_ids.npy.
Chay: ~/p2/venv/bin/python p2_extract_faces.py <Split>   # Train|Validation|Test
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DAISEE_ROOT", os.path.expanduser("~/daisee")))
OUT = Path(os.path.expanduser("~/p2/faces"))
N_FRAMES, SIZE = 16, 160

EYE_X1 = [f"eye_lmk_x_{i}" for i in range(28)]
EYE_Y1 = [f"eye_lmk_y_{i}" for i in range(28)]
EYE_X2 = [f"eye_lmk_x_{i}" for i in range(28, 56)]
EYE_Y2 = [f"eye_lmk_y_{i}" for i in range(28, 56)]


def clip_bbox(csv_path, w, h):
    """Bbox vuong (x0,y0,x1,y1) tu median eye landmarks; None -> fallback."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        ok = df[(df["success"] == 1) & (df["confidence"] >= 0.8)]
        if len(ok) < 10:
            return None
        e1x, e1y = ok[EYE_X1].mean(axis=1).median(), ok[EYE_Y1].mean(axis=1).median()
        e2x, e2y = ok[EYE_X2].mean(axis=1).median(), ok[EYE_Y2].mean(axis=1).median()
        d = float(np.hypot(e2x - e1x, e2y - e1y))
        if d < 10:
            return None
        cx, cy, side = (e1x + e2x) / 2, (e1y + e2y) / 2 + 0.55 * d, 3.2 * d
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        return x0, y0, x1, y1
    except Exception:
        return None


def center_square(w, h):
    s = min(w, h)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return x0, y0, x0 + s, y0 + s


def decode(args):
    """Ben voi video hong duoi/thieu khung: pad khung cuoi; chi None khi 0 khung."""
    path_str, csv_str = args
    cap = cv2.VideoCapture(path_str)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n <= 0:
        n = 400  # metadata hong -> doc tuan tu toi da 400 khung
    box = clip_bbox(csv_str, w, h) if w > 0 and h > 0 else None
    idxs = np.linspace(0, n - 1, N_FRAMES).astype(int)
    frames, j = [], 0
    for i in range(n):
        if not cap.grab():
            break
        if j < len(idxs) and i >= idxs[j]:
            ok, fr = cap.retrieve()
            if not ok:
                break
            if box is None:
                box = center_square(fr.shape[1], fr.shape[0])
            x0, y0, x1, y1 = box
            fr = cv2.resize(fr[y0:y1, x0:x1], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            while j < len(idxs) and i >= idxs[j]:
                j += 1
    cap.release()
    if not frames:
        return None
    while len(frames) < N_FRAMES:
        frames.append(frames[-1])
    return np.stack(frames[:N_FRAMES])  # [16,160,160,3] uint8


def main():
    split = sys.argv[1]
    clips = sorted((ROOT / "DAiSEE" / "DataSet" / split).rglob("*.avi")) + \
            sorted((ROOT / "DAiSEE" / "DataSet" / split).rglob("*.mp4"))
    print(f"{split}: {len(clips)} clips", flush=True)
    jobs = [(str(c), str(ROOT / "OpenFace" / split / f"{c.stem}.csv")) for c in clips]

    feats, ids, n_fallback = [], [], 0
    with Pool(16) as pool:
        for c, arr in zip(clips, pool.imap(decode, jobs, chunksize=4)):
            if arr is None:
                continue
            feats.append(arr)
            ids.append(c.stem)
            if len(ids) % 500 == 0:
                print(f"  {len(ids)}/{len(clips)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{split}_X.npy", np.stack(feats))
    np.save(OUT / f"{split}_ids.npy", np.array(ids))
    print(f"{split}: saved {len(ids)}/{len(clips)} clips -> {OUT}/{split}_X.npy", flush=True)


if __name__ == "__main__":
    main()
