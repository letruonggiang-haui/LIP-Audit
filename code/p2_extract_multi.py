"""P2 giai doan 3 — trich 5 CUA SO x 16 khung o sampling rate 4, KHOP domain pretraining.

Chan doan 15/08: AUC phan biet muc 2 vs muc 3 chi .60 (can ~.79 de cham SOTA). Mot trong
cac nguyen nhan chua loai tru duoc la CACH LAY MAU KHUNG. Ban cu lay 16 khung TRAI DEU tren
10 giay (~0,63s/khung) — MAE-DFER duoc pretrain tren 16 khung LIEN TIEP o sampling rate 4
(~2,1s). Sai khac nay vua lech domain, vua vut bo 95% video (16/300 khung).

Ban nay sua MOT LUC BON thieu sot da ghi nhan:
  1. Khop domain pretraining: moi cua so = 16 khung x sr4 = 2,1s (dung nhu DFEW).
  2. Dung 80/300 khung thay vi 16/300 — nhieu thong tin thoi gian hon 5 lan.
  3. Cho phep TANG CUONG THOI GIAN luc train (moi epoch chon ngau nhien 1 trong 5 cua so) —
     truoc gio khung co dinh nen KHONG co temporal aug, da ghi la han che tu Phase 0.
  4. Cho phep GOP DA CUA SO luc test (giong --test_num_segment cua cong thuc goc).

Dau ra: ~/p2/faces5/<Split>_X.npy [N,5,16,160,160,3] uint8, <Split>_ids.npy.
Kich thuoc: Train 33,7GB + Val 10,6GB + Test 11,5GB = 55,8GB (con 112GB trong).

Chay: ~/p2/venv/bin/python p2_extract_multi.py <Split> [--limit N]
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_extract_faces import center_square, clip_bbox  # dung lai y het logic bbox

ROOT = Path(os.environ.get("DAISEE_ROOT", os.path.expanduser("~/daisee")))
OUT = Path(os.path.expanduser("~/p2/faces5"))
N_WIN, N_FRAMES, SR, SIZE = 5, 16, 4, 160


def window_starts(n):
    """5 diem bat dau phu deu clip; moi cua so trai dai span = 16*sr khung."""
    span = N_FRAMES * SR
    if n >= span:
        return np.linspace(0, n - span, N_WIN).astype(int), SR
    # clip qua ngan: giam sr de cua so van nam gon trong clip
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

    # pad ben: khung thieu -> lay khung da giai ma gan nhat
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
            if mm is None:  # ghi thang ra memmap, khong giu 34GB trong RAM
                mm = np.lib.format.open_memmap(
                    OUT / f"{split}_X.npy", mode="w+", dtype=np.uint8,
                    shape=(len(clips), N_WIN, N_FRAMES, SIZE, SIZE, 3))
            mm[len(ids)] = arr
            ids.append(c.stem)
            if len(ids) % 500 == 0:
                print(f"  {len(ids)}/{len(clips)}", flush=True)
    mm.flush()
    del mm
    # cat bot phan du neu co clip hong
    if len(ids) < len(clips):
        full = np.load(OUT / f"{split}_X.npy", mmap_mode="r")
        np.save(OUT / f"{split}_X_tmp.npy", np.asarray(full[:len(ids)]))
        os.replace(OUT / f"{split}_X_tmp.npy", OUT / f"{split}_X.npy")
    np.save(OUT / f"{split}_ids.npy", np.array(ids))
    print(f"{split}: xong {len(ids)}/{len(clips)} clip -> {OUT}/{split}_X.npy", flush=True)


if __name__ == "__main__":
    main()
