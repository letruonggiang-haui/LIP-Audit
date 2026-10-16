"""Test the container-timestamp hypothesis for the ~19 % of EngageNet clips whose
header frame count reads 8,900-10,000 instead of ~300.

The hypothesis, recorded in E3a_data_integrity.md as unverified: the media is
ten seconds long like every other clip, and only the container's frame count
(or its timebase) is wrong. If that is right, decoding one of those clips to
the end yields ~300 real frames, not ~10,000, and OpenFace's per-frame
descriptor was computed over ~300 frames of real video.

Method, on the local validation clips (no server needed):
  1. read the header of every clip: CAP_PROP_FRAME_COUNT, FPS;
  2. for a sample of the anomalous clips, decode every frame and count;
  3. report header count, decoded count, file size, and the ratio.

Usage:  python p4_frame_hypothesis.py [n_sample]
"""
import glob
import os
import sys
import time

import cv2
import numpy as np

SRC = r"E:\datasets\EmotiW_2023_EngageNet\Validation"


def header(path):
    c = cv2.VideoCapture(path)
    n, fps = int(c.get(cv2.CAP_PROP_FRAME_COUNT)), c.get(cv2.CAP_PROP_FPS)
    w, h = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c.release()
    return n, fps, w, h


def decode_count(path):
    c = cv2.VideoCapture(path)
    n = 0
    last_ms = 0.0
    while True:
        ok, _ = c.read()
        if not ok:
            break
        n += 1
        last_ms = c.get(cv2.CAP_PROP_POS_MSEC)
    c.release()
    return n, last_ms


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    files = sorted(glob.glob(os.path.join(SRC, "*.mp4")))
    heads = {f: header(f) for f in files}
    counts = np.array([h[0] for h in heads.values()])
    anomalous = [f for f, h in heads.items() if h[0] > 2000]
    normal = [f for f, h in heads.items() if h[0] <= 2000]
    print("validation clips: %d; header count median %d; > 2000 frames: %d (%.1f %%)"
          % (len(files), int(np.median(counts)), len(anomalous),
             100.0 * len(anomalous) / len(files)))
    fps_anom = sorted(set(round(heads[f][1], 1) for f in anomalous))
    fps_norm = sorted(set(round(heads[f][1], 1) for f in normal))
    print("header fps, anomalous clips:", fps_anom)
    print("header fps, normal clips   :", fps_norm)
    sz_a = np.mean([os.path.getsize(f) for f in anomalous]) if anomalous else 0
    sz_n = np.mean([os.path.getsize(f) for f in normal])
    print("mean file size: anomalous %.2f MB, normal %.2f MB, ratio %.2f"
          % (sz_a / 1e6, sz_n / 1e6, sz_a / sz_n if sz_n else 0))

    rng = np.random.default_rng(20260907)
    sample = list(rng.choice(anomalous, min(k, len(anomalous)), replace=False))
    sample += list(rng.choice(normal, 2, replace=False))
    print("\n%-42s %8s %8s %9s %8s %s" % ("clip", "header", "decoded", "last ms", "fps", "verdict"))
    for f in sample:
        t0 = time.time()
        n_hdr, fps, w, h = heads[f]
        n_dec, last_ms = decode_count(f)
        verdict = ("HEADER WRONG, media ~%.1f s" % (last_ms / 1000) if n_dec < 0.5 * n_hdr
                   else "media really is long" if n_hdr > 2000 else "normal")
        print("%-42s %8d %8d %9.0f %8.2f %s   (%.0fs)" % (
            os.path.basename(f)[:42], n_hdr, n_dec, last_ms, fps, verdict, time.time() - t0),
            flush=True)


if __name__ == "__main__":
    main()
