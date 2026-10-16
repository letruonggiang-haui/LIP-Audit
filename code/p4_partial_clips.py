"""E3a: the clips OpenFace cannot finish, handled explicitly rather than hidden.

Twelve of the 7,983 Train clips stop OpenFace part-way through, always at the
same frame. The cause is inside OpenFace, not in our pipeline: when the detected
face box runs past the edge of the image, the ROI is not clamped and OpenCV
throws

  Assertion failed (0 <= roi.x && ... roi.x + roi.width <= m.cols) in Mat

which aborts the process. Re-running changes nothing, because the offending
frame is the same every time.

This runs each affected clip on its own, keeps whatever the crash left behind,
reduces it, and writes a manifest recording the coverage of every one. The
descriptors are then complete in shape for all 7,983 clips and the twelve
partial ones are declared, which is the convention P2 used for DAiSEE and the
one PriorNet's own ablation shows matters.

Usage:  python p4_partial_clips.py Train
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

P4 = os.path.expanduser("~/p4")


def source_frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    split = sys.argv[1]
    listing = os.path.join(P4, "logs", "redo_%s.txt" % split)
    clips = [l.strip() for l in open(listing) if l.strip()]
    print("%s: %d clips OpenFace could not finish" % (split, len(clips)))

    csv_dir = os.path.join(P4, "openface", split)
    os.makedirs(csv_dir, exist_ok=True)
    for i, path in enumerate(clips, 1):
        base = os.path.splitext(os.path.basename(path))[0]
        inner = path.replace(os.path.expanduser("~/p4"), "/data")
        cmd = ("/home/openface-build/build/bin/FeatureExtraction -f %s "
               "-aus -pose -gaze -out_dir /data/openface/%s > /dev/null 2>&1 || true"
               % (inner, split))
        subprocess.run(["docker", "run", "--rm", "--entrypoint", "/bin/bash",
                        "-v", P4 + ":/data", "algebr/openface:latest", "-c", cmd],
                       check=False)
        got = os.path.join(csv_dir, base + ".csv")
        rows = sum(1 for _ in open(got)) - 1 if os.path.exists(got) else 0
        print("  %2d/%d %-46s %d rows" % (i, len(clips), base, rows), flush=True)

    # reduce everything now sitting in the CSV directory
    subprocess.run([os.path.expanduser("~/p2/venv/bin/python"),
                    os.path.join(P4, "code", "p4_reduce_stream.py"),
                    split, "--age", "5"], check=False)

    manifest = []
    red_dir = os.path.join(P4, "feats", "percli", split)
    clip_dir = os.path.join(P4, "clips", split)
    for path in clips:
        base = os.path.splitext(os.path.basename(path))[0]
        npz = os.path.join(red_dir, base + ".npz")
        n_src = source_frames(os.path.join(clip_dir, base + ".mp4"))
        if os.path.exists(npz):
            z = np.load(npz)
            got_t, got_ok = int(z["frames_total"]), int(z["frames_ok"])
        else:
            got_t = got_ok = 0
        manifest.append({"chunk": base + ".mp4", "source_frames": n_src,
                         "frames_extracted": got_t, "frames_tracked": got_ok,
                         "coverage": round(got_t / n_src, 4) if n_src else 0.0,
                         "reason": "OpenFace aborts on an out-of-bounds face ROI"})

    out = os.path.join(P4, "logs", "partial_%s.json" % split)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    cov = [m["coverage"] for m in manifest]
    print("\nmanifest written to", out)
    print("coverage: min %.3f, median %.3f, %d clips with none"
          % (min(cov), float(np.median(cov)), sum(1 for c in cov if c == 0)))
    missing = [m["chunk"] for m in manifest if m["frames_extracted"] == 0]
    if missing:
        print("no descriptor at all:", missing)


if __name__ == "__main__":
    main()
