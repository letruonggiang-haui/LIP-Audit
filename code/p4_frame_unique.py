"""For one 1000-fps-header clip: are the 10,000 decoded frames distinct images
or a ~300-frame video re-emitted at a 1 ms timebase?

Decodes every frame, downsamples it to 32x18 grey, hashes it, and counts
distinct hashes and the run lengths of identical consecutive frames. If the
media is ordinary 30 fps video under a wrong timebase, there will be ~300
distinct frames in runs of ~33 identical ones.

Usage:  python p4_frame_unique.py <clip.mp4>
"""
import hashlib
import sys

import cv2
import numpy as np


def main(path):
    c = cv2.VideoCapture(path)
    hashes, runs, prev, run = [], [], None, 0
    while True:
        ok, fr = c.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (32, 18))
        h = hashlib.md5(small.tobytes()).hexdigest()
        hashes.append(h)
        if h == prev:
            run += 1
        else:
            if prev is not None:
                runs.append(run)
            prev, run = h, 1
    runs.append(run)
    c.release()
    n = len(hashes)
    distinct = len(set(hashes))
    print("%s\n  decoded %d, distinct %d, runs of identical frames: median %d, min %d, max %d"
          % (path, n, distinct, int(np.median(runs)), min(runs), max(runs)))
    print("  implied real frame rate over 10 s: %.1f fps" % (distinct / 10.0))


if __name__ == "__main__":
    main(sys.argv[1])
