"""E3a integrity gate: does the frame count we extracted agree with the
sequence length of the released MARLIN tensor, clip by clip?

The reducer deletes each OpenFace CSV once it has turned it into a vector, and
it decides a CSV is finished by its modification time. If it ever sweeps a file
while OpenFace is still writing, the descriptor is computed on a prefix of the
clip and nothing downstream would notice: the vector has the right shape and a
plausible magnitude.

The released MARLIN tensors give an independent handle. Their length is 9 for a
clip the decoder reads as roughly 300 frames and 312 for one it reads as
roughly 8,900, and the two populations are disjoint. Every clip therefore has a
predicted population, and a clip that lands in the wrong one is a clip whose
extraction was cut short.

`labels_<split>.csv` carries `marlin_seq_len`, written by `p4_pack_marlin.py`,
so this runs on the server without the tensors themselves.

Exit code 1 if any clip fails, so the campaign script stops instead of spending
nine hours on the same settings.

Usage:  python p4_check_frames.py Validation
"""
import os
import sys

import numpy as np
import pandas as pd

P4 = os.path.expanduser("~/p4")
LONG_MARLIN = 50      # a MARLIN length above this marks the long population
LONG_FRAMES = 2000    # a frame count above this marks the long population
SHORT_FRACTION = 0.5   # of the clip's own recording median


def main():
    split = sys.argv[1]
    lab = pd.read_csv(os.path.join(P4, "data", "labels_%s.csv" % split.lower()))
    lab["chunk"] = lab["chunk"].astype(str)
    if "marlin_seq_len" not in lab.columns:
        sys.exit("labels_%s.csv has no marlin_seq_len column; repack with "
                 "p4_pack_marlin.py" % split.lower())

    d = os.path.join(P4, "feats", "percli", split)
    rows = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npz"):
            continue
        z = np.load(os.path.join(d, f))
        rows.append((f[:-4] + ".mp4", int(z["frames_total"]), int(z["frames_ok"])))
    got = pd.DataFrame(rows, columns=["chunk", "frames_total", "frames_ok"])
    m = lab.merge(got, on="chunk", how="inner")
    print("%s: %d labelled chunks, %d reduced, %d joined"
          % (split, len(lab), len(got), len(m)))

    m["marlin_long"] = m["marlin_seq_len"] > LONG_MARLIN
    m["frames_long"] = m["frames_total"] > LONG_FRAMES
    bad_short = m[m["marlin_long"] & ~m["frames_long"]]
    bad_long = m[~m["marlin_long"] & m["frames_long"]]
    # Short is relative, not absolute. EngageNet was captured at heterogeneous
    # frame rates: of 95 participants, 69 give 300 frames for a ten-second chunk
    # and the rest give 75, 100, 125, 150, 166, 196 and 250, which is 7.5 to 25
    # frames per second. An absolute floor would condemn whole participants. A
    # truncated extraction, in contrast, makes one clip short against the other
    # clips of its own recording. The last chunk of a recording is whatever was
    # left over, so it is exempt.
    m["rec"] = (m["subj"].astype(str) + "|" + m["sess"].astype(str) + "|"
                + m["vid"].astype(str))
    m["is_last"] = m["t"] == m.groupby("rec")["t"].transform("max")
    rec_median = m.groupby("rec")["frames_total"].transform("median")
    # Clips OpenFace aborts on are declared in the manifest and exempt here;
    # p4_verify_frames.py owns that list and prints it on every run.
    manifest = os.path.join(P4, "logs", "partial_%s.json" % split)
    declared = set()
    if os.path.exists(manifest):
        import json
        declared = set(x["chunk"] for x in json.load(open(manifest)))
        print("  declared partial and exempt: %d clips" % len(declared))
    truncated = m[~m["marlin_long"] & ~m["is_last"]
                  & ~m["chunk"].isin(declared)
                  & (m["frames_total"] < SHORT_FRACTION * rec_median)]
    noface = m[m["frames_ok"] == 0]

    fps = (m.loc[m["frames_total"] < LONG_FRAMES]
           .groupby("subj")["frames_total"].median() / 10.0).round(1)
    print("  inferred capture rate, frames per second, over %d participants: %s"
          % (len(fps), dict(sorted(fps.value_counts().items()))))

    print("  long population: MARLIN says %d, frame count says %d"
          % (int(m["marlin_long"].sum()), int(m["frames_long"].sum())))
    print("  frames_total: median %.0f, 90th pct %.0f, max %d"
          % (m["frames_total"].median(), m["frames_total"].quantile(0.9),
             m["frames_total"].max()))
    print("  tracked-frame rate: median %.3f, min %.3f"
          % ((m["frames_ok"] / m["frames_total"].clip(lower=1)).median(),
             (m["frames_ok"] / m["frames_total"].clip(lower=1)).min()))
    print("  clips with no tracked frame: %d" % len(noface))

    fail = False
    for name, sub in (("long by MARLIN but short by frame count", bad_short),
                      ("short by MARLIN but long by frame count", bad_long),
                      ("interior clips under half their own recording median", truncated)):
        if len(sub):
            fail = True
            print("  FAIL %s: %d clips, first: %s"
                  % (name, len(sub), sub["chunk"].head(3).tolist()))
    if len(m) != len(lab):
        fail = True
        print("  FAIL only %d of %d labelled chunks reduced" % (len(m), len(lab)))

    if fail:
        print("CHECK FAILED for %s" % split)
        sys.exit(1)
    print("CHECK PASSED for %s" % split)


if __name__ == "__main__":
    main()
