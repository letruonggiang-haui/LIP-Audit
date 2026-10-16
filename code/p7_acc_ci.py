# -*- coding: utf-8 -*-
"""P7 - participant-clustered 95% intervals of accuracy for each reimplemented
EngageNet system (seed mean inside each replicate), validation and the
frozen-checkpoint test read, for Table 5 of the main text (internal-review item
EIC-W2). Stored probabilities only; nothing retrained, test not re-read.

  python code/p7_acc_ci.py -> results_p4/json/p7_acc_ci.json
"""
import glob
import io
import json
import os

import numpy as np

from label_join import load_probs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# probabilities: the project staging copy if present, else the repository's engagenet/probs/
_CANDIDATES = [os.path.join(ROOT, "release_lip_audit_stage", "engagenet", "probs"),
               os.path.join(ROOT, "engagenet", "probs")]
PROBS = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[-1])
_RES = os.path.join(ROOT, "results_p4", "json")
OUT = os.path.join(_RES if os.path.isdir(_RES) else os.path.join(ROOT, "results"), "p7_acc_ci.json")
B, SEED = 2000, 20260925
SYSTEMS = ["of_mlp", "of_lstm", "of_cnnlstm", "of_tcn", "of_transformer", "marlin"]


def main():
    rng = np.random.default_rng(SEED)
    out = {"n_boot": B, "seed": SEED, "rows": []}
    for split, suffix in (("validation", "_val_probs.npz"), ("test_frozen", "_test_probs.npz")):
        for sysname in SYSTEMS:
            files = sorted(glob.glob(os.path.join(PROBS, sysname + "_s*" + suffix)))
            if not files:
                continue
            runs = [load_probs(f) for f in files]  # labels joined from the released label file
            if "y" not in runs[0].files:
                print("skip %s: no label file for this split under ENGAGENET_ROOT (labels are not redistributed)" % split)
                break
            y, subj = runs[0]["y"], runs[0]["subj"]
            preds = [z["P"].argmax(1) for z in runs]
            hit = np.mean([(p == y) for p in preds], axis=0)  # per-chunk seed-mean correctness
            point = float(hit.mean())
            us = np.unique(subj)
            index = {s: np.flatnonzero(subj == s) for s in us}
            vals = []
            for _ in range(B):
                pick = rng.choice(us, size=len(us), replace=True)
                rows = np.concatenate([index[s] for s in pick])
                vals.append(float(hit[rows].mean()))
            lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
            out["rows"].append(dict(system=sysname, split=split, seeds=len(files), acc=point, ci=[lo, hi]))
            print("%-11s %-15s acc %.3f [%.3f, %.3f]" % (split, sysname, point, lo, hi), flush=True)
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
