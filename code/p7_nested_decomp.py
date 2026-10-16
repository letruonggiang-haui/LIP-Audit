# -*- coding: utf-8 -*-
"""P7 - nested decomposition of EngageNet label and score variance:
participant / recording-within-participant / chunk-within-recording.

Internal-review item R2-M6 (25/09/2026): Eq. (6) attributes to the participant every
component of variance that is constant within a participant, but chunks are
nested in recordings (session code + video index). This script splits the
between-participant share into a participant part and a recording-within-
participant part, for the labels of each split and for the ordinal score of
each reimplemented system (posterior mean level, as in p5_person_variance_ci),
and repeats the label decomposition for single- and multi-recording
participants separately. Runs on the released label files and the stored
per-chunk probabilities; nothing is retrained and the test split is not read
again (the stored test predictions carry their labels from the single read).

  python code/p7_nested_decomp.py  -> results_p4/json/p7_nested_decomp.json
"""
import glob
import io
import json
import os

import numpy as np
import pandas as pd

import p4_labels as L
from label_join import engagenet_labels, engagenet_root, load_probs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(ROOT, "staging_p4")
# probabilities: the project staging copy if present, else the repository's engagenet/probs/
_CANDIDATES = [os.path.join(ROOT, "release_lip_audit_stage", "engagenet", "probs"),
               os.path.join(ROOT, "engagenet", "probs")]
PROBS = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[-1])
_RES = os.path.join(ROOT, "results_p4", "json")
OUT = os.path.join(_RES if os.path.isdir(_RES) else os.path.join(ROOT, "results"), "p7_nested_decomp.json")
LEVELS = np.arange(4, dtype=float)


def nested(v, subj, rec):
    """Sums of squares of v across participant, recording|participant, chunk|recording."""
    v = np.asarray(v, float)
    gm = v.mean()
    ss_tot = ((v - gm) ** 2).sum()
    ss_p = 0.0
    ss_r = 0.0
    for s in np.unique(subj):
        ks = subj == s
        ms = v[ks].mean()
        ss_p += ks.sum() * (ms - gm) ** 2
        for r in np.unique(rec[ks]):
            kr = ks & (rec == r)
            ss_r += kr.sum() * (v[kr].mean() - ms) ** 2
    ss_w = ss_tot - ss_p - ss_r
    return dict(participant=ss_p / ss_tot, recording_within_participant=ss_r / ss_tot,
                chunk_within_recording=ss_w / ss_tot,
                between_participant_eq6=ss_p / ss_tot,
                between_recording_total=(ss_p + ss_r) / ss_tot)


def load_labels(split):
    """Labels of one split: the project staging copy if present, else the released
    EngageNet label file under ENGAGENET_ROOT (None if that file is absent)."""
    path = os.path.join(LAB, "labels_%s.csv" % split)
    if os.path.isfile(path):
        df = pd.read_csv(path)
    elif engagenet_labels(split) is None:
        return None
    else:
        df = L.load_engagenet(split, engagenet_root())
    df = df[df["y"].notna()].copy()
    df["y"] = df["y"].astype(int)
    df["rec"] = df["subj"].astype(str) + "_" + df["sess"].astype(str) + "_" + df["vid"].astype(str)
    return df


def main():
    res = {"note": "shares of total variance; between_participant_eq6 is the share Eq. (6) "
                   "of the main text reports; between_recording_total adds the recording-"
                   "within-participant component"}
    for split in ("train", "validation", "test"):
        df = load_labels(split)
        if df is None:
            print("skip %s: no label file for this split under ENGAGENET_ROOT" % split)
            continue
        y = df["y"].to_numpy(float)
        subj = df["subj"].to_numpy()
        rec = df["rec"].to_numpy()
        n_rec_per = df.groupby("subj")["rec"].nunique()
        multi = df["subj"].map(n_rec_per) > 1
        r = {"n_chunks": int(len(df)), "n_participants": int(len(np.unique(subj))),
             "n_recordings": int(len(np.unique(rec))),
             "participants_multi_recording": int((n_rec_per > 1).sum()),
             "labels_all": nested(y, subj, rec),
             "labels_multi_recording_participants": nested(y[multi.to_numpy()], subj[multi.to_numpy()], rec[multi.to_numpy()]),
             "labels_single_recording_participants": (
                 {"between_participant_eq6": float(
                     nested(y[~multi.to_numpy()], subj[~multi.to_numpy()], rec[~multi.to_numpy()])["participant"])}
                 if (~multi).sum() > 0 else None),
             "binary_decisive_axis_labels": None}
        # alternative coding: binary decisive axis (level 3 vs 2 among y >= 2)
        m = y >= 2
        if m.sum() > 0:
            r["binary_decisive_axis_labels"] = nested((y[m] == 3).astype(float), subj[m], rec[m])
        # model scores: posterior mean level, one entry per system (mean of per-seed shares)
        pat = {"validation": "*_val_probs.npz", "test": "*_test_probs.npz"}.get(split)
        if pat:
            by_sys = {}
            for f in sorted(glob.glob(os.path.join(PROBS, pat))):
                tag = os.path.basename(f).split("_val_probs")[0].split("_test_probs")[0]
                if tag.startswith("smoke"):
                    continue
                z = load_probs(f)  # labels joined from the released label file
                P, sj, rc = z["P"], z["subj"], z["rec"]
                keep = np.ones(len(P), bool)
                score = (P * LEVELS).sum(1)
                sysname = tag.rsplit("_s", 1)[0]
                by_sys.setdefault(sysname, []).append(nested(score[keep], sj[keep], rc[keep].astype(str)))
                if "y" in z.files and sysname not in r.get("score_label_check", {}):
                    r.setdefault("score_label_check", {})[sysname] = nested(z["y"].astype(float), sj, rc.astype(str))["participant"]
            r["scores"] = {k: {kk: float(np.mean([d[kk] for d in v])) for kk in v[0]} for k, v in by_sys.items()}
        res[split] = r
        print("== %s: %d chunks, %d participants, %d recordings (%d with >1 recording)" % (
            split, r["n_chunks"], r["n_participants"], r["n_recordings"], r["participants_multi_recording"]))
        la = r["labels_all"]
        print("   labels: participant %.3f | recording|participant %.3f | chunk %.3f" % (
            la["participant"], la["recording_within_participant"], la["chunk_within_recording"]))
        lm = r["labels_multi_recording_participants"]
        print("   multi-recording participants only: participant %.3f | recording %.3f | chunk %.3f" % (
            lm["participant"], lm["recording_within_participant"], lm["chunk_within_recording"]))
        if r["binary_decisive_axis_labels"]:
            b = r["binary_decisive_axis_labels"]
            print("   binary decisive-axis labels: participant %.3f | recording %.3f" % (b["participant"], b["recording_within_participant"]))
        for k, v in r.get("scores", {}).items():
            print("   score %-12s participant %.3f | recording %.3f | chunk %.3f" % (
                k, v["participant"], v["recording_within_participant"], v["chunk_within_recording"]))
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
