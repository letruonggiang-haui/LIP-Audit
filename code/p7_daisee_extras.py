# -*- coding: utf-8 -*-
"""P7 - two small DAiSEE checks from the internal review round (R1-m9, R1-m3), from the
released label files and the stored sealed-test probabilities only.

  leave-one-out Acc_maj : the constant-per-subject oracle of Section V-A, with
                          each clip's own label removed from its subject's mode
                          (validation and test, decisive-axis clips y >= 2)
  within-person AUC_23  : clip-count-weighted (Eq. 5), unweighted, and
                          positive-negative-pair-weighted means over the 21 test
                          subjects, on the stored ensemble probabilities

  python code/p7_daisee_extras.py -> results_p4/json/p7_daisee_extras.json
"""
import io
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(ROOT, "data_local", "daisee_labels")
LOGS = os.path.join(ROOT, "release_repo", "logs")
OUT = os.path.join(ROOT, "results_p4", "json", "p7_daisee_extras.json")


def labels(split):
    df = pd.read_csv(os.path.join(LAB, "%sLabels.csv" % split))
    df.columns = [c.strip() for c in df.columns]
    df["stem"] = df["ClipID"].astype(str).str.strip().str.replace(r"\.(avi|mp4)$", "", regex=True)
    df["subj"] = df["stem"].str[:6]
    df["y"] = df["Engagement"].astype(int)
    return df


def acc_maj(y, subj, loo):
    hit = 0
    for s in np.unique(subj):
        k = np.flatnonzero(subj == s)
        yy = y[k]
        if not loo:
            vals, cnt = np.unique(yy, return_counts=True)
            hit += cnt.max()
        else:
            for i in range(len(yy)):
                other = np.delete(yy, i)
                if len(other) == 0:
                    continue
                vals, cnt = np.unique(other, return_counts=True)
                mode = vals[np.argmax(cnt)]  # ties: lowest level (deterministic)
                hit += int(yy[i] == mode)
    return hit / len(y)


def main():
    res = {}
    for split in ("Validation", "Test"):
        df = labels(split)
        d = df[df["y"] >= 2]
        y = d["y"].to_numpy(); subj = d["subj"].to_numpy()
        floor = max((y == 2).mean(), (y == 3).mean())
        res["acc_maj_%s" % split.lower()] = dict(
            decisive_clips=int(len(y)), subjects=int(len(np.unique(subj))), floor=round(float(floor), 4),
            in_sample=round(acc_maj(y, subj, False), 4), leave_one_out=round(acc_maj(y, subj, True), 4))
        print(split, res["acc_maj_%s" % split.lower()])
    # within-person AUC_23 on the stored sealed-test ensemble
    z = np.load(os.path.join(LOGS, "TEST_FINAL_probs.npz"), allow_pickle=True)
    ids = [str(s) for s in np.load(os.path.join(LOGS, "Test_ids.npy"), allow_pickle=True)]
    lab = labels("Test"); eng = dict(zip(lab["stem"], lab["y"]))
    stems = [s for s in ids if s in eng]
    assert len(stems) == len(z["ens"]), (len(stems), len(z["ens"]))
    P = z["ens"]; y = np.array([eng[s] for s in stems]); subj = np.array([s[:6] for s in stems])
    if "y" in z.files:
        assert (y == z["y"]).all()
    m = y >= 2
    score = P[:, 3] / (P[:, 2] + P[:, 3] + 1e-9)
    per = []
    for s in np.unique(subj[m]):
        k = m & (subj == s)
        t = (y[k] == 3).astype(int)
        if t.min() == t.max():
            continue
        per.append(dict(subj=s, n=int(k.sum()), pairs=int(t.sum() * (len(t) - t.sum())), auc=float(roc_auc_score(t, score[k]))))
    n = np.array([p["n"] for p in per]); pr = np.array([p["pairs"] for p in per]); a = np.array([p["auc"] for p in per])
    res["within_person_auc23_test_ensemble"] = dict(
        subjects_with_both_classes=int(len(per)),
        clip_weighted=round(float((n * a).sum() / n.sum()), 4), unweighted=round(float(a.mean()), 4),
        pair_weighted=round(float((pr * a).sum() / pr.sum()), 4))
    print("within-person AUC23 (test ensemble): clip-weighted %.3f | unweighted %.3f | pair-weighted %.3f over %d subjects" % (
        res["within_person_auc23_test_ensemble"]["clip_weighted"], res["within_person_auc23_test_ensemble"]["unweighted"],
        res["within_person_auc23_test_ensemble"]["pair_weighted"], len(per)))
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
