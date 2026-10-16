"""P5/R6 - participant-clustered intervals for the person-level quantities.

Section V-C reports the between-participant share of a model's ordinal score
against the same share of the labels, and reads an ordering off point values
("every system is at least as person-driven as its target").  Three seats asked
for the interval that claim needs.  This script supplies it, on the stored
predictions only: no model is retrained and the sealed test split is not read
again, exactly as in p5_paired_bootstrap.py.

For each system and split it reports, with a participant-clustered percentile
bootstrap:

  eta2_b(score)          between-participant share of the posterior mean level
  eta2_b(label)          the same share for the label
  difference             eta2_b(score) - eta2_b(label), paired inside a replicate
  rho_b                  Spearman between per-participant mean decisive score
                         and per-participant fraction of the upper class
  auc_within_person      the decisive-axis within-person AUC

A bootstrap replicate draws participants with replacement and treats each draw
as a separate group, so a participant drawn twice contributes two groups; this
keeps the between-group sum of squares from being inflated by the duplication.

Output: ~/p4/analysis/p5_person_variance_ci.json
"""
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from label_join import load_probs

# the server copy if present, else the repository's engagenet/ directory
P4 = os.environ.get("P4_DIR") or next(
    (p for p in (os.path.expanduser("~/p4"),
                 os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engagenet"))
     if os.path.isdir(os.path.join(p, "probs"))), os.path.expanduser("~/p4"))
OUT = os.path.join(P4, "analysis")
LEVELS = np.arange(4)
B = 2000
SEED = 20260910


def eta2_between(v, grp):
    gm = v.mean()
    sst = ((v - gm) ** 2).sum()
    if sst <= 0:
        return float("nan")
    ssb = 0.0
    for g in np.unique(grp):
        k = grp == g
        ssb += k.sum() * (v[k].mean() - gm) ** 2
    return float(ssb / sst)


def within_person_auc(score, target, subj):
    aucs, ns = [], 0
    for s in np.unique(subj):
        k = subj == s
        t = target[k]
        if t.min() == t.max():
            continue
        aucs.append(roc_auc_score(t, score[k]))
        ns += 1
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), ns


def rho_between(score, target, subj):
    us = np.unique(subj)
    if len(us) < 3:
        return float("nan")
    means = np.array([score[subj == s].mean() for s in us])
    frac = np.array([target[subj == s].mean() for s in us])
    if np.all(means == means[0]) or np.all(frac == frac[0]):
        return float("nan")
    return float(spearmanr(means, frac).statistic)


def point_stats(P, y, subj):
    ord_score = (P * LEVELS).sum(1)
    m = y >= 2
    sc = P[:, 3] / (P[:, 2] + P[:, 3] + 1e-9)
    t = (y[m] == 3).astype(int)
    s_dec, sj_dec = sc[m], subj[m]
    wp, n_sub = within_person_auc(s_dec, t, sj_dec)
    return {
        "eta2_score_ordinal": eta2_between(ord_score, subj),
        "eta2_label_ordinal": eta2_between(y.astype(float), subj),
        "eta2_score_decisive": eta2_between(s_dec, sj_dec),
        "eta2_label_decisive": eta2_between(t.astype(float), sj_dec),
        "rho_b": rho_between(s_dec, t, sj_dec),
        "auc_within_person": wp,
        "n_participants_usable": n_sub,
    }


def ci(vals, lo=2.5, hi=97.5):
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if len(vals) < 50:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]


def bootstrap(P, y, subj, rng, b=B):
    us = np.unique(subj)
    index = {s: np.flatnonzero(subj == s) for s in us}
    keys = ["eta2_score_ordinal", "eta2_label_ordinal", "eta2_score_decisive",
            "eta2_label_decisive", "rho_b", "auc_within_person"]
    draws = {k: [] for k in keys}
    draws["diff_ordinal"] = []
    draws["diff_decisive"] = []
    for _ in range(b):
        pick = rng.choice(us, size=len(us), replace=True)
        rows, grp = [], []
        for j, s in enumerate(pick):
            rows.append(index[s])
            grp.append(np.full(len(index[s]), j))
        rows = np.concatenate(rows)
        grp = np.concatenate(grp)
        st = point_stats(P[rows], y[rows], grp)
        for k in keys:
            draws[k].append(st[k])
        draws["diff_ordinal"].append(st["eta2_score_ordinal"]
                                     - st["eta2_label_ordinal"])
        draws["diff_decisive"].append(st["eta2_score_decisive"]
                                      - st["eta2_label_decisive"])
    return {k: ci(v) for k, v in draws.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    out = {"n_boot": B, "seed": SEED, "rows": []}

    for split, pat in (("validation", "*_val_probs.npz"),
                       ("test_frozen", "*_test_probs.npz")):
        for path in sorted(glob.glob(os.path.join(P4, "probs", pat))):
            tag = os.path.basename(path).split("_val_probs")[0].split("_test_probs")[0]
            z = load_probs(path)  # labels joined from the released label file (label_join.py)
            if "y" not in z:
                continue
            P, y, subj = z["P"], z["y"], z["subj"]
            pt = point_stats(P, y, subj)
            iv = bootstrap(P, y, subj, rng)
            row = {"tag": tag, "split": split, "n_clips": int(len(y)),
                   "n_participants": int(len(np.unique(subj))), "point": pt,
                   "ci": iv,
                   "diff_ordinal_point": pt["eta2_score_ordinal"]
                   - pt["eta2_label_ordinal"],
                   "diff_decisive_point": pt["eta2_score_decisive"]
                   - pt["eta2_label_decisive"]}
            out["rows"].append(row)
            print("%-12s %-18s eta2 score %.3f %s  label %.3f %s  diff %+.3f %s | "
                  "rho_b %.2f %s | within-person AUC %.3f %s"
                  % (split, tag,
                     pt["eta2_score_ordinal"], iv["eta2_score_ordinal"],
                     pt["eta2_label_ordinal"], iv["eta2_label_ordinal"],
                     row["diff_ordinal_point"], iv["diff_ordinal"],
                     pt["rho_b"], iv["rho_b"],
                     pt["auc_within_person"], iv["auc_within_person"]),
                  flush=True)

    with open(os.path.join(OUT, "p5_person_variance_ci.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", os.path.join(OUT, "p5_person_variance_ci.json"))


if __name__ == "__main__":
    main()
