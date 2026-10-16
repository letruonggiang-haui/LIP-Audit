# -*- coding: utf-8 -*-
"""P7 - participant-clustered intervals for the person-level quantities of
Supplementary Table S15, with the seed mean taken INSIDE each replicate.

Internal-review item R1-W2 (25/09/2026): p5_person_variance_ci.py averaged the three
per-seed percentile intervals, which has no defined coverage. Here every
bootstrap replicate draws participants once, evaluates each of the system's
three seeds on that draw, averages the statistic over seeds, and the interval
is taken over replicates (the construction p5_paired_bootstrap.py already
uses). Quantities and estimators are those of p5_person_variance_ci.py:
eta2_b of the posterior-mean ordinal score and of the labels, their paired
difference, rho_b between per-participant mean decisive score and fraction of
level-3 labels, and the within-person decisive-axis AUC.

Runs on the stored per-chunk probabilities (validation; frozen-checkpoint test
read with its stored labels) and retrains nothing.

  python code/p7_person_variance_ci.py -> results_p4/json/p7_person_variance_ci.json
"""
import glob
import io
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from label_join import load_probs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# probabilities: the project staging copy if present, else the repository's engagenet/probs/
_CANDIDATES = [os.path.join(ROOT, "release_lip_audit_stage", "engagenet", "probs"),
               os.path.join(ROOT, "engagenet", "probs")]
PROBS = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[-1])
_RES = os.path.join(ROOT, "results_p4", "json")
OUT = os.path.join(_RES if os.path.isdir(_RES) else os.path.join(ROOT, "results"), "p7_person_variance_ci.json")
LEVELS = np.arange(4)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = 20260925
SYSTEMS = ["of_lstm", "of_cnnlstm", "of_tcn", "of_transformer", "of_mlp", "marlin"]


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
    aucs = []
    for s in np.unique(subj):
        k = subj == s
        t = target[k]
        if t.min() == t.max():
            continue
        aucs.append(roc_auc_score(t, score[k]))
    return float(np.mean(aucs)) if aucs else float("nan")


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
    e_s = eta2_between(ord_score, subj)
    e_l = eta2_between(y.astype(float), subj)
    return dict(eta2_score=e_s, eta2_label=e_l, diff=e_s - e_l,
                rho_b=rho_between(s_dec, t, sj_dec),
                auc_within=within_person_auc(s_dec, t, sj_dec))


def ci(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    rng = np.random.default_rng(SEED)
    out = {"n_boot": B, "seed": SEED, "construction": "seed mean inside each participant replicate", "rows": []}
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
            for z in runs[1:]:
                assert (z["y"] == y).all() and (z["subj"] == subj).all()
            Ps = [z["P"] for z in runs]
            # point estimate: seed mean of the statistic on the full sample
            pts = [point_stats(P, y, subj) for P in Ps]
            point = {k: float(np.mean([p[k] for p in pts])) for k in pts[0]}
            us = np.unique(subj)
            index = {s: np.flatnonzero(subj == s) for s in us}
            draws = {k: [] for k in pts[0]}
            for _ in range(B):
                pick = rng.choice(us, size=len(us), replace=True)
                rows = np.concatenate([index[s] for s in pick])
                grp = np.concatenate([np.full(len(index[s]), j) for j, s in enumerate(pick)])
                per_seed = [point_stats(P[rows], y[rows], grp) for P in Ps]
                for k in draws:
                    draws[k].append(float(np.nanmean([p[k] for p in per_seed])))
            ivs = {k: ci(v) for k, v in draws.items()}
            row = dict(system=sysname, split=split, seeds=len(files), n_chunks=int(len(y)),
                       n_participants=int(len(us)), point=point, ci=ivs)
            out["rows"].append(row)
            print("%-11s %-15s eta2 score %.3f [%.2f, %.2f]  label %.3f  diff %+.3f [%+.2f, %+.2f]  rho_b %.2f [%.2f, %.2f]  AUC_w %.3f [%.2f, %.2f]" % (
                split, sysname, point["eta2_score"], *ivs["eta2_score"], point["eta2_label"],
                point["diff"], *ivs["diff"], point["rho_b"], *ivs["rho_b"], point["auc_within"], *ivs["auc_within"]), flush=True)
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
