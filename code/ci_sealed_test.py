# -*- coding: utf-8 -*-
"""Reproduce the sealed-test table of the article from the stored predictions.

Inputs (all in this repository except the released DAiSEE label file):
  logs/TEST_FINAL_probs.npz   per-clip class probabilities of the three seeds (P0, P1, P2),
                              their average (ens) and the clip identifiers (ids), in the
                              order of logs/Test_ids.npy restricted to labeled clips; no labels
  logs/Test_ids.npy           clip identifiers in extraction order (1,866 clips on disk)
  <DAiSEE>/Labels/TestLabels.csv   official test labels (1,784 labelled clips)

Outputs: accuracy, UAR, macro-F1, quadratic-weighted kappa, AUC_dis and AUC_23 per seed
and for the ensemble; subject-clustered 95 % bootstrap intervals of the ensemble (2,000
replicates, subjects resampled with replacement, seed 3); per-class recall; margins over
the majority-class and random-guess predictors; within-person AUC_23 with its within-subject
permutation p; Spearman rho_b between a subject's mean score and its share of level-3
labels; the session-majority oracle.

  python ci_sealed_test.py --labels E:/datasets/DAiSEE/DAiSEE/Labels
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from label_join import daisee_sealed_test

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "logs")


def load_labels(labels_dir):
    path = os.path.join(labels_dir, "TestLabels.csv")
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    lab = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    lab.columns = [c.strip() for c in lab.columns]
    lab["stem"] = lab["ClipID"].astype(str).str.strip().str.replace(r"\.(avi|mp4)$", "", regex=True)
    return dict(zip(lab["stem"], lab["Engagement"].astype(int)))


def metrics(P, y):
    pred = P.argmax(1)
    rec = [float((pred[y == c] == c).mean()) if (y == c).any() else float("nan") for c in range(4)]
    m = y >= 2
    return dict(acc=float((pred == y).mean()), uar=float(np.nanmean(rec)),
                f1=float(f1_score(y, pred, average="macro")),
                kw=float(cohen_kappa_score(y, pred, weights="quadratic")),
                auc_dis=float(roc_auc_score((y >= 2).astype(int), P[:, 2] + P[:, 3])),
                auc23=float(roc_auc_score((y[m] == 3).astype(int), P[m, 3] / (P[m, 2] + P[m, 3] + 1e-9))),
                recall=rec)


def within_person(P, y, subj, n_perm=1000, seed=0):
    m = y >= 2
    s = P[m, 3] / (P[m, 2] + P[m, 3] + 1e-9)
    yy, sj = (y[m] == 3).astype(int), subj[m]
    rng = np.random.default_rng(seed)

    def wauc(labels):
        num, den = 0.0, 0
        for u in np.unique(sj):
            k = sj == u
            if labels[k].min() == labels[k].max():
                continue
            num += roc_auc_score(labels[k], s[k]) * k.sum()
            den += k.sum()
        return num / den
    obs = wauc(yy)
    draws = []
    for _ in range(n_perm):
        yp = yy.copy()
        for u in np.unique(sj):
            k = np.flatnonzero(sj == u)
            yp[k] = rng.permutation(yy[k])
        draws.append(wauc(yp))
    p = (int((np.asarray(draws) >= obs).sum()) + 1) / (n_perm + 1)
    means = pd.DataFrame(dict(s=s, y=yy, sj=sj)).groupby("sj").mean()
    rho = spearmanr(means["s"], means["y"]).correlation
    return obs, p, float(rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="DAiSEE Labels directory")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    # labels are joined from TestLabels.csv on the clip identifiers (label_join.py);
    # the deposited probability file carries no labels
    d = daisee_sealed_test(LOGS, args.labels)
    y, stems = d["y"], [str(s) for s in d["stems"]]
    subj = np.array([s[:6] for s in stems])
    sess = np.array([s[:7] for s in stems])
    print("clips %d, subjects %d, sessions %d" % (len(y), len(np.unique(subj)), len(np.unique(sess))))

    for name, P in (("Seed 0", d["P0"]), ("Seed 1", d["P1"]), ("Seed 2", d["P2"]), ("Ensemble", d["ens"])):
        m = metrics(P, y)
        print("%-9s acc %.3f  UAR %.3f  F1 %.3f  kw %.3f  AUCdis %.3f  AUC23 %.3f  recall %s"
              % (name, m["acc"], m["uar"], m["f1"], m["kw"], m["auc_dis"], m["auc23"],
                 "/".join("%.3f" % r for r in m["recall"])))

    P = d["ens"]
    floor = np.bincount(y, minlength=4).max() / len(y)
    rand = ((np.bincount(y, minlength=4) / len(y)) ** 2).sum()
    acc = (P.argmax(1) == y).mean()
    print("majority floor %.3f  random guess %.3f  margins %.1f / %.1f points"
          % (floor, rand, 100 * (acc - floor), 100 * (acc - rand)))

    rng = np.random.default_rng(3)
    us = np.unique(subj)
    idx_by = {s: np.flatnonzero(subj == s) for s in us}
    keys = ("acc", "uar", "f1", "kw", "auc_dis", "auc23")
    draws = {k: [] for k in keys}
    for _ in range(args.n_boot):
        sel = np.concatenate([idx_by[s] for s in rng.choice(us, len(us), replace=True)])
        mm = metrics(P[sel], y[sel])
        for k in keys:
            draws[k].append(mm[k])
    for k in keys:
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        print("ensemble %-8s 95%% subject-clustered CI [%.3f, %.3f]" % (k, lo, hi))

    w, p, rho = within_person(P, y, subj)
    print("within-person AUC23 %.3f (permutation p = %.3f); rho_b %.2f" % (w, p, rho))
    oracle = np.mean([(y[sess == u] == np.bincount(y[sess == u]).argmax()).mean() * (sess == u).sum()
                      for u in np.unique(sess)]) * len(np.unique(sess)) / len(y)
    print("session-majority oracle accuracy %.3f" % oracle)


if __name__ == "__main__":
    main()
