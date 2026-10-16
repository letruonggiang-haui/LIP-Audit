# -*- coding: utf-8 -*-
"""Recomputations requested by the independent review of 09/09/2026 that need
only the released label files (C1, M3, M4). Writes results_p4/json/review_recompute.json
and prints a short report.

  C1  noise ceiling from exact pair counts: q, 1-q on the binary axis, the
      four-level conversion 0.951*(1-q)+0.049 (minority levels granted), at the
      point estimate and both CI ends, for r = 0 and r = 0.2.
  M3  AUC required by Eq. (map) for 63.9 % and 67.48 % at eps = 0 and 0.05.
  M4  EngageNet adjacency without spanning a removed SNP chunk (strict) beside
      the published gap-tolerant version; Cohen's kappa with the chance term
      taken from the two pair columns' marginals beside the split-marginal
      version, on both benchmarks.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4_labels as L  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_p4", "json",
                   "review_recompute.json")


def pair_kappa(y, a_idx, b_idx, keep):
    """Agreement with chance from the pair columns' own marginals (Cohen's
    two-rater form) beside the split-marginal form of Eq. (agree)."""
    m = keep[a_idx] & keep[b_idx]
    ya, yb = y[a_idx[m]], y[b_idx[m]]
    obs = float(np.mean(ya == yb))
    levels = np.unique(np.r_[ya, yb])
    pa = np.array([(ya == k).mean() for k in levels])
    pb = np.array([(yb == k).mean() for k in levels])
    pe_pair = float((pa * pb).sum())
    _, cnt = np.unique(y[keep], return_counts=True)
    p = cnt / cnt.sum()
    pe_split = float((p ** 2).sum())
    return dict(pairs=int(m.sum()), a=obs,
                a_iid_split=pe_split, kappa_split=(obs - pe_split) / (1 - pe_split),
                a_iid_pair=pe_pair, kappa_pair=(obs - pe_pair) / (1 - pe_pair))


def q_of_a(a, r=0.0):
    """Solve a = (1-r)[(1-q)^2+q^2] + 2 r q (1-q) for q in [0, .5]."""
    # (1-r)(1 - 2q + 2q^2) + 2r(q - q^2) = a
    # q^2 [2(1-r) - 2r] + q [-2(1-r) + 2r] + (1-r) - a = 0
    A = 2 * (1 - r) - 2 * r
    B = -2 * (1 - r) + 2 * r
    C = (1 - r) - a
    disc = B * B - 4 * A * C
    if disc < 0:
        return float("nan")
    q = (-B - np.sqrt(disc)) / (2 * A)
    return float(q)


def main():
    res = {}
    # ---------------- C1: caps from exact counts ----------------
    caps = {}
    for split, ci in (("Validation", (.536, .630)), ("Test", (.495, .560))):
        d = L.load_daisee(split)
        d = d[d["y"].notna()].reset_index(drop=True)
        d, a1, b1 = L.build_pairs(d, "rec", strict=True, distance=1)
        keep = L.space_decisive(d)
        y = d["y"].values.astype(int)
        st = L.agreement(y, a1, b1, keep)
        # share of levels 2-3 in this split (four-level conversion weight)
        yy = d["y"].values
        share23 = float((yy >= 2).mean())
        row = dict(pairs=st["pairs"], a=st["a"], a_iid=st["a_iid"], kappa=st["kappa"],
                   share23=share23, ci=ci)
        for r in (0.0, 0.2):
            for name, a in (("point", st["a"]), ("ci_lo", ci[0]), ("ci_hi", ci[1])):
                q = q_of_a(a, r)
                cap2 = 1 - q
                cap4 = share23 * cap2 + (1 - share23)
                row["r%.1f_%s" % (r, name)] = dict(q=q, cap_binary=cap2, cap_four_level=cap4)
        caps[split] = row
    res["C1_caps"] = caps

    # ---------------- M3: AUC required by Eq. (map) ----------------
    def auc_needed(acc, eps, share=0.95):
        z = norm.ppf((acc - eps) / share)
        return float(norm.cdf(z * np.sqrt(2)))
    m3 = {}
    for acc in (0.639, 0.6748, 0.552):
        m3["%.4f" % acc] = {"eps0": auc_needed(acc, 0.0), "eps0.05": auc_needed(acc, 0.05)}
    res["M3_auc_required"] = m3

    # ---------------- M4: strict adjacency and pair-marginal kappa ----------------
    m4 = {"EngageNet": {}, "DAiSEE": {}}
    for split in ("train", "validation", "test"):
        e = L.load_engagenet(split)
        e = e[e["y"].notna()].reset_index(drop=True)
        y = e["y"].values.astype(int)
        keep = L.space_full(e)
        out = {}
        for name, strict in (("gap_tolerant_published", False), ("strict_no_snp_span", True)):
            e2, a1, b1 = L.build_pairs(e, "rec", strict=strict, distance=1)
            out[name] = pair_kappa(y, a1, b1, keep)
        m4["EngageNet"][split] = out
    for split in ("Validation", "Test"):
        d = L.load_daisee(split)
        d = d[d["y"].notna()].reset_index(drop=True)
        d, a1, b1 = L.build_pairs(d, "rec", strict=True, distance=1)
        y = d["y"].values.astype(int)
        m4["DAiSEE"][split] = pair_kappa(y, a1, b1, L.space_decisive(d))
    res["M4_adjacency"] = m4

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
