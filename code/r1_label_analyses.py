# -*- coding: utf-8 -*-
"""R1 registration (REGISTRATION_R1_20261026.md), Sections 1, 2 and 3.

  kappa   1a  participant-clustered intervals for Audit 1
              DAiSEE: Table 2 pairing (rule S3.1, decisive axis) from the released label files,
              subjects resampled with all their sessions; EngageNet: the subject-clustered
              intervals already stored in results/E2_results.json (no label file read)
  coding  1b  between-person share of the labels under four codings, with the cumulative-logit
              mixed model checked on simulated data first
  equiv   2   equivalence bounds and minimum detectable differences from the stored paired
              intervals of results/p5_paired_bootstrap.json (no label read)
  resid   3   within-participant residuals by true label and the label-matched between-person
              share of scores, validation splits only

EngageNet labels are read from a directory that holds the train and validation label files only
(ENGAGENET_ROOT below); the test split is never requested.

Usage: python r1_label_analyses.py kappa|coding|equiv|resid|all
"""
import glob
import json
import os
import sys
import time

TRAINVAL_ONLY = r"E:\datasets\EngageNet_trainval_only"
assert sorted(os.listdir(TRAINVAL_ONLY)) == ["train_engagement_labels.xlsx", "validation_engagement_labels.xlsx"]
os.environ["ENGAGENET_ROOT"] = TRAINVAL_ONLY

ROOT = r"E:\OneDrive\R&D\Education\P2-Identity-Invariant-Engagement"
REPO = os.path.join(ROOT, "release_lip_audit")
sys.path.insert(0, os.path.join(REPO, "code"))
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit, logsumexp  # noqa: E402

import label_join as LJ  # noqa: E402
import p4_labels as L  # noqa: E402
import p7_audit1_daisee as A1  # noqa: E402

DAISEE_LAB = os.path.join(ROOT, "data_local", "daisee_labels")  # the copy Table 2 was computed from
OUT = os.path.join(ROOT, "paper", "review_R1_20261024", "results")
B = 2000
SEED = 20261026
os.makedirs(OUT, exist_ok=True)


def save(name, obj):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    print("wrote", name, flush=True)


def en_split(split):
    assert split in ("train", "validation"), split
    df = L.load_engagenet(split, TRAINVAL_ONLY)
    return df[df["y"].notna()].assign(y=lambda d: d["y"].astype(int)).reset_index(drop=True)


# ============================================================== 1a
def kappa():
    rng = np.random.default_rng(SEED)
    res = {"DAiSEE": {}, "EngageNet_stored_E2": {}}
    for split in ("Train", "Validation", "Test"):
        df = A1.load(split, DAISEE_LAB)
        pairs = A1.decisive(A1.pairs_for(df, "s31_allwidths"))
        dec = df[df["y"] >= 2].reset_index(drop=True)
        a, ch, k = A1.stats(pairs, dec["y"].to_numpy())
        # resample subjects with replacement; rebuild pairs and the chance term in each replicate
        subj_p = np.array([p[1] for p in pairs])
        keys = np.unique(np.concatenate([subj_p, dec["subj"].unique()]))
        by_p = {s: np.flatnonzero(subj_p == s) for s in keys}
        by_c = {s: np.flatnonzero(dec["subj"].to_numpy() == s) for s in keys}
        yd = dec["y"].to_numpy()
        ks = []
        for _ in range(B):
            draw = rng.choice(keys, len(keys), replace=True)
            pi = np.concatenate([by_p[s] for s in draw])
            ci = np.concatenate([by_c[s] for s in draw])
            if len(pi) < 10:
                continue
            ks.append(A1.stats([pairs[i] for i in pi], yd[ci])[2])
        res["DAiSEE"][split] = {"pairs": len(pairs), "a": a, "a_iid": ch, "kappa": k,
                                "n_subjects": int(len(keys)),
                                "kappa_ci_subject_clustered": [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))]}
        print("DAiSEE", split, res["DAiSEE"][split], flush=True)
    e2 = json.load(open(os.path.join(REPO, "results", "E2_results.json"), encoding="utf-8"))["EngageNet"]
    for split, d in e2.items():
        res["EngageNet_stored_E2"][split] = {
            ax: {"recording_clustered": d[ax]["ci_recording_clustered"], "subject_clustered": d[ax]["ci_subject_clustered"]}
            for ax in ("four_class", "decisive")}
    res["note"] = ("EngageNet intervals are copied from results/E2_results.json (generated 2026-09-04, "
                   "2,000 replicates, same pairing as E1); no EngageNet label file was read for them.")
    save("r1_1a_kappa_participant_clustered.json", res)


# ============================================================== 1b
GRID = np.linspace(-15.0, 15.0, 1201)
DW = GRID[1] - GRID[0]


def log_pk(theta):
    """log P(Y = k | w) on GRID for cumulative-logit thresholds theta (K-1)."""
    c = np.concatenate([[-np.inf], theta, [np.inf]])
    F = expit(c[:, None] - GRID[None, :])  # (K+1, G)
    P = np.clip(F[1:] - F[:-1], 1e-300, None)
    return np.log(P)  # (K, G)


def unpack(x, K, two_level):
    th = np.cumsum(np.concatenate([[x[0]], np.exp(x[1:K - 1])]))
    su = np.exp(x[K - 1])
    sv = np.exp(x[K]) if two_level else None
    return th, su, sv


def negll(x, counts_unit, unit_to_cluster, n_cluster, K, two_level):
    th, su, sv = unpack(x, K, two_level)
    lw = counts_unit @ log_pk(th)  # (units, G): log prod_i P(y_i | w)
    if not two_level:
        lphi = -0.5 * (GRID / su) ** 2 - np.log(su * np.sqrt(2 * np.pi)) + np.log(DW)
        return -float(logsumexp(lw + lphi[None, :], axis=1).sum())
    m = lw.max(1, keepdims=True)
    E = np.exp(lw - m)
    Kmat = np.exp(-0.5 * ((GRID[:, None] - GRID[None, :]) / sv) ** 2) / (sv * np.sqrt(2 * np.pi)) * DW
    with np.errstate(divide="ignore"):
        lI = np.log(E @ Kmat) + m  # (units, G over u)
    lc = np.zeros((n_cluster, len(GRID)))
    np.add.at(lc, unit_to_cluster, lI)
    lphi = -0.5 * (GRID / su) ** 2 - np.log(su * np.sqrt(2 * np.pi)) + np.log(DW)
    return -float(logsumexp(lc + lphi[None, :], axis=1).sum())


def fit_clmm(y, unit, cluster=None, K=4):
    """Cumulative-logit model with a random intercept per cluster (and, if `cluster` is given,
    a nested random intercept per unit). Returns thresholds, SDs and the latent ICC of the
    top level. Labels enter only through per-unit class counts."""
    two = cluster is not None
    uu, ui = np.unique(unit, return_inverse=True)
    counts = np.zeros((len(uu), K))
    np.add.at(counts, (ui, y), 1)
    if two:
        u2c = {}
        for a, b in zip(ui, cluster):
            u2c[a] = b
        cl, unit_to_cluster = np.unique([u2c[i] for i in range(len(uu))], return_inverse=True)
        ncl = len(cl)
    else:
        unit_to_cluster, ncl = None, len(uu)
    cum = np.cumsum(np.bincount(y, minlength=K))[:-1] / len(y)
    cum = np.clip(cum, 1e-3, 1 - 1e-3)
    th0 = np.log(cum / (1 - cum))
    th0 = np.maximum.accumulate(th0 + np.arange(K - 1) * 1e-3)
    x0 = np.concatenate([[th0[0]], np.log(np.maximum(np.diff(th0), 0.05)), [0.0], [np.log(0.5)] if two else []])
    r = minimize(negll, x0, args=(counts, unit_to_cluster, ncl, K, two), method="L-BFGS-B",
                 bounds=[(None, None)] + [(np.log(1e-3), None)] * (K - 2) + [(np.log(0.02), np.log(20))]
                 + ([(np.log(0.05), np.log(20))] if two else []))
    th, su, sv = unpack(r.x, K, two)
    v = su ** 2 + (sv ** 2 if two else 0.0) + np.pi ** 2 / 3
    out = {"thresholds": th.tolist(), "sd_cluster": float(su), "icc_latent": float(su ** 2 / v),
           "converged": bool(r.success), "negll": float(r.fun)}
    if two:
        out["sd_unit"] = float(sv)
        out["unit_share_latent"] = float(sv ** 2 / v)
    return out


def simulate_clmm(unit, cluster, thresholds, icc, sv=None, rng=None):
    K = len(thresholds) + 1
    vres = np.pi ** 2 / 3 + (sv ** 2 if sv is not None else 0.0)
    su = np.sqrt(icc * vres / (1 - icc))
    top = cluster if cluster is not None else unit
    tu, ti = np.unique(top, return_inverse=True)
    w = rng.normal(0, su, len(tu))[ti]
    if sv is not None:
        uu, ui = np.unique(unit, return_inverse=True)
        w = w + rng.normal(0, sv, len(uu))[ui]
    ystar = w + rng.logistic(0, 1, len(w))
    return np.searchsorted(np.asarray(thresholds), ystar).astype(int).clip(0, K - 1)


def ridit(y, K=4):
    p = np.bincount(y, minlength=K) / len(y)
    r = np.cumsum(p) - p / 2
    return r[y]


def coding():
    rng = np.random.default_rng(SEED + 1)
    data = {}
    for split in ("Train", "Validation", "Test"):
        d = L.load_daisee(split, DAISEE_LAB)
        data[("DAiSEE", split)] = (d["y"].to_numpy().astype(int), d["subj"].to_numpy(), None)
    for split in ("train", "validation"):
        d = en_split(split)
        data[("EngageNet", split)] = (d["y"].to_numpy(), d["rec"].to_numpy(), d["subj"].to_numpy())
    res = {"estimator_check": [], "splits": {}}
    # estimator check on the real structure of the two training splits
    for corpus, split in (("DAiSEE", "Train"), ("EngageNet", "train")):
        y, a, b = data[(corpus, split)]
        unit, cluster = (a, None) if b is None else (a, b)
        fit_real = fit_clmm(y, unit, cluster)
        for icc in (0.1, 0.3, 0.5):
            for rep in range(3):
                ys = simulate_clmm(unit, cluster, fit_real["thresholds"], icc,
                                   sv=(0.5 if cluster is not None else None), rng=rng)
                f = fit_clmm(ys, unit, cluster)
                res["estimator_check"].append({"structure": "%s %s" % (corpus, split), "true_icc": icc,
                                               "rep": rep, "estimated_icc": f["icc_latent"]})
                print("check", corpus, icc, rep, round(f["icc_latent"], 3), flush=True)
    for (corpus, split), (y, a, b) in data.items():
        if b is None:  # DAiSEE: subject is the group
            subj, unit, cluster = a, a, None
        else:
            subj, unit, cluster = b, a, b
        keep = np.ones(len(y), bool)
        dec = y >= 2
        row = {"n_clips": int(len(y)), "n_participants": int(len(np.unique(subj))),
               "eta2_equal_interval": L.eta2_between(y.astype(float), keep, subj),
               "eta2_decisive": L.eta2_between((y[dec] == 3).astype(float), np.ones(dec.sum(), bool), subj[dec]),
               "eta2_ridit": L.eta2_between(ridit(y), keep, subj),
               "clmm": fit_clmm(y, unit, cluster)}
        res["splits"]["%s %s" % (corpus, split)] = row
        print(corpus, split, json.dumps({k: (v if not isinstance(v, dict) else v["icc_latent"]) for k, v in row.items()}), flush=True)
    save("r1_1b_coding_sensitivity.json", res)


# ============================================================== 2
def equiv():
    d = json.load(open(os.path.join(REPO, "results", "p5_paired_bootstrap.json"), encoding="utf-8"))
    res = {}
    for block, v in d.items():
        rows = {}
        for pair, q in v["paired"].items():
            r = {}
            for met in ("acc", "uar"):
                lo, hi = q["d_%s_ci" % met]
                se = (hi - lo) / 3.92
                r[met] = {"d": q["d_%s" % met], "ci95": [lo, hi],
                          "equivalence_bound": max(abs(lo), abs(hi)),
                          "mdd80": (1.96 + 0.84) * se}
            rows[pair] = r
        res[block] = {"n_participants": v["n_participants"], "pairs": rows,
                      "max_equivalence_bound_acc": max(r["acc"]["equivalence_bound"] for r in rows.values()),
                      "max_mdd80_acc": max(r["acc"]["mdd80"] for r in rows.values()),
                      "max_equivalence_bound_uar": max(r["uar"]["equivalence_bound"] for r in rows.values()),
                      "max_mdd80_uar": max(r["uar"]["mdd80"] for r in rows.values())}
        print(block, {k: round(x, 4) for k, x in res[block].items() if k.startswith("max")}, flush=True)
    res["definitions"] = ("equivalence_bound = max(|lower|, |upper|) of the stored 95% paired participant-clustered "
                          "interval (differences beyond it excluded at 2.5% per side, a conservative TOST); "
                          "mdd80 = 2.80 x SE with SE = (upper - lower)/3.92 (normal approximation).")
    save("r1_2_equivalence_mdd.json", res)


# ============================================================== 3
def weighted_eta2(s, g, w):
    m = np.average(s, weights=w)
    sst = (w * (s - m) ** 2).sum()
    ssb = 0.0
    for u in np.unique(g):
        k = g == u
        ssb += w[k].sum() * (np.average(s[k], weights=w[k]) - m) ** 2
    return float(ssb / sst)


def resid_stats(score, t, subj):
    """(a) mean within-participant residual for t=1 and t=0; (b) label-matched eta2."""
    r = np.empty_like(score)
    for u in np.unique(subj):
        k = subj == u
        r[k] = score[k] - score[k].mean()
    a1, a0 = float(r[t == 1].mean()), float(r[t == 0].mean())
    both = np.array([len(np.unique(t[subj == u])) == 2 for u in subj])
    s2, t2, g2 = score[both], t[both], subj[both]
    pooled = np.array([1 - t2.mean(), t2.mean()])
    w = np.empty(len(t2))
    for u in np.unique(g2):
        k = g2 == u
        pu = np.array([1 - t2[k].mean(), t2[k].mean()])
        w[k] = pooled[t2[k]] / pu[t2[k]]
    return {"resid_mean_t1": a1, "resid_mean_t0": a0, "resid_gap": a1 - a0,
            "eta2_score_unmatched": float(L.eta2_between(score, np.ones(len(score), bool), subj)),
            "eta2_score_label_matched": weighted_eta2(s2, g2, w),
            "n_participants_matched": int(len(np.unique(g2)))}


def resid_boot(score, t, subj, rng):
    us = np.unique(subj)
    pos = {u: np.flatnonzero(subj == u) for u in us}
    out = {"resid_gap": [], "eta2_score_label_matched": []}
    for _ in range(B):
        draw = rng.choice(us, len(us), replace=True)
        idx = np.concatenate([pos[u] for u in draw])
        rel = np.concatenate([np.full(len(pos[u]), j) for j, u in enumerate(draw)])
        st = resid_stats(score[idx], t[idx], rel)
        for k in out:
            out[k].append(st[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in out.items()}


def resid():
    rng = np.random.default_rng(SEED + 3)
    res = {"DAiSEE_validation": {}, "EngageNet_validation": {}}
    for f in sorted(glob.glob(os.path.join(REPO, "logs", "val_probs", "*_val_probs.npz"))):
        z = LJ.daisee_probs(f, DAISEE_LAB, "Validation")
        P, y = z["P"], z["y"]
        subj = np.array([str(s)[:6] for s in z["ids"]])
        m = y >= 2
        sc = P[m, 3] / (P[m, 2] + P[m, 3] + 1e-9)
        t, sj = (y[m] == 3).astype(int), subj[m]
        st = resid_stats(sc, t, sj)
        st["ci"] = resid_boot(sc, t, sj, rng)
        res["DAiSEE_validation"][os.path.basename(f)] = st
        print(os.path.basename(f), round(st["resid_gap"], 4), round(st["eta2_score_label_matched"], 4), flush=True)
    for f in sorted(glob.glob(os.path.join(REPO, "engagenet", "probs", "of_*_val_probs.npz"))):
        z = LJ.load_probs(f, TRAINVAL_ONLY)
        P, y, subj = z["P"], z["y"], np.asarray(z["subj"])
        m = y >= 2
        sc = P[m, 3] / (P[m, 2] + P[m, 3] + 1e-9)
        t, sj = (y[m] == 3).astype(int), subj[m]
        st = resid_stats(sc, t, sj)
        st["ci"] = resid_boot(sc, t, sj, rng)
        res["EngageNet_validation"][os.path.basename(f)] = st
        print(os.path.basename(f), round(st["resid_gap"], 4), round(st["eta2_score_label_matched"], 4), flush=True)
    save("r1_3_residuals.json", res)


if __name__ == "__main__":
    t0 = time.time()
    what = sys.argv[1]
    for name, fn in (("kappa", kappa), ("equiv", equiv), ("resid", resid), ("coding", coding)):
        if what in (name, "all"):
            fn()
    print("minutes %.1f" % ((time.time() - t0) / 60))
