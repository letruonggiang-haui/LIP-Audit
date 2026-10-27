# -*- coding: utf-8 -*-
"""R1 registration (REGISTRATION_R1_20261026.md), Sections 4A and 4B.

  A  synthetic Audit 1: label sequences generated on the real recording structure of DAiSEE
     train and EngageNet train with a known state-switch rate r and label-flip rate q; Audit 1
     (four-level and decisive kappa, the code of record) applied to each dataset
  B  semi-synthetic Audit 2: two synthetic score models on the real participant structure of
     the EngageNet training labels (decisive axis), tuned to equal accuracy: S carries only a
     person offset, W only within-person signal; plus a mixture grid

Only training-split labels are read (DAiSEE TrainLabels.csv for the structure and marginal of A;
EngageNet train for A and B, from a directory that holds no test file).

Usage: python r1_controlled_sim.py A|B
"""
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

import p4_labels as L  # noqa: E402
import p7_audit1_daisee as A1  # noqa: E402
from p5_person_variance_ci import eta2_between, rho_between, within_person_auc  # noqa: E402

DAISEE_LAB = os.path.join(ROOT, "data_local", "daisee_labels")
OUT = os.path.join(ROOT, "paper", "review_R1_20261024", "results")
SEED = 20261026
GRID = (0.0, 0.05, 0.1, 0.2, 0.3)
NREP = 200
os.makedirs(OUT, exist_ok=True)


def save(name, obj):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    print("wrote", name, flush=True)


# ============================================================== A
def structure(corpus):
    """Positions of every labelled clip, grouped by recording, and the pair index of record."""
    if corpus == "DAiSEE":
        df = A1.load("Train", DAISEE_LAB)
        rec, pos, y = df["sess"].to_numpy(), df["pos"].to_numpy(), df["y"].to_numpy()
        # rule S3.1: two clips of one session whose positions differ by exactly one
        order = np.lexsort((pos, rec))
        a, b = [], []
        for i, j in zip(order[:-1], order[1:]):
            if rec[i] == rec[j] and pos[j] - pos[i] == 1:
                a.append(i)
                b.append(j)
    else:
        df = L.load_engagenet("train", TRAINVAL_ONLY)
        df = df[df["y"].notna()].reset_index(drop=True)
        d, a, b = L.build_pairs(df, strict=False)
        rec, pos, y = d["rec"].to_numpy(), d["t"].to_numpy(), d["y"].to_numpy().astype(int)
    return rec, pos.astype(int), y.astype(int), np.asarray(a), np.asarray(b)


class Timeline(object):
    """Every integer position of every recording, from its first to its last labelled clip,
    laid out flat (for the segment form) and padded (for the stepwise form)."""

    def __init__(self, rec, pos):
        self.us, inv = np.unique(rec, return_inverse=True)
        lo = np.array([pos[inv == i].min() for i in range(len(self.us))])
        hi = np.array([pos[inv == i].max() for i in range(len(self.us))])
        self.T = hi - lo + 1
        self.off = np.concatenate([[0], np.cumsum(self.T)[:-1]])
        self.flat_idx = self.off[inv] + (pos - lo[inv])        # clip -> flat timeline position
        self.row, self.col = inv, pos - lo[inv]                # clip -> padded (recording, step)
        self.L, self.Tmax = int(self.T.sum()), int(self.T.max())


def simulate(tl, pi, r, q, variant, rng):
    """One synthetic label vector for the clips of `tl`. Marginal variant: at each step the
    state is redrawn from the marginal with probability r. Boundary variant: with probability r
    it moves one ordinal level up or down (clipped at the ends). Each observed label is the state,
    or with probability q a different level drawn from the marginal."""
    K = len(pi)
    if variant == "marginal":
        sw = rng.random(tl.L) < r
        sw[tl.off] = True
        seg = np.cumsum(sw) - 1
        state = rng.choice(K, size=seg[-1] + 1, p=pi)[seg]
        y = state[tl.flat_idx]
    else:
        n = len(tl.us)
        S = np.empty((n, tl.Tmax), int)
        S[:, 0] = rng.choice(K, size=n, p=pi)
        move = rng.random((n, tl.Tmax)) < r
        step = rng.choice((-1, 1), size=(n, tl.Tmax))
        for t in range(1, tl.Tmax):
            S[:, t] = np.clip(S[:, t - 1] + np.where(move[:, t], step[:, t], 0), 0, K - 1)
        y = S[tl.row, tl.col]
    y = y.copy()
    flip = np.flatnonzero(rng.random(len(y)) < q)
    new = y[flip].copy()
    todo = np.ones(len(flip), bool)
    while todo.any():  # draw from the marginal until the level differs from the state
        draw = rng.choice(K, size=int(todo.sum()), p=pi)
        idx = np.flatnonzero(todo)
        ok = draw != y[flip][idx]
        new[idx[ok]] = draw[ok]
        todo[idx[ok]] = False
    y[flip] = new
    return y


def kap(y, a, b):
    full = L.agreement(y, a, b, np.ones(len(y), bool))["kappa"]
    dec = L.agreement(y, a, b, y >= 2)["kappa"]
    return full, dec


def run_A():
    rng = np.random.default_rng(SEED)
    res = {"grid": GRID, "n_rep": NREP, "observed": {
        "DAiSEE train decisive (Table 2)": 0.144, "DAiSEE validation decisive (Table 2)": 0.085,
        "DAiSEE test decisive (Table 2)": 0.053, "EngageNet train four-level (E1)": 0.534}, "cells": []}
    for corpus in ("DAiSEE", "EngageNet"):
        rec, pos, y, a, b = structure(corpus)
        tl = Timeline(rec, pos)
        pi = np.bincount(y, minlength=4) / len(y)
        obs_full, obs_dec = kap(y, a, b)
        res["observed_recomputed_%s" % corpus] = {"four_level": obs_full, "decisive": obs_dec, "pairs": int(len(a))}
        print(corpus, "pairs", len(a), "recordings", len(tl.us), "timeline", tl.L, "max length", tl.Tmax,
              "observed kappa", round(obs_full, 3), round(obs_dec, 3), flush=True)
        for variant in ("marginal", "boundary"):
            for r in GRID:
                for q in GRID:
                    ks = np.array([kap(simulate(tl, pi, r, q, variant, rng), a, b) for _ in range(NREP)])
                    cell = {"corpus": corpus, "variant": variant, "r": r, "q": q}
                    for j, name in enumerate(("four_level", "decisive")):
                        v = ks[:, j]
                        cell[name] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
                                      "p2.5": float(np.nanpercentile(v, 2.5)), "p97.5": float(np.nanpercentile(v, 97.5))}
                    res["cells"].append(cell)
                print(corpus, variant, "r", r, "done", flush=True)
    # H3: monotone decrease in r (fixed q) and in q (fixed r), on the replicate means
    mono = {}
    for corpus in ("DAiSEE", "EngageNet"):
        for variant in ("marginal", "boundary"):
            for name in ("four_level", "decisive"):
                tab = {(c["r"], c["q"]): c[name]["mean"] for c in res["cells"]
                       if c["corpus"] == corpus and c["variant"] == variant}
                in_r = all(tab[(GRID[i], q)] >= tab[(GRID[i + 1], q)] for q in GRID for i in range(len(GRID) - 1))
                in_q = all(tab[(r, GRID[i])] >= tab[(r, GRID[i + 1])] for r in GRID for i in range(len(GRID) - 1))
                mono["%s %s %s" % (corpus, variant, name)] = {"decreasing_in_r": in_r, "decreasing_in_q": in_q}
    res["H3_monotonicity"] = mono
    # H4: distinct (r, q) cells whose replicate 95% ranges overlap although the cells differ in
    # both directions (more switching / less flipping against the reverse)
    over = []
    for corpus in ("DAiSEE", "EngageNet"):
        name = "decisive" if corpus == "DAiSEE" else "four_level"
        cs = [c for c in res["cells"] if c["corpus"] == corpus and c["variant"] == "marginal"]
        for i, c1 in enumerate(cs):
            for c2 in cs[i + 1:]:
                if (c1["r"] - c2["r"]) * (c1["q"] - c2["q"]) < 0:
                    m1, m2 = c1[name], c2[name]
                    if abs(m1["mean"] - m2["mean"]) < min(m1["sd"], m2["sd"]):
                        over.append({"corpus": corpus, "axis": name, "cell_1": [c1["r"], c1["q"]],
                                     "cell_2": [c2["r"], c2["q"]], "mean_1": m1["mean"], "mean_2": m2["mean"]})
    res["H4_overlapping_cells"] = over
    save("r1_4A_synthetic_audit1.json", res)


# ============================================================== B
def run_B():
    rng = np.random.default_rng(SEED + 1)
    df = L.load_engagenet("train", TRAINVAL_ONLY)
    df = df[df["y"].notna() & (df["y"] >= 2)].reset_index(drop=True)
    t = (df["y"].to_numpy() == 3).astype(int)
    subj = df["subj"].to_numpy()
    n = np.array([np.sum(subj == s) for s in subj])
    tot = np.array([t[subj == s].sum() for s in subj])
    loo = np.where(n > 1, (tot - t) / np.maximum(n - 1, 1), 0.5)
    pmean = tot / n

    def metrics(score):
        wp, _ = within_person_auc(score, t, subj)
        return {"acc": float(((score > 0.5).astype(int) == t).mean()), "auc_within_person": wp,
                "eta2_score": eta2_between(score, subj), "rho_b": rho_between(score, t, subj)}

    sigma_s = 0.15

    def acc_of(sig_w, reps=40, seed=0):
        g = np.random.default_rng(seed)
        return np.mean([((0.5 + (t - pmean) + g.normal(0, sig_w, len(t)) > 0.5) == t).mean() for _ in range(reps)])

    g0 = np.random.default_rng(7)
    acc_s = np.mean([((loo + g0.normal(0, sigma_s, len(t)) > 0.5) == t).mean() for _ in range(40)])
    lo, hi = 0.01, 5.0
    for _ in range(40):  # bisection: accuracy of W falls as its noise grows
        mid = (lo + hi) / 2
        if acc_of(mid) > acc_s:
            lo = mid
        else:
            hi = mid
    sigma_w = (lo + hi) / 2
    reps = {"S": [], "W": []}
    lam_grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    mix = {lam: [] for lam in lam_grid}
    for _ in range(NREP):
        sS = loo + rng.normal(0, sigma_s, len(t))
        sW = 0.5 + (t - pmean) + rng.normal(0, sigma_w, len(t))
        reps["S"].append(metrics(sS))
        reps["W"].append(metrics(sW))
        for lam in lam_grid:
            mix[lam].append(metrics(lam * sS + (1 - lam) * sW))

    def summ(rows):
        return {k: {"mean": float(np.nanmean([r[k] for r in rows])),
                    "p2.5": float(np.nanpercentile([r[k] for r in rows], 2.5)),
                    "p97.5": float(np.nanpercentile([r[k] for r in rows], 97.5))} for k in rows[0]}
    res = {"n_clips": int(len(t)), "n_participants": int(len(np.unique(subj))),
           "sigma_S": sigma_s, "sigma_W_matched": sigma_w, "target_accuracy_S": float(acc_s),
           "S_person_shortcut": summ(reps["S"]), "W_within_person": summ(reps["W"]),
           "mixture_lambda_weight_on_S": {str(lam): summ(v) for lam, v in mix.items()}}
    s, w = res["S_person_shortcut"], res["W_within_person"]
    res["H5"] = {"accuracy_matched_within_0.005": abs(s["acc"]["mean"] - w["acc"]["mean"]) <= 0.005,
                 "within_auc_W_gt_S": w["auc_within_person"]["mean"] > s["auc_within_person"]["mean"],
                 "S_within_auc_within_.02_of_.5": abs(s["auc_within_person"]["mean"] - 0.5) <= 0.02,
                 "eta2_S_gt_W": s["eta2_score"]["mean"] > w["eta2_score"]["mean"],
                 "rho_b_S_gt_W": s["rho_b"]["mean"] > w["rho_b"]["mean"]}
    print(json.dumps({k: res[k] for k in ("sigma_W_matched", "target_accuracy_S", "H5")}), flush=True)
    print("S", json.dumps({k: round(v["mean"], 3) for k, v in s.items()}), flush=True)
    print("W", json.dumps({k: round(v["mean"], 3) for k, v in w.items()}), flush=True)
    save("r1_4B_semisynthetic_audit2.json", res)


if __name__ == "__main__":
    t0 = time.time()
    {"A": run_A, "B": run_B}[sys.argv[1]]()
    print("minutes %.1f" % ((time.time() - t0) / 60), flush=True)
