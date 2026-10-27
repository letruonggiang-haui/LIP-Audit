# -*- coding: utf-8 -*-
"""Turn the R1 result files into the LaTeX table bodies of Supplementary Section S11 and a flat
numbers file used to check every figure quoted in the main text and the response letter.

Reads paper/review_R1_20261024/results/*.json (and server/*.json when fetched); writes
results/r1_tables.tex and results/r1_numbers.json. Missing inputs are skipped and listed.
"""
import glob
import json
import os

import numpy as np

ROOT = r"E:\OneDrive\R&D\Education\P2-Identity-Invariant-Engagement"
RES = os.path.join(ROOT, "paper", "review_R1_20261024", "results")
REPO = os.path.join(ROOT, "release_lip_audit")
N = {}
T = []
missing = []


def load(path):
    if not os.path.exists(path):
        missing.append(os.path.basename(path))
        return None
    return json.load(open(path, encoding="utf-8"))


def f3(x):
    if x == x and abs(x) < 0.0005:
        return ".000"
    return ("%.3f" % x).replace("-0.", "$-$0.").replace("0.", ".", 1) if x == x else "--"


def k3(x):
    s = "%.3f" % x
    return s.replace("-", "$-$")


def ci(lo, hi, f=None):
    f = f or f3
    return "[%s, %s]" % (f(lo), f(hi))


def sgn(x):
    return ("$+$" if x >= 0 or round(x, 3) == 0 else "$-$") + f3(abs(x))


def pts(x):
    return "%.1f" % (100 * x)


# ---------------------------------------------------------------- S11.2 Audit 1 intervals
a1 = load(os.path.join(REPO, "results", "p7_audit1_daisee.json"))
k = load(os.path.join(RES, "r1_1a_kappa_participant_clustered.json"))
if a1 and k:
    rows = []
    for s in ("Train", "Validation", "Test"):
        r = a1[s]["s31_allwidths"]
        q = k["DAiSEE"][s]
        rows.append("DAiSEE %s, decisive & %d & %d & %s & %s & %s\\\\" % (
            s.lower(), q["n_subjects"], r["pairs_decisive"], k3(r["kappa_split_marginals"]),
            ci(*r["kappa_ci"], f=k3), ci(*q["kappa_ci_subject_clustered"], f=k3)))
        N["daisee_%s_kappa_subjci" % s.lower()] = q["kappa_ci_subject_clustered"]
        N["daisee_%s_kappa_recci" % s.lower()] = r["kappa_ci"]
    e1 = load(os.path.join(REPO, "results", "E1_results.json"))
    for s, v in k["EngageNet_stored_E2"].items():
        for ax, name in (("four_class", "four-level"), ("decisive", "decisive")):
            kap = e1["EngageNet"][s][ax if ax == "decisive" else "four_class"]["adjacent"]["kappa"] if e1 else float("nan")
            rc, sc = v[ax]["recording_clustered"], v[ax]["subject_clustered"]
            rows.append("EngageNet %s, %s & %d & %d & %s & %s & %s\\\\" % (
                s, name, sc["n_clusters"], e1["EngageNet"][s][ax]["adjacent"]["pairs"] if e1 else 0, k3(kap),
                ci(rc["kappa_lo"], rc["kappa_hi"], f=k3), ci(sc["kappa_lo"], sc["kappa_hi"], f=k3)))
            N["engagenet_%s_%s_kappa_subjci" % (s, ax)] = [sc["kappa_lo"], sc["kappa_hi"]]
    T.append(("tab:r1-kappa", "\n".join(rows)))

# ---------------------------------------------------------------- S11.3 coding
c = load(os.path.join(RES, "r1_1b_coding_sensitivity.json"))
if c:
    rows = []
    for key, v in c["splits"].items():
        m = v["clmm"]
        rows.append("%s & %d & %s & %s & %s & %s & %s\\\\" % (
            key, v["n_participants"], f3(v["eta2_equal_interval"]), f3(v["eta2_decisive"]),
            f3(v["eta2_ridit"]), f3(m["icc_latent"]), f3(m.get("unit_share_latent", float("nan")))))
        N["coding_" + key.replace(" ", "_").lower()] = {
            "equal": v["eta2_equal_interval"], "decisive": v["eta2_decisive"], "ridit": v["eta2_ridit"],
            "icc": m["icc_latent"], "rec": m.get("unit_share_latent")}
    T.append(("tab:r1-coding", "\n".join(rows)))
    chk = {}
    for r in c["estimator_check"]:
        chk.setdefault((r["structure"], r["true_icc"]), []).append(r["estimated_icc"])
    T.append(("tab:r1-coding-check", "\n".join(
        "%s & %.1f & %s\\\\" % (s, t, ", ".join(f3(x) for x in v)) for (s, t), v in chk.items())))

# ---------------------------------------------------------------- S11.4 equivalence
e = load(os.path.join(RES, "r1_2_equivalence_mdd.json"))
if e:
    names = {"validation_best_epoch": "Validation, best epoch", "validation_last_epoch": "Validation, last epoch",
             "test_frozen": "Test, frozen checkpoints", "test_trainval": "Test, Train+Val arm"}
    rows = []
    for b, nm in names.items():
        v = e[b]
        rows.append("%s & %d & %d & %s & %s & %s & %s\\\\" % (
            nm, v["n_participants"], len(v["pairs"]), pts(v["max_equivalence_bound_acc"]), pts(v["max_mdd80_acc"]),
            pts(v["max_equivalence_bound_uar"]), pts(v["max_mdd80_uar"])))
        N["equiv_" + b] = {k2: v[k2] for k2 in v if k2.startswith("max")}
    T.append(("tab:r1-equiv", "\n".join(rows)))

# ---------------------------------------------------------------- S11.5 residuals
r = load(os.path.join(RES, "r1_3_residuals.json"))
if r:
    rows = []
    groups = {}
    for corpus in ("DAiSEE_validation", "EngageNet_validation"):
        for fn, v in r[corpus].items():
            fam = fn.replace("_val_probs.npz", "")
            fam = fam.rsplit("_s", 1)[0]
            groups.setdefault((corpus, fam), []).append(v)
    labels = {"au": "AU stream", "w5": "Five-window", "of_cnnlstm": "CNN--LSTM", "of_lstm": "LSTM",
              "of_mlp": "MLP", "of_tcn": "TCN", "of_transformer": "Transformer"}
    for (corpus, fam), vs in groups.items():
        gap = [x["resid_gap"] for x in vs]
        lo = min(x["ci"]["resid_gap"][0] for x in vs)
        em = [x["eta2_score_label_matched"] for x in vs]
        emlo = min(x["ci"]["eta2_score_label_matched"][0] for x in vs)
        un = [x["eta2_score_unmatched"] for x in vs]
        rows.append("%s & %s & %d & %s & %s & %s & %s & %s\\\\" % (
            corpus.split("_")[0], labels.get(fam, fam), len(vs), f3(np.mean(gap)), f3(lo),
            f3(np.mean(un)), f3(np.mean(em)), f3(emlo)))
        N["resid_%s_%s" % (corpus.split("_")[0], fam)] = {"gap_mean": float(np.mean(gap)), "gap_min_lo": lo,
                                                          "eta2_matched_mean": float(np.mean(em)), "eta2_matched_min_lo": emlo,
                                                          "gap_range": [min(gap), max(gap)], "eta2m_range": [min(em), max(em)]}
    T.append(("tab:r1-resid", "\n".join(rows)))

# ---------------------------------------------------------------- S11.6 synthetic Audit 1
s = load(os.path.join(RES, "r1_4A_synthetic_audit1.json"))
if s:
    G = s["grid"]
    rows = []
    for corpus, ax in (("DAiSEE", "decisive"), ("EngageNet", "four_level")):
        for rr in G:
            cells = []
            for q in G:
                cc = [x for x in s["cells"] if x["corpus"] == corpus and x["variant"] == "marginal"
                      and x["r"] == rr and x["q"] == q][0][ax]
                cells.append("%s" % f3(cc["mean"]))
            rows.append("%s & %.2f & %s\\\\" % (corpus if rr == G[0] else "", rr, " & ".join(cells)))
        rows.append("\\midrule")
    T.append(("tab:r1-synth", "\n".join(rows[:-1])))
    N["synth_H3"] = s["H3_monotonicity"]
    N["synth_H4"] = s["H4_overlapping_cells"]

# ---------------------------------------------------------------- S11.7 semi-synthetic Audit 2
b = load(os.path.join(RES, "r1_4B_semisynthetic_audit2.json"))
if b:
    rows = []
    for nm, key in (("S (person offset only)", "S_person_shortcut"), ("W (within-person only)", "W_within_person")):
        v = b[key]
        rows.append("%s & %s & %s & %s & %s\\\\" % (nm, f3(v["acc"]["mean"]), f3(v["auc_within_person"]["mean"]),
                                                  f3(v["eta2_score"]["mean"]), f3(v["rho_b"]["mean"])))
    rows.append("\\midrule")
    for lam, v in b["mixture_lambda_weight_on_S"].items():
        rows.append("Mixture, weight %s on S & %s & %s & %s & %s\\\\" % (
            lam, f3(v["acc"]["mean"]), f3(v["auc_within_person"]["mean"]), f3(v["eta2_score"]["mean"]), f3(v["rho_b"]["mean"])))
    T.append(("tab:r1-semi", "\n".join(rows)))
    N["semi"] = {k2: {m: b[k2][m]["mean"] for m in b[k2]} for k2 in ("S_person_shortcut", "W_within_person")}
    N["semi"]["sigma_W"] = b["sigma_W_matched"]
    N["semi"]["H5"] = b["H5"]

# ---------------------------------------------------------------- S11.8-10 server results
for name, key in (("C_identity_overlap.json", "C"), ("D_identity_futile.json", "D"),
                  ("F1_temporal_shuffle_retrained.json", "F1"), ("F2_temporal_shuffle_frozen.json", "F2")):
    v = load(os.path.join(RES, "server", name))
    if v is not None:
        N[key] = v
if "C" in N:
    rows = []
    for fam, v in N["C"]["families"].items():
        for cond in ("overlap", "disjoint"):
            x = v[cond]
            rows.append("%s & %s & %s & %s & %s & %s & %s\\\\" % (
                fam.upper() if cond == "overlap" else "", cond, f3(x["acc"]), f3(x["uar"]), f3(x["auc_within_person"]),
                f3(x["eta2_score_ordinal"]), f3(x["rho_b"])))
        d = v["overlap_minus_disjoint"]
        rows.append(" & difference & %s %s & %s %s & %s %s & %s %s & %s %s\\\\" % tuple(
            z for m in ("acc", "uar", "auc_within_person", "eta2_score_ordinal", "rho_b")
            for z in (sgn(d[m]["point"]), ci(*d[m]["ci"]))))
        rows.append("\\midrule")
    T.append(("tab:r1-overlap", "\n".join(rows[:-1])))
if "D" in N:
    rows = []
    for fam, v in N["D"]["families"].items():
        for cond in ("balanced", "natural"):
            x = v["eval_on_balanced_heldout"][cond]
            p = v["identity_probe"][cond]
            rows.append("%s & %s & %s & %s & %s & %.1f (%.1f)\\\\" % (
                fam.upper() if cond == "balanced" else "", cond, f3(x["acc"]), f3(x["auc_within_person"]),
                f3(x["eta2_score"]), p["times_chance_mean"], p["times_chance_min"]))
        d = v["balanced_minus_natural_on_balanced_heldout"]
        rows.append(" & difference & %s & %s & %s & \\\\" % tuple(
            sgn(d[m]["point"]) + " " + ci(*d[m]["ci"])
            for m in ("acc", "auc_within_person", "eta2_score")))
        rows.append("\\midrule")
    T.append(("tab:r1-futile", "\n".join(rows[:-1])))
if "F1" in N:
    rows = []
    for fam, v in N["F1"]["families"].items():
        p = v["point"]
        if fam == "mlp":
            rows.append("MLP (no time axis) & %s & -- & -- & -- & %s & -- & --\\\\" % (f3(p["ordered"]["acc"]), f3(p["ordered"]["auc_within_person"])))
            continue
        d = v["shuffled_minus_ordered"]
        # UAR difference added 27/10/2026: a registered primary outcome (registration 4F)
        rows.append("%s & %s & %s & %s %s & %s %s & %s & %s & %s %s\\\\" % (
            fam.upper() if fam in ("lstm", "tcn") else ("CNN--LSTM" if fam == "cnnlstm" else "Transformer"),
            f3(p["ordered"]["acc"]), f3(p["shuffled"]["acc"]),
            sgn(d["acc"]["point"]), ci(*d["acc"]["ci"]),
            sgn(d["uar"]["point"]), ci(*d["uar"]["ci"]),
            f3(p["ordered"]["auc_within_person"]), f3(p["shuffled"]["auc_within_person"]),
            sgn(d["auc_within_person"]["point"]),
            ci(*d["auc_within_person"]["ci"])))
    T.append(("tab:r1-temporal", "\n".join(rows)))

with open(os.path.join(RES, "r1_tables.tex"), "w", encoding="utf-8") as fh:
    for lab, body in T:
        fh.write("%% ---- %s\n%s\n\n" % (lab, body))
with open(os.path.join(RES, "r1_numbers.json"), "w", encoding="utf-8") as fh:
    json.dump(N, fh, indent=1, default=float)
print("tables:", [t[0] for t in T], "| missing:", missing)
