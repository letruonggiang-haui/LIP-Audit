# -*- coding: utf-8 -*-
"""P7 - two controls the internal review round of 25/09/2026 asked for on the DAiSEE
identity readings (items R1-M2 and R1-M3), computed from the cached frozen
embeddings (~/p2/probe/*_z.npz, 512-d pooled backbone output of every labelled
Train+Validation clip) and the saved identity-head weights. Nothing is
retrained and the test split is not touched.

A. Recording-disjoint probe on DAiSEE (as Table S16 does for EngageNet).
   A DAiSEE session is the seven-character prefix of the clip identifier. For
   every subject with at least two sessions one whole session is held out; the
   probe (standardized, l2-regularized multinomial logistic regression, C = 1)
   is fitted on the remaining sessions and read on the held-out one. Compared
   with the published random 75/25 within-subject split on the same subject
   pool, under no centering, oracle centering (subject mean over all clips) and
   fitted centering (subject mean over probe-training clips only). Pools: the
   88 subjects of Train+Validation and the 69 of Train.

B. The adversary's own head on the probe's pool and held-out clips.
   Each adversarial run's subj_head (linear, or 512-256-69 MLP with GELU) is
   applied to the same frozen z it was trained against, over the 69 training
   subjects it has outputs for, (i) on all training-split clips (as reported in
   v6, "the clips it trained on") and (ii) on the probe's held-out 25%. The
   retrained probe is read on the same held-out clips, so the two columns of
   Table 6 are for the first time on one pool and one clip set.

  python code/p7_daisee_probe_controls.py -> results_p4/json/p7_daisee_probe_controls.json
"""
import glob
import io
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.special import erf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data_local", "daisee_probe")
LAB = os.path.join(ROOT, "data_local", "daisee_labels")
OUT = os.path.join(ROOT, "results_p4", "json", "p7_daisee_probe_controls.json")
SEEDS = (0, 1, 2)
MODES = ("none", "fitted")  # oracle centering is the published control and is not re-run here

TABLE6 = {
    "no_penalty_baseline": ["base_s0_last"],
    "adversary": ["grl025_s0_last", "grl05_s0_last", "grl10_s0_last", "grl20_s0_last",
                  "mlp025_s0_last", "mlp025_s1_last", "mlp025_s2_last", "mlp05_s0_last", "mlp10_s0_last"],
    "separation_penalty": ["idsep4_s0_last"],
    "prototype_confusion_global": ["gproto2_s0_best_acc_late"],
    "prototype_confusion_local": ["proto2_s0_best_acc_late", "proto2_s1_best_acc_late", "proto2_s2_best_acc_late"],
}


def stems_in_pool():
    """Clip stems in the order p2_probe.extract wrote the cache: labelled Train clips in
    Train_ids order, then labelled Validation clips in Validation_ids order."""
    out, split_of = [], []
    for split in ("Train", "Validation"):
        ids = [str(s) for s in np.load(os.path.join(D, "%s_ids.npy" % split), allow_pickle=True)]
        lab = pd.read_csv(os.path.join(LAB, "%sLabels.csv" % split))
        lab.columns = [c.strip() for c in lab.columns]
        keep = set(lab["ClipID"].astype(str).str.replace(r"\.(avi|mp4)$", "", regex=True))
        for s in ids:
            if s in keep:
                out.append(s); split_of.append(split)
    return np.array(out), np.array(split_of)


def random_split(subj, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s)
        if len(idx) < 4:
            continue
        idx = rng.permutation(idx)
        k = max(1, int(round(len(idx) * 0.25)))
        te.extend(idx[:k]); tr.extend(idx[k:])
    return np.array(tr), np.array(te)


def recording_split(subj, sess, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s)
        recs = np.unique(sess[idx])
        if len(recs) < 2:
            continue
        held = recs[rng.integers(len(recs))]
        te.extend(idx[sess[idx] == held]); tr.extend(idx[sess[idx] != held])
    return np.array(tr), np.array(te)


def centre(Z, subj, tr, mode):
    if mode == "none":
        return Z
    Zc = Z.copy()
    in_tr = np.zeros(len(Z), bool); in_tr[tr] = True
    for s in np.unique(subj):
        k = np.flatnonzero(subj == s)
        src = k if mode == "oracle" else k[in_tr[k]]
        if len(src) == 0:
            continue
        Zc[k] = Z[k] - Z[src].mean(0, keepdims=True)
    return Zc


def probe(Z, subj, tr, te):
    sc = StandardScaler().fit(Z[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(Z[tr]), subj[tr])
    acc = float((clf.predict(sc.transform(Z[te])) == subj[te]).mean())
    n = len(np.unique(subj[tr]))
    return acc, acc * n, n


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def head_predict(h, Z):
    if "weight" in h:  # linear
        logits = Z @ h["weight"].T + h["bias"]
    else:  # Sequential(Linear, GELU, Linear): keys 0.weight, 0.bias, 2.weight, 2.bias
        logits = gelu(Z @ h["0.weight"].T + h["0.bias"]) @ h["2.weight"].T + h["2.bias"]
    return logits.argmax(1)


def main():
    stems, split_of = stems_in_pool()
    subj_all = np.array([s[:6] for s in stems]); sess_all = np.array([s[:7] for s in stems])
    train_subjects = sorted(set(subj_all[split_of == "Train"]))  # Clips.subj_to_idx order
    idx_of = {s: k for k, s in enumerate(train_subjects)}
    res = {"pool88": {"subjects": len(set(subj_all)), "clips": int(len(stems)),
                      "subjects_multi_session": int(sum(len(set(sess_all[subj_all == s])) > 1 for s in set(subj_all))),
                      "sessions": int(len(set(sess_all)))},
           "pool69": {"subjects": len(train_subjects), "clips": int((split_of == "Train").sum()),
                      "subjects_multi_session": int(sum(len(set(sess_all[(subj_all == s) & (split_of == "Train")])) > 1 for s in train_subjects))},
           "A_recording_disjoint": {}, "B_adversary_head": {}}
    print("pool88:", res["pool88"]); print("pool69:", res["pool69"])

    for group, runs in TABLE6.items():
        for run in runs:
            f = os.path.join(D, run + "_z.npz")
            if not os.path.exists(f):
                print("missing", run); continue
            z = np.load(f, allow_pickle=True)
            Z = z["z"].astype(np.float64); subj = np.array([str(s) for s in z["subj"]])
            assert len(Z) == len(stems) and (subj == subj_all).all(), run
            seeds = SEEDS if group == "no_penalty_baseline" else SEEDS[:1]
            pools = (("pool88", np.ones(len(Z), bool)), ("pool69", split_of == "Train")) if group == "no_penalty_baseline" else (("pool69", split_of == "Train"),)
            for pool, mask in pools:
                Zp, sp, ssp = Z[mask], subj[mask], sess_all[mask]
                rows = {}
                for split_name, splitter in (("random", random_split), ("recording", recording_split)):
                    for mode in MODES:
                        vals = []
                        for seed in seeds:
                            tr, te = splitter(sp, seed) if split_name == "random" else splitter(sp, ssp, seed)
                            keep = np.isin(sp, np.unique(sp[tr]))  # restrict to subjects present in this split
                            # for the recording split only multi-session subjects enter; same pool for random by construction
                            Zc = centre(Zp, sp, tr, mode)
                            acc, M, n = probe(Zc, sp, tr, te)
                            vals.append((acc, M, n))
                        rows["%s/%s" % (split_name, mode)] = dict(
                            top1=[round(v[0], 4) for v in vals], M=[round(v[1], 1) for v in vals], n_classes=vals[0][2])
                # random split restricted to the multi-session subjects (same pool as the recording split)
                multi = np.array([len(set(ssp[sp == s])) > 1 for s in sp])
                vals = []
                for seed in seeds:
                    tr, te = random_split(sp[multi], seed)
                    acc, M, n = probe(Zp[multi], sp[multi], tr, te)
                    vals.append((acc, M, n))
                rows["random_multi_session_pool/none"] = dict(top1=[round(v[0], 4) for v in vals], M=[round(v[1], 1) for v in vals], n_classes=vals[0][2])
                res["A_recording_disjoint"].setdefault(group, {}).setdefault(run, {})[pool] = rows
                r1, r2, r3 = rows["random/none"]["M"], rows["recording/none"]["M"], rows["random_multi_session_pool/none"]["M"]
                print("%-28s %-6s random %s | recording-disjoint %s (n=%d) | random on same pool %s | fitted-centred rec %s" % (
                    run, pool, r1, r2, rows["recording/none"]["n_classes"], r3, rows["recording/fitted"]["M"]))
            # B: adversary head on the 69-pool
            if group == "adversary":
                hf = os.path.join(D, "heads", run.replace("_last", "_last") + "_subjhead.npz")
                if not os.path.exists(hf):
                    print("no head for", run); continue
                h = dict(np.load(hf))
                m69 = split_of == "Train"
                Z69, s69 = Z[m69], subj[m69]
                y69 = np.array([idx_of[s] for s in s69])
                pred_all = head_predict(h, Z69)
                acc_all = float((pred_all == y69).mean())
                per_seed = []
                for seed in SEEDS:
                    tr, te = random_split(s69, seed)
                    acc_te = float((head_predict(h, Z69[te]) == y69[te]).mean())
                    acc_probe, M_probe, n = probe(Z69, s69, tr, te)
                    per_seed.append(dict(seed=seed, head_heldout_top1=round(acc_te, 4), head_heldout_M=round(acc_te * 69, 2),
                                         probe_heldout_top1=round(acc_probe, 4), probe_heldout_M=round(M_probe, 1)))
                res["B_adversary_head"][run] = dict(head_all_train_clips_top1=round(acc_all, 4), head_all_train_clips_M=round(acc_all * 69, 2),
                                                    n_outputs=int(h["weight"].shape[0] if "weight" in h else h["2.weight"].shape[0]),
                                                    per_seed=per_seed)
                print("   head %-14s all-train-clips M %.2f | held-out M %s | retrained probe held-out M %s" % (
                    run, acc_all * 69, [p["head_heldout_M"] for p in per_seed], [p["probe_heldout_M"] for p in per_seed]))
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
