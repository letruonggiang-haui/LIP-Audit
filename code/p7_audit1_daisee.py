# -*- coding: utf-8 -*-
"""P7 - Audit 1 on DAiSEE from the released label files alone, under the pairing
rule written in Supplementary S3.1, plus the variants needed to find which rule
produced the pair counts of Table 2 (3,225 / 748 / 1,045).

Rule S3.1: characters 1-6 = subject, 1-7 = session, the rest = position (any
identifier width); a pair is two labelled clips of one session whose positions
differ by exactly one; the decisive axis keeps pairs with both labels >= 2.

Variants:
  s31_allwidths   the S3.1 rule as written
  ten_char_only   S3.1 restricted to ten-character identifiers (p6_controls.py)
  gap_le_2        positions differing by 1 or 2 (the smoothing script p2_session.py)
  sorted_neighbours   consecutive labelled clips in sorted order, any gap

For each: pairs, agreement a, a_iid (split marginals over the decisive-axis
clips), Cohen's kappa, recording-clustered percentile bootstrap (2,000
replicates, chance term recomputed inside each replicate), within-recording and
within-subject permutation nulls (10,000 draws).

  python code/p7_audit1_daisee.py [--labels <dir with TrainLabels.csv, ValidationLabels.csv,
                                   TestLabels.csv>] [--out <json path>]
  defaults: --labels data_local/daisee_labels  --out results_p4/json/p7_audit1_daisee.json
"""
import argparse
import io
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(ROOT, "data_local", "daisee_labels")
OUT = os.path.join(ROOT, "results_p4", "json", "p7_audit1_daisee.json")
B_BOOT, B_PERM = 2000, 10000
RNG = np.random.default_rng(20260925)


def load(split, lab=LAB):
    df = pd.read_csv(os.path.join(lab, "%sLabels.csv" % split))
    df.columns = [c.strip() for c in df.columns]
    df["stem"] = df["ClipID"].astype(str).str.strip().str.replace(r"\.(avi|mp4)$", "", regex=True)
    df = df[df["stem"].str.fullmatch(r"\d+")]
    df["subj"] = df["stem"].str[:6]
    df["sess"] = df["stem"].str[:7]
    df["pos"] = df["stem"].str[7:].astype(int)
    df["y"] = df["Engagement"].astype(int)
    return df.reset_index(drop=True)


def pairs_for(df, rule):
    out = []  # (sess, subj, y1, y2)
    for s, g in df.groupby("sess"):
        if rule == "ten_char_only":
            g = g[g["stem"].str.len() == 10]
        g = g.sort_values("pos")
        pos = g["pos"].to_numpy()
        y = g["y"].to_numpy()
        for i in range(len(g) - 1):
            d = pos[i + 1] - pos[i]
            ok = (d == 1) if rule in ("s31_allwidths", "ten_char_only") else (d <= 2) if rule == "gap_le_2" else True
            if ok:
                out.append((s, s[:6], y[i], y[i + 1]))
    return out


def decisive(pairs):
    return [p for p in pairs if p[2] >= 2 and p[3] >= 2]


def chance_split(y_dec):
    """a_iid of Eq. (2): squared marginals of the decisive-axis label over ALL decisive-axis
    clips of the split (the main text's convention), not over the paired clips only."""
    pk = np.array([(y_dec == k).mean() for k in np.unique(y_dec)])
    return float((pk ** 2).sum())


def stats(pairs, y_dec=None):
    a1 = np.array([p[2] for p in pairs]); a2 = np.array([p[3] for p in pairs])
    agree = float((a1 == a2).mean())
    if y_dec is None:  # pair-marginal form (Supplementary S3.3)
        allv = np.concatenate([a1, a2])
        pk = np.array([(allv == k).mean() for k in np.unique(allv)])
        chance = float((pk ** 2).sum())
    else:
        chance = chance_split(y_dec)
    return agree, chance, (agree - chance) / (1 - chance)


def boot(pairs, df_dec):
    """Recording-clustered percentile bootstrap: resample sessions with replacement; inside each
    replicate rebuild the pair set AND recompute the chance term from the resampled clips."""
    sess = np.array([p[0] for p in pairs])
    keys = np.unique(np.concatenate([sess, df_dec["sess"].unique()]))
    kidx = {k: i for i, k in enumerate(keys)}
    by_pairs = [[] for _ in keys]
    for i, s in enumerate(sess):
        by_pairs[kidx[s]].append(i)
    by_clips = [[] for _ in keys]
    ydec = df_dec["y"].to_numpy()
    for i, s in enumerate(df_dec["sess"].to_numpy()):
        by_clips[kidx[s]].append(i)
    ks, as_ = [], []
    for _ in range(B_BOOT):
        draw = RNG.integers(0, len(keys), len(keys))
        pidx = np.concatenate([np.array(by_pairs[d], int) for d in draw])
        cidx = np.concatenate([np.array(by_clips[d], int) for d in draw])
        if len(pidx) < 10:
            continue
        a_, _, k_ = stats([pairs[i] for i in pidx], ydec[cidx])
        ks.append(k_); as_.append(a_)
    return (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5)),
            float(np.percentile(as_, 2.5)), float(np.percentile(as_, 97.5)))


def distant_agreement(df_dec):
    """Agreement between non-adjacent decisive-axis clips of the same subject: every same-subject
    pair except those within two positions of one another in one session (Table 2, last column)."""
    agree = 0; total = 0
    for s, g in df_dec.groupby("subj"):
        y = g["y"].to_numpy(); sess = g["sess"].to_numpy(); pos = g["pos"].to_numpy()
        n = len(g)
        for i in range(n):
            same = (sess[i + 1:] == sess[i]) & (np.abs(pos[i + 1:] - pos[i]) <= 2)
            keep = ~same
            total += keep.sum()
            agree += (y[i + 1:][keep] == y[i]).sum()
    return int(total), float(agree / total) if total else float("nan")


def perm_null(df_dec, pairs_idx, group_col):
    """Shuffle decisive-axis labels within group (session or subject), rebuild pairs by index."""
    y = df_dec["y"].to_numpy().copy()
    groups = df_dec[group_col].to_numpy()
    i1 = np.array([i for i, _ in pairs_idx]); i2 = np.array([j for _, j in pairs_idx])
    obs = float((y[i1] == y[i2]).mean())
    hits = 0
    vals = []
    for _ in range(B_PERM):
        yp = y.copy()
        for g in np.unique(groups):
            m = np.where(groups == g)[0]
            yp[m] = RNG.permutation(yp[m])
        a = float((yp[i1] == yp[i2]).mean())
        vals.append(a)
        hits += a >= obs
    return obs, float(np.mean(vals)), (hits + 1) / (B_PERM + 1)


def pairs_idx_for(df_dec):
    """Strict S3.1 pairs as row indices into the decisive-axis frame (for the permutation nulls,
    positions are those of the full label file, so a pair must be index-adjacent among ALL clips)."""
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=LAB,
                    help="directory holding TrainLabels.csv, ValidationLabels.csv, TestLabels.csv (default: %(default)s)")
    ap.add_argument("--out", default=OUT, help="output JSON path (default: %(default)s)")
    args = ap.parse_args()
    lab, out = args.labels, args.out
    res = {}
    for split in ("Train", "Validation", "Test"):
        df = load(split, lab)
        r = {"clips": int(len(df)), "subjects": int(df["subj"].nunique()), "sessions": int(df["sess"].nunique()),
             "id_widths": {int(k): int(v) for k, v in df["stem"].str.len().value_counts().sort_index().items()}}
        dfd = df[df["y"] >= 2].reset_index(drop=True)
        for rule in ("s31_allwidths", "ten_char_only", "gap_le_2", "sorted_neighbours"):
            allp = pairs_for(df, rule)
            dec = decisive(allp)
            a, c, k = stats(dec, dfd["y"].to_numpy())
            _, cp, kp = stats(dec)
            lo, hi, alo, ahi = boot(dec, dfd)
            r[rule] = dict(pairs_all=len(allp), pairs_decisive=len(dec), a=round(a, 4),
                           a_ci=[round(alo, 4), round(ahi, 4)],
                           a_iid_split_marginals=round(c, 4), kappa_split_marginals=round(k, 4),
                           kappa_ci=[round(lo, 4), round(hi, 4)],
                           a_iid_pair_marginals=round(cp, 4), kappa_pair_marginals=round(kp, 4))
            print("%-10s %-17s pairs %5d (all %5d)  a %.3f [%.3f, %.3f]  a_iid %.3f  kappa %.3f [%.3f, %.3f]  (pair-marginal kappa %.3f)" % (
                split, rule, len(dec), len(allp), a, alo, ahi, c, k, lo, hi, kp))
        nd, ad = distant_agreement(dfd)
        r["distant_pairs"] = nd; r["distant_agreement"] = round(ad, 4)
        print("           distant same-subject pairs %d  agreement %.3f" % (nd, ad))
        # permutation nulls for the S3.1 rule on the decisive axis
        key = {(s, p): i for i, (s, p) in enumerate(zip(dfd["sess"], dfd["pos"]))}
        # strict adjacency in the FULL file: rebuild pairs from df then map to decisive rows
        pidx = []
        for s, g in df.groupby("sess"):
            g = g.sort_values("pos"); pos = g["pos"].to_numpy(); yy = g["y"].to_numpy()
            for i in range(len(g) - 1):
                if pos[i + 1] - pos[i] == 1 and yy[i] >= 2 and yy[i + 1] >= 2:
                    pidx.append((key[(s, pos[i])], key[(s, pos[i + 1])]))
        for col, name in (("sess", "within_recording"), ("subj", "within_subject")):
            obs, mean_null, p = perm_null(dfd, pidx, col)
            r["s31_allwidths"]["null_" + name] = dict(observed=round(obs, 4), mean=round(mean_null, 4), p=round(p, 4))
            print("           null %-16s observed %.3f  mean %.3f  p %.3f" % (name, obs, mean_null, p))
        res[split] = r
    io.open(out, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
