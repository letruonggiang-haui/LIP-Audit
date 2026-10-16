"""Shared label-audit core for the two-benchmark article (E1, E2).

Loads DAiSEE and EngageNet label files, builds pair indices once, and computes
agreement / kappa / permutation nulls / clustered bootstraps on top of them.

Conventions kept identical to the DAiSEE analysis already in the manuscript:
  DAiSEE  recording = ClipID[:7], subject = ClipID[:6], order = int(ClipID[7:]),
          strict adjacency (order difference of exactly one),
          decisive axis = Engagement 3 versus 2 among clips with Engagement >= 2.
  EngageNet recording = (subject, session code, video index), order = chunk index,
          SNP removed before pairing so a pair spanning a removed chunk is kept,
          four ordinal levels; decisive axis = Highly-Engaged versus Engaged
          among clips at Engaged or above.
"""
import os
import re
import numpy as np
import pandas as pd

# set DAISEE_ROOT / ENGAGENET_ROOT in the environment to point at the released corpora
DAISEE_ROOT = os.environ.get("DAISEE_ROOT", r"E:\datasets\DAiSEE\DAiSEE")
ENGAGENET_ROOT = os.environ.get("ENGAGENET_ROOT", r"E:\datasets\EmotiW_2023_EngageNet")

EN_LEVELS = {"Not-Engaged": 0, "Barely-engaged": 1, "Engaged": 2, "Highly-Engaged": 3}
EN_SNP = "SNP(Subject Not Present)"


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_daisee(split, labels_dir=None):
    """Return a frame with stem, subj, rec, t, y (Engagement 0-3).

    The delimiter is detected per file: the second copy of the label directory
    that ships with the download uses semicolons for Train and Validation and
    commas for Test, and reading all three with commas corrupts the semicolon
    files without raising. The expected columns are asserted after the split.
    """
    path = os.path.join(labels_dir or os.path.join(DAISEE_ROOT, "Labels"),
                        split + "Labels.csv")
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    need = ["ClipID", "Boredom", "Engagement", "Confusion", "Frustration"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(path + ": delimiter " + repr(sep) +
                         " left columns " + repr(df.columns.tolist()))
    n_raw = len(df)
    df["stem"] = (df["ClipID"].astype(str).str.strip()
                  .str.replace(r"\.(avi|mp4)$", "", regex=True))
    df["y"] = pd.to_numeric(df["Engagement"], errors="coerce")
    df = df[df["y"].notna()].copy()
    df["y"] = df["y"].astype(int)
    df["subj"] = df["stem"].str[:6]
    df["rec"] = df["stem"].str[:7]
    df["t"] = pd.to_numeric(df["stem"].str[7:], errors="coerce")
    df["idlen"] = df["stem"].str.len()
    df["wellformed"] = df["stem"].str.fullmatch(r"\d{10}")
    df = df[df["t"].notna()].copy()
    df["t"] = df["t"].astype(int)
    df["split"] = split
    out = df[["stem", "subj", "rec", "t", "y", "idlen", "wellformed", "split"]].copy()
    out.attrs["n_raw"] = n_raw
    out.attrs["sep"] = sep
    out.attrs["path"] = path
    return out


def daisee_disk_stems(split, root=None):
    """Clip stems present on disk; the face-extraction stage enumerated these."""
    out = set()
    base = os.path.join(root or DAISEE_ROOT, "DataSet", split)
    for dirpath, _, files in os.walk(base):
        for fn in files:
            if fn.lower().endswith((".avi", ".mp4")):
                out.add(os.path.splitext(fn)[0])
    return out


_EN_PAT = re.compile(r"^subject_(\d+)_([A-Za-z0-9]+)_vid_(\d+)_(\d+)\.mp4$")


def load_engagenet(split, root=None):
    """Return a frame with chunk, subj, sess, vid, rec, t, y, label.

    y is NaN for SNP. The recording key is (subject, session code, video index):
    five participants carry two session codes and the chunk index restarts at
    each session, so grouping by (subject, video index) alone merges two
    recordings and lowers every agreement figure.
    """
    base = root or ENGAGENET_ROOT
    if split == "test":
        df = pd.read_csv(os.path.join(base, "test_engagement_labels.csv"))
    else:
        df = pd.read_excel(os.path.join(base, split + "_engagement_labels.xlsx"))
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df[["chunk", "label"]].copy()
    df["chunk"] = df["chunk"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    m = df["chunk"].apply(_EN_PAT.match)
    if m.isna().any():
        raise ValueError("unparsed chunk names: " +
                         repr(df.loc[m.isna(), "chunk"].head().tolist()))
    df["subj"] = m.apply(lambda x: int(x.group(1)))
    df["sess"] = m.apply(lambda x: x.group(2))
    df["vid"] = m.apply(lambda x: int(x.group(3)))
    df["t"] = m.apply(lambda x: int(x.group(4)))
    df["rec"] = [str(a) + "|" + str(b) + "|" + str(c)
                 for a, b, c in zip(df["subj"], df["sess"], df["vid"])]
    df["y"] = df["label"].map(EN_LEVELS)
    df["split"] = split
    return df


# --------------------------------------------------------------------------- #
# pair index
# --------------------------------------------------------------------------- #
def build_pairs(df, rec_col="rec", strict=True, distance=1):
    """Index pairs of clips separated by `distance` in recording time.

    strict=True pairs a clip at order t with the clip at order t + distance of
    the same recording whenever both exist, so a distance-two pair does not
    require the clip between them to be present (DAiSEE convention).
    strict=False pairs rows that are `distance` apart in the sorted list of
    surviving rows regardless of gaps in the order index (EngageNet convention,
    which keeps a pair spanning a removed SNP chunk).
    Returns the reset frame and integer positions into it.
    """
    d = df.reset_index(drop=True)
    a_idx, b_idx = [], []
    for _, g in d.groupby(rec_col, sort=False):
        g = g.sort_values("t")
        pos = g.index.values
        tv = g["t"].values
        if strict:
            first = {}
            for k, t in enumerate(tv):
                first.setdefault(int(t), pos[k])
            for k, t in enumerate(tv):
                nxt = first.get(int(t) + distance)
                if nxt is not None and pos[k] == first.get(int(t)):
                    a_idx.append(pos[k])
                    b_idx.append(nxt)
        else:
            for i in range(len(pos) - distance):
                a_idx.append(pos[i])
                b_idx.append(pos[i + distance])
    return d, np.asarray(a_idx, dtype=int), np.asarray(b_idx, dtype=int)


def build_nonadjacent_pairs(df, subj_col="subj", rec_col="rec", near=2,
                            seed=20260904, cap=None):
    """Same-subject pairs that are not near-neighbours inside one recording.

    A pair is excluded when the two clips share a recording and their order
    indices differ by at most `near`, which is the exclusion the manuscript's
    ``Distant`` column uses. Every remaining pair is enumerated rather than
    sampled, so the figure is deterministic.
    """
    d = df.reset_index(drop=True)
    rec = d[rec_col].values
    t = d["t"].values
    a_idx, b_idx = [], []
    for _, g in d.groupby(subj_col, sort=False):
        pos = np.sort(g.index.values)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                p, q = pos[i], pos[j]
                if rec[p] == rec[q] and abs(int(t[p]) - int(t[q])) <= near:
                    continue
                a_idx.append(p)
                b_idx.append(q)
    a_idx = np.asarray(a_idx, dtype=int)
    b_idx = np.asarray(b_idx, dtype=int)
    if cap and len(a_idx) > cap:
        rng = np.random.default_rng(seed)
        k = np.sort(rng.choice(len(a_idx), cap, replace=False))
        a_idx, b_idx = a_idx[k], b_idx[k]
    return d, a_idx, b_idx


# --------------------------------------------------------------------------- #
# label spaces
# --------------------------------------------------------------------------- #
def space_full(d):
    """Four ordinal levels; selects rows with a defined level."""
    return d["y"].notna().values


def space_decisive(d):
    """Binary decisive axis: top level versus the one below, among y >= 2."""
    return (d["y"].notna() & (d["y"] >= 2)).values


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def agreement(y, a_idx, b_idx, keep):
    """Observed agreement, chance baseline and kappa on the selected rows.

    A pair is kept when both members are selected. The chance baseline is
    sum_k p_k^2 over the marginal of every selected clip in the split, not only
    those that sit in a pair, matching Eq. (agree) of the manuscript.
    """
    m = keep[a_idx] & keep[b_idx]
    ya, yb = y[a_idx[m]], y[b_idx[m]]
    obs = float(np.mean(ya == yb)) if len(ya) else float("nan")
    _, cnt = np.unique(y[keep], return_counts=True)
    p = cnt / cnt.sum()
    aiid = float((p ** 2).sum())
    kappa = (obs - aiid) / (1 - aiid) if aiid < 1 else float("nan")
    return dict(pairs=int(m.sum()), n_clips=int(keep.sum()), a=obs, a_iid=aiid,
                kappa=float(kappa), floor=float(p.max()))


def permutation_null(y, a_idx, b_idx, keep, groups, n_perm=1000, seed=20260904):
    """Shuffle labels inside each group, keeping the pair index fixed.

    Only the selected rows are shuffled, so the statistic runs on the same set
    of pairs in every replicate and the null isolates temporal order from the
    group's label composition. The reported p is (exceedances + 1) / (B + 1),
    so its floor is 1 / (B + 1) and it is never zero.
    """
    rng = np.random.default_rng(seed)
    m = keep[a_idx] & keep[b_idx]
    ai, bi = a_idx[m], b_idx[m]
    if len(ai) == 0:
        return dict(pairs=0)
    obs = float(np.mean(y[ai] == y[bi]))
    sel = np.flatnonzero(keep)
    gs = np.asarray(groups)[sel]
    order = np.argsort(gs, kind="stable")
    sel = sel[order]
    gs = gs[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    blocks = [sel[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    blocks = [b for b in blocks if len(b) > 1]
    yw = y.copy()
    draws = np.empty(n_perm)
    for r in range(n_perm):
        for blk in blocks:
            yw[blk] = rng.permutation(y[blk])
        draws[r] = np.mean(yw[ai] == yw[bi])
    p = (int((draws >= obs).sum()) + 1) / (n_perm + 1)
    return dict(pairs=int(len(ai)), a=obs, null_mean=float(draws.mean()),
                null_sd=float(draws.std()), null_max=float(draws.max()),
                p=float(p), n_perm=n_perm, n_groups=len(blocks))


def cluster_bootstrap(y, a_idx, b_idx, keep, clusters, n_boot=2000, seed=20260904):
    """Percentile intervals for agreement and kappa, resampling clusters.

    A cluster is a recording or a participant; every clip of a drawn cluster
    enters the replicate, so within-cluster dependence is preserved. The chance
    baseline is recomputed inside each replicate, so the kappa interval carries
    the uncertainty of both terms.
    """
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    m = keep[a_idx] & keep[b_idx]
    ai, bi = a_idx[m], b_idx[m]
    cl_of_pair = clusters[ai]
    uc = np.unique(clusters[keep])
    pair_by_cl, clip_by_cl = {}, {}
    for c in uc:
        pair_by_cl[c] = np.flatnonzero(cl_of_pair == c)
        clip_by_cl[c] = np.flatnonzero(keep & (clusters == c))
    A, K = [], []
    for _ in range(n_boot):
        draw = rng.choice(uc, len(uc), replace=True)
        pi = np.concatenate([pair_by_cl[c] for c in draw])
        ci = np.concatenate([clip_by_cl[c] for c in draw])
        if len(pi) == 0 or len(ci) == 0:
            continue
        a = np.mean(y[ai[pi]] == y[bi[pi]])
        _, cnt = np.unique(y[ci], return_counts=True)
        p = cnt / cnt.sum()
        aiid = (p ** 2).sum()
        A.append(a)
        K.append((a - aiid) / (1 - aiid) if aiid < 1 else np.nan)
    A, K = np.asarray(A), np.asarray(K)
    return dict(n_boot=int(len(A)), n_clusters=int(len(uc)),
                a_lo=float(np.percentile(A, 2.5)), a_hi=float(np.percentile(A, 97.5)),
                kappa_lo=float(np.nanpercentile(K, 2.5)),
                kappa_hi=float(np.nanpercentile(K, 97.5)))


def eta2_between(y, keep, groups):
    """Share of label variance lying between groups (Eq. eta_b)."""
    groups = np.asarray(groups)
    yy = y[keep].astype(float)
    gg = groups[keep]
    gm = yy.mean()
    sst = ((yy - gm) ** 2).sum()
    if sst <= 0:
        return float("nan")
    ssb = 0.0
    for g in np.unique(gg):
        v = yy[gg == g]
        ssb += len(v) * (v.mean() - gm) ** 2
    return float(ssb / sst)


def eta2_bootstrap(y, keep, groups, n_boot=2000, seed=20260904):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    gg = groups[keep]
    yy = y[keep].astype(float)
    uc = np.unique(gg)
    by = {g: yy[gg == g] for g in uc}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(uc, len(uc), replace=True)
        vals = [by[g] for g in draw]
        cat = np.concatenate(vals)
        gm = cat.mean()
        sst = ((cat - gm) ** 2).sum()
        if sst <= 0:
            continue
        ssb = sum(len(v) * (v.mean() - gm) ** 2 for v in vals)
        out.append(ssb / sst)
    out = np.asarray(out)
    return dict(lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)),
                n_boot=int(len(out)))


def subject_majority_oracle(y, keep, subjects):
    """Accuracy of assigning every clip of a participant that participant's
    modal label. It reads the held-out labels, so it is a ceiling, not a
    predictor."""
    subjects = np.asarray(subjects)
    yy = y[keep]
    ss = subjects[keep]
    hit = 0
    for s in np.unique(ss):
        v = yy[ss == s]
        _, cnt = np.unique(v, return_counts=True)
        hit += cnt.max()
    return float(hit / len(yy))


def purity(y, keep, subjects, thresh=0.8):
    subjects = np.asarray(subjects)
    yy, ss = y[keep], subjects[keep]
    pur, maj = [], []
    for s in np.unique(ss):
        v = yy[ss == s]
        vals, cnt = np.unique(v, return_counts=True)
        pur.append(cnt.max() / len(v))
        maj.append(vals[np.argmax(cnt)])
    pur = np.asarray(pur)
    vals, cnt = np.unique(np.asarray(maj), return_counts=True)
    return dict(n_subjects=int(len(pur)), n_at_or_above=int((pur >= thresh).sum()),
                thresh=thresh,
                majority_counts=dict((float(v), int(c)) for v, c in zip(vals, cnt)))
