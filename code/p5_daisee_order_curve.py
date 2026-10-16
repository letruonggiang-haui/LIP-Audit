"""P5/R1 - positive control for the DAiSEE clip-ordering convention.

Audit 1 on DAiSEE treats two clips as temporally adjacent when their ClipID
trailing indices differ by exactly one inside the same seven-character session
prefix.  The convention is inferred, not documented.  This script supplies the
behavioural check the internal reviewers asked for: if the trailing index really orders
clips in time, then the distance between two clips' OpenFace descriptors,
averaged over pairs of a session, must grow with the index separation, because
a face-video recording drifts.  If the index is an arbitrary key the curve is
flat.

Input  : ~/daisee/seq/{Train,Validation,Test}.npz  (X 300x22 AU sequences, ids)
Output : ~/p4/analysis/p5_daisee_order_curve.json

No labels are read and no model is trained, so the result is independent of
every engagement number in the paper.
"""
import json
import os

import numpy as np

SEQ = os.path.expanduser("~/daisee/seq")
OUT = os.path.expanduser("~/p4/analysis")
SPLITS = ["Train", "Validation", "Test"]
MAXLAG = 10
B_BOOT = 2000
N_PERM = 1000
SEED = 20260910


def clip_features(X):
    """Per clip: channel means and channel standard deviations over frames."""
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    return np.concatenate([mu, sd], axis=1)


def standardise(F):
    return (F - F.mean(0)) / (F.std(0) + 1e-8)


def session_and_index(ids):
    sess = np.array([s[:7] for s in ids])
    idx = np.array([int(s[7:]) for s in ids], dtype=np.int64)
    return sess, idx


def pair_table(F, sess, idx):
    """Every within-session pair: separation, distance, session key."""
    lag, dist, key = [], [], []
    for s in np.unique(sess):
        m = np.flatnonzero(sess == s)
        if len(m) < 2:
            continue
        Fi = F[m]
        ti = idx[m]
        d = np.sqrt(((Fi[:, None, :] - Fi[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(m), k=1)
        lag.append(np.abs(ti[iu[0]] - ti[iu[1]]))
        dist.append(d[iu])
        key.append(np.full(len(iu[0]), s))
    return (np.concatenate(lag), np.concatenate(dist), np.concatenate(key))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def within_session_rho(lag, dist, key):
    """Mean over sessions of the within-session Spearman rho(separation, distance)."""
    rhos, weights = [], []
    for s in np.unique(key):
        m = key == s
        if m.sum() < 6 or len(np.unique(lag[m])) < 2:
            continue
        r = spearman(lag[m], dist[m])
        if np.isfinite(r):
            rhos.append(r)
            weights.append(m.sum())
    rhos = np.asarray(rhos)
    weights = np.asarray(weights, dtype=float)
    return float((rhos * weights).sum() / weights.sum()), rhos, weights


def curve(lag, dist, maxlag=MAXLAG):
    out = []
    for k in range(1, maxlag + 1):
        m = lag == k
        if m.sum() == 0:
            continue
        out.append({"lag": k, "n_pairs": int(m.sum()),
                    "mean_distance": float(dist[m].mean())})
    m = lag > maxlag
    if m.sum():
        out.append({"lag": ">%d" % maxlag, "n_pairs": int(m.sum()),
                    "mean_distance": float(dist[m].mean())})
    return out


def cluster_bootstrap_diff(lag, dist, key, rng, b=B_BOOT):
    """Session-clustered interval on mean distance at lag>=5 minus lag==1."""
    sessions = np.unique(key)
    index = {s: np.flatnonzero(key == s) for s in sessions}
    vals = []
    for _ in range(b):
        draw = rng.choice(sessions, size=len(sessions), replace=True)
        rows = np.concatenate([index[s] for s in draw])
        l, d = lag[rows], dist[rows]
        m1, m5 = l == 1, l >= 5
        if m1.sum() == 0 or m5.sum() == 0:
            continue
        vals.append(d[m5].mean() - d[m1].mean())
    vals = np.asarray(vals)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def permutation_p(F, sess, idx, observed, rng, n_perm=N_PERM):
    """Shuffle the index labels inside each session; recompute the pooled rho."""
    hits = 0
    for _ in range(n_perm):
        idx_p = idx.copy()
        for s in np.unique(sess):
            m = np.flatnonzero(sess == s)
            idx_p[m] = rng.permutation(idx[m])
        lag, dist, key = pair_table(F, sess, idx_p)
        r, _, _ = within_session_rho(lag, dist, key)
        if r >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    report = {"seed": SEED, "n_perm": N_PERM, "n_boot": B_BOOT, "splits": {}}

    for split in SPLITS:
        z = np.load(os.path.join(SEQ, split + ".npz"), allow_pickle=True)
        ids = np.asarray(z["ids"], dtype=str)
        keep = np.array([len(s) == 10 and s[7:].isdigit() for s in ids])
        X = z["X"][keep].astype(np.float64)
        ids = ids[keep]
        F = standardise(clip_features(X))
        sess, idx = session_and_index(ids)

        lag, dist, key = pair_table(F, sess, idx)
        rho, per_sess, weights = within_session_rho(lag, dist, key)
        lo, hi = cluster_bootstrap_diff(lag, dist, key, rng)
        p = permutation_p(F, sess, idx, rho, rng)

        d1 = dist[lag == 1].mean()
        d5 = dist[lag >= 5].mean()
        report["splits"][split] = {
            "clips_used": int(keep.sum()),
            "clips_dropped_not_ten_char": int((~keep).sum()),
            "sessions": int(len(np.unique(sess))),
            "within_session_pairs": int(len(lag)),
            "sessions_scored": int(len(per_sess)),
            "sessions_with_positive_rho": int((per_sess > 0).sum()),
            "pooled_within_session_rho": rho,
            "permutation_p": float(p),
            "mean_distance_lag1": float(d1),
            "mean_distance_lag_ge5": float(d5),
            "difference_ge5_minus_1": float(d5 - d1),
            "difference_ci95_session_clustered": [lo, hi],
            "curve": curve(lag, dist),
        }
        print("%-11s clips %5d  sessions %4d  pairs %7d  rho %+.3f (p=%.4f)  "
              "d(lag1) %.3f  d(lag>=5) %.3f  diff %+.3f [%+.3f, %+.3f]"
              % (split, keep.sum(), len(np.unique(sess)), len(lag), rho, p,
                 d1, d5, d5 - d1, lo, hi), flush=True)

    with open(os.path.join(OUT, "p5_daisee_order_curve.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", os.path.join(OUT, "p5_daisee_order_curve.json"))


if __name__ == "__main__":
    main()
