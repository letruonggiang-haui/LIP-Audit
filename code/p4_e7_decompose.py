"""E7 - within-person decomposition of the reproduced EngageNet systems.

Runs on the stored per-clip probabilities, so nothing is retrained and no split
is touched again. For every system it separates two abilities that a blended AUC
conflates: ranking people by their propensity toward a class, and ranking the
clips of one person against each other. Identity shortcuts supply the first and
never the second.

Two axes are read, both defined on the four ordinal levels:
  decisive        Highly-Engaged against Engaged, among clips at Engaged or above
  disengagement   the two lower levels against the two upper ones

and one ordinal score, the posterior mean level, for the variance decomposition.

It also measures the placeholder effect that E3a exposed: face-detection failure
predicts Not-Engaged, every model here sees a zero vector for those clips, and
the question is how much of the reported accuracy that accounts for.

Usage:  python p4_e7_decompose.py
"""
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

P4 = os.path.expanduser("~/p4")
OUT = os.path.join(P4, "analysis")
LEVELS = np.arange(4)


def within_person_auc(score, target, subj):
    """Clip-count weighted mean of the per-participant AUC, Eq. (5)."""
    aucs, w = [], []
    for s in np.unique(subj):
        k = subj == s
        if len(np.unique(target[k])) == 2:
            aucs.append(roc_auc_score(target[k], score[k]))
            w.append(int(k.sum()))
    if not aucs:
        return float("nan"), 0
    return float(np.average(aucs, weights=w)), len(aucs)


def within_person_null(score, target, subj, n_perm=1000, seed=20260905):
    """Shuffle the target inside each participant; the pair of abilities that
    identity can supply is destroyed and the within-person one is not."""
    rng = np.random.default_rng(seed)
    obs, _ = within_person_auc(score, target, subj)
    blocks = [np.flatnonzero(subj == s) for s in np.unique(subj)]
    draws = np.empty(n_perm)
    t = target.copy()
    for r in range(n_perm):
        for b in blocks:
            t[b] = rng.permutation(target[b])
        draws[r], _ = within_person_auc(score, t, subj)
    p = (int((draws >= obs).sum()) + 1) / (n_perm + 1)
    return dict(observed=obs, null_mean=float(np.nanmean(draws)),
                p=float(p), n_perm=n_perm)


def centre_per_subject(v, subj):
    out = v.copy()
    for s in np.unique(subj):
        k = subj == s
        out[k] = v[k] - v[k].mean() + v.mean()
    return out


def eta2_between(v, subj):
    gm = v.mean()
    sst = ((v - gm) ** 2).sum()
    if sst <= 0:
        return float("nan")
    ssb = sum((subj == s).sum() * (v[subj == s].mean() - gm) ** 2
              for s in np.unique(subj))
    return float(ssb / sst)


def analyse(path):
    z = np.load(path, allow_pickle=True)
    P, y, subj = z["P"], z["y"], z["subj"]
    # the MARLIN runs predate the tracked-frame record
    frames_ok = z["frames_ok"] if "frames_ok" in z.files else None
    tag = os.path.basename(path)[:-len("_val_probs.npz")]
    row = {"tag": tag, "n": int(len(y))}

    # ---- decisive axis: 3 against 2, among clips at 2 or above -------------
    m = y >= 2
    sc = P[:, 3] / (P[:, 2] + P[:, 3] + 1e-9)
    t = (y[m] == 3).astype(int)
    s_dec, sj_dec = sc[m], subj[m]
    blended = float(roc_auc_score(t, s_dec))
    wp, n_sub = within_person_auc(s_dec, t, sj_dec)
    frac = np.array([t[sj_dec == s].mean() for s in np.unique(sj_dec)])
    means = np.array([s_dec[sj_dec == s].mean() for s in np.unique(sj_dec)])
    null = within_person_null(s_dec, t, sj_dec)
    cen = centre_per_subject(s_dec, sj_dec)
    row["decisive"] = {
        "n_clips": int(m.sum()), "n_participants_usable": n_sub,
        "auc_blended": blended, "auc_within_person": wp,
        "auc_after_centring": float(roc_auc_score(t, cen)),
        "rho_between": float(spearmanr(means, frac).statistic),
        "within_person_null": null,
        "eta2_between_score": eta2_between(s_dec, sj_dec),
        "eta2_between_label": eta2_between(t.astype(float), sj_dec)}

    # ---- disengagement axis: 0-1 against 2-3 ------------------------------
    td = (y <= 1).astype(int)
    sd = P[:, 0] + P[:, 1]
    wp_d, n_sub_d = within_person_auc(sd, td, subj)
    row["disengagement"] = {
        "auc_blended": float(roc_auc_score(td, sd)),
        "auc_within_person": wp_d, "n_participants_usable": n_sub_d,
        "auc_after_centring": float(roc_auc_score(td, centre_per_subject(sd, subj))),
        "within_person_null": within_person_null(sd, td, subj)}

    # ---- ordinal score, variance decomposition ----------------------------
    ord_score = (P * LEVELS).sum(1)
    row["ordinal"] = {"eta2_between_score": eta2_between(ord_score, subj),
                      "eta2_between_label": eta2_between(y.astype(float), subj)}

    # ---- the placeholder effect -------------------------------------------
    pred = P.argmax(1)
    if frames_ok is None:
        row["placeholder"] = None
        return row
    untr = frames_ok == 0
    row["placeholder"] = {
        "n_untracked": int(untr.sum()),
        "accuracy_all": float((pred == y).mean()),
        "accuracy_tracked_only": float((pred[~untr] == y[~untr]).mean()),
        "accuracy_on_untracked": float((pred[untr] == y[untr]).mean()) if untr.any() else None,
        "untracked_predicted_not_engaged": float((pred[untr] == 0).mean()) if untr.any() else None,
        "untracked_truly_not_engaged": float((y[untr] == 0).mean()) if untr.any() else None}
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(P4, "probs", "*_val_probs.npz")))
    files = [f for f in files if "smoke" not in f]
    rows = []
    for f in files:
        r = analyse(f)
        rows.append(r)
        d, g = r["decisive"], r["disengagement"]
        print("%-20s decisive blended %.4f within %.4f (p %.4f) centred %.4f rho_b %+.2f | "
              "diseng blended %.4f within %.4f | eta2 score %.3f vs label %.3f"
              % (r["tag"], d["auc_blended"], d["auc_within_person"],
                 d["within_person_null"]["p"], d["auc_after_centring"],
                 d["rho_between"], g["auc_blended"], g["auc_within_person"],
                 r["ordinal"]["eta2_between_score"], r["ordinal"]["eta2_between_label"]),
              flush=True)

    with open(os.path.join(OUT, "E7_decompose.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nwrote", os.path.join(OUT, "E7_decompose.json"))

    by = {}
    for r in rows:
        by.setdefault(r["tag"].rsplit("_s", 1)[0], []).append(r)
    print("\n%-16s %-14s %-14s %-14s %-10s %s"
          % ("system", "blended AUC", "within-person", "after centring", "rho_b", "placeholder cost"))
    for m, rs in sorted(by.items()):
        f = lambda p, k: np.mean([r[p][k] for r in rs])
        ph = [r["placeholder"] for r in rs if r["placeholder"]]
        cost = np.mean([p["accuracy_all"] - p["accuracy_tracked_only"]
                        for p in ph]) if ph else float("nan")
        print("%-16s %-14.4f %-14.4f %-14.4f %-10.2f %+.4f"
              % (m, f("decisive", "auc_blended"), f("decisive", "auc_within_person"),
                 f("decisive", "auc_after_centring"), f("decisive", "rho_between"), cost))


if __name__ == "__main__":
    main()
