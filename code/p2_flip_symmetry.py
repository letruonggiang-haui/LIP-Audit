"""Additional check from the internal review round before submission (Major 1): the
feature-independence assumption of the flip-rate model Eq.(1).

Question: do adjacent clip pairs (gap-1, same session, both at level >=2) that DISAGREE in
label differ systematically in observable behaviour (17 AU intensity + pose 3 + gaze 2,
time-averaged, z-scored within split) from pairs that AGREE?

- If no difference is detected -> supports the reading "flips are not tied to visible change"
  (consistent with the feature-independence assumption of Eq. 1).
- If there is a difference -> flips come with real behavioural change -> the "state dynamics"
  reading (r>0), which is already bounded by the sensitivity curve q(r) in the paper.
Both directions are reported faithfully. Uses only the public labels + the existing seq npz;
NOT a model evaluation (does not touch the sealed-test protocol).

Run: ~/p2/venv/bin/python p2_flip_symmetry.py
"""
import os
import numpy as np
import pandas as pd

LABELS = os.path.expanduser("~/daisee/DAiSEE/Labels")
SEQ = os.path.expanduser("~/daisee/seq")
rng = np.random.default_rng(0)
B = 10000


def load_split(split):
    d = np.load(os.path.join(SEQ, f"{split}.npz"), allow_pickle=True)
    X, ids = d["X"], [str(s) for s in d["ids"]]
    lab = pd.read_csv(os.path.join(LABELS, f"{split}Labels.csv"))
    lab.columns = [c.strip() for c in lab.columns]
    lab["stem"] = lab["ClipID"].astype(str).str.replace(r"\.(avi|mp4)$", "", regex=True)
    ymap = dict(zip(lab["stem"], lab["Engagement"].astype(int)))
    keep = [i for i, s in enumerate(ids) if s in ymap]
    X = X[keep]
    ids = [ids[i] for i in keep]
    y = np.array([ymap[s] for s in ids])
    # clip features: time average, then z-score each dimension within the split
    F = np.nanmean(X, axis=1)
    F = (F - np.nanmean(F, 0)) / (np.nanstd(F, 0) + 1e-9)
    return F, ids, y


def analyze(split):
    F, ids, y = load_split(split)
    sess = np.array([s[:7] for s in ids])
    order = np.array([int(s[7:]) for s in ids])
    d_agree, d_dis = [], []
    for s in np.unique(sess):
        idx = np.where(sess == s)[0]
        idx = idx[np.argsort(order[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            if order[b] - order[a] != 1:
                continue
            if y[a] < 2 or y[b] < 2:
                continue
            dist = float(np.linalg.norm(F[a] - F[b]))
            (d_agree if y[a] == y[b] else d_dis).append(dist)
    d_agree, d_dis = np.array(d_agree), np.array(d_dis)
    na, nd = len(d_agree), len(d_dis)
    diff = d_dis.mean() - d_agree.mean()
    pooled_sd = np.concatenate([d_agree, d_dis]).std()
    # permute the agree/disagree labels
    allv = np.concatenate([d_agree, d_dis])
    perm = np.empty(B)
    for i in range(B):
        pr = rng.permutation(allv)
        perm[i] = pr[na:].mean() - pr[:na].mean()
    p_two = float((np.abs(perm) >= abs(diff)).mean())
    # Cliff's delta
    gt = sum((d_dis[:, None] > d_agree[None, :]).sum() for _ in [0]) / (na * nd)
    lt = (d_dis[:, None] < d_agree[None, :]).sum() / (na * nd)
    cliff = float(gt - lt)
    print(f"{split}: cap agree n={na} (dist TB {d_agree.mean():.3f}), "
          f"disagree n={nd} (dist TB {d_dis.mean():.3f})")
    print(f"  chenh = {diff:+.3f} ({diff/ (pooled_sd+1e-9):+.2f} SD don vi), "
          f"hoan vi 2 phia p = {p_two:.3f}, Cliff's delta = {cliff:+.3f}")


for sp in ("Train", "Validation", "Test"):
    analyze(sp)
