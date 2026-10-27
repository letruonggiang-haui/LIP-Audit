"""R1 registration (REGISTRATION_R1_20261026.md), Sections 4C, 4D and 4F, on
EngageNet train and validation only.

  C   identity overlap against identity disjointness at matched training size
  D   a condition in which identity cannot help (decisive axis, per-participant
      class balance) against natural class proportions
  F1  temporal order: retrained with the nine sub-windows shuffled, against the
      same families retrained in order (MLP anchor)
  F2  temporal order: the stored frozen checkpoints read on validation with
      reversed and randomly permuted sub-windows (secondary)

Data come from ~/r1/data, which holds copies of the train and validation label
and descriptor files only; every load asserts its split and path.

Model families, training loop and hyperparameters are those of p4_train_of.py
(imported). Each run stops after the median validation-selected epoch of its
family in the stored runs (EPOCHS, 0-indexed, cosine schedule of the original
80-epoch loop), so no held-out data is used for selection. The person-level
metrics are copied verbatim from p5_person_variance_ci.py of the repository
(the server copy predates the label-free release and imports differently).

Usage: python r1_engagenet.py C|D|F1|F2
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score

sys.path.insert(0, os.path.expanduser("~/p4/code"))
from p4_train_of import BUILDERS  # noqa: E402

DATA = os.path.expanduser("~/r1/data")
RUNS = os.path.expanduser("~/p4/runs")
OUT = os.path.expanduser("~/r1/e")
SPLITS = ("train", "validation")
EPOCHS = {"mlp": 3, "lstm": 4, "cnnlstm": 1, "tcn": 6, "transformer": 9}
SEEDS = (0, 1, 2)
LEVELS = np.arange(4)
B = 2000
SEED = 20261026
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- metrics
# verbatim from release_lip_audit/code/p5_person_variance_ci.py
def eta2_between(v, grp):
    gm = v.mean()
    sst = ((v - gm) ** 2).sum()
    if sst <= 0:
        return float("nan")
    ssb = 0.0
    for g in np.unique(grp):
        k = grp == g
        ssb += k.sum() * (v[k].mean() - gm) ** 2
    return float(ssb / sst)


def within_person_auc(score, target, subj):
    aucs, ns = [], 0
    for s in np.unique(subj):
        k = subj == s
        t = target[k]
        if t.min() == t.max():
            continue
        aucs.append(roc_auc_score(t, score[k]))
        ns += 1
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), ns


def rho_between(score, target, subj):
    us = np.unique(subj)
    if len(us) < 3:
        return float("nan")
    means = np.array([score[subj == s].mean() for s in us])
    frac = np.array([target[subj == s].mean() for s in us])
    if np.all(means == means[0]) or np.all(frac == frac[0]):
        return float("nan")
    return float(spearmanr(means, frac).statistic)


def point_stats(P, y, subj):
    ord_score = (P * LEVELS).sum(1)
    m = y >= 2
    sc = P[:, 3] / (P[:, 2] + P[:, 3] + 1e-9)
    t = (y[m] == 3).astype(int)
    s_dec, sj_dec = sc[m], subj[m]
    wp, n_sub = within_person_auc(s_dec, t, sj_dec)
    return {
        "eta2_score_ordinal": eta2_between(ord_score, subj),
        "eta2_label_ordinal": eta2_between(y.astype(float), subj),
        "eta2_score_decisive": eta2_between(s_dec, sj_dec),
        "eta2_label_decisive": eta2_between(t.astype(float), sj_dec),
        "rho_b": rho_between(s_dec, t, sj_dec),
        "auc_within_person": wp,
        "n_participants_usable": n_sub,
    }
# ---------------------------------------------------------------- end verbatim


def four_class(P, y, subj):
    r = point_stats(P, y, subj)
    pred = P.argmax(1)
    r["acc"] = float((pred == y).mean())
    r["uar"] = float(recall_score(y, pred, average="macro", zero_division=0))
    return r


def binary(P, t, subj):
    s = P[:, 1]
    wp, n = within_person_auc(s, t, subj)
    return {"acc": float(((s > 0.5).astype(int) == t).mean()),
            "auc_within_person": wp, "n_participants_usable": n,
            "eta2_score": eta2_between(s, subj),
            "auc_blended": float(roc_auc_score(t, s))}


# ---------------------------------------------------------------- data
def load(split):
    assert split in SPLITS, split
    fz, fl = os.path.join(DATA, "openface98_%s.npz" % split), os.path.join(DATA, "labels_%s.csv" % split)
    for f in (fz, fl):
        assert "test" not in f.lower(), f
    z = np.load(fz, allow_pickle=True)
    chunk = np.asarray(z["chunk"], dtype=str)
    lab = pd.read_csv(fl)
    lab["chunk"] = lab["chunk"].astype(str)
    order = {c: i for i, c in enumerate(chunk)}
    absent = set(lab["chunk"]) - set(order)
    if absent:
        raise SystemExit("%s: %d labelled chunks without descriptor" % (split, len(absent)))
    lab["row"] = lab["chunk"].map(order)
    lab = lab[lab["y"].notna()].reset_index(drop=True)
    rows = lab["row"].values
    return {"Xw": z["X"].astype(np.float32)[rows], "Xs": z["X_seq"].astype(np.float32)[rows],
            "y": lab["y"].values.astype(np.int64), "subj": lab["subj"].values.astype(int),
            "rec": (lab["subj"].astype(str) + "|" + lab["sess"].astype(str) + "|" + lab["vid"].astype(str)).values,
            "chunk": lab["chunk"].values}


# ---------------------------------------------------------------- training
def fit(fam, X, y, seed, n_class=4, shuffle_order=False):
    """The p4_train_of.py loop, stopped after EPOCHS[fam]."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    flat = X.reshape(-1, X.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    Xn = torch.from_numpy((X - mu) / sd)
    yt = torch.from_numpy(y)
    model = BUILDERS[fam](X.shape[-1], n_class=n_class).to(DEV)
    w = torch.tensor(len(y) / (n_class * np.bincount(y, minlength=n_class).clip(1)),
                     dtype=torch.float32, device=DEV)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    n = len(y)
    for ep in range(EPOCHS[fam] + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            k = perm[i:i + 64]
            if len(k) < 2:
                continue
            xb = Xn[k]
            if shuffle_order:
                idx = torch.argsort(torch.rand(xb.shape[0], xb.shape[1]), dim=1)
                xb = torch.gather(xb, 1, idx.unsqueeze(-1).expand(-1, -1, xb.shape[2]))
            xb, yb = xb.to(DEV), yt[k].to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model, mu, sd


def predict(model, X, mu, sd, perm=None):
    Xn = (X - mu) / sd
    if perm is not None:
        Xn = np.take_along_axis(Xn, perm[:, :, None], axis=1)
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(np.ascontiguousarray(Xn)).to(DEV)), 1).cpu().numpy()


def penultimate(model, fam, X, mu, sd):
    x = torch.from_numpy((X - mu) / sd).to(DEV)
    with torch.no_grad():
        if fam == "mlp":
            h = model.net[:-1](x)
        else:  # tcn
            h = model.head[:-1](model.body(x.transpose(1, 2))[:, :, -1])
    return h.cpu().numpy()


def pick(d, idx):
    return {k: v[idx] for k, v in d.items()}


# ---------------------------------------------------------------- bootstrap
def boot_diff(groups, fa, fb, metrics, rng):
    """Participant-clustered percentile intervals of mean_seeds metric(a) -
    mean_seeds metric(b), for every metric in one pass. fa/fb(index,
    relabelled subj) -> list of per-seed metric dicts. A participant drawn twice
    counts as two groups, as in p5_person_variance_ci.py."""
    us = np.unique(groups)
    pos = {u: np.flatnonzero(groups == u) for u in us}
    out = {k: [] for k in metrics}
    for _ in range(B):
        draw = rng.choice(us, len(us), replace=True)
        idx = np.concatenate([pos[u] for u in draw])
        rel = np.concatenate([np.full(len(pos[u]), j) for j, u in enumerate(draw)])
        ma, mb = fa(idx, rel), fb(idx, rel)
        for k in metrics:
            out[k].append(np.nanmean([m[k] for m in ma]) - np.nanmean([m[k] for m in mb]))
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in out.items()}


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=1)
    print("wrote", os.path.join(OUT, name), flush=True)


# ---------------------------------------------------------------- C
def run_C():
    d = load("train")
    rng = np.random.default_rng(SEED)
    nrec = pd.Series(d["rec"]).groupby(d["subj"]).nunique()
    elig = np.array(sorted(nrec[nrec >= 2].index))
    print("participants %d, with >= 2 recordings %d" % (len(nrec), len(elig)), flush=True)
    folds = np.array_split(rng.permutation(elig), 5)
    H_idx, probs = [], {}
    for f, fold in enumerate(folds):
        held = []
        for p in fold:
            recs = sorted(set(d["rec"][d["subj"] == p]))
            r = recs[rng.integers(len(recs))]
            held.append(np.flatnonzero(d["rec"] == r))
        H = np.concatenate(held)
        inH = np.zeros(len(d["y"]), bool)
        inH[H] = True
        infold = np.isin(d["subj"], fold)
        disj = np.flatnonzero(~infold)
        over_all = np.flatnonzero(~inH)
        extra = len(over_all) - len(disj)
        drop = rng.choice(np.flatnonzero(~infold & ~inH), extra, replace=False)
        over = np.setdiff1d(over_all, drop)
        assert len(over) == len(disj)
        print("fold %d: held-out %d chunks of %d participants; train %d chunks each"
              % (f, len(H), len(fold), len(disj)), flush=True)
        H_idx.append(H)
        for fam in ("mlp", "tcn"):
            key = "Xw" if fam == "mlp" else "Xs"
            for cond, tr in (("overlap", over), ("disjoint", disj)):
                for s in SEEDS:
                    m, mu, sd = fit(fam, d[key][tr], d["y"][tr], s)
                    probs.setdefault((fam, cond, s), []).append(predict(m, d[key][H], mu, sd))
    H = np.concatenate(H_idx)
    y, subj = d["y"][H], d["subj"][H]
    res = {"n_heldout_chunks": int(len(H)), "n_heldout_participants": int(len(np.unique(subj))),
           "folds": [sorted(int(p) for p in f) for f in folds], "families": {}}
    np.savez_compressed(os.path.join(OUT, "C_probs.npz"), H=H, y=y, subj=subj,
                        **{"%s_%s_s%d" % k: np.concatenate(v) for k, v in probs.items()})
    for fam in ("mlp", "tcn"):
        P = {c: [np.concatenate(probs[(fam, c, s)]) for s in SEEDS] for c in ("overlap", "disjoint")}
        point = {c: [four_class(p, y, subj) for p in P[c]] for c in P}
        r = {c: {k: float(np.nanmean([m[k] for m in point[c]])) for k in point[c][0]} for c in point}
        brng = np.random.default_rng(SEED + 1)
        fa = lambda idx, rel: [four_class(p[idx], y[idx], rel) for p in P["overlap"]]  # noqa: E731
        fb = lambda idx, rel: [four_class(p[idx], y[idx], rel) for p in P["disjoint"]]  # noqa: E731
        keys = ("acc", "uar", "auc_within_person", "eta2_score_ordinal", "eta2_score_decisive", "rho_b")
        cis = boot_diff(subj, fa, fb, keys, brng)
        diffs = {k: {"point": r["overlap"][k] - r["disjoint"][k], "ci": cis[k]} for k in keys}
        res["families"][fam] = {"overlap": r["overlap"], "disjoint": r["disjoint"], "overlap_minus_disjoint": diffs}
        print(fam, json.dumps(diffs), flush=True)
    save("C_identity_overlap.json", res)


# ---------------------------------------------------------------- D
def run_D():
    d = load("train")
    rng = np.random.default_rng(SEED + 10)
    m = d["y"] >= 2
    dd = pick(d, np.flatnonzero(m))
    dd["t"] = (dd["y"] == 3).astype(np.int64)
    keep, bal, nat = [], [], []
    for p in np.unique(dd["subj"]):
        k = np.flatnonzero(dd["subj"] == p)
        n1 = int(dd["t"][k].sum())
        n0 = len(k) - n1
        if min(n0, n1) < 8:
            continue
        keep.append(p)
        c = min(n0, n1)
        i0 = rng.choice(k[dd["t"][k] == 0], c, replace=False)
        i1 = rng.choice(k[dd["t"][k] == 1], c, replace=False)
        bal.append(np.concatenate([i0, i1]))
        nat.append(rng.choice(k, 2 * c, replace=False))
    keep = np.array(keep)
    bal, nat = np.concatenate(bal), np.concatenate(nat)
    print("participants with >= 8 chunks of each decisive class: %d; balanced %d chunks"
          % (len(keep), len(bal)), flush=True)
    folds = np.array_split(rng.permutation(keep), 5)
    out = {(fam, c, s): [] for fam in ("mlp", "tcn") for c in ("balanced", "natural") for s in SEEDS}
    out_nat = {k: [] for k in out}
    probe = {k: [] for k in out}
    E_bal, E_nat = [], []
    for f, fold in enumerate(folds):
        te_b = bal[np.isin(dd["subj"][bal], fold)]
        te_n = nat[np.isin(dd["subj"][nat], fold)]
        E_bal.append(te_b)
        E_nat.append(te_n)
        for cond, pool in (("balanced", bal), ("natural", nat)):
            tr = pool[~np.isin(dd["subj"][pool], fold)]
            for fam in ("mlp", "tcn"):
                key = "Xw" if fam == "mlp" else "Xs"
                for s in SEEDS:
                    mdl, mu, sd = fit(fam, dd[key][tr], dd["t"][tr], s, n_class=2)
                    out[(fam, cond, s)].append(predict(mdl, dd[key][te_b], mu, sd))
                    out_nat[(fam, cond, s)].append(predict(mdl, dd[key][te_n], mu, sd))
                    # identity probe on the representation, training participants
                    H = penultimate(mdl, fam, dd[key][tr], mu, sd)
                    sj = dd["subj"][tr]
                    prng = np.random.default_rng(SEED + 100 + s)
                    is_tr = np.zeros(len(tr), bool)
                    for p in np.unique(sj):
                        k = np.flatnonzero(sj == p)
                        is_tr[prng.choice(k, int(round(0.75 * len(k))), replace=False)] = True
                    mu_h, sd_h = H[is_tr].mean(0), H[is_tr].std(0) + 1e-6
                    clf = LogisticRegression(C=1.0, max_iter=3000)
                    clf.fit((H[is_tr] - mu_h) / sd_h, sj[is_tr])
                    acc = float((clf.predict((H[~is_tr] - mu_h) / sd_h) == sj[~is_tr]).mean())
                    K = len(np.unique(sj))
                    probe[(fam, cond, s)].append({"fold": f, "top1": acc, "n_identities": K,
                                                  "times_chance": acc * K})
        print("fold %d done" % f, flush=True)
    Eb, En = np.concatenate(E_bal), np.concatenate(E_nat)
    tb, sb = dd["t"][Eb], dd["subj"][Eb]
    tn, sn = dd["t"][En], dd["subj"][En]
    res = {"n_participants": int(len(keep)), "n_eval_balanced": int(len(Eb)), "families": {}}
    for fam in ("mlp", "tcn"):
        Pb = {c: [np.concatenate(out[(fam, c, s)]) for s in SEEDS] for c in ("balanced", "natural")}
        Pn = {c: [np.concatenate(out_nat[(fam, c, s)]) for s in SEEDS] for c in ("balanced", "natural")}
        rb = {c: {k: float(np.nanmean([binary(p, tb, sb)[k] for p in Pb[c]])) for k in
                  ("acc", "auc_within_person", "eta2_score", "auc_blended")} for c in Pb}
        rn = {c: {k: float(np.nanmean([binary(p, tn, sn)[k] for p in Pn[c]])) for k in
                  ("acc", "auc_within_person", "eta2_score", "auc_blended")} for c in Pn}
        brng = np.random.default_rng(SEED + 11)
        fa = lambda idx, rel: [binary(p[idx], tb[idx], rel) for p in Pb["balanced"]]  # noqa: E731
        fb = lambda idx, rel: [binary(p[idx], tb[idx], rel) for p in Pb["natural"]]  # noqa: E731
        keys = ("eta2_score", "auc_within_person", "acc")
        cis = boot_diff(sb, fa, fb, keys, brng)
        diffs = {k: {"point": rb["balanced"][k] - rb["natural"][k], "ci": cis[k]} for k in keys}
        pr = {c: {"times_chance_mean": float(np.mean([q["times_chance"] for s in SEEDS for q in probe[(fam, c, s)]])),
                  "times_chance_min": float(np.min([q["times_chance"] for s in SEEDS for q in probe[(fam, c, s)]])),
                  "top1_mean": float(np.mean([q["top1"] for s in SEEDS for q in probe[(fam, c, s)]]))}
              for c in ("balanced", "natural")}
        res["families"][fam] = {"eval_on_balanced_heldout": rb, "eval_on_natural_heldout": rn,
                                "balanced_minus_natural_on_balanced_heldout": diffs, "identity_probe": pr}
        print(fam, json.dumps(diffs), json.dumps(pr), flush=True)
    save("D_identity_futile.json", res)


# ---------------------------------------------------------------- F1, F2
def run_F1():
    tr, va = load("train"), load("validation")
    yv, sv = va["y"], va["subj"]
    P = {}
    for fam in ("mlp", "lstm", "cnnlstm", "tcn", "transformer"):
        key = "Xw" if fam == "mlp" else "Xs"
        conds = ("ordered",) if fam == "mlp" else ("ordered", "shuffled")
        for c in conds:
            for s in SEEDS:
                m, mu, sd = fit(fam, tr[key], tr["y"], s, shuffle_order=(c == "shuffled"))
                P[(fam, c, s)] = predict(m, va[key], mu, sd)
                if c == "shuffled":
                    prng = np.random.default_rng(SEED + 200 + s)
                    perm = np.argsort(prng.random((len(yv), va[key].shape[1])), axis=1)
                    P[(fam, "shuffled_readshuffled", s)] = predict(m, va[key], mu, sd, perm=perm)
        print(fam, "done", flush=True)
    np.savez_compressed(os.path.join(OUT, "F1_probs.npz"), y=yv, subj=sv,
                        **{"%s_%s_s%d" % k: v for k, v in P.items()})
    res = {"families": {}}
    brng = np.random.default_rng(SEED + 21)
    for fam in ("mlp", "lstm", "cnnlstm", "tcn", "transformer"):
        conds = [c for c in ("ordered", "shuffled", "shuffled_readshuffled") if (fam, c, 0) in P]
        r = {c: {k: float(np.nanmean([four_class(P[(fam, c, s)], yv, sv)[k] for s in SEEDS]))
                 for k in ("acc", "uar", "auc_within_person", "eta2_score_ordinal")} for c in conds}
        entry = {"point": r}
        if fam != "mlp":
            fa = lambda idx, rel, fam=fam: [four_class(P[(fam, "shuffled", s)][idx], yv[idx], rel) for s in SEEDS]  # noqa: E731
            fb = lambda idx, rel, fam=fam: [four_class(P[(fam, "ordered", s)][idx], yv[idx], rel) for s in SEEDS]  # noqa: E731
            keys = ("acc", "uar", "auc_within_person")
            cis = boot_diff(sv, fa, fb, keys, brng)
            entry["shuffled_minus_ordered"] = {k: {"point": r["shuffled"][k] - r["ordered"][k], "ci": cis[k]}
                                               for k in keys}
        res["families"][fam] = entry
        print(fam, json.dumps(entry.get("shuffled_minus_ordered", r)), flush=True)
    save("F1_temporal_shuffle_retrained.json", res)


def run_F2():
    va = load("validation")
    yv, sv = va["y"], va["subj"]
    res = {}
    for fam in ("lstm", "cnnlstm", "tcn", "transformer"):
        rows = []
        for s in SEEDS:
            ck = torch.load(os.path.join(RUNS, "of_%s_s%d_best.pth" % (fam, s)), map_location=DEV, weights_only=False)
            m = BUILDERS[fam](98).to(DEV)
            m.load_state_dict(ck["state"])
            m.eval()
            mu, sd = ck["mu"], ck["sd"]
            X = va["Xs"]
            T = X.shape[1]
            r = {"original": four_class(predict(m, X, mu, sd), yv, sv),
                 "reversed": four_class(predict(m, X, mu, sd, perm=np.tile(np.arange(T)[::-1], (len(yv), 1))), yv, sv)}
            prng = np.random.default_rng(SEED + 300 + s)
            perms = [four_class(predict(m, X, mu, sd, perm=np.argsort(prng.random((len(yv), T)), axis=1)), yv, sv)
                     for _ in range(20)]
            r["permuted_mean"] = {k: float(np.nanmean([q[k] for q in perms])) for k in ("acc", "uar", "auc_within_person")}
            rows.append(r)
        res[fam] = {c: {k: float(np.nanmean([row[c][k] for row in rows])) for k in ("acc", "uar", "auc_within_person")}
                    for c in ("original", "reversed", "permuted_mean")}
        print(fam, json.dumps(res[fam]), flush=True)
    save("F2_temporal_shuffle_frozen.json", res)


if __name__ == "__main__":
    t0 = time.time()
    {"C": run_C, "D": run_D, "F1": run_F1, "F2": run_F2}[sys.argv[1]]()
    print("minutes %.1f" % ((time.time() - t0) / 60), flush=True)
