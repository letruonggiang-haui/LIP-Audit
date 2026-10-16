"""P6 - the three probe controls the round-2 panel asked for (RV5-A1, A4, A5).

None of them retrains anything.  Every representation is either a stored
descriptor or the vector entering the final classifier layer of a stored
checkpoint, exactly as in p4_e6_probe.py and p5_probe_recdisjoint.py.  Only the
EngageNet training split is touched; no sealed partition is read.

A1  The recording-disjoint control of p5 moves two things at once: it removes
    the recording overlap AND it drops the pool from 91 participants to the 78
    that carry two recordings, which is an easier problem at the same multiple
    of chance.  Here the random split is re-run on exactly those 78, so the two
    splits differ in recording overlap alone.

A4  The probe may be reading participant-level acquisition signatures rather
    than anything about the face.  Two measurements:
      (a) fit the same probe on the acquisition metadata alone -- the chunk's
          usable frame count and its container frame count -- and see what
          multiple of chance that reaches.  This bounds the confound directly.
      (b) restrict the pool to participants that share one capture rate and
          re-run the representation probe there, so the signature carries no
          information between classes.

A5  The probe is credited to the models, but every OpenFace-derived system sits
    on the same 98-d descriptor.  Two representations with no model in them are
    probed for comparison: the raw 98-d whole-chunk descriptor and the raw
    MARLIN tensor.  The difference between a fine-tuned checkpoint and its own
    input descriptor is the model-attributable increment.

Output: ~/p4/analysis/p6_probe_controls.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/p4/code"))
from p4_train_of import BUILDERS, load_split  # noqa: E402

P4 = os.path.expanduser("~/p4")
OUT = os.path.join(P4, "analysis")
SEED = 0


# --------------------------------------------------------------- helpers
def embed(model, X, batch=256):
    feats = []
    hook_out = {}

    def hook(_m, inp, _out):
        hook_out["v"] = inp[0].detach().cpu().numpy()

    last = model.head[-1] if hasattr(model, "head") else model.net[-1]
    h = last.register_forward_hook(hook)
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(X), batch):
            model(torch.from_numpy(X[i:i + batch]).to(dev))
            feats.append(hook_out["v"])
    h.remove()
    return np.concatenate(feats)


def random_split(subj, seed, keep=None):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for s in np.unique(subj):
        if keep is not None and s not in keep:
            continue
        idx = np.flatnonzero(subj == s)
        rng.shuffle(idx)
        k = max(1, int(round(0.75 * len(idx))))
        tr.append(idx[:k])
        te.append(idx[k:])
    return (np.concatenate(tr),
            np.concatenate([t for t in te if len(t)]))


def recording_split(subj, rec, seed, keep=None):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for s in np.unique(subj):
        if keep is not None and s not in keep:
            continue
        idx = np.flatnonzero(subj == s)
        recs = np.unique(rec[idx])
        if len(recs) < 2:
            continue
        held = recs[rng.integers(len(recs))]
        te.append(idx[rec[idx] == held])
        tr.append(idx[rec[idx] != held])
    return np.concatenate(tr), np.concatenate(te)


def fit_logistic(Z, subj, tr, te):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=1000, tol=1e-3)
    clf.fit(Z[tr], subj[tr])
    return float((clf.predict(Z[te]) == subj[te]).mean())


def standardise(Z):
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-6)


def probe(Z, subj, rec, seed, keep, split_kind):
    if split_kind == "random":
        tr, te = random_split(subj, seed, keep)
    else:
        tr, te = recording_split(subj, rec, seed, keep)
    classes = len(np.unique(subj[tr]))
    acc = fit_logistic(Z, subj, tr, te)
    return dict(split=split_kind, n_classes=classes, chance=1.0 / classes,
                acc=round(acc, 4), M=round(acc * classes, 2),
                n_probe_train=int(len(tr)), n_probe_test=int(len(te)))


# --------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    Xw, Xs, _y, lab = load_split("train")
    subj = lab["subj"].values
    rec = (lab["subj"].astype(str) + "|" + lab["sess"].astype(str) + "|"
           + lab["vid"].astype(str)).values
    frames_ok = lab["frames_ok"].values.astype(float)

    z = np.load(os.path.join(P4, "feats", "openface98_train.npz"),
                allow_pickle=True)
    chunk_all = np.asarray(z["chunk"], dtype=str)
    order = {c: i for i, c in enumerate(chunk_all)}
    rows = np.array([order[c] for c in lab["chunk"].astype(str)])
    frames_total = z["frames_total"][rows].astype(float)

    multi = np.array([len(np.unique(rec[subj == s])) >= 2
                      for s in np.unique(subj)])
    keep78 = set(np.unique(subj)[multi])

    res = {"n_clips": int(len(subj)),
           "n_participants": int(len(np.unique(subj))),
           "n_participants_multi_recording": int(len(keep78))}

    # ---------------- A4a: what the acquisition metadata alone can decode
    A = np.stack([frames_ok, frames_total,
                  frames_total - frames_ok,
                  np.log1p(frames_total)], axis=1)
    A = standardise(A)
    res["A4a_acquisition_only_probe"] = {
        "features": ["frames_ok", "frames_total", "frames_total-frames_ok",
                     "log1p(frames_total)"],
        "random_all91": probe(A, subj, rec, SEED, None, "random"),
        "recording_78": probe(A, subj, rec, SEED, keep78, "recording"),
    }

    # per-participant capture signature, for the matched subgroup
    fps = {}
    for s in np.unique(subj):
        m = subj == s
        fps[s] = round(float(np.median(frames_ok[m])) / 10.0, 1)
    from collections import Counter
    modal_fps, n_modal = Counter(fps.values()).most_common(1)[0]
    keep_fps = set(s for s in fps if fps[s] == modal_fps)
    keep_fps_multi = keep_fps & keep78
    res["A4b_capture_groups"] = {
        "modal_fps": modal_fps,
        "participants_at_modal_fps": int(n_modal),
        "of_which_multi_recording": int(len(keep_fps_multi)),
        "fps_histogram": dict(Counter(fps.values())),
    }

    # ---------------- A5 + A1: representations
    reps = {}
    reps["raw_openface98"] = standardise(Xw.astype(np.float32))
    reps["raw_openface98_seq"] = standardise(
        Xs.reshape(len(Xs), -1).astype(np.float32))
    try:
        zm = np.load(os.path.join(P4, "data", "marlin_train.npz"),
                     allow_pickle=True)
        cm = np.asarray(zm["chunk"], dtype=str)
        om = {c: i for i, c in enumerate(cm)}
        have = np.array([c in om for c in lab["chunk"].astype(str)])
        if have.all():
            rm = np.array([om[c] for c in lab["chunk"].astype(str)])
            Xm = zm["X"][rm].astype(np.float32)
            reps["raw_marlin"] = standardise(Xm.reshape(len(Xm), -1))
        else:
            res["marlin_note"] = ("%d of %d chunks absent from marlin_train"
                                  % int((~have).sum()))
    except Exception as exc:                      # noqa: BLE001
        res["marlin_note"] = "marlin tensors unavailable: %s" % exc

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ck in sorted(glob.glob(os.path.join(P4, "runs", "of_*_s0_best.pth"))):
        tag = os.path.basename(ck)[:-len("_best.pth")]
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        name = blob["args"]["model"]
        X = (Xw if name == "mlp" else Xs)
        X = (X - blob["mu"]) / blob["sd"]
        model = BUILDERS[name](X.shape[-1]).to(dev)
        model.load_state_dict(blob["state"])
        reps["model_" + tag] = standardise(embed(model, X.astype(np.float32)))
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    rows_out = []
    for rname, Z in reps.items():
        entry = {"representation": rname, "dim": int(Z.shape[1])}
        # A1: the two splits on the SAME 78 participants
        entry["random_78"] = probe(Z, subj, rec, SEED, keep78, "random")
        entry["recording_78"] = probe(Z, subj, rec, SEED, keep78, "recording")
        # the published framing, for continuity
        entry["random_all91"] = probe(Z, subj, rec, SEED, None, "random")
        # A4b: acquisition-matched subgroup
        if len(keep_fps_multi) >= 10:
            entry["recording_fps_matched"] = probe(
                Z, subj, rec, SEED, keep_fps_multi, "recording")
        rows_out.append(entry)
        print(json.dumps(entry), flush=True)

    res["representations"] = rows_out
    path = os.path.join(OUT, "p6_probe_controls.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
