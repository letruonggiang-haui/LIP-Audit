"""Four checks on the E6 and E7 claims before they enter the manuscript.

1. E6 tap point. The 128-d vector entering the final classifier sits one layer
   from the logits and is compressed toward the four engagement classes, so a
   probe read there is not the same measurement as DAiSEE's 512-d embedding.
   The probe is re-read at the pooled representation before the head, and both
   numbers are reported with their dimensions. If the ordering survives the tap,
   it is a fact about the models; if not, it was an artefact of where we looked.

2. E7 within-person AUC. The reported figure is a clip-count weighted mean over
   the participants that carry both classes, and there are eleven of them. The
   spread across participants and the count that qualify say whether one or two
   people carry it.

3. The frame-rate confound. Capture rate identifies the participant, so a
   within-person ranking could be tracking a per-participant scale. The
   within-person AUC is split by frame-rate group.

4. CNN-LSTM returns the same accuracy on all three seeds with best epochs 0, 1
   and 5. Its confusion matrix says whether that is a reproduced system or a
   degenerate one.

Usage:  python p4_e67_checks.py
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p4_train_of import BUILDERS, load_split  # noqa: E402
from p4_e6_probe import logistic_probe, ncm_probe, centre_per_subject  # noqa: E402

P4 = os.path.expanduser("~/p4")
OUT = os.path.join(P4, "analysis")


def pooled_module(model, name):
    """The module whose input is the pooled representation, before the head."""
    if name == "mlp":
        return model.net[4]
    if name == "transformer":
        return model.head[0]
    return model.head[1]          # lstm, cnnlstm, tcn: head[0] is a Dropout


def tap(model, module, X, batch=256):
    got = {}

    def hook(_m, inp, _o):
        got.setdefault("v", []).append(inp[0].detach().cpu().numpy())

    h = module.register_forward_hook(hook)
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(X), batch):
            model(torch.from_numpy(X[i:i + batch]).to(dev))
    h.remove()
    return np.concatenate(got["v"])


# --------------------------------------------------------------------------- #
def check_probe_tap():
    Xw, Xs, ytr, ltr = load_split("train")
    subj = ltr["subj"].values
    n_subj = len(np.unique(subj))
    rows = []
    for ck in sorted(glob.glob(os.path.join(P4, "runs", "of_*_best.pth"))):
        tag = os.path.basename(ck)[:-len("_best.pth")]
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        name = blob["args"]["model"]
        X = ((Xw if name == "mlp" else Xs) - blob["mu"]) / blob["sd"]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = BUILDERS[name](X.shape[-1]).to(dev)
        model.load_state_dict(blob["state"])
        Z = tap(model, pooled_module(model, name), X.astype(np.float32))
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
        acc, _, _ = logistic_probe(Z, subj)
        ncm = ncm_probe(Z, subj)
        accc, _, _ = logistic_probe(centre_per_subject(Z, subj), subj)
        rows.append({"tag": tag, "model": name, "dim": int(Z.shape[1]),
                     "logistic_M": acc * n_subj, "ncm_M": ncm * n_subj,
                     "logistic_M_centred": accc * n_subj})
        print("  %-20s pooled dim %3d | logistic %5.1fx  ncm %5.1fx  centred %4.1fx"
              % (tag, Z.shape[1], acc * n_subj, ncm * n_subj, accc * n_subj), flush=True)
    return rows


# --------------------------------------------------------------------------- #
def per_participant_auc(score, target, subj):
    out = {}
    for s in np.unique(subj):
        k = subj == s
        if len(np.unique(target[k])) == 2:
            out[int(s)] = (float(roc_auc_score(target[k], score[k])), int(k.sum()))
    return out


def check_within_person(fps_by_subject):
    rows = []
    for f in sorted(glob.glob(os.path.join(P4, "probs", "of_*_val_probs.npz"))):
        z = np.load(f, allow_pickle=True)
        P, y, subj = z["P"], z["y"], z["subj"]
        m = y >= 2
        sc = (P[:, 3] / (P[:, 2] + P[:, 3] + 1e-9))[m]
        t = (y[m] == 3).astype(int)
        sj = subj[m]
        per = per_participant_auc(sc, t, sj)
        vals = np.array([v for v, _ in per.values()])
        w = np.array([n for _, n in per.values()])
        hi = {s: v for s, (v, _) in per.items() if fps_by_subject.get(s, 0) >= 25}
        lo = {s: v for s, (v, _) in per.items() if 0 < fps_by_subject.get(s, 0) < 25}
        rows.append({"tag": os.path.basename(f)[:-len("_val_probs.npz")],
                     "n_participants": len(per), "weighted_mean": float(np.average(vals, weights=w)),
                     "unweighted_mean": float(vals.mean()), "min": float(vals.min()),
                     "max": float(vals.max()),
                     "clips_per_participant": [int(x) for x in w],
                     "mean_30fps_group": float(np.mean(list(hi.values()))) if hi else None,
                     "n_30fps": len(hi),
                     "mean_low_fps_group": float(np.mean(list(lo.values()))) if lo else None,
                     "n_low_fps": len(lo)})
        r = rows[-1]
        print("  %-20s n=%2d weighted %.3f unweighted %.3f range [%.3f, %.3f] | "
              "30fps %s (n=%d)  slower %s (n=%d)"
              % (r["tag"], r["n_participants"], r["weighted_mean"], r["unweighted_mean"],
                 r["min"], r["max"],
                 "%.3f" % r["mean_30fps_group"] if r["mean_30fps_group"] else "-",
                 r["n_30fps"],
                 "%.3f" % r["mean_low_fps_group"] if r["mean_low_fps_group"] else "-",
                 r["n_low_fps"]), flush=True)
    return rows


# --------------------------------------------------------------------------- #
def check_zero_input():
    """Where does an all-zero descriptor land? That is what every untracked clip
    presents to the model, so the answer is the placeholder rule the model has
    learned."""
    out = []
    for ck in sorted(glob.glob(os.path.join(P4, "runs", "of_*_best.pth"))):
        tag = os.path.basename(ck)[:-len("_best.pth")]
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        name = blob["args"]["model"]
        d = len(blob["mu"]) if np.ndim(blob["mu"]) == 1 else blob["mu"].shape[-1]
        raw = np.zeros((1, d), dtype=np.float32)
        x = (raw - blob["mu"]) / blob["sd"]
        if name != "mlp":
            x = np.repeat(x[:, None, :], 9, axis=1)
        model = BUILDERS[name](x.shape[-1])
        model.load_state_dict(blob["state"])
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(torch.from_numpy(x.astype(np.float32))), 1).numpy()[0]
        out.append({"tag": tag, "model": name, "probs": [round(float(v), 4) for v in p],
                    "argmax": int(p.argmax())})
        print("  %-20s zero descriptor -> class %d, p = %s"
              % (tag, p.argmax(), np.round(p, 3)), flush=True)
    return out


def check_confusions():
    out = {}
    for f in sorted(glob.glob(os.path.join(P4, "probs", "of_*_val_probs.npz"))):
        z = np.load(f, allow_pickle=True)
        pred, y = z["P"].argmax(1), z["y"]
        cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
        tag = os.path.basename(f)[:-len("_val_probs.npz")]
        out[tag] = cm.tolist()
        used = sorted(set(pred.tolist()))
        print("  %-20s predicts classes %s   rows(true 0..3) = %s"
              % (tag, used, cm.tolist()), flush=True)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    import pandas as pd
    lab = pd.read_csv(os.path.join(P4, "data", "labels_validation.csv"))
    zf = np.load(os.path.join(P4, "feats", "openface98_validation.npz"), allow_pickle=True)
    fr = pd.DataFrame({"chunk": np.asarray(zf["chunk"], dtype=str),
                       "tot": zf["frames_total"]})
    mm = lab.merge(fr, on="chunk")
    fps = (mm[mm.tot < 2000].groupby("subj")["tot"].median() / 10.0).to_dict()
    print("validation participants and their capture rate:",
          {int(k): round(v, 1) for k, v in sorted(fps.items())})

    print("\n=== 1. probe read at the pooled representation ===")
    tap_rows = check_probe_tap()
    print("\n=== 2 and 3. within-person AUC, spread and frame-rate split ===")
    wp_rows = check_within_person({int(k): v for k, v in fps.items()})
    print("\n=== 4a. what an all-zero descriptor predicts ===")
    zero_rows = check_zero_input()
    print("\n=== 4b. confusion matrices ===")
    cms = check_confusions()

    json.dump({"probe_pooled_tap": tap_rows, "within_person_detail": wp_rows,
               "zero_descriptor": zero_rows, "confusions": cms},
              open(os.path.join(OUT, "E67_checks.json"), "w"), indent=2)
    print("\nwrote", os.path.join(OUT, "E67_checks.json"))


if __name__ == "__main__":
    main()
