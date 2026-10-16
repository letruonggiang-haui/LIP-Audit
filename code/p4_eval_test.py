"""The single, pre-registered evaluation on the EngageNet test partition.

Registered in DANG_KY_TRUOC_TEST_ENGAGENET_2026-09-05.md before any test
descriptor existed. What that registration fixes, and what this script therefore
refuses to let drift:

  - the configurations are the eighteen checkpoints already trained on the
    official training partition, unchanged, at the epoch already selected on
    validation and frozen there. Nothing is retrained here and no hyper-parameter
    is touched;
  - the metrics are accuracy, UAR, macro-F1, quadratic-weighted kappa and MAE,
    beside the test majority-class floor and a participant-clustered bootstrap,
    B = 2,000;
  - the comparison is against the only published test column, the dataset
    paper's own: LSTM 62.37, CNN-LSTM 65.16, TCN 65.60, Transformer 67.61,
    MARLIN 65.2.

Section 2(d) registers a second, separate arm: the same five families retrained
on train and validation combined, so that the dataset paper's own Train+Val row
has something to be compared with. That is a different set of checkpoints and a
different single evaluation, so it is a different stage here and writes its own
result file.

Each stage runs once. A results file already on disk is not overwritten without
--i-am-rerunning-and-will-say-so, which exists so that a second run has to be a
deliberate act that leaves a trace in the shell history rather than an
absent-minded one.

Usage:  python p4_eval_test.py --stage frozen
        python p4_eval_test.py --stage trainval
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p4_train_of import BUILDERS, load_split  # noqa: E402
from p4_train_marlin import SeqTransformer  # noqa: E402
from p4_train_marlin import load_split as load_marlin  # noqa: E402

P4 = os.path.expanduser("~/p4")
OUT = os.path.join(P4, "analysis")
STAGES = {"frozen": ("of_*_best.pth", "E10_test.json", True),
          "trainval": ("tv_*_best.pth", "E10_test_trainval.json", False)}

# The dataset paper's test column. The Train+Val figure it publishes, 69.29, is
# for feature fusion and not for any single stream, so no per-family Train+Val
# number exists to compare against and the trainval stage carries none.
PUBLISHED = {"of_lstm": 62.37, "of_cnnlstm": 65.16, "of_tcn": 65.60,
             "of_transformer": 67.61, "marlin": 65.2, "of_mlp": None}


def metrics(pred, y):
    return {"acc": float((pred == y).mean()),
            "uar": float(recall_score(y, pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
            "mae": float(np.abs(pred - y).mean())}


def bootstrap(pred, y, clusters, n_boot=2000, seed=20260905):
    rng = np.random.default_rng(seed)
    uc = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uc}
    draws = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.choice(uc, len(uc), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        m = metrics(pred[idx], y[idx])
        for k, v in m.items():
            draws[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in draws.items()}


def infer(model, X, batch=256):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logit = model(torch.from_numpy(X[i:i + batch]).to(dev))
            out.append(torch.softmax(logit, 1).cpu().numpy())
    return np.concatenate(out)


def run_openface(dev, rows, pattern, prefix):
    Xw, Xs, y, lab = load_split("test")
    print("test descriptors: %d chunks, %d participants, floor %.4f"
          % (len(y), lab["subj"].nunique(), np.bincount(y, minlength=4).max() / len(y)))
    found = sorted(glob.glob(os.path.join(P4, "runs", pattern)))
    if not found:
        raise SystemExit("no checkpoint matches %s" % pattern)
    for ck in found:
        tag = os.path.basename(ck)[:-len("_best.pth")]
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        name = blob["args"]["model"]
        X = ((Xw if name == "mlp" else Xs) - blob["mu"]) / blob["sd"]
        model = BUILDERS[name](X.shape[-1]).to(dev)
        model.load_state_dict(blob["state"])
        P = infer(model, X.astype(np.float32))
        rows.append(record(tag, prefix + name, P, y, lab))


def run_marlin(dev, rows):
    X0, y, lab = load_marlin("test")
    for ck in sorted(glob.glob(os.path.join(P4, "runs", "marlin_*_best.pth"))):
        tag = os.path.basename(ck)[:-len("_best.pth")]
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        X = (X0 - blob["mu"]) / blob["sd"]
        model = SeqTransformer(d_in=X.shape[-1]).to(dev)
        model.load_state_dict(blob["state"])
        P = infer(model, X.astype(np.float32))
        rows.append(record(tag, "marlin", P, y, lab))


def record(tag, system, P, y, lab):
    pred = P.argmax(1)
    np.savez_compressed(
        os.path.join(P4, "probs", "%s_test_probs.npz" % tag),
        P=P, y=y, chunk=lab["chunk"].values.astype(str),
        subj=lab["subj"].values, rec=(lab["sess"].astype(str) + "_"
                                      + lab["vid"].astype(str)).values,
        t=lab["t"].values)
    row = {"tag": tag, "system": system, "n": int(len(y)),
           "point": metrics(pred, y),
           "ci_subject": bootstrap(pred, y, lab["subj"].values),
           "confusion": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
           "n_subjects": int(lab["subj"].nunique())}
    p = row["point"]
    ci = row["ci_subject"]["acc"]
    print("%-20s acc %.4f  CI [%.4f, %.4f]  UAR %.3f  F1 %.3f  QWK %.3f  MAE %.3f"
          % (tag, p["acc"], ci[0], ci[1], p["uar"], p["macro_f1"], p["qwk"], p["mae"]),
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--i-am-rerunning-and-will-say-so", action="store_true",
                    dest="rerun")
    args = ap.parse_args()
    pattern, result_name, with_marlin = STAGES[args.stage]
    result = os.path.join(OUT, result_name)
    if os.path.exists(result) and not args.rerun:
        sys.exit("%s already exists. Each test evaluation is registered as a single "
                 "run; every later analysis is post hoc on the stored per-clip "
                 "probabilities in ~/p4/probs/*_test_probs.npz." % result)

    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    run_openface(dev, rows, pattern, "of_" if args.stage == "frozen" else "tv_")
    if with_marlin:
        run_marlin(dev, rows)

    lab = pd.read_csv(os.path.join(P4, "data", "labels_test.csv"))
    lab = lab[lab["y"].notna()]
    floor = float(lab["y"].value_counts().max() / len(lab))

    by = defaultdict(list)
    for r in rows:
        by[r["system"]].append(r)
    summary = {}
    for s, rs in sorted(by.items()):
        accs = [r["point"]["acc"] for r in rs]
        summary[s] = {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                      "published": PUBLISHED.get(s),
                      "gap_points": (float(np.mean(accs) * 100 - PUBLISHED[s])
                                     if PUBLISHED.get(s) else None),
                      "uar": float(np.mean([r["point"]["uar"] for r in rs])),
                      "macro_f1": float(np.mean([r["point"]["macro_f1"] for r in rs])),
                      "qwk": float(np.mean([r["point"]["qwk"] for r in rs])),
                      "mae": float(np.mean([r["point"]["mae"] for r in rs]))}

    json.dump({"registered": "DANG_KY_TRUOC_TEST_ENGAGENET_2026-09-05.md",
               "stage": args.stage, "floor": floor, "runs": rows,
               "summary": summary}, open(result, "w"), indent=2)

    print("\nmajority-class floor on test: %.4f" % floor)
    print("%-16s %-16s %-10s %-8s %-7s %-7s %s"
          % ("system", "acc (3 seeds)", "published", "gap", "UAR", "QWK", "MAE"))
    for s, v in sorted(summary.items()):
        print("%-16s %.4f +- %.4f  %-10s %-8s %-7.3f %-7.3f %.3f"
              % (s, v["acc_mean"], v["acc_sd"],
                 "%.2f" % v["published"] if v["published"] else "-",
                 "%+.1f" % v["gap_points"] if v["gap_points"] is not None else "-",
                 v["uar"], v["qwk"], v["mae"]))
    print("\nwrote", result)


if __name__ == "__main__":
    main()
