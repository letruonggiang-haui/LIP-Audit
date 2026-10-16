"""E4 - TCCT-Net released weights on the EngageNet validation partition, with
per-clip probabilities kept so the within-person decomposition can be read.

Runs from inside the TCCT_Net checkout (its modules are imported relative to
the working directory) after p4_tcct_build.py has written inference_files/ and
labels_inference.csv. Everything model-side is the authors' own code and
weights, unchanged: Decision_Fusion, get_source_data_inference, batch_cwt,
config.json. What is added is the softmax, the join back to chunk and
participant ids, UAR / macro-F1 / kappa_w / MAE beside accuracy, and a
participant-clustered bootstrap.

Usage:  cd ~/TCCT_Net && python ~/p4/code/p4_e4_tcct.py Validation
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, recall_score

sys.path.insert(0, os.getcwd())
from models.feature_fusion import Decision_Fusion  # noqa: E402
from data.data_loader import get_source_data_inference  # noqa: E402
from data.data_processing import batch_cwt  # noqa: E402

P4 = os.path.expanduser("~/p4")


def metrics(pred, y):
    return {"acc": float((pred == y).mean()),
            "uar": float(recall_score(y, pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
            "mae": float(np.abs(pred - y).mean())}


def bootstrap(pred, y, clusters, n_boot=2000, seed=20260907):
    rng = np.random.default_rng(seed)
    uc = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uc}
    draws = defaultdict(list)
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[c] for c in rng.choice(uc, len(uc), replace=True)])
        for k, v in metrics(pred[idx], y[idx]).items():
            draws[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in draws.items()}


def main(split):
    cfg = json.load(open("config.json"))
    torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(Decision_Fusion(cfg["n_classes"])).to(dev)
    model.load_state_dict(torch.load(cfg["final_model_weights"], map_location=dev))
    model.eval()

    X, y = get_source_data_inference(cfg["inference_folder_path"], cfg["label_file_inference"],
                                     cfg["behavioral_features"], cfg["target_mean"],
                                     cfg["target_std"])
    # the loader walks os.listdir, so recover the file order it used
    ids = [f[:-4] for f in os.listdir(cfg["inference_folder_path"]) if f.endswith(".csv")]
    man = {m["id"]: m for m in json.load(open(os.path.join(
        P4, "analysis", "E4_tcct_manifest_%s.json" % split)))}
    assert len(ids) == len(y) == len(man), (len(ids), len(y), len(man))

    Xt = torch.from_numpy(X).float().to(dev)
    freqs = np.linspace(cfg["freq_min"], cfg["freq_max"], cfg["tensor_height"])
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xt), 64):
            xb = Xt[i:i + 64]
            cwt = batch_cwt(xb, freqs, sampling_frequency=cfg["sampling_frequency"]).to(dev)
            probs.append(torch.softmax(model(xb, cwt), 1).cpu().numpy())
    P = np.concatenate(probs)
    y = np.asarray(y).astype(int)
    pred = P.argmax(1)
    subj = np.array([man[i]["subj"] for i in ids])
    chunk = np.array([man[i]["chunk"] for i in ids])

    os.makedirs(os.path.join(P4, "probs"), exist_ok=True)
    np.savez_compressed(os.path.join(P4, "probs", "tcct_%s_probs.npz" % split.lower()),
                        P=P, y=y, chunk=chunk, subj=subj)
    out = {"split": split, "n": int(len(y)), "n_subjects": int(len(np.unique(subj))),
           "published_val_acc": 68.91, "point": metrics(pred, y),
           "ci_subject": bootstrap(pred, y, subj),
           "confusion": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
           "floor": float(np.bincount(y, minlength=4).max() / len(y)),
           "note": ("authors' code, weights and config unchanged; inputs rebuilt from our "
                    "OpenFace run, 1000 Hz-timebase clips resampled to 30 Hz before the "
                    "280-row cut (see E4_tcct_manifest)")}
    json.dump(out, open(os.path.join(P4, "analysis", "E4_tcct_%s.json" % split.lower()), "w"),
              indent=2)
    p = out["point"]
    print("TCCT-Net on %s: acc %.4f (published 68.91)  CI [%.4f, %.4f]  UAR %.3f  F1 %.3f  "
          "QWK %.3f  MAE %.3f  floor %.3f" % (split, p["acc"], out["ci_subject"]["acc"][0],
                                             out["ci_subject"]["acc"][1], p["uar"],
                                             p["macro_f1"], p["qwk"], p["mae"], out["floor"]))
    print("confusion (rows true 0..3):", out["confusion"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Validation")
