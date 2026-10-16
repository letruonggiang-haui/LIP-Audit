"""P2 Phase 1 — do identity leakage cua model fine-tuned (giao thuc ke thua A3/P1).

1. Trich embedding z [N,512] tu ckpt cho moi clip CO NHAN cua Train+Validation
   (Test khong dung). Cache ra ~/p2/probe/<run>_z.npz.
2. Identity-probe: logistic regression du doan subject ID tu z — split 75/25 clip
   TRONG tung subject (identity chi dinh nghia cho subject da thay). Top-1 acc vs
   chance 1/88. Do tren z tho va z da per-subject centering (doi chieu).
3. Collapse test: linear probe 4 bai toan nhi phan (giao thuc P1), 5 resplit doc lap
   doi tuong (GroupShuffleSplit 75/25, random_state=rep), none vs trans centering.
   Khac A3: dau vao la embedding clip-level 512d (model tu pool) -> linear probe
   thay MLP/BiLSTM; khai bao khi viet.
   ⚠️ CANH BAO NHIEM: voi model fine-tuned, backbone DA HOC nhan cua clip Train
   -> resplit tren pool Train+Val bi thoi phong boi memorization. Chi dung muc nay
   de doi chieu voi A3 (backbone dong bang). So SACH la muc 4.
4. Official-probe: probe hoc tren embedding Train, test tren Validation (doi tuong
   doc lap, clip/nhan chua tung thay) — none vs trans centering. Day la con so
   subject-fair chinh thuc cho model fine-tuned.

Chay: ~/p2/venv/bin/python p2_probe.py --ckpt ~/p2/runs/base_s0/best.pth
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_train import Clips, P2Model  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DAISEE = Path(os.path.expanduser("~/daisee"))
OUT = Path(os.path.expanduser("~/p2/probe"))
TASKS = {"disengagement": ("Engagement", "le1"), "boredom": ("Boredom", "ge2"),
         "confusion": ("Confusion", "ge2"), "frustration": ("Frustration", "ge2")}


def labels_for(split, stems):
    lab = pd.read_csv(DAISEE / "DAiSEE" / "Labels" / f"{split}Labels.csv")
    lab.columns = [c.strip() for c in lab.columns]
    lab["stem"] = lab["ClipID"].astype(str).str.replace(r"\.(avi|mp4)$", "", regex=True)
    lab = lab.set_index("stem").loc[list(stems)]
    return lab[["Boredom", "Engagement", "Confusion", "Frustration"]].to_numpy()


@torch.no_grad()
def extract(ckpt_path, cache):
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["z"], d["y"], d["subj"]
    dev = "cuda"
    model = P2Model(n_subj=2)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("subj_head.")}
    model.load_state_dict(sd, strict=False)
    model = model.to(dev).eval()
    zs, ys, gs = [], [], []
    for split in ("Train", "Validation"):
        ds = Clips(split, train_aug=False)
        dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=8, pin_memory=True)
        stems = [s for _, s in ds.rows]
        y = labels_for(split, stems)
        zb = []
        for x, _, _ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, z = model(x.to(dev, non_blocking=True))
            zb.append(z.float().cpu().numpy())
        zs.append(np.concatenate(zb))
        ys.append(y)
        gs.append(np.array([s[:6] for s in stems]))
    z, y, subj = np.concatenate(zs), np.concatenate(ys), np.concatenate(gs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, z=z, y=y, subj=subj)
    return z, y, subj


def center(z, subj):
    z = z.copy()
    for s in np.unique(subj):
        m = subj == s
        z[m] -= z[m].mean(axis=0, keepdims=True)
    return z


def identity_probe(z, subj, seed=0):
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        if len(idx) < 4:
            continue
        idx = rng.permutation(idx)
        k = max(1, int(round(len(idx) * 0.25)))
        te_idx.extend(idx[:k])
        tr_idx.extend(idx[k:])
    sc = StandardScaler().fit(z[tr_idx])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(sc.transform(z[tr_idx]), subj[tr_idx])
    acc = float((clf.predict(sc.transform(z[te_idx])) == subj[te_idx]).mean())
    return acc, len(np.unique(subj))


def make_y(y, task):
    col, rule = TASKS[task]
    v = y[:, ["Boredom", "Engagement", "Confusion", "Frustration"].index(col)]
    return (v <= 1).astype(int) if rule == "le1" else (v >= 2).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    ck = Path(os.path.expanduser(args.ckpt))
    run = f"{ck.parent.name}_{ck.stem}"  # vd base_s0_last — tranh va cham cache giua best/last
    z0, y, subj = extract(os.path.expanduser(args.ckpt), OUT / f"{run}_z.npz")
    print(f"pool {z0.shape}, {len(np.unique(subj))} doi tuong")

    for mode, z in (("none", z0), ("trans", center(z0, subj))):
        acc, n = identity_probe(z, subj)
        print(f"identity-probe[{mode:5s}]: top-1 acc {acc:.3f} (chance {1/n:.4f}, {n} lop)",
              flush=True)

    rows = []
    for mode, z in (("none", z0), ("trans", center(z0, subj))):
        for task in TASKS:
            yt = make_y(y, task)
            aucs = []
            for rep in range(5):
                tr, te = next(GroupShuffleSplit(1, test_size=0.25,
                                                random_state=rep).split(z, yt, subj))
                sc = StandardScaler().fit(z[tr])
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                clf.fit(sc.transform(z[tr]), yt[tr])
                p = clf.predict_proba(sc.transform(z[te]))[:, 1]
                aucs.append(float(roc_auc_score(yt[te], p)))
                rows.append((run, task, mode, rep, aucs[-1]))
            print(f"[{mode:5s}|{task:13s}] AUC {np.mean(aucs):.3f}+-{np.std(aucs):.3f}",
                  flush=True)

    # official-probe: hoc tren Train, test tren Validation (doi tuong doc lap)
    val_lab = pd.read_csv(DAISEE / "DAiSEE" / "Labels" / "ValidationLabels.csv")
    val_lab.columns = [c.strip() for c in val_lab.columns]
    val_subjects = set(val_lab["ClipID"].astype(str).str[:6])
    is_val = np.array([s in val_subjects for s in subj])
    print(f"official-probe: train {int((~is_val).sum())} clip / test {int(is_val.sum())} clip")
    for mode, z in (("none", z0), ("trans", center(z0, subj))):
        for task in TASKS:
            yt = make_y(y, task)
            sc = StandardScaler().fit(z[~is_val])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(z[~is_val]), yt[~is_val])
            p = clf.predict_proba(sc.transform(z[is_val]))[:, 1]
            auc = float(roc_auc_score(yt[is_val], p))
            rows.append((run, task, f"official-{mode}", -1, auc))
            print(f"[official-{mode:5s}|{task:13s}] AUC {auc:.3f}", flush=True)

    out_csv = OUT / f"{run}_probe.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "task", "norm", "resplit", "auc"])
        w.writerows(rows)
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
