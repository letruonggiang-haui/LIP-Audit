"""Phan tach tin hieu 2-vs-3: TRONG-doi-tuong vs GIUA-doi-tuong, va do xang cua
transductive per-subject calibration.

KHONG cham Test them lan nao: dung xac suat da dump (TEST_FINAL_probs.npz) +
w5_s0_val_probs.npz (model Train-only -> Val la held-out sach, subject-independent).
Day la buoc PHAT TRIEN tren Val, XAC NHAN mot lan tren file Test da luu.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HOME = os.path.expanduser('~')  # the server account that ran this script

P23_TRAIN = 0.465 / (0.488 + 0.465)      # P(y=3 | y>=2) theo prior Train


def load_ids(split):
    ids = [s for s in np.load(f'{HOME}/p2/faces5/{split}_ids.npy', allow_pickle=True)]
    lab = pd.read_csv(f'{HOME}/daisee/DAiSEE/Labels/{split}Labels.csv')
    lab.columns = [c.strip() for c in lab.columns]
    lab['stem'] = lab['ClipID'].astype(str).str.replace(r'\.(avi|mp4)$', '', regex=True)
    keep = set(lab['stem'])
    return [s for s in ids if s in keep]


def analyze(P, y, ids, name):
    subj = np.array([s[:6] for s in ids])
    m = y >= 2
    s23 = P[:, 3] / (P[:, 2] + P[:, 3] + 1e-9)
    t = (y[m] == 3).astype(int)
    sc = s23[m]
    sj = subj[m]
    us = np.unique(sj)
    print(f'\n===== {name}: n23={m.sum()}, {len(us)} doi tuong =====')
    print('AUC23 raw (tron):            %.4f' % roc_auc_score(t, sc))
    aucs, ws = [], []
    for s in us:
        k = sj == s
        if len(np.unique(t[k])) == 2:
            aucs.append(roc_auc_score(t[k], sc[k]))
            ws.append(k.sum())
    print('AUC23 TRONG-doi-tuong (TB):  %.4f  (n_subj=%d)' % (np.average(aucs, weights=ws), len(aucs)))
    fr = np.array([t[sj == s].mean() for s in us])
    ms = np.array([sc[sj == s].mean() for s in us])
    print('GIUA-doi-tuong: Spearman(TB diem, ti le 3) = %.3f' % spearmanr(ms, fr).statistic)

    pred = P.argmax(1)
    print('acc(2/3) argmax goc:         %.4f' % (pred[m] == y[m]).mean())
    # A: nguong toan cuc theo prior Train (khong transductive) — doi chung
    th = np.quantile(sc, 1 - P23_TRAIN)
    print('acc(2/3) nguong toan cuc:    %.4f' % ((np.where(sc >= th, 3, 2)) == y[m]).mean())
    # B: per-subject centering (shrinkage lam) + nguong toan cuc
    for lam in (0.5, 1.0):
        scc = sc.copy()
        for s in us:
            k = sj == s
            scc[k] = sc[k] - lam * (sc[k].mean() - sc.mean())
        th = np.quantile(scc, 1 - P23_TRAIN)
        print('acc(2/3) centering lam=%.1f:  %.4f' % (lam, ((np.where(scc >= th, 3, 2)) == y[m]).mean()))
    # C: per-subject quantile matching (moi doi tuong nhan dung prior toan cuc)
    p23 = np.empty(len(sc), int)
    for s in us:
        k = sj == s
        ths = np.quantile(sc[k], 1 - P23_TRAIN)
        p23[k] = np.where(sc[k] >= ths, 3, 2)
    print('acc(2/3) per-subj quantile:  %.4f' % (p23 == y[m]).mean())
    # D: ORACLE nguong toi uu tung doi tuong — TRAN cua moi phep transductive
    accо = []
    for s in us:
        k = sj == s
        best = max(((sc[k] >= c).astype(int) + 2 == y[m][k]).mean() for c in
                   np.unique(np.concatenate([sc[k], [sc[k].max() + 1]])))
        accо.append(best * k.sum())
    print('acc(2/3) ORACLE per-subj th: %.4f  <- tran transductive' % (np.sum(accо) / len(sc)))
    # E: ORACLE prior dung + nguong quantile per-subject
    p23o = np.empty(len(sc), int)
    for s in us:
        k = sj == s
        tho = np.quantile(sc[k], 1 - max(t[sj == s].mean(), 1e-9))
        p23o[k] = np.where(sc[k] >= tho, 3, 2)
    print('acc(2/3) oracle-prior quant: %.4f' % (p23o == y[m]).mean())


ids_v = load_ids('Validation')
d = np.load(f'{HOME}/p2/probe/w5_s0_val_probs.npz', allow_pickle=True)
analyze(d['P'], d['y'], ids_v, 'VAL (w5_s0 Train-only — held-out sach, giuong phat trien)')

ids_t = load_ids('Test')
dt = np.load(f'{HOME}/p2/runs/TEST_FINAL_probs.npz', allow_pickle=True)
assert len(ids_t) == len(dt['y']), (len(ids_t), len(dt['y']))
analyze(dt['ens'], dt['y'], ids_t, 'TEST (ensemble w5m — phan tich hau nghiem tren file da dump)')
