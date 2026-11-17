"""Adjudicate the ResNet-TCN reproduction by the PRE-REGISTERED gates (NHAT_KY_P2.md 20/08):
(1) pass <=> peak Test acc within [0.619, 0.659]
(2) within-person AUC23 at best-acc < 0.65 => defuse DA-C2; >= 0.65 => DA-C2 STANDS
(3) record both the last-epoch acc and the peak (evidence of test-set selection)

AUC definition identical to audits/p2_decompose.py: s23 = P3/(P2+P3),
within-person = mean weighted by each subject's clip count (only subjects with both classes).
Also: WITHIN-subject label permutation B=1000 for the within-AUC (p-value).
Runs on CPU, does not touch the GPU: ~/p2/venv/bin/python ~/p2/repro/repro_audit2.py
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

OUT = os.path.expanduser('~/p2/repro')
rng = np.random.default_rng(0)

# (1)+(3) acc curve
recs = [json.loads(l) for l in open(os.path.join(OUT, 'repro_log.jsonl'))]
accs = np.array([r['test_acc'] for r in recs])
best_ep = int(accs.argmax())
print(f'epochs={len(recs)} PEAK acc={accs.max():.4f} @ep{best_ep}  FINAL acc={accs[-1]:.4f} @ep{recs[-1]["ep"]}')
print(f'GATE1 (peak in [.619,.659]): {"DAT" if 0.619 <= accs.max() <= 0.659 else "KHONG DAT"}')
print(f'GATE3 peak-to-final drop = {accs[-1]-accs.max():+.4f}')

ids = [os.path.basename(p) for p in pd.read_csv(os.path.join(OUT, 'test_repro.csv'))['path']]
subj_all = np.array([s[:6] for s in ids])

def analyze(tag):
    d = np.load(os.path.join(OUT, f'repro_test_probs_{tag}.npz'))
    P, y = d['P'], d['y']
    assert len(y) == len(ids), (len(y), len(ids))
    print(f'\n===== {tag} (ep{int(d["ep"])}) acc={(P.argmax(1)==y).mean():.4f} =====')
    m = y >= 2
    s23 = P[:, 3] / (P[:, 2] + P[:, 3] + 1e-9)
    t, sc, sj = (y[m] == 3).astype(int), s23[m], subj_all[m]
    us = np.unique(sj)
    print('AUC23 raw (tron):           %.4f' % roc_auc_score(t, sc))
    def within(tv):
        aucs, ws = [], []
        for s in us:
            k = sj == s
            if len(np.unique(tv[k])) == 2:
                aucs.append(roc_auc_score(tv[k], sc[k])); ws.append(k.sum())
        return np.average(aucs, weights=ws)
    w_obs = within(t)
    print('AUC23 TRONG-doi-tuong (TB): %.4f' % w_obs)
    # within-person label permutation
    perm = []
    for _ in range(1000):
        tp = t.copy()
        for s in us:
            k = np.where(sj == s)[0]
            tp[k] = tp[rng.permutation(k)]
        perm.append(within(tp))
    perm = np.array(perm)
    print('  hoan vi trong-nguoi: mean=%.4f  p(perm>=obs)=%.4f' % (perm.mean(), (perm >= w_obs).mean()))
    fr = np.array([t[sj == s].mean() for s in us])
    ms = np.array([sc[sj == s].mean() for s in us])
    print('GIUA-doi-tuong Spearman:    %.3f' % spearmanr(ms, fr).statistic)
    if tag == 'best':
        print(f'GATE2 (within<0.65 tai best-acc): {"DEFUSE DA-C2" if w_obs < 0.65 else "DA-C2 DUNG"}')

analyze('best')
analyze('last')
