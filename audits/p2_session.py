"""The last signal source not yet measured: SESSION CONTEXT.

DAiSEE 10s clips are cut from long recording sessions (ID = subj6 + session1 + order3).
Engagement is a slowly changing state -> are the labels of adjacent clips autocorrelated?
If so, smoothing scores over neighbours in the SAME session is a source of gains nobody
has exploited yet (every model, SOTA included, processes each clip independently).

Developed on Val (w5_s0 Train-only, clean held-out); confirmed ONCE on the dumped Test
probabilities (post hoc, no further Test forward pass).
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HOME = os.path.expanduser('~')  # the server account that ran this script

P23 = 0.465 / (0.488 + 0.465)


def load_ids(split):
    ids = [s for s in np.load(f'{HOME}/p2/faces5/{split}_ids.npy', allow_pickle=True)]
    lab = pd.read_csv(f'{HOME}/daisee/DAiSEE/Labels/{split}Labels.csv')
    lab.columns = [c.strip() for c in lab.columns]
    lab['stem'] = lab['ClipID'].astype(str).str.replace(r'\.(avi|mp4)$', '', regex=True)
    keep = set(lab['stem'])
    return [s for s in ids if s in keep]


def analyze(P, y, ids, name):
    sess = np.array([s[:7] for s in ids])
    order = np.array([int(s[7:]) for s in ids])
    print(f'\n===== {name}: n={len(y)}, {len(np.unique(sess))} phien =====')

    # 1. LABEL AUTOCORRELATION between adjacent clips in the same session (2/3 axis and 4 levels)
    agree23, agree4, gap1 = [], [], 0
    for s in np.unique(sess):
        k = np.where(sess == s)[0]
        k = k[np.argsort(order[k])]
        for a, b in zip(k[:-1], k[1:]):
            if order[b] - order[a] <= 2:               # adjacent (one missing clip allowed)
                gap1 += 1
                agree4.append(y[a] == y[b])
                if y[a] >= 2 and y[b] >= 2:
                    agree23.append(y[a] == y[b])
    m = y >= 2
    base23 = max((y[m] == 2).mean(), (y[m] == 3).mean())
    print('cap lien ke: %d | dong nhan 4-muc %.3f | dong nhan 2/3 %.3f (san doan da so %.3f)'
          % (gap1, np.mean(agree4), np.mean(agree23), base23))

    # 2. SMOOTHING probabilities over same-session neighbours: P'_i = (1-w)P_i + w*mean(neighbours)
    t = (y[m] == 3).astype(int)
    pred0 = P.argmax(1)
    print('goc      : acc %.4f | AUC23 %.4f' % ((pred0 == y).mean(),
          roc_auc_score(t, (P[:, 3] / (P[:, 2] + P[:, 3] + 1e-9))[m])))
    for W in (1, 2, 3):
        for w in (0.3, 0.5, 0.7):
            Ps = P.copy()
            for s in np.unique(sess):
                k = np.where(sess == s)[0]
                k = k[np.argsort(order[k])]
                nb = np.zeros((len(k), P.shape[1]))
                cnt = np.zeros(len(k))
                for d in range(1, W + 1):
                    nb[d:] += P[k[:-d]]; cnt[d:] += 1
                    nb[:-d] += P[k[d:]]; cnt[:-d] += 1
                ok = cnt > 0
                Ps[k[ok]] = (1 - w) * P[k[ok]] + w * nb[ok] / cnt[ok, None]
            pred = Ps.argmax(1)
            s23 = (Ps[:, 3] / (Ps[:, 2] + Ps[:, 3] + 1e-9))[m]
            print('W=%d w=%.1f: acc %.4f | AUC23 %.4f' % (W, w, (pred == y).mean(),
                  roc_auc_score(t, s23)))

    # 3. CEILING: oracle "session majority label" (transductive oracle, only to know the limit)
    accs = 0
    for s in np.unique(sess):
        k = sess == s
        vals, cnts = np.unique(y[k], return_counts=True)
        accs += cnts.max()
    print('oracle nhan-da-so-phien: acc %.4f' % (accs / len(y)))


ids_v = load_ids('Validation')
d = np.load(f'{HOME}/p2/probe/w5_s0_val_probs.npz', allow_pickle=True)
analyze(d['P'], d['y'], ids_v, 'VAL (w5_s0 Train-only)')

ids_t = load_ids('Test')
dt = np.load(f'{HOME}/p2/runs/TEST_FINAL_probs.npz', allow_pickle=True)
analyze(dt['ens'], dt['y'], ids_t, 'TEST (ensemble w5m, hau nghiem tren dump)')
