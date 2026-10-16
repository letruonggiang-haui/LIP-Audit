"""Attach the released labels to the deposited prediction files.

The prediction files in this repository carry identifiers and class
probabilities only. The labels are not redistributed: they are read here from
the label files that ship with each corpus and joined on the identifier.

EngageNet  the *_probs.npz files carry `chunk` (the released chunk file name).
           Labels come from <root>/validation_engagement_labels.xlsx and
           <root>/test_engagement_labels.csv, where <root> is the `root`
           argument, else the ENGAGENET_ROOT environment variable, else
           p4_labels.ENGAGENET_ROOT. SNP chunks carry no label; no stored
           prediction file contains one. If a split's label file is absent the
           file is returned without `y`, and the calling script skips that split.
DAiSEE     logs/TEST_FINAL_probs.npz, logs/repro_test_probs_*.npz and
           logs/val_probs/*_val_probs.npz carry their clip identifiers in `ids`
           (for TEST_FINAL_probs.npz the order is also that of logs/Test_ids.npy
           restricted to the clips labeled in TestLabels.csv).
"""
import os

import numpy as np

import p4_labels as L


class Stored(dict):
    """A loaded npz as a dict, with the `.files` attribute scripts test for."""

    @property
    def files(self):
        return list(self.keys())


def engagenet_root(root=None):
    return root or os.environ.get("ENGAGENET_ROOT") or L.ENGAGENET_ROOT


_EN = {}


def engagenet_labels(split, root=None):
    """chunk -> level (0-3) for one EngageNet split, or None if its file is absent."""
    base = engagenet_root(root)
    key = (split, base)
    if key not in _EN:
        fn = "test_engagement_labels.csv" if split == "test" else split + "_engagement_labels.xlsx"
        if not os.path.isfile(os.path.join(base, fn)):
            _EN[key] = None
        else:
            df = L.load_engagenet(split, base)
            df = df[df["y"].notna()]
            _EN[key] = dict(zip(df["chunk"], df["y"].astype(int)))
    return _EN[key]


def load_probs(path, root=None):
    """Load an EngageNet *_probs.npz and add `y` from the released label file."""
    z = Stored(np.load(path, allow_pickle=True))
    if "y" not in z:
        split = "validation" if "_val_probs" in os.path.basename(path) else "test"
        lab = engagenet_labels(split, root)
        if lab is not None:
            z["y"] = np.array([lab[str(c)] for c in z["chunk"]], dtype=np.int64)
    return z


def daisee_test_labels(labels_dir):
    """stem -> Engagement level for the DAiSEE test split."""
    df = L.load_daisee("Test", labels_dir)
    return dict(zip(df["stem"], df["y"].astype(int)))


def daisee_sealed_test(logs_dir, labels_dir):
    """TEST_FINAL_probs.npz with `y` and `stems` attached from TestLabels.csv."""
    z = Stored(np.load(os.path.join(logs_dir, "TEST_FINAL_probs.npz"), allow_pickle=True))
    eng = daisee_test_labels(labels_dir)
    if "ids" in z:
        stems = [str(s) for s in z["ids"]]
    else:
        ids = [str(s) for s in np.load(os.path.join(logs_dir, "Test_ids.npy"), allow_pickle=True)]
        stems = [s for s in ids if s in eng]
    y = np.array([eng[s] for s in stems], dtype=np.int64)
    assert len(y) == len(z["ens"]), (len(y), len(z["ens"]))
    z["y"], z["stems"] = y, np.array(stems)
    return z


def daisee_probs(path, labels_dir, split):
    """A DAiSEE prediction file keyed by `ids`, with `y` joined from <split>Labels.csv."""
    z = Stored(np.load(path, allow_pickle=True))
    df = L.load_daisee(split, labels_dir)
    eng = dict(zip(df["stem"], df["y"].astype(int)))
    z["y"] = np.array([eng[str(s)] for s in z["ids"]], dtype=np.int64)
    return z
