"""Pack the EngageNet test split: MARLIN tensors and labels_test.csv.

Separate from p4_pack_marlin.py --include-test, which would repack train and
validation as well and take twenty minutes to reproduce two files that already
exist. The test split is opened under the pre-registration recorded in
DANG_KY_TRUOC_TEST_ENGAGENET_2026-09-05.md: the dataset authors granted access
to the full corpus including the test labels, which Audit 1 and Audit 2 already
use. What this step does not touch is model performance on that split.

Usage:  python p4_pack_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4_pack_marlin as M  # noqa: E402


def main():
    shape, missing = M.pack("test", M.DEFAULT_OUT)
    print("test packed: X %s, %d tensors missing" % (shape, missing))
    for f in ("marlin_test.npz", "labels_test.csv"):
        p = os.path.join(M.DEFAULT_OUT, f)
        print("  %s  %.1f MB" % (p, os.path.getsize(p) / 1e6))


if __name__ == "__main__":
    main()
