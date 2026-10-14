"""Regenerate the controlled imbalance-ratio (IR) training subsets from the fold splits.

Usage:
    python make_ir_subsets.py --dataset DiaMOS --out ir_sweep_regenerated

Only the training portion of each fold is subsampled; validation and test
images are never changed. The output lists should be identical to the files
in ir_sweep/.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from ir_dataset import compute_ir_adjusted_indices, ir_label

MINORITY_CLASS_IDX = {"DiaMOS": 1, "RoCoLe": 1}  # Healthy, Red Spider Mite
IR_LEVELS = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(MINORITY_CLASS_IDX))
    parser.add_argument("--out", default="ir_sweep_regenerated")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    rows = list(csv.DictReader(open(os.path.join(here, "splits", f"{args.dataset}_splits.csv"))))

    n = max(int(r["index"]) for r in rows) + 1
    image = [None] * n
    cls = [None] * n
    labels = np.zeros(n, dtype=int)
    train = defaultdict(list)
    for r in rows:
        i = int(r["index"])
        image[i], cls[i], labels[i] = r["image"], r["class"], int(r["label"])
        if r["split"] == "train":
            train[int(r["fold"])].append(i)
    num_classes = int(labels.max()) + 1

    out_dir = os.path.join(args.out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    for fold in sorted(train):
        for ir in IR_LEVELS:
            idx, cfg = compute_ir_adjusted_indices(sorted(train[fold]), labels, num_classes,
                                                   ir, MINORITY_CLASS_IDX[args.dataset], SEED)
            path = os.path.join(out_dir, f"ir_{ir_label(ir)}_fold_{fold}_train.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["index", "image", "class", "label"])
                for i in sorted(idx):
                    w.writerow([i, image[i], cls[i], labels[i]])
            print(f"fold {fold}  IR {ir:>4}: {cfg['train_size_after']} training images "
                  f"({cfg['mode']}, actual IR {cfg['actual_ir']})")


if __name__ == "__main__":
    main()
