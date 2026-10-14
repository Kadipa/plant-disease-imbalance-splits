# ir_dataset.py
"""
IR-controlled training index selection for the controlled imbalance sweep.

Two modes:
  1. target_ir <= natural_ir  →  subsample each majority class down to
     minority_count * target_ir samples (majority gets rarer, minority fixed).
  2. target_ir >  natural_ir  →  subsample the minority class down to
     max_majority_count / target_ir samples (minority gets even rarer).

Val/test indices are never touched — call this only on train_idx.
"""
import numpy as np


def compute_ir_adjusted_indices(train_idx, labels, num_classes,
                                 target_ir, minority_class_idx, seed):
    """
    Returns (new_train_idx: list, ir_config: dict).

    Parameters
    ----------
    train_idx         : array-like of int — original training sample indices
    labels            : np.ndarray of int — class label for every sample
                        in the FULL dataset (indexed by dataset index, not fold index)
    num_classes       : int
    target_ir         : float — desired imbalance ratio (majority / minority)
    minority_class_idx: int   — class index of the rarest class
    seed              : int   — RNG seed for reproducibility
    """
    rng = np.random.RandomState(seed)
    train_idx = np.array(train_idx)
    labels_train = labels[train_idx]
    counts = np.bincount(labels_train, minlength=num_classes)

    minority_count = int(counts[minority_class_idx])
    majority_counts = {i: int(counts[i]) for i in range(num_classes) if i != minority_class_idx}
    max_majority = max(majority_counts.values()) if majority_counts else 1
    natural_ir = max_majority / minority_count if minority_count > 0 else float("inf")

    new_idx_parts = []
    actual_counts = {}

    if target_ir <= natural_ir:
        # Mode 1: cap each majority class individually
        cap_majority = max(1, int(round(minority_count * target_ir)))
        for cls in range(num_classes):
            cls_mask = labels_train == cls
            cls_idx = train_idx[cls_mask]
            if cls == minority_class_idx:
                selected = cls_idx          # minority unchanged
            elif len(cls_idx) > cap_majority:
                selected = rng.choice(cls_idx, size=cap_majority, replace=False)
            else:
                selected = cls_idx          # already at or below cap
            new_idx_parts.append(selected)
            actual_counts[cls] = len(selected)
    else:
        # Mode 2: cap minority to achieve higher IR
        cap_minority = max(1, int(round(max_majority / target_ir)))
        for cls in range(num_classes):
            cls_mask = labels_train == cls
            cls_idx = train_idx[cls_mask]
            if cls == minority_class_idx and len(cls_idx) > cap_minority:
                selected = rng.choice(cls_idx, size=cap_minority, replace=False)
            else:
                selected = cls_idx
            new_idx_parts.append(selected)
            actual_counts[cls] = len(selected)

    new_train_idx = np.concatenate(new_idx_parts).tolist()

    actual_max = max(actual_counts[i] for i in range(num_classes) if i != minority_class_idx)
    actual_min = actual_counts[minority_class_idx]
    actual_ir = round(actual_max / actual_min, 2) if actual_min > 0 else float("inf")

    ir_config = {
        "target_ir":         target_ir,
        "actual_ir":         actual_ir,
        "natural_ir":        round(natural_ir, 2),
        "mode":              "subsample_majority" if target_ir <= natural_ir else "subsample_minority",
        "minority_class_idx": minority_class_idx,
        "counts_before":     {str(i): int(counts[i]) for i in range(num_classes)},
        "counts_after":      {str(i): int(actual_counts[i]) for i in range(num_classes)},
        "train_size_before": len(train_idx),
        "train_size_after":  len(new_train_idx),
        "seed":              seed,
    }
    return new_train_idx, ir_config


def ir_label(target_ir):
    """Convert float IR to a filesystem-safe string: 1.0 → '1_0', 4.7 → '4_7'."""
    return str(target_ir).replace(".", "_")
