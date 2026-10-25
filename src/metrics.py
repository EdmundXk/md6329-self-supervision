"""Healthcare-oriented evaluation metrics."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, accuracy_score)


def expected_calibration_error(probs, y, n_bins=15):
    """ECE with equal-width confidence bins (Guo et al. 2017)."""
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(probs, y, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append(((lo + hi) / 2, acc[m].mean() if m.sum() else np.nan, int(m.sum())))
    return np.array(out)


def evaluate(probs, y, n_classes):
    """AUROC/AUPRC are macro one-vs-rest for multiclass; sensitivity and
    specificity are macro-averaged per-class rates (for binary these are the
    usual recall of the positive / negative class)."""
    pred = probs.argmax(1)
    res = {}
    if n_classes == 2:
        res["auroc"] = roc_auc_score(y, probs[:, 1])
        res["auprc"] = average_precision_score(y, probs[:, 1])
    else:
        present = np.unique(y)
        res["auroc"] = roc_auc_score(y, probs, multi_class="ovr", average="macro",
                                     labels=np.arange(n_classes))
        res["auprc"] = np.mean([average_precision_score((y == c).astype(int), probs[:, c])
                                for c in present])
    cm = confusion_matrix(y, pred, labels=np.arange(n_classes))
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
    res["sensitivity"] = float(np.nanmean(sens))
    res["specificity"] = float(np.nanmean(spec))
    res["balanced_acc"] = balanced_accuracy_score(y, pred)
    res["accuracy"] = accuracy_score(y, pred)
    res["ece"] = expected_calibration_error(probs, y)
    res["per_class_recall"] = sens.tolist()
    return res
