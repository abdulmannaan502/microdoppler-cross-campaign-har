import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def ece(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error (top-label, equal-width bins)."""
    conf = prob.max(1)
    pred = prob.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def nll(prob, y):
    return float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, None))))


def summarize(prob, y, n_classes):
    pred = prob.argmax(1)
    return dict(acc=accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0),
                ece=ece(prob, y), nll=nll(prob, y),
                cm=confusion_matrix(y, pred, labels=list(range(n_classes))).tolist())
