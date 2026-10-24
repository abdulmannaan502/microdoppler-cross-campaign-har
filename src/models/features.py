"""Hand-crafted micro-Doppler features: moments of the Doppler centroid and bandwidth
trajectories (the feature set used by Shah et al., Neural Comput. Appl. 2022, on this radar),
plus envelope percentiles. Computed on the same front-end image as the neural models."""
import numpy as np
from scipy import stats


def _moments(v):
    return [v.mean(), np.sqrt((v ** 2).mean()), np.median(v), stats.skew(v), v.var(), v.std(), stats.kurtosis(v)]


def md_features(img: np.ndarray) -> np.ndarray:
    """img: [F, T] non-negative image (Doppler x time). Returns a 1-D feature vector."""
    w = np.clip(img.astype(np.float64), 0, None) + 1e-9
    f = np.linspace(-1.0, 1.0, w.shape[0])[:, None]
    s = w.sum(0)
    c = (f * w).sum(0) / s
    b = np.sqrt(((f - c) ** 2 * w).sum(0) / s)
    cdf = np.cumsum(w, 0) / s
    up = f[np.argmax(cdf >= 0.95, axis=0), 0]
    lo = f[np.argmax(cdf >= 0.05, axis=0), 0]
    energy_t = s / s.sum()
    return np.nan_to_num(np.array(_moments(c) + _moments(b) + _moments(up) + _moments(lo) +
                                  [stats.entropy(energy_t), energy_t.std()], dtype=np.float32))
