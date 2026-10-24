"""FMCW radar front-end: range compression, MTI, micro-Doppler spectrograms and the
range-gating / power-referencing variants compared in the paper.

Conventions: `rt` is a complex range-time matrix [n_range, n_slow] after MTI; the PRF is
1/Tsweep (1 kHz for this dataset). All spectrograms are power, shape [n_doppler, n_frames].
"""
from __future__ import annotations

import numpy as np
from scipy import signal

C0 = 299_792_458.0


def range_compress(chirps: np.ndarray, n_keep: int = 48, window: str = "rect") -> np.ndarray:
    """Range FFT per chirp (rectangular window by default, as in the dataset authors' reference
    code; "hann" lowers range sidelobes), keep the positive-beat half, drop the DC bin.
    Returns [n_keep, n_chirps] complex64."""
    nts = chirps.shape[1]
    if window == "hann":
        chirps = (chirps - chirps.mean(1, keepdims=True)) * signal.windows.hann(nts, sym=False)[None, :]
    rp = np.fft.fftshift(np.fft.fft(chirps, axis=1), axes=1)[:, nts // 2:]
    return rp[:, 1:1 + n_keep].T.astype(np.complex64)


def mti(rt: np.ndarray, wn: float = 0.0075, order: int = 4) -> np.ndarray:
    """4th-order Butterworth high-pass along slow time (static-clutter suppression),
    same cut-off as the dataset reference code (0.0075 x Nyquist = 3.75 Hz at 1 kHz PRF)."""
    b, a = signal.butter(order, wn, "high")
    return signal.lfilter(b, a, rt, axis=1).astype(np.complex64)


def stft_bins(rt: np.ndarray, win: int = 200, hop: int = 20, nfft: int = 256) -> np.ndarray:
    """Per-range-bin STFT power. Returns [n_range, nfft, n_frames] with Doppler fftshifted."""
    w = signal.windows.hann(win, sym=False).astype(np.float32)
    n = rt.shape[1]
    starts = np.arange(0, n - win + 1, hop)
    idx = starts[:, None] + np.arange(win)[None, :]
    seg = rt[:, idx] * w  # [R, T, win]
    spec = np.fft.fftshift(np.fft.fft(seg, n=nfft, axis=2), axes=2)
    p = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return np.transpose(p, (0, 2, 1))  # [R, F, T]


def range_bin_m(bw: float) -> float:
    """FMCW range-bin size c / (2B) (one beat-frequency bin per 1/T_chirp)."""
    return C0 / (2.0 * bw)


def select_window(rt: np.ndarray, n_win: int) -> np.ndarray:
    """Keep the n_win slow-time samples with the largest moving (post-MTI) energy.
    Label-agnostic activity localisation; removes the recording-duration shortcut
    (walking files are 10 s, other activities 5 s)."""
    n = rt.shape[1]
    if n <= n_win:
        return rt
    e = (np.abs(rt) ** 2).sum(0)
    c = np.concatenate([[0.0], np.cumsum(e)])
    s = c[n_win:] - c[:-n_win]
    a = int(np.argmax(s))
    return rt[:, a:a + n_win]


def range_gate(P: np.ndarray, mode: str, bin_m: float = 0.375, fixed_m=(3.75, 11.25),
               half_width_m: float = 1.5, smooth_frames: int = 9,
               exclude_bins: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-bin spectrograms P [R, F, T] to one spectrogram [F, T].
    Stored range index i corresponds to range (i + 1) * bin_m (DC bin dropped).

    fixed : sum over a fixed range interval (default 3.75-11.25 m = bins 10-30 of the dataset
            reference script at 400 MHz), converted to bins with the file's own bandwidth.
    all   : sum over all stored bins.
    track : target-adaptive gate. For each frame, the target range bin is the argmax of the
            moving-energy profile (Doppler power excluding the |f| <= exclude_bins MTI notch),
            median-smoothed over time; bins within +-half_width_m of it are summed.
    Returns (spectrogram, per-frame centre bin or -1)."""
    R, F, T = P.shape
    if mode == "fixed":
        lo = max(int(round(fixed_m[0] / bin_m)) - 1, 0)
        hi = min(int(round(fixed_m[1] / bin_m)) - 1, R - 1)
        return P[lo:hi + 1].sum(0), np.full(T, -1)
    if mode == "all":
        return P.sum(0), np.full(T, -1)
    if mode != "track":
        raise ValueError(mode)
    c = F // 2
    moving = P.copy()
    moving[:, c - exclude_bins:c + exclude_bins + 1, :] = 0.0
    e = moving.sum(1)  # [R, T]
    centre = np.argmax(e, axis=0)
    k = smooth_frames if smooth_frames % 2 else smooth_frames + 1
    centre = signal.medfilt(centre.astype(float), k).astype(int)
    half_width = max(int(round(half_width_m / bin_m)), 1)
    r = np.arange(R)[:, None]
    mask = (np.abs(r - centre[None, :]) <= half_width).astype(np.float32)  # [R, T]
    return (P * mask[:, None, :]).sum(0), centre


def crop_doppler(S: np.ndarray, n_keep: int = 128) -> np.ndarray:
    c = S.shape[0] // 2
    return S[c - n_keep // 2:c + n_keep // 2]


def resample_time(S: np.ndarray, n_out: int = 128) -> np.ndarray:
    """Average-pool / linearly interpolate the frame axis to a fixed length."""
    T = S.shape[1]
    edges = np.linspace(0, T, n_out + 1)
    out = np.empty((S.shape[0], n_out), np.float32)
    for i in range(n_out):
        a, b = int(np.floor(edges[i])), int(np.ceil(edges[i + 1]))
        out[:, i] = S[:, a:max(b, a + 1)].mean(1)
    return out


def reference_db(S: np.ndarray, mode: str, dyn_db: float = 40.0, pct: float = 20.0,
                 margin_db: float = 0.0) -> np.ndarray:
    """Convert power spectrogram to a bounded [0, 1] log image.

    max      : dB relative to the image maximum, clipped to [-dyn_db, 0] (standard practice).
    noise    : dB relative to a scalar noise floor (pct-th percentile of all cells) plus margin_db,
               clipped to [0, dyn_db].
    noise_f  : dB relative to a per-Doppler-bin noise floor N(f) (pct-th percentile over time) plus a
               CFAR-style threshold margin, i.e. an ordered-statistic CFAR normalisation along slow
               time; clipped to [0, dyn_db].
    zscore   : dB, then per-image standardisation (no clipping); rescaled by 1/4 for range.
    """
    L = 10.0 * np.log10(S + 1e-12)
    if mode == "max":
        x = (np.clip(L - L.max(), -dyn_db, 0.0) + dyn_db) / dyn_db
    elif mode == "noise":
        x = np.clip(L - np.percentile(L, pct) - margin_db, 0.0, dyn_db) / dyn_db
    elif mode == "noise_f":
        nf = np.percentile(L, pct, axis=1, keepdims=True)
        x = np.clip(L - nf - margin_db, 0.0, dyn_db) / dyn_db
    elif mode == "zscore":
        x = (L - L.mean()) / (L.std() + 1e-6) / 4.0
    else:
        raise ValueError(mode)
    return x.astype(np.float32)


def make_image(P: np.ndarray, gate: str = "fixed", ref: str = "max", n_dop: int = 128,
               n_time: int = 128, **kw) -> np.ndarray:
    """Full front-end from per-bin STFT power to a model input [n_dop, n_time]."""
    gate_kw = {k: kw[k] for k in ("bin_m", "fixed_m", "half_width_m", "smooth_frames", "exclude_bins") if k in kw}
    ref_kw = {k: kw[k] for k in ("dyn_db", "pct", "margin_db") if k in kw}
    S, _ = range_gate(P, gate, **gate_kw)
    S = resample_time(crop_doppler(S, n_dop), n_time)
    return reference_db(S, ref, **ref_kw)
