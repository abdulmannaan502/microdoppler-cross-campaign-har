import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.gla848 import parse_name, read_dat  # noqa: E402
from src.radar_processing.frontend import (range_compress, mti, stft_bins, range_gate,  # noqa: E402
                                           make_image, reference_db)

PRF, NTS, TS = 1000.0, 128, 1e-3


def synth(n_chirps=3000, beat_bin=20, f_d=100.0, snr_db=10.0, seed=0):
    """Point target at a known beat frequency (range bin) with Doppler f_d, plus white noise."""
    rng = np.random.default_rng(seed)
    fast = np.arange(NTS)
    slow = np.arange(n_chirps)[:, None]
    s = np.exp(2j * np.pi * (beat_bin / NTS * fast[None, :] + f_d / PRF * slow))
    n = (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape)) / np.sqrt(2) * 10 ** (-snr_db / 20)
    return (s + n).astype(np.complex64)


def test_parse_name():
    assert parse_name("1P36A01R01.dat") == (36, 1, 1)
    assert parse_name("4P08A04R3.dat") == (8, 4, 3)
    assert parse_name("2P01A05R1.dat") is None  # inconsistent activity code


def test_range_and_doppler_location():
    rt = mti(range_compress(synth(beat_bin=20, f_d=100.0)))
    # DC dropped -> beat bin 20 sits at stored index 19
    assert np.argmax(np.abs(rt).mean(1)) == 19
    P = stft_bins(rt)
    S, centre = range_gate(P, "track")
    assert np.all(np.abs(centre - 19) <= 1)
    f = np.argmax(S.mean(1)) - 128  # nfft=256, fftshifted, bin = f_d / (PRF/256)
    assert abs(f - round(100.0 / (PRF / 256))) <= 1


def test_track_gate_robust_to_range_offset():
    """Moving the target outside the fixed band must not change the tracked spectrogram much."""
    imgs = []
    for b in (15, 40):
        P = stft_bins(mti(range_compress(synth(beat_bin=b, f_d=60.0, seed=b))))
        imgs.append((make_image(P, "track", "noise_f"), make_image(P, "fixed", "max")))
    d_track = np.abs(imgs[0][0] - imgs[1][0]).mean()
    d_fixed = np.abs(imgs[0][1] - imgs[1][1]).mean()
    assert d_track < 0.5 * d_fixed


def test_reference_invariance_to_gain():
    rng = np.random.default_rng(1)
    S = rng.gamma(1.0, 1.0, (128, 128)).astype(np.float32)
    for mode in ("max", "noise", "noise_f"):
        a, b = reference_db(S, mode), reference_db(S * 1e3, mode)
        assert np.allclose(a, b, atol=1e-4), mode
        assert a.min() >= 0 and a.max() <= 1


def test_read_dat(tmp_path):
    p = tmp_path / "1P01A01R1.dat"
    vals = ["5800000000", "1", "4", "400000000"] + ["1+2i", "-0.5-1i", "3+0i", "0-4i"] * 2
    p.write_text("\n".join(vals))
    hdr, d = read_dat(p)
    assert hdr["nts"] == 4 and d.shape == (2, 4)
    assert d[0, 1] == np.complex64(-0.5 - 1j)


def test_adabn_order_invariance():
    """Regression test: AdaBN statistics must not depend on the (class-sorted) storage order."""
    import torch
    from src.models.nets import LightCNN, adabn
    torch.manual_seed(0)
    x = torch.cat([torch.rand(64, 1, 32, 32) * s for s in (0.2, 1.0, 3.0)])  # three "classes", sorted
    m1, m2 = LightCNN(3), LightCNN(3)
    m2.load_state_dict(m1.state_dict())
    adabn(m1, x, batch=32, seed=0)
    adabn(m2, x[torch.randperm(len(x))], batch=32, seed=1)
    bn1 = [m for m in m1.modules() if isinstance(m, torch.nn.BatchNorm2d)][0]
    bn2 = [m for m in m2.modules() if isinstance(m, torch.nn.BatchNorm2d)][0]
    assert torch.allclose(bn1.running_var, bn2.running_var, rtol=0.1)
    assert torch.allclose(bn1.running_mean, bn2.running_mean, rtol=0.05, atol=1e-3)
