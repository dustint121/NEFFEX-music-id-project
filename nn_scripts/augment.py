"""
Audio degradations: what happens to music between a speaker and a microphone.

augment() is the batched, on-device version used for training.
degrade() is the simple NumPy version used by selftest (same as neffex_id.py).
"""

import numpy as np
import torch

from .config import SAMPLE_RATE


def _scale_to_snr(signal, noise, snr_db):
    """
    Scale noise so that signal/noise power equals snr_db for each row.

    Args:
        signal: torch tensor (B, N).
        noise: torch tensor (B, N).
        snr_db: torch tensor (B,) of target SNRs in dB.

    Returns:
        torch tensor (B, N), the rescaled noise.
    """
    ps = signal.pow(2).mean(dim=1, keepdim=True)
    pn = noise.pow(2).mean(dim=1, keepdim=True) + 1e-10
    return noise * torch.sqrt(ps / (pn * 10 ** (snr_db[:, None] / 10)))


def augment(x, background):
    """
    Degrade clean segments: reverb, background music, band-pass, noise, gain.

    Args:
        x: torch float tensor (B, N) of clean audio.
        background: torch float tensor (B, N) of unrelated audio to mix in.

    Returns:
        torch float tensor (B, N) of degraded audio.
    """
    b, n = x.shape
    dev = x.device

    def rand(lo, hi):
        return torch.empty(b, device=dev).uniform_(lo, hi)

    def chance(p):
        return (torch.rand(b, device=dev) < p)[:, None]

    # Room reverb: exponentially decaying noise impulse response, FFT convolution
    ir_len = SAMPLE_RATE // 2
    t = torch.arange(ir_len, device=dev) / SAMPLE_RATE
    ir = torch.randn(b, ir_len, device=dev) * torch.exp(-6.9 * t[None] / rand(0.05, 0.5)[:, None])
    ir[:, 0] = rand(1.0, 6.0)
    ir = ir / ir.norm(dim=1, keepdim=True)
    m = n + ir_len
    wet = torch.fft.irfft(torch.fft.rfft(x, m) * torch.fft.rfft(ir, m), m)[:, :n]
    x = torch.where(chance(0.7), wet, x)

    # Background music / chatter from another random segment
    x = x + chance(0.5) * _scale_to_snr(x, background, rand(3, 20))

    # Speaker + microphone band-limiting via a random band-pass mask
    spec = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(n, 1 / SAMPLE_RATE).to(dev)
    band = (freqs[None] >= rand(50, 400)[:, None]) & (freqs[None] <= rand(2500, 3950)[:, None])
    x = torch.where(chance(0.8), torch.fft.irfft(spec * band, n), x)

    # White or pink noise
    white = torch.randn(b, n, device=dev)
    pink_spec = torch.fft.rfft(torch.randn(b, n, device=dev)) / torch.sqrt(freqs.clamp(min=20))[None]
    pink = torch.fft.irfft(pink_spec, n)
    noise = torch.where(chance(0.5), white, pink)
    x = x + _scale_to_snr(x, noise, rand(-3, 20))

    return x * rand(0.1, 1.0)[:, None]


def degrade(y, snr_db, rng, sr=SAMPLE_RATE):
    """
    Same "phone near a speaker" simulation as neffex_id.selftest.

    Args:
        y: 1-D numpy float array of clean audio.
        snr_db: float signal-to-noise ratio in dB.
        rng: numpy.random.Generator.
        sr: int sample rate of y.

    Returns:
        1-D numpy float32 array, same length as y.
    """
    from scipy.signal import butter, sosfilt
    sos = butter(4, [300, min(3400, sr / 2 - 100)], btype="band", fs=sr, output="sos")
    y = sosfilt(sos, y) * rng.uniform(0.3, 1.0)
    noise = rng.normal(size=len(y))
    noise *= np.sqrt(np.mean(y ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10) + 1e-12))
    return (y + noise).astype(np.float32)
