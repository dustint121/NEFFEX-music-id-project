"""
Noise and silence rejection for the neural matcher.

Why this exists:
    LogMel normalises every 1 s window to zero mean / unit variance, so a
    quiet microphone's noise floor reaches the network looking as loud as
    music. The model was trained only on music, so it has no "not music"
    region: noise fingerprints land close (cosine 0.75-0.95) to ordinary
    song windows, and digital silence matches silent intros/outros in the
    index exactly (cosine 1.0).

What this does (no retraining needed):
    1. Energy gate: query windows quieter than SILENCE_DBFS are dropped.
    2. Noise bank: the model fingerprints a set of synthetic noises
       (silence, dither, white/pink/brown noise, mains hum, clicks) once per process.
       A query window is kept only if it is clearly closer to a song window
       than to any noise (margin NOISE_MARGIN).
    3. Index mask: index windows that look like noise/silence
       (similarity >= DB_NOISE_SIM to the bank) are never used as neighbours.
"""

import numpy as np
import torch

from .config import DB_NOISE_SIM, SAMPLE_RATE, SEG, SILENCE_DBFS
from .index import embed_windows


def noise_windows(seed=1234):
    """
    Synthetic 1 s noise windows covering what a mic hears with no music.

    Args:
        seed: int random seed (fixed, so the bank is identical every run).

    Returns:
        2-D numpy float32 array (n, SEG).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(SEG) / SAMPLE_RATE

    def colored(alpha):
        # 1/f^alpha noise: 0 = white, 1 = pink, 2 = brown
        spec = np.fft.rfft(rng.standard_normal(SEG))
        spec /= np.maximum(np.arange(len(spec)), 1) ** (alpha / 2)
        x = np.fft.irfft(spec, SEG)
        return x / (np.abs(x).max() + 1e-12)

    bank = [np.zeros(SEG), rng.integers(-1, 2, SEG) / 32768.0]  # silence, 16-bit dither
    for level_db in (-80, -60, -40, -20):
        gain = 10 ** (level_db / 20)
        for alpha in (0, 1, 2):
            bank.append(colored(alpha) * gain)
        for hz in (50, 60):
            hum = np.sin(2 * np.pi * hz * t) + 0.3 * np.sin(2 * np.pi * 3 * hz * t)
            bank.append(hum * gain + rng.standard_normal(SEG) * gain * 0.01)
    # Taps, bumps and clicks on a quiet floor (handling noise, mic pops)
    for n_clicks in (1, 2, 4):
        x = rng.standard_normal(SEG) * 10 ** (-70 / 20)
        for pos in rng.integers(100, SEG - 400, size=n_clicks):
            decay = rng.uniform(5, 40)
            x[pos:pos + 300] += rng.uniform(0.05, 0.5) * np.exp(-np.arange(300) / decay) * rng.standard_normal(300)
        bank.append(x)
    return np.asarray(bank, dtype=np.float32)


def window_dbfs(windows):
    """
    RMS level of each window in dBFS (0 dBFS = full-scale sine-ish level 1.0).

    Args:
        windows: 2-D numpy float array (n, SEG).

    Returns:
        1-D numpy float array (n,) of levels in dB (silence -> -200).
    """
    rms = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-10))


class NoiseGuard:
    """
    Noise-bank fingerprints plus the index mask, computed once per
    (model, index) pair and cached on the index object.

    Attributes:
        bank: torch float tensor (n_noise, EMBED_DIM) of unit vectors.
        db_mask: torch bool tensor (n_index_windows,), True = usable window.
        n_masked: int number of index windows excluded as noise/silence.
    """

    def __init__(self, model, index, device):
        self.bank = torch.from_numpy(embed_windows(model, noise_windows(), device)).to(device)
        db = index.matrix(device)
        noise_sim = (db.float() @ self.bank.T).max(dim=1).values
        self.db_mask = noise_sim < DB_NOISE_SIM
        self.n_masked = int((~self.db_mask).sum())

    def noise_sim(self, q):
        """
        Best similarity of each query window to any noise in the bank.

        Args:
            q: torch float tensor (n, EMBED_DIM) of unit query fingerprints.

        Returns:
            torch float tensor (n,).
        """
        return (q.float() @ self.bank.T).max(dim=1).values


def get_guard(model, index, device):
    """
    Return the cached NoiseGuard for this model/index, building it on first use.

    Args:
        model: FingerprintNet in eval mode.
        index: NNIndex.
        device: torch.device.

    Returns:
        NoiseGuard.
    """
    guard = getattr(index, "_noise_guard", None)
    if guard is None or getattr(index, "_noise_guard_model", None) is not model:
        guard = NoiseGuard(model, index, device)
        index._noise_guard, index._noise_guard_model = guard, model
    return guard


def loud_enough(windows):
    """
    Energy gate for query windows.

    Args:
        windows: 2-D numpy float array (n, SEG).

    Returns:
        1-D numpy bool array (n,), True = louder than SILENCE_DBFS.
    """
    return window_dbfs(windows) >= SILENCE_DBFS
