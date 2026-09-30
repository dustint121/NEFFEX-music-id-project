"""
Shared settings, file paths, and small helpers for the neural fingerprinting package.
"""

import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths (project root = the folder containing neffex_nn.py)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MUSIC_DIR = ROOT_DIR / "Neffex mp3"
AUDIO_CACHE = ROOT_DIR / "neffex_audio_8k.npy"        # int16 memmap of every song
AUDIO_META = ROOT_DIR / "neffex_audio_8k_meta.npz"
MODEL_PATH = ROOT_DIR / "neffex_nn_model.pt"
INDEX_PATH = ROOT_DIR / "neffex_nn_index.npz"
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

# ---------------------------------------------------------------------------
# Audio framing
# ---------------------------------------------------------------------------
SAMPLE_RATE = 8000            # 8 kHz is standard for neural fingerprinting (mic/phone band)
SEG = SAMPLE_RATE             # 1.0 s segments
DB_HOP = SAMPLE_RATE // 4     # index a window every 0.25 s
QUERY_HOP = SAMPLE_RATE // 2  # query a window every 0.5 s (= 2 index steps)
MAX_SHIFT = int(0.2 * SAMPLE_RATE)  # training time-shift covers the 0.125 s grid error

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
N_FFT = 1024
STFT_HOP = 256                # -> 32 frames per 1 s segment
N_MELS = 64
EMBED_DIM = 128
TEMPERATURE = 0.05            # NT-Xent temperature
TRAIN_SNR_DB = (-3, 20)       # training noise range (dB). (-12, 20) handles -10 dB clips
                              # but costs some accuracy on cleaner clips at equal steps

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
TOP_K = 5                     # neighbours per query window
MIN_SIM = 0.3                 # ignore neighbours below this cosine similarity
MIN_MATCH_SCORE = 2.35        # summed aligned similarity needed to report a result...
MIN_MEAN_SIM = 0.67           # ...capped at this x number of query windows, so clips
                              # under 2 s (1-2 windows) can still pass. 0.67 x 3 = 2.0,
                              # so clips of 2 s and longer use the same bar as before
MIN_CONFIDENCE = 1.0          # winner must beat the runner-up by this factor


def pick_device():
    """
    Choose the fastest available torch device.

    Returns:
        torch.device ("cuda", "mps", or "cpu").
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_songs(music_dir=MUSIC_DIR):
    """
    List audio files in the music folder (exits if none are found).

    Args:
        music_dir: pathlib.Path of the folder to scan.

    Returns:
        Sorted list of pathlib.Path objects.
    """
    if not music_dir.is_dir():
        sys.exit(f'Music folder not found: "{music_dir}"')
    files = sorted(p for p in music_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        sys.exit(f'No audio files found in "{music_dir}"')
    return files


def folder_signature(files):
    """
    Build a string that changes whenever the song files change.

    Args:
        files: list of pathlib.Path audio files.

    Returns:
        str signature.
    """
    return f"{SAMPLE_RATE}|" + "|".join(
        f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in files)


def format_duration(seconds):
    """
    Format a duration for log messages.

    Args:
        seconds: float number of seconds.

    Returns:
        str such as "42s", "3m 05s", or "1h 38m 10s".
    """
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
