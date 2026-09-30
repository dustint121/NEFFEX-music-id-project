"""
Audio decoding and the on-disk audio cache.

Every song is decoded once to 8 kHz int16 and stored back-to-back in a single
.npy memmap, so training can read thousands of random 1 s windows per second
without re-decoding MP3s.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .config import AUDIO_CACHE, AUDIO_META, SAMPLE_RATE, folder_signature, list_songs


def load_audio(path, sr=SAMPLE_RATE):
    """
    Decode an audio file to mono float32.

    Args:
        path: str or pathlib.Path to an audio file.
        sr: int target sample rate.

    Returns:
        1-D numpy float32 array in [-1, 1].
    """
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _decode_int16(path):
    """
    Worker: decode one file to int16 at SAMPLE_RATE.

    Args:
        path: str path to an audio file.

    Returns:
        1-D numpy int16 array.
    """
    y = load_audio(path)
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


class AudioLibrary:
    """
    Every song decoded to 8 kHz int16 and stored back-to-back in one memmap.

    Attributes:
        audio: numpy int16 memmap holding all songs concatenated.
        starts: 1-D numpy int64 array, start sample of each song in audio.
        lengths: 1-D numpy int64 array, length in samples of each song.
        names: list of song names (str).
        signature: str folder signature this cache was built from.
    """

    def __init__(self, audio, starts, lengths, names, signature):
        self.audio = audio
        self.starts = starts
        self.lengths = lengths
        self.names = names
        self.signature = signature

    def song(self, i):
        """
        Return one song as float32.

        Args:
            i: int song index.

        Returns:
            1-D numpy float32 array in [-1, 1].
        """
        s = self.starts[i]
        return self.audio[s:s + self.lengths[i]].astype(np.float32) / 32767.0

    @classmethod
    def get(cls, workers=None):
        """
        Open the cache, decoding the music folder first if it is missing or stale.

        Args:
            workers: int number of decode processes, or None for (CPU count - 1).

        Returns:
            AudioLibrary instance.
        """
        files = list_songs()
        sig = folder_signature(files)
        if AUDIO_CACHE.exists() and AUDIO_META.exists():
            meta = np.load(AUDIO_META, allow_pickle=False)
            if str(meta["signature"]) == sig:
                return cls(np.load(AUDIO_CACHE, mmap_mode="r"), meta["starts"], meta["lengths"],
                           [str(n) for n in meta["names"]], sig)

        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        print(f"Decoding {len(files)} songs to {SAMPLE_RATE} Hz with {workers} worker(s)...")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            songs = list(pool.map(_decode_int16, map(str, files)))
        lengths = np.array([len(s) for s in songs], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        out = np.lib.format.open_memmap(AUDIO_CACHE, mode="w+", dtype=np.int16,
                                        shape=(int(lengths.sum()),))
        for s, y in zip(starts, songs):
            out[s:s + len(y)] = y
        out.flush()
        del out, songs
        names = [p.stem for p in files]
        np.savez(AUDIO_META, starts=starts, lengths=lengths, names=np.array(names),
                 signature=np.array(sig))
        hours = lengths.sum() / SAMPLE_RATE / 3600
        print(f"Cached {hours:.1f} h of audio ({AUDIO_CACHE.stat().st_size / 1e6:.0f} MB) "
              f"in {time.perf_counter() - t0:.1f}s")
        return cls(np.load(AUDIO_CACHE, mmap_mode="r"), starts, lengths, names, sig)
