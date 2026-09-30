"""
Shazam-style audio fingerprinting for a local folder of NEFFEX MP3s.

Based on:
    Avery Li-Chun Wang, "An Industrial-Strength Audio Search Algorithm",
    Proc. ISMIR 2003, Baltimore, MD.
    https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf

Pipeline:
    audio -> mono, 11,025 Hz -> STFT spectrogram (dB)
          -> local peaks ("constellation map")
          -> hashes from peak pairs: (freq_anchor, freq_target, time_delta)
          -> index: hash -> (song_id, anchor_time)
    query -> same hashes -> look up -> histogram of (db_time - query_time)
          -> the song with the tallest aligned histogram bin wins

Usage (one script, no server/database):
    python neffex_id.py build                  # fingerprint "Neffex mp3/" -> neffex_index.npz
    python neffex_id.py listen --seconds 8     # record from mic and identify
    python neffex_id.py file some_clip.wav     # identify an audio file
    python neffex_id.py selftest --trials 20   # random noisy excerpts, reports accuracy
    python neffex_id.py                        # same as "listen"

The index is rebuilt automatically when MP3s are added, removed, or modified.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MUSIC_DIR = SCRIPT_DIR / "Neffex mp3"
INDEX_PATH = SCRIPT_DIR / "neffex_index.npz"
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

SAMPLE_RATE = 11025       # Most song energy is < 5 kHz; low SR = faster + smaller index
N_FFT = 2048              # ~186 ms window -> 1025 frequency bins (~5.4 Hz each)
HOP = 512                 # ~46 ms per frame

PEAK_NEIGHBORHOOD = (25, 15)  # (freq bins, time frames) for the local-max filter
PEAK_MIN_DB = -60.0           # peaks must be within 60 dB of the clip's loudest point
PEAKS_PER_SECOND = 30         # cap density so quiet/loud tracks get similar coverage
MIN_FREQ_BIN = 5              # ignore DC / sub-bass rumble (< ~27 Hz)

FAN_OUT = 10              # pair each anchor with up to 10 peaks in its target zone
TARGET_DT = (1, 63)       # target zone: 1..63 frames ahead (~0.05 s .. 2.9 s)
TARGET_DF = 200           # and within +/-200 bins (~1 kHz) of the anchor

MIN_MATCH_SCORE = 8       # aligned hash matches required to report a result
MIN_CONFIDENCE = 1.5      # ...and the winner must beat the runner-up by this factor


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------
def load_audio(path):
    """
    Decode an audio file to mono at SAMPLE_RATE.

    Args:
        path: str or pathlib.Path to an .mp3/.wav/.flac/.ogg/.m4a file.

    Returns:
        1-D numpy float32 array of samples in [-1, 1].
    """
    import librosa  # imported lazily so worker processes stay light
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def spectrogram_db(y):
    """
    Magnitude STFT in decibels, computed with NumPy only.

    Args:
        y: 1-D numpy float array of audio samples at SAMPLE_RATE.

    Returns:
        2-D numpy float array of shape (N_FFT // 2 + 1, n_frames),
        rows = frequency bins, columns = time frames.
    """
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    n_frames = 1 + (len(y) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n_frames, N_FFT), strides=(y.strides[0] * HOP, y.strides[0])
    )
    window = np.hanning(N_FFT).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * window, axis=1)).T  # (freq, time)
    return 20.0 * np.log10(mag + 1e-10)


def find_peaks(S):
    """
    Find local maxima in the spectrogram (the "constellation map").

    Args:
        S: 2-D numpy array from spectrogram_db(), shape (freq_bins, frames).

    Returns:
        Tuple (freqs, times) of equal-length 1-D numpy int arrays holding the
        frequency bin and time frame of each peak, sorted by time then frequency.
    """
    local_max = maximum_filter(S, size=PEAK_NEIGHBORHOOD, mode="constant", cval=-np.inf) == S
    mask = local_max & (S > S.max() + PEAK_MIN_DB)
    mask[:MIN_FREQ_BIN, :] = False
    f, t = np.nonzero(mask)

    # Keep only the strongest peaks if density is too high
    max_peaks = int(PEAKS_PER_SECOND * S.shape[1] * HOP / SAMPLE_RATE) + 1
    if len(f) > max_peaks:
        keep = np.argpartition(S[f, t], -max_peaks)[-max_peaks:]
        f, t = f[keep], t[keep]

    order = np.lexsort((f, t))  # sort by time, then frequency
    return f[order], t[order]


def hash_peaks(f, t):
    """
    Combinatorial hashing: pair each anchor peak with nearby future peaks.
    Packed layout: f_anchor (11 bits) | f_target (11 bits) | dt (6 bits).

    Args:
        f: 1-D numpy int array of peak frequency bins (sorted by time).
        t: 1-D numpy int array of peak time frames, same length as f.

    Returns:
        Tuple (hashes, anchor_times):
            hashes: 1-D numpy uint32 array of packed hash values.
            anchor_times: 1-D numpy uint32 array, time frame of each hash's anchor.
    """
    hashes, times = [], []
    n = len(f)
    for i in range(n):
        paired = 0
        for j in range(i + 1, n):
            dt = t[j] - t[i]
            if dt < TARGET_DT[0]:
                continue
            if dt > TARGET_DT[1]:
                break
            if abs(int(f[j]) - int(f[i])) > TARGET_DF:
                continue
            hashes.append((int(f[i]) << 17) | (int(f[j]) << 6) | int(dt))
            times.append(int(t[i]))
            paired += 1
            if paired >= FAN_OUT:
                break
    return np.asarray(hashes, dtype=np.uint32), np.asarray(times, dtype=np.uint32)


def fingerprint(y):
    """
    Full fingerprint pipeline: spectrogram -> peaks -> hashes.

    Args:
        y: 1-D numpy float array of audio samples at SAMPLE_RATE.

    Returns:
        Tuple (hashes, anchor_times) as described in hash_peaks().
    """
    return hash_peaks(*find_peaks(spectrogram_db(y)))


def _fingerprint_file(path):
    """
    Worker function for parallel index building.

    Args:
        path: str path to an audio file.

    Returns:
        Tuple (path, hashes, anchor_times, duration_seconds).
    """
    y = load_audio(path)
    h, t = fingerprint(y)
    return path, h, t, len(y) / SAMPLE_RATE


# ---------------------------------------------------------------------------
# Index (sorted NumPy arrays + binary search instead of a database)
# ---------------------------------------------------------------------------
class Index:
    """
    In-memory fingerprint index.

    Attributes:
        hashes: 1-D numpy uint32 array, sorted ascending.
        song_ids: 1-D numpy uint16 array aligned with hashes.
        offsets: 1-D numpy uint32 array aligned with hashes (anchor frame in song).
        names: list of song names (str), indexed by song_id.
        signature: str describing the music folder + settings, used to
            detect when a rebuild is needed.
    """

    def __init__(self, hashes, song_ids, offsets, names, signature):
        self.hashes = hashes
        self.song_ids = song_ids
        self.offsets = offsets
        self.names = names
        self.signature = signature

    def save(self, path=INDEX_PATH):
        """
        Write the index to a compressed .npz file.

        Args:
            path: destination pathlib.Path or str (default INDEX_PATH).

        Returns:
            None.
        """
        np.savez_compressed(path, hashes=self.hashes, song_ids=self.song_ids,
                            offsets=self.offsets, names=np.array(self.names),
                            signature=np.array(self.signature))

    @classmethod
    def load(cls, path=INDEX_PATH):
        """
        Read an index previously written by save().

        Args:
            path: source pathlib.Path or str (default INDEX_PATH).

        Returns:
            Index instance.
        """
        d = np.load(path, allow_pickle=False)
        return cls(d["hashes"], d["song_ids"], d["offsets"],
                   [str(n) for n in d["names"]], str(d["signature"]))


def list_songs(music_dir=MUSIC_DIR):
    """
    List audio files in the music folder (exits if none are found).

    Args:
        music_dir: pathlib.Path of the folder to scan (default MUSIC_DIR).

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
    Build a string that changes whenever files or fingerprint settings change.

    Args:
        files: list of pathlib.Path audio files.

    Returns:
        str signature.
    """
    cfg = f"{SAMPLE_RATE},{N_FFT},{HOP},{PEAK_NEIGHBORHOOD},{PEAKS_PER_SECOND},{FAN_OUT},{TARGET_DT},{TARGET_DF}"
    parts = [f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in files]
    return cfg + "|" + "|".join(parts)


def build_index(music_dir=MUSIC_DIR, workers=None):
    """
    Fingerprint every song in parallel, sort by hash, and save to INDEX_PATH.

    Args:
        music_dir: pathlib.Path of the music folder (default MUSIC_DIR).
        workers: int number of processes, or None for (CPU count - 1).

    Returns:
        Index instance.
    """
    files = list_songs(music_dir)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"Fingerprinting {len(files)} songs with {workers} worker(s)...")
    start = time.perf_counter()

    all_h, all_ids, all_t, names = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (path, h, t, dur) in enumerate(pool.map(_fingerprint_file, map(str, files))):
            names.append(Path(path).stem)
            all_h.append(h)
            all_t.append(t)
            all_ids.append(np.full(len(h), i, dtype=np.uint16))
            print(f"  [{i + 1:>3}/{len(files)}] {Path(path).stem:<45} {dur:6.1f}s  {len(h):>7,} hashes")

    hashes = np.concatenate(all_h)
    order = np.argsort(hashes, kind="stable")
    index = Index(hashes[order], np.concatenate(all_ids)[order],
                  np.concatenate(all_t)[order], names, folder_signature(files))
    index.save()
    print(f"Built index: {len(hashes):,} hashes in {time.perf_counter() - start:.1f}s -> {INDEX_PATH.name}")
    return index


def get_index(force_rebuild=False):
    """
    Load the cached index, rebuilding it if missing or out of date.

    Args:
        force_rebuild: bool, True to ignore the cache.

    Returns:
        Index instance.
    """
    files = list_songs()
    if not force_rebuild and INDEX_PATH.exists():
        index = Index.load()
        if index.signature == folder_signature(files):
            return index
        print("Music folder changed; rebuilding index.")
    return build_index()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
class Match:
    """
    One candidate result.

    Attributes:
        name: str song name.
        score: int number of hashes agreeing on a single time offset.
        offset_sec: float estimated position in the song where the clip starts.
        confidence: float ratio of this score to the runner-up's score.
    """

    def __init__(self, name, score, offset_sec, confidence):
        self.name = name
        self.score = score
        self.offset_sec = offset_sec
        self.confidence = confidence


def identify(y, index, top_k=3):
    """
    Match a clip against the index.

    Args:
        y: 1-D numpy float array of audio samples at SAMPLE_RATE.
        index: Index instance.
        top_k: int number of candidates to return.

    Returns:
        List of up to top_k Match objects, best first (empty if nothing matched).
    """
    q_hash, q_time = fingerprint(y)
    if len(q_hash) == 0:
        return []

    # Binary-search each query hash in the sorted index
    lo = np.searchsorted(index.hashes, q_hash, side="left")
    hi = np.searchsorted(index.hashes, q_hash, side="right")
    counts = hi - lo
    if counts.sum() == 0:
        return []

    # Expand every (query hash -> all db rows with that hash) pair, vectorized
    q_rep = np.repeat(np.arange(len(q_hash)), counts)
    starts = np.repeat(lo, counts)
    within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    db_rows = starts + within

    songs = index.song_ids[db_rows].astype(np.int64)
    delta = index.offsets[db_rows].astype(np.int64) - q_time[q_rep].astype(np.int64)

    # Histogram over (song, offset): true matches pile up on one offset
    key = songs * 1_000_000 + (delta + 500_000)
    uniq, freq = np.unique(key, return_counts=True)
    key_song = uniq // 1_000_000
    key_delta = uniq % 1_000_000 - 500_000

    best_per_song = {}
    for s, d, c in zip(key_song, key_delta, freq):
        if c > best_per_song.get(s, (0, 0))[0]:
            best_per_song[s] = (int(c), int(d))

    ranked = sorted(best_per_song.items(), key=lambda kv: kv[1][0], reverse=True)
    runner_up = ranked[1][1][0] if len(ranked) > 1 else 1
    results = []
    for s, (score, d) in ranked[:top_k]:
        results.append(Match(index.names[s], score, d * HOP / SAMPLE_RATE,
                             score / max(runner_up, 1)))
    return results


def is_confident(matches):
    """
    Decide whether the top match is strong enough to report.

    Args:
        matches: list of Match objects from identify().

    Returns:
        bool, True if the best match passes MIN_MATCH_SCORE and MIN_CONFIDENCE.
    """
    return bool(matches) and matches[0].score >= MIN_MATCH_SCORE and matches[0].confidence >= MIN_CONFIDENCE


def print_result(matches, elapsed):
    """
    Pretty-print identification results.

    Args:
        matches: list of Match objects from identify().
        elapsed: float seconds spent matching.

    Returns:
        None.
    """
    print(f"\nMatched in {elapsed * 1000:.0f} ms")
    if not is_confident(matches):
        print("No confident match. Try a longer or louder recording.")
        if matches:
            print(f"  (closest: {matches[0].name}, score {matches[0].score})")
        return
    best = matches[0]
    m, s = divmod(max(best.offset_sec, 0), 60)
    print(f"  >>> {best.name}")
    print(f"      score {best.score}, {best.confidence:.1f}x runner-up, clip starts ~{int(m)}:{s:04.1f}")
    for alt in matches[1:]:
        print(f"      alt: {alt.name} (score {alt.score})")


# ---------------------------------------------------------------------------
# Microphone capture (sounddevice, as in the starter test.py)
# ---------------------------------------------------------------------------
def record(seconds, device=None):
    """
    Record from the microphone at its native rate, then resample.

    Args:
        seconds: float recording length.
        device: sounddevice input device id (int), name (str), or None for default.

    Returns:
        1-D numpy float32 array of samples at SAMPLE_RATE.
    """
    import librosa
    import sounddevice as sd

    # Many mics reject 11,025 Hz, so record natively and resample
    native_sr = int(sd.query_devices(device, kind="input")["default_samplerate"])
    print(f"Recording {seconds:g}s at {native_sr} Hz... play a NEFFEX song near the mic.")
    audio = sd.rec(int(seconds * native_sr), samplerate=native_sr, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    y = audio[:, 0]
    peak = float(np.abs(y).max())
    print(f"Captured. Peak level {peak:.3f}" + ("  (very quiet - check input device)" if peak < 0.01 else ""))
    return librosa.resample(y, orig_sr=native_sr, target_sr=SAMPLE_RATE).astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test: random excerpts + noise, no microphone needed
# ---------------------------------------------------------------------------
def degrade(y, snr_db, rng):
    """
    Rough "phone near a speaker" simulation: band-limit, gain change, noise.

    Args:
        y: 1-D numpy float array of clean samples at SAMPLE_RATE.
        snr_db: float target signal-to-noise ratio in dB.
        rng: numpy.random.Generator.

    Returns:
        1-D numpy float32 array, same length as y.
    """
    from scipy.signal import butter, sosfilt
    sos = butter(4, [300, 3400], btype="band", fs=SAMPLE_RATE, output="sos")
    y = sosfilt(sos, y) * rng.uniform(0.3, 1.0)
    noise = rng.normal(size=len(y))
    noise *= np.sqrt(np.mean(y ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10) + 1e-12))
    return (y + noise).astype(np.float32)


def selftest(index, trials, seconds, snr_db, seed):
    """
    Identify random degraded excerpts from the library and report accuracy.

    Args:
        index: Index instance.
        trials: int number of excerpts to test.
        seconds: float excerpt length.
        snr_db: float signal-to-noise ratio for added noise.
        seed: int random seed for reproducibility.

    Returns:
        None (prints per-trial results and overall accuracy).
    """
    rng = np.random.default_rng(seed)
    files = list_songs()
    correct, times = 0, []
    print(f"Self-test: {trials} trials, {seconds:g}s clips, {snr_db:g} dB SNR, telephone band-pass\n")
    for i in range(trials):
        path = files[rng.integers(len(files))]
        y = load_audio(path)
        n = int(seconds * SAMPLE_RATE)
        start = int(rng.integers(0, max(1, len(y) - n)))
        clip = degrade(y[start:start + n], snr_db, rng)

        t0 = time.perf_counter()
        matches = identify(clip, index)
        times.append(time.perf_counter() - t0)
        ok = is_confident(matches) and matches[0].name == path.stem
        correct += ok
        got = matches[0].name if matches else "-"
        score = matches[0].score if matches else 0
        print(f"  {'OK ' if ok else 'MISS'} {path.stem:<35} @ {start / SAMPLE_RATE:6.1f}s -> {got} (score {score})")
    print(f"\nAccuracy {correct}/{trials} = {100 * correct / trials:.0f}%   "
          f"median match time {1000 * np.median(times):.0f} ms")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    """
    Command-line entry point. See module docstring for usage.

    Args:
        None (reads sys.argv).

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Identify NEFFEX songs from audio.")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("build", help="(Re)build the fingerprint index")
    p_listen = sub.add_parser("listen", help="Record from the microphone and identify")
    p_listen.add_argument("--seconds", type=float, default=8)
    p_listen.add_argument("--device", default=None, help="sounddevice input device id/name")
    p_file = sub.add_parser("file", help="Identify an audio file")
    p_file.add_argument("path")
    p_test = sub.add_parser("selftest", help="Accuracy test on noisy random excerpts")
    p_test.add_argument("--trials", type=int, default=20)
    p_test.add_argument("--seconds", type=float, default=5)
    p_test.add_argument("--snr", type=float, default=5, help="signal-to-noise ratio in dB")
    p_test.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.cmd == "build":
        build_index()
        return

    index = get_index()
    print(f"Index ready: {len(index.names)} songs, {len(index.hashes):,} hashes")

    if args.cmd == "selftest":
        selftest(index, args.trials, args.seconds, args.snr, args.seed)
        return

    if args.cmd == "file":
        y = load_audio(args.path)
    else:  # "listen" or no command
        seconds = getattr(args, "seconds", 8)
        device = getattr(args, "device", None)
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        y = record(seconds, device)

    t0 = time.perf_counter()
    matches = identify(y, index)
    print_result(matches, time.perf_counter() - t0)


if __name__ == "__main__":  # required for ProcessPoolExecutor on Windows/macOS
    main()
