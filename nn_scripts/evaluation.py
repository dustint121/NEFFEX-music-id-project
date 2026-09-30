"""
Accuracy tests: selftest (neural only) and compare (neural vs neffex_id.py).
"""

import time

import numpy as np

from .audio_library import AudioLibrary
from .augment import degrade
from .config import SAMPLE_RATE, pick_device
from .index import get_index
from .matching import identify, is_confident
from .model import load_model


def selftest(trials, seconds, snr_db, seed):
    """
    Identify random degraded excerpts; report accuracy for seen vs held-out songs.

    Args:
        trials: int number of excerpts.
        seconds: float excerpt length.
        snr_db: float signal-to-noise ratio.
        seed: int random seed.
    """
    lib = AudioLibrary.get()
    device = pick_device()
    model, holdout = load_model(device)
    index = get_index()
    rng = np.random.default_rng(seed)
    stats = {"seen": [0, 0], "held-out": [0, 0]}
    times = []
    print(f"Self-test: {trials} trials, {seconds:g}s clips, {snr_db:g} dB SNR, telephone band-pass\n")
    for _ in range(trials):
        sid = int(rng.integers(len(lib.names)))
        y = lib.song(sid)
        n = int(seconds * SAMPLE_RATE)
        start = int(rng.integers(0, max(1, len(y) - n)))
        clip = degrade(y[start:start + n], snr_db, rng)
        t0 = time.perf_counter()
        matches = identify(clip, model, index, device)
        times.append(time.perf_counter() - t0)
        name = lib.names[sid]
        ok = is_confident(matches) and matches[0].name == name
        group = "held-out" if name in holdout else "seen"
        stats[group][0] += ok
        stats[group][1] += 1
        got = matches[0].name if matches else "-"
        score = matches[0].score if matches else 0
        tag = " (held out)" if group == "held-out" else ""
        print(f"  {'OK ' if ok else 'MISS'} {name + tag:<46} @ {start / SAMPLE_RATE:6.1f}s -> {got} ({score:.2f})")
    total = sum(v[0] for v in stats.values())
    print(f"\nAccuracy {total}/{trials} = {100 * total / trials:.0f}%   median match time "
          f"{1000 * np.median(times):.0f} ms")
    for g, (c, n) in stats.items():
        if n:
            print(f"  {g:<9} songs: {c}/{n} = {100 * c / n:.0f}%")


def compare(trials, seconds, snr_db, seed):
    """
    Run identical degraded clips through neffex_id.py (landmark hashes) and the
    network, and print a side-by-side accuracy / latency / index-size summary.

    Args:
        trials: int number of excerpts.
        seconds: float excerpt length.
        snr_db: float signal-to-noise ratio.
        seed: int random seed.
    """
    import librosa
    import neffex_id as classic

    c_index = classic.get_index()
    device = pick_device()
    model, _ = load_model(device)
    n_index = get_index()
    files = classic.list_songs()
    rng = np.random.default_rng(seed)
    res = {"landmark": [0, []], "neural": [0, []]}
    print(f"Compare: {trials} trials, {seconds:g}s clips, {snr_db:g} dB SNR\n")
    for _ in range(trials):
        path = files[rng.integers(len(files))]
        y = classic.load_audio(path)
        n = int(seconds * classic.SAMPLE_RATE)
        start = int(rng.integers(0, max(1, len(y) - n)))
        clip = classic.degrade(y[start:start + n], snr_db, rng)  # one clip for both systems

        t0 = time.perf_counter()
        cm = classic.identify(clip, c_index)
        res["landmark"][1].append(time.perf_counter() - t0)
        c_ok = classic.is_confident(cm) and cm[0].name == path.stem
        res["landmark"][0] += c_ok

        clip8 = librosa.resample(clip, orig_sr=classic.SAMPLE_RATE, target_sr=SAMPLE_RATE)
        t0 = time.perf_counter()
        nm = identify(clip8, model, n_index, device)
        res["neural"][1].append(time.perf_counter() - t0)
        n_ok = is_confident(nm) and nm[0].name == path.stem
        res["neural"][0] += n_ok
        print(f"  landmark {'OK ' if c_ok else 'MISS'}  neural {'OK ' if n_ok else 'MISS'}  {path.stem}")

    c_mb = (c_index.hashes.nbytes + c_index.song_ids.nbytes + c_index.offsets.nbytes) / 1e6
    n_mb = n_index.emb.nbytes / 1e6
    print(f"\n{'system':<10}{'accuracy':>12}{'median ms':>12}{'index MB':>11}")
    for name, mb in (("landmark", c_mb), ("neural", n_mb)):
        ok, ts = res[name]
        print(f"{name:<10}{f'{ok}/{trials}':>12}{1000 * np.median(ts):>12.0f}{mb:>11.0f}")
