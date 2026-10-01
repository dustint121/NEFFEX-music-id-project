"""
Query matching: nearest-neighbour search over the index + time-offset voting.
"""

import numpy as np
import torch

from .config import (DB_HOP, MIN_CONFIDENCE, MIN_MATCH_SCORE, MIN_MEAN_SIM, MIN_SIM,
                     NOISE_MARGIN, QUERY_HOP, SAMPLE_RATE, TOP_K)
from .index import embed_windows, frame_audio
from .noise import get_guard, loud_enough


class Match:
    """
    One candidate result.

    Attributes:
        name: str song name.
        score: float summed cosine similarity of time-aligned windows.
        offset_sec: float estimated position in the song where the clip starts.
        confidence: float ratio of this score to the runner-up's score.
        n_windows: int number of query windows the clip was split into
    """

    def __init__(self, name, score, offset_sec, confidence, n_windows):
        self.name = name
        self.score = score
        self.offset_sec = offset_sec
        self.confidence = confidence
        self.n_windows = n_windows


def identify(y, model, index, device, top_k=3):
    """
    Identify a clip: embed windows, nearest-neighbour search, offset voting.

    Windows that are near-silent or sound more like noise than music cast no
    votes (see nn_scripts/noise.py). n_windows still counts every window, so
    the score threshold is not lowered when windows are dropped.

    Args:
        y: 1-D numpy float array of audio at SAMPLE_RATE.
        model: FingerprintNet in eval mode.
        index: NNIndex instance.
        device: torch.device.
        top_k: int number of candidates to return.

    Returns:
        List of up to top_k Match objects, best first (empty if nothing matched).
    """
    windows = frame_audio(y, QUERY_HOP)
    q = embed_windows(model, windows, device)
    if len(q) == 0:
        return []

    # Cosine similarity of every query window vs every index window (unit vectors).
    # Index windows that are themselves silence/noise are excluded as neighbours.
    guard = get_guard(model, index, device)
    db = index.matrix(device)
    qt = torch.from_numpy(q).to(device=device, dtype=db.dtype)
    sims = (qt @ db.T).float()
    sims[:, ~guard.db_mask] = -1.0
    top_s, top_i = sims.topk(min(TOP_K, sims.shape[1]), dim=1)

    # A window votes only if it has sound and is clearly closer to music than to noise
    margin = top_s[:, 0] - guard.noise_sim(qt)
    voiced = loud_enough(windows) & (margin >= NOISE_MARGIN).cpu().numpy()
    top_s, top_i = top_s.cpu().numpy(), top_i.cpu().numpy()
    top_s[~voiced] = -1.0   # below MIN_SIM, so removed by the filter below

    # Vote: neighbours that agree on (song, db_window - query_window) add up
    step = QUERY_HOP // DB_HOP
    q_idx = np.repeat(np.arange(len(q)), top_s.shape[1])
    s, i = top_s.ravel(), top_i.ravel()
    keep = s >= MIN_SIM
    if not keep.any():
        return []
    s, i, q_idx = s[keep], i[keep], q_idx[keep]
    songs = index.song_ids[i].astype(np.int64)
    delta = index.seg_idx[i].astype(np.int64) - q_idx * step
    key = songs * 1_000_000 + delta + 500_000
    uniq, inv = np.unique(key, return_inverse=True)
    votes = np.zeros(len(uniq))
    # One vote per (key, query window): keep its best neighbour, which comes
    # first because topk sorts descending, so adjacent neighbours don't double count
    pair = np.unique(inv * 100_000 + q_idx, return_index=True)[1]
    np.add.at(votes, inv[pair], s[pair])

    best = {}
    for k, v in zip(uniq, votes):
        sid, d = int(k // 1_000_000), int(k % 1_000_000 - 500_000)
        if v > best.get(sid, (0.0, 0))[0]:
            best[sid] = (float(v), d)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    runner_up = ranked[1][1][0] if len(ranked) > 1 else 0.0
    return [Match(index.names[sid], v, d * DB_HOP / SAMPLE_RATE, v / max(runner_up, 1e-6), len(q))
            for sid, (v, d) in ranked[:top_k]]


def score_threshold(n_windows):
    """
    Minimum score for a clip split into n_windows query windows.

    Each window adds at most 1.0, so a fixed threshold would make short clips
    impossible to match; the bar is capped at MIN_MEAN_SIM per window.

    Args:
        n_windows: int number of query windows.

    Returns:
        float threshold.
    """
    return min(MIN_MATCH_SCORE, MIN_MEAN_SIM * n_windows)


def is_confident(matches):
    """
    Decide whether the top match is strong enough to report.

    Args:
        matches: list of Match objects from identify().

    Returns:
        bool.
    """
    return (bool(matches) and matches[0].score >= score_threshold(matches[0].n_windows)
            and matches[0].confidence >= MIN_CONFIDENCE)


def print_result(matches, elapsed):
    """
    Pretty-print identification results.

    Args:
        matches: list of Match objects.
        elapsed: float seconds spent matching.
    """
    print(f"\nMatched in {elapsed * 1000:.0f} ms")
    if not is_confident(matches):
        print("No confident match. Try a longer or louder recording.")
        if matches:
            print(f"  (closest: {matches[0].name}, score {matches[0].score:.2f})")
        return
    best = matches[0]
    m, sec = divmod(max(best.offset_sec, 0), 60)
    print(f"  >>> {best.name}")
    print(f"      score {best.score:.2f}, {min(best.confidence, 99):.1f}x runner-up, "
          f"clip starts ~{int(m)}:{sec:04.1f}")
    for alt in matches[1:]:
        print(f"      alt: {alt.name} (score {alt.score:.2f})")
