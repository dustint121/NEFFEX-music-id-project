"""
Query matching: nearest-neighbour search over the index + time-offset voting.
"""

import numpy as np
import torch

from .config import (DB_HOP, MIN_CONFIDENCE, MIN_MATCH_SCORE, MIN_SIM, QUERY_HOP,
                     SAMPLE_RATE, TOP_K)
from .index import embed_windows, frame_audio


class Match:
    """
    One candidate result.

    Attributes:
        name: str song name.
        score: float summed cosine similarity of time-aligned windows.
        offset_sec: float estimated position in the song where the clip starts.
        confidence: float ratio of this score to the runner-up's score.
    """

    def __init__(self, name, score, offset_sec, confidence):
        self.name = name
        self.score = score
        self.offset_sec = offset_sec
        self.confidence = confidence


def identify(y, model, index, device, top_k=3):
    """
    Identify a clip: embed windows, nearest-neighbour search, offset voting.

    Args:
        y: 1-D numpy float array of audio at SAMPLE_RATE.
        model: FingerprintNet in eval mode.
        index: NNIndex instance.
        device: torch.device.
        top_k: int number of candidates to return.

    Returns:
        List of up to top_k Match objects, best first (empty if nothing matched).
    """
    q = embed_windows(model, frame_audio(y, QUERY_HOP), device)
    if len(q) == 0:
        return []

    # Cosine similarity of every query window vs every index window (unit vectors)
    db = index.matrix(device)
    sims = (torch.from_numpy(q).to(device=device, dtype=db.dtype) @ db.T).float()
    top_s, top_i = sims.topk(min(TOP_K, sims.shape[1]), dim=1)
    top_s, top_i = top_s.cpu().numpy(), top_i.cpu().numpy()

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
    return [Match(index.names[sid], v, d * DB_HOP / SAMPLE_RATE, v / max(runner_up, 1e-6))
            for sid, (v, d) in ranked[:top_k]]


def is_confident(matches):
    """
    Decide whether the top match is strong enough to report.

    Args:
        matches: list of Match objects from identify().

    Returns:
        bool.
    """
    return (bool(matches) and matches[0].score >= MIN_MATCH_SCORE
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
