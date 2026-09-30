"""
Neural audio fingerprinting for a local folder of NEFFEX MP3s (PyTorch).

The deep-learning counterpart to neffex_id.py (the Wang 2003 landmark-hash baseline).
It uses the neural approach Google described for Now Playing / Sound Search:

    1-second audio segment -> log-mel spectrogram -> small CNN -> 128-d unit vector

The CNN is trained with self-supervised contrastive learning (NT-Xent / SimCLR-style).
A clean segment and a degraded copy of the same segment (noise, reverb, band-pass,
background music, small time shift) must map to nearby vectors. Every other
segment in the batch must map far away. No labels are needed.

Matching:
    index  : embed every song in 1 s windows every 0.25 s -> matrix of vectors
    query  : embed the clip in 1 s windows every 0.5 s
    search : cosine nearest neighbours for each query window (one matrix multiply)
    vote   : windows that agree on the same (song, time offset) add their
             similarities; the song with the tallest aligned vote wins
             (the same offset-alignment idea as Wang 2003, applied to embeddings)

Implementation lives in the nn_scripts/ package; this file is the command line.

References:
    Google Research 2018, Google's Next Generation Music Recognition:
        https://research.google/blog/googles-next-generation-music-recognition/
    Chang et al. 2021, Neural Audio Fingerprint for High-specific Audio Retrieval
        based on Contrastive Learning: https://arxiv.org/abs/2010.11910

Usage:
    python neffex_nn.py train --steps 3000      # decode audio cache + train the CNN
    python neffex_nn.py build                   # embed the library -> neffex_nn_index.npz
    python neffex_nn.py listen --seconds 8      # record from mic and identify
    python neffex_nn.py file some_clip.wav      # identify an audio file
    python neffex_nn.py selftest --trials 30    # noisy random excerpts, reports accuracy
    python neffex_nn.py compare --trials 30     # same clips through both systems

Requirements:
    pip install torch numpy scipy librosa soundfile sounddevice
    neffex_id.py must be in the same folder (shared mic capture + comparison).
"""

import argparse
import time

from nn_scripts.audio_library import load_audio
from nn_scripts.config import SAMPLE_RATE, pick_device
from nn_scripts.evaluation import compare, selftest
from nn_scripts.index import build_index, get_index
from nn_scripts.matching import identify, print_result
from nn_scripts.model import load_model
from nn_scripts.training import train


def record(seconds, device):
    """
    Record from the microphone (via neffex_id.py) and resample to SAMPLE_RATE.

    Args:
        seconds: float recording length.
        device: sounddevice input device id (int), name (str), or None.

    Returns:
        1-D numpy float32 array at SAMPLE_RATE.
    """
    import librosa
    import neffex_id as classic
    y = classic.record(seconds, device)
    return librosa.resample(y, orig_sr=classic.SAMPLE_RATE, target_sr=SAMPLE_RATE)


def identify_clip(y):
    """
    Load the model and index, identify one clip, and print the result.

    Args:
        y: 1-D numpy float array at SAMPLE_RATE.

    Returns:
        None.
    """
    device = pick_device()
    model, _ = load_model(device)
    index = get_index()
    print(f"Index ready: {len(index.names)} songs, {len(index.emb):,} windows")
    t0 = time.perf_counter()
    matches = identify(y, model, index, device)
    print_result(matches, time.perf_counter() - t0)


def main():
    """
    Command-line entry point. See module docstring for usage.

    Args:
        None (reads sys.argv).

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Neural NEFFEX song identifier.")
    sub = parser.add_subparsers(dest="cmd")
    p_train = sub.add_parser("train", help="Train the fingerprint network")
    p_train.add_argument("--steps", type=int, default=3000)
    p_train.add_argument("--batch", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=2e-3)
    p_train.add_argument("--holdout", type=float, default=0.1,
                         help="fraction of songs never used in training")
    p_train.add_argument("--seed", type=int, default=0)
    sub.add_parser("build", help="Embed the library into the index")
    p_listen = sub.add_parser("listen", help="Record from the microphone and identify")
    p_listen.add_argument("--seconds", type=float, default=8)
    p_listen.add_argument("--device", default=None)
    p_file = sub.add_parser("file", help="Identify an audio file")
    p_file.add_argument("path")
    for name in ("selftest", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--trials", type=int, default=30)
        p.add_argument("--seconds", type=float, default=5)
        p.add_argument("--snr", type=float, default=5)
        p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.cmd == "train":
        train(args.steps, args.batch, args.lr, args.holdout, args.seed)
    elif args.cmd == "build":
        build_index()
    elif args.cmd == "selftest":
        selftest(args.trials, args.seconds, args.snr, args.seed)
    elif args.cmd == "compare":
        compare(args.trials, args.seconds, args.snr, args.seed)
    elif args.cmd == "file":
        identify_clip(load_audio(args.path))
    else:  # "listen" or no command
        dev = getattr(args, "device", None)
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)
        identify_clip(record(getattr(args, "seconds", 8), dev))


if __name__ == "__main__":  # required for ProcessPoolExecutor on Windows/macOS
    main()
