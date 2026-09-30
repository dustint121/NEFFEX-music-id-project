"""
Contrastive training loop for FingerprintNet.
"""

import time

import numpy as np
import torch

from .audio_library import AudioLibrary
from .augment import augment
from .config import (EMBED_DIM, MAX_SHIFT, MODEL_PATH, N_MELS, SAMPLE_RATE, SEG,
                     format_duration, pick_device)
from .model import FingerprintNet, nt_xent


def split_songs(n_songs, holdout):
    """
    Deterministically hold out a fraction of songs from training.

    Args:
        n_songs: int number of songs in the library.
        holdout: float fraction in [0, 1) never seen during training.

    Returns:
        Tuple (train_ids, holdout_ids) of 1-D numpy int arrays.
    """
    ids = np.random.default_rng(1234).permutation(n_songs)
    k = int(round(holdout * n_songs))
    return np.sort(ids[k:]), np.sort(ids[:k])


def sample_batch(lib, song_ids, batch, rng):
    """
    Draw anchor/positive/background windows from the audio memmap.

    Args:
        lib: AudioLibrary instance.
        song_ids: 1-D numpy int array of songs to sample from.
        batch: int number of pairs.
        rng: numpy.random.Generator.

    Returns:
        Tuple (anchor, positive, background) of numpy float32 arrays (batch, SEG).
        positive is the same audio as anchor shifted by up to MAX_SHIFT samples.
    """
    lengths = lib.lengths[song_ids]
    songs = rng.choice(song_ids, size=batch, p=lengths / lengths.sum())
    span = lib.lengths[songs] - SEG - 2 * MAX_SHIFT
    starts = lib.starts[songs] + MAX_SHIFT + (rng.random(batch) * span).astype(np.int64)
    shifts = rng.integers(-MAX_SHIFT, MAX_SHIFT + 1, batch)
    bg_songs = rng.choice(len(lib.names), size=batch)
    bg_starts = lib.starts[bg_songs] + (rng.random(batch) * (lib.lengths[bg_songs] - SEG)).astype(np.int64)

    offs = np.arange(SEG)
    anchor = lib.audio[starts[:, None] + offs]
    positive = lib.audio[(starts + shifts)[:, None] + offs]
    background = lib.audio[bg_starts[:, None] + offs]
    return tuple(a.astype(np.float32) / 32767.0 for a in (anchor, positive, background))


def train(steps, batch, lr, holdout, seed):
    """
    Train FingerprintNet with contrastive learning and save it to MODEL_PATH.

    Args:
        steps: int optimisation steps.
        batch: int anchor/positive pairs per step (2 * batch views).
        lr: float peak learning rate (AdamW, one-cycle cosine decay).
        holdout: float fraction of songs excluded from training (to test
            generalisation to songs the network has never heard).
        seed: int random seed.
    """
    lib = AudioLibrary.get()
    device = pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_ids, holdout_ids = split_songs(len(lib.names), holdout)

    model = FingerprintNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training on {device} | {n_params / 1e6:.2f}M params | {len(train_ids)} train songs, "
          f"{len(holdout_ids)} held out | {steps} steps x {batch} pairs | AMP={use_amp}")
    if device.type == "cpu":
        print("  Note: no GPU detected - training will be slow. Colab/AWS GPU recommended.")

    model.train()
    t0 = time.perf_counter()
    loss_avg, acc_avg = None, None
    for step in range(1, steps + 1):
        a, p, bg = (torch.from_numpy(v).to(device, non_blocking=True)
                    for v in sample_batch(lib, train_ids, batch, rng))
        with torch.no_grad():
            p = augment(p, bg)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            za, zp = model(a), model(p)
        loss, top1 = nt_xent(za, zp)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        loss_avg = loss.item() if loss_avg is None else 0.95 * loss_avg + 0.05 * loss.item()
        acc_avg = top1 if acc_avg is None else 0.95 * acc_avg + 0.05 * top1
        if step % 50 == 0 or step == steps:
            elapsed = time.perf_counter() - t0
            rate = step / elapsed
            eta = (steps - step) / rate
            print(f"  step {step:>5}/{steps}  loss {loss_avg:.3f}  in-batch top1 {acc_avg:.3f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {rate:.1f} steps/s  "
                  f"elapsed {format_duration(elapsed)}  eta {format_duration(eta)}")

    torch.save({"state_dict": model.state_dict(), "holdout_names": [lib.names[i] for i in holdout_ids],
                "config": {"sr": SAMPLE_RATE, "seg": SEG, "n_mels": N_MELS, "dim": EMBED_DIM}},
               MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH.name} (trained in {format_duration(time.perf_counter() - t0)})")
