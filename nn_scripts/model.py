"""
The fingerprint network, its contrastive loss, and checkpoint loading.
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EMBED_DIM, MODEL_PATH, N_FFT, N_MELS, SAMPLE_RATE, STFT_HOP, TEMPERATURE


class LogMel(nn.Module):
    """
    Waveform -> per-example normalised log-mel spectrogram, computed on-device.

    forward(x):
        Args:
            x: torch float tensor (batch, samples) at SAMPLE_RATE.
        Returns:
            torch float tensor (batch, 1, N_MELS, frames).
    """

    def __init__(self):
        super().__init__()
        import librosa
        mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=50,
                                  fmax=SAMPLE_RATE / 2)
        self.register_buffer("mel", torch.from_numpy(mel).float())
        self.register_buffer("window", torch.hann_window(N_FFT))

    def forward(self, x):
        x = x.float()
        spec = torch.stft(x, N_FFT, STFT_HOP, window=self.window, return_complex=True).abs() ** 2
        logmel = torch.log(self.mel @ spec + 1e-6)
        mu = logmel.mean(dim=(1, 2), keepdim=True)
        sd = logmel.std(dim=(1, 2), keepdim=True) + 1e-5
        return ((logmel - mu) / sd).unsqueeze(1)  # gain-invariant


class FingerprintNet(nn.Module):
    """
    Small CNN encoder: 1 s waveform -> EMBED_DIM unit-length fingerprint.

    forward(x):
        Args:
            x: torch float tensor (batch, SEG).
        Returns:
            torch float tensor (batch, EMBED_DIM), L2-normalised rows.
    """

    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.frontend = LogMel()

        def block(cin, cout, double):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                      nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            if double:
                layers += [nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                           nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers, nn.MaxPool2d(2))

        # (64 mels x 32 frames) -> 32x16 -> 16x8 -> 8x4 -> 4x2 -> global average
        # Single convs at high resolution (cheap), double convs deeper (capacity)
        self.encoder = nn.Sequential(block(1, 32, False), block(32, 64, False),
                                     block(64, 128, True), block(128, 256, True),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Linear(512, dim))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            feats = self.frontend(x)  # STFT stays in float32 even under AMP
        return F.normalize(self.head(self.encoder(feats)).float(), dim=1)


def nt_xent(za, zp, temperature=TEMPERATURE):
    """
    SimCLR contrastive loss with in-batch negatives.

    Args:
        za: torch tensor (B, D) of anchor embeddings (unit length).
        zp: torch tensor (B, D) of positive embeddings, row i pairs with za[i].
        temperature: float softmax temperature.

    Returns:
        Tuple (loss, top1): loss is a scalar tensor; top1 is a float, the fraction
        of anchors whose nearest neighbour in the batch is their own positive.
    """
    z = torch.cat([za, zp])
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))
    b = za.shape[0]
    target = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    loss = F.cross_entropy(sim, target)
    top1 = (sim.argmax(dim=1) == target).float().mean().item()
    return loss, top1


def load_model(device):
    """
    Load the trained network.

    Args:
        device: torch.device to place the model on.

    Returns:
        Tuple (model, holdout_names): model in eval mode, and the list of song
        names that were excluded from training.
    """
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run: python neffex_nn.py train")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = FingerprintNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("holdout_names", [])
