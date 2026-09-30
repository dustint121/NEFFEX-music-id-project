"""
Embedding index: every song embedded in 1 s windows every 0.25 s.
"""

import sys
import time

import numpy as np
import torch

from .audio_library import AudioLibrary
from .config import (DB_HOP, EMBED_DIM, INDEX_PATH, MODEL_PATH, SEG, folder_signature,
                     list_songs, pick_device)
from .model import load_model


def frame_audio(y, hop):
    """
    Slice audio into overlapping SEG-length windows.

    Args:
        y: 1-D numpy float array at SAMPLE_RATE.
        hop: int samples between window starts.

    Returns:
        2-D numpy float32 array (n_windows, SEG). Short input is zero-padded.
    """
    y = np.ascontiguousarray(y, dtype=np.float32)
    if len(y) < SEG:
        y = np.pad(y, (0, SEG - len(y)))
    n = 1 + (len(y) - SEG) // hop
    return np.lib.stride_tricks.as_strided(
        y, shape=(n, SEG), strides=(y.strides[0] * hop, y.strides[0])).copy()


@torch.no_grad()
def embed_windows(model, windows, device, batch=512):
    """
    Run the network over many windows.

    Args:
        model: FingerprintNet in eval mode.
        windows: 2-D numpy float32 array (n, SEG).
        device: torch.device.
        batch: int windows per forward pass.

    Returns:
        2-D numpy float32 array (n, EMBED_DIM) of unit vectors.
    """
    out = []
    for i in range(0, len(windows), batch):
        x = torch.from_numpy(windows[i:i + batch]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, EMBED_DIM), np.float32)


class NNIndex:
    """
    Embedding index.

    Attributes:
        emb: 2-D numpy float16 array (n_windows, EMBED_DIM), unit rows.
        song_ids: 1-D numpy int32 array, song of each row.
        seg_idx: 1-D numpy int32 array, window number within its song (x DB_HOP).
        names: list of song names (str).
        signature: str identifying the songs + model the index was built from.
    """

    def __init__(self, emb, song_ids, seg_idx, names, signature):
        self.emb = emb
        self.song_ids = song_ids
        self.seg_idx = seg_idx
        self.names = names
        self.signature = signature
        self._matrix = None

    def matrix(self, device):
        """
        Index embeddings as a torch tensor on device (cached after first call).

        Args:
            device: torch.device.

        Returns:
            torch tensor (n_windows, EMBED_DIM); float16 on GPU, float32 on CPU.
        """
        if self._matrix is None or self._matrix.device.type != device.type:
            dtype = torch.float32 if device.type == "cpu" else torch.float16
            self._matrix = torch.from_numpy(self.emb).to(device=device, dtype=dtype)
        return self._matrix

    def save(self, path=INDEX_PATH):
        """
        Write the index to an .npz file.

        Args:
            path: destination pathlib.Path or str.

        """
        np.savez(path, emb=self.emb, song_ids=self.song_ids, seg_idx=self.seg_idx,
                 names=np.array(self.names), signature=np.array(self.signature))

    @classmethod
    def load(cls, path=INDEX_PATH):
        """
        Read an index written by save().

        Args:
            path: source pathlib.Path or str.

        Returns:
            NNIndex instance.
        """
        d = np.load(path, allow_pickle=False)
        return cls(d["emb"], d["song_ids"], d["seg_idx"], [str(n) for n in d["names"]],
                   str(d["signature"]))


def index_signature(song_signature):
    """
    Build a string that changes if the songs or the model change.

    Args:
        song_signature: str from config.folder_signature().

    Returns:
        str signature.
    """
    st = MODEL_PATH.stat()
    return f"{song_signature}|model:{st.st_size}:{int(st.st_mtime)}|hop:{DB_HOP}"


def build_index():
    """
    Embed every song in the library and save to INDEX_PATH.

    Returns:
        NNIndex instance.
    """
    lib = AudioLibrary.get()
    device = pick_device()
    model, _ = load_model(device)
    t0 = time.perf_counter()
    embs, ids, segs = [], [], []
    for i, name in enumerate(lib.names):
        e = embed_windows(model, frame_audio(lib.song(i), DB_HOP), device)
        embs.append(e.astype(np.float16))
        ids.append(np.full(len(e), i, np.int32))
        segs.append(np.arange(len(e), dtype=np.int32))
        print(f"  [{i + 1:>3}/{len(lib.names)}] {name:<45} {len(e):>5} windows")
    index = NNIndex(np.concatenate(embs), np.concatenate(ids), np.concatenate(segs),
                    lib.names, index_signature(lib.signature))
    index.save()
    print(f"Built index: {len(index.emb):,} windows x {EMBED_DIM}-d "
          f"({index.emb.nbytes / 1e6:.0f} MB) in {time.perf_counter() - t0:.1f}s")
    return index


def get_index():
    """
    Load the cached index, rebuilding it only if the songs or model changed.
    Does not touch the audio cache unless a rebuild is needed.

    Returns:
        NNIndex instance.
    """
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run: python neffex_nn.py train")
    if INDEX_PATH.exists():
        index = NNIndex.load()
        if index.signature == index_signature(folder_signature(list_songs())):
            return index
        print("Songs or model changed; rebuilding index.")
    return build_index()
