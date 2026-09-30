"""
Flask web interface for the NEFFEX song identifier.

Serves a single page with a record button and a Landmark / Neural switch.
The browser records the microphone with the Web Audio API and uploads a WAV
file; the server decodes it and runs either:
    landmark : neffex_id.py      (Wang 2003 spectrogram-peak hashing)
    neural   : nn_scripts/       (contrastive CNN embeddings)

The server never touches a microphone itself, so sounddevice/PortAudio are
not needed on the server.

Usage:
    python app.py                         # http://localhost:5000 (mic works on localhost)
    python app.py --host 0.0.0.0 --https  # self-signed HTTPS for testing on a LAN/server
    gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 app:app   # production, behind HTTPS proxy

Browsers only allow microphone access on HTTPS pages (or localhost), so a
deployed server must be served over HTTPS (e.g. nginx + Let's Encrypt).

Requirements:
    pip install flask numpy scipy librosa soundfile torch
    (pip install pyopenssl for --https)
"""

import argparse
import io
import threading
import time

import librosa
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request

import neffex_id as landmark
from nn_scripts import config as nn_config
from nn_scripts.index import NNIndex
from nn_scripts.index import get_index as nn_get_index
from nn_scripts.matching import identify as nn_identify
from nn_scripts.matching import is_confident as nn_is_confident
from nn_scripts.matching import score_threshold as nn_score_threshold
from nn_scripts.model import load_model

MIN_SECONDS = 1.0
RECORD_SECONDS = 8      # how long the browser listens before auto-stopping
MAX_SECONDS = 30.0
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# ---------------------------------------------------------------------------
# Engines: load both systems once and share them across requests
# ---------------------------------------------------------------------------
def load_landmark_index():
    """
    Load the landmark index. Uses the MP3 folder (auto-rebuild) when present,
    otherwise the cached neffex_index.npz, so a server only needs the index file.

    Args:
        None.

    Returns:
        neffex_id.Index instance.
    """
    if landmark.MUSIC_DIR.is_dir():
        return landmark.get_index()
    if landmark.INDEX_PATH.exists():
        return landmark.Index.load()
    raise FileNotFoundError(f"{landmark.INDEX_PATH.name} not found and no music folder to build it from")


def load_neural():
    """
    Load the neural model and index. Uses the MP3 folder (auto-rebuild) when
    present, otherwise the cached neffex_nn_index.npz.

    Args:
        None.

    Returns:
        Tuple (model, index, device).
    """
    device = nn_config.pick_device()
    model, _ = load_model(device)
    if nn_config.MUSIC_DIR.is_dir():
        index = nn_get_index()
    elif nn_config.INDEX_PATH.exists():
        index = NNIndex.load()
    else:
        raise FileNotFoundError(f"{nn_config.INDEX_PATH.name} not found and no music folder to build it from")
    return model, index, device


class Engines:
    """
    Lazily loaded, thread-safe holder for both identification systems.

    Attributes:
        landmark_index: neffex_id.Index or None if it failed to load.
        nn_model: FingerprintNet or None.
        nn_index: NNIndex or None.
        nn_device: torch.device or None.
        errors: dict mapping "landmark"/"neural" to a load error message.
    """

    def __init__(self):
        self._load_lock = threading.Lock()
        self._nn_lock = threading.Lock()
        self._loaded = False
        self.landmark_index = None
        self.nn_model = None
        self.nn_index = None
        self.nn_device = None
        self.errors = {}

    def load(self):
        """
        Load both systems once. Failures are recorded, not raised, so one
        missing system does not take the whole site down.

        Args:
            None.

        Returns:
            None.
        """
        with self._load_lock:
            if self._loaded:
                return
            try:
                self.landmark_index = load_landmark_index()
                print(f"Landmark ready: {len(self.landmark_index.names)} songs")
            except (Exception, SystemExit) as e:  # get_index() uses sys.exit on missing files
                self.errors["landmark"] = str(e)
                print(f"Landmark unavailable: {e}")
            try:
                self.nn_model, self.nn_index, self.nn_device = load_neural()
                print(f"Neural ready: {len(self.nn_index.names)} songs on {self.nn_device}")
            except (Exception, SystemExit) as e:
                self.errors["neural"] = str(e)
                print(f"Neural unavailable: {e}")
            self._loaded = True

    def available(self):
        """
        Report which systems loaded.

        Args:
            None.

        Returns:
            dict {"landmark": bool, "neural": bool}.
        """
        self.load()
        return {"landmark": self.landmark_index is not None, "neural": self.nn_index is not None}

    def identify(self, y, sr, mode):
        """
        Run one system on a clip.

        Args:
            y: 1-D numpy float32 array of mono audio.
            sr: int sample rate of y.
            mode: str, "landmark" or "neural".

        Returns:
            Tuple (matches, confident, threshold): list of Match objects from
            the chosen system, bool whether the top match passes that system's
            thresholds, and float score threshold that applied.
        """
        self.load()
        if mode == "landmark":
            clip = librosa.resample(y, orig_sr=sr, target_sr=landmark.SAMPLE_RATE)
            matches = landmark.identify(clip, self.landmark_index)
            return matches, landmark.is_confident(matches), float(landmark.MIN_MATCH_SCORE)
        clip = librosa.resample(y, orig_sr=sr, target_sr=nn_config.SAMPLE_RATE)
        with self._nn_lock:  # one forward pass at a time on the shared model
            matches = nn_identify(clip, self.nn_model, self.nn_index, self.nn_device)
        threshold = nn_score_threshold(matches[0].n_windows) if matches else nn_config.MIN_MATCH_SCORE
        return matches, nn_is_confident(matches), float(threshold)


engines = Engines()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decode_upload(data):
    """
    Decode an uploaded WAV/FLAC/OGG file to mono float32.

    Args:
        data: bytes of the uploaded file.

    Returns:
        Tuple (y, sr): 1-D numpy float32 array and int sample rate.
    """
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return y.mean(axis=1), int(sr)


def match_to_dict(m):
    """
    Convert a Match from either system to JSON-safe values.

    Args:
        m: neffex_id.Match or nn_scripts.matching.Match.

    Returns:
        dict with name, score, confidence, offset_sec.
    """
    return {"name": m.name, "score": round(float(m.score), 2),
            "confidence": round(min(float(m.confidence), 99.0), 2),
            "offset_sec": round(max(float(m.offset_sec), 0.0), 1)}


def error(message, status=400):
    """
    Build a JSON error response.

    Args:
        message: str shown to the user.
        status: int HTTP status code.

    Returns:
        Flask response tuple.
    """
    return jsonify({"error": message}), status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.after_request
def security_headers(response):
    """
    Allow the microphone for this origin only (Permissions-Policy).

    Args:
        response: flask.Response.

    Returns:
        flask.Response with headers added.
    """
    response.headers["Permissions-Policy"] = "microphone=(self)"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/")
def index():
    """
    Serve the single-page interface.

    Args:
        None.

    Returns:
        Rendered templates/index.html.
    """
    return render_template("index.html", min_seconds=MIN_SECONDS, record_seconds=RECORD_SECONDS)


@app.get("/api/status")
def status():
    """
    Report which systems are available and how many songs they know.

    Returns:
        JSON {"available": {...}, "songs": {...}, "errors": {...}}.
    """
    avail = engines.available()
    songs = {"landmark": len(engines.landmark_index.names) if avail["landmark"] else 0,
             "neural": len(engines.nn_index.names) if avail["neural"] else 0}
    return jsonify({"available": avail, "songs": songs, "errors": engines.errors})


@app.post("/api/identify")
def identify():
    """
    Identify an uploaded recording.

    Args (multipart form):
        audio: WAV file recorded by the browser.
        mode: "landmark" or "neural".

    Returns:
        JSON with match (bool), best, alternatives, threshold, timing, and
        input diagnostics (duration, sample rate, peak level).
    """
    mode = request.form.get("mode", "landmark")
    if mode not in ("landmark", "neural"):
        return error("mode must be 'landmark' or 'neural'")
    if not engines.available()[mode]:
        return error(f"The {mode} system is not available on this server: "
                     f"{engines.errors.get(mode, 'unknown error')}", 503)
    upload = request.files.get("audio")
    if upload is None:
        return error("No audio file received")

    try:
        y, sr = decode_upload(upload.read())
    except Exception:
        return error("Could not decode the audio. Upload a WAV file.")
    duration = len(y) / sr
    if duration < MIN_SECONDS:
        return error(f"Recording too short ({duration:.1f}s). Record at least {MIN_SECONDS:g}s.")
    if duration > MAX_SECONDS:
        y = y[:int(MAX_SECONDS * sr)]
        duration = MAX_SECONDS
    peak = float(np.abs(y).max()) if len(y) else 0.0

    t0 = time.perf_counter()
    matches, confident, threshold = engines.identify(y, sr, mode)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return jsonify({
        "mode": mode,
        "match": bool(confident),
        "best": match_to_dict(matches[0]) if matches else None,
        "alternatives": [match_to_dict(m) for m in matches[1:]],
        "threshold": round(threshold, 2),
        "elapsed_ms": round(elapsed_ms),
        "input": {"duration_sec": round(duration, 2), "sample_rate": sr, "peak": round(peak, 4),
                  "too_quiet": peak < 0.01},
    })


@app.errorhandler(413)
def too_large(_):
    """
    Handle uploads over MAX_UPLOAD_BYTES.

    Args:
        _: the werkzeug exception (unused).

    Returns:
        JSON error response.
    """
    return error(f"Upload too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)", 413)


def main():
    """
    Run the development server.

    Args:
        None (reads sys.argv).
    """
    parser = argparse.ArgumentParser(description="NEFFEX identifier web interface")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other devices")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--https", action="store_true",
                        help="self-signed certificate (needs pyopenssl); browsers block the mic on plain HTTP")
    args = parser.parse_args()

    engines.load()  # load before the first request, not during it
    if args.host not in ("127.0.0.1", "localhost") and not args.https:
        print("Warning: browsers only allow the microphone on HTTPS or localhost. "
              "Use --https or put the app behind an HTTPS proxy.")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False,
            ssl_context="adhoc" if args.https else None)


if __name__ == "__main__":
    main()
