"""
Flask web interface for the NEFFEX Music Identifier.

Serves a single page with a record button and a Landmark / Neural switch.
The browser records the microphone with the Web Audio API and uploads a WAV
file; the server decodes it and runs either:
    landmark : neffex_id.py      (Wang 2003 spectrogram-peak hashing)
    neural   : nn_scripts/       (contrastive CNN embeddings)

Song titles and SoundCloud / YouTube / Spotify links come from
song_catalog.json, matched by the number prefix of each song name
("87_Take Me Back Again" -> song_number 87).

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
import json
import re
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request, send_from_directory

import neffex_id as landmark
from nn_scripts import config as nn_config
from nn_scripts.index import NNIndex
from nn_scripts.index import get_index as nn_get_index
from nn_scripts.matching import identify as nn_identify
from nn_scripts.matching import is_confident as nn_is_confident
from nn_scripts.matching import score_threshold as nn_score_threshold
from nn_scripts.model import load_model

ROOT_DIR = Path(__file__).resolve().parent
CATALOG_PATH = ROOT_DIR / "song_catalog.json"
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
# Song catalog: song_catalog.json maps the number prefix of each MP3/index
# name ("87_Take Me Back Again") to its title and SoundCloud/YouTube/Spotify links
# ---------------------------------------------------------------------------
YOUTUBE_ID = re.compile(r"(?:v=|youtu\.be/|/embed/|/shorts/)([\w-]{11})")
NUMBER_PREFIX = re.compile(r"^\s*(\d+)\s*[_\-. ]\s*(.*)$")


def normalize_title(name):
    """
    Lower-case a song title and drop an "NEFFEX - " prefix and punctuation.

    Args:
        name: str title, e.g. "NEFFEX - Take Me Back".

    Returns:
        str key, e.g. "take me back".
    """
    name = re.sub(r"^\s*neffex\s*[-_:]\s*", "", name, flags=re.IGNORECASE)
    return " ".join(re.sub(r"[^\w\s]", " ", name.lower()).split())


class Catalog:
    """
    song_catalog.json loaded into a dict keyed by song_number.
    Reloads automatically when the file changes on disk.

    Attributes:
        path: pathlib.Path of the JSON file.
        songs: dict {int song_number: dict entry from the JSON}.
        by_title: dict {normalized song_name: dict entry}, used when a name
            has no number prefix (e.g. "NEFFEX - Cold").
    """

    def __init__(self, path=CATALOG_PATH):
        self.path = path
        self.songs = {}
        self.by_title = {}
        self._mtime = None
        self._lock = threading.Lock()

    def _refresh(self):
        """
        Re-read the JSON file if it is new or has been modified.

        Args:
            None.

        Returns:
            None.
        """
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self.songs, self._mtime = {}, None
            return
        if mtime == self._mtime:
            return
        with self._lock:
            with open(self.path, encoding="utf-8") as f:
                entries = json.load(f)
            self.songs = {int(e["song_number"]): e for e in entries if "song_number" in e}
            self.by_title = {normalize_title(e.get("song_name", "")): e for e in entries}
            self._mtime = mtime
            print(f"Catalog loaded: {len(self.songs)} songs from {self.path.name}")

    def lookup(self, index_name):
        """
        Resolve an index/MP3 name to display info.

        Args:
            index_name: str song name as stored in the index, e.g. "87_Take Me Back Again".

        Returns:
            dict with id (int or None), title (str), links (dict of
            soundcloud/youtube/spotify URLs, only those that exist), and
            thumbnail (YouTube thumbnail URL or None).
        """
        self._refresh()
        m = NUMBER_PREFIX.match(index_name)
        number = int(m.group(1)) if m else None
        fallback_title = m.group(2) if m and m.group(2) else index_name
        entry = self.songs.get(number) if number is not None else None
        if entry is None:
            entry = self.by_title.get(normalize_title(fallback_title))
            number = entry.get("song_number") if entry else number
        if entry is None:
            return {"id": number, "title": fallback_title, "links": {}, "thumbnail": None}

        links = {}
        for key in ("soundcloud", "youtube", "spotify"):
            url = entry.get(f"{key}_url")
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                links[key] = url
        thumbnail = None
        yt = YOUTUBE_ID.search(links.get("youtube", ""))
        if yt:
            thumbnail = f"https://i.ytimg.com/vi/{yt.group(1)}/hqdefault.jpg"
        return {"id": number, "title": entry.get("song_name") or fallback_title,
                "links": links, "thumbnail": thumbnail}


catalog = Catalog()


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
    Convert a Match from either system to JSON-safe values, with catalog info.

    Args:
        m: neffex_id.Match or nn_scripts.matching.Match.

    Returns:
        dict with name (index name), id, title, links, thumbnail, score,
        confidence, offset_sec.
    """
    return {"name": m.name, **catalog.lookup(m.name), "score": round(float(m.score), 2),
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


@app.get("/favicon.ico")
def favicon():
    """
    Browser tab icon (static/favicon.ico). Browsers also request /favicon.ico
    directly, so serve it at the root as well as through the <link> tag.

    Returns:
        flask.Response with the .ico file, cached for a day.
    """
    return send_from_directory(app.static_folder, "favicon.ico",
                               mimetype="image/vnd.microsoft.icon", max_age=86400)


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
    catalog.lookup("")  # load song_catalog.json and report it
    if args.host not in ("127.0.0.1", "localhost") and not args.https:
        print("Warning: browsers only allow the microphone on HTTPS or localhost. "
              "Use --https or put the app behind an HTTPS proxy.")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False,
            ssl_context="adhoc" if args.https else None)


if __name__ == "__main__":
    main()
