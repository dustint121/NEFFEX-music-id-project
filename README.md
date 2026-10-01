# NEFFEX Music Identifier

A Shazam-style music recognition app for the songs of [NEFFEX](https://www.neffexmusic.com). Hold up a microphone to a song, and the app names it, shows where in the song the clip starts, and links to the track on YouTube, Spotify and SoundCloud.

The project implements and compares two identification methods side by side:

| | Landmark (classic) | Neural (deep learning) |
|---|---|---|
| Idea | Hash pairs of spectrogram peaks ([Wang, 2003](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)) | Learn a 128-d fingerprint per 1 s of audio with contrastive learning ([Chang et al., 2021](https://arxiv.org/abs/2010.11910)) |
| Code | `neffex_id.py` | `neffex_nn.py` + `nn_scripts/` |
| Setup time (250 songs, CPU) | Under 10 minutes (index build, no training) | About 2 hours (training) + index build |

Both run from the command line or through a Flask web app with a single "Listen" button and a Landmark / Neural switch. A live deployment runs at [neffex-id.dustintran.dev](https://neffex-id.dustintran.dev/).

---


## The dataset: NEFFEX's first 250 songs

NEFFEX numbers the copyright-free releases in order (for example "Go! ... No.147" on YouTube). This project uses **songs No. 1 to No. 250**.

- **Audio:** one MP3 per song in `Neffex mp3/`, named with the release number, e.g. `87_Take Me Back Again.mp3`. The MP3s are not included in this repository
- **Metadata:** `song_catalog.json` maps each release number to its title and links:

  ```json
  {
    "song_number": 87,
    "song_name": "Take Me Back Again",
    "soundcloud_url": "https://soundcloud.com/neffexmusic/take-me-back-again",
    "youtube_url": "https://www.youtube.com/watch?v=Pdj_RtBHkKo",
    "spotify_url": "https://open.spotify.com/track/1EJ5sjEx4asa3acWZuolmV"
  }
  ```

  The web app reads the number at the start of each indexed name, looks it up here, and shows the real title, the YouTube thumbnail and the three source links. Five songs (131, 132, 133, 142, 144) have no Spotify link, so the app simply hides that button.

---

## How the two methods work

### 1. Landmark fingerprinting (`neffex_id.py`)

The method behind the original Shazam ([Wang, 2003](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)).

1. **Spectrogram:** audio is resampled to 11,025 Hz and turned into a log-magnitude spectrogram (2048-point FFT, 512-sample hop, about 46 ms per frame).
2. **Peaks ("constellation map"):** local maxima are found with a 25 x 15 maximum filter, capped at 30 peaks per second. Peaks survive noise and EQ much better than the full spectrogram.
3. **Hashes:** each peak (anchor) is paired with up to 10 later peaks in a target zone (1 to 63 frames ahead, within +/-200 bins). Each pair becomes a 32-bit hash: `f1 << 17 | f2 << 6 | dt`.
4. **Index:** every hash is stored with its song ID and time, in sorted NumPy arrays searched with `searchsorted` (`neffex_index.npz`).
5. **Matching:** a recording's hashes are looked up, and each hit votes for (song, time offset). The right song has many hits sharing **one** offset; random hits scatter. The score is the size of the largest aligned group.

For the 250 songs the index holds 13.7 million hashes (137 MB).

### 2. Neural fingerprinting (`neffex_nn.py` + `nn_scripts/`)

Based on the contrastive approach of [Chang et al., 2021](https://arxiv.org/abs/2010.11910), the same family of technique as Google's on-device "Now Playing" ([Google Research](https://research.google/blog/googles-next-generation-music-recognition/)).

1. **Input:** audio at 8 kHz, cut into 1 s windows, converted to a 64-band log-mel spectrogram. Each window is normalized to zero mean and unit variance (gain-invariant).
2. **Model:** a small CNN (1.32 M parameters) maps each window to a unit-length 128-d vector.
3. **Training:** each step takes 256 random windows and makes a degraded copy of each (room reverb, background music from other songs, phone-style band-pass, white or pink noise at -3 to 20 dB SNR, gain changes, and a small time shift). The NT-Xent contrastive loss from [SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709) pulls each window toward its degraded copy and away from the other 255. My model trained for 3,000 steps (AdamW, one-cycle LR).
4. **Index:** every song is embedded every 0.25 s (`neffex_nn_index.npz`, 47 MB).
5. **Matching:** the recording is embedded every 0.5 s. Each window finds its 5 nearest index windows by cosine similarity, and the same time-offset voting as the landmark method picks the song. The score is the summed similarity of the aligned windows.
6. **Noise guard:** windows that are near-silent, or look more like noise than music, cast no votes (`nn_scripts/noise.py`).

---

## Results

All tests below ran on the full 250-song library. Recordings were random excerpts with added noise and phone-style filtering (`python neffex_nn.py compare`).

### Accuracy vs. clip length (0 dB SNR)

| Clip length | Landmark | Neural |
|---|---|---|
| 1 s | **140/250** | 108/250 |
| 1.5 s | 209/250 | **232/250** |
| 2 s | **236/250** | 232/250 |
| 3 s | **245/250** | 241/250 |
| 5 s | **249/250** | 248/250 |
| 8 s | 250/250 | 250/250 |

### Accuracy vs. noise (5 s clips)

| SNR | Landmark | Neural |
|---|---|---|
| 5 dB | 249/250 | 249/250 |
| -5 dB | **247/250** | 240/250 |
| -8 dB | **223/250** | 117/250 |
| -10 dB | **177/250** | 31/250 |

### Speed and size

| | Landmark | Neural |
|---|---|---|
| Median match time (5 s clip) | 10 to 15 ms | 18 to 38 ms |
| Index size | 137 MB | 47 MB (+ about 5 MB model) |
| Growth per song | about 0.55 MB | about 0.19 MB |

### False matches (211 clips of 10 non-NEFFEX songs, x3 lengths)

| Cutoffs | Landmark | Neural |
|---|---|---|
| Original (8 / 2.0, confidence 1.5) | 8 to 9 of 211 per length (about 4%) | 2 to 3 of 211 (about 1%) |
| Tuned (12.5 / 2.35, confidence 1.0) | **0 of 633** | **0 of 633** |

Real NEFFEX clips accepted with the tuned cutoffs: 594/600 (landmark) and 596/600 (neural).

---

## Pros and cons

### Landmark

**Pros**
- **Fast setup:** no training. Indexing all 250 songs takes under 10 minutes on a CPU, and adding a song only means hashing that song.
- **More robust to noise:** still 177/250 at -10 dB, where the neural model gets 31/250.
- **Wider safety margin:** in testing, a real 8 s match typically scored in the hundreds (about 160 to 400) against a cutoff of 12.5, while non-NEFFEX music stayed at or below 12. Right and wrong answers are far apart, so the cutoff is easy to set.
- **Silence-safe:** silence and steady noise produce almost no peaks, so they never match.
- **Explainable:** every match can be traced back to the exact peak pairs that agreed.

**Cons**
- **Large index:** 137 MB for 250 songs, about 3x the neural index.
- **Needs a few seconds** of audio to collect enough aligned hashes (weaker at 1.5 s).
- **Hand-tuned:** the peak picking and target zone are fixed rules, not learned.

### Neural (as currently implemented)

**Pros**
- **Smaller index:** 47 MB, about 3x smaller, and it grows more slowly per song.
- **Better on very short clips:** 232/250 at 1.5 s, against 209/250 for landmark.
- **Learned robustness:** handles the kinds of distortion it was trained on (filtering, reverb, moderate noise) without hand-written rules.
- **Room to improve:** a GPU, a bigger model, more training steps or harder augmentation could all raise accuracy, and the same model can index new songs without retraining.

**Cons**
- **Slow setup:** about 2 hours of training on a CPU for 3,000 steps, before the index can even be built.
- **More sensitive to noise:** accuracy collapses below about -5 dB SNR (117/250 at -8 dB, 31/250 at -10 dB). Training with more noise (-12 to 20 dB) helped at -10 dB but cost accuracy on cleaner clips.
- **Small margin between right and wrong:** in testing, a real 8 s match typically scored about 5 to 10, while non-NEFFEX music and noise reached about 2.5, against a cutoff of 2.35. The maximum possible score is only the number of windows (1.0 each), so there is little room between the two. Similar NEFFEX songs (same production style) also sit closer together, so the runner-up is often near the winner. A small shift in recording conditions can push a wrong song over the cutoff, so the thresholds need careful calibration.
- **No "not music" concept:** trained only on music, the model maps silence and background noise close to real song windows. Before the noise guard was added, the live site named a song for 11 of 48 no-music recordings.
- **Slower per query** (CNN forward pass), though still far below what a user notices.

**Bottom line:** with the current implementation, landmark is the more reliable default, especially in noisy rooms. The neural method wins on index size and very short clips, and is the more interesting one to keep improving.

---

## Project layout

```
neffex_id/
├── neffex_id.py            # landmark method: build / listen / file / selftest
├── neffex_nn.py            # neural method CLI: train / build / listen / file / selftest / compare
├── nn_scripts/
│   ├── config.py           # paths, audio settings, model and matching settings
│   ├── audio_library.py    # loads and caches the 8 kHz song audio
│   ├── model.py            # LogMel front end, FingerprintNet CNN, NT-Xent loss
│   ├── augment.py          # training degradations (noise, filtering, reverb, shift)
│   ├── training.py         # training loop (AdamW, one-cycle LR, AMP on CUDA)
│   ├── index.py            # embedding index build / load
│   ├── matching.py         # nearest-neighbour search + offset voting
│   ├── noise.py            # silence / noise guard
│   └── evaluation.py       # selftest and compare
├── app.py                  # Flask web server (both methods behind /api/identify)
├── templates/index.html    # web page
├── static/                 # app.js, recorder-worklet.js, style.css, favicon.ico
├── song_catalog.json       # titles + YouTube / Spotify / SoundCloud links
├── make_fp_clips.py        # cut non-NEFFEX songs into test clips
├── false_positive_test.py  # false-match test + threshold suggestions
├── noise_test.py           # silence / background-noise test (local or a live URL)
└── Neffex mp3/             # the 250 songs (not included)
```

Generated files (not all in the repository): `neffex_index.npz`, `neffex_nn_model.pt`, `neffex_nn_index.npz`, `neffex_audio_8k.npy`, `neffex_audio_8k_meta.npz`.

---

## Usage

### Install

```bash
pip install numpy scipy librosa soundfile torch flask
pip install sounddevice      # only for the command-line "listen" commands
# Linux: sudo apt install libportaudio2  (needed by sounddevice)
```

### Landmark method

```bash
python neffex_id.py build                 # index "Neffex mp3/" (under 10 minutes)
python neffex_id.py listen --seconds 8    # record from the mic and identify
python neffex_id.py file clip.mp3         # identify an audio file
python neffex_id.py selftest --seconds 5 --snr 0
```

### Neural method

```bash
python neffex_nn.py train --steps 3000 --batch 256   # about 2 hours on CPU, much less on a GPU
python neffex_nn.py build                            # embed the library
python neffex_nn.py listen --seconds 8
python neffex_nn.py file clip.mp3
python neffex_nn.py compare --trials 250 --seconds 5 --snr 0   # both methods side by side
```

### Web app

```bash
python app.py                          # http://localhost:5000
python app.py --host 0.0.0.0 --https   # self-signed HTTPS for phones on your network (pip install pyopenssl)
gunicorn -w 1 --threads 4 -b 127.0.0.1:8000 app:app   # production, behind nginx + HTTPS
```

- **HTTPS is required.** Browsers only allow the microphone on HTTPS pages (or `localhost`), so a deployed server needs a certificate.
- **Use one worker.** Both indexes and the model are held in memory; each extra worker would load its own copy.
- **The server never touches a microphone.** The browser records and uploads a WAV file, so the server needs neither sounddevice nor ffmpeg.
- **The server only needs** `neffex_index.npz`, `neffex_nn_index.npz`, `neffex_nn_model.pt` and `song_catalog.json`. The MP3 folder is optional.
- **Browser audio processing is turned off.** Echo cancellation, noise suppression and auto gain are designed for voices and damage music.
- **Phones work** in Safari and Chrome over HTTPS. Play the song from another device; phones usually can't record what they are playing themselves.

### Testing

```bash
python make_fp_clips.py                    # cut test_mp3/ into 8 s clips -> fp_clips/
python false_positive_test.py              # false matches + suggested cutoffs (3/5/8 s, 5 dB noise)
python false_positive_test.py --seconds 1.5 3 8 --snr 0
python noise_test.py                       # silence / noise test on local models
python noise_test.py --url https://neffex-id.dustintran.dev   # same test against a live server
```

### Key settings

| File | Setting | Value | Meaning |
|---|---|---|---|
| `neffex_id.py` | `MIN_MATCH_SCORE` | 12.5 | aligned hash matches needed |
| `neffex_id.py` | `MIN_CONFIDENCE` | 1.0 | winner / runner-up ratio (1.0 = off) |
| `nn_scripts/config.py` | `MIN_MATCH_SCORE` | 2.35 | summed aligned similarity needed (lower for clips under 2.5 s) |
| `nn_scripts/config.py` | `MIN_CONFIDENCE` | 1.0 | winner / runner-up ratio |
| `nn_scripts/config.py` | `TRAIN_SNR_DB` | (-3, 20) | training noise range |
| `nn_scripts/config.py` | `SILENCE_DBFS` | -60 | quieter windows are ignored |
| `nn_scripts/config.py` | `NOISE_MARGIN` | 0.10 | how much more a window must look like music than noise |
| `nn_scripts/config.py` | `DB_NOISE_SIM` | 0.97 | silent stretches of the index are never matched |

Re-run `false_positive_test.py` after changing the library or retraining; the best cutoffs depend on both.

---

## Lessons learned

- **An easy test hides differences.** At 5 s and 5 dB both methods scored 249/250. Only shorter and noisier clips separated them.
- **A fixed threshold breaks short clips.** The neural score is a sum over windows, so a 1.5 s clip (2 windows) could never reach a cutoff of 2.0. It scored 0/250 until the cutoff was capped at 0.67 per window.
- **Measure false matches, not just accuracy.** The original cutoffs named the wrong song for about 4% (landmark) and 1% (neural) of non-NEFFEX clips. Wrongly naming a song is worse than missing one.
- **Normalization has a cost.** Making every window gain-invariant also turns a muted mic's faint hiss into "loud" input. Combined with music-only training, silence matched silent intros in the index perfectly. Without retraining, the fix drops quiet windows, compares each window against fingerprints of synthetic noise, and masks silent index windows. No-music false matches on the 12-song test library went to 0, with real-match acceptance almost unchanged.

---

## References

- A. Wang, "An Industrial-Strength Audio Search Algorithm," ISMIR 2003. [PDF](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)
- S. Chang et al., "Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrastive Learning," ICASSP 2021. [arXiv:2010.11910](https://arxiv.org/abs/2010.11910)
- T. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR, NT-Xent loss), ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Google Research, "Google's Next-Generation Music Recognition." [Blog](https://research.google/blog/googles-next-generation-music-recognition/)
- Apple, ShazamKit documentation. [developer.apple.com](https://developer.apple.com/documentation/shazamkit)
- MDN, `MediaDevices.getUserMedia()`. [developer.mozilla.org](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)

## Credits

All music by [NEFFEX](https://www.neffexmusic.com) ([YouTube](https://www.youtube.com/channel/UCBefBxNTPoNCQBU_Lta6Nvg)). This is an unofficial fan and portfolio project and is not affiliated with NEFFEX.
