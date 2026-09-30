/*
 * microphone capture, WAV encoding, and upload for the NEFFEX identifier.
 *
 * Flow: button tap -> getUserMedia (browser asks for mic permission)
 *       -> AudioWorklet collects raw samples -> 16-bit WAV -> POST /api/identify
 */
(() => {
  "use strict";

  const { minSeconds, recordSeconds, workletUrl } = window.APP_CONFIG;
  const MODE_KEY = "neffex-mode";

  const els = {
    button: document.getElementById("record"),
    label: document.getElementById("record-label"),
    progress: document.getElementById("progress"),
    level: document.getElementById("level"),
    modeSwitch: document.getElementById("mode-switch"),
    modeHint: document.getElementById("mode-hint"),
    modeNames: document.querySelectorAll(".mode-name"),
    status: document.getElementById("status"),
    library: document.getElementById("library"),
    result: document.getElementById("result"),
    kicker: document.getElementById("result-kicker"),
    title: document.getElementById("result-title"),
    meta: document.getElementById("result-meta"),
    alternatives: document.getElementById("alternatives"),
  };

  const CIRCUMFERENCE = 2 * Math.PI * 54;
  els.progress.style.strokeDasharray = `${CIRCUMFERENCE}`;
  els.progress.style.strokeDashoffset = `${CIRCUMFERENCE}`;

  let state = "idle";     // idle | requesting | recording | uploading
  let session = null;     // { stream, ctx, node, chunks, sampleRate, startedAt, timer, raf }
  let available = { landmark: true, neural: true };

  // -------------------------------------------------------------------------
  // UI helpers
  // -------------------------------------------------------------------------

  /*
   * Update the status line.
   * Args: message - string; kind - "" | "error" | "warn".
   * Returns: nothing.
   */
  function setStatus(message, kind = "") {
    els.status.textContent = message;
    els.status.className = `status ${kind}`.trim();
  }

  /*
   * Switch the button between idle / requesting / recording / uploading looks.
   * Args: next - state string. Returns: nothing.
   */
  function setState(next) {
    state = next;
    document.body.dataset.state = next;
    const labels = { idle: "Listen", requesting: "Allow mic", recording: "Stop", uploading: "Matching" };
    els.label.textContent = labels[next];
    els.button.setAttribute("aria-pressed", String(next === "recording"));
    els.button.setAttribute("aria-label", next === "recording" ? "Stop listening" : "Start listening");
    els.button.disabled = next === "requesting" || next === "uploading";
    els.modeSwitch.disabled = next !== "idle";
    if (next !== "recording") {
      els.progress.style.strokeDashoffset = `${CIRCUMFERENCE}`;
      els.level.style.transform = "scale(0)";
    }
  }

  /*
   * Current recognition mode from the switch.
   * Args: none. Returns: "landmark" or "neural".
   */
  function currentMode() {
    return els.modeSwitch.checked ? "neural" : "landmark";
  }

  /*
   * Highlight the active mode name and show a one-line description.
   * Args: none. Returns: nothing.
   */
  function renderMode() {
    const mode = currentMode();
    els.modeNames.forEach((el) => el.classList.toggle("active", el.dataset.mode === mode));
    els.modeHint.textContent = mode === "neural"
      ? "CNN embeddings trained with contrastive learning"
      : "Spectrogram peak hashing (Wang 2003, Shazam)";
    localStorage.setItem(MODE_KEY, mode);
  }

  /*
   * Format seconds as m:ss.s.
   * Args: seconds - number. Returns: string.
   */
  function formatOffset(seconds) {
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1).padStart(4, "0");
    return `${m}:${s}`;
  }

  /*
   * Show the server's answer.
   * Args: data - JSON from /api/identify. Returns: nothing.
   */
  function showResult(data) {
    els.result.hidden = false;
    els.result.classList.toggle("no-match", !data.match);
    els.alternatives.replaceChildren();
    const method = data.mode === "neural" ? "Neural" : "Landmark";

    if (data.match) {
      const b = data.best;
      els.kicker.textContent = "Match found";
      els.title.textContent = b.name;
      els.meta.textContent = `${method} · starts ~${formatOffset(b.offset_sec)} · score ${b.score} ` +
        `(needs ${data.threshold}) · ${b.confidence}x runner-up · ${data.elapsed_ms} ms`;
      data.alternatives.forEach((alt) => {
        const li = document.createElement("li");
        li.textContent = `${alt.name} (score ${alt.score})`;
        els.alternatives.appendChild(li);
      });
      setStatus("Tap again to identify another song.");
    } else {
      els.kicker.textContent = "No confident match";
      els.title.textContent = data.best ? `Closest: ${data.best.name}` : "Nothing matched";
      els.meta.textContent = data.best
        ? `${method} · score ${data.best.score} (needs ${data.threshold}) · ${data.best.confidence}x runner-up`
        : `${method} · no fingerprints matched`;
      const tip = data.input.too_quiet
        ? "The recording was very quiet. Move closer to the speaker or check the input device."
        : "Try a longer recording, closer to the speaker, or switch methods.";
      setStatus(tip, data.input.too_quiet ? "warn" : "");
    }
  }

  // -------------------------------------------------------------------------
  // Microphone permission
  // -------------------------------------------------------------------------

  /*
   * Explain why the microphone cannot be used at all on this page.
   * Args: none. Returns: string message, or "" if the mic API is usable.
   */
  function micUnsupportedReason() {
    if (!window.isSecureContext) {
      return "Microphone access requires HTTPS. Open this site with https:// " +
        "(or on localhost during development).";
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return "This browser does not support microphone recording.";
    }
    if (!window.AudioWorkletNode) {
      return "This browser does not support AudioWorklet. Please use a current Chrome, Edge, Firefox, or Safari.";
    }
    return "";
  }

  /*
   * Translate getUserMedia errors into instructions.
   * Args: err - DOMException from getUserMedia. Returns: string.
   */
  function micErrorMessage(err) {
    switch (err && err.name) {
      case "NotAllowedError":
      case "SecurityError":
        return "Microphone access was blocked. Click the lock or site-settings icon in the address bar, " +
          "allow the microphone, then tap Listen again.";
      case "NotFoundError":
      case "OverconstrainedError":
        return "No microphone was found. Connect one and try again.";
      case "NotReadableError":
      case "AbortError":
        return "The microphone is in use by another app or could not be started.";
      default:
        return `Could not access the microphone (${err && err.name ? err.name : "unknown error"}).`;
    }
  }

  /*
   * Show a hint up front if the user previously blocked the mic.
   * Uses the Permissions API where supported (not all browsers support "microphone").
   * Args: none. Returns: Promise<void>.
   */
  async function watchPermission() {
    if (!navigator.permissions || !navigator.permissions.query) return;
    try {
      const perm = await navigator.permissions.query({ name: "microphone" });
      const update = () => {
        if (perm.state === "denied" && state === "idle") {
          setStatus("Microphone access is blocked for this site. Allow it in the address bar's site settings.", "warn");
        } else if (perm.state !== "denied" && els.status.classList.contains("warn") && state === "idle") {
          setStatus("Tap the button and hold your device near the music.");
        }
      };
      update();
      perm.onchange = update;
    } catch (_) {
      // Permission name not supported in this browser; getUserMedia will still prompt.
    }
  }

  // -------------------------------------------------------------------------
  // Recording
  // -------------------------------------------------------------------------

  /*
   * Ask for the microphone and start collecting samples.
   * Args: none. Returns: Promise<void>.
   */
  async function startRecording() {
    const reason = micUnsupportedReason();
    if (reason) {
      setStatus(reason, "error");
      return;
    }

    setState("requesting");
    setStatus("Waiting for microphone permission...");
    let stream;
    try {
      // Voice-call processing (echo cancellation, noise suppression, auto gain)
      // is designed for speech and damages music, so it is turned off.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (err) {
      setState("idle");
      setStatus(micErrorMessage(err), "error");
      return;
    }

    try {
      const ctx = new AudioContext();
      await ctx.audioWorklet.addModule(workletUrl);
      const source = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, "recorder");
      const mute = ctx.createGain();
      mute.gain.value = 0;  // keep the graph running without playing the mic back
      source.connect(node).connect(mute).connect(ctx.destination);
      if (ctx.state === "suspended") await ctx.resume();

      session = { stream, ctx, node, chunks: [], sampleRate: ctx.sampleRate, startedAt: performance.now() };
      node.port.onmessage = (event) => {
        if (event.data.samples) {
          session.chunks.push(event.data.samples);
          const level = Math.min(1, event.data.rms * 8);
          els.level.style.transform = `scale(${0.35 + level * 0.65})`;
        }
      };

      setState("recording");
      els.result.hidden = true;
      setStatus(`Listening... tap to stop (${recordSeconds}s max).`);
      session.timer = setTimeout(stopRecording, recordSeconds * 1000);
      tick();
    } catch (err) {
      stream.getTracks().forEach((t) => t.stop());
      setState("idle");
      setStatus(`Could not start audio processing: ${err.message || err}`, "error");
    }
  }

  /*
   * Animate the progress ring while recording.
   * Args: none. Returns: nothing.
   */
  function tick() {
    if (state !== "recording" || !session) return;
    const elapsed = (performance.now() - session.startedAt) / 1000;
    const fraction = Math.min(1, elapsed / recordSeconds);
    els.progress.style.strokeDashoffset = `${CIRCUMFERENCE * (1 - fraction)}`;
    session.raf = requestAnimationFrame(tick);
  }

  /*
   * Stop the mic, gather the samples, and send them to the server.
   * Args: none. Returns: Promise<void>.
   */
  async function stopRecording() {
    if (state !== "recording" || !session) return;
    const s = session;
    const elapsed = (performance.now() - s.startedAt) / 1000;
    if (elapsed < minSeconds) {
      setStatus(`Keep listening a little longer (at least ${minSeconds}s).`, "warn");
      return;
    }
    clearTimeout(s.timer);
    cancelAnimationFrame(s.raf);
    setState("uploading");
    setStatus("Matching...");

    // Ask the worklet for its last partial chunk, then shut everything down
    await new Promise((resolve) => {
      const done = setTimeout(resolve, 500);
      const previous = s.node.port.onmessage;
      s.node.port.onmessage = (event) => {
        previous(event);
        if (event.data.done) { clearTimeout(done); resolve(); }
      };
      s.node.port.postMessage("flush");
    });
    s.stream.getTracks().forEach((t) => t.stop());  // turns off the browser's mic indicator
    await s.ctx.close();
    session = null;

    const wav = encodeWav(s.chunks, s.sampleRate);
    await upload(wav);
  }

  /*
   * Encode Float32 chunks as a mono 16-bit PCM WAV file.
   * Args: chunks - array of Float32Array; sampleRate - number.
   * Returns: Blob of type audio/wav.
   */
  function encodeWav(chunks, sampleRate) {
    const length = chunks.reduce((n, c) => n + c.length, 0);
    const buffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(buffer);
    const writeString = (offset, text) => {
      for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
    };
    writeString(0, "RIFF");
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);          // fmt chunk size
    view.setUint16(20, 1, true);           // PCM
    view.setUint16(22, 1, true);           // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);           // block align
    view.setUint16(34, 16, true);          // bits per sample
    writeString(36, "data");
    view.setUint32(40, length * 2, true);
    let offset = 44;
    for (const chunk of chunks) {
      for (let i = 0; i < chunk.length; i++, offset += 2) {
        const v = Math.max(-1, Math.min(1, chunk[i]));
        view.setInt16(offset, v < 0 ? v * 0x8000 : v * 0x7fff, true);
      }
    }
    return new Blob([buffer], { type: "audio/wav" });
  }

  /*
   * POST the recording and display the result.
   * Args: wav - Blob. Returns: Promise<void>.
   */
  async function upload(wav) {
    const form = new FormData();
    form.append("audio", wav, "clip.wav");
    form.append("mode", currentMode());
    try {
      const response = await fetch("api/identify", { method: "POST", body: form });
      const data = await response.json().catch(() => ({ error: `Server error (${response.status})` }));
      if (!response.ok || data.error) throw new Error(data.error || `Server error (${response.status})`);
      showResult(data);
    } catch (err) {
      setStatus(err.message || "Could not reach the server.", "error");
    } finally {
      setState("idle");
    }
  }

  // -------------------------------------------------------------------------
  // Startup
  // -------------------------------------------------------------------------

  /*
   * Load which systems the server has and how many songs they know.
   * Args: none. Returns: Promise<void>.
   */
  async function loadStatus() {
    try {
      const data = await (await fetch("api/status")).json();
      available = data.available;
      const songs = Math.max(data.songs.landmark, data.songs.neural);
      els.library.textContent = songs ? `${songs} songs in the library` : "Library unavailable";
      if (!available.neural && available.landmark) {
        els.modeSwitch.checked = false;
        els.modeSwitch.disabled = true;
        els.modeHint.textContent = "Neural model not available on this server";
      } else if (!available.landmark && available.neural) {
        els.modeSwitch.checked = true;
        els.modeSwitch.disabled = true;
      }
      if (!available.landmark && !available.neural) {
        els.button.disabled = true;
        setStatus("No recognition system is available on the server.", "error");
      }
      renderMode();
    } catch (_) {
      els.library.textContent = "Could not reach the server";
    }
  }

  els.modeSwitch.checked = localStorage.getItem(MODE_KEY) === "neural";
  els.modeSwitch.addEventListener("change", renderMode);
  els.modeNames.forEach((el) => el.addEventListener("click", (event) => {
    event.preventDefault();
    if (els.modeSwitch.disabled) return;
    els.modeSwitch.checked = el.dataset.mode === "neural";
    renderMode();
  }));
  els.button.addEventListener("click", () => {
    if (state === "idle") startRecording();
    else if (state === "recording") stopRecording();
  });

  renderMode();
  setState("idle");
  const reason = micUnsupportedReason();
  if (reason) setStatus(reason, "error");
  else watchPermission();
  loadStatus();
})();
