/*
 * AudioWorklet that copies raw microphone samples to the
 * main thread in ~4096-sample chunks, plus an RMS level for the meter.
 *
 * Runs on the browser's audio rendering thread.
 * Messages posted: { samples: Float32Array, rms: number } and, after the main
 * thread sends "flush", a final partial chunk followed by { done: true }.
 */
class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(4096);
    this.length = 0;
    this.port.onmessage = (event) => {
      if (event.data === "flush") {
        if (this.length) this.flush();
        this.port.postMessage({ done: true });
      }
    };
  }

  /*
   * Called for every 128-sample render quantum.
   * Args: inputs - array of inputs, each an array of channel Float32Arrays.
   * Returns: true to keep the processor alive.
   */
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this.buffer[this.length++] = channel[i];
      if (this.length === this.buffer.length) this.flush();
    }
    return true;
  }

  /*
   * Send the filled buffer to the main thread and start a new one.
   * Args: none. Returns: nothing.
   */
  flush() {
    let sum = 0;
    for (let i = 0; i < this.length; i++) sum += this.buffer[i] * this.buffer[i];
    const samples = this.buffer.slice(0, this.length);
    this.port.postMessage({ samples, rms: Math.sqrt(sum / this.length) }, [samples.buffer]);
    this.length = 0;
  }
}

registerProcessor("recorder", RecorderProcessor);
