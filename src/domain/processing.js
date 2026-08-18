const WINDOWS = {
  rectangular: () => 1,
  hann: (index, length) => (length <= 1 ? 1 : 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / (length - 1))),
  hamming: (index, length) => (length <= 1 ? 1 : 0.54 - 0.46 * Math.cos((2 * Math.PI * index) / (length - 1))),
};

export function applyProcessing(samples, settings) {
  const windowFn = WINDOWS[settings.window] ?? WINDOWS.rectangular;
  const windowed = samples.map((sample, index) => sample * windowFn(index, samples.length));
  const gain = Number.isFinite(settings.gain) ? settings.gain : 1;
  let processed = windowed.map((sample) => sample * gain);

  if (settings.removeDc && processed.length > 0) {
    const mean = processed.reduce((total, value) => total + value, 0) / processed.length;
    processed = processed.map((sample) => sample - mean);
  }

  if (settings.normalize && processed.length > 0) {
    const peak = Math.max(...processed.map(Math.abs));
    if (peak > 1e-12) {
      processed = processed.map((sample) => sample / peak);
    }
  }

  return processed;
}

export function defaultProcessingSettings() {
  return {
    window: 'hann',
    gain: 1,
    normalize: false,
    removeDc: true,
  };
}
