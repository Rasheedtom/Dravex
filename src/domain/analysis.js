function safeRms(samples) {
  if (samples.length === 0) return 0;
  const energy = samples.reduce((total, value) => total + value * value, 0);
  return Math.sqrt(energy / samples.length);
}

export function zeroCrossingCount(samples) {
  let crossings = 0;
  for (let index = 1; index < samples.length; index += 1) {
    if ((samples[index - 1] < 0 && samples[index] >= 0) || (samples[index - 1] >= 0 && samples[index] < 0)) {
      crossings += 1;
    }
  }
  return crossings;
}

export function estimateDominantFrequency(samples, sampleRate) {
  if (samples.length < 4 || sampleRate <= 0) return 0;
  const maxFrequency = Math.min(sampleRate / 2, 1600);
  const step = Math.max(1, Math.floor(sampleRate / 240));
  let bestFrequency = 0;
  let bestMagnitude = -Infinity;

  for (let frequency = step; frequency <= maxFrequency; frequency += step) {
    let real = 0;
    let imaginary = 0;
    const angular = (2 * Math.PI * frequency) / sampleRate;
    for (let index = 0; index < samples.length; index += 1) {
      const phase = angular * index;
      real += samples[index] * Math.cos(phase);
      imaginary -= samples[index] * Math.sin(phase);
    }
    const magnitude = real * real + imaginary * imaginary;
    if (magnitude > bestMagnitude) {
      bestMagnitude = magnitude;
      bestFrequency = frequency;
    }
  }

  return bestFrequency;
}

export function buildSpectrum(samples, sampleRate, bucketCount = 72) {
  if (samples.length === 0 || sampleRate <= 0) return [];
  const maxFrequency = sampleRate / 2;
  const points = [];
  const stride = Math.max(1, Math.floor(samples.length / 768));
  const compact = samples.filter((_, index) => index % stride === 0).slice(0, 768);

  for (let bucket = 1; bucket <= bucketCount; bucket += 1) {
    const frequency = (bucket / bucketCount) * maxFrequency;
    const angular = (2 * Math.PI * frequency) / sampleRate;
    let real = 0;
    let imaginary = 0;
    for (let index = 0; index < compact.length; index += 1) {
      const phase = angular * index * stride;
      real += compact[index] * Math.cos(phase);
      imaginary -= compact[index] * Math.sin(phase);
    }
    points.push({
      frequency,
      magnitude: Math.sqrt(real * real + imaginary * imaginary) / compact.length,
    });
  }

  const peak = Math.max(...points.map((point) => point.magnitude), 1e-12);
  return points.map((point) => ({ ...point, normalized: point.magnitude / peak }));
}

export function analyzeSignal(samples, sampleRate) {
  const peak = samples.length ? Math.max(...samples.map(Math.abs)) : 0;
  const rms = safeRms(samples);
  const mean = samples.length ? samples.reduce((total, value) => total + value, 0) / samples.length : 0;
  const duration = sampleRate > 0 ? samples.length / sampleRate : 0;
  const zeroCrossings = zeroCrossingCount(samples);
  const dominantFrequency = estimateDominantFrequency(samples, sampleRate);

  return {
    sampleCount: samples.length,
    sampleRate,
    duration,
    peak,
    rms,
    mean,
    crestFactor: rms > 1e-12 ? peak / rms : 0,
    zeroCrossings,
    dominantFrequency,
    spectrum: buildSpectrum(samples, sampleRate),
  };
}

export function formatMetric(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}
