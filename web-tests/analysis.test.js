import { describe, expect, it } from 'vitest';
import { analyzeSignal, estimateDominantFrequency, zeroCrossingCount } from '../src/domain/analysis.js';
import { generatePresetSignal } from '../src/domain/signal.js';
import { SIGNAL_PRESETS } from '../src/data/presets.js';

describe('signal analysis', () => {
  it('computes stable scalar metrics', () => {
    const result = analyzeSignal([1, -1, 1, -1], 4);
    expect(result.peak).toBe(1);
    expect(result.rms).toBe(1);
    expect(result.duration).toBe(1);
    expect(result.zeroCrossings).toBe(3);
  });

  it('counts sign transitions', () => {
    expect(zeroCrossingCount([-1, -0.5, 0.2, 0.5, -0.1])).toBe(2);
  });

  it('finds the reference tone near 440 Hz', () => {
    const signal = generatePresetSignal(SIGNAL_PRESETS[0]);
    const estimated = estimateDominantFrequency(signal.samples.slice(0, 1200), signal.sampleRate);
    expect(estimated).toBeGreaterThan(410);
    expect(estimated).toBeLessThan(470);
  });

  it('returns a normalized spectrum bounded by one', () => {
    const signal = generatePresetSignal(SIGNAL_PRESETS[1]);
    const result = analyzeSignal(signal.samples.slice(0, 1000), signal.sampleRate);
    expect(result.spectrum).toHaveLength(72);
    expect(Math.max(...result.spectrum.map((point) => point.normalized))).toBeCloseTo(1);
  });
});
