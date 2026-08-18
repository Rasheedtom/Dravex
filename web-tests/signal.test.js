import { describe, expect, it } from 'vitest';
import { SIGNAL_PRESETS } from '../src/data/presets.js';
import { generatePresetSignal, parseSampleText, sliceSignal } from '../src/domain/signal.js';

describe('signal sources', () => {
  it('generates deterministic preset samples at the declared duration', () => {
    const preset = SIGNAL_PRESETS[0];
    const signal = generatePresetSignal(preset);
    expect(signal.samples).toHaveLength(preset.sampleRate * preset.duration);
    expect(signal.sampleRate).toBe(preset.sampleRate);
    expect(signal.samples[0]).toBe(0);
  });

  it('parses CSV and whitespace-separated sample values', () => {
    const signal = parseSampleText('0, 0.5\n-0.25; 1');
    expect(signal.samples).toEqual([0, 0.5, -0.25, 1]);
  });

  it('reports invalid imported values clearly', () => {
    expect(() => parseSampleText('0, nope, 1')).toThrow('Sample 2 is not a finite number');
  });

  it('slices a signal using normalized range boundaries', () => {
    const signal = { samples: [0, 1, 2, 3, 4, 5, 6, 7], sampleRate: 8 };
    expect(sliceSignal(signal, { start: 0.25, end: 0.75 }).samples).toEqual([2, 3, 4, 5]);
  });
});
