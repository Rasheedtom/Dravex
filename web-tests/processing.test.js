import { describe, expect, it } from 'vitest';
import { applyProcessing } from '../src/domain/processing.js';

describe('processing pipeline', () => {
  it('can remove DC offset without mutating input samples', () => {
    const input = [1, 2, 3];
    const output = applyProcessing(input, { window: 'rectangular', gain: 1, removeDc: true, normalize: false });
    expect(output[0]).toBe(-1);
    expect(output[2]).toBe(1);
    expect(input).toEqual([1, 2, 3]);
  });

  it('normalizes the processed peak when requested', () => {
    const output = applyProcessing([0.25, -0.5, 0.2], {
      window: 'rectangular', gain: 1, removeDc: false, normalize: true,
    });
    expect(Math.max(...output.map(Math.abs))).toBeCloseTo(1);
  });

  it('applies gain after the selected window', () => {
    const output = applyProcessing([1, 1, 1], {
      window: 'rectangular', gain: 1.5, removeDc: false, normalize: false,
    });
    expect(output).toEqual([1.5, 1.5, 1.5]);
  });
});
