import { describe, expect, it } from 'vitest';
import { initialWorkbenchState, workbenchReducer } from '../src/domain/workbenchReducer.js';

describe('workbench state', () => {
  it('keeps range handles ordered', () => {
    const next = workbenchReducer(initialWorkbenchState, { type: 'set-range-start', value: 1 });
    expect(next.range.start).toBeLessThan(next.range.end);
  });

  it('resets processing without changing the selected source', () => {
    const selected = { ...initialWorkbenchState, selectedPresetId: 'chirp', processing: { ...initialWorkbenchState.processing, gain: 1.8 } };
    const next = workbenchReducer(selected, { type: 'reset-processing' });
    expect(next.selectedPresetId).toBe('chirp');
    expect(next.processing.gain).toBe(1);
  });

  it('resets the selection when a different preset is chosen', () => {
    const narrowed = { ...initialWorkbenchState, range: { start: 0.2, end: 0.6 } };
    const next = workbenchReducer(narrowed, { type: 'select-preset', id: 'dual-tone' });
    expect(next.range).toEqual({ start: 0, end: 1 });
  });
});
