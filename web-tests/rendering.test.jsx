import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { AppHeader } from '../src/components/AppHeader.jsx';
import { SignalLibrary } from '../src/components/SignalLibrary.jsx';
import { ProcessingControls } from '../src/components/ProcessingControls.jsx';
import { SIGNAL_PRESETS } from '../src/data/presets.js';

describe('React semantics', () => {
  it('exposes keyboard help as a labelled control', () => {
    const html = renderToStaticMarkup(<AppHeader onOpenHelp={() => {}} />);
    expect(html).toContain('Keyboard help');
    expect(html).toContain('aria-keyshortcuts="?"');
  });

  it('marks the selected signal source with aria-pressed', () => {
    const html = renderToStaticMarkup(
      <SignalLibrary presets={SIGNAL_PRESETS} selectedId="tone-440" onSelect={() => {}} onImport={() => {}} />,
    );
    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain('Import samples');
  });

  it('keeps processing toggles associated with visible labels', () => {
    const settings = { window: 'hann', gain: 1, removeDc: true, normalize: false };
    const html = renderToStaticMarkup(<ProcessingControls settings={settings} onChange={() => {}} onReset={() => {}} />);
    expect(html).toContain('Remove DC offset');
    expect(html).toContain('Normalize peak');
    expect(html).toContain('<select');
  });
});
