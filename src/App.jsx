import { useCallback, useMemo, useReducer, useState } from 'react';
import { AppHeader } from './components/AppHeader.jsx';
import { SignalLibrary } from './components/SignalLibrary.jsx';
import { WaveformChart } from './components/WaveformChart.jsx';
import { RangeControls } from './components/RangeControls.jsx';
import { ProcessingControls } from './components/ProcessingControls.jsx';
import { AnalysisPanel } from './components/AnalysisPanel.jsx';
import { HelpDialog } from './components/HelpDialog.jsx';
import { SIGNAL_PRESETS } from './data/presets.js';
import { generatePresetSignal, parseSampleText, sliceSignal } from './domain/signal.js';
import { applyProcessing } from './domain/processing.js';
import { initialWorkbenchState, workbenchReducer } from './domain/workbenchReducer.js';
import { useSignalAnalysis } from './hooks/useSignalAnalysis.js';
import { useWorkbenchShortcuts } from './hooks/useWorkbenchShortcuts.js';
import './styles/app.css';

export default function App() {
  const [state, dispatch] = useReducer(workbenchReducer, initialWorkbenchState);
  const [importedSignal, setImportedSignal] = useState(null);
  const [importError, setImportError] = useState('');

  const preset = SIGNAL_PRESETS.find((item) => item.id === state.selectedPresetId) ?? SIGNAL_PRESETS[0];
  const presetSignal = useMemo(() => generatePresetSignal(preset), [preset]);
  const sourceSignal = importedSignal ?? presetSignal;
  const rangedSignal = useMemo(() => sliceSignal(sourceSignal, state.range), [sourceSignal, state.range]);
  const processedSamples = useMemo(
    () => applyProcessing(rangedSignal.samples, state.processing),
    [rangedSignal.samples, state.processing],
  );
  const analysis = useSignalAnalysis(processedSamples, sourceSignal.sampleRate);

  const openHelp = useCallback(() => dispatch({ type: 'open-help' }), []);
  const resetWorkbench = useCallback(() => dispatch({ type: 'reset-processing' }), []);
  const closeHelp = useCallback(() => dispatch({ type: 'close-help' }), []);
  useWorkbenchShortcuts({ onHelp: openHelp, onReset: resetWorkbench });

  function selectPreset(id) {
    setImportedSignal(null);
    setImportError('');
    dispatch({ type: 'select-preset', id });
  }

  async function importSamples(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const signal = parseSampleText(text);
      setImportedSignal(signal);
      setImportError('');
      dispatch({ type: 'select-preset', id: state.selectedPresetId });
    } catch (error) {
      setImportError(error instanceof Error ? error.message : 'The sample file could not be read.');
    } finally {
      event.target.value = '';
    }
  }

  return (
    <div className="app-shell">
      <AppHeader onOpenHelp={openHelp} />
      <main className="workspace">
        <SignalLibrary
          presets={SIGNAL_PRESETS}
          selectedId={importedSignal ? '' : state.selectedPresetId}
          onSelect={selectPreset}
          onImport={importSamples}
        />

        <section className="canvas-column" aria-labelledby="workspace-title">
          <div className="signal-hero">
            <div className="section-heading hero-heading">
              <div>
                <p className="eyebrow">Active signal</p>
                <h2 id="workspace-title">{sourceSignal.name}</h2>
                <p>{sourceSignal.description}</p>
              </div>
              <div className="source-meta" aria-label="Signal metadata">
                <span>{sourceSignal.samples.length.toLocaleString()} samples</span>
                <span>{sourceSignal.sampleRate.toLocaleString()} Hz</span>
              </div>
            </div>
            {importError && <p className="inline-error" role="alert">{importError}</p>}
            <WaveformChart samples={sourceSignal.samples} range={state.range} label={sourceSignal.name} />
            <RangeControls
              range={state.range}
              onStartChange={(value) => dispatch({ type: 'set-range-start', value })}
              onEndChange={(value) => dispatch({ type: 'set-range-end', value })}
            />
          </div>
          <AnalysisPanel
            analysis={analysis}
            activeView={state.activeView}
            onViewChange={(view) => dispatch({ type: 'set-view', view })}
          />
        </section>

        <ProcessingControls
          settings={state.processing}
          onChange={(key, value) => dispatch({ type: 'set-processing', key, value })}
          onReset={resetWorkbench}
        />
      </main>
      <footer className="app-footer">
        <span>Dravex Signal Workbench</span>
        <span>Browser-side exploration · no samples leave this device</span>
      </footer>
      <HelpDialog open={state.helpOpen} onClose={closeHelp} />
    </div>
  );
}
