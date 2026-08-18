import { MetricsGrid } from './MetricsGrid.jsx';
import { SpectrumChart } from './SpectrumChart.jsx';

export function AnalysisPanel({ analysis, activeView, onViewChange }) {
  const tabs = [
    ['overview', 'Overview'],
    ['spectrum', 'Spectrum'],
  ];

  return (
    <section className="analysis-panel" aria-labelledby="analysis-title">
      <div className="analysis-header">
        <div>
          <p className="eyebrow">Inspector</p>
          <h2 id="analysis-title">Analysis</h2>
        </div>
        <div className="analysis-tabs" aria-label="Analysis views">
          {tabs.map(([id, label]) => (
            <button
              key={id}
              type="button"
              aria-pressed={activeView === id}
              className={activeView === id ? 'is-active' : ''}
              onClick={() => onViewChange(id)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {analysis.status === 'loading' && (
        <div className="analysis-state" role="status">
          <span className="spinner" aria-hidden="true" />
          <div><strong>Recomputing analysis</strong><span>Using the active range and processing pipeline.</span></div>
        </div>
      )}

      {analysis.status === 'error' && (
        <div className="analysis-state error-state" role="alert">
          <strong>Analysis could not be completed.</strong>
          <span>{analysis.error?.message ?? 'Unexpected analysis error.'}</span>
        </div>
      )}

      {analysis.status === 'ready' && analysis.result && (
        <div className="analysis-content">
          {activeView === 'overview' ? (
            <>
              <MetricsGrid result={analysis.result} />
              <div className="analysis-footnote">
                <span>{analysis.result.sampleCount.toLocaleString()} selected samples</span>
                <span>{analysis.result.sampleRate.toLocaleString()} Hz sample rate</span>
                <span>{analysis.result.duration.toFixed(3)} s duration</span>
              </div>
            </>
          ) : (
            <SpectrumChart points={analysis.result.spectrum} />
          )}
        </div>
      )}
    </section>
  );
}
