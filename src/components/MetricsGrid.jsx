import { formatMetric } from '../domain/analysis.js';

const METRICS = [
  ['Peak', (result) => formatMetric(result.peak, 3), 'amplitude'],
  ['RMS', (result) => formatMetric(result.rms, 3), 'amplitude'],
  ['Crest factor', (result) => formatMetric(result.crestFactor, 2), 'ratio'],
  ['Dominant', (result) => `${formatMetric(result.dominantFrequency, 0)} Hz`, 'frequency'],
  ['Zero crossings', (result) => String(result.zeroCrossings), 'count'],
  ['Mean', (result) => formatMetric(result.mean, 4), 'amplitude'],
];

export function MetricsGrid({ result }) {
  return (
    <dl className="metrics-grid">
      {METRICS.map(([label, getValue, unit]) => (
        <div className="metric-card" key={label}>
          <dt>{label}</dt>
          <dd>{getValue(result)}</dd>
          <span>{unit}</span>
        </div>
      ))}
    </dl>
  );
}
