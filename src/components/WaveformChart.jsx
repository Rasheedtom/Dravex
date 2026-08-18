function pointsFromSamples(samples, width, height) {
  if (samples.length === 0) return '';
  const bucketCount = Math.min(samples.length, 480);
  const stride = Math.max(1, Math.floor(samples.length / bucketCount));
  const compact = samples.filter((_, index) => index % stride === 0).slice(0, bucketCount);
  return compact
    .map((sample, index) => {
      const x = compact.length === 1 ? 0 : (index / (compact.length - 1)) * width;
      const y = height / 2 - Math.max(-1.2, Math.min(1.2, sample)) * (height * 0.36);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

export function WaveformChart({ samples, range, label }) {
  const width = 920;
  const height = 250;
  const points = pointsFromSamples(samples, width, height);
  const selectionX = range.start * width;
  const selectionWidth = Math.max(1, (range.end - range.start) * width);

  return (
    <figure className="waveform-figure">
      <figcaption className="sr-only">{label} waveform with selected analysis range</figcaption>
      <svg className="waveform-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} waveform`}>
        <defs>
          <linearGradient id="waveStroke" x1="0" x2="1">
            <stop offset="0" stopColor="currentColor" stopOpacity="0.65" />
            <stop offset="1" stopColor="currentColor" stopOpacity="1" />
          </linearGradient>
        </defs>
        <g className="chart-grid" aria-hidden="true">
          {[0.25, 0.5, 0.75].map((ratio) => (
            <line key={`h-${ratio}`} x1="0" y1={height * ratio} x2={width} y2={height * ratio} />
          ))}
          {[0.2, 0.4, 0.6, 0.8].map((ratio) => (
            <line key={`v-${ratio}`} x1={width * ratio} y1="0" x2={width * ratio} y2={height} />
          ))}
        </g>
        <rect className="selection-fill" x={selectionX} y="0" width={selectionWidth} height={height} />
        <line className="zero-line" x1="0" y1={height / 2} x2={width} y2={height / 2} />
        {points && <polyline className="waveform-line" points={points} fill="none" />}
      </svg>
    </figure>
  );
}
