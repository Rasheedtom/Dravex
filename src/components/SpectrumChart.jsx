export function SpectrumChart({ points }) {
  const width = 760;
  const height = 210;
  const bars = points.map((point, index) => ({
    ...point,
    x: (index / points.length) * width,
    width: Math.max(1, width / points.length - 2),
    y: height - point.normalized * (height - 12),
    height: point.normalized * (height - 12),
  }));

  return (
    <figure className="spectrum-figure">
      <figcaption>Normalized spectrum</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Normalized frequency spectrum">
        {bars.map((bar) => (
          <rect
            key={bar.frequency}
            x={bar.x}
            y={bar.y}
            width={bar.width}
            height={bar.height}
            rx="1"
          >
            <title>{`${Math.round(bar.frequency)} Hz: ${Math.round(bar.normalized * 100)}%`}</title>
          </rect>
        ))}
      </svg>
      <div className="spectrum-axis" aria-hidden="true"><span>0 Hz</span><span>Nyquist</span></div>
    </figure>
  );
}
