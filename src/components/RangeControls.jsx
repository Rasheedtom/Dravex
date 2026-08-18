function percent(value) {
  return Math.round(value * 100);
}

export function RangeControls({ range, onStartChange, onEndChange }) {
  return (
    <fieldset className="range-controls">
      <legend>Analysis range</legend>
      <div className="range-summary" aria-live="polite">
        <strong>{percent(range.start)}%</strong>
        <span>to</span>
        <strong>{percent(range.end)}%</strong>
      </div>
      <label>
        <span>Start</span>
        <input
          type="range"
          min="0"
          max="0.99"
          step="0.01"
          value={range.start}
          onChange={(event) => onStartChange(Number(event.target.value))}
        />
      </label>
      <label>
        <span>End</span>
        <input
          type="range"
          min="0.01"
          max="1"
          step="0.01"
          value={range.end}
          onChange={(event) => onEndChange(Number(event.target.value))}
        />
      </label>
    </fieldset>
  );
}
