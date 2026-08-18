export function ProcessingControls({ settings, onChange, onReset }) {
  return (
    <section className="processing-panel" aria-labelledby="processing-title">
      <div className="section-heading inline-heading">
        <div>
          <p className="eyebrow">Pipeline</p>
          <h2 id="processing-title">Processing</h2>
        </div>
        <button type="button" className="text-button" onClick={onReset} aria-keyshortcuts="r">Reset</button>
      </div>
      <label className="control-stack">
        <span>Window</span>
        <select value={settings.window} onChange={(event) => onChange('window', event.target.value)}>
          <option value="rectangular">Rectangular</option>
          <option value="hann">Hann</option>
          <option value="hamming">Hamming</option>
        </select>
      </label>
      <label className="control-stack">
        <span className="control-line"><span>Gain</span><output>{settings.gain.toFixed(1)}×</output></span>
        <input
          type="range"
          min="0.2"
          max="2"
          step="0.1"
          value={settings.gain}
          onChange={(event) => onChange('gain', Number(event.target.value))}
        />
      </label>
      <label className="switch-row">
        <span>
          <strong>Remove DC offset</strong>
          <small>Center the selected samples before analysis.</small>
        </span>
        <input
          type="checkbox"
          checked={settings.removeDc}
          onChange={(event) => onChange('removeDc', event.target.checked)}
        />
      </label>
      <label className="switch-row">
        <span>
          <strong>Normalize peak</strong>
          <small>Scale the selected range to unit peak amplitude.</small>
        </span>
        <input
          type="checkbox"
          checked={settings.normalize}
          onChange={(event) => onChange('normalize', event.target.checked)}
        />
      </label>
    </section>
  );
}
