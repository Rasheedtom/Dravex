export function SignalLibrary({ presets, selectedId, onSelect, onImport }) {
  return (
    <aside className="signal-library" aria-labelledby="signal-library-title">
      <div className="section-heading compact-heading">
        <div>
          <p className="eyebrow">Sources</p>
          <h2 id="signal-library-title">Signal library</h2>
        </div>
      </div>
      <div className="signal-list">
        {presets.map((preset) => {
          const active = preset.id === selectedId;
          return (
            <button
              key={preset.id}
              type="button"
              className={`signal-card${active ? ' is-active' : ''}`}
              aria-pressed={active}
              onClick={() => onSelect(preset.id)}
            >
              <span className="signal-card-title">{preset.name}</span>
              <span className="signal-card-description">{preset.description}</span>
              <span className="signal-card-meta">{preset.sampleRate / 1000} kHz · {preset.duration}s</span>
            </button>
          );
        })}
      </div>
      <label className="import-button">
        <span>Import samples</span>
        <span className="import-hint">CSV or text</span>
        <input type="file" accept=".csv,.txt,text/csv,text/plain" onChange={onImport} />
      </label>
    </aside>
  );
}
