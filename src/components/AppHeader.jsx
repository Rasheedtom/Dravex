export function AppHeader({ onOpenHelp }) {
  return (
    <header className="app-header">
      <div className="brand-lockup">
        <span className="brand-mark" aria-hidden="true">D</span>
        <div>
          <p className="eyebrow">Dravex</p>
          <h1>Signal Workbench</h1>
        </div>
      </div>
      <div className="header-actions">
        <span className="local-badge">Runs locally</span>
        <button className="quiet-button" type="button" onClick={onOpenHelp} aria-keyshortcuts="?">
          Keyboard help
          <kbd>?</kbd>
        </button>
      </div>
    </header>
  );
}
