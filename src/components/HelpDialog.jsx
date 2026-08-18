import { useEffect, useRef } from 'react';

export function HelpDialog({ open, onClose }) {
  const closeRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const previous = document.activeElement;
    closeRef.current?.focus();

    function onKeyDown(event) {
      if (event.key === 'Escape') onClose();
      if (event.key === 'Tab') {
        event.preventDefault();
        closeRef.current?.focus();
      }
    }

    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      if (previous instanceof HTMLElement) previous.focus();
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section className="help-dialog" role="dialog" aria-modal="true" aria-labelledby="help-title">
        <button ref={closeRef} type="button" className="dialog-close" onClick={onClose} aria-label="Close keyboard help">×</button>
        <p className="eyebrow">Shortcuts</p>
        <h2 id="help-title">Keyboard help</h2>
        <dl className="shortcut-list">
          <div><dt><kbd>?</kbd></dt><dd>Open keyboard help</dd></div>
          <div><dt><kbd>R</kbd></dt><dd>Reset processing and analysis range</dd></div>
          <div><dt><kbd>Esc</kbd></dt><dd>Close this dialog</dd></div>
        </dl>
      </section>
    </div>
  );
}
