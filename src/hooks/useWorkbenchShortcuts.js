import { useEffect } from 'react';

function isTypingTarget(target) {
  return target instanceof HTMLElement && (
    target.matches('input, textarea, select') || target.isContentEditable
  );
}

export function useWorkbenchShortcuts({ onHelp, onReset }) {
  useEffect(() => {
    function onKeyDown(event) {
      if (event.ctrlKey || event.metaKey || event.altKey) return;

      if (event.key === '?' && !isTypingTarget(event.target)) {
        event.preventDefault();
        onHelp();
      }
      if (event.key.toLowerCase() === 'r' && !isTypingTarget(event.target)) {
        event.preventDefault();
        onReset();
      }
    }

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onHelp, onReset]);
}
