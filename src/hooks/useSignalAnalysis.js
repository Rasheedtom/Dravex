import { useEffect, useRef, useState } from 'react';
import { analyzeSignal } from '../domain/analysis.js';

const ANALYSIS_DELAY_MS = 120;

export function useSignalAnalysis(samples, sampleRate) {
  const requestRef = useRef(0);
  const [state, setState] = useState({ status: 'idle', result: null, error: null });

  useEffect(() => {
    const requestId = requestRef.current + 1;
    requestRef.current = requestId;
    const controller = new AbortController();

    setState((current) => ({ ...current, status: 'loading', error: null }));

    const timer = setTimeout(() => {
      if (controller.signal.aborted || requestRef.current !== requestId) return;
      try {
        const result = analyzeSignal(samples, sampleRate);
        if (!controller.signal.aborted && requestRef.current === requestId) {
          setState({ status: 'ready', result, error: null });
        }
      } catch (error) {
        if (!controller.signal.aborted && requestRef.current === requestId) {
          setState({ status: 'error', result: null, error });
        }
      }
    }, ANALYSIS_DELAY_MS);

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [samples, sampleRate]);

  return state;
}
