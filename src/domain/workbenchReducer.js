export const initialWorkbenchState = {
  range: { start: 0, end: 1 },
  selectedPresetId: 'tone-440',
  processing: {
    window: 'hann',
    gain: 1,
    normalize: false,
    removeDc: true,
  },
  activeView: 'overview',
  helpOpen: false,
};

export function workbenchReducer(state, action) {
  switch (action.type) {
    case 'select-preset':
      return { ...state, selectedPresetId: action.id, range: { start: 0, end: 1 } };
    case 'set-range-start': {
      const start = Math.max(0, Math.min(action.value, state.range.end - 0.01));
      return { ...state, range: { ...state.range, start } };
    }
    case 'set-range-end': {
      const end = Math.min(1, Math.max(action.value, state.range.start + 0.01));
      return { ...state, range: { ...state.range, end } };
    }
    case 'set-processing':
      return { ...state, processing: { ...state.processing, [action.key]: action.value } };
    case 'set-view':
      return { ...state, activeView: action.view };
    case 'open-help':
      return { ...state, helpOpen: true };
    case 'close-help':
      return { ...state, helpOpen: false };
    case 'reset-processing':
      return {
        ...state,
        processing: initialWorkbenchState.processing,
        range: { start: 0, end: 1 },
      };
    default:
      return state;
  }
}
