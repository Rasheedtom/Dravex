const TAU = Math.PI * 2;

function seededNoise(index) {
  const value = Math.sin(index * 12.9898 + 78.233) * 43758.5453;
  return (value - Math.floor(value)) * 2 - 1;
}

export function generatePresetSignal(preset) {
  const sampleCount = Math.max(1, Math.round(preset.sampleRate * preset.duration));
  const samples = Array.from({ length: sampleCount }, (_, index) => {
    const time = index / preset.sampleRate;

    switch (preset.kind) {
      case 'tone':
        return Math.sin(TAU * preset.frequency * time) * 0.72;
      case 'noisy-tone':
        return Math.sin(TAU * preset.frequency * time) * 0.58 + seededNoise(index) * 0.16;
      case 'dual-tone':
        return (
          Math.sin(TAU * preset.frequency * time) * 0.52 +
          Math.sin(TAU * preset.frequency * 2 * time) * 0.26
        );
      case 'chirp': {
        const sweep = (preset.endFrequency - preset.startFrequency) / preset.duration;
        const phase = TAU * (preset.startFrequency * time + 0.5 * sweep * time * time);
        return Math.sin(phase) * 0.68;
      }
      case 'transient': {
        const period = Math.max(1, Math.round(preset.sampleRate * 0.12));
        const offset = index % period;
        if (offset > preset.sampleRate * 0.035) return 0;
        return Math.exp(-offset / (preset.sampleRate * 0.009)) * (offset % 2 === 0 ? 0.88 : -0.72);
      }
      default:
        return 0;
    }
  });

  return {
    id: preset.id,
    name: preset.name,
    description: preset.description,
    sampleRate: preset.sampleRate,
    samples,
    source: 'preset',
  };
}

export function parseSampleText(text, sampleRate = 8000) {
  const tokens = text
    .split(/[\s,;]+/)
    .map((token) => token.trim())
    .filter(Boolean);

  if (tokens.length === 0) {
    throw new Error('No numeric samples were found in the file.');
  }

  const samples = tokens.map((token, index) => {
    const value = Number(token);
    if (!Number.isFinite(value)) {
      throw new Error(`Sample ${index + 1} is not a finite number.`);
    }
    return value;
  });

  if (samples.length > 100_000) {
    throw new Error('Sample files are limited to 100,000 values in the workbench.');
  }

  return {
    id: `import-${samples.length}`,
    name: 'Imported samples',
    description: `${samples.length.toLocaleString()} values loaded from a local text or CSV file.`,
    sampleRate,
    samples,
    source: 'import',
  };
}

export function sliceSignal(signal, range) {
  const length = signal.samples.length;
  if (length === 0) return signal;
  const start = Math.max(0, Math.min(length - 1, Math.floor(range.start * length)));
  const rawEnd = Math.ceil(range.end * length);
  const end = Math.max(start + 1, Math.min(length, rawEnd));
  return { ...signal, samples: signal.samples.slice(start, end) };
}
