# Dravex

Dravex is a pure-Python digital signal processing library with a browser-based React workbench for exploring signals interactively. The Python package has zero runtime dependencies and targets Python 3.12+. The workbench runs locally in the browser and targets Node.js 20.

## Python library

### Features

- **Core**: Signal, Frame, Spectrum, MultiChannel containers
- **Transforms**: FFT, DCT, STFT, Hilbert, Wavelet, Cepstrum
- **Filters**: FIR/IIR design, Butterworth, Chebyshev, Kalman, median
- **Spectral**: PSD, MFCC, pitch detection, formant tracking, chroma
- **Convolution**: Direct, overlap-add/save, FFT-based, deconvolution
- **Resampling**: Interpolation, decimation, polyphase, sinc
- **Modulation**: AM/FM/PM, QAM, BPSK
- **Noise**: Generation, SNR estimation, spectral subtraction
- **Compression**: μ-law, A-law, ADPCM, quantization with dithering
- **Detection**: Peaks, onsets, silence, clipping, zero crossings
- **I/O**: WAV, raw PCM, CSV, JSON

### Quick start

```python
from dravex.core.signal import Signal
from dravex.transform.fft import rfft
from dravex.io.tone_gen import sine_wave

samples = sine_wave(440, 1.0, 44100)
signal = Signal(samples, 44100)
magnitudes = rfft(signal.samples)
```

Run the Python test suite with:

```bash
python -m pytest
```

## Signal Workbench

The React workbench is a self-contained companion for inspecting DSP behavior without sending samples to a backend. It includes deterministic signal presets, local CSV/text sample import, range selection, windowing, gain, DC-offset removal, peak normalization, scalar metrics, and a normalized spectrum view.

### Requirements

- Node.js 20
- npm

### Run locally

```bash
npm ci
npm run dev
```

Then open the local Vite address printed in the terminal.

### Verify the workbench

```bash
npm test
npm run lint
npm run check
npm run build
```

The JavaScript tests are self-contained and do not require credentials, network services, or machine-specific files.

### Workbench architecture

- `src/data/` — deterministic signal presets
- `src/domain/` — signal generation, processing, analysis, and reducer logic
- `src/hooks/` — async analysis lifecycle and keyboard shortcuts
- `src/components/` — accessible React UI components
- `src/styles/` — responsive workbench presentation
- `web-tests/` — Vitest coverage for domain behavior, state, rendering semantics, and repository constraints

The workbench intentionally keeps DSP calculations local and dependency-light. It is a UI companion rather than a browser port of the complete Python implementation.

## License

MIT
