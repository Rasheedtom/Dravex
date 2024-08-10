# Dravex

Pure-Python digital signal processing library. Zero runtime dependencies, Python 3.12+.

## Features

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

## Quick Start

```python
from dravex.core.signal import Signal
from dravex.transform.fft import rfft
from dravex.io.tone_gen import sine_wave

# Generate and analyze a 440 Hz tone
samples = sine_wave(440, 1.0, 44100)
signal = Signal(samples, 44100)
magnitudes = rfft(signal.samples)
```

## License

MIT
