# Dravex Architecture

## Package Structure

```
dravex/
├── core/        Signal containers and utilities
├── transform/   FFT, DCT, STFT, wavelet, Hilbert
├── filter/      FIR, IIR, Butterworth, Chebyshev, Kalman
├── window/      Hamming, Hann, Blackman, Kaiser, Gaussian
├── spectral/    PSD, MFCC, pitch, formants, chroma
├── convolve/    Direct, overlap-add, overlap-save, FFT
├── resample/    Interpolation, decimation, polyphase
├── modulate/    AM, FM, PM, QAM, BPSK
├── noise/       Generation, SNR, denoising
├── correlation/ Autocorrelation, cross-correlation, coherence
├── compress/    mu-law, A-law, ADPCM, quantization
├── detect/      Peaks, onsets, silence, energy
├── io/          WAV, raw, CSV, tone generation
├── stats/       Moments, histogram, entropy, stationarity
└── lpc/         Linear prediction analysis and synthesis
```

## Design Principles

- Pure stdlib Python 3.12 — zero runtime dependencies
- Immutable Signal type for thread safety
- Functional style for DSP operations
- Streaming support via RingBuffer and frame processing
