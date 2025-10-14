# Changelog

## [0.9.0] - 2025-11-15

### Added
- Core signal processing types (Signal, Frame, Spectrum, MultiChannel)
- FFT/IFFT with Cooley-Tukey radix-2 implementation
- DCT-II/III, Hilbert transform, STFT/ISTFT
- Wavelet transform (Haar and Daubechies-4)
- Cepstral analysis and Goertzel algorithm
- FIR filter design (lowpass, highpass, bandpass, bandstop)
- IIR filter application and biquad cascade
- Butterworth and Chebyshev filter design
- Median, Kalman, notch, and allpass filters
- Window functions (Hamming, Hann, Blackman, Kaiser, Gaussian, Tukey)
- Spectral analysis (PSD, MFCC, pitch detection, formant tracking)
- Convolution methods (direct, overlap-add, overlap-save, FFT)
- Resampling (interpolation, decimation, polyphase, sinc)
- Modulation (AM, FM, PM, QAM, BPSK)
- Noise generation and SNR estimation
- Correlation and coherence analysis
- Compression codecs (mu-law, A-law, ADPCM, quantization)
- Signal detection (peaks, onsets, zero crossings, silence)
- WAV and raw PCM I/O
- Statistical signal analysis
- LPC analysis, synthesis, and PARCOR coefficients

### Fixed
- FFT bit-reversal for non-power-of-2 lengths
- Kalman filter convergence for high measurement noise
