"""Tests for signal detection."""
from dravex.detect.peaks import find_peaks, peak_interpolate
from dravex.detect.zero_cross import zero_crossing_rate, zero_crossings
from dravex.detect.energy import frame_energy, total_energy
from dravex.detect.silence import detect_silence, voice_activity
from dravex.detect.clipping import detect_clipping, clipping_ratio
import math
from dravex.detect.peaks import find_peaks, find_valleys, prominence, peak_widths, adaptive_threshold_peaks
from dravex.detect.onset import spectral_flux, energy_onset, pick_onsets, hfc_onset, onset_strength
from dravex.detect.zero_cross import zero_crossing_rate, classify_voiced_unvoiced
from dravex.detect.energy import frame_energy, frame_rms, loudness_curve
from dravex.detect.silence import detect_silence, voice_activity, trim_silence, silence_ratio
from dravex.detect.threshold import fixed_threshold, cfar_detector, hysteresis_threshold
from dravex.detect.clipping import detect_clipping, consecutive_clipping, headroom_db, declip

class TestDetect:
    def test_find_peaks(self):
        x = [0, 1, 0, 2, 0, 3, 0]
        peaks = find_peaks(x)
        assert 1 in peaks and 3 in peaks and 5 in peaks

    def test_peak_interpolate(self):
        x = [0, 1, 3, 2, 0]
        pos, val = peak_interpolate(x, 2)
        assert 1.5 < pos < 2.5

    def test_zcr(self):
        x = [1, -1, 1, -1, 1]
        rate = zero_crossing_rate(x)
        assert rate == 1.0

    def test_zero_crossings(self):
        x = [1, -1, 1]
        zc = zero_crossings(x)
        assert len(zc) == 2

    def test_frame_energy(self):
        x = [1.0] * 2048
        e = frame_energy(x, 1024, 1024)
        assert len(e) == 2
        assert abs(e[0] - 1.0) < 1e-10

    def test_total_energy(self):
        assert abs(total_energy([3, 4]) - 25) < 1e-10

    def test_silence(self):
        x = [0.001] * 1024 + [0.5] * 1024 + [0.001] * 1024
        regions = detect_silence(x, threshold_db=-30, frame_size=256)
        assert len(regions) >= 1

    def test_clipping(self):
        x = [0.5, 0.99, 1.0, 0.99, 0.5]
        clips = detect_clipping(x, 0.99)
        assert 2 in clips

    def test_clipping_ratio(self):
        x = [1.0] * 3 + [0.5] * 7
        r = clipping_ratio(x, 0.99)
        assert abs(r - 0.3) < 1e-10



class TestDetectAdvanced:
    def test_find_valleys(self):
        x = [5, 1, 5, 0, 5]
        valleys = find_valleys(x)
        assert 1 in valleys and 3 in valleys

    def test_prominence(self):
        x = [0, 5, 0, 10, 0]
        peaks = find_peaks(x)
        proms = prominence(x, peaks)
        assert len(proms) == len(peaks)

    def test_peak_widths(self):
        x = [0]*5 + [1, 2, 3, 2, 1] + [0]*5
        peaks = find_peaks(x)
        widths = peak_widths(x, peaks)
        assert all(w > 0 for w in widths)

    def test_adaptive_peaks(self):
        x = [1]*100
        x[50] = 10
        peaks = adaptive_threshold_peaks(x, 20, 3.0)
        assert 50 in peaks

    def test_spectral_flux(self):
        frames = [[float(i)] * 10 for i in range(20)]
        sf = spectral_flux(frames)
        assert len(sf) == 20

    def test_hfc(self):
        frames = [[float(k+1)] * 10 for k in range(5)]
        hfc = hfc_onset(frames)
        assert len(hfc) == 5

    def test_onset_strength(self):
        x = [0]*2048 + [math.sin(2*math.pi*440*i/44100) for i in range(2048)]
        os_env = onset_strength(x, 44100, 512, 256)
        assert len(os_env) > 0

    def test_zcr_frames(self):
        x = [math.sin(0.5*i) for i in range(1024)]
        rates = zero_crossing_rate(x, 256)
        assert len(rates) == 4

    def test_classify_vu(self):
        voiced = [math.sin(2*math.pi*200*i/8000) for i in range(2048)]
        labels = classify_voiced_unvoiced(voiced, 256, 0.3, 0.001)
        assert "voiced" in labels

    def test_frame_rms(self):
        x = [1.0] * 2048
        r = frame_rms(x, 1024, 1024)
        assert abs(r[0] - 1.0) < 1e-10

    def test_loudness(self):
        x = [math.sin(2*math.pi*1000*i/44100) for i in range(8192)]
        lc = loudness_curve(x, 2048, 512)
        assert len(lc) > 0

    def test_trim_silence(self):
        x = [0.001]*1000 + [0.5]*2000 + [0.001]*1000
        trimmed = trim_silence(x, -30, 256)
        assert len(trimmed) < len(x)

    def test_silence_ratio(self):
        x = [0.001]*500 + [0.5]*500
        r = silence_ratio(x, -30, 256)
        assert 0 < r < 1

    def test_cfar(self):
        x = [0.1]*50 + [5.0] + [0.1]*50
        dets = cfar_detector(x, 2, 8, 5.0)
        assert 50 in dets

    def test_hysteresis(self):
        x = [0]*10 + [2]*10 + [0]*10
        regions = hysteresis_threshold(x, 1.5, 0.5)
        assert len(regions) >= 1

    def test_consecutive_clipping(self):
        x = [0.5]*10 + [1.0]*10 + [0.5]*10
        regions = consecutive_clipping(x, 0.99, 3)
        assert len(regions) >= 1

    def test_headroom(self):
        x = [0.5]*10
        h = headroom_db(x)
        assert h > 0

    def test_declip(self):
        x = [0.5, 0.8, 1.0, 1.0, 0.8, 0.5]
        fixed = declip(x, 0.99, 5)
        assert fixed[2] < 1.0


class TestSegmentExtra:
    def test_sliding_window(self):
        from dravex.core.segment import sliding_window
        x = list(range(20))
        windows = sliding_window(x, 5, 3)
        assert len(windows) > 0
        assert len(windows[0]) == 5

    def test_zero_crossing_segments(self):
        from dravex.core.segment import zero_crossing_segments
        from dravex.core.signal import Signal
        x = [math.sin(2*math.pi*i/20) for i in range(200)]
        s = Signal(x, 44100)
        segs = zero_crossing_segments(s, 5)
        assert len(segs) > 1

    def test_adaptive_segmentation(self):
        from dravex.core.segment import adaptive_segmentation
        from dravex.core.signal import Signal
        quiet = [0.01] * 1024
        loud = [0.5] * 1024
        s = Signal(quiet + loud + quiet, 44100)
        segs = adaptive_segmentation(s, 256, 4096, 5.0)
        assert len(segs) >= 1
