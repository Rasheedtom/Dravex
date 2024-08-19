"""Global configuration for dravex computations."""

class Config:
    """DSP configuration singleton."""
    _instance = None
    _defaults = {
        "default_sample_rate": 44100,
        "fft_size": 1024,
        "hop_size": 256,
        "epsilon": 1e-12,
        "max_iterations": 10000,
        "window_type": "hann",
        "overlap_ratio": 0.5,
        "filter_order": 8,
        "kalman_process_noise": 1e-5,
        "kalman_measurement_noise": 1e-2,
        "lpc_order": 12,
        "mfcc_n_coeffs": 13,
        "pitch_min_hz": 50.0,
        "pitch_max_hz": 2000.0,
    }

    def __init__(self):
        self._values = dict(self._defaults)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._values:
            return self._values[name]
        raise AttributeError(f"Unknown config: {name}")

    def set(self, key, value):
        if key not in self._defaults:
            raise KeyError(f"Unknown config: {key}")
        self._values[key] = value
        return self

    def snapshot(self):
        return dict(self._values)

    def restore(self, snap):
        for k, v in snap.items():
            if k in self._defaults:
                self._values[k] = v
