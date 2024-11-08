"""Tests for DCT."""
from dravex.transform.dct import dct_ii, dct_iii, dct_energy_compaction

class TestDCT:
    def test_dc_input(self):
        result = dct_ii([1, 1, 1, 1])
        assert abs(result[0] - 8.0) < 1e-8

    def test_inverse(self):
        x = [1.0, 2.0, 3.0, 4.0]
        X = dct_ii(x)
        y = dct_iii(X)
        for a, b in zip(x, y):
            assert abs(a - b) < 1e-6

    def test_energy_compaction(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        compressed = dct_energy_compaction(x, keep_ratio=0.5)
        assert len(compressed) == len(x)
