"""Tests for complex signal support."""
import math
from dravex.core.complex_signal import ComplexSignal

class TestComplexSignal:
    def test_create(self):
        cs = ComplexSignal([1, 2, 3], [4, 5, 6])
        assert cs.length == 3

    def test_magnitude(self):
        cs = ComplexSignal([3], [4])
        assert abs(cs.magnitude()[0] - 5.0) < 1e-10

    def test_phase(self):
        cs = ComplexSignal([1], [1])
        assert abs(cs.phase()[0] - math.pi/4) < 1e-10

    def test_conjugate(self):
        cs = ComplexSignal([1], [2])
        conj = cs.conjugate()
        assert conj.imag[0] == -2.0

    def test_add(self):
        a = ComplexSignal([1, 2], [3, 4])
        b = ComplexSignal([5, 6], [7, 8])
        c = a + b
        assert c.real == [6.0, 8.0]
        assert c.imag == [10.0, 12.0]

    def test_multiply_complex(self):
        a = ComplexSignal([1], [2])
        b = ComplexSignal([3], [4])
        c = a * b
        assert abs(c.real[0] - (-5)) < 1e-10
        assert abs(c.imag[0] - 10) < 1e-10

    def test_multiply_scalar(self):
        cs = ComplexSignal([1, 2], [3, 4])
        r = cs * 2
        assert r.real == [2.0, 4.0]

    def test_energy(self):
        cs = ComplexSignal([3], [4])
        assert abs(cs.energy() - 25) < 1e-10

    def test_from_polar(self):
        cs = ComplexSignal.from_polar([1.0], [0.0])
        assert abs(cs.real[0] - 1.0) < 1e-10
        assert abs(cs.imag[0]) < 1e-10

    def test_from_analytic(self):
        x = [math.sin(2*math.pi*10*i/256) for i in range(256)]
        cs = ComplexSignal.from_analytic(x)
        assert cs.length == 256
