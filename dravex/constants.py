"""Mathematical and DSP constants."""
import math

PI = math.pi
TAU = 2.0 * math.pi
HALF_PI = math.pi / 2.0
SQRT2 = math.sqrt(2.0)
INV_SQRT2 = 1.0 / SQRT2
LN2 = math.log(2.0)
LN10 = math.log(10.0)

DEFAULT_SAMPLE_RATE = 44100
CD_SAMPLE_RATE = 44100
DVD_SAMPLE_RATE = 48000
TELEPHONE_RATE = 8000

DEFAULT_EPSILON = 1e-12

# dB conversion
def amplitude_to_db(amp, ref=1.0):
    if amp <= 0:
        return -120.0
    return 20.0 * math.log10(amp / ref)

def db_to_amplitude(db, ref=1.0):
    return ref * (10.0 ** (db / 20.0))

def power_to_db(power, ref=1.0):
    if power <= 0:
        return -120.0
    return 10.0 * math.log10(power / ref)

def db_to_power(db, ref=1.0):
    return ref * (10.0 ** (db / 10.0))

# frequency conversion
def hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def hz_to_midi(hz):
    if hz <= 0:
        return 0
    return 69 + 12 * math.log2(hz / 440.0)

def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def hz_to_bark(hz):
    return 13.0 * math.atan(0.00076 * hz) + 3.5 * math.atan((hz / 7500.0) ** 2)

def next_power_of_2(n):
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p

def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0

def sinc(x):
    if abs(x) < DEFAULT_EPSILON:
        return 1.0
    return math.sin(PI * x) / (PI * x)

def rms(samples):
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))
