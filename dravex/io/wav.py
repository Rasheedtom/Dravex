"""WAV file reading and writing."""
import struct

def write_wav(filename, samples, sample_rate=44100, bits=16):
    """Write PCM WAV file.

    Supports 8-bit and 16-bit integer PCM formats.
    """
    """Write PCM WAV file."""
    n = len(samples)
    bytes_per_sample = bits // 8
    data_size = n * bytes_per_sample
    max_val = (2 ** (bits - 1)) - 1
    with open(filename, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * bytes_per_sample))
        f.write(struct.pack('<H', bytes_per_sample))
        f.write(struct.pack('<H', bits))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        for s in samples:
            clamped = max(-1.0, min(1.0, s))
            val = int(clamped * max_val)
            if bits == 16:
                f.write(struct.pack('<h', val))
            elif bits == 8:
                f.write(struct.pack('B', val + 128))

def read_wav(filename):
    """Read PCM WAV file."""
    with open(filename, 'rb') as f:
        riff = f.read(4)
        if riff != b'RIFF':
            return [], 0
        f.read(4)
        wave = f.read(4)
        if wave != b'WAVE':
            return [], 0
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id == b'fmt ':
                fmt = struct.unpack('<HHIIHH', f.read(16))
                audio_fmt, channels, sr, _, _, bits = fmt
                if chunk_size > 16:
                    f.read(chunk_size - 16)
            elif chunk_id == b'data':
                max_val = (2 ** (bits - 1))
                samples = []
                bytes_per = bits // 8
                n_samples = chunk_size // bytes_per
                for _ in range(n_samples):
                    raw = f.read(bytes_per)
                    if len(raw) < bytes_per:
                        break
                    if bits == 16:
                        val = struct.unpack('<h', raw)[0]
                    elif bits == 8:
                        val = struct.unpack('B', raw)[0] - 128
                    else:
                        val = 0
                    samples.append(val / max_val)
                return samples, sr
            else:
                f.read(chunk_size)
    return [], 0
