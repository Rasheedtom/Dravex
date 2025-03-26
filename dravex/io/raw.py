"""Raw PCM signal I/O."""
import struct

def write_raw_f32(filename, samples):
    """Write raw 32-bit float PCM."""
    with open(filename, 'wb') as f:
        for s in samples:
            f.write(struct.pack('<f', s))

def read_raw_f32(filename, count=-1):
    """Read raw 32-bit float PCM."""
    samples = []
    with open(filename, 'rb') as f:
        while True:
            if count >= 0 and len(samples) >= count:
                break
            data = f.read(4)
            if len(data) < 4:
                break
            samples.append(struct.unpack('<f', data)[0])
    return samples

def write_raw_i16(filename, samples):
    """Write raw 16-bit integer PCM."""
    with open(filename, 'wb') as f:
        for s in samples:
            val = int(max(-1.0, min(1.0, s)) * 32767)
            f.write(struct.pack('<h', val))

def read_raw_i16(filename, count=-1):
    """Read raw 16-bit integer PCM."""
    samples = []
    with open(filename, 'rb') as f:
        while True:
            if count >= 0 and len(samples) >= count:
                break
            data = f.read(2)
            if len(data) < 2:
                break
            val = struct.unpack('<h', data)[0]
            samples.append(val / 32767.0)
    return samples

def samples_to_bytes(samples, bits=16):
    """Convert float samples to byte string."""
    max_val = (2 ** (bits - 1)) - 1
    result = bytearray()
    for s in samples:
        val = int(max(-1.0, min(1.0, s)) * max_val)
        if bits == 16:
            result.extend(struct.pack('<h', val))
        elif bits == 8:
            result.append(val + 128)
    return bytes(result)
