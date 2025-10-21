"""Ring buffer for streaming signal processing."""
from dravex.errors import EmptySignalError

class RingBuffer:
    """Fixed-size circular buffer for real-time processing."""

    def __init__(self, capacity):
        self._capacity = capacity
        self._data = [0.0] * capacity
        self._write_pos = 0
        self._read_pos = 0
        self._count = 0

    @property
    def capacity(self):
        return self._capacity

    @property
    def count(self):
        return self._count

    @property
    def is_full(self):
        return self._count >= self._capacity

    @property
    def is_empty(self):
        return self._count == 0

    def push(self, sample):
        self._data[self._write_pos] = sample
        self._write_pos = (self._write_pos + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1
        else:
            self._read_pos = (self._read_pos + 1) % self._capacity

    def push_many(self, samples):
        for s in samples:
            self.push(s)

    def pop(self):
        if self.is_empty:
            raise EmptySignalError()
        val = self._data[self._read_pos]
        self._read_pos = (self._read_pos + 1) % self._capacity
        self._count -= 1
        return val

    def peek(self, offset=0):
        if offset >= self._count:
            raise IndexError(f"Offset {offset} exceeds count {self._count}")
        idx = (self._read_pos + offset) % self._capacity
        return self._data[idx]

    def read_all(self):
        result = []
        for i in range(self._count):
            idx = (self._read_pos + i) % self._capacity
            result.append(self._data[idx])
        return result

    def clear(self):
        self._write_pos = 0
        self._read_pos = 0
        self._count = 0

    def latest(self, n):
        if self._count == 0:
            return []
        n = min(n, self._count)
        result = []
        for i in range(n):
            idx = (self._write_pos - n + i) % self._capacity
            result.append(self._data[idx])
        return result

    def running_mean(self):
        if self._count == 0:
            return 0.0
        return sum(self.read_all()) / self._count

    def running_rms(self):
        if self._count == 0:
            return 0.0
        import math
        data = self.read_all()
        return math.sqrt(sum(s * s for s in data) / len(data))

class OverlapBuffer:
    """Buffer for overlap-add/save processing."""
    def __init__(self, frame_size, hop_size):
        self._frame_size = frame_size
        self._hop_size = hop_size
        self._buffer = [0.0] * frame_size
        self._output = []
        self._write_pos = 0

    def push_frame(self, frame):
        """Add a processed frame to the overlap buffer."""
        while len(self._output) < self._write_pos + self._frame_size:
            self._output.append(0.0)
        for i in range(min(len(frame), self._frame_size)):
            self._output[self._write_pos + i] += frame[i]
        self._write_pos += self._hop_size

    def get_output(self):
        """Return accumulated output samples."""
        return list(self._output)

    def clear(self):
        self._output = []
        self._write_pos = 0

    @property
    def output_length(self):
        return len(self._output)


class DelayLine:
    """Simple delay line for effects processing."""
    def __init__(self, max_delay):
        self._buffer = [0.0] * max_delay
        self._write_pos = 0
        self._max_delay = max_delay

    def write(self, sample):
        self._buffer[self._write_pos] = sample
        self._write_pos = (self._write_pos + 1) % self._max_delay

    def read(self, delay):
        idx = (self._write_pos - int(delay)) % self._max_delay
        return self._buffer[idx]

    def read_interpolated(self, delay):
        int_delay = int(delay)
        frac = delay - int_delay
        s1 = self.read(int_delay)
        s2 = self.read(int_delay + 1)
        return s1 * (1 - frac) + s2 * frac

    def process(self, sample, delay, feedback=0.0):
        output = self.read(delay)
        self.write(sample + output * feedback)
        return output

class SlidingWindow:
    """Efficient sliding window for streaming analysis."""
    def __init__(self, size):
        self._size = size
        self._data = []
        self._sum = 0.0
        self._sum_sq = 0.0

    def push(self, value):
        self._data.append(value)
        self._sum += value
        self._sum_sq += value * value
        if len(self._data) > self._size:
            old = self._data.pop(0)
            self._sum -= old
            self._sum_sq -= old * old

    def mean(self):
        n = len(self._data)
        return self._sum / n if n > 0 else 0.0

    def variance(self):
        n = len(self._data)
        if n < 2:
            return 0.0
        mean_val = self._sum / n
        return self._sum_sq / n - mean_val * mean_val

    def rms(self):
        import math
        n = len(self._data)
        if n == 0:
            return 0.0
        return math.sqrt(self._sum_sq / n)

    @property
    def count(self):
        return len(self._data)

    def get_data(self):
        return list(self._data)
