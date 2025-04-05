"""JSON signal serialization."""
import json

def to_json(signal_samples, sample_rate=44100, metadata=None):
    """Serialize signal to JSON string."""
    obj = {
        "sample_rate": sample_rate,
        "length": len(signal_samples),
        "samples": [round(s, 8) for s in signal_samples],
    }
    if metadata:
        obj["metadata"] = metadata
    return json.dumps(obj)

def from_json(json_str):
    """Deserialize signal from JSON string."""
    obj = json.loads(json_str)
    samples = [float(s) for s in obj.get("samples", [])]
    sample_rate = int(obj.get("sample_rate", 44100))
    metadata = obj.get("metadata")
    return samples, sample_rate, metadata

def save_json(filename, signal_samples, sample_rate=44100, metadata=None):
    """Save signal to JSON file."""
    with open(filename, 'w') as f:
        f.write(to_json(signal_samples, sample_rate, metadata))

def load_json(filename):
    """Load signal from JSON file."""
    with open(filename, 'r') as f:
        return from_json(f.read())

def multi_signal_json(signals, names, sample_rate=44100):
    """Serialize multiple named signals."""
    obj = {
        "sample_rate": sample_rate,
        "signals": {}
    }
    for name, sig in zip(names, signals):
        obj["signals"][name] = [round(s, 8) for s in sig]
    return json.dumps(obj)
