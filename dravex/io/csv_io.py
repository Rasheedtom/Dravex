"""CSV signal import/export.

Supports time-indexed and raw sample CSV formats
with optional headers and multi-signal output.
"""

def write_csv(filename, signals, headers=None, sample_rate=None):
    """Write signals to CSV file."""
    with open(filename, 'w') as f:
        if headers:
            line = ','.join(headers)
            if sample_rate:
                line = 'time,' + line
            f.write(line + '\n')
        n = max(len(s) for s in signals) if signals else 0
        for i in range(n):
            parts = []
            if sample_rate:
                parts.append(f"{i / sample_rate:.6f}")
            for sig in signals:
                parts.append(f"{sig[i]:.8f}" if i < len(sig) else "0.0")
            f.write(','.join(parts) + '\n')

def read_csv(filename, skip_header=True, time_column=False):
    """Read signals from CSV file."""
    signals = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    start = 1 if skip_header else 0
    if not lines:
        return []
    first_data = lines[start].strip().split(',') if start < len(lines) else []
    n_cols = len(first_data)
    start_col = 1 if time_column else 0
    for _ in range(start_col, n_cols):
        signals.append([])
    for line in lines[start:]:
        parts = line.strip().split(',')
        for j in range(start_col, min(len(parts), n_cols)):
            try:
                signals[j - start_col].append(float(parts[j]))
            except (ValueError, IndexError):
                signals[j - start_col].append(0.0)
    return signals
