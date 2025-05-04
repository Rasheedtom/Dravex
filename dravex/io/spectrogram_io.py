"""Spectrogram serialization and rendering."""
import math
from dravex.constants import amplitude_to_db

def spectrogram_to_ascii(spectrogram, n_rows=20, n_cols=80):
    """Render spectrogram as ASCII art."""
    if not spectrogram:
        return ""
    n_frames = len(spectrogram)
    n_bins = len(spectrogram[0])
    chars = " .:-=+*#%@"
    db_frames = []
    for frame in spectrogram:
        db_frames.append([amplitude_to_db(m) for m in frame])
    all_db = [d for frame in db_frames for d in frame]
    min_db = min(all_db) if all_db else -80
    max_db = max(all_db) if all_db else 0
    rng = max_db - min_db
    if rng < 1:
        rng = 1
    lines = []
    for row in range(n_rows - 1, -1, -1):
        line = []
        for col in range(n_cols):
            fi = int(col * n_frames / n_cols)
            bi = int(row * n_bins / n_rows)
            if fi < len(db_frames) and bi < len(db_frames[fi]):
                val = (db_frames[fi][bi] - min_db) / rng
            else:
                val = 0.0
            idx = int(val * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            line.append(chars[idx])
        lines.append(''.join(line))
    return '\n'.join(lines)

def spectrogram_to_matrix(spectrogram):
    """Convert spectrogram list-of-lists to flat matrix dict."""
    if not spectrogram:
        return {"rows": 0, "cols": 0, "data": []}
    return {
        "rows": len(spectrogram[0]),
        "cols": len(spectrogram),
        "data": [[spectrogram[t][f] for t in range(len(spectrogram))]
                 for f in range(len(spectrogram[0]))],
    }

def save_spectrogram_csv(filename, spectrogram, sample_rate=44100, hop_size=512):
    """Save spectrogram to CSV."""
    with open(filename, 'w') as f:
        n_frames = len(spectrogram)
        n_bins = len(spectrogram[0]) if spectrogram else 0
        header = ','.join([f"t={i*hop_size/sample_rate:.4f}" for i in range(n_frames)])
        f.write('freq,' + header + '\n')
        for b in range(n_bins):
            freq = b * sample_rate / (2 * (n_bins - 1))
            vals = ','.join([f"{spectrogram[t][b]:.6f}" for t in range(n_frames)])
            f.write(f"{freq:.1f},{vals}\n")
