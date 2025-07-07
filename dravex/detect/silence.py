"""Silence detection and voice activity detection."""
from dravex.constants import power_to_db

def detect_silence(x, threshold_db=-40, frame_size=1024, hop_size=512):
    """Detect silent frames. Returns list of (start, end) sample indices."""
    n = len(x)
    silent_regions = []
    in_silence = False
    silence_start = 0
    pos = 0
    frame_idx = 0
    while pos + frame_size <= n:
        energy = sum(s * s for s in x[pos:pos + frame_size]) / frame_size
        db = power_to_db(energy)
        if db < threshold_db:
            if not in_silence:
                in_silence = True
                silence_start = pos
        else:
            if in_silence:
                silent_regions.append((silence_start, pos))
                in_silence = False
        pos += hop_size
    if in_silence:
        silent_regions.append((silence_start, n))
    return silent_regions

def voice_activity(x, threshold_db=-30, frame_size=512, hop_size=256):
    """Simple VAD returning list of boolean per frame."""
    n = len(x)
    activity = []
    pos = 0
    while pos + frame_size <= n:
        energy = sum(s * s for s in x[pos:pos + frame_size]) / frame_size
        db = power_to_db(energy)
        activity.append(db >= threshold_db)
        pos += hop_size
    return activity

def trim_silence(x, threshold_db=-40, frame_size=256):
    """Trim leading and trailing silence."""
    n = len(x)
    start = 0
    while start + frame_size <= n:
        energy = sum(s*s for s in x[start:start+frame_size]) / frame_size
        if power_to_db(energy) >= threshold_db:
            break
        start += frame_size
    end = n
    while end - frame_size >= start:
        energy = sum(s*s for s in x[end-frame_size:end]) / frame_size
        if power_to_db(energy) >= threshold_db:
            break
        end -= frame_size
    return x[start:end]

def silence_ratio(x, threshold_db=-40, frame_size=1024):
    """Fraction of signal that is silence."""
    regions = detect_silence(x, threshold_db, frame_size)
    silent_samples = sum(end - start for start, end in regions)
    return silent_samples / max(1, len(x))
