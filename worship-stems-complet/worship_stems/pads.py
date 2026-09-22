"""Ambient pad generator.

MultiTracks sells key pads separately, and for most churches the pad matters
more than any individual stem: it glues the room together between songs and
under spoken word. Since we already detect the key, generating one is free.

Signal chain: a detuned saw stack on root / third / fifth, a gentle low-pass,
a slow breathing envelope, then a Schroeder reverb for the tail.
"""

from __future__ import annotations

import numpy as np

PITCH_INDEX = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def note_hz(pitch: str, octave: int = 3) -> float:
    semitone = PITCH_INDEX.get(pitch, 0) + 12 * (octave - 4) - 9  # A4 = 440
    return 440.0 * (2.0 ** (semitone / 12.0))


def _supersaw(freq: float, n: int, sr: int, voices: int = 5, detune_cents: float = 11.0, seed: int = 0) -> np.ndarray:
    """Band-limited-ish saw stack built from additive harmonics."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    nyq = sr / 2
    for v in range(voices):
        cents = detune_cents * (v - (voices - 1) / 2) / max(1, (voices - 1) / 2)
        f = freq * (2.0 ** (cents / 1200.0))
        # Slow random drift keeps it from sounding static.
        drift = 1.0 + 0.0009 * np.sin(2 * np.pi * (0.05 + 0.03 * v) * t + rng.random() * 6.28)
        phase0 = rng.random() * 2 * np.pi
        wave = np.zeros(n, dtype=np.float32)
        for h in range(1, 13):
            if f * h >= nyq * 0.9:
                break
            wave += (1.0 / h) * np.sin(2 * np.pi * f * h * t * drift + phase0 * h).astype(np.float32)
        out += wave / voices
    return out


def _lowpass(x: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    sos = butter(order, min(cutoff, sr / 2 * 0.98), btype="low", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def _reverb(x: np.ndarray, sr: int, decay: float = 4.5, mix: float = 0.55) -> np.ndarray:
    """Schroeder reverb: four combs into two allpasses."""
    comb_ms = [29.7, 37.1, 41.1, 43.7]
    out = np.zeros_like(x)
    for ms in comb_ms:
        d = max(1, int(sr * ms / 1000.0))
        g = float(10 ** (-3.0 * (ms / 1000.0) / decay))
        # y[n] = x[n] + g*y[n-d]  -- block recursion, d samples at a time.
        y = np.zeros(len(x) + d, dtype=np.float32)
        y[d : d + len(x)] = x
        for start in range(d, len(y), d):
            end = min(len(y), start + d)
            y[start:end] += g * y[start - d : start - d + (end - start)]
        out += y[d : d + len(x)] / len(comb_ms)

    for ms, g in ((5.0, 0.7), (1.7, 0.7)):
        d = max(1, int(sr * ms / 1000.0))
        y = np.zeros(len(out) + d, dtype=np.float32)
        y[d : d + len(out)] = out
        for start in range(d, len(y), d):
            end = min(len(y), start + d)
            y[start:end] += g * y[start - d : start - d + (end - start)]
        out = (-g * out + y[d : d + len(out)] * (1 - g * g)).astype(np.float32)

    return ((1 - mix) * x + mix * out).astype(np.float32)


def _tile(core: np.ndarray, total: int, sr: int, xfade: float = 4.0) -> np.ndarray:
    """Loop a rendered core to `total` samples with equal-power crossfades."""
    n_core = core.shape[-1]
    if n_core >= total:
        return core[..., :total]
    x = min(int(xfade * sr), n_core // 3)
    step = n_core - x
    out = np.zeros((core.shape[0], total + n_core), dtype=np.float32)
    fade_in = np.sin(np.linspace(0, np.pi / 2, x)) ** 2
    fade_out = np.cos(np.linspace(0, np.pi / 2, x)) ** 2
    pos = 0
    first = True
    while pos < total:
        chunk = core.copy()
        if not first:
            chunk[..., :x] *= fade_in
        end = min(out.shape[-1], pos + n_core)
        seg = chunk[..., : end - pos]
        if not first:
            out[..., pos : pos + x] *= fade_out[: max(0, min(x, end - pos))]
        out[..., pos:end] += seg
        first = False
        pos += step
    return out[..., :total]


def build_pad(
    key: str,
    mode: str,
    duration: float,
    sr: int,
    level_db: float = -20.0,
    brightness: float = 1.0,
    core_seconds: float = 24.0,
) -> np.ndarray:
    """Stereo ambient pad in the given key, `duration` seconds long.

    A sustained drone is rendered once as a `core_seconds` loop and tiled,
    which keeps a six-minute pad down to a couple of seconds of work.
    """
    total_n = int(duration * sr)
    if total_n <= 0:
        return np.zeros((2, 0), dtype=np.float32)
    n = min(total_n, int(core_seconds * sr))

    third = 3 if mode == "minor" else 4
    intervals = [(0, 2, 1.0), (7, 2, 0.55), (0, 3, 0.8), (third, 3, 0.45), (7, 3, 0.35), (0, 4, 0.18)]

    root = PITCH_INDEX.get(key, 0)
    channels = []
    for ch in range(2):
        acc = np.zeros(n, dtype=np.float32)
        for i, (semi, octave, amp) in enumerate(intervals):
            pitch = [p for p, v in PITCH_INDEX.items() if v == (root + semi) % 12][0]
            oct_adj = octave + (root + semi) // 12
            f = note_hz(pitch, oct_adj)
            acc += amp * _supersaw(f, n, sr, seed=i * 7 + ch * 101)
        acc = _lowpass(acc, sr, 1800.0 * brightness)
        acc += 0.25 * _lowpass(acc, sr, 500.0 * brightness)
        channels.append(acc)

    core = np.stack([_reverb(c, sr) for c in channels])
    pad = _tile(core, total_n, sr)

    # Breathing envelope + fade in/out, applied across the whole length.
    t = np.arange(total_n) / sr
    breath = 0.82 + 0.18 * np.sin(2 * np.pi * 0.045 * t) * np.sin(2 * np.pi * 0.017 * t + 1.1)
    fade = min(total_n // 2, int(sr * 3.0))
    env = np.ones(total_n, dtype=np.float32)
    if fade > 1:
        env[:fade] = np.linspace(0, 1, fade) ** 2
        env[-fade:] = np.linspace(1, 0, fade) ** 2
    pad = pad * (breath * env)[None, :]

    peak = float(np.max(np.abs(pad))) or 1.0
    target = 10.0 ** (level_db / 20.0)
    return np.clip(pad * (target / peak), -1.0, 1.0).astype(np.float32)
