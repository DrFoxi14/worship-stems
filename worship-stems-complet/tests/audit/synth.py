"""Synthesise worship-like material from a known note list.

The point is ground truth: every test signal is built from an explicit
(start, end, pitch) list, so transcription output can be scored against
what was actually played rather than against a guess.

The synthesis is deliberately not a sine. Real instruments give the
transcriber harmonics to latch onto and vibrato/onset noise to trip over,
and a pure tone would make basic-pitch look far better than it is.
"""

from __future__ import annotations

import numpy as np

SR = 44100


def midi_to_hz(p: float) -> float:
    return 440.0 * (2.0 ** ((p - 69) / 12.0))


def adsr(n: int, sr: int, a=0.01, d=0.08, s=0.7, r=0.12) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    na, nd, nr = int(a * sr), int(d * sr), int(r * sr)
    na, nd, nr = min(na, n), min(nd, n), min(nr, n)
    if na:
        env[:na] = np.linspace(0, 1, na)
    if nd and na + nd <= n:
        env[na:na + nd] = np.linspace(1, s, nd)
    if na + nd < n - nr:
        env[na + nd:n - nr] = s
    if nr:
        env[n - nr:] = np.linspace(env[n - nr - 1] if n - nr - 1 >= 0 else s, 0, nr)
    return env


def voice(f0, n, sr, rng, vibrato_hz=5.2, vibrato_cents=35, breath=0.006):
    """A sung note: strong harmonics, vibrato, a little breath noise."""
    t = np.arange(n) / sr
    vib = (vibrato_cents / 1200.0) * np.sin(2 * np.pi * vibrato_hz * t) * np.log(2)
    phase_f = f0 * np.exp(vib)
    phase = 2 * np.pi * np.cumsum(phase_f) / sr
    amps = [1.0, 0.52, 0.34, 0.20, 0.12, 0.07, 0.04]
    x = sum(a * np.sin(h * phase) for h, a in enumerate(amps, start=1))
    x = x / max(sum(amps), 1e-9)
    x = x + breath * rng.standard_normal(n)
    return (x * adsr(n, sr, a=0.04, d=0.10, s=0.82, r=0.10)).astype(np.float32)


def piano(f0, n, sr, rng):
    """Struck string: bright onset, inharmonic-ish decay."""
    t = np.arange(n) / sr
    amps = [1.0, 0.45, 0.28, 0.18, 0.11, 0.07, 0.05, 0.03]
    x = np.zeros(n, dtype=np.float64)
    for h, a in enumerate(amps, start=1):
        # slight stretch, as a real string has
        f = f0 * h * (1.0 + 0.0003 * h * h)
        x += a * np.sin(2 * np.pi * f * t) * np.exp(-t * (1.2 + 0.35 * h))
    x /= max(sum(amps), 1e-9)
    click = rng.standard_normal(n) * np.exp(-t * 400.0) * 0.05
    return ((x + click) * adsr(n, sr, a=0.002, d=0.25, s=0.35, r=0.15)).astype(np.float32)


def bass(f0, n, sr, rng):
    """Electric bass: fundamental plus a few strong low harmonics."""
    t = np.arange(n) / sr
    amps = [1.0, 0.65, 0.30, 0.15, 0.08]
    x = sum(a * np.sin(2 * np.pi * f0 * h * t) for h, a in enumerate(amps, start=1))
    x /= max(sum(amps), 1e-9)
    x = x * np.exp(-t * 0.9)
    return (x * adsr(n, sr, a=0.006, d=0.12, s=0.6, r=0.10)).astype(np.float32)


PATCHES = {"voice": voice, "piano": piano, "bass": bass}


def render(notes, patch="voice", sr=SR, duration=None, seed=0, gain=0.25):
    """notes: list of (start_s, end_s, midi_pitch). Returns mono float32."""
    rng = np.random.default_rng(seed)
    dur = duration or (max(e for _, e, _ in notes) + 0.5)
    out = np.zeros(int(dur * sr), dtype=np.float32)
    fn = PATCHES[patch]
    for start, end, pitch in notes:
        n = int((end - start) * sr)
        if n <= 0:
            continue
        i = int(start * sr)
        seg = fn(midi_to_hz(pitch), n, sr, rng)
        out[i:i + len(seg)] += seg[: max(0, len(out) - i)]
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * gain).astype(np.float32)


def stereo(x):
    return np.stack([x, x]).astype(np.float32)
