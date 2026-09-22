"""Click track generation.

A usable worship click is not a metronome bleep: the downbeat has to be
unmistakable through in-ears, the subdivision has to sit behind it, and the
count-in has to start before bar 1 without shifting the song. Timing comes
from the detected beat grid, so the click follows tempo drift in a live
recording instead of fighting it.
"""

from __future__ import annotations

import numpy as np

from .analyze import Analysis


def _tick(sr: int, freq: float, ms: float, amp: float, noise: float = 0.0) -> np.ndarray:
    """Short pitched tick with a fast exponential decay - a woodblock, roughly."""
    n = max(4, int(sr * ms / 1000.0))
    t = np.arange(n) / sr
    env = np.exp(-t * (6000.0 / ms))
    body = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 2 * t)
    if noise > 0:
        rng = np.random.default_rng(1)
        body = body + noise * rng.standard_normal(n) * np.exp(-t * 900.0)
    click = (body * env * amp).astype(np.float32)
    # Tiny fade-in so there is no DC step.
    fade = min(n, max(2, int(sr * 0.0005)))
    click[:fade] *= np.linspace(0.0, 1.0, fade)
    return click


def _place(buffer: np.ndarray, sample: np.ndarray, at: int) -> None:
    if at < 0:
        sample = sample[-at:]
        at = 0
    end = min(buffer.shape[-1], at + sample.shape[-1])
    if end <= at:
        return
    buffer[..., at:end] += sample[: end - at]


def build_click(
    analysis: Analysis,
    sr: int,
    total_samples: int,
    count_in_bars: int = 2,
    subdivision: int = 1,
    accent_freq: float = 1600.0,
    beat_freq: float = 1000.0,
    sub_freq: float = 1000.0,
    stereo: bool = True,
) -> tuple[np.ndarray, float]:
    """Return (click audio, count_in_offset_seconds).

    The count-in is placed *before* time zero when there is room (songs almost
    always start with a little silence); if there is not, the returned offset
    tells the caller how much everything else must be shifted.
    """
    beats = np.asarray(analysis.beats, dtype=float)
    if beats.size < 2:
        return np.zeros((2 if stereo else 1, total_samples), dtype=np.float32), 0.0

    bpb = max(1, analysis.beats_per_bar)
    downbeats = set(np.round(np.asarray(analysis.downbeats, dtype=float), 4).tolist())
    period = float(np.median(np.diff(beats)))

    # Count-in beats, extrapolated backwards from the first beat.
    first = float(beats[0])
    count_beats = count_in_bars * bpb
    pre = [first - period * (count_beats - i) for i in range(count_beats)]
    offset = max(0.0, -(min(pre) - 0.15)) if pre else 0.0

    length = total_samples + int(np.ceil(offset * sr))
    buf = np.zeros((1, length), dtype=np.float32)

    accent = _tick(sr, accent_freq, 34, 0.60, noise=0.10)
    normal = _tick(sr, beat_freq, 26, 0.38)
    sub = _tick(sr, sub_freq, 16, 0.15)

    # Count-in: accent on the first beat of each count-in bar.
    for i, t in enumerate(pre):
        _place(buf, accent if i % bpb == 0 else normal, int(round((t + offset) * sr)))

    # Song beats.
    for i, t in enumerate(beats):
        is_down = round(float(t), 4) in downbeats or (not downbeats and i % bpb == 0)
        _place(buf, accent if is_down else normal, int(round((t + offset) * sr)))
        if subdivision > 1:
            step = period / subdivision
            for s in range(1, subdivision):
                _place(buf, sub, int(round((t + offset + s * step) * sr)))

    if stereo:
        buf = np.repeat(buf, 2, axis=0)
    return np.clip(buf, -1.0, 1.0), offset


def tempo_map(analysis: Analysis) -> list[dict]:
    """Per-bar tempo, for DAW import and for spotting a tempo that drifts."""
    downbeats = np.asarray(analysis.downbeats, dtype=float)
    if downbeats.size < 2:
        return [{"bar": 1, "time": 0.0, "bpm": analysis.bpm}]
    bpb = max(1, analysis.beats_per_bar)
    out = []
    for i in range(len(downbeats) - 1):
        bar_len = float(downbeats[i + 1] - downbeats[i])
        bpm = 60.0 * bpb / bar_len if bar_len > 0 else analysis.bpm
        out.append({"bar": i + 1, "time": float(downbeats[i]), "bpm": round(bpm, 2)})
    out.append({"bar": len(downbeats), "time": float(downbeats[-1]), "bpm": out[-1]["bpm"] if out else analysis.bpm})
    return out
