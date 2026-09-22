"""Putting the song on a grid.

A studio multitrack is grid-locked: bar 17 starts exactly where the click
says bar 17 starts, every time. A recording of a band playing is not. It
drifts - a little faster into the chorus, a little slower at the end of a
phrase - which is what makes it sound human and what makes it useless to
play along to. Give a band a click taken from a drifting record and they
spend the whole song fighting it.

The fix is not a better click. It is to move the audio onto a steady grid:
stretch each beat interval by exactly the amount needed for its downbeat to
land where a constant tempo says it should. Then the tracks and the click
agree, and the band plays with the record instead of chasing it.

The stretching is a phase vocoder driven by a variable rate - the rate
changes every frame, following the map from where the beat actually is to
where it ought to be. Every stem gets the *same* map, so they stay locked to
each other to the sample.

What this costs: the performance is no longer exactly what was played. A
drummer's deliberate pull-back before a chorus gets flattened. That is a real
loss, which is why it is off by default and why the untouched stems are still
in the folder.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from .analyze import Analysis
from .audio import EPS

log = logging.getLogger(__name__)


@dataclass
class RegridReport:
    target_bpm: float
    beats_used: int
    max_stretch: float  # largest local rate, 1.0 = untouched
    mean_stretch: float
    drift_before_ms: float  # how far the record wandered
    duration_before: float
    duration_after: float

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# The map
# --------------------------------------------------------------------------


def build_warp(analysis: Analysis, target_bpm: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (source beat times, target beat times, bpm).

    The target grid starts at the first detected beat, so the beginning of
    the song does not slide away from the count-in.
    """
    beats = np.asarray(analysis.beats, dtype=float)
    if beats.size < 4:
        return beats, beats, analysis.bpm

    intervals = np.diff(beats)
    # The median beat is a better anchor than the mean: a couple of missed
    # beats would drag an average badly.
    bpm = float(target_bpm or (60.0 / np.median(intervals)))
    period = 60.0 / bpm

    target = beats[0] + np.arange(beats.size) * period
    return beats, target, bpm


def drift_ms(source: np.ndarray, target: np.ndarray) -> float:
    if source.size == 0:
        return 0.0
    return float(np.max(np.abs(source - target)) * 1000.0)


# --------------------------------------------------------------------------
# Variable-rate stretch
# --------------------------------------------------------------------------


def _variable_phase_vocoder(
    spec: np.ndarray,
    time_steps: np.ndarray,
    hop: int,
) -> np.ndarray:
    """Phase vocoder that follows an arbitrary time path.

    librosa's own phase vocoder takes a single constant rate, which cannot
    express "speed up slightly through bar 12 and slow down again". This is
    the same algorithm with the frame positions supplied per output frame.
    """
    n_freq, n_frames = spec.shape
    out = np.zeros((n_freq, len(time_steps)), dtype=np.complex64)

    expected = 2.0 * np.pi * hop * np.arange(n_freq) / (2 * (n_freq - 1))
    phase_acc = np.angle(spec[:, 0])

    padded = np.concatenate([spec, np.zeros((n_freq, 2), dtype=spec.dtype)], axis=1)

    for i, step in enumerate(time_steps):
        base = int(np.floor(step))
        frac = step - base
        if base >= n_frames:
            break
        a, b = padded[:, base], padded[:, base + 1]
        mag = (1.0 - frac) * np.abs(a) + frac * np.abs(b)
        out[:, i] = mag * np.exp(1j * phase_acc)

        # Advance the phase by the measured deviation from the expected rate.
        delta = np.angle(b) - np.angle(a) - expected
        delta = delta - 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
        phase_acc += expected + delta

    return out


def time_steps_from_warp(
    source: np.ndarray,
    target: np.ndarray,
    duration: float,
    sr: int,
    hop: int,
) -> np.ndarray:
    """Output frame -> fractional input frame, following the beat map."""
    if source.size < 2:
        return np.arange(0, duration * sr / hop)

    out_duration = float(target[-1] + (duration - source[-1]))
    n_out = max(1, int(np.ceil(out_duration * sr / hop)))
    out_times = np.arange(n_out) * hop / sr

    # Map output time back to input time: the inverse of target -> source.
    src = np.concatenate([[0.0], source, [duration]])
    dst = np.concatenate([[0.0], target, [out_duration]])
    in_times = np.interp(out_times, dst, src)
    return in_times * sr / hop


def regrid(
    audio: np.ndarray,
    analysis: Analysis,
    sr: int,
    target_bpm: float | None = None,
    n_fft: int = 2048,
    warp: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, RegridReport]:
    """Stretch `audio` so its beats land on a constant grid."""
    import librosa

    hop = n_fft // 4
    duration = audio.shape[-1] / sr

    if warp is None:
        source, target, bpm = build_warp(analysis, target_bpm)
    else:
        source, target = warp
        bpm = float(target_bpm or analysis.bpm)

    if source.size < 4:
        return audio, RegridReport(bpm, 0, 1.0, 1.0, 0.0, duration, duration)

    steps = time_steps_from_warp(source, target, duration, sr, hop)

    out_channels = []
    for ch in range(audio.shape[0]):
        spec = librosa.stft(np.ascontiguousarray(audio[ch]), n_fft=n_fft, hop_length=hop)
        warped = _variable_phase_vocoder(spec, steps, hop)
        out_channels.append(librosa.istft(warped, hop_length=hop, n_fft=n_fft))

    length = min(len(c) for c in out_channels)
    result = np.stack([c[:length] for c in out_channels]).astype(np.float32)

    rates = np.diff(steps)
    rates = rates[np.isfinite(rates) & (rates > EPS)]
    report = RegridReport(
        target_bpm=round(bpm, 2),
        beats_used=int(source.size),
        max_stretch=round(float(np.max(rates)) if rates.size else 1.0, 4),
        mean_stretch=round(float(np.mean(rates)) if rates.size else 1.0, 4),
        drift_before_ms=round(drift_ms(source, target), 1),
        duration_before=round(duration, 2),
        duration_after=round(length / sr, 2),
    )
    return result, report


def regrid_all(
    stems: dict[str, np.ndarray],
    analysis: Analysis,
    sr: int,
    target_bpm: float | None = None,
    skip: tuple[str, ...] = ("click", "guide", "click_guide", "pad_key"),
) -> tuple[dict[str, np.ndarray], Analysis, RegridReport | None]:
    """Apply one shared map to every stem, and move the analysis with it.

    The same map for all of them is the whole point: a per-stem map would
    pull them apart.
    """
    source, target, bpm = build_warp(analysis, target_bpm)
    if source.size < 4:
        return stems, analysis, None

    out: dict[str, np.ndarray] = {}
    report: RegridReport | None = None
    for key, audio in stems.items():
        if key in skip:
            out[key] = audio
            continue
        warped, rep = regrid(audio, analysis, sr, target_bpm=bpm, warp=(source, target))
        out[key] = warped
        report = report or rep

    if report is None:
        return stems, analysis, None

    # Everything the analysis knows about time has moved with the audio.
    duration = analysis.duration
    out_duration = report.duration_after
    src = np.concatenate([[0.0], source, [duration]])
    dst = np.concatenate([[0.0], target, [out_duration]])

    def move(times):
        return [float(v) for v in np.interp(np.asarray(times, dtype=float), src, dst)]

    analysis.beats = move(analysis.beats)
    analysis.downbeats = move(analysis.downbeats)
    for section in analysis.sections:
        section.start, section.end = move([section.start, section.end])
    analysis.bpm = report.target_bpm
    analysis.duration = out_duration

    # Trim or pad every stem to one length so nothing drifts at the end.
    longest = max(v.shape[-1] for v in out.values())
    from .audio import match_length

    out = {k: match_length(v, longest) for k, v in out.items()}
    return out, analysis, report
