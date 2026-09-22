"""Changing key, one stem at a time.

Worship teams change key constantly - whoever is leading this week sits a
tone lower, so the whole song moves. Pitch-shifting a finished mix smears
everything at once, because one set of settings has to serve a kick drum and
a cymbal and a voice.

With the stems separated, each one can be shifted on its own terms, and two
things follow that a mix cannot do:

  drums stay put   percussion has no key. Shifting a kick down two
                   semitones makes it flabby and shifting a snare up makes
                   it ring, for no musical gain at all. Percussive stems are
                   left exactly as they were.

  per-stem care    a bass shifted down needs different handling from a vocal
                   shifted up, and separated stems allow that.

The generated tracks move too: the ambient pad is re-rendered in the new key
rather than pitch-shifted, and the key written into every file name, chart
and session file follows.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from .analyze import PITCHES, Analysis
from .audio import EPS

log = logging.getLogger(__name__)

# Percussion has no pitch to change.
PERCUSSIVE = {"drums", "kick", "snare", "toms", "hihat", "cymbals"}
# Generated in the target key directly, never shifted.
REGENERATED = {"pad_key", "click", "guide", "click_guide"}


@dataclass
class TransposeReport:
    semitones: int
    from_key: str
    to_key: str
    shifted: list[str]
    left_alone: list[str]
    regenerated: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def transposed_key(key: str, mode: str, semitones: int) -> tuple[str, str]:
    idx = PITCHES.index(key) if key in PITCHES else 0
    return PITCHES[(idx + semitones) % 12], mode


def shift(audio: np.ndarray, sr: int, semitones: float, quality: str = "high") -> np.ndarray:
    """Pitch-shift without changing length."""
    if abs(semitones) < 1e-6:
        return audio
    import librosa

    # librosa counts n_steps in units of bins_per_octave, not semitones. With
    # 24 bins per octave a "2" is one semitone, not two - so convert.
    bins = 24 if quality == "high" else 12
    steps = float(semitones) * bins / 12.0
    out = [
        librosa.effects.pitch_shift(
            np.ascontiguousarray(ch), sr=sr, n_steps=steps, bins_per_octave=bins
        )
        for ch in audio
    ]
    length = min(len(c) for c in out)
    return np.stack([c[:length] for c in out]).astype(np.float32)


def transpose_stems(
    stems: dict[str, np.ndarray],
    analysis: Analysis,
    sr: int,
    semitones: int,
    shift_percussion: bool = False,
) -> tuple[dict[str, np.ndarray], Analysis, TransposeReport]:
    """Move every pitched stem by `semitones`, leaving percussion alone."""
    from .pads import build_pad

    old_key = analysis.key_display
    new_key, new_mode = transposed_key(analysis.key, analysis.mode, semitones)

    out: dict[str, np.ndarray] = {}
    shifted: list[str] = []
    left: list[str] = []
    regenerated: list[str] = []

    for key, audio in stems.items():
        if key in REGENERATED:
            out[key] = audio
            continue
        if key in PERCUSSIVE and not shift_percussion:
            out[key] = audio
            left.append(key)
            continue
        try:
            out[key] = shift(audio, sr, semitones)
            shifted.append(key)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not transpose %s: %s", key, exc)
            out[key] = audio
            left.append(key)

    # The mix and the proof are rebuilt from the shifted parts rather than
    # shifted themselves, so they still describe what is in the folder.
    length = max((v.shape[-1] for v in out.values()), default=0)
    if length:
        from .audio import match_length

        out = {k: match_length(v, length) for k, v in out.items()}

    if "pad_key" in out and length:
        out["pad_key"] = build_pad(new_key, new_mode, length / sr, sr)
        regenerated.append("pad_key")

    analysis.key, analysis.mode = new_key, new_mode
    from .analyze import CAMELOT

    analysis.camelot = CAMELOT.get((new_key, new_mode), "")

    report = TransposeReport(
        semitones=semitones,
        from_key=old_key,
        to_key=analysis.key_display,
        shifted=shifted,
        left_alone=left,
        regenerated=regenerated,
    )
    return out, analysis, report


def suggest_for_singer(analysis: Analysis, comfortable_root: str) -> int:
    """Semitones needed to move the song to a key a singer asked for.

    Returns the smaller move: going up 7 and going down 5 land in the same
    place, and the smaller one keeps the arrangement closer to the record.
    """
    if comfortable_root not in PITCHES or analysis.key not in PITCHES:
        return 0
    delta = (PITCHES.index(comfortable_root) - PITCHES.index(analysis.key)) % 12
    return delta if delta <= 6 else delta - 12


def peak_safe(stems: dict[str, np.ndarray], ceiling_db: float = -1.0) -> dict[str, np.ndarray]:
    """Pitch shifting can add a little level; keep the set under the ceiling."""
    ceiling = 10.0 ** (ceiling_db / 20.0)
    worst = max((float(np.max(np.abs(v))) for v in stems.values() if v.size), default=0.0)
    if worst <= ceiling or worst < EPS:
        return stems
    scale = ceiling / worst
    return {k: (v * scale).astype(np.float32) for k, v in stems.items()}
