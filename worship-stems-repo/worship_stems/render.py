"""Playing the part instead of extracting it.

Once the notes and the timbre are separated from each other, the timbre stops
being fixed. The same part can be played back with the instrument that
recorded it, or with a completely different one - the score does not care.

That matters more than it first sounds. An extracted keys stem out of a dense
worship mix is always going to be smeared, because the keys were sharing
every bar with a guitar and two vocals. But the *part* is simple: a handful
of held chords. Rendered with a clean patch it has no bleed, no holes and no
artefacts at all - it is not the record any more, but for a band playing
along it is often the more useful track of the two.

So this offers both, side by side, and lets the ear decide:

  learned   the instrument as it was measured in this recording
  patch     a clean sound of your choosing, playing the same notes

Nothing here is guesswork about what was played - the notes come from the
transcription, which you can open in any DAW and check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .audio import EPS
from .harmonic import Note
from .resynth import Timbre

log = logging.getLogger(__name__)


@dataclass
class Patch:
    """A sound: how the partials are balanced, and how a note behaves."""

    name: str
    label: str
    harmonics: tuple[float, ...]
    attack_ms: float = 8.0
    decay: float = 0.8  # exponential decay rate; 0 = sustains forever
    release_ms: float = 120.0
    detune_cents: float = 0.0  # >0 stacks a second voice for width
    odd_only: bool = False
    program: int = 0  # General MIDI, for the exported .mid
    tags: tuple[str, ...] = field(default_factory=tuple)

    def profile(self, count: int) -> np.ndarray:
        out = np.zeros(count)
        n = min(count, len(self.harmonics))
        out[:n] = self.harmonics[:n]
        if self.odd_only:
            out[1::2] *= 0.15
        total = out.sum()
        return out / total if total > EPS else out


# A small, deliberately plain set. These are not emulations of real
# instruments - they are clean, predictable sounds that sit well under a band
# and do not fight the rest of the mix.
PATCHES: dict[str, Patch] = {
    "learned": Patch("learned", "As recorded", (), program=0),
    "piano": Patch(
        "piano", "Piano", (1.0, 0.55, 0.32, 0.20, 0.13, 0.08, 0.05, 0.03),
        attack_ms=4, decay=1.6, release_ms=200, program=0, tags=("keys",),
    ),
    "epiano": Patch(
        # Bell-like: the fundamental and a strong upper pair, little between.
        "epiano", "Electric Piano", (1.0, 0.10, 0.42, 0.06, 0.50, 0.04, 0.16, 0.03),
        attack_ms=6, decay=1.1, release_ms=260, detune_cents=4.0, program=4, tags=("keys",),
    ),
    "organ": Patch(
        # Drawbars: the low partials nearly as loud as the fundamental.
        "organ", "Organ", (1.0, 0.88, 0.72, 0.62, 0.14, 0.48, 0.09, 0.36),
        attack_ms=12, decay=0.0, release_ms=90, program=19, tags=("keys",),
    ),
    "pad": Patch(
        # Almost all fundamental - a pad should sit under everything, not on top.
        "pad", "Warm Pad", (1.0, 0.32, 0.11, 0.045, 0.02, 0.01),
        attack_ms=320, decay=0.0, release_ms=900, detune_cents=9.0, program=89, tags=("pads",),
    ),
    "strings": Patch(
        "strings", "Strings", (1.0, 0.72, 0.58, 0.46, 0.37, 0.29, 0.22, 0.16, 0.11, 0.08),
        attack_ms=180, decay=0.1, release_ms=600, detune_cents=6.0, program=48, tags=("pads",),
    ),
    "pluck": Patch(
        "pluck", "Clean Guitar", (1.0, 0.68, 0.52, 0.44, 0.31, 0.26, 0.19, 0.13, 0.09),
        attack_ms=3, decay=2.4, release_ms=140, program=27, tags=("guitar",),
    ),
    "bass": Patch(
        "bass", "Bass", (1.0, 0.28, 0.09, 0.035, 0.015),
        attack_ms=6, decay=1.0, release_ms=120, program=33, tags=("bass",),
    ),
    "flute": Patch(
        "flute", "Soft Lead", (1.0, 0.045, 0.018, 0.008),
        attack_ms=60, decay=0.05, release_ms=220, odd_only=False, program=73, tags=("lead",),
    ),
}

# What to reach for when a stem is rendered without a patch being named.
DEFAULT_PATCH = {
    "keys": "piano",
    "pads": "pad",
    "electric_guitar": "pluck",
    "acoustic_guitar": "pluck",
    "bass": "bass",
    "lead_vocal": "flute",
    "bgv": "pad",
}


def _envelope(length: int, sr: int, patch: Patch, sustain_samples: int) -> np.ndarray:
    attack = max(1, min(length, int(patch.attack_ms * 1e-3 * sr)))
    release = max(1, min(length, int(patch.release_ms * 1e-3 * sr)))
    t = np.arange(length) / sr

    env = np.ones(length)
    env[:attack] = np.linspace(0.0, 1.0, attack) ** 2
    if patch.decay > 0:
        env *= np.exp(-t * patch.decay)
    # Release starts when the note ends, not at the end of the buffer.
    tail_start = min(length - 1, max(attack, sustain_samples))
    tail = length - tail_start
    if tail > 1:
        env[tail_start:] *= np.linspace(1.0, 0.0, tail) ** 2
    return env


def render(
    notes: list[Note],
    duration: float,
    sr: int,
    patch: Patch | str = "piano",
    timbre: Timbre | None = None,
    channels: int = 2,
    max_harmonics: int = 16,
    level_db: float = -14.0,
) -> np.ndarray:
    """Play `notes` with the chosen sound.

    `patch="learned"` uses the timbre measured from the recording; anything
    else uses one of the clean patches, which is how the same part ends up
    playable on a different instrument.
    """
    if isinstance(patch, str):
        patch = PATCHES.get(patch, PATCHES["piano"])
    n = int(duration * sr)
    if n <= 0 or not notes:
        return np.zeros((channels, max(n, 0)), dtype=np.float32)

    if patch.name == "learned" and timbre is not None:
        profile = timbre.amplitudes(max_harmonics)
        total = profile.sum()
        profile = profile / total if total > EPS else profile
        decay, attack_ms, release_ms, detune = 0.7, 8.0, 150.0, 0.0
    else:
        profile = patch.profile(max_harmonics)
        decay, attack_ms, release_ms, detune = patch.decay, patch.attack_ms, patch.release_ms, patch.detune_cents

    voice = Patch(patch.name, patch.label, tuple(profile), attack_ms, decay, release_ms, detune)

    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    for note in notes:
        start = int(note.start * sr)
        sustain = max(1, int((note.end - note.start) * sr))
        tail = int(voice.release_ms * 1e-3 * sr)
        end = min(n, start + sustain + tail)
        if end - start < 8:
            continue
        length = end - start
        tt = np.arange(length) / sr
        env = _envelope(length, sr, voice, sustain)

        wave_l = np.zeros(length)
        wave_r = np.zeros(length)
        for h in range(1, max_harmonics + 1):
            amp = profile[h - 1] if h - 1 < len(profile) else 0.0
            if amp <= 0:
                continue
            f = h * note.f0
            if f >= sr / 2 * 0.92:
                break
            phase = (h * 0.37) % (2 * np.pi)
            wave_l += amp * np.sin(2 * np.pi * f * tt + phase)
            if voice.detune_cents > 0:
                f2 = f * (2.0 ** (voice.detune_cents / 1200.0))
                wave_r += amp * np.sin(2 * np.pi * f2 * tt + phase + 0.4)
            else:
                wave_r += amp * np.sin(2 * np.pi * f * tt + phase)

        gain = float(np.clip(note.amplitude, 0.05, 1.0))
        left[start:end] += wave_l * env * gain
        right[start:end] += wave_r * env * gain

    out = np.stack([left, right] if channels >= 2 else [left])
    peak = float(np.max(np.abs(out)))
    if peak > EPS:
        out = out * (10.0 ** (level_db / 20.0) / peak)
    return out.astype(np.float32)


def render_stem(
    key: str,
    notes: list[Note],
    duration: float,
    sr: int,
    timbre: Timbre | None = None,
    patch_name: str | None = None,
) -> tuple[np.ndarray, str]:
    """Render one stem's part, choosing a sensible sound if none is named."""
    name = patch_name or DEFAULT_PATCH.get(key, "piano")
    audio = render(notes, duration, sr, patch=name, timbre=timbre)
    label = PATCHES.get(name, PATCHES["piano"]).label
    return audio, label


def list_patches() -> list[dict]:
    return [
        {"name": p.name, "label": p.label, "tags": list(p.tags)}
        for p in PATCHES.values()
        if p.name != "learned"
    ]
