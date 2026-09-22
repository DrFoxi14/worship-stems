"""Second pass: sharpen the split without breaking the sum.

The first pass assigns every time-frequency bin proportionally, so a bin that
is 55% guitar and 45% keys gets shared out - which is why a first-pass stem
still has its neighbours faintly behind it. The second pass goes back over
the same bins with the first result as its prior and pushes the ambiguous
ones toward whichever stem already owns them, while cutting each stem where
that instrument physically cannot be playing.

Two mechanisms:

  sharpening   raise the mask exponent. p=2 shares a contested bin; p=6 gives
               it almost entirely to the leader. Bleed drops, and the risk is
               chopping a genuinely shared bin, so it is applied gradually.

  band gating  a bass guitar has nothing to say at 8 kHz and a hi-hat has
               nothing at 40 Hz. Energy outside an instrument's plausible
               range is demoted rather than deleted - it moves to the stems
               that can plausibly own it.

Because both mechanisms only redistribute mask weight, and the masks are
renormalised to sum to one afterwards, the parts still add back up to the
parent exactly. The proof file stays a proof.
"""

from __future__ import annotations

import logging

import numpy as np

from .audio import EPS, _istft, _stft, match_shape
from .inventory import PROFILES

log = logging.getLogger(__name__)


def _band_weight(freqs: np.ndarray, key: str, sr: int, floor: float = 0.25) -> np.ndarray:
    """Per-frequency plausibility for an instrument, between `floor` and 1."""
    prof = PROFILES.get(key)
    if prof is None:
        return np.ones_like(freqs)

    lo, hi = prof.f_low, min(prof.f_high * 6.0, sr / 2)
    w = np.ones_like(freqs)

    # Below the fundamental range nothing legitimate can live: roll off hard.
    below = freqs < lo
    if below.any():
        ratio = np.clip(freqs[below] / max(lo, 1e-6), 0.0, 1.0)
        w[below] = floor + (1.0 - floor) * ratio**2

    # Above the harmonic reach, roll off gently - cymbals and air are real.
    above = freqs > hi
    if above.any():
        span = max(sr / 2 - hi, 1e-6)
        ratio = np.clip((freqs[above] - hi) / span, 0.0, 1.0)
        w[above] = floor + (1.0 - floor) * (1.0 - ratio) ** 2
    return w


def refine_partition(
    parent: np.ndarray,
    parts: dict[str, np.ndarray],
    sr: int,
    iterations: int = 2,
    power_start: float = 2.0,
    power_end: float = 5.0,
    band_gating: bool = True,
    band_floor: float = 0.25,
    n_fft: int = 4096,
) -> dict[str, np.ndarray]:
    """Re-mask `parts` against `parent`, sharper each round. Sum stays exact."""
    keys = list(parts)
    if len(keys) < 2 or iterations < 1:
        return parts

    hop = n_fft // 4
    n = parent.shape[-1]
    X_parent = _stft(parent, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    band = None
    if band_gating:
        band = np.stack([_band_weight(freqs, k, sr, band_floor) for k in keys])  # (parts, freq)
        band = band[:, None, :, None]  # broadcast over channel and frame

    current = {k: match_shape(np.asarray(v, dtype=np.float32), parent) for k, v in parts.items()}

    for it in range(iterations):
        t = (it + 1) / iterations
        power = power_start + (power_end - power_start) * t

        M = np.stack([np.abs(_stft(current[k], n_fft, hop)) ** power for k in keys])
        if band is not None:
            M = M * band

        total = M.sum(axis=0)
        silent = total < EPS
        M = np.where(silent[None, ...], 1.0 / len(keys), M / np.where(silent, 1.0, total)[None, ...])

        current = {k: _istft(M[i] * X_parent, hop, n, n_fft) for i, k in enumerate(keys)}

    # Absorb iSTFT edge error so the sum is bit-exact again.
    residual = parent - sum(current.values())
    anchor = max(keys, key=lambda k: float(np.sum(current[k] ** 2)))
    current[anchor] = current[anchor] + residual
    return current


def fold_back(
    parts: dict[str, np.ndarray],
    drop: list[str],
    into: str,
) -> dict[str, np.ndarray]:
    """Remove rejected stems, adding their audio to `into`.

    This is how a stem the inventory rejected disappears without any audio
    being lost: nothing is deleted, it just stops being labelled.
    """
    keep = {k: v for k, v in parts.items() if k not in drop}
    if not keep:
        return parts
    target = into if into in keep else max(keep, key=lambda k: float(np.sum(keep[k] ** 2)))
    for k in drop:
        if k in parts:
            keep[target] = keep[target] + parts[k]
    return keep
