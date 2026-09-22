"""Splitting by position in the stereo image.

The 2025 Interspeech work on string quartets ("Position also matters")
settles the two-violins question: timbre alone is not enough, because two
violins have none to spare. What separates them is *where they are*. Their
model reaches 8.26 dB on violin 2 across seating arrangements - but only
because it was trained on that instrument pair and that room, and it drops
to -0.36 dB when the players move. There is no pretrained model that does
this for an arbitrary mix.

What is honestly available from a stereo file is the panning itself. If the
engineer panned two guitars left and right, that cue is still in the file and
can be recovered. If everything sits in the middle, it is gone, and no amount
of processing brings it back.

So this module measures first and splits second. `pan_diversity` says how
much positional information the material actually contains; the split is
only offered when there is something to work with. Refusing to split a mono
source is the honest answer, not a failure.
"""

from __future__ import annotations

import numpy as np

from .audio import EPS, _istft, _stft, match_shape


def _pan_map(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin pan position in [-1, 1] and the magnitude that carries it."""
    L, R = np.abs(X[0]), np.abs(X[1])
    total = L + R
    pan = np.where(total > EPS, (R - L) / np.where(total > EPS, total, 1.0), 0.0)
    return pan, total


def pan_diversity(stem: np.ndarray, sr: int, n_fft: int = 4096) -> dict:
    """How much of this stem's energy is actually spread across the image."""
    if stem.shape[0] < 2:
        return {"usable": False, "spread": 0.0, "side_ratio": 0.0, "reason": "mono source"}

    X = _stft(stem, n_fft, n_fft // 4)
    pan, mag = _pan_map(X)
    w = mag / (mag.sum() + EPS)

    mean = float(np.sum(pan * w))
    spread = float(np.sqrt(np.sum(((pan - mean) ** 2) * w)))

    mid = (stem[0] + stem[1]) / 2.0
    side = (stem[0] - stem[1]) / 2.0
    side_ratio = float(np.sqrt(np.mean(side**2)) / (np.sqrt(np.mean(mid**2)) + EPS))

    usable = spread > 0.22 and side_ratio > 0.12
    reason = (
        "clear left/right placement" if usable
        else ("almost entirely centred" if spread <= 0.22 else "too little stereo difference")
    )
    return {
        "usable": bool(usable),
        "spread": round(spread, 3),
        "side_ratio": round(side_ratio, 3),
        "reason": reason,
    }


def split_by_position(
    stem: np.ndarray,
    sr: int,
    names: tuple[str, str, str] = ("left", "centre", "right"),
    width: float = 0.38,
    n_fft: int = 4096,
) -> dict[str, np.ndarray]:
    """Split one stem into left / centre / right layers that sum back to it.

    Soft, overlapping pan windows - a hard cut would chop a source that
    wanders across the image and leave holes on both sides.
    """
    stem = np.asarray(stem, dtype=np.float32)
    if stem.shape[0] < 2:
        return {names[1]: stem.copy()}

    hop = n_fft // 4
    n = stem.shape[-1]
    X = _stft(stem, n_fft, hop)
    pan, _ = _pan_map(X)

    centres = np.array([-1.0, 0.0, 1.0])
    masks = np.stack([np.exp(-((pan - c) ** 2) / (2 * width**2)) for c in centres])
    total = masks.sum(axis=0)
    masks = masks / np.where(total > EPS, total, 1.0)

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        m = masks[i][None, ...]  # same mask for both channels: position, not level
        out[name] = _istft(m * X, hop, n, n_fft)

    residual = stem - sum(out.values())
    anchor = max(out, key=lambda k: float(np.sum(out[k] ** 2)))
    out[anchor] = out[anchor] + residual
    return {k: match_shape(v, stem) for k, v in out.items()}


def describe(stem: np.ndarray, sr: int, label: str) -> str:
    d = pan_diversity(stem, sr)
    if d["usable"]:
        return f"{label}: can be split by position ({d['reason']}, spread {d['spread']})"
    return f"{label}: not splittable by position - {d['reason']}"
