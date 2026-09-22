"""Does it sound right - not just does it score well.

SDR is the standard number and it has a known ceiling: past roughly 9 dB on
vocals, listeners stop hearing the difference in blind tests, so pushing the
number higher stops meaning anything. The 2025 work on evaluating generative
separation puts numbers on it - BSS-Eval metrics correlate 0.6-0.7 with human
judgement for masking models like ours, but below 0.5 for generative ones,
and MERT-L12 embedding distance is the best single proxy they found.

Our pipeline is a masking pipeline, so SDR-family numbers are still
meaningful here. But they answer the wrong question for a sound engineer.
What he needs to know is not "how many dB" but "what is wrong with this
track": is the guitar still audible behind the keys, and does it have that
warbling underwater sound. So this module reports those two directly.

  bleed     how much of a stem's envelope is explained by its siblings
  artifacts isolated time-frequency islands - the signature of musical noise
  fullness  how much of the parent's energy this stem carries
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from .audio import EPS, _stft

log = logging.getLogger(__name__)


@dataclass
class StemQuality:
    key: str
    label: str
    bleed: float  # 0 = clean, 1 = indistinguishable from siblings
    artifacts: float  # 0 = smooth, 1 = heavily fragmented
    fullness: float  # share of the parent's energy
    verdict: str
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _envelope(x: np.ndarray, bins: int = 2000) -> np.ndarray:
    y = x.mean(axis=0) if x.ndim > 1 else x
    win = max(1, len(y) // bins)
    usable = (len(y) // win) * win
    if usable < win:
        return np.abs(y)
    return np.abs(y[:usable]).reshape(-1, win).mean(axis=1)


def bleed_score(stem: np.ndarray, siblings: list[np.ndarray]) -> float:
    """Highest envelope correlation with any sibling."""
    if not siblings:
        return 0.0
    e = _envelope(stem)
    e = e - e.mean()
    ne = float(np.linalg.norm(e))
    if ne < EPS:
        return 0.0
    worst = 0.0
    for sib in siblings:
        s = _envelope(sib)
        m = min(len(e), len(s))
        s = s[:m] - s[:m].mean()
        ns = float(np.linalg.norm(s))
        if ns < EPS:
            continue
        worst = max(worst, abs(float(np.dot(e[:m], s)) / (float(np.linalg.norm(e[:m])) * ns + EPS)))
    return float(np.clip(worst, 0.0, 1.0))


def artifact_score(stem: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    """Musical noise: loud bins with quiet neighbours in both time and frequency."""
    from scipy.ndimage import uniform_filter

    S = np.abs(_stft(stem, n_fft, n_fft // 4)).mean(axis=0)
    if S.size == 0 or S.max() < EPS:
        return 0.0
    S = S / S.max()
    loud = S > 0.05
    if not loud.any():
        return 0.0
    local = uniform_filter(S, size=(3, 5), mode="nearest")
    # A real partial is supported by its neighbourhood; an artifact is alone.
    isolation = np.clip(1.0 - local / (S + EPS), 0.0, 1.0)
    return float(np.clip(isolation[loud].mean() * 1.8, 0.0, 1.0))


def assess(
    stems: dict[str, np.ndarray],
    parents: dict[str, str],
    labels: dict[str, str],
    sr: int,
    keys: list[str],
) -> dict[str, StemQuality]:
    out: dict[str, StemQuality] = {}
    for key in keys:
        if key not in stems:
            continue
        audio = stems[key]
        parent_key = parents.get(key)
        parent = stems.get(parent_key) if parent_key else stems.get("mix")
        siblings = [v for k, v in stems.items() if k != key and parents.get(k) == parent_key]

        try:
            bleed = bleed_score(audio, siblings)
            art = artifact_score(audio, sr)
            full = 0.0
            if parent is not None:
                pe = float(np.mean(parent**2))
                full = float(np.mean(audio**2) / pe) if pe > EPS else 0.0
        except Exception as exc:  # noqa: BLE001
            log.warning("quality check failed for %s: %s", key, exc)
            continue

        notes: list[str] = []
        if bleed > 0.75:
            notes.append("strong bleed from the neighbouring stems")
        elif bleed > 0.55:
            notes.append("some bleed audible behind it")
        if art > 0.6:
            notes.append("watery / metallic artefacts likely")
        elif art > 0.45:
            notes.append("mild artefacts on quiet passages")
        if full < 0.005:
            notes.append("carries very little of the mix")
        if not notes:
            notes.append("clean")

        verdict = "good" if (bleed < 0.55 and art < 0.45) else ("usable" if (bleed < 0.78 and art < 0.62) else "rough")
        out[key] = StemQuality(key, labels.get(key, key), round(bleed, 3), round(art, 3), round(full, 5), verdict, notes)
    return out


# --------------------------------------------------------------------------
# Optional perceptual distance
# --------------------------------------------------------------------------


def mert_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def mert_distance(reference: np.ndarray, estimate: np.ndarray, sr: int) -> float | None:
    """MSE between MERT layer-12 embeddings - the best perceptual proxy found
    in the 2025 evaluation study. Optional; needs transformers + weights."""
    try:
        import torch
        from transformers import AutoModel

        from .audio import resample

        model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).eval()

        def embed(x: np.ndarray) -> torch.Tensor:
            y = resample(x.mean(axis=0, keepdims=True), sr, 24000)[0]
            with torch.no_grad():
                out = model(torch.tensor(y)[None, :], output_hidden_states=True)
            return out.hidden_states[12].mean(dim=1).squeeze(0)

        a, b = embed(reference), embed(estimate)
        return float(torch.mean((a - b) ** 2))
    except Exception as exc:  # noqa: BLE001
        log.warning("MERT distance unavailable: %s", exc)
        return None
