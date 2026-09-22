"""Does the better setting survive messier material?

Synthetic material is unfairly clean. Real worship recordings have room
ambience, softer entries and wider vibrato, which is exactly what onset
detection struggles with. If the gain only exists on clean synthesis it is
not a gain.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/claude/worship-stems/worship-stems")

import mir_eval  # noqa: E402

from material import MELODY, SATB  # noqa: E402
from synth import SR, midi_to_hz, render, stereo  # noqa: E402

from worship_stems.harmonic import transcribe  # noqa: E402

rng = np.random.default_rng(11)


def reverb(x, sr=SR, rt60=1.4, mix=0.35):
    """Cheap exponential-decay room, like a church."""
    n = int(rt60 * sr)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / (rt60 * sr / 6.9))
    ir[0] = 1.0
    wet = np.convolve(x, ir)[: len(x)]
    wet /= max(float(np.max(np.abs(wet))), 1e-9)
    return ((1 - mix) * x + mix * wet * float(np.max(np.abs(x)))).astype(np.float32)


def soften_onsets(notes, jitter=0.06):
    """Human timing: entries are not all exactly on the grid."""
    out = []
    for s, e, p in notes:
        d = float(rng.normal(0, jitter))
        out.append((max(0.0, s + d), max(0.05, e + d), p))
    return out


def to_arrays(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros((0,))
    iv = np.array([[s, e] for s, e, _ in notes], dtype=float)
    p = np.array([midi_to_hz(x) for _, _, x in notes], dtype=float)
    return iv, p


def score(ref, est):
    ri, rp = to_arrays(ref)
    ei, ep = to_arrays(est)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    return p, r, f


SETTINGS = {
    "actual (0.5/0.3/60ms)": (0.5, 0.3, 60),
    "propus  (0.7/0.4/250ms)": (0.7, 0.4, 250),
    "propus  (0.8/0.6/400ms)": (0.8, 0.6, 400),
}


def variants(ref, patch):
    yield "curat", ref, stereo(render(ref, patch=patch, seed=7))
    hum = soften_onsets(ref)
    yield "timing uman", hum, stereo(render(hum, patch=patch, seed=8))
    yield "cu reverb de biserică", hum, stereo(reverb(render(hum, patch=patch, seed=9)))
    noisy = render(hum, patch=patch, seed=10)
    noisy = (noisy + 0.004 * rng.standard_normal(len(noisy))).astype(np.float32)
    yield "reverb + zgomot", hum, stereo(reverb(noisy))


def main():
    for name, ref0, patch in (("MELODIE SOLO", MELODY, "voice"), ("COR SATB", SATB, "voice")):
        print(f"\n=== {name} (ref {len(ref0)} note) ===")
        for vname, ref, audio in variants(ref0, patch):
            print(f"\n  {vname}")
            for label, (on, fr, ml) in SETTINGS.items():
                est, _ = transcribe(audio, SR, onset_threshold=on,
                                    frame_threshold=fr, min_note_len_ms=ml)
                en = [(n.start, n.end, n.pitch) for n in est]
                p, r, f = score(ref, en)
                print(f"    {label:<26} est {len(en):>4}   P {p:5.2f}  R {r:5.2f}  F1 {f:5.2f}")


if __name__ == "__main__":
    main()
