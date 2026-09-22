"""Score basic-pitch against known ground truth on worship-shaped material."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/claude/worship-stems/worship-stems")

import mir_eval  # noqa: E402
import soundfile as sf  # noqa: E402

from material import CASES  # noqa: E402
from synth import SR, render, stereo  # noqa: E402

from worship_stems.harmonic import transcribe  # noqa: E402

ONSET_TOL = 0.05   # 50 ms, the MIREX convention
CENT_TOL = 50.0    # half a semitone


def to_arrays(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros((0,))
    iv = np.array([[s, e] for s, e, _ in notes], dtype=float)
    pitch = np.array([440.0 * 2.0 ** ((p - 69) / 12.0) for _, _, p in notes], dtype=float)
    return iv, pitch


def octave_errors(ref_notes, est_notes, tol=ONSET_TOL):
    """Estimated notes that match a reference onset but land in the wrong octave."""
    bad = 0
    for s, _e, p in est_notes:
        for rs, _re, rp in ref_notes:
            if abs(s - rs) <= tol:
                d = p - rp
                if d != 0 and d % 12 == 0:
                    bad += 1
                    break
    return bad


def run_case(name, patch, notes, label=""):
    audio = render(notes, patch=patch, seed=7)
    est, _midi = transcribe(stereo(audio), SR)
    est_notes = [(n.start, n.end, n.pitch) for n in est]

    ref_iv, ref_p = to_arrays(notes)
    est_iv, est_p = to_arrays(est_notes)

    # onset + pitch only (offsets are not meaningful for sustained choir parts)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=ONSET_TOL, pitch_tolerance=CENT_TOL,
        offset_ratio=None,
    )
    oct_bad = octave_errors(notes, est_notes)
    print(f"  {name+label:<24} ref {len(notes):>3}  est {len(est_notes):>3}   "
          f"P {p:5.2f}  R {r:5.2f}  F1 {f:5.2f}   octave-errors {oct_bad}")
    return dict(name=name, ref=len(notes), est=len(est_notes), p=p, r=r, f=f, oct=oct_bad)


def main():
    print("\nbasic-pitch pe material de laudă cu ground truth exact")
    print(f"(toleranță onset {ONSET_TOL*1000:.0f} ms, pitch {CENT_TOL:.0f} cenți, offset ignorat)\n")
    results = []
    for name, patch, notes in CASES:
        results.append(run_case(name, patch, notes))

    # And the case the pipeline actually hands it: a stem with bleed left in.
    print("\n  cu bleed (stem imperfect, vecinul la -18 dB):")
    from material import PIANO, SATB
    satb = render(SATB, patch="voice", seed=7)
    pia = render(PIANO, patch="piano", seed=9)
    m = min(len(satb), len(pia))
    mixed = (satb[:m] + 0.126 * pia[:m]).astype(np.float32)
    est, _ = transcribe(stereo(mixed), SR)
    est_notes = [(n.start, n.end, n.pitch) for n in est]
    ref_iv, ref_p = to_arrays(SATB)
    est_iv, est_p = to_arrays(est_notes)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=ONSET_TOL,
        pitch_tolerance=CENT_TOL, offset_ratio=None)
    print(f"  {'Cor SATB + bleed pian':<24} ref {len(SATB):>3}  est {len(est_notes):>3}   "
          f"P {p:5.2f}  R {r:5.2f}  F1 {f:5.2f}")
    return results


if __name__ == "__main__":
    main()
