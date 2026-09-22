"""Is the over-segmentation basic-pitch's fault, or the parameters'?"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/claude/worship-stems/worship-stems")

import mir_eval  # noqa: E402

from material import MELODY, PIANO, SATB  # noqa: E402
from synth import SR, render, stereo  # noqa: E402

from worship_stems.harmonic import transcribe  # noqa: E402

ONSET_TOL, CENT_TOL = 0.05, 50.0


def to_arrays(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros((0,))
    iv = np.array([[s, e] for s, e, _ in notes], dtype=float)
    p = np.array([440.0 * 2.0 ** ((x - 69) / 12.0) for _, _, x in notes], dtype=float)
    return iv, p


def score(ref, est_notes):
    ri, rp = to_arrays(ref)
    ei, ep = to_arrays(est_notes)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=ONSET_TOL,
        pitch_tolerance=CENT_TOL, offset_ratio=None)
    return p, r, f


CASES = {
    "melodie solo": (MELODY, "voice"),
    "cor SATB": (SATB, "voice"),
    "pian": (PIANO, "piano"),
}

GRID = [
    # (onset, frame, min_len_ms)
    (0.5, 0.3, 60),    # ce e in cod acum
    (0.5, 0.3, 128),   # default basic-pitch
    (0.5, 0.3, 250),
    (0.5, 0.3, 400),
    (0.7, 0.3, 250),
    (0.7, 0.5, 250),
    (0.3, 0.3, 250),
    (0.8, 0.6, 400),
    (0.9, 0.7, 500),
]


def main():
    for name, (ref, patch) in CASES.items():
        audio = stereo(render(ref, patch=patch, seed=7))
        print(f"\n{name}  (ref {len(ref)} note)")
        print(f"  {'onset':>6} {'frame':>6} {'min_ms':>7}   {'est':>4}  {'P':>5} {'R':>5} {'F1':>5}")
        best = None
        for on, fr, ml in GRID:
            t0 = time.time()
            est, _ = transcribe(audio, SR, onset_threshold=on,
                                frame_threshold=fr, min_note_len_ms=ml)
            en = [(n.start, n.end, n.pitch) for n in est]
            p, r, f = score(ref, en)
            mark = ""
            if best is None or f > best[0]:
                best, mark = (f, on, fr, ml), ""
            print(f"  {on:>6.1f} {fr:>6.1f} {ml:>7.0f}   {len(en):>4}  "
                  f"{p:5.2f} {r:5.2f} {f:5.2f}   {time.time()-t0:4.1f}s{mark}")
        print(f"  -> cel mai bun F1 {best[0]:.2f} la onset={best[1]} frame={best[2]} min_len={best[3]}ms")


if __name__ == "__main__":
    main()
