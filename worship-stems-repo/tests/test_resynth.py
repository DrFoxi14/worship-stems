"""The loop: score -> timbre -> synthesis -> residual -> missed notes.

What is checked is not that the resynthesis sounds identical - it will not,
it is an additive model of a real instrument. What is checked is that the
loop behaves like an honest measurement: it explains a real fraction of the
signal, the residual shrinks as it finds more notes, and the number it
reports is one you could verify by listening to the residual.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worship_stems import audio as A  # noqa: E402
from worship_stems.harmonic import Note  # noqa: E402
from worship_stems.resynth import (  # noqa: E402
    explain,
    fill_from_synthesis,
    fit_gain,
    learn_timbre,
    synthesize,
)

SR = 44100
DUR = 8.0
N = int(SR * DUR)
rng = np.random.default_rng(31)

# A simple line with a distinctive timbre: strong 3rd harmonic, weak 2nd.
PROFILE = np.array([1.0, 0.15, 0.7, 0.2, 0.35, 0.1, 0.12, 0.05])
PITCHES = [55, 57, 59, 60, 59, 57, 55, 55]  # G3 A3 B3 C4 B3 A3 G3 G3
NOTE_LEN = 1.0


def build_line() -> tuple[np.ndarray, list[Note]]:
    y = np.zeros(N)
    notes: list[Note] = []
    for i, pitch in enumerate(PITCHES):
        start = i * NOTE_LEN
        end = start + NOTE_LEN * 0.9
        a, b = int(start * SR), int(end * SR)
        tt = np.arange(b - a) / SR
        f0 = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        wave = np.zeros(b - a)
        for h, amp in enumerate(PROFILE, start=1):
            wave += amp * np.sin(2 * np.pi * f0 * h * tt)
        env = np.exp(-tt * 0.7)
        env[: int(0.008 * SR)] *= np.linspace(0, 1, int(0.008 * SR)) ** 2
        y[a:b] += wave * env
        notes.append(Note(start, end, pitch, 0.8))
    y = y / np.max(np.abs(y)) * 0.6
    return np.stack([y, y * 0.98]).astype(np.float32), notes


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return cond


def main() -> int:
    ok = True
    stem, true_notes = build_line()

    print("\n[1] the timbre is learned from the recording, not assumed")
    timbre = learn_timbre(stem, true_notes, SR)
    learned = timbre.amplitudes(8)
    expected = PROFILE / PROFILE.sum()
    print(f"      learned  {np.round(learned[:5], 3)}")
    print(f"      actual   {np.round(expected[:5], 3)}")
    ok &= check("it measured real notes", timbre.observations >= len(PITCHES) - 1,
                f"{timbre.observations} notes")
    ok &= check("the strong 3rd harmonic is found", learned[2] > learned[1],
                f"h3={learned[2]:.3f} vs h2={learned[1]:.3f}")
    ok &= check("the fundamental leads", float(np.argmax(learned)) == 0.0, f"peak at h{int(np.argmax(learned))+1}")

    print("\n[2] synthesis from the score explains most of the signal")
    synth = synthesize(true_notes, timbre, DUR, SR, channels=2)
    synth = A.match_shape(synth, stem) * fit_gain(stem, synth)
    residual = stem - synth
    explained = 1.0 - float(np.mean(residual**2)) / float(np.mean(stem**2))
    print(f"      explained {explained*100:.1f}% of the energy")
    ok &= check("a real fraction is accounted for", explained > 0.35, f"{explained*100:.1f}%")

    print("\n[3] the loop reports what it could not explain")
    synthesis, resid, report = explain(stem, SR, "electric_guitar", "Guitars", rounds=2, notes=true_notes)
    print(f"      rounds {report.rounds}, notes {report.notes_initial} -> {report.notes_final}, "
          f"explained {report.explained*100:.1f}%, residual {report.residual_db} dB")
    ok &= check("it ran", report.rounds >= 1)
    ok &= check("the explained share is reported honestly",
                0.0 < report.explained < 1.0, f"{report.explained}")
    ok &= check("residual is below the signal", report.residual_db < 0, f"{report.residual_db} dB")
    ok &= check("synthesis plus residual is the original",
                A.reconstruction_error_db(stem, [synthesis, resid]) < -100)

    print("\n[4] the synthesis can fill what the cut destroyed")
    noise = np.stack([rng.standard_normal(N), rng.standard_normal(N)]).astype(np.float32)
    burst = np.zeros(N)
    for start in (1.2, 3.4, 5.6):
        i = int(start * SR)
        L = int(0.7 * SR)
        tt = np.arange(L) / SR
        burst[i : i + L] += 3.0 * np.exp(-tt * 3.0)
    interferer = (noise * burst).astype(np.float32)
    mix = (stem + interferer).astype(np.float32)
    parts = A.partition_exact(mix, {"g": stem * 0.85, "n": interferer * 0.9}, power=2.0)
    cut = parts["g"]

    # Measure where the damage is.
    w0, w1 = int(1.2 * SR), int(1.9 * SR)
    sdr_cut = A.si_sdr(stem[:, w0:w1], cut[:, w0:w1])
    filled, count = fill_from_synthesis(cut, mix, synthesis, SR)
    sdr_filled = A.si_sdr(stem[:, w0:w1], filled[:, w0:w1])
    print(f"      filled {count:,} bins; damaged window {sdr_cut:.2f} -> {sdr_filled:.2f} dB SI-SDR")
    ok &= check("it filled something", count > 0)
    ok &= check("the damaged window improves", sdr_filled > sdr_cut + 0.5,
                f"{sdr_filled:.2f} vs {sdr_cut:.2f}")

    print("\n[5] no notes means no claims")
    silence = np.zeros((2, SR * 2), dtype=np.float32)
    s2, r2, rep2 = explain(silence, SR, "x", "X", rounds=1, notes=[])
    ok &= check("nothing synthesised", float(np.max(np.abs(s2))) == 0.0)
    ok &= check("residual is the input", A.reconstruction_error_db(silence, [r2]) <= 0)
    ok &= check("and it says so", rep2.notes_initial == 0)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
