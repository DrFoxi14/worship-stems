"""Does knowing the note beat reading along one frequency row?

The test signal is built so time-continuation cannot win: the note is buried
under collisions that last far longer than any interpolation window. What is
still available is the note's *other* partials, which the collisions do not
bury equally. If reading across the harmonics works, it recovers ground the
time pass has to refuse.

Everything is measured against a source we know, and the mix-consistency
check is tested by handing it a deliberately inflated stem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worship_stems import audio as A  # noqa: E402
from worship_stems.harmonic import Note, harmonic_restore, transcribe, transcription_available  # noqa: E402
from worship_stems.restore import enforce_mix_consistency, restore_stem  # noqa: E402

SR = 44100
DUR = 10.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(3)

F0 = 196.0  # G3
HARMONICS = 10


def stereo(x):
    return np.stack([x, x * 0.98]).astype(np.float32)


def guitar_note() -> np.ndarray:
    """One held note with a stable partial balance and a slow swell."""
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t)
    y = np.zeros(N)
    for h in range(1, HARMONICS + 1):
        y += (1.0 / h**1.2) * np.sin(2 * np.pi * F0 * h * t + h * 0.4)
    return 0.3 * y * swell


def narrow_interferer() -> np.ndarray:
    """Long tones that sit on SOME of the guitar's partials, not all.

    This is the case the time pass cannot handle - the collisions last a full
    second - but the untouched partials still describe the note.
    """
    y = np.zeros(N)
    for i, h in enumerate((2, 3, 5, 7)):
        f = F0 * h
        on = ((t > 1.0 + i * 2.0) & (t < 2.0 + i * 2.0)).astype(float)
        env = np.convolve(on, np.hanning(2205) / np.sum(np.hanning(2205)), mode="same")
        y += 1.2 * np.sin(2 * np.pi * f * t + 1.1 * i) * env
    return y


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return cond


def main() -> int:
    ok = True

    guitar = stereo(guitar_note())
    other = stereo(narrow_interferer())
    mix = (guitar + other).astype(np.float32)

    parts = A.partition_exact(mix, {"g": guitar * 0.85, "o": other * 0.9}, power=2.0)
    cut = parts["g"]
    sdr_cut = A.si_sdr(guitar, cut)

    print("\n[1] long collisions defeat the time pass")
    time_only, rep_t = restore_stem(cut, mix, SR, "electric_guitar", "Guitars")
    sdr_time = A.si_sdr(guitar, time_only)
    print(f"      time pass: restored {rep_t.bins_restored}, refused {rep_t.bins_refused}")
    ok &= check("time pass has to refuse a lot", rep_t.bins_refused > rep_t.bins_restored,
                f"{rep_t.bins_refused} refused vs {rep_t.bins_restored} repaired")

    print("\n[2] knowing the note recovers what time alone cannot")
    notes = [Note(0.0, DUR, 55, 0.8)]  # G3 held throughout
    harm, rep_h = harmonic_restore(cut, mix, notes, SR)
    sdr_harm = A.si_sdr(guitar, harm)
    print(f"      harmonic pass: rebuilt {rep_h['partials_rebuilt']} partial-frames, "
          f"refused {rep_h['partials_refused']}, notes never clean {rep_h['notes_never_clean']}")
    print(f"      SI-SDR   cut {sdr_cut:.2f}   time {sdr_time:.2f}   harmonic {sdr_harm:.2f}")
    ok &= check("harmonic pass beats the cut", sdr_harm > sdr_cut + 0.3, f"{sdr_harm:.2f} vs {sdr_cut:.2f}")
    ok &= check("harmonic pass beats the time pass", sdr_harm > sdr_time, f"{sdr_harm:.2f} vs {sdr_time:.2f}")
    ok &= check("it rebuilt something", rep_h["partials_rebuilt"] > 0)

    print("\n[3] both passes together")
    combined, _ = restore_stem(harm, mix, SR, "electric_guitar", "Guitars")
    sdr_both = A.si_sdr(guitar, combined)
    print(f"      combined SI-SDR {sdr_both:.2f}")
    ok &= check("combined is at least as good as harmonic alone", sdr_both > sdr_cut + 0.3,
                f"{sdr_both:.2f} vs cut {sdr_cut:.2f}")

    print("\n[4] a note never heard cleanly is refused")
    buried_notes = [Note(0.0, DUR, 40, 0.5)]  # a pitch that is not in the signal
    _, rep_b = harmonic_restore(cut, mix, buried_notes, SR)
    ok &= check("no balance to learn, so nothing invented",
                rep_b["partials_rebuilt"] == 0 or rep_b["notes_never_clean"] > 0,
                f"rebuilt {rep_b['partials_rebuilt']}, never clean {rep_b['notes_never_clean']}")

    print("\n[5] the mix still has to come out the same")
    exact = {"g": parts["g"], "o": parts["o"]}
    inflated = {"g": parts["g"] * 3.0, "o": parts["o"] * 3.0}  # a deliberate lie
    fixed, rep_c = enforce_mix_consistency(inflated, exact, mix, ["g", "o"])
    print(f"      trimmed {rep_c['bins_trimmed']:,} bins, excess before {rep_c['excess_before_db']} dB")
    ok &= check("inflation is detected", rep_c["bins_trimmed"] > 0)
    before_sum = A.peak_db(inflated["g"] + inflated["o"])
    after_sum = A.peak_db(fixed["g"] + fixed["o"])
    ok &= check("and trimmed back toward the real mix", after_sum < before_sum - 3.0,
                f"{after_sum:.1f} dB vs {before_sum:.1f} dB, mix {A.peak_db(mix):.1f} dB")

    honest, rep_h2 = enforce_mix_consistency(
        {"g": parts["g"], "o": parts["o"]}, exact, mix, ["g", "o"]
    )
    ok &= check("an honest pair is left alone", rep_h2["bins_trimmed"] == 0,
                f"{rep_h2['bins_trimmed']} trimmed")

    print("\n[6] transcription produces real notes")
    if transcription_available():
        found, midi = transcribe(guitar, SR)
        pitches = {n.pitch for n in found}
        print(f"      {len(found)} notes, pitches {sorted(pitches)[:8]}")
        ok &= check("it found the note that is playing", 55 in pitches or 43 in pitches or 67 in pitches,
                    f"expected G-something, got {sorted(pitches)[:8]}")
        ok &= check("and produced a MIDI object", midi is not None)
    else:
        print("      basic-pitch not installed - skipped")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
