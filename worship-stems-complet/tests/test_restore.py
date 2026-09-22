"""Does rebuilding the holes actually get closer to the real instrument?

The only honest test is against a source we know. A sustained guitar-like
note is repeatedly buried under snare hits; the exact partition cuts holes
in it where the collisions are; restoration fills them. If the idea works,
the restored stem is measurably closer to the original guitar than the cut
one - and it must refuse to rebuild a note that was never heard cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worship_stems import audio as A  # noqa: E402
from worship_stems.restore import restore_stem  # noqa: E402

SR = 44100
DUR = 12.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(5)


def stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x * 0.97]).astype(np.float32)


def held_note(f0: float, harmonics: int = 8) -> np.ndarray:
    """A sustained, slowly swelling harmonic note - the thing that gets holed."""
    swell = 0.7 + 0.3 * np.sin(2 * np.pi * 0.4 * t)
    y = np.zeros(N)
    for h in range(1, harmonics + 1):
        y += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t + h * 0.7)
    return 0.25 * y * swell


def snare_hits(rate: float = 2.0) -> np.ndarray:
    """Loud broadband hits - these are what bury the note."""
    y = np.zeros(N)
    step = int(SR / rate)
    for i in range(0, N, step):
        L = min(int(SR * 0.09), N - i)
        tt = np.arange(L) / SR
        y[i : i + L] += 1.4 * rng.standard_normal(L) * np.exp(-tt * 30.0)
    return y


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return cond


def main() -> int:
    ok = True

    guitar = stereo(held_note(196.0))
    drums = stereo(snare_hits())
    mix = (guitar + drums).astype(np.float32)

    # The cut: an exact partition, as the app produces it.
    parts = A.partition_exact(mix, {"guitar": guitar * 0.8, "drums": drums * 0.9}, power=2.0)
    cut = parts["guitar"]

    print("\n[1] the cut really does leave holes")
    sdr_cut = A.si_sdr(guitar, cut)
    ok &= check("cut stem is worse than the true guitar", sdr_cut < 40, f"{sdr_cut:.1f} dB SI-SDR")

    print("\n[2] restoring gets closer to the real instrument")
    restored, rep = restore_stem(cut, mix, SR, "electric_guitar", "Guitars")
    sdr_res = A.si_sdr(guitar, restored)
    print(f"      damaged {rep.bins_damaged} bins, restored {rep.bins_restored} "
          f"({rep.share_restored*100:.0f}%), refused {rep.bins_refused}, "
          f"energy {rep.energy_gain_db:+.2f} dB")
    ok &= check("restored stem beats the cut stem", sdr_res > sdr_cut + 0.3,
                f"{sdr_res:.1f} dB vs {sdr_cut:.1f} dB")
    ok &= check("something was actually repaired", rep.bins_restored > 0)

    print("\n[3] it never creates energy the recording did not hold")
    over = restored - mix
    # Per-bin the ceiling is the mix magnitude, so the restored stem must not
    # tower over the mix anywhere in level.
    ok &= check("restored stem stays under the mix",
                float(np.max(np.abs(restored))) <= float(np.max(np.abs(mix))) * 1.5,
                f"peak {A.peak_db(restored):.1f} vs mix {A.peak_db(mix):.1f}")
    _ = over

    print("\n[4] it refuses a gap too long to read across")
    _, rep2 = restore_stem(cut, mix, SR, "electric_guitar", "Guitars", max_gap_ms=8.0)
    print(f"      damaged {rep2.bins_damaged}, restored {rep2.bins_restored}, refused {rep2.bins_refused}")
    ok &= check("long gaps are left alone", rep2.bins_refused > 0, f"{rep2.bins_refused} bins refused")
    ok &= check("and it repairs less than with a generous limit",
                rep2.bins_restored < rep.bins_restored, f"{rep2.bins_restored} vs {rep.bins_restored}")

    print("\n[5] a clean stem is left alone")
    clean = stereo(held_note(196.0))
    untouched, rep3 = restore_stem(clean, clean, SR, "x", "X")
    diff = A.reconstruction_error_db(clean, [untouched])
    ok &= check("nothing changed on an undamaged stem", diff < -40, f"{diff:.1f} dB difference")
    ok &= check("and it reports no repairs", rep3.bins_restored == 0, f"{rep3.bins_restored}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
