"""Borrowing from another time the band played the same thing.

The test song has a chorus that appears three times with identical guitar.
A crash lands on bar 2 of chorus 1 only. The guitar there is destroyed by the
cut, and neither time-continuation nor the harmonic pass can help much -
but choruses 2 and 3 have that bar completely in the clear.

Also tested: a last chorus that is genuinely different must be refused, not
smeared over the others.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worship_stems import audio as A  # noqa: E402
from worship_stems.analyze import Analysis, Section  # noqa: E402
from worship_stems.repeat import borrow_from_repeats, warp_times  # noqa: E402

SR = 44100
BPM = 80.0
BEAT = 60.0 / BPM
BAR = BEAT * 4
BARS_PER_SECTION = 8
SEC_LEN = BAR * BARS_PER_SECTION  # 24 s
N_SECTIONS = 3
DUR = SEC_LEN * N_SECTIONS
N = int(SR * DUR)
rng = np.random.default_rng(17)


def stereo(x):
    return np.stack([x, x * 0.97]).astype(np.float32)


def chorus_guitar(n_samples: int, seed: int = 0) -> np.ndarray:
    """A repeating four-bar guitar figure, identical every chorus."""
    t = np.arange(n_samples) / SR
    y = np.zeros(n_samples)
    pattern = [196.0, 246.94, 293.66, 246.94]  # G B D B
    step = int(BEAT * SR)
    for i in range(0, n_samples, step):
        L = min(step, n_samples - i)
        tt = np.arange(L) / SR
        f = pattern[(i // step) % len(pattern)]
        env = np.exp(-tt * 1.2)
        note = sum((1.0 / h**1.1) * np.sin(2 * np.pi * f * h * tt) for h in range(1, 9))
        y[i : i + L] += note * env
    _ = t
    return 0.28 * y


def crash_at(time_s: float, n_samples: int) -> np.ndarray:
    """One loud cymbal, in one place only."""
    y = np.zeros(n_samples)
    i = int(time_s * SR)
    L = min(int(SR * 1.6), n_samples - i)
    if L <= 0:
        return y
    tt = np.arange(L) / SR
    y[i : i + L] += 2.2 * rng.standard_normal(L) * np.exp(-tt * 2.2)
    return y


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return cond


def build_analysis(last_is_different: bool = False) -> Analysis:
    downbeats = [i * BAR for i in range(int(DUR / BAR) + 1)]
    beats = [i * BEAT for i in range(int(DUR / BEAT) + 1)]
    sections = [
        Section(k * SEC_LEN, (k + 1) * SEC_LEN, "chorus", k + 1) for k in range(N_SECTIONS)
    ]
    _ = last_is_different
    return Analysis(
        bpm=BPM, beats=beats, downbeats=downbeats, beats_per_bar=4,
        key="G", mode="major", key_confidence=0.9, camelot="9B",
        duration=DUR, sections=sections,
    )


def main() -> int:
    ok = True
    per_section = int(SEC_LEN * SR)

    print("\n[1] bar-for-bar alignment survives a tempo difference")
    grid_a = np.array([0.0, 3.0, 6.0, 9.0])
    grid_b = np.array([0.0, 3.3, 6.6, 9.9])  # same music, 10% slower
    mapped = warp_times(np.array([0.0, 1.5, 3.0, 4.5, 6.0]), grid_a, grid_b)
    ok &= check("bar starts map onto bar starts", abs(mapped[2] - 3.3) < 1e-6, f"{mapped[2]:.3f}")
    ok &= check("mid-bar maps proportionally", abs(mapped[1] - 1.65) < 1e-6, f"{mapped[1]:.3f}")

    print("\n[2] a crash in one chorus only")
    guitar_one = chorus_guitar(per_section)
    guitar = stereo(np.tile(guitar_one, N_SECTIONS)[:N])
    crash = stereo(crash_at(BAR * 1.0, N))  # bar 2 of chorus 1
    mix = (guitar + crash).astype(np.float32)

    parts = A.partition_exact(mix, {"g": guitar * 0.85, "c": crash * 0.9}, power=2.0)
    cut = parts["g"]

    # Measure where the damage actually is. Averaging over the whole 24 s
    # section dilutes a 1.6 s crash into the noise.
    a0 = int(BAR * 1.0 * SR)
    a1 = a0 + int(1.6 * SR)
    sdr_cut = A.si_sdr(guitar[:, a0:a1], cut[:, a0:a1])
    print(f"      the crashed bar after the cut: {sdr_cut:.2f} dB SI-SDR")

    analysis = build_analysis()
    fixed, rep = borrow_from_repeats(cut, mix, analysis, SR, "electric_guitar", "Guitars")
    sdr_fixed = A.si_sdr(guitar[:, a0:a1], fixed[:, a0:a1])
    print(f"      borrowed {rep.bins_borrowed:,} bins from {rep.instances} instances, "
          f"{rep.pairs_rejected} pairs rejected")
    print(f"      the crashed bar after borrowing: {sdr_fixed:.2f} dB SI-SDR")
    ok &= check("it borrowed something", rep.bins_borrowed > 0)
    ok &= check("the crashed bar improves", sdr_fixed > sdr_cut + 1.0,
                f"{sdr_fixed:.2f} vs {sdr_cut:.2f}")

    print("\n[3] the undamaged choruses are left alone")
    b0, b1 = int(SEC_LEN * SR), int(2 * SEC_LEN * SR)
    before = A.si_sdr(guitar[:, b0:b1], cut[:, b0:b1])
    after = A.si_sdr(guitar[:, b0:b1], fixed[:, b0:b1])
    # Chorus 2 was never damaged, so both are perfect; the only difference is
    # the STFT round trip, which lands at float precision.
    ok &= check("chorus 2 is still effectively perfect", after > 100.0,
                f"{after:.1f} dB (was {before:.1f})")

    print("\n[4] a chorus that is genuinely different is refused")
    different = guitar.copy()
    # Last chorus: a different part entirely (transposed up a fourth).
    c0 = int(2 * SEC_LEN * SR)
    other = chorus_guitar(per_section, seed=1)
    t = np.arange(len(other)) / SR
    shifted = other * np.cos(2 * np.pi * 120.0 * t)  # deliberately unlike the rest
    different[:, c0 : c0 + len(shifted)] = stereo(shifted)[:, : different.shape[1] - c0]
    mix2 = (different + crash).astype(np.float32)
    p2 = A.partition_exact(mix2, {"g": different * 0.85, "c": crash * 0.9}, power=2.0)
    _, rep2 = borrow_from_repeats(p2["g"], mix2, analysis, SR, "electric_guitar", "Guitars")
    print(f"      borrowed {rep2.bins_borrowed:,}, rejected {rep2.pairs_rejected} pairs")
    ok &= check("mismatched instances are rejected", rep2.pairs_rejected > 0,
                f"{rep2.pairs_rejected} rejected")

    print("\n[5] nothing to borrow from when there are no repeats")
    single = Analysis(
        bpm=BPM, beats=analysis.beats, downbeats=analysis.downbeats, beats_per_bar=4,
        key="G", mode="major", key_confidence=0.9, camelot="9B", duration=DUR,
        sections=[Section(0.0, DUR, "verse", 1)],
    )
    same, rep3 = borrow_from_repeats(cut, mix, single, SR)
    ok &= check("no groups, no changes", rep3.bins_borrowed == 0 and rep3.groups == 0)
    ok &= check("audio untouched", A.reconstruction_error_db(cut, [same]) < -100)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
