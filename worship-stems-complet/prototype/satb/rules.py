"""Voice-leading rules, checked against a corpus that is known to obey them.

A rule checker written from a textbook is a guess until it is calibrated. If
my parallel-fifth detector says Bach wrote hundreds of them, the detector is
wrong, not Bach. So every rule here is run over the 400-odd chorales in the
music21 corpus first, and the violation rate is the calibration.

Voices are indexed 0=soprano, 1=alto, 2=tenor, 3=bass.
"""

from __future__ import annotations

from dataclasses import dataclass


# Standard SATB ranges, as written in every harmony textbook (MIDI numbers).
RANGES = {
    0: (60, 81),   # soprano  C4 - A5
    1: (55, 74),   # alto     G3 - D5
    2: (48, 69),   # tenor    C3 - A4
    3: (40, 62),   # bass     E2 - D4
}

# Maximum spacing between adjacent upper voices, in semitones.
# S-A and A-T should stay within an octave; T-B may exceed it.
MAX_SPACING = {(0, 1): 12, (1, 2): 12, (2, 3): 24}


@dataclass
class Violation:
    rule: str
    index: int      # which chord (or the first of the pair)
    voices: tuple
    detail: str


def _pairs(n=4):
    return [(a, b) for a in range(n) for b in range(a + 1, n)]


def check_ranges(chords) -> list[Violation]:
    out = []
    for i, ch in enumerate(chords):
        for v, p in enumerate(ch):
            if p is None:
                continue
            lo, hi = RANGES[v]
            if p < lo or p > hi:
                out.append(Violation("range", i, (v,), f"voice {v} at {p}, range {lo}-{hi}"))
    return out


def check_spacing(chords) -> list[Violation]:
    out = []
    for i, ch in enumerate(chords):
        for (a, b), limit in MAX_SPACING.items():
            if ch[a] is None or ch[b] is None:
                continue
            gap = ch[a] - ch[b]
            if gap > limit:
                out.append(Violation("spacing", i, (a, b), f"{gap} semitones, max {limit}"))
    return out


def check_crossing(chords) -> list[Violation]:
    out = []
    for i, ch in enumerate(chords):
        for a in range(3):
            if ch[a] is None or ch[a + 1] is None:
                continue
            if ch[a] < ch[a + 1]:
                out.append(Violation("crossing", i, (a, a + 1), f"{ch[a]} below {ch[a+1]}"))
    return out


def _parallel(chords, interval: int, name: str) -> list[Violation]:
    """Two voices a perfect interval apart that move to the same interval,
    both actually moving, and in the same direction."""
    out = []
    for i in range(len(chords) - 1):
        c1, c2 = chords[i], chords[i + 1]
        for a, b in _pairs():
            if None in (c1[a], c1[b], c2[a], c2[b]):
                continue
            i1 = abs(c1[a] - c1[b]) % 12
            i2 = abs(c2[a] - c2[b]) % 12
            if i1 != interval or i2 != interval:
                continue
            m1, m2 = c2[a] - c1[a], c2[b] - c1[b]
            if m1 == 0 and m2 == 0:          # repeated chord: not parallel motion
                continue
            if m1 == 0 or m2 == 0:            # one voice held: oblique, allowed
                continue
            if (m1 > 0) != (m2 > 0):          # contrary motion
                continue
            out.append(Violation(name, i, (a, b), f"{c1[a]},{c1[b]} -> {c2[a]},{c2[b]}"))
    return out


def check_parallel_fifths(chords):
    return _parallel(chords, 7, "parallel_fifth")


def check_parallel_octaves(chords):
    return _parallel(chords, 0, "parallel_octave")


def check_leaps(chords, max_leap=12, upper_max=9) -> list[Violation]:
    """Large melodic leaps. The bass is allowed an octave; upper voices are
    conventionally kept inside a sixth, with the octave as the hard limit."""
    out = []
    for i in range(len(chords) - 1):
        for v in range(4):
            a, b = chords[i][v], chords[i + 1][v]
            if a is None or b is None:
                continue
            leap = abs(b - a)
            limit = max_leap if v == 3 else upper_max
            if leap > limit:
                out.append(Violation("leap", i, (v,), f"voice {v} leaps {leap}, max {limit}"))
    return out


ALL_RULES = {
    "range": check_ranges,
    "spacing": check_spacing,
    "crossing": check_crossing,
    "parallel_fifth": check_parallel_fifths,
    "parallel_octave": check_parallel_octaves,
    "leap": check_leaps,
}


def check_all(chords) -> dict[str, list[Violation]]:
    return {name: fn(chords) for name, fn in ALL_RULES.items()}
