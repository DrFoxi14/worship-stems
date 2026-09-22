"""SATB harmonisation as an exact shortest-path problem.

The point this prototype is meant to establish: every hard voice-leading rule
in the textbook is either *within* one chord (ranges, spacing, chord tones) or
*between two adjacent* chords (parallel fifths and octaves, leaps). None of
them reach further than one chord back.

That is the whole game. A constraint graph whose edges only ever join
neighbours is a chain, and a chain is solved exactly by dynamic programming -
Viterbi over voicings. No search heuristics, no backtracking, no model. The
optimum is found, and it is provably rule-clean, because illegal transitions
are simply absent from the graph.

So the learned part is not needed for correctness. It is needed only to
choose among the many correct answers, which is exactly what the 2026
constraints survey means by constraints becoming "a language of control"
rather than the engine.

Input is a melody plus chord symbols - not a harmony guessed from audio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rules import MAX_SPACING, RANGES

# --- chords ---------------------------------------------------------------

NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
        "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
        "Bb": 10, "B": 11}

QUALITY = {"": (0, 4, 7), "maj": (0, 4, 7), "m": (0, 3, 7), "min": (0, 3, 7),
           "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
           "sus4": (0, 5, 7), "dim": (0, 3, 6)}


def parse_chord(sym: str) -> tuple[int, tuple[int, ...]]:
    """'G' -> (7, (0,4,7)).  'Em' -> (4, (0,3,7))."""
    i = 1
    if len(sym) > 1 and sym[1] in "#b":
        i = 2
    root = NOTE[sym[:i]]
    qual = QUALITY[sym[i:]]
    return root, tuple((root + s) % 12 for s in qual)


# --- voicings -------------------------------------------------------------


def voicings(pcs: tuple[int, ...], root: int, soprano: int | None = None):
    """Every (S,A,T,B) inside the ranges, spaced legally, no crossing."""
    out = []
    s_lo, s_hi = RANGES[0]
    sopranos = [soprano] if soprano is not None else range(s_lo, s_hi + 1)
    for s in sopranos:
        if not (s_lo <= s <= s_hi) or s % 12 not in pcs:
            continue
        for a in range(RANGES[1][0], RANGES[1][1] + 1):
            if a % 12 not in pcs or a > s or s - a > MAX_SPACING[(0, 1)]:
                continue
            for t in range(RANGES[2][0], RANGES[2][1] + 1):
                if t % 12 not in pcs or t > a or a - t > MAX_SPACING[(1, 2)]:
                    continue
                for b in range(RANGES[3][0], RANGES[3][1] + 1):
                    if b % 12 not in pcs or b > t or t - b > MAX_SPACING[(2, 3)]:
                        continue
                    # the root should be present, and the third should not be
                    # doubled - the two doubling conventions that matter
                    notes = {s % 12, a % 12, t % 12, b % 12}
                    if root not in notes:
                        continue
                    third = (root + (3 if (root + 3) % 12 in pcs else 4)) % 12
                    if [x % 12 for x in (s, a, t, b)].count(third) > 1:
                        continue
                    out.append((s, a, t, b))
    return out


# --- transition legality (the only thing that reaches back) ---------------


def _is_parallel(c1, c2, interval):
    for x in range(4):
        for y in range(x + 1, 4):
            if abs(c1[x] - c1[y]) % 12 != interval:
                continue
            if abs(c2[x] - c2[y]) % 12 != interval:
                continue
            m1, m2 = c2[x] - c1[x], c2[y] - c1[y]
            if m1 == 0 or m2 == 0:
                continue
            if (m1 > 0) == (m2 > 0):
                return True
    return False


def legal_transition(c1, c2, max_leap=12, upper_max=9) -> bool:
    if _is_parallel(c1, c2, 7) or _is_parallel(c1, c2, 0):
        return False
    for v in range(4):
        limit = max_leap if v == 3 else upper_max
        if abs(c2[v] - c1[v]) > limit:
            return False
    return True


def transition_cost(c1, c2, crossing_penalty=0.0) -> float:
    """Smaller is smoother. Plain total motion, with the upper voices weighted
    a little more because the ear follows them."""
    w = (1.2, 1.0, 1.0, 0.8)
    cost = sum(wi * abs(c2[v] - c1[v]) for v, wi in enumerate(w))
    # common tones held are the classic preference
    held = sum(1 for v in range(4) if c1[v] == c2[v])
    return cost - 0.5 * held


# --- exact solve ----------------------------------------------------------


@dataclass
class Result:
    chords: list
    cost: float
    considered: int


def harmonize(chord_syms, melody=None, crossing_penalty=0.0) -> Result | None:
    """Viterbi over voicings. Returns the globally cheapest legal path."""
    melody = melody or [None] * len(chord_syms)
    layers = []
    for sym, mel in zip(chord_syms, melody):
        root, pcs = parse_chord(sym)
        v = voicings(pcs, root, soprano=mel)
        if not v:
            return None
        layers.append(v)

    considered = sum(len(l) for l in layers)
    best = {i: 0.0 for i in range(len(layers[0]))}
    back = [{} for _ in layers]

    for k in range(1, len(layers)):
        new = {}
        for j, c2 in enumerate(layers[k]):
            bi, bc = None, math.inf
            for i, c1 in enumerate(layers[k - 1]):
                if i not in best:
                    continue
                if not legal_transition(c1, c2):
                    continue
                c = best[i] + transition_cost(c1, c2, crossing_penalty)
                if c < bc:
                    bi, bc = i, c
            if bi is not None:
                new[j] = bc
                back[k][j] = bi
        if not new:
            return None
        best = new

    end = min(best, key=best.get)
    path = [end]
    for k in range(len(layers) - 1, 0, -1):
        path.append(back[k][path[-1]])
    path.reverse()
    return Result([layers[k][i] for k, i in enumerate(path)], best[end], considered)
