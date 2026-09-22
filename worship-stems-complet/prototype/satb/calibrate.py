"""Run the rule checkers over the Bach chorales and see what they say.

Bach is the reference the textbooks were written from. If a rule fires often
on Bach, either the rule is stated too strictly or my implementation is wrong.
Either way the number is what tells me, not the textbook.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from music21 import converter, corpus  # noqa: E402

from rules import ALL_RULES  # noqa: E402


def chorale_to_chords(score, max_events=400):
    """Four-voice score -> list of [soprano, alto, tenor, bass] MIDI numbers.

    Uses the chordified score so every point where any voice changes becomes
    one event, which is how voice-leading rules are actually applied.
    """
    parts = score.parts
    if len(parts) != 4:
        return None
    try:
        flat = [p.flatten().notes.stream() for p in parts]
    except Exception:
        return None

    # Sample at every offset where any part has a note onset.
    offsets = sorted({float(n.offset) for part in flat for n in part})
    if not offsets or len(offsets) > max_events:
        offsets = offsets[:max_events]

    chords = []
    for off in offsets:
        ev = []
        for part in flat:
            sounding = None
            for n in part:
                if float(n.offset) <= off < float(n.offset) + float(n.quarterLength):
                    sounding = n
                    break
            if sounding is None or not hasattr(sounding, "pitch"):
                ev.append(None)
            else:
                ev.append(int(sounding.pitch.midi))
        chords.append(ev)
    return chords


def main(limit=60):
    paths = corpus.getComposer("bach")
    totals = Counter()
    events = 0
    used = 0
    skipped = 0

    for p in paths:
        if used >= limit:
            break
        try:
            score = converter.parse(p)
        except Exception:
            skipped += 1
            continue
        chords = chorale_to_chords(score)
        if not chords or len(chords) < 8:
            skipped += 1
            continue
        used += 1
        events += len(chords)
        for name, fn in ALL_RULES.items():
            totals[name] += len(fn(chords))

    print(f"\n{used} corale Bach, {events} evenimente armonice "
          f"({skipped} sărite: nu sunt la 4 voci sau nu s-au parsat)\n")
    print(f"  {'regulă':<20} {'încălcări':>10} {'la 100 evenimente':>20}")
    for name in ALL_RULES:
        n = totals[name]
        print(f"  {name:<20} {n:>10} {100.0*n/max(events,1):>20.2f}")
    return totals, events


if __name__ == "__main__":
    main()
