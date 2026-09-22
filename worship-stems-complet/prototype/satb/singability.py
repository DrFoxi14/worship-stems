"""Al doilea scor: cat de cantabila e linia, nu cat de mult seamana cu Bach.

Obiectia e corecta: asemanarea cu Bach masoara lucrul gresit. Ce conteaza
pentru un cor de biserica e daca linia se invata la o repetitie. Astea sunt
masurabile, si se pot calibra pe ORICE corpus - inclusiv pe cele 124.
"""
import sys, warnings, statistics as st
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from music21 import converter, corpus
from calibrate import chorale_to_chords
from harmonize import harmonize
from rules import RANGES

NAMES = ["S","A","T","B"]

def singability(chords):
    """Per voce: % miscare prin grad alaturat, % salturi > cvarta,
    % timp petrecut in extremele registrului, % voicing-uri repetate."""
    out = {}
    for v in range(4):
        line = [c[v] for c in chords if c[v] is not None]
        if len(line) < 3: continue
        steps = [abs(line[i+1]-line[i]) for i in range(len(line)-1)]
        moving = [s for s in steps if s > 0]
        lo, hi = RANGES[v]; span = hi - lo
        extreme = sum(1 for p in line if p < lo + 0.15*span or p > hi - 0.15*span)
        out[NAMES[v]] = {
            "grad_alaturat": 100*sum(1 for s in moving if s<=2)/max(len(moving),1),
            "salt_mare":     100*sum(1 for s in moving if s>5)/max(len(moving),1),
            "extreme":       100*extreme/len(line),
            "static":        100*sum(1 for s in steps if s==0)/max(len(steps),1),
        }
    return out

def show(chords, label):
    s = singability(chords)
    print(f"\n  {label}")
    print(f"    {'voce':<5} {'grad alăturat':>14} {'salturi >cvartă':>16} {'în extreme':>12} {'static':>8}")
    for v in NAMES:
        if v not in s: continue
        d = s[v]
        print(f"    {v:<5} {d['grad_alaturat']:>13.0f}% {d['salt_mare']:>15.0f}% "
              f"{d['extreme']:>11.0f}% {d['static']:>7.0f}%")

allc=[]; used=0
for p in corpus.getComposer("bach"):
    if used>=40: break
    try: sc=converter.parse(p)
    except Exception: continue
    ch=chorale_to_chords(sc)
    if not ch or len(ch)<8: continue
    used+=1; allc.extend(ch)
clean=[c for c in allc if None not in c]

print("\nCântabilitate — al doilea scor, lângă numărul de încălcări")
show(clean, f"Bach ({used} corale) — referință de calibrare, NU țintă de stil")
show(harmonize(["G","D","Em","C"]*4).chords, "generat de armonizator")
