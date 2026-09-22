"""Rule-clean is not the same as good. Measure the gap against Bach."""
import sys, warnings, statistics as st
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from music21 import converter, corpus
from calibrate import chorale_to_chords
from harmonize import harmonize

def profile(chords, label):
    sop=[]; sa=[]; at=[]; tb=[]; uni=0; n=0
    for c in chords:
        if None in c: continue
        n+=1; sop.append(c[0]); sa.append(c[0]-c[1]); at.append(c[1]-c[2]); tb.append(c[2]-c[3])
        uni += sum(1 for i in range(3) if c[i]==c[i+1])
    print(f"  {label:<28} sopran {st.mean(sop):5.1f}  "
          f"S-A {st.mean(sa):4.1f}  A-T {st.mean(at):4.1f}  T-B {st.mean(tb):4.1f}  "
          f"unisoane {100*uni/max(n,1):5.1f}%")

print("\nProfil de scriitură (medii, în semitonuri)\n")
allc=[]; used=0
for p in corpus.getComposer("bach"):
    if used>=40: break
    try: s=converter.parse(p)
    except Exception: continue
    ch=chorale_to_chords(s)
    if not ch or len(ch)<8: continue
    used+=1; allc.extend(ch)
profile(allc, f"Bach ({used} corale)")

prog=["G","D","Em","C"]*4
r=harmonize(prog)
profile(r.chords, "armonizatorul meu")
