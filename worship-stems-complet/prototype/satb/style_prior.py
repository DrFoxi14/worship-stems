"""Can a prior learned from a corpus fix the shape without breaking the rules?

The claim to test: constraints give correctness, the corpus gives taste, and
the two compose. If adding a style term moves the writing profile toward the
corpus while violations stay at zero, the architecture works.
"""
import sys, warnings, math, statistics as st
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from music21 import converter, corpus
from calibrate import chorale_to_chords
from harmonize import voicings, parse_chord, legal_transition, transition_cost
from rules import check_all

# --- learn the prior ------------------------------------------------------
allc=[]; used=0
for p in corpus.getComposer("bach"):
    if used>=40: break
    try: s=converter.parse(p)
    except Exception: continue
    ch=chorale_to_chords(s)
    if not ch or len(ch)<8: continue
    used+=1; allc.extend(ch)
clean=[c for c in allc if None not in c]
PRIOR = {
    "sop": (st.mean(c[0] for c in clean), st.pstdev(c[0] for c in clean)),
    "sa":  (st.mean(c[0]-c[1] for c in clean), st.pstdev(c[0]-c[1] for c in clean)),
    "at":  (st.mean(c[1]-c[2] for c in clean), st.pstdev(c[1]-c[2] for c in clean)),
    "tb":  (st.mean(c[2]-c[3] for c in clean), st.pstdev(c[2]-c[3] for c in clean)),
}
print(f"Prior invatat din {used} corale Bach:")
for k,(m,s) in PRIOR.items(): print(f"  {k}: media {m:.1f}, abatere {s:.1f}")

def style_cost(c):
    z=0.0
    for key,val in (("sop",c[0]),("sa",c[0]-c[1]),("at",c[1]-c[2]),("tb",c[2]-c[3])):
        m,s = PRIOR[key]
        z += ((val-m)/max(s,1e-6))**2
    return z

# --- Viterbi with node costs ---------------------------------------------
def harmonize_styled(syms, weight=1.0):
    layers=[]
    for sym in syms:
        root,pcs = parse_chord(sym)
        v = voicings(pcs, root)
        if not v: return None
        layers.append(v)
    best={i: weight*style_cost(c) for i,c in enumerate(layers[0])}
    back=[{} for _ in layers]
    for k in range(1,len(layers)):
        new={}
        for j,c2 in enumerate(layers[k]):
            bi,bc=None,math.inf
            for i,c1 in enumerate(layers[k-1]):
                if i not in best or not legal_transition(c1,c2): continue
                c=best[i]+transition_cost(c1,c2)+weight*style_cost(c2)
                if c<bc: bi,bc=i,c
            if bi is not None: new[j]=bc; back[k][j]=bi
        if not new: return None
        best=new
    end=min(best,key=best.get); path=[end]
    for k in range(len(layers)-1,0,-1): path.append(back[k][path[-1]])
    path.reverse()
    return [layers[k][i] for k,i in enumerate(path)]

def profile(ch,label):
    n=len(ch)
    print(f"  {label:<34} sopran {st.mean(c[0] for c in ch):5.1f}  "
          f"S-A {st.mean(c[0]-c[1] for c in ch):4.1f}  "
          f"A-T {st.mean(c[1]-c[2] for c in ch):4.1f}  "
          f"T-B {st.mean(c[2]-c[3] for c in ch):4.1f}  "
          f"incalcari {sum(len(x) for x in check_all(ch).values())}")

print("\nProfil:")
profile(clean, "Bach (referinta)")
prog=["G","D","Em","C"]*4
for w in (0.0, 0.5, 2.0, 5.0):
    ch=harmonize_styled(prog, weight=w)
    profile(ch, f"cu prior, pondere {w}")
