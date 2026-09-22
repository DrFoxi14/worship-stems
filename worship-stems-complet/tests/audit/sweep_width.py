"""Latimea nucleului: calibrata pe formanti cunoscuti, nu ghicita."""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/claude/worship-stems/worship-stems"); sys.path.insert(0,"/home/claude/midi_audit")
import worship_stems.resynth as R
from formant_test import vowel, stereo, FORMANTS, BW
from synth import SR, midi_to_hz
from worship_stems.harmonic import Note

notes, audio = [], np.zeros(0, dtype=np.float32)
for i,p in enumerate((55,57,59,60,62,64,65,67,64,60)):
    audio=np.concatenate([audio, vowel(midi_to_hz(p), int(0.7*SR), SR)])
    notes.append(Note(i*0.7+0.02,(i+1)*0.7-0.02,p,0.6))
st = stereo(audio)

def truth(f):
    return sum(a*np.exp(-0.5*((f-fc)/BW)**2) for fc,a in FORMANTS)

print("eroare fata de curba reala de formanti (mai mic = mai bine)\n")
print(f"  {'latime':>8} {'eroare log':>12}   formanti gasiti (real 700/1150/2600)")
best=(None,1e9)
for w in (0.06,0.09,0.12,0.16,0.22,0.30):
    orig = R._build_envelope
    R._build_envelope = lambda g,of,oa,width_oct=w,_o=orig: _o(g,of,oa,width_oct)
    tb = R.learn_timbre(st, notes, SR)
    R._build_envelope = orig
    if not tb.has_envelope(): print(f"  {w:>8} fara anvelopa"); continue
    e,f = tb.envelope, tb.env_freqs
    band=(f>=400)&(f<=3500)
    t = np.array([truth(x) for x in f[band]]); t/=t.max()
    a = e[band]/e[band].max()
    err = float(np.mean(np.abs(np.log10(a+1e-3)-np.log10(t+1e-3))))
    # varfuri distincte
    pk=[]
    for i in range(1,len(e)-1):
        if e[i]>e[i-1] and e[i]>=e[i+1] and e[i]>0.03:
            if all(abs(np.log2(f[i]/k))>0.3 for k in pk): pk.append(f[i])
    print(f"  {w:>8} {err:>12.3f}   {[round(x) for x in pk if 400<x<3500]}")
    if err<best[1]: best=(w,err)
print(f"\n  cea mai buna: {best[0]} (eroare {best[1]:.3f})")
