import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/claude/worship-stems/worship-stems"); sys.path.insert(0,"/home/claude/midi_audit")
from formant_test import vowel, stereo, FORMANTS
from synth import SR, midi_to_hz
from worship_stems.harmonic import Note
from worship_stems.resynth import learn_timbre

notes, audio = [], np.zeros(0, dtype=np.float32)
for i,p in enumerate((55,59,62,67,64,60,57,64)):
    audio=np.concatenate([audio, vowel(midi_to_hz(p), int(0.8*SR), SR)])
    notes.append(Note(i*0.8+0.02,(i+1)*0.8-0.02,p,0.6))
tb = learn_timbre(stereo(audio), notes, SR)
e,f = tb.envelope, tb.env_freqs
print("Anvelopa invatata, 300-3500 Hz (formanti reali: 700 / 1150 / 2600)\n")
for i in range(len(f)):
    if 300 <= f[i] <= 3500:
        bar = "#"*int(round(e[i]*50))
        print(f"  {f[i]:7.0f} Hz  {e[i]:.3f}  {bar}")
