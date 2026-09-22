"""Synthesise a fake worship song with a known structure, for testing."""
import numpy as np, sys
sys.path.insert(0, '.')
from worship_stems.audio import save_audio

SR = 44100
BPM = 76.0
BEAT = 60.0 / BPM
BAR = BEAT * 4

def chord(freqs, n, sr=SR, amp=0.2, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(n)/sr
    out = np.zeros(n)
    for f in freqs:
        for h,(a) in enumerate([1.0,0.5,0.28,0.15], start=1):
            out += a*np.sin(2*np.pi*f*h*t + rng.random()*6.28)
    return amp*out/len(freqs)

def note(p, o=3):
    idx = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}[p]
    return 440.0*2**((idx + 12*(o-4) - 9)/12)

def drums(n, intensity=1.0, sr=SR, seed=1):
    rng = np.random.default_rng(seed); out = np.zeros(n)
    beat_n = int(BEAT*sr)
    for i in range(0, n, beat_n):
        b = (i//beat_n) % 4
        L = min(beat_n, n-i); t = np.arange(L)/sr
        if b in (0,2):  # kick
            out[i:i+L] += 0.9*intensity*np.sin(2*np.pi*55*np.exp(-t*12)*t*8)*np.exp(-t*18)
        if b in (1,3):  # snare
            out[i:i+L] += 0.5*intensity*rng.standard_normal(L)*np.exp(-t*24)
        out[i:i+L] += 0.12*intensity*rng.standard_normal(L)*np.exp(-t*60)  # hats
    return out

def bassline(root, n, sr=SR):
    t = np.arange(n)/sr
    f = note(root, 1)
    return 0.35*(np.sin(2*np.pi*f*t) + 0.3*np.sin(2*np.pi*f*2*t))*np.clip(np.sin(2*np.pi*t/(BEAT*2))*3,0,1)

def vocal(n, sr=SR, seed=3, amp=0.25):
    rng = np.random.default_rng(seed); t = np.arange(n)/sr
    mel = np.zeros(n); phrase = int(BEAT*2*sr)
    scale = [0,2,4,7,9,12]
    for i in range(0, n, phrase):
        L = min(phrase, n-i); tt = np.arange(L)/sr
        f = note('G',4)*2**(scale[rng.integers(0,len(scale))]/12)
        vib = 1+0.012*np.sin(2*np.pi*5.2*tt)
        env = np.clip(np.minimum(tt*12,(L/sr-tt)*12),0,1)
        mel[i:i+L] += (np.sin(2*np.pi*f*tt*vib)+0.4*np.sin(2*np.pi*2*f*tt*vib)+0.2*np.sin(2*np.pi*3*f*tt*vib))*env
    return amp*mel

# G major: I=G, V=D, vi=Em, IV=C
CH = {
 'G': [note('G',3), note('B',3), note('D',4)],
 'D': [note('D',3), note('F#',3), note('A',3)],
 'Em':[note('E',3), note('G',3), note('B',3)],
 'C': [note('C',3), note('E',3), note('G',3)],
}

PLAN = [   # (label, bars, chords, drum intensity, has_vocal, pad_amp)
 ('intro',  8, ['G','D','Em','C'], 0.0, False, 0.22),
 ('verse',  16,['G','D','Em','C'], 0.6, True,  0.16),
 ('chorus', 16,['C','G','D','Em'], 1.0, True,  0.26),
 ('verse',  16,['G','D','Em','C'], 0.7, True,  0.16),
 ('chorus', 16,['C','G','D','Em'], 1.0, True,  0.26),
 ('bridge', 12,['Em','C','G','D'], 0.5, True,  0.20),
 ('chorus', 16,['C','G','D','Em'], 1.1, True,  0.28),
 ('outro',  8, ['G','D','G','G'],  0.2, False, 0.18),
]

pieces = []; truth = []; t0 = 0.0
for label, bars, chords, di, has_voc, pa in PLAN:
    n = int(bars*BAR*SR); seg = np.zeros(n)
    bar_n = int(BAR*SR)
    for b in range(bars):
        c = chords[b % len(chords)]
        s = b*bar_n; L = min(bar_n, n-s)
        seg[s:s+L] += chord(CH[c], L, amp=pa, seed=b)
        seg[s:s+L] += bassline({'G':'G','D':'D','Em':'E','C':'C'}[c], L)
    if di > 0: seg += drums(n, di, seed=int(t0))
    if has_voc: seg += vocal(n, seed=int(t0)+7)
    pieces.append(seg); truth.append((label, t0, t0+n/SR)); t0 += n/SR

y = np.concatenate(pieces)
y = y/np.max(np.abs(y))*0.85
stereo = np.stack([y, np.roll(y, 40)*0.97]).astype(np.float32)
save_audio('tests/fake_song.wav', stereo, SR, 'wav24')
print(f"wrote tests/fake_song.wav  {len(y)/SR:.1f}s  BPM={BPM} key=G major")
for l,a,b in truth: print(f"  {l:8s} {a:7.2f} -> {b:7.2f}")
