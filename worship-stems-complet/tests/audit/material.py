"""Test material: worship-shaped parts with exact ground truth.

Each entry is (name, patch, notes) where notes is [(start, end, midi_pitch)].
Everything sits in G major at 76 BPM, which is ordinary worship territory.
"""

BEAT = 60.0 / 76.0


def b(n):
    """beats -> seconds"""
    return n * BEAT


# --- 1. solo vocal melody, monophonic, legato ------------------------------
# G4 A4 B4 | D5 C5 B4 | A4 G4 A4 | B4 ... - a plain stepwise worship melody
MELODY = []
_seq = [(67, 1), (69, 1), (71, 2), (74, 2), (72, 1), (71, 1),
        (69, 2), (67, 1), (69, 1), (71, 3), (69, 1),
        (67, 2), (64, 2), (62, 4)]
_t = 0.0
for _p, _d in _seq:
    MELODY.append((b(_t), b(_t + _d) - 0.02, _p))
    _t += _d

# --- 2. piano, I-V-vi-IV in G, four-note voicings --------------------------
# G major, D major, E minor, C major - the most common worship loop there is
CHORDS = [
    [55, 62, 67, 71],   # G3 D4 G4 B4
    [50, 57, 62, 66],   # D3 A3 D4 F#4
    [52, 59, 64, 67],   # E3 B3 E4 G4
    [48, 55, 60, 64],   # C3 G3 C4 E4
]
PIANO = []
_t = 0.0
for _rep in range(2):
    for _ch in CHORDS:
        for _p in _ch:
            PIANO.append((b(_t), b(_t + 4) - 0.05, _p))
        _t += 4

# --- 3. SATB choir, four sustained voices in close harmony -----------------
# The hard case, and the one the product depends on.
# S / A / T / B over the same I-V-vi-IV loop.
SATB_VOICES = {
    "soprano": [71, 69, 67, 64],   # B4 A4 G4 E4
    "alto":    [67, 66, 64, 60],   # G4 F#4 E4 C4
    "tenor":   [62, 62, 59, 55],   # D4 D4 B3 G3
    "bass":    [55, 50, 52, 48],   # G3 D3 E3 C3
}
SATB = []
_t = 0.0
for _rep in range(2):
    for _i in range(4):
        for _v in SATB_VOICES.values():
            SATB.append((b(_t), b(_t + 4) - 0.05, _v[_i]))
        _t += 4

# --- 4. bass line, low register -------------------------------------------
BASS = []
_t = 0.0
for _rep in range(2):
    for _root in (43, 38, 40, 36):  # G2 D2 E2 C2
        BASS.append((b(_t), b(_t + 4) - 0.1, _root))
        _t += 4


CASES = [
    ("Linie vocală solo", "voice", MELODY),
    ("Pian, acorduri", "piano", PIANO),
    ("Cor SATB", "voice", SATB),
    ("Bas", "bass", BASS),
]
