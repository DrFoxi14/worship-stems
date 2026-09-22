"""Does the frequency envelope actually fix the three things it was meant to?

  1. the same vowel measured at different pitches gives the same shape
  2. a transposed note keeps its formants
  3. resynthesis explains more of a real stem than the harmonic-index model

Ground truth throughout: the test signals are built with formants at known
fixed frequencies, so "did the formants move" is a question with an answer.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/worship-stems/worship-stems")
sys.path.insert(0, "/home/claude/midi_audit")

from synth import SR, midi_to_hz  # noqa: E402

from worship_stems.harmonic import Note  # noqa: E402
from worship_stems.resynth import Timbre, explain, learn_timbre, synthesize  # noqa: E402

FORMANTS = ((700.0, 1.0), (1150.0, 0.7), (2600.0, 0.35))
BW = 220.0


def vowel(f0, n, sr, formants=FORMANTS, bw=BW):
    """A sung vowel: harmonics of f0 shaped by formants FIXED in Hz."""
    t = np.arange(n) / sr
    x = np.zeros(n)
    for h in range(1, 40):
        f = f0 * h
        if f >= sr / 2:
            break
        g = sum(a * np.exp(-0.5 * ((f - fc) / bw) ** 2) for fc, a in formants)
        x += g * np.sin(2 * np.pi * f * t)
    m = float(np.abs(x).max())
    return (x / (m or 1.0) * 0.3).astype(np.float32)


def stereo(x):
    return np.stack([x, x * 0.98]).astype(np.float32)


def envelope_peaks(tb, n=3):
    """Where does the learned envelope say the formants are?"""
    e, f = tb.envelope, tb.env_freqs
    peaks = []
    for i in range(1, len(e) - 1):
        if e[i] >= e[i - 1] and e[i] >= e[i + 1]:
            peaks.append((e[i], f[i]))
    peaks.sort(reverse=True)
    return [round(fr) for _, fr in peaks[:n]]


def main():
    dur = 2.0
    n = int(dur * SR)

    print("\n[1] aceeași vocală la înălțimi diferite -> aceeași anvelopă?")
    print(f"    formanți reali: {[int(f) for f, _ in FORMANTS]} Hz")
    for pitch, label in ((55, "G3 196 Hz"), (62, "D4 294 Hz"), (67, "G4 392 Hz")):
        f0 = midi_to_hz(pitch)
        tb = learn_timbre(stereo(vowel(f0, n, SR)), [Note(0.05, dur - 0.05, pitch, 0.6)], SR)
        old_peak = int(np.argmax(tb.profile)) + 1
        print(f"    {label:<12} anvelopă: vârfuri la {envelope_peaks(tb)} Hz"
              f"   | vechiul model: armonica {old_peak} = {round(old_peak*f0)} Hz")

    print("\n[2] o singură voce care cântă mai multe înălțimi (ca în realitate)")
    notes, audio = [], np.zeros(0, dtype=np.float32)
    for i, pitch in enumerate((55, 59, 62, 67, 64, 60)):
        seg = vowel(midi_to_hz(pitch), int(0.8 * SR), SR)
        notes.append(Note(i * 0.8 + 0.02, (i + 1) * 0.8 - 0.02, pitch, 0.6))
        audio = np.concatenate([audio, seg])
    st = stereo(audio)
    tb = learn_timbre(st, notes, SR)
    print(f"    învățat din {tb.observations} note")
    print(f"    anvelopa găsește formanții la {envelope_peaks(tb)} Hz")

    print("\n[3] transpunere: se mișcă formanții?")
    for semis in (0, 3, 7, -4):
        moved = [Note(x.start, x.end, x.pitch + semis, x.amplitude) for x in notes]
        syn = synthesize(moved, tb, duration=len(audio) / SR, sr=SR)
        from worship_stems.audio import _stft
        S = np.abs(_stft(syn, 4096, 1024)).mean(axis=0).mean(axis=-1)
        fr = np.fft.rfftfreq(4096, 1.0 / SR)
        band = (fr > 400) & (fr < 3200)
        top = fr[band][np.argsort(S[band])[-40:]]
        centre = float(np.median(top))
        print(f"    {semis:+3d} semitonuri -> centrul de energie 400-3200 Hz: {centre:6.0f} Hz")

    print("\n[4] resinteza explică mai mult din stem?")
    rep_new = explain(st, SR, "bgv", "Cor", rounds=2, notes=notes)[2]
    tb_old = Timbre(tb.profile, tb.observations)  # fără anvelopă
    syn_old = synthesize(notes, tb_old, duration=len(audio) / SR, sr=SR)
    syn_new = synthesize(notes, tb, duration=len(audio) / SR, sr=SR)
    from worship_stems.audio import si_sdr
    print(f"    model vechi (pe armonice) SI-SDR {si_sdr(st, syn_old):6.2f} dB")
    print(f"    cu anvelopă              SI-SDR {si_sdr(st, syn_new):6.2f} dB")
    print(f"    explain() raportează {rep_new.explained*100:.0f}% din energia stem-ului")


if __name__ == "__main__":
    main()
