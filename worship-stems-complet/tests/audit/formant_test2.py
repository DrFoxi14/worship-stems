"""Two of the first tests measured the wrong thing. Redo them properly.

[2] used a naive peak finder that reports every local maximum, so a flat
    region came back as three "formants". Needs prominence and separation.

[4] used SI-SDR, which is a waveform metric: it is dominated by phase and by
    the fixed decay envelope in `synthesize`, neither of which is what the
    timbre model controls. The right question is spectral - does the
    synthesis put energy where the recording puts it - so compare log
    spectra instead.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/worship-stems/worship-stems")
sys.path.insert(0, "/home/claude/midi_audit")

from formant_test import FORMANTS, stereo, vowel  # noqa: E402
from synth import SR, midi_to_hz  # noqa: E402

from worship_stems.audio import _stft  # noqa: E402
from worship_stems.harmonic import Note  # noqa: E402
from worship_stems.resynth import Timbre, learn_timbre, synthesize  # noqa: E402


def formant_peaks(tb, n=3, min_sep_oct=0.35, min_prom=0.12):
    """Real peaks only: prominent, and not two bins of the same hump."""
    e, f = np.asarray(tb.envelope), np.asarray(tb.env_freqs)
    cand = []
    for i in range(1, len(e) - 1):
        if e[i] > e[i - 1] and e[i] >= e[i + 1]:
            lo = e[max(0, i - 6):i].min() if i else e[i]
            hi = e[i + 1:i + 7].min() if i + 1 < len(e) else e[i]
            if e[i] - max(lo, hi) >= min_prom:
                cand.append((e[i], f[i]))
    cand.sort(reverse=True)
    kept = []
    for amp, fr in cand:
        if all(abs(np.log2(fr / k)) > min_sep_oct for k in kept):
            kept.append(fr)
        if len(kept) >= n:
            break
    return sorted(round(x) for x in kept)


def log_spectrum(x, sr=SR, n_fft=4096):
    S = np.abs(_stft(x, n_fft, n_fft // 4)).mean(axis=0).mean(axis=-1)
    return np.log10(S + 1e-8), np.fft.rfftfreq(n_fft, 1.0 / sr)


def spectral_distance(a, b, sr=SR, lo=300.0, hi=4000.0):
    """Mean absolute log-spectral difference over the band that carries the
    vowel. Lower is a better timbre match; phase and note envelope drop out."""
    la, fr = log_spectrum(a, sr)
    lb, _ = log_spectrum(b, sr)
    band = (fr >= lo) & (fr <= hi)
    la = la[band] - la[band].mean()
    lb = lb[band] - lb[band].mean()
    return float(np.mean(np.abs(la - lb)))


def main():
    print(f"\nformanți reali: {[int(f) for f, _ in FORMANTS]} Hz")

    # --- a voice singing several pitches, as in a real stem ---------------
    notes, audio = [], np.zeros(0, dtype=np.float32)
    for i, pitch in enumerate((55, 59, 62, 67, 64, 60, 57, 64)):
        audio = np.concatenate([audio, vowel(midi_to_hz(pitch), int(0.8 * SR), SR)])
        notes.append(Note(i * 0.8 + 0.02, (i + 1) * 0.8 - 0.02, pitch, 0.6))
    st = stereo(audio)
    dur = len(audio) / SR

    print("\n[2] anvelopa învățată din 8 note la înălțimi diferite")
    tb = learn_timbre(st, notes, SR)
    print(f"    formanți găsiți: {formant_peaks(tb)} Hz")

    print("\n[4] potrivirea spectrală (mai mic = mai bun), 300-4000 Hz")
    tb_old = Timbre(tb.profile, tb.observations)  # same data, no envelope
    syn_old = synthesize(notes, tb_old, duration=dur, sr=SR)
    syn_new = synthesize(notes, tb, duration=dur, sr=SR)
    d_old = spectral_distance(st, syn_old)
    d_new = spectral_distance(st, syn_new)
    print(f"    model vechi (pe armonice)  {d_old:.3f}")
    print(f"    cu anvelopă                {d_new:.3f}")
    print(f"    îmbunătățire               {100*(d_old-d_new)/d_old:.0f}%")

    print("\n[5] transpunere: unde ajunge formantul, vechi vs nou")
    for semis in (0, 5, -5):
        moved = [Note(x.start, x.end, x.pitch + semis, x.amplitude) for x in notes]
        for tag, t in (("vechi", tb_old), ("nou  ", tb)):
            syn = synthesize(moved, t, duration=dur, sr=SR)
            ls, fr = log_spectrum(syn)
            band = (fr > 400) & (fr < 3500)
            w = 10 ** ls[band]
            centre = float(np.sum(fr[band] * w) / np.sum(w))
            print(f"    {semis:+3d} semitonuri  {tag}  centru spectral {centre:6.0f} Hz")


if __name__ == "__main__":
    main()
