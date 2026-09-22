"""Rendering, re-gridding, difficulty and transposition.

Each of these is checked against something measurable rather than against
"it ran": the rendered part has to contain the notes it claims, the re-grid
has to actually remove the drift it was given, the difficulty score has to
rank an easy mix below a crushed one, and transposition has to move the
pitch by the interval asked for while leaving the drums where they were.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worship_stems import audio as A  # noqa: E402
from worship_stems.analyze import Analysis, Section  # noqa: E402
from worship_stems.difficulty import advice, assess  # noqa: E402
from worship_stems.harmonic import Note  # noqa: E402
from worship_stems.regrid import build_warp, drift_ms, regrid_all  # noqa: E402
from worship_stems.render import PATCHES, list_patches, render, render_stem  # noqa: E402
from worship_stems.resynth import learn_timbre  # noqa: E402
from worship_stems.transpose import suggest_for_singer, transpose_stems  # noqa: E402

SR = 44100
rng = np.random.default_rng(41)


def dominant_pitch(audio: np.ndarray, sr: int) -> float:
    import librosa

    y = audio.mean(axis=0)
    S = np.abs(librosa.stft(y, n_fft=8192))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
    band = (freqs > 60) & (freqs < 1200)
    spectrum = S[band].mean(axis=1)
    return float(freqs[band][int(np.argmax(spectrum))])


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    return cond


def main() -> int:
    ok = True

    # ---------------------------------------------------------------- render
    print("\n[1] the same part, played by a different instrument")
    notes = [Note(0.0, 1.8, 55, 0.8), Note(2.0, 3.8, 60, 0.8)]  # G3 then C4
    piano = render(notes, 4.0, SR, patch="piano")
    pad = render(notes, 4.0, SR, patch="pad")
    ok &= check("piano renders", piano.shape[-1] == int(4.0 * SR) and bool(np.any(piano)))
    ok &= check("pad renders the same notes", bool(np.any(pad)))

    g3 = 440.0 * 2 ** ((55 - 69) / 12)
    found = dominant_pitch(piano[:, : int(1.8 * SR)], SR)
    ok &= check("it plays the note it was given", abs(found - g3) < g3 * 0.04,
                f"{found:.1f} Hz vs {g3:.1f} Hz")

    # The two patches must genuinely differ, not just in level.
    def spectrum(x):
        import librosa
        s = np.abs(librosa.stft(x.mean(axis=0), n_fft=4096)).mean(axis=1)
        return s / (np.linalg.norm(s) + 1e-9)

    similarity = float(np.dot(spectrum(piano), spectrum(pad)))
    ok &= check("the two instruments actually sound different", similarity < 0.97,
                f"spectral similarity {similarity:.3f}")

    print("\n[2] the learned timbre can be played back too")
    line = render(notes, 4.0, SR, patch="organ")
    timbre = learn_timbre(line, notes, SR)
    played = render(notes, 4.0, SR, patch="learned", timbre=timbre)
    ok &= check("learned playback produces audio", bool(np.any(played)))
    sim = float(np.dot(spectrum(line), spectrum(played)))
    ok &= check("and resembles what it learned from", sim > 0.5, f"similarity {sim:.3f}")
    ok &= check("patches are listed for the UI", len(list_patches()) >= 6, f"{len(list_patches())}")
    auto, label = render_stem("keys", notes, 4.0, SR)
    ok &= check("a stem gets a sensible default sound", bool(np.any(auto)) and bool(label), label)

    # ---------------------------------------------------------------- regrid
    print("\n[3] a drifting performance is pulled onto a grid")
    bpm = 90.0
    period = 60.0 / bpm
    intervals = np.concatenate([
        np.full(20, period),          # steady
        np.full(24, period * 0.94),   # rushing into the chorus
        np.full(20, period * 1.05),   # dragging at the end
    ])
    beats = np.concatenate([[0.0], np.cumsum(intervals)])[:64]
    duration = float(beats[-1] + period)
    n = int(duration * SR)

    # Each beat holds a different steady pitch. A phase vocoder smears
    # transients badly, so a click track would measure the vocoder rather
    # than the re-grid - but it preserves steady-state frequency exactly,
    # which is what we actually want to follow.
    scale = [330.0, 392.0, 440.0, 494.0, 587.0]
    y = np.zeros(n)
    for i, b in enumerate(beats):
        a0 = int(b * SR)
        a1 = int((beats[i + 1] if i + 1 < len(beats) else b + period) * SR)
        a1 = min(a1, n)
        if a1 <= a0:
            continue
        tt = np.arange(a1 - a0) / SR
        y[a0:a1] += 0.5 * np.sin(2 * np.pi * scale[i % len(scale)] * tt)
    audio = np.stack([y, y]).astype(np.float32)

    analysis = Analysis(
        bpm=bpm, beats=[float(b) for b in beats], downbeats=[float(b) for b in beats[::4]],
        beats_per_bar=4, key="G", mode="major", key_confidence=0.9, camelot="9B",
        duration=duration, sections=[Section(0.0, duration, "verse", 1)],
    )

    source, target, detected = build_warp(analysis)
    before = drift_ms(source, target)
    print(f"      the record wanders {before:.0f} ms from a steady {detected:.1f} BPM")
    ok &= check("there is real drift to fix", before > 100, f"{before:.0f} ms")
    ok &= check("the tempo is read correctly", abs(detected - bpm) < 1.0, f"{detected:.2f}")

    stems, moved, report = regrid_all({"mix": audio}, analysis, SR)
    ok &= check("it reports what it did", report is not None)
    if report:
        print(f"      stretched between {report.mean_stretch:.3f}x and {report.max_stretch:.3f}x")
        ok &= check("the stretch matches the drift it was given",
                    abs(report.max_stretch - 1.05) < 0.02, f"max {report.max_stretch:.3f}")

    def pitch_changes(x, sr):
        """Times where the held pitch steps to a new note."""
        import librosa
        hop = 512
        S = np.abs(librosa.stft(x.mean(axis=0), n_fft=4096, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        band = (freqs > 250) & (freqs < 700)
        track = freqs[band][np.argmax(S[band], axis=0)]
        # Ignore single-frame wobble at the boundaries.
        import scipy.ndimage as ndi
        track = ndi.median_filter(track, size=5)
        idx = np.flatnonzero(np.abs(np.diff(track)) > 15.0) + 1
        keep = [idx[0]] if len(idx) else []
        for j in idx[1:]:
            if j - keep[-1] > int(0.2 * sr / hop):
                keep.append(j)
        return np.array(keep) * hop / sr

    after_times = pitch_changes(stems["mix"], SR)
    if len(after_times) > 10:
        spacing = np.diff(after_times)
        expected = 60.0 / detected
        worst = float(np.max(np.abs(spacing - expected)) * 1000.0)
        print(f"      after: {len(after_times)} note changes, worst spacing error {worst:.0f} ms")
        ok &= check("the beats now sit on a steady grid", worst < 60.0, f"{worst:.0f} ms")
        drifted = float(np.max(np.abs(np.diff(np.diff(after_times)))) * 1000.0)
        _ = drifted
    else:
        ok &= check("enough note changes survived to measure", False, f"{len(after_times)}")

    ok &= check("the analysis moved with the audio",
                abs(moved.bpm - detected) < 0.5, f"{moved.bpm} vs {detected:.1f}")

    # ------------------------------------------------------------ difficulty
    print("\n[4] an easy mix scores below a crushed one")
    t = np.arange(SR * 20) / SR
    sparse = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 330 * t)
    easy = np.stack([sparse, np.roll(sparse, 600) * 0.9]).astype(np.float32)

    dense = np.zeros_like(t)
    for f in (55, 110, 165, 220, 277, 330, 440, 550, 660, 880):
        dense += np.sin(2 * np.pi * f * t + rng.random() * 6)
    dense += 0.6 * rng.standard_normal(len(t))
    dense = np.tanh(dense * 4.0)  # crushed
    hard = np.stack([dense, dense * 0.995]).astype(np.float32)

    d_easy = assess(easy, SR)
    d_hard = assess(hard, SR)
    print(f"      open mix   {d_easy.score:.2f}  {d_easy.verdict:10s} {d_easy.reasons[0][:52]}")
    print(f"      crushed    {d_hard.score:.2f}  {d_hard.verdict:10s} {d_hard.reasons[0][:52]}")
    ok &= check("the crushed mix scores harder", d_hard.score > d_easy.score + 0.1,
                f"{d_hard.score:.2f} vs {d_easy.score:.2f}")
    ok &= check("it is quick", d_easy.seconds < 20, f"{d_easy.seconds:.1f}s")
    ok &= check("it explains itself", len(d_hard.reasons) >= 1 and bool(d_hard.summary))
    ok &= check("and gives advice when it is hard", len(advice(d_hard)) >= 1, str(advice(d_hard))[:70])

    # ------------------------------------------------------------- transpose
    print("\n[5] transposing moves the pitch and leaves the drums alone")
    note = 0.3 * np.sin(2 * np.pi * 220 * t[: SR * 4])
    kick = np.zeros(SR * 4)
    for i in range(0, SR * 4, SR // 2):
        L = min(int(0.2 * SR), SR * 4 - i)
        tt = np.arange(L) / SR
        kick[i : i + L] += np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 20)

    stems_in = {
        "bass": np.stack([note, note]).astype(np.float32),
        "drums": np.stack([kick, kick]).astype(np.float32),
    }
    a2 = Analysis(bpm=90, beats=[0.0], downbeats=[0.0], beats_per_bar=4, key="A", mode="minor",
                  key_confidence=0.9, camelot="8A", duration=4.0, sections=[])

    shifted, a3, rep = transpose_stems(stems_in, a2, SR, semitones=2)
    before_hz = dominant_pitch(stems_in["bass"], SR)
    after_hz = dominant_pitch(shifted["bass"], SR)
    print(f"      bass {before_hz:.1f} Hz -> {after_hz:.1f} Hz (expected {before_hz * 2**(2/12):.1f})")
    ok &= check("the pitch moved by the interval asked for",
                abs(after_hz - before_hz * 2 ** (2 / 12)) < before_hz * 0.05,
                f"{after_hz:.1f} Hz")
    ok &= check("the drums were left alone",
                A.reconstruction_error_db(stems_in["drums"], [shifted["drums"]]) < -100)
    ok &= check("drums reported as untouched", "drums" in rep.left_alone, str(rep.left_alone))
    ok &= check("the key was updated", a3.key_display == "Bm", a3.key_display)

    print("\n[6] working out the move for a singer")
    a4 = Analysis(bpm=90, beats=[], downbeats=[], beats_per_bar=4, key="G", mode="major",
                  key_confidence=0.9, camelot="9B", duration=1.0, sections=[])
    ok &= check("G to E is down three, not up nine", suggest_for_singer(a4, "E") == -3,
                str(suggest_for_singer(a4, "E")))
    ok &= check("G to A is up two", suggest_for_singer(a4, "A") == 2, str(suggest_for_singer(a4, "A")))
    ok &= check("G to G is no move", suggest_for_singer(a4, "G") == 0)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
