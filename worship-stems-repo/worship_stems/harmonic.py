"""Knowing what note was playing, and using it to rebuild what got buried.

The previous repair could only read *along* one frequency row: if a partial
was steady before a collision and steady after, fill the middle. That fails
whenever the collision is long, which is most of the time - a snare rings for
longer than a held note stays still.

Knowing the notes changes the problem. A note at 196 Hz has partials at 196,
392, 588, 784... and a snare hit does not bury all of them equally. If the
third partial is gone but the first, second and fifth are clean *at that same
instant*, the third is not unknown - it is determined by the other four,
because the partials of one note keep a roughly fixed balance while their
overall level moves together. That is the Common Amplitude Modulation
assumption, and it is what lets us read across the harmonics of a note
instead of only along the time axis.

Knowing the note also gives the exact phase advance: a partial at h*f0 turns
by 2*pi*h*f0*hop/sr per frame. No estimation needed.

The balance between partials is not assumed - it is measured, from the frames
of that same note where nothing was covering it. If a note is never heard
cleanly, there is no balance to measure and the hole stays.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np

from .audio import EPS, _istft, _stft, match_shape, save_audio

log = logging.getLogger(__name__)


@dataclass
class Note:
    start: float
    end: float
    pitch: int  # MIDI note number
    amplitude: float

    @property
    def f0(self) -> float:
        return 440.0 * (2.0 ** ((self.pitch - 69) / 12.0))

    @property
    def name(self) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return f"{names[self.pitch % 12]}{self.pitch // 12 - 1}"


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------


def transcription_available() -> bool:
    try:
        import basic_pitch  # noqa: F401

        from basic_pitch import ONNX_PRESENT, TF_PRESENT  # noqa: F401

        return True
    except Exception:
        return False


def _model_path():
    """Prefer the bundled ONNX model: it needs no TensorFlow, and TensorFlow
    pins numpy below 2, which collides with audio-separator."""
    import basic_pitch

    root = pathlib.Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022"
    onnx = root / "nmp.onnx"
    if onnx.exists():
        return onnx
    from basic_pitch import ICASSP_2022_MODEL_PATH

    return ICASSP_2022_MODEL_PATH


_MODEL_CACHE: dict[str, object] = {}


def transcribe(
    audio: np.ndarray,
    sr: int,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_len_ms: float = 60.0,
) -> tuple[list[Note], object | None]:
    """Polyphonic note transcription. Returns (notes, pretty_midi object)."""
    if not transcription_available():
        return [], None

    from basic_pitch.inference import Model, predict

    work = pathlib.Path(tempfile.mkdtemp(prefix="ws-midi-"))
    try:
        wav = work / "stem.wav"
        save_audio(wav, audio, sr, "wav16")

        key = "onnx"
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = Model(_model_path())

        _out, midi, note_events = predict(
            str(wav),
            _MODEL_CACHE[key],
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_len_ms,
        )
        notes = [Note(float(n[0]), float(n[1]), int(n[2]), float(n[3])) for n in note_events]
        notes.sort(key=lambda n: n.start)
        return notes, midi
    except Exception as exc:  # noqa: BLE001
        log.warning("transcription failed: %s", exc)
        return [], None
    finally:
        shutil.rmtree(work, ignore_errors=True)


MIDI_PROGRAMS = {
    "lead_vocal": 53, "bgv": 52, "bass": 33, "electric_guitar": 27,
    "keys": 0, "pads": 89, "drums": 0, "kick": 0, "snare": 0,
}


def write_midi(midi_obj, path: pathlib.Path, program: int = 0, is_drum: bool = False) -> pathlib.Path | None:
    """Write a transcription to a .mid file with a sensible instrument sound."""
    if midi_obj is None:
        return None
    try:
        for inst in midi_obj.instruments:
            inst.program = program
            inst.is_drum = is_drum
        path.parent.mkdir(parents=True, exist_ok=True)
        midi_obj.write(str(path))
        return path
    except Exception as exc:  # noqa: BLE001
        log.warning("could not write %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------
# Note-informed reconstruction
# --------------------------------------------------------------------------


def _harmonic_bins(f0: float, n_fft: int, sr: int, max_harmonics: int) -> list[tuple[int, int]]:
    """(harmonic number, bin index) for the partials that fit below Nyquist."""
    out = []
    for h in range(1, max_harmonics + 1):
        f = h * f0
        if f >= sr / 2 * 0.92:
            break
        k = int(round(f * n_fft / sr))
        if 0 < k < n_fft // 2:
            out.append((h, k))
    return out


def harmonic_restore(
    stem: np.ndarray,
    parent: np.ndarray,
    notes: list[Note],
    sr: int,
    max_harmonics: int = 14,
    damage_ratio: float = 0.35,
    presence_floor_db: float = -48.0,
    min_clean_siblings: int = 3,
    fit_tolerance: float = 0.45,
    min_low_harmonics: int = 3,
    n_fft: int = 4096,
) -> tuple[np.ndarray, dict]:
    """Rebuild buried partials from the clean partials of the same note."""
    hop = n_fft // 4
    n = stem.shape[-1]
    stem = np.ascontiguousarray(stem, dtype=np.float32)
    parent = match_shape(np.asarray(parent, dtype=np.float32), stem)

    if not notes:
        return stem, {"notes": 0, "partials_rebuilt": 0, "partials_refused": 0, "notes_never_clean": 0}

    S = _stft(stem, n_fft, hop)
    P = _stft(parent, n_fft, hop)
    out = S.copy()

    n_frames = S.shape[-1]
    frame_times = np.arange(n_frames) * hop / sr

    rebuilt = refused = never_clean = 0

    for ch in range(S.shape[0]):
        ms, mp = np.abs(S[ch]), np.abs(P[ch])
        peak = ms.max()
        floor = peak * (10.0 ** (presence_floor_db / 20.0)) if peak > EPS else EPS
        share = ms / np.maximum(mp, EPS)

        # Owning a bin is not the same as having something in it. An empty
        # bin has a share near 1 simply because nobody else is there either,
        # so "clean" alone would accept silence as evidence of a partial -
        # which is how a wrong pitch talks itself into existence. A real
        # partial also stands above its neighbours, so require a local peak.
        from scipy.ndimage import maximum_filter1d

        neighbourhood = maximum_filter1d(ms, size=5, axis=0, mode="nearest")
        peaky = ms >= neighbourhood * 0.75
        clean = (ms > floor) & (share > damage_ratio) & peaky

        for note in notes:
            a = int(np.searchsorted(frame_times, note.start))
            b = int(np.searchsorted(frame_times, note.end))
            if b - a < 2:
                continue
            bins = _harmonic_bins(note.f0, n_fft, sr, max_harmonics)
            if len(bins) < min_clean_siblings + 1:
                continue

            ks = np.array([k for _h, k in bins])
            amp = ms[ks, a:b]  # (harmonics, frames)
            ok = clean[ks, a:b]

            # The note's own balance between partials, measured only on the
            # frames where that partial was not covered by anything.
            ratios = np.zeros(len(ks))
            for i in range(len(ks)):
                vals = amp[i][ok[i]]
                ratios[i] = float(np.median(vals)) if vals.size >= 2 else 0.0
            total = ratios.sum()
            if total < EPS or (ratios > 0).sum() < min_clean_siblings + 1:
                # Never heard cleanly enough to learn anything from.
                never_clean += 1
                continue
            # A real note shows itself in its lowest partials. If those were
            # never clean, what we are looking at is most likely another
            # note's harmonics that happen to land nearby.
            low = min(min_low_harmonics + 1, len(ratios))
            if int((ratios[:low] > 0).sum()) < min_low_harmonics:
                never_clean += 1
                continue
            ratios = ratios / total

            for t in range(b - a):
                good = ok[:, t] & (ratios > 0)
                if good.sum() < min_clean_siblings:
                    refused += int((~ok[:, t]).sum())
                    continue

                # Least-squares level of the note at this instant, from the
                # partials that are visible: a_h ~= level * ratio_h.
                r = ratios[good]
                obs = amp[good, t]
                denom = float(np.dot(r, r))
                if denom < EPS:
                    continue
                level = float(np.dot(obs, r)) / denom
                if level <= 0:
                    continue

                # Does this note actually explain what is there? A wrong
                # pitch still finds energy - some of its harmonics land near
                # a real note's partials by coincidence - and would then
                # happily rebuild the rest out of nothing. So the hypothesis
                # has to fit the partials we can see before it is allowed to
                # say anything about the ones we cannot.
                predicted = level * r
                err = float(np.linalg.norm(obs - predicted))
                scale = float(np.linalg.norm(obs))
                if scale < EPS or err / scale > fit_tolerance:
                    refused += int((~ok[:, t]).sum())
                    continue

                frame = a + t
                for i, (h, k) in enumerate(bins):
                    if ok[i, t] or ratios[i] <= 0:
                        continue
                    est = level * ratios[i]
                    if est <= ms[k, frame]:
                        continue  # already louder than we would predict
                    # Never claim more than the recording held in that bin.
                    est = min(est, mp[k, frame])
                    if est <= ms[k, frame]:
                        continue

                    # A partial at h*f0 turns by exactly this much per frame.
                    advance = 2.0 * np.pi * (h * note.f0) * hop / sr
                    has_clean_previous = t > 0 and ok[i, t - 1]
                    if has_clean_previous:
                        phase = float(np.angle(S[ch, k, frame - 1])) + advance
                    else:
                        phase = float(np.angle(out[ch, k, frame]))
                    out[ch, k, frame] = est * np.exp(1j * phase)
                    rebuilt += 1

    result = _istft(out, hop, n, n_fft).astype(np.float32)
    report = {
        "notes": len(notes),
        "partials_rebuilt": rebuilt,
        "partials_refused": refused,
        "notes_never_clean": never_clean,
    }
    return result, report
