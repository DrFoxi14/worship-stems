"""Tempo, key, beat grid and song structure.

Two backends:

  allin1  - the All-In-One Music Structure Analyzer (mir-aidj). One pass gives
            tempo, beats, downbeats and functional segments labelled
            intro / verse / chorus / bridge / inst / solo / break / outro.
            This is what makes a usable Guide track possible.

  builtin - a librosa-only fallback (beat tracking + Laplacian structure
            segmentation + heuristic labelling). Less accurate on labels but
            needs no extra models, so the app still works if allin1 is absent.

Either way the result is the same dataclass and the boundaries are snapped to
downbeats, because a Guide cue that lands half a bar early is useless.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

log = logging.getLogger(__name__)

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Camelot wheel, for transposing to a key the band can actually sing.
CAMELOT = {
    ("B", "major"): "1B", ("F#", "major"): "2B", ("C#", "major"): "3B", ("G#", "major"): "4B",
    ("D#", "major"): "5B", ("A#", "major"): "6B", ("F", "major"): "7B", ("C", "major"): "8B",
    ("G", "major"): "9B", ("D", "major"): "10B", ("A", "major"): "11B", ("E", "major"): "12B",
    ("G#", "minor"): "1A", ("D#", "minor"): "2A", ("A#", "minor"): "3A", ("F", "minor"): "4A",
    ("C", "minor"): "5A", ("G", "minor"): "6A", ("D", "minor"): "7A", ("A", "minor"): "8A",
    ("E", "minor"): "9A", ("B", "minor"): "10A", ("F#", "minor"): "11A", ("C#", "minor"): "12A",
}

SECTION_ORDER = ["intro", "verse", "prechorus", "chorus", "bridge", "inst", "solo", "break", "outro"]


@dataclass
class Section:
    start: float
    end: float
    label: str
    index: int = 1  # "Verse 2" -> index 2

    @property
    def display(self) -> str:
        pretty = {
            "intro": "Intro", "verse": "Verse", "prechorus": "Pre-Chorus", "chorus": "Chorus",
            "bridge": "Bridge", "inst": "Instrumental", "solo": "Solo", "break": "Break",
            "outro": "Outro", "end": "End",
        }.get(self.label, self.label.title())
        multi = {"verse", "chorus", "bridge"}
        return f"{pretty} {self.index}" if self.label in multi and self.index > 0 else pretty


@dataclass
class Analysis:
    bpm: float
    beats: list[float]
    downbeats: list[float]
    beats_per_bar: int
    key: str
    mode: str
    key_confidence: float
    camelot: str
    duration: float
    sections: list[Section] = field(default_factory=list)
    backend: str = "builtin"

    @property
    def key_display(self) -> str:
        return f"{self.key}{'m' if self.mode == 'minor' else ''}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sections"] = [{**asdict(s), "display": s.display} for s in self.sections]
        d["key_display"] = self.key_display
        return d


# --------------------------------------------------------------------------
# Key
# --------------------------------------------------------------------------


def detect_key(audio: np.ndarray, sr: int) -> tuple[str, str, float]:
    import librosa

    y = audio.mean(axis=0) if audio.ndim > 1 else audio
    y_h = librosa.effects.harmonic(y, margin=3.0)
    chroma = librosa.feature.chroma_cens(y=y_h, sr=sr, bins_per_octave=36)
    profile = chroma.mean(axis=1)
    if profile.sum() <= 0:
        return "C", "major", 0.0
    profile = profile / profile.sum()

    scores: list[tuple[float, str, str]] = []
    for i in range(12):
        rotated = np.roll(profile, -i)
        for name, ref in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            r = float(np.corrcoef(rotated, ref)[0, 1])
            scores.append((r, PITCHES[i], name))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    confidence = float(np.clip((best[0] - second[0]) * 4.0 + best[0], 0.0, 1.0))
    return best[1], best[2], confidence


# --------------------------------------------------------------------------
# Beats
# --------------------------------------------------------------------------


def detect_beats(audio: np.ndarray, sr: int) -> tuple[float, np.ndarray, np.ndarray, int]:
    """Return (bpm, beat times, downbeat times, beats per bar)."""
    import librosa

    y = audio.mean(axis=0) if audio.ndim > 1 else audio
    onset = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, trim=False)
    bpm = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beats) < 4:
        return bpm, beats, beats[:1], 4

    bpb = _guess_meter(onset, beat_frames, sr)
    phase = _best_downbeat_phase(y, sr, beats, bpb)
    downbeats = beats[phase::bpb]
    return bpm, beats, downbeats, bpb


def _guess_meter(onset: np.ndarray, beat_frames: np.ndarray, sr: int) -> int:
    """Pick 4 or 3 by how strongly the onset envelope repeats every N beats."""
    strengths = onset[np.clip(beat_frames, 0, len(onset) - 1)]
    best, best_score = 4, -np.inf
    for bpb in (4, 3):
        scores = [strengths[p::bpb].mean() if len(strengths[p::bpb]) else 0.0 for p in range(bpb)]
        contrast = max(scores) - np.mean(scores)
        if contrast > best_score:
            best, best_score = bpb, contrast
    return best


def _best_downbeat_phase(y: np.ndarray, sr: int, beats: np.ndarray, bpb: int) -> int:
    """Downbeats carry the low end - pick the phase with the most bass energy."""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[freqs < 180].sum(axis=0)
    idx = librosa.time_to_frames(beats, sr=sr, hop_length=512)
    idx = np.clip(idx, 0, len(low) - 1)
    energy = low[idx]
    scores = [energy[p::bpb].mean() if len(energy[p::bpb]) else 0.0 for p in range(bpb)]
    return int(np.argmax(scores))


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def _enforce_min_length(labels: np.ndarray, min_beats: int) -> np.ndarray:
    """Absorb runs shorter than `min_beats` into the stronger neighbour.

    Without this the clustering chops a song into 20+ fragments, which is
    useless for a Guide track: real sections are 8 or 16 bars.
    """
    labels = labels.copy()
    while True:
        bounds = [0, *(np.flatnonzero(np.diff(labels)) + 1), len(labels)]
        runs = list(zip(bounds[:-1], bounds[1:]))
        if len(runs) <= 1:
            return labels
        lengths = [b - a for a, b in runs]
        shortest = int(np.argmin(lengths))
        if lengths[shortest] >= min_beats:
            return labels
        a, b = runs[shortest]
        prev_len = lengths[shortest - 1] if shortest > 0 else -1
        next_len = lengths[shortest + 1] if shortest < len(runs) - 1 else -1
        take_prev = prev_len >= next_len
        labels[a:b] = labels[a - 1] if take_prev else labels[b]


def _segment_builtin(
    audio: np.ndarray,
    sr: int,
    beats: np.ndarray,
    beats_per_bar: int = 4,
    n_segments: int = 6,
    min_bars: int = 4,
) -> list[tuple[float, float, int]]:
    """Laplacian structure segmentation (McFee & Ellis) over beat-synchronous features."""
    import librosa
    from scipy import ndimage

    y = audio.mean(axis=0) if audio.ndim > 1 else audio
    if len(beats) < 8:
        return [(0.0, len(y) / sr, 0)]

    beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=512)
    beat_frames = np.unique(np.clip(beat_frames, 0, None))

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, bins_per_octave=36)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    Csync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    Msync = librosa.util.sync(mfcc, beat_frames, aggregate=np.mean)
    n = Csync.shape[1]
    if n < 8:
        return [(0.0, len(y) / sr, 0)]

    R = librosa.segment.recurrence_matrix(Csync, width=3, mode="affinity", sym=True)
    R = ndimage.median_filter(R, size=(1, 7))
    path_dist = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_dist) or 1.0
    path_sim = np.exp(-path_dist / sigma)
    Rf = np.diag(path_sim, 1) + np.diag(path_sim, -1)

    deg_R, deg_path = np.sum(R, axis=1), np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_R) / (np.sum((deg_path + deg_R) ** 2) or 1.0)
    A = mu * R + (1 - mu) * Rf

    L = np.diag(np.sum(A, axis=1)) - A
    try:
        evals, evecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return [(0.0, len(y) / sr, 0)]
    k = int(min(n_segments, max(2, n // 6)))
    X = evecs[:, :k]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms > 0, norms, 1.0)

    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    # Smooth over roughly two bars, then force sections to a musical minimum.
    filt = max(5, beats_per_bar * 2 + 1)
    labels = ndimage.median_filter(labels, size=filt, mode="nearest")
    labels = _enforce_min_length(labels, min_beats=beats_per_bar * min_bars)

    bounds = [0] + list(np.flatnonzero(np.diff(labels)) + 1) + [n]
    times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        start = float(times[a])
        end = float(times[b]) if b < len(times) else len(y) / sr
        out.append((start, end, int(labels[a])))
    return out


def _label_segments(raw: list[tuple[float, float, int]], audio: np.ndarray, sr: int, duration: float) -> list[Section]:
    """Turn anonymous cluster ids into intro/verse/chorus/bridge/outro."""
    if not raw:
        return [Section(0.0, duration, "verse", 1)]

    y = audio.mean(axis=0) if audio.ndim > 1 else audio
    energy: dict[int, list[float]] = {}
    span: dict[int, float] = {}
    for start, end, cid in raw:
        seg = y[int(start * sr) : int(end * sr)]
        energy.setdefault(cid, []).append(float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0)
        span[cid] = span.get(cid, 0.0) + (end - start)

    counts = {cid: len(v) for cid, v in energy.items()}
    mean_energy = {cid: float(np.mean(v)) for cid, v in energy.items()}

    # Chorus: repeats most, and among the repeaters it is the loudest.
    repeated = [c for c, n in counts.items() if n >= 2] or list(counts)
    chorus_id = max(repeated, key=lambda c: (counts[c], mean_energy[c]))
    remaining = [c for c in counts if c != chorus_id]
    verse_id = max(remaining, key=lambda c: (counts[c], span[c]), default=None)

    labels: list[Section] = []
    for i, (start, end, cid) in enumerate(raw):
        if i == 0 and mean_energy[cid] < mean_energy[chorus_id] * 0.85:
            label = "intro"
        elif i == len(raw) - 1 and end >= duration - 1.0 and cid != chorus_id:
            label = "outro"
        elif cid == chorus_id:
            label = "chorus"
        elif cid == verse_id:
            label = "verse"
        elif counts[cid] == 1 and i > len(raw) // 2:
            label = "bridge"
        else:
            label = "verse"
        labels.append(Section(start, end, label, 0))

    # Merge neighbours that ended up with the same label.
    merged: list[Section] = []
    for s in labels:
        if merged and merged[-1].label == s.label:
            merged[-1].end = s.end
        else:
            merged.append(s)

    seen: dict[str, int] = {}
    for s in merged:
        if s.label in {"verse", "chorus", "bridge"}:
            seen[s.label] = seen.get(s.label, 0) + 1
            s.index = seen[s.label]
    return merged


def rebuild_grid(analysis: Analysis, bpm: float, anchor: float | None = None) -> Analysis:
    """Lay a constant-tempo grid at `bpm`, keeping everything else.

    Used when the musician sets the tempo themselves: the click then becomes
    something to play *to*, rather than a copy of the record's own drift.
    The grid is anchored to the first detected downbeat so it still lines up
    with the recording at the start.
    """
    if bpm <= 0:
        return analysis
    period = 60.0 / bpm
    start = anchor
    if start is None:
        start = float(analysis.downbeats[0]) if analysis.downbeats else (float(analysis.beats[0]) if analysis.beats else 0.0)
    count = int(np.floor((analysis.duration - start) / period)) + 1
    beats = [start + i * period for i in range(max(0, count))]
    bpb = max(1, analysis.beats_per_bar)
    analysis.beats = beats
    analysis.downbeats = beats[::bpb]
    analysis.bpm = round(float(bpm), 2)
    return analysis


def _snap_to_downbeats(sections: list[Section], downbeats: np.ndarray, duration: float) -> list[Section]:
    if len(downbeats) < 2:
        return sections
    db = np.asarray(downbeats)
    for s in sections:
        s.start = float(db[np.argmin(np.abs(db - s.start))])
    sections.sort(key=lambda s: s.start)
    out = []
    for s in sections:
        if out and s.start <= out[-1].start:
            continue
        if out:
            out[-1].end = s.start
        out.append(s)
    if out:
        out[-1].end = duration
    return [s for s in out if s.end - s.start > 1.0]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def analyze(
    audio: np.ndarray,
    sr: int,
    source_path: str | None = None,
    prefer_allin1: bool = True,
) -> Analysis:
    duration = audio.shape[-1] / sr

    result = None
    if prefer_allin1 and source_path:
        result = _try_allin1(source_path, audio, sr, duration)
    if result is not None:
        return result

    # Analysis runs on a 22.05 kHz mono copy: same answers, a few times faster.
    from .audio import resample

    a_sr = 22050
    mono = audio.mean(axis=0, keepdims=True) if audio.ndim > 1 else audio[None, :]
    small = resample(mono, sr, a_sr) if sr != a_sr else mono

    bpm, beats, downbeats, bpb = detect_beats(small, a_sr)
    key, mode, conf = detect_key(small, a_sr)
    raw = _segment_builtin(small, a_sr, beats, beats_per_bar=bpb)
    sections = _label_segments(raw, small, a_sr, duration)
    sections = _snap_to_downbeats(sections, downbeats, duration)
    return Analysis(
        bpm=round(bpm, 2),
        beats=[float(b) for b in beats],
        downbeats=[float(d) for d in downbeats],
        beats_per_bar=bpb,
        key=key,
        mode=mode,
        key_confidence=round(conf, 3),
        camelot=CAMELOT.get((key, mode), ""),
        duration=duration,
        sections=sections,
        backend="builtin",
    )


def _try_allin1(source_path: str, audio: np.ndarray, sr: int, duration: float) -> Analysis | None:
    try:
        import allin1
    except Exception:
        return None
    try:
        res = allin1.analyze(source_path, keep_byproducts=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("allin1 failed (%s), falling back to the built-in analyser", exc)
        return None

    beats = [float(b) for b in getattr(res, "beats", [])]
    downbeats = [float(b) for b in getattr(res, "downbeats", [])]
    bpb = 4
    if len(downbeats) >= 2 and beats:
        per_bar = np.median(np.diff(downbeats)) / (np.median(np.diff(beats)) or 1.0)
        bpb = int(np.clip(round(per_bar), 2, 7))

    sections: list[Section] = []
    seen: dict[str, int] = {}
    for seg in getattr(res, "segments", []):
        label = str(getattr(seg, "label", "verse")).lower()
        if label in {"start", "end"}:
            continue
        idx = 0
        if label in {"verse", "chorus", "bridge"}:
            seen[label] = seen.get(label, 0) + 1
            idx = seen[label]
        sections.append(Section(float(seg.start), float(seg.end), label, idx))
    if not sections:
        return None

    key, mode, conf = detect_key(audio, sr)
    return Analysis(
        bpm=round(float(getattr(res, "bpm", 0.0)) or 0.0, 2),
        beats=beats,
        downbeats=downbeats,
        beats_per_bar=bpb,
        key=key,
        mode=mode,
        key_confidence=round(conf, 3),
        camelot=CAMELOT.get((key, mode), ""),
        duration=duration,
        sections=sections,
        backend="allin1",
    )
