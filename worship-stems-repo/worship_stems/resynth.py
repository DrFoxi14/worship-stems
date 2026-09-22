"""Closing the loop: build the instrument back from its own score.

Everything so far repairs a stem locally - fill this hole, borrow that bar.
This does the thing properly: work out what the instrument played, learn how
that instrument sounds, rebuild it from scratch, and then check the rebuild
against the recording. What the rebuild fails to explain is the residual,
and the residual is not an error to hide - it is the list of what we got
wrong, which is exactly what tells us where to look next.

    notes ---> timbre ---> synthesis ---> residual ---> missed notes
      ^                                                      |
      `------------------------------------------------------'

Each round the residual shrinks. When it stops shrinking, the score and the
timbre together are as much of that instrument as we can account for, and
the fraction of the stem's energy they explain is an honest measure of how
well we actually understood it - far more honest than any dB figure, because
a number you cannot resynthesise is a number you cannot check.

The timbre is learned from the recording itself, per instrument: the average
balance of partials over the notes we heard clearly. Nothing generic, nothing
pretrained - this guitar, in this room, on this day.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

from .audio import EPS, _istft, _stft, match_shape
from .harmonic import Note, transcribe

log = logging.getLogger(__name__)


@dataclass
class Timbre:
    """How this instrument distributes energy across its partials."""

    profile: np.ndarray  # relative amplitude per harmonic, normalised
    observations: int = 0

    def amplitudes(self, count: int) -> np.ndarray:
        out = np.zeros(count)
        n = min(count, len(self.profile))
        out[:n] = self.profile[:n]
        return out


@dataclass
class ResynthReport:
    key: str
    label: str
    rounds: int = 0
    notes_initial: int = 0
    notes_final: int = 0
    explained: float = 0.0  # 0..1 of the stem's energy
    residual_db: float = 0.0
    history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Timbre
# --------------------------------------------------------------------------


def learn_timbre(
    stem: np.ndarray,
    notes: list[Note],
    sr: int,
    max_harmonics: int = 16,
    n_fft: int = 4096,
) -> Timbre:
    """Measure the instrument's partial balance from the notes we can hear."""
    hop = n_fft // 4
    S = _stft(stem, n_fft, hop)
    mag = np.abs(S).mean(axis=0)
    n_frames = mag.shape[-1]
    frame_times = np.arange(n_frames) * hop / sr

    rows: list[np.ndarray] = []
    for note in notes:
        a = int(np.searchsorted(frame_times, note.start))
        b = int(np.searchsorted(frame_times, note.end))
        if b - a < 2:
            continue
        profile = np.zeros(max_harmonics)
        for h in range(1, max_harmonics + 1):
            f = h * note.f0
            if f >= sr / 2 * 0.92:
                break
            k = int(round(f * n_fft / sr))
            if k >= mag.shape[0]:
                break
            profile[h - 1] = float(np.median(mag[k, a:b]))
        total = profile.sum()
        if total > EPS:
            rows.append(profile / total)

    if not rows:
        # Nothing measurable: a gently falling series, the shape almost every
        # plucked or bowed instrument has, used only as a last resort.
        fallback = np.array([1.0 / (h**1.2) for h in range(1, max_harmonics + 1)])
        return Timbre(fallback / fallback.sum(), 0)

    stacked = np.stack(rows)
    profile = np.median(stacked, axis=0)
    total = profile.sum()
    return Timbre(profile / total if total > EPS else profile, len(rows))


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------


def synthesize(
    notes: list[Note],
    timbre: Timbre,
    duration: float,
    sr: int,
    channels: int = 2,
    max_harmonics: int = 16,
    attack_ms: float = 8.0,
    release_ms: float = 60.0,
) -> np.ndarray:
    """Render the score with the learned timbre."""
    n = int(duration * sr)
    out = np.zeros(n, dtype=np.float64)
    if n <= 0 or not notes:
        return np.zeros((channels, max(n, 0)), dtype=np.float32)

    amps = timbre.amplitudes(max_harmonics)
    attack = max(1, int(attack_ms * 1e-3 * sr))
    release = max(1, int(release_ms * 1e-3 * sr))

    for note in notes:
        a = int(note.start * sr)
        b = min(n, int(note.end * sr) + release)
        if b - a < attack + 2:
            continue
        length = b - a
        tt = np.arange(length) / sr

        env = np.ones(length)
        env[:attack] = np.linspace(0.0, 1.0, attack) ** 2
        env[-release:] *= np.linspace(1.0, 0.0, release) ** 2
        # A plucked note decays; a bowed one does not. Half-way is closer to
        # both than either extreme, and the residual corrects the rest.
        env *= np.exp(-tt * 0.7)

        wave = np.zeros(length)
        for h in range(1, max_harmonics + 1):
            f = h * note.f0
            if f >= sr / 2 * 0.92 or amps[h - 1] <= 0:
                continue
            wave += amps[h - 1] * np.sin(2 * np.pi * f * tt + h * 0.3)

        out[a:b] += wave * env * note.amplitude

    peak = float(np.max(np.abs(out)))
    if peak > EPS:
        out = out / peak
    return np.stack([out] * channels).astype(np.float32)


def fit_gain(reference: np.ndarray, estimate: np.ndarray) -> float:
    """The single scale that makes the synthesis fit the recording best."""
    r = reference.reshape(-1).astype(np.float64)
    e = match_shape(estimate, reference).reshape(-1).astype(np.float64)
    denom = float(np.dot(e, e))
    if denom < EPS:
        return 0.0
    return float(np.dot(r, e)) / denom


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def explain(
    stem: np.ndarray,
    sr: int,
    key: str = "",
    label: str = "",
    rounds: int = 3,
    notes: list[Note] | None = None,
    min_improvement: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, ResynthReport]:
    """Rebuild a stem from its score, feeding the residual back each round.

    Returns (synthesis, residual, report).
    """
    stem = np.ascontiguousarray(stem, dtype=np.float32)
    duration = stem.shape[-1] / sr
    report = ResynthReport(key=key, label=label or key)

    found = notes if notes is not None else transcribe(stem, sr)[0]
    report.notes_initial = len(found)
    if not found:
        return np.zeros_like(stem), stem.copy(), report

    all_notes = list(found)
    best_synth = np.zeros_like(stem)
    best_residual = stem.copy()
    best_explained = 0.0
    total_energy = float(np.mean(stem**2)) + EPS

    for round_index in range(max(1, rounds)):
        timbre = learn_timbre(stem, all_notes, sr)
        synth = synthesize(all_notes, timbre, duration, sr, channels=stem.shape[0])
        synth = match_shape(synth, stem) * fit_gain(stem, synth)

        residual = stem - synth
        explained = 1.0 - float(np.mean(residual**2)) / total_energy
        report.history.append(round(float(explained), 4))
        report.rounds = round_index + 1

        if explained > best_explained:
            best_explained, best_synth, best_residual = explained, synth, residual
        if round_index and explained - report.history[-2] < min_improvement:
            break

        # What the score could not account for is where the missed notes are.
        if round_index < rounds - 1:
            extra, _midi = transcribe(residual, sr, onset_threshold=0.4, frame_threshold=0.25)
            if not extra:
                break
            known = {(round(nn.start, 2), nn.pitch) for nn in all_notes}
            fresh = [nn for nn in extra if (round(nn.start, 2), nn.pitch) not in known]
            if not fresh:
                break
            all_notes.extend(fresh)

    report.notes_final = len(all_notes)
    report.explained = round(float(np.clip(best_explained, 0.0, 1.0)), 4)
    ref = float(np.sqrt(np.mean(stem**2)))
    err = float(np.sqrt(np.mean(best_residual**2)))
    report.residual_db = round(20.0 * np.log10(err / ref), 1) if ref > EPS and err > EPS else 0.0
    return best_synth, best_residual, report


def fill_from_synthesis(
    stem: np.ndarray,
    parent: np.ndarray,
    synthesis: np.ndarray,
    sr: int,
    damage_ratio: float = 0.35,
    presence_floor_db: float = -55.0,
    n_fft: int = 4096,
) -> tuple[np.ndarray, int]:
    """Use the resynthesis to fill bins the cut destroyed.

    The synthesis has no holes in it - it was built from the score, not cut
    out of a mix - so it is a clean source for exactly the bins that are
    missing. It is still capped by what the recording held, and it is only
    ever allowed into bins something else took.
    """
    hop = n_fft // 4
    n = stem.shape[-1]
    stem = np.ascontiguousarray(stem, dtype=np.float32)
    parent = match_shape(np.asarray(parent, dtype=np.float32), stem)
    synthesis = match_shape(np.asarray(synthesis, dtype=np.float32), stem)

    S = _stft(stem, n_fft, hop)
    P = _stft(parent, n_fft, hop)
    Y = _stft(synthesis, n_fft, hop)
    out = S.copy()
    filled = 0

    from scipy.ndimage import maximum_filter1d

    for ch in range(S.shape[0]):
        ms, mp, my = np.abs(S[ch]), np.abs(P[ch]), np.abs(Y[ch])
        peak = ms.max()
        floor = peak * (10.0 ** (presence_floor_db / 20.0)) if peak > EPS else EPS
        share = ms / np.maximum(mp, EPS)

        # The synthesis is only worth trusting at its own partials. Away from
        # them it is the additive model's own smear, and letting that in fills
        # broadband damage with a hum that was never played - which scores
        # worse than leaving the hole alone.
        syn_peak = my.max()
        syn_floor = syn_peak * (10.0 ** (-40.0 / 20.0)) if syn_peak > EPS else EPS
        neighbourhood = maximum_filter1d(my, size=7, axis=0, mode="nearest")
        on_partial = (my >= neighbourhood * 0.85) & (my > syn_floor)

        damaged = (mp > floor) & (share <= damage_ratio) & on_partial & (my > ms * 1.5)

        if not damaged.any():
            continue
        new_mag = np.where(damaged, np.minimum(my, mp), ms)

        # The synthesis knows each partial's exact frequency but its absolute
        # phase is arbitrary - it was rendered, not recorded. Dropping that
        # phase in fights the surviving audio instead of joining it. So take
        # the phase *advance* from the synthesis and anchor it to the last
        # frame the recording itself was heard clearly.
        clean = ~damaged & (ms > floor)
        width = ms.shape[1]
        rows = np.arange(ms.shape[0])[:, None]
        advance = np.angle(Y[ch][:, 1:] * np.conj(Y[ch][:, :-1]))
        advance = np.concatenate([np.zeros((ms.shape[0], 1)), advance], axis=1)
        cumulative = np.cumsum(advance, axis=1)

        order = np.where(clean, np.arange(width)[None, :], -1)
        last = np.maximum.accumulate(order, axis=1)
        usable = last >= 0
        safe_last = np.maximum(last, 0)
        host_phase = np.angle(S[ch])
        continued = host_phase[rows, safe_last] + (cumulative - cumulative[rows, safe_last])

        apply = damaged & usable
        new_phase = np.where(apply, continued, host_phase)
        out[ch] = np.where(apply, new_mag, ms) * np.exp(1j * new_phase)
        filled += int(apply.sum())

    return _istft(out, hop, n, n_fft).astype(np.float32), filled
