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
    """How this instrument distributes energy across the spectrum.

    Two representations, and the difference between them is the difference
    between a voice that keeps its vowel when transposed and one that does
    not.

    `profile` is the old one: amplitude per harmonic *number*. It cannot
    represent a formant, because a formant sits at a fixed frequency while
    the harmonic that lands on it changes with the note. A vowel with
    formants at 700/1150/2600 Hz measured on G3 and on G4 gives two
    completely different harmonic profiles, and averaging them - which is
    what learning from a whole stem does - smears both into something that
    is neither.

    `envelope` is amplitude per *frequency*, on the fixed grid `env_freqs`.
    Every note's observations land in the same bins no matter what pitch it
    was sung at, so the average is meaningful, and a note re-rendered at a
    new pitch reads its harmonic amplitudes off the same fixed curve - which
    is exactly what preserving formants means. Transposition then needs no
    pitch shifting at all: move f0, leave the envelope alone.

    `profile` is kept so anything that asks for it still works, and so a
    timbre measured from too few notes to build an envelope still renders.
    """

    profile: np.ndarray  # relative amplitude per harmonic, normalised
    observations: int = 0
    envelope: np.ndarray | None = None  # amplitude per frequency
    env_freqs: np.ndarray | None = None  # the frequencies, Hz, ascending

    def has_envelope(self) -> bool:
        return (
            self.envelope is not None
            and self.env_freqs is not None
            and len(self.envelope) == len(self.env_freqs)
            and float(np.max(self.envelope)) > EPS
        )

    def amplitudes(self, count: int, f0: float | None = None) -> np.ndarray:
        """Harmonic amplitudes for a note at `f0`.

        With an envelope and a pitch, the amplitudes are read off the fixed
        curve at f0, 2*f0, 3*f0 ... With neither, this is the old behaviour.
        """
        if f0 is not None and f0 > 0 and self.has_envelope():
            freqs = np.arange(1, count + 1, dtype=np.float64) * f0
            out = np.interp(freqs, self.env_freqs, self.envelope, left=0.0, right=0.0)
            total = out.sum()
            if total > EPS:
                return out / total
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


ENV_BINS = 96
ENV_F_LOW = 55.0


def _env_grid(sr: int, bins: int = ENV_BINS) -> np.ndarray:
    """Log-spaced frequencies for the envelope.

    Log spacing because formants are a constant fraction of an octave wide,
    not a constant number of Hz - at 96 bins this is about a tenth of an
    octave, which is roughly 45 Hz around the first formant of a sung vowel.
    """
    return np.geomspace(ENV_F_LOW, sr / 2 * 0.92, bins)


def learn_timbre(
    stem: np.ndarray,
    notes: list[Note],
    sr: int,
    max_harmonics: int = 16,
    n_fft: int = 4096,
) -> Timbre:
    """Measure the instrument's spectral shape from the notes we can hear.

    Two things are built from the same measurements: the old per-harmonic
    profile, and the frequency envelope. The envelope is what makes the
    average over notes of different pitches mean anything - see `Timbre`.
    """
    hop = n_fft // 4
    S = _stft(stem, n_fft, hop)
    mag = np.abs(S).mean(axis=0)
    n_frames = mag.shape[-1]
    frame_times = np.arange(n_frames) * hop / sr

    grid = _env_grid(sr)
    obs_f: list[float] = []
    obs_a: list[float] = []
    rows: list[np.ndarray] = []

    for note in notes:
        a = int(np.searchsorted(frame_times, note.start))
        b = int(np.searchsorted(frame_times, note.end))
        if b - a < 2:
            continue
        profile = np.zeros(max_harmonics)
        freqs = np.zeros(max_harmonics)
        for h in range(1, max_harmonics + 1):
            f = h * note.f0
            if f >= sr / 2 * 0.92:
                break
            k = int(round(f * n_fft / sr))
            if k >= mag.shape[0]:
                break
            profile[h - 1] = float(np.median(mag[k, a:b]))
            freqs[h - 1] = f

        total = profile.sum()
        if total <= EPS:
            continue
        rows.append(profile / total)

        # Each note contributes shape, not level. Normalising by the note's
        # peak looks right and is not: it forces every note to put a 1.0
        # somewhere, and different notes put it in different bins, so the
        # median across notes comes back as a flat plateau that merges
        # neighbouring formants into one. Centring the log amplitudes on
        # each note's own mean removes the level without anchoring any bin.
        usable = [(a, f) for a, f in zip(profile, freqs) if f > 0.0 and a > EPS]
        if len(usable) < 2:
            continue
        logs = np.log10([a for a, _ in usable])
        logs = logs - logs.mean()
        for lg, (_, f) in zip(logs, usable):
            obs_f.append(float(f))
            obs_a.append(float(lg))

    if not rows:
        # Nothing measurable: a gently falling series, the shape almost every
        # plucked or bowed instrument has, used only as a last resort.
        fallback = np.array([1.0 / (h**1.2) for h in range(1, max_harmonics + 1)])
        return Timbre(fallback / fallback.sum(), 0)

    profile = np.median(np.stack(rows), axis=0)
    total = profile.sum()
    profile = profile / total if total > EPS else profile

    envelope, env_freqs = _build_envelope(grid, np.array(obs_f), np.array(obs_a))
    return Timbre(profile, len(rows), envelope, env_freqs)


def _build_envelope(grid, obs_f, obs_a, width_oct: float = 0.16):
    """Kernel regression over every observation, in log-frequency.

    Bucketing and taking a median does not work here: the bins are populated
    very unevenly, because which bins a note touches depends on its pitch.
    Bins fed by many notes end up stable and bins fed by one end up noise,
    and a median cannot tell the two apart.

    Smoothing over all the observations at once fixes that. Each grid point
    is a weighted average of every observation near it, with a Gaussian
    weight in log-frequency, so the curve follows real formants and ignores
    the gaps between one note's harmonics.

    The width is calibrated, not guessed: on synthesised vowels with formants
    known to be at 700/1150/2600 Hz, widths from 0.06 to 0.30 octaves were
    scored against the true curve. Narrow widths chase the gaps between
    harmonics and invent peaks; wide ones smooth the first two formants into
    one hump, which is what tells /a/ from /o/. 0.16 was the compromise: log
    error 0.286 against 0.273 for the flattest, while still resolving three
    distinct formants. Re-check it on a real voice before trusting it.
    """
    if obs_f.size < 6:
        return None, None
    lg_grid = np.log2(grid)
    lg_obs = np.log2(np.maximum(obs_f, 1e-6))
    out = np.empty(len(grid))
    for i, g in enumerate(lg_grid):
        w = np.exp(-0.5 * ((lg_obs - g) / width_oct) ** 2)
        s = w.sum()
        out[i] = float(np.dot(w, obs_a) / s) if s > 1e-9 else np.nan
    known = ~np.isnan(out)
    if known.sum() < 3:
        return None, None
    out = np.interp(lg_grid, lg_grid[known], out[known])
    envelope = 10.0 ** out
    peak = float(envelope.max())
    if peak <= EPS:
        return None, None
    return envelope / peak, grid


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

    attack = max(1, int(attack_ms * 1e-3 * sr))
    release = max(1, int(release_ms * 1e-3 * sr))

    for note in notes:
        # Per note, not once for all of them: with a frequency envelope the
        # harmonic amplitudes depend on where this note's harmonics land.
        amps = timbre.amplitudes(max_harmonics, note.f0)
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
