"""Reverse, then forward again: rebuilding what the cut took out.

A masking separator does not damage a stem by adding noise to it. It damages
it by *removing* things. Where the guitar and the snare land on the same
time-frequency bin, the bin goes to one of them, and the guitar is left with
a hole exactly where the snare hit. That hole is the warbling, underwater
sound - it is an absence, not an addition.

So cutting harder can never fix it. The only fix is to put back what was
taken, and the honest way to do that is to work out what *must* have been
there rather than invent something plausible:

  time continuation   a partial that is steady before the collision and
                      steady after it was steady during it. Interpolating
                      across 30 ms of a held note is reading a line between
                      two known points, not guessing.

  the mix ceiling     the rebuilt value is never allowed to exceed what the
                      original mix actually contained in that bin. We can
                      restore energy that was there and got assigned
                      elsewhere; we can never create energy the recording
                      never held.

  refusal             if a note starts inside the collision and is never
                      heard cleanly, there is nothing to continue from. The
                      hole stays. That is the line between reconstruction
                      and fiction, and the report says how often we stopped
                      at it.

The result no longer sums to the original - it contains energy that was
restored rather than allocated - which is why the exact stems are kept
alongside it. One chain is the proof, the other is the one you play.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from .audio import EPS, _istft, _stft, match_shape

log = logging.getLogger(__name__)


@dataclass
class RestoreReport:
    key: str
    label: str
    bins_damaged: int
    bins_restored: int
    bins_refused: int
    energy_gain_db: float
    share_restored: float

    def to_dict(self) -> dict:
        return asdict(self)


def _rebuild_gaps(
    spec: np.ndarray,
    alive: np.ndarray,
    ceiling: np.ndarray,
    max_gap: int,
    hop: int,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild each frequency row across the short gaps between live frames.

    Magnitude is continued as a straight line in log-amplitude: a held
    partial swells or decays smoothly, so reading between two known points
    is continuation, not invention.

    Phase is continued at the partial's *own* rate, measured from the last
    two frames before the gap. This matters more than the magnitude. The
    obvious shortcut is to borrow the phase from the original mix, but
    during a snare hit the mix's phase at that bin *is* the snare's - so
    that puts back the right amount of energy in the wrong place and the
    result sounds no better than the hole. A steady partial advances by a
    fixed amount per hop, and that is what gets extrapolated.

    Returns (rebuilt spectrum, repaired mask, refused mask).
    """
    out = spec.copy()
    repaired = np.zeros_like(alive, dtype=bool)
    refused = np.zeros_like(alive, dtype=bool)

    n_freq, _ = spec.shape
    mag = np.abs(spec)
    log_mag = np.log(np.maximum(mag, EPS))
    phase = np.angle(spec)

    for k in range(n_freq):
        row = alive[k]
        if not row.any():
            continue
        idx = np.flatnonzero(row)
        if idx.size < 2:
            continue

        # Expected phase advance per hop for a partial sitting at this bin.
        nominal = 2.0 * np.pi * k * hop / n_fft

        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if gap <= 0:
                continue
            if gap > max_gap:
                refused[k, a + 1 : b] = True
                continue

            # Measure the partial's real advance if we have two clean frames
            # in a row before the gap; otherwise fall back to the bin rate.
            if a >= 1 and alive[k, a - 1]:
                delta = phase[k, a] - phase[k, a - 1]
                delta = nominal + np.angle(np.exp(1j * (delta - nominal)))
            else:
                delta = nominal

            steps = np.arange(1, gap + 1)
            w = steps / (gap + 1)
            new_mag = np.exp(log_mag[k, a] * (1 - w) + log_mag[k, b] * w)
            new_mag = np.minimum(new_mag, ceiling[k, a + 1 : b])
            new_phase = phase[k, a] + delta * steps

            out[k, a + 1 : b] = new_mag * np.exp(1j * new_phase)
            repaired[k, a + 1 : b] = True

    return out, repaired, refused


def restore_stem(
    stem: np.ndarray,
    parent: np.ndarray,
    sr: int,
    key: str = "",
    label: str = "",
    max_gap_ms: float = 250.0,
    damage_ratio: float = 0.35,
    presence_floor_db: float = -60.0,
    n_fft: int = 4096,
    headroom_db: float = 0.0,
) -> tuple[np.ndarray, RestoreReport]:
    """Fill the holes a collision left in `stem`, bounded by `parent`."""
    hop = n_fft // 4
    n = stem.shape[-1]
    stem = np.ascontiguousarray(stem, dtype=np.float32)
    parent = match_shape(np.asarray(parent, dtype=np.float32), stem)

    S = _stft(stem, n_fft, hop)
    P = _stft(parent, n_fft, hop)
    mag_s = np.abs(S)
    mag_p = np.abs(P)

    ceiling = mag_p * (10.0 ** (headroom_db / 20.0))

    out = np.empty_like(S)
    damaged = restored = refused_total = 0

    max_gap = max(1, int(max_gap_ms * 1e-3 * sr / hop))

    for ch in range(S.shape[0]):
        ms, mp = mag_s[ch], mag_p[ch]

        # "Alive" = this stem genuinely owns the bin right now.
        peak = ms.max()
        floor = peak * (10.0 ** (presence_floor_db / 20.0)) if peak > EPS else EPS
        share = ms / np.maximum(mp, EPS)
        alive = (ms > floor) & (share > damage_ratio)

        # A bin only counts as damage if something actually took it: the mix
        # had real energy there and this stem ended up with a small share of
        # it. A bin that is simply quiet was not stolen - the note just is
        # not playing - and rebuilding those would be inventing notes.
        lost = (mp > floor) & (share <= damage_ratio)

        rebuilt_spec, repaired, refused = _rebuild_gaps(
            S[ch], alive, ceiling[ch], max_gap, hop, n_fft
        )

        apply = repaired & lost
        damaged += int(((repaired | refused) & lost).sum())
        restored += int(apply.sum())
        refused_total += int((refused & lost).sum())
        out[ch] = np.where(apply, rebuilt_spec, S[ch])

    rebuilt = _istft(out, hop, n, n_fft)

    before = float(np.mean(stem**2))
    after = float(np.mean(rebuilt**2))
    gain_db = 10.0 * np.log10(after / before) if before > EPS and after > EPS else 0.0

    report = RestoreReport(
        key=key,
        label=label or key,
        bins_damaged=damaged,
        bins_restored=restored,
        bins_refused=refused_total,
        energy_gain_db=round(gain_db, 2),
        share_restored=round(restored / damaged, 4) if damaged else 0.0,
    )
    return rebuilt.astype(np.float32), report


def enforce_mix_consistency(
    restored: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    mix: np.ndarray,
    keys: list[str],
    n_fft: int = 4096,
    tolerance_db: float = 1.5,
) -> tuple[dict[str, np.ndarray], dict]:
    """Keep the rebuilt stems reconcilable with the original recording.

    Each stem is capped against the mix on its own, but that is not enough:
    five stems can each stay under the mix and still add up to more than the
    mix ever contained. This is the joint version of the question - if every
    instrument really played what we say it played, does the record still
    come out the same?

    Where the rebuilt stems together exceed what the recording held, the
    *rebuilt part* is scaled back proportionally until they fit. The exact
    chain is the floor and is never touched, so this can only remove
    invention, never real signal.
    """
    hop = n_fft // 4
    n = mix.shape[-1]
    present = [k for k in keys if k in restored and k in exact]
    if not present:
        return restored, {"bins_trimmed": 0, "excess_before_db": 0.0}

    X_mix = _stft(mix, n_fft, hop)
    mag_mix = np.abs(X_mix)

    specs_r = {k: _stft(match_shape(restored[k], mix), n_fft, hop) for k in present}
    specs_e = {k: _stft(match_shape(exact[k], mix), n_fft, hop) for k in present}

    # Work on the complex sum, not the sum of magnitudes. The exact stems add
    # up to the recording bit for bit, but their magnitudes add up to far more
    # than it wherever their phases differ - measuring that way would flag an
    # honest set of stems as over budget everywhere.
    added = {k: specs_r[k] - specs_e[k] for k in present}
    total_added = sum(added.values())
    mag_added = np.abs(total_added)

    allowance = mag_mix * (10.0 ** (tolerance_db / 20.0))
    # |mix + s*A| <= |mix| + s*|A|, so this bound is safe and monotone in s.
    headroom = np.maximum(allowance - mag_mix, 0.0)
    scale = np.where(mag_added > EPS, np.clip(headroom / np.maximum(mag_added, EPS), 0.0, 1.0), 1.0)

    over = mag_added > headroom + EPS
    trimmed = int(over.sum())
    before = float(np.sum(np.maximum(mag_added - headroom, 0.0) ** 2))

    out: dict[str, np.ndarray] = dict(restored)
    for k in present:
        out[k] = _istft(specs_e[k] + added[k] * scale, hop, n, n_fft).astype(np.float32)

    report = {
        "bins_trimmed": trimmed,
        "excess_before_db": round(10.0 * np.log10(before + EPS), 1),
        "tolerance_db": tolerance_db,
    }
    return out, report


def restore_all(
    stems: dict[str, np.ndarray],
    parents: dict[str, str],
    labels: dict[str, str],
    keys: list[str],
    sr: int,
    max_gap_ms: float = 250.0,
    headroom_db: float = 0.0,
    notes: dict[str, list] | None = None,
    analysis=None,
    offset: float = 0.0,
    min_agreement: float = 0.45,
    syntheses: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Restore every requested stem against its own parent.

    Four mechanisms, strongest evidence first:

      repeats    the same bar, actually recorded, from a chorus where
                 nothing was on top of it. Not an inference at all.
      harmonics  the note's other partials at the same instant.
      score      the instrument rebuilt from its own notes and timbre.
      time       the partial's own past and future.

    Each pass only touches what the ones before it left alone.
    """
    out: dict[str, np.ndarray] = {}
    reports: dict[str, dict] = {}
    for key in keys:
        if key not in stems:
            continue
        parent_key = parents.get(key)
        parent = stems.get(parent_key) if parent_key else stems.get("mix")
        if parent is None:
            continue
        try:
            working = stems[key]
            harmonic_report: dict = {}
            repeat_report: dict = {}
            synth_filled = 0

            if analysis is not None and getattr(analysis, "sections", None):
                from .repeat import borrow_from_repeats

                working, rep = borrow_from_repeats(
                    working, parent, analysis, sr, key, labels.get(key, key),
                    offset=offset, min_agreement=min_agreement,
                )
                repeat_report = rep.to_dict()

            note_list = (notes or {}).get(key)
            if note_list:
                from .harmonic import harmonic_restore

                working, harmonic_report = harmonic_restore(working, parent, note_list, sr)

            synthesis = (syntheses or {}).get(key)
            if synthesis is not None:
                from .resynth import fill_from_synthesis

                working, synth_filled = fill_from_synthesis(working, parent, synthesis, sr)

            rebuilt, report = restore_stem(
                working, parent, sr, key, labels.get(key, key),
                max_gap_ms=max_gap_ms, headroom_db=headroom_db,
            )
            out[key] = rebuilt
            payload = report.to_dict()
            if repeat_report:
                payload["repeats"] = repeat_report
            if harmonic_report:
                payload["harmonic"] = harmonic_report
            if synth_filled:
                payload["from_score"] = synth_filled
            reports[key] = payload
        except Exception as exc:  # noqa: BLE001
            log.warning("restore failed for %s: %s", key, exc)
    return out, reports
