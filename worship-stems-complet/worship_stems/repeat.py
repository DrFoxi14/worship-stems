"""Taking the clean version from another time the band played the same thing.

Everything else in this app *infers* what was in a hole - from the note's
other partials, from the partial's own past. This does not infer anything.
A worship song plays its chorus three or four times, and the collisions land
differently each time: if the guitar is buried under a crash in chorus 1,
chorus 2 almost certainly has that bar in the clear. So instead of working
out what the guitar probably did, take the bar where it was actually
recorded.

That is the strongest evidence available anywhere in this pipeline. It is
not reconstruction at all - it is the same part, genuinely played, borrowed
from a moment when nothing was on top of it.

Two things make it work rather than smear:

  alignment    repeats are never the same length. The two instances are
               warped onto each other bar by bar, using the downbeats we
               already detected, so bar 3 lines up with bar 3 even when the
               tempo drifted between them.

  agreement    the last chorus usually is not the same as the first - extra
               instruments, a key change, a different ending. So before
               borrowing, the two instances are compared where both are
               clean. If they do not already agree there, they are not the
               same performance and nothing is taken.

Only the magnitude is borrowed, never the phase: two performances are not
phase-locked at the sample level, so a borrowed phase would fight the host
instance instead of joining it. The phase comes from the local context, as
everywhere else.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from .analyze import Analysis, Section
from .audio import EPS, _istft, _stft, match_shape

log = logging.getLogger(__name__)

REPEATABLE = {"verse", "chorus", "bridge", "prechorus"}


@dataclass
class RepeatReport:
    key: str
    label: str
    groups: int
    instances: int
    bins_borrowed: int
    pairs_rejected: int
    energy_gain_db: float

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def _bar_grid(section: Section, downbeats: np.ndarray) -> np.ndarray:
    """The downbeat times inside a section, plus its end as the closing edge."""
    inside = downbeats[(downbeats >= section.start - 1e-6) & (downbeats < section.end - 1e-6)]
    if inside.size == 0:
        return np.array([section.start, section.end])
    return np.concatenate([inside, [section.end]])


def warp_times(src_times: np.ndarray, grid_src: np.ndarray, grid_dst: np.ndarray) -> np.ndarray:
    """Map times from one section onto another, bar for bar.

    Piecewise-linear between downbeats, so a tempo difference between the
    two repeats stretches the bars instead of sliding everything out of step.
    """
    n = min(len(grid_src), len(grid_dst))
    if n < 2:
        return np.full_like(src_times, np.nan, dtype=float)
    return np.interp(src_times, grid_src[:n], grid_dst[:n], left=np.nan, right=np.nan)


# --------------------------------------------------------------------------
# Borrowing
# --------------------------------------------------------------------------


def _agreement(mag_a: np.ndarray, mag_b: np.ndarray, clean_a: np.ndarray, clean_b: np.ndarray) -> float:
    """How alike two instances are where both were heard clearly."""
    both = clean_a & clean_b
    if both.sum() < 200:
        return 0.0
    a = np.log(np.maximum(mag_a[both], EPS))
    b = np.log(np.maximum(mag_b[both], EPS))
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def borrow_from_repeats(
    stem: np.ndarray,
    parent: np.ndarray,
    analysis: Analysis,
    sr: int,
    key: str = "",
    label: str = "",
    offset: float = 0.0,
    damage_ratio: float = 0.35,
    presence_floor_db: float = -55.0,
    min_agreement: float = 0.45,
    n_fft: int = 4096,
) -> tuple[np.ndarray, RepeatReport]:
    """Fill damaged bins from another instance of the same section."""
    hop = n_fft // 4
    n = stem.shape[-1]
    stem = np.ascontiguousarray(stem, dtype=np.float32)
    parent = match_shape(np.asarray(parent, dtype=np.float32), stem)

    groups: dict[str, list[Section]] = defaultdict(list)
    for s in analysis.sections:
        if s.label in REPEATABLE:
            groups[s.label].append(s)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}

    report = RepeatReport(key, label or key, len(groups), sum(len(v) for v in groups.values()), 0, 0, 0.0)
    if not groups:
        return stem, report

    downbeats = np.asarray(analysis.downbeats, dtype=float) + offset
    S = _stft(stem, n_fft, hop)
    P = _stft(parent, n_fft, hop)
    out = S.copy()

    n_frames = S.shape[-1]
    frame_times = np.arange(n_frames) * hop / sr

    borrowed = rejected = 0

    for _label, sections in groups.items():
        shifted = [Section(s.start + offset, s.end + offset, s.label, s.index) for s in sections]
        grids = [_bar_grid(s, downbeats) for s in shifted]
        spans = []
        for s in shifted:
            a = int(np.searchsorted(frame_times, s.start))
            b = int(np.searchsorted(frame_times, s.end))
            spans.append((a, b))

        for ch in range(S.shape[0]):
            ms, mp = np.abs(S[ch]), np.abs(P[ch])
            peak = ms.max()
            floor = peak * (10.0 ** (presence_floor_db / 20.0)) if peak > EPS else EPS
            share = ms / np.maximum(mp, EPS)
            clean = (ms > floor) & (share > damage_ratio)
            damaged = (mp > floor) & (share <= damage_ratio)

            for i, (ai, bi) in enumerate(spans):
                if bi - ai < 4:
                    continue
                host_times = frame_times[ai:bi]
                host_damaged = damaged[:, ai:bi]
                if not host_damaged.any():
                    continue

                donors: list[np.ndarray] = []
                donor_clean: list[np.ndarray] = []
                donor_spec: list[np.ndarray] = []
                donor_score: list[float] = []
                for j, (aj, bj) in enumerate(spans):
                    if j == i or bj - aj < 4:
                        continue
                    mapped = warp_times(host_times, grids[i], grids[j])
                    idx = np.searchsorted(frame_times, mapped)
                    valid = np.isfinite(mapped) & (idx >= 0) & (idx < n_frames)
                    if valid.sum() < 4:
                        continue
                    idx = np.clip(idx, 0, n_frames - 1)

                    donor_mag = ms[:, idx]
                    donor_ok = clean[:, idx] & valid[None, :]

                    score = _agreement(ms[:, ai:bi], donor_mag, clean[:, ai:bi], donor_ok)
                    if score < min_agreement:
                        rejected += 1
                        continue
                    donors.append(donor_mag)
                    donor_clean.append(donor_ok)
                    donor_spec.append(S[ch][:, idx])
                    donor_score.append(score)

                if not donors:
                    continue

                stack = np.stack(donors)
                stack_ok = np.stack(donor_clean)
                have = stack_ok.any(axis=0)
                if not have.any():
                    continue
                # Median of the instances that were clean here - one odd
                # take cannot drag the answer on its own.
                filled = np.where(stack_ok, stack, np.nan)
                with np.errstate(all="ignore"):
                    estimate = np.nanmedian(np.where(have[None, ...], filled, 0.0), axis=0)
                estimate = np.nan_to_num(estimate, nan=0.0)

                target = host_damaged & have & (estimate > ms[:, ai:bi])
                if not target.any():
                    continue

                # Never claim more than the recording held in that bin.
                capped = np.minimum(estimate, mp[:, ai:bi])
                new_mag = np.where(target, capped, ms[:, ai:bi])

                # The donor's absolute phase is useless - two performances are
                # not phase-locked - and the damaged bin's own phase belongs
                # to whatever buried it. But the donor's phase *advance* per
                # frame is the partial's true instantaneous frequency, and
                # that does transfer. So: take the frequency trajectory from
                # the donor, and anchor it to the last frame this row was
                # heard clearly here. Continuing at the bin's nominal rate
                # instead drifts into noise within about a second.
                seg = S[ch, :, ai:bi]
                phase = np.angle(seg)
                clean_seg = clean[:, ai:bi]
                width = bi - ai
                rows = np.arange(seg.shape[0])[:, None]

                best = donor_spec[int(np.argmax(donor_score))]
                advance = np.angle(best[:, 1:] * np.conj(best[:, :-1]))
                advance = np.concatenate([np.zeros((seg.shape[0], 1)), advance], axis=1)
                cumulative = np.cumsum(advance, axis=1)

                order = np.where(clean_seg, np.arange(width)[None, :], -1)
                last = np.maximum.accumulate(order, axis=1)
                usable = last >= 0
                safe_last = np.maximum(last, 0)
                continued = phase[rows, safe_last] + (cumulative - cumulative[rows, safe_last])
                new_phase = np.where(target & usable, continued, phase)

                out[ch, :, ai:bi] = new_mag * np.exp(1j * new_phase)
                borrowed += int((target & usable).sum())

    rebuilt = _istft(out, hop, n, n_fft).astype(np.float32)
    before = float(np.mean(stem**2))
    after = float(np.mean(rebuilt**2))
    report.bins_borrowed = borrowed
    report.pairs_rejected = rejected
    report.energy_gain_db = round(10.0 * np.log10(after / before), 2) if before > EPS and after > EPS else 0.0
    return rebuilt, report
