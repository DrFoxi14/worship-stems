"""Will this song come out well? Ask before spending fifteen minutes on it.

Separation quality is mostly decided by the recording, not by the settings.
A sparse arrangement with room to breathe comes apart cleanly; a dense,
heavily limited mix with everything in the same octave does not, and no
amount of waiting changes that. Since the whole pipeline takes a quarter of
an hour at maximum quality, it is worth saying so in five seconds rather
than fifteen minutes.

The measurements are deliberately simple and each one has a reason you can
argue with:

  density      how much of the time-frequency plane is occupied. When
               everything is playing at once there are no free bins to give
               anyone, so every stem is a compromise.
  crest        peaks against average level. A crushed master has had its
               transients flattened, and transients are most of what tells a
               snare from a bass note.
  width        stereo information. A wide mix hands the models a second,
               independent view; a mono-ish one gives them one.
  register     how much energy is piled into the same octaves. Instruments
               sharing a register share bins.
  brightness   the top end. Cymbals and air help separate percussion; a dull
               transfer takes that away.

This is a heuristic, not a trained predictor. It is calibrated on reasoning
about what makes separation hard, so treat it as a weather forecast: useful
for deciding whether to bother, not a promise.

One known quirk: pure synthetic tones have a naturally low crest factor and
get flagged as "compressed" even when they are not. Real recordings have
transients, so this does not come up in use - but it does mean a test signal
made of sine waves will score harder than it deserves.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

from .audio import EPS

log = logging.getLogger(__name__)


@dataclass
class Difficulty:
    score: float  # 0 = easy, 1 = hopeless
    verdict: str  # "easy" | "fair" | "hard" | "very hard"
    summary: str
    reasons: list[str] = field(default_factory=list)
    measures: dict = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile_ratio(x: np.ndarray, high: float = 95.0, low: float = 50.0) -> float:
    a, b = np.percentile(x, high), np.percentile(x, low)
    return float(a / b) if b > EPS else 0.0


def assess(audio: np.ndarray, sr: int, max_seconds: float = 60.0) -> Difficulty:
    """Estimate how hard this mix will be to take apart."""
    import time

    import librosa

    started = time.time()

    # A minute from the middle is enough and keeps this quick: the busiest
    # part of a worship song is almost always the middle.
    n = audio.shape[-1]
    want = int(max_seconds * sr)
    if n > want:
        start = (n - want) // 2
        audio = audio[:, start : start + want]

    mono = audio.mean(axis=0) if audio.ndim > 1 else audio
    small_sr = 22050
    if sr != small_sr:
        from .audio import resample

        mono = resample(mono[None, :], sr, small_sr)[0]
        work_sr = small_sr
    else:
        work_sr = sr

    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
    power = S**2
    total = power.sum()
    if total < EPS:
        return Difficulty(0.0, "easy", "Silence.", ["nothing to separate"], {}, 0.0)

    # --- density ---------------------------------------------------------
    # The share of bins holding real energy. Dense mixes leave nothing spare.
    peak = S.max()
    occupied = float(np.mean(S > peak * 0.02))

    # --- crest -----------------------------------------------------------
    crest = _percentile_ratio(np.abs(mono), 99.5, 50.0)
    crest_db = 20.0 * np.log10(max(crest, EPS))

    # --- stereo width ----------------------------------------------------
    if audio.ndim > 1 and audio.shape[0] > 1:
        mid = (audio[0] + audio[1]) / 2.0
        side = (audio[0] - audio[1]) / 2.0
        width = float(np.sqrt(np.mean(side**2)) / (np.sqrt(np.mean(mid**2)) + EPS))
    else:
        width = 0.0

    # --- register crowding -----------------------------------------------
    freqs = librosa.fft_frequencies(sr=work_sr, n_fft=2048)
    bands = [(0, 120), (120, 400), (400, 1200), (1200, 3500), (3500, work_sr / 2)]
    shares = []
    for lo, hi in bands:
        sel = (freqs >= lo) & (freqs < hi)
        shares.append(float(power[sel].sum() / total) if sel.any() else 0.0)
    shares = np.array(shares)
    # One band holding most of the energy means everything is stacked there.
    crowding = float(shares.max())

    # --- brightness ------------------------------------------------------
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=work_sr)))

    # --- harmonic vs percussive -----------------------------------------
    harmonic, percussive = librosa.decompose.hpss(S)
    h_energy = float((harmonic**2).sum())
    p_energy = float((percussive**2).sum())
    percussive_share = p_energy / (h_energy + p_energy + EPS)

    measures = {
        "occupied_bins": round(occupied, 3),
        "crest_db": round(crest_db, 1),
        "stereo_width": round(width, 3),
        "register_crowding": round(crowding, 3),
        "centroid_hz": round(centroid, 1),
        "percussive_share": round(percussive_share, 3),
    }

    # --- scoring ---------------------------------------------------------
    # Each term is 0 (easy) to 1 (hard).
    s_density = float(np.clip((occupied - 0.10) / 0.30, 0, 1))
    s_crest = float(np.clip((14.0 - crest_db) / 9.0, 0, 1))
    s_width = float(np.clip((0.25 - width) / 0.25, 0, 1))
    s_crowd = float(np.clip((crowding - 0.40) / 0.35, 0, 1))
    s_dull = float(np.clip((1400.0 - centroid) / 900.0, 0, 1))

    weights = {"density": 0.30, "crest": 0.25, "crowding": 0.20, "width": 0.15, "dull": 0.10}
    score = (
        weights["density"] * s_density
        + weights["crest"] * s_crest
        + weights["crowding"] * s_crowd
        + weights["width"] * s_width
        + weights["dull"] * s_dull
    )

    reasons: list[str] = []
    if s_density > 0.5:
        reasons.append("the arrangement is dense - most of the spectrum is busy most of the time")
    if s_crest > 0.5:
        reasons.append(f"heavily compressed ({crest_db:.0f} dB crest) - the transients are flattened")
    if s_crowd > 0.5:
        reasons.append("the instruments are piled into the same register")
    if s_width > 0.5:
        reasons.append("almost mono - there is no stereo information to help")
    if s_dull > 0.5:
        reasons.append("dull top end - little air to separate the cymbals with")
    if not reasons:
        reasons.append("open arrangement with room between the parts")

    if score < 0.28:
        verdict, summary = "easy", "This should come out well."
    elif score < 0.48:
        verdict, summary = "fair", "This should be usable, with some bleed."
    elif score < 0.68:
        verdict, summary = "hard", "Expect audible artefacts; listen before Sunday."
    else:
        verdict, summary = "very hard", "This one will be rough whatever settings you use."

    return Difficulty(
        score=round(float(score), 3),
        verdict=verdict,
        summary=summary,
        reasons=reasons,
        measures=measures,
        seconds=round(time.time() - started, 2),
    )


def advice(d: Difficulty) -> list[str]:
    """What to change, given what the mix is like."""
    out: list[str] = []
    m = d.measures
    if m.get("crest_db", 99) < 10:
        out.append("If a less-compressed version of this song exists, use it - it will separate better than any setting.")
    if m.get("stereo_width", 1) < 0.12:
        out.append("Almost mono, so the stereo-position split will refuse; that is expected, not a fault.")
    if d.verdict in {"hard", "very hard"}:
        out.append("Run it on Fast first and listen. If it is rough there, Maximum will be rough too, just slower.")
        out.append("Consider keeping drums as one track rather than splitting the kit.")
    return out
