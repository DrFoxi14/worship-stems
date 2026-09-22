"""Does this instrument actually exist in the song?

A separator can never answer "no". Ask it for piano and it returns a piano
file, even when there is no piano within a hundred kilometres - it gathers
whatever frequencies look piano-ish and hands them over with total
confidence. Exporting that file is a lie, and the band will hear it.

So the separator is not allowed to decide what exists. This module decides,
by looking for the things a human ear actually uses:

  presence     is there enough energy here to matter at all?
  onsets       real instruments start notes; a ghost stem is a smear
  harmonicity  real pitched instruments stack harmonics over a noise floor
  range        a bass line does not live at 4 kHz
  distinctness a ghost is mostly a scaled copy of its neighbours

A stem that fails is not deleted - it is folded back into the residual, so
the partition stays exact and the audio is not lost, it just stops
pretending to be an instrument it is not.

An optional CLAP backend adds a zero-shot opinion from a model trained on
audio-text pairs, which is closer still to how a person would label a sound.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

log = logging.getLogger(__name__)

EPS = 1e-10


@dataclass
class InstrumentProfile:
    """What this instrument should look like if it is really there."""

    key: str
    label: str
    f_low: float  # plausible fundamental range
    f_high: float
    energy_low: float  # min band energy, dB relative to the parent
    percussive: bool = False
    min_onsets_per_min: float = 8.0
    min_harmonicity: float = 0.12


PROFILES: dict[str, InstrumentProfile] = {
    "lead_vocal": InstrumentProfile("lead_vocal", "Lead Vocal", 80, 1200, -30, False, 15, 0.16),
    "bgv": InstrumentProfile("bgv", "Background Vocals", 100, 1400, -34, False, 4, 0.10),
    "bass": InstrumentProfile("bass", "Bass", 30, 400, -30, False, 8, 0.18),
    # Onset expectations stay low for guitars and keys: worship guitar is
    # often swells and volume-pedal ambience with barely an attack in it, and
    # a held organ chord has none at all. Demanding attacks here would delete
    # exactly the parts that make these arrangements what they are.
    "electric_guitar": InstrumentProfile("electric_guitar", "Guitars", 80, 1400, -32, False, 5, 0.12),
    "keys": InstrumentProfile("keys", "Piano / Keys", 55, 2200, -32, False, 4, 0.12),
    "pads": InstrumentProfile("pads", "Pads / Synth / Strings", 100, 6000, -36, False, 0.0, 0.04),
    "drums": InstrumentProfile("drums", "Drums", 40, 12000, -30, True, 30, 0.0),
    "kick": InstrumentProfile("kick", "Kick", 35, 160, -34, True, 20, 0.0),
    "snare": InstrumentProfile("snare", "Snare", 120, 8000, -34, True, 15, 0.0),
    "toms": InstrumentProfile("toms", "Toms", 60, 500, -40, True, 2, 0.0),
    "hihat": InstrumentProfile("hihat", "Hi-Hat", 3000, 16000, -40, True, 25, 0.0),
    "cymbals": InstrumentProfile("cymbals", "Cymbals / OH", 2000, 16000, -40, True, 4, 0.0),
}

# Stems that are structural, not claims about instruments - never gated.
NEVER_GATE = {"mix", "vocals", "instrumental", "recombined", "click", "guide", "click_guide", "pad_key", "minus_one", "acapella"}


@dataclass
class Verdict:
    key: str
    label: str
    confidence: float
    decision: str  # "present" | "uncertain" | "absent"
    reasons: list[str]
    evidence: dict

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) if x.ndim > 1 else x


def relative_energy_db(stem: np.ndarray, parent: np.ndarray) -> float:
    a = float(np.sqrt(np.mean(stem**2)))
    b = float(np.sqrt(np.mean(parent**2)))
    if b < EPS:
        return -np.inf
    if a < EPS:
        return -np.inf
    return 20.0 * np.log10(a / b)


def band_occupancy(stem: np.ndarray, sr: int, f_low: float, f_high: float) -> float:
    """Fraction of the stem's energy that sits in its plausible range."""
    import librosa

    y = _mono(stem)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = float(S.sum())
    if total < EPS:
        return 0.0
    # Allow harmonics well above the fundamental range.
    in_band = (freqs >= f_low * 0.8) & (freqs <= min(f_high * 6.0, sr / 2))
    return float(S[in_band].sum() / total)


def onset_rate(stem: np.ndarray, sr: int) -> float:
    """Note starts per minute. A ghost stem has almost none."""
    import librosa

    y = _mono(stem)
    if np.sqrt(np.mean(y**2)) < 1e-6:
        return 0.0
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    minutes = max(len(y) / sr / 60.0, 1e-6)
    return float(len(onsets) / minutes)


def harmonicity(stem: np.ndarray, sr: int) -> float:
    """How much the spectrum looks like stacked partials rather than mush.

    Ratio of harmonic to total energy via librosa's HPSS, tempered by
    spectral flatness - noise-like content scores near zero.
    """
    import librosa

    y = _mono(stem)
    if np.sqrt(np.mean(y**2)) < 1e-6:
        return 0.0
    h = librosa.effects.harmonic(y, margin=2.0)
    h_energy = float(np.mean(h**2))
    t_energy = float(np.mean(y**2)) + EPS
    ratio = h_energy / t_energy
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    return float(np.clip(ratio * (1.0 - flat), 0.0, 1.0))


def distinctness(stem: np.ndarray, siblings: list[np.ndarray]) -> float:
    """1.0 = its own source; near 0 = a scaled shadow of its neighbours."""
    y = _mono(stem)
    ny = float(np.linalg.norm(y))
    if ny < EPS or not siblings:
        return 0.0
    worst = 0.0
    for sib in siblings:
        s = _mono(sib)
        ns = float(np.linalg.norm(s))
        if ns < EPS:
            continue
        # Correlate envelopes, not waveforms: masked stems share phase by
        # construction, so raw correlation would always look high.
        n = min(len(y), len(s))
        win = max(1, int(len(y) / 2000))
        ey = np.abs(y[:n]).reshape(-1, win).mean(axis=1) if n >= win else np.abs(y[:n])
        es = np.abs(s[:n]).reshape(-1, win).mean(axis=1) if n >= win else np.abs(s[:n])
        m = min(len(ey), len(es))
        ey, es = ey[:m] - ey[:m].mean(), es[:m] - es[:m].mean()
        d = float(np.linalg.norm(ey) * np.linalg.norm(es))
        if d < EPS:
            continue
        worst = max(worst, abs(float(np.dot(ey, es)) / d))
    return float(np.clip(1.0 - worst, 0.0, 1.0))


# --------------------------------------------------------------------------
# Judgement
# --------------------------------------------------------------------------


def judge_stem(
    key: str,
    stem: np.ndarray,
    parent: np.ndarray,
    siblings: list[np.ndarray],
    sr: int,
    strictness: float = 1.0,
) -> Verdict:
    prof = PROFILES.get(key)
    label = prof.label if prof else key
    if prof is None:
        return Verdict(key, label, 1.0, "present", ["no profile, not gated"], {})

    rel = relative_energy_db(stem, parent)
    if not np.isfinite(rel):
        return Verdict(key, label, 0.0, "absent", ["silent"], {"relative_db": None})

    occ = band_occupancy(stem, sr, prof.f_low, prof.f_high)
    rate = onset_rate(stem, sr)
    harm = harmonicity(stem, sr) if not prof.percussive else 0.0
    dist = distinctness(stem, siblings)

    evidence = {
        "relative_db": round(rel, 1),
        "band_occupancy": round(occ, 3),
        "onsets_per_min": round(rate, 1),
        "harmonicity": round(harm, 3),
        "distinctness": round(dist, 3),
    }

    # Each score is 0..1; they are deliberately independent kinds of evidence.
    s_energy = float(np.clip((rel - prof.energy_low) / 12.0, 0.0, 1.0))
    s_band = float(np.clip((occ - 0.45) / 0.35, 0.0, 1.0))
    s_onset = float(np.clip(rate / max(prof.min_onsets_per_min, 1e-6), 0.0, 1.0)) if prof.min_onsets_per_min > 0 else 1.0
    s_harm = float(np.clip(harm / max(prof.min_harmonicity, 1e-6), 0.0, 1.0))
    s_dist = float(np.clip(dist / 0.35, 0.0, 1.0))

    # Distinctness carries the most weight because it is the one test a ghost
    # cannot pass: a stem the model assembled out of its neighbours moves
    # exactly when they move. Everything else - energy, attacks, a plausible
    # range - a ghost inherits for free from the instruments it was scraped
    # from, which is precisely why the other tests alone let toms through.
    if prof.percussive:
        # Percussion has no harmonics to test, so that weight goes here too.
        weights = {"energy": 0.20, "band": 0.15, "onset": 0.15, "harm": 0.0, "dist": 0.50}
    else:
        weights = {"energy": 0.20, "band": 0.15, "onset": 0.15, "harm": 0.15, "dist": 0.35}

    confidence = (
        weights["energy"] * s_energy
        + weights["band"] * s_band
        + weights["onset"] * s_onset
        + weights["harm"] * s_harm
        + weights["dist"] * s_dist
    )

    # Veto 1: if it moves with its neighbours it is not a separate source, no
    # matter how healthy the rest of the evidence looks.
    vetoed = dist < 0.15
    if vetoed:
        confidence = min(confidence, 0.30)

    # Veto 2: audibility. A track sitting 100 dB under its parent is silence,
    # and silence has a perfect onset rate and a perfect frequency range - a
    # nulled ghost scores beautifully on every other test precisely because
    # there is nothing left in it to fail them.
    audible = float(np.clip((rel - (prof.energy_low - 10.0)) / 8.0, 0.0, 1.0))
    if audible < 1.0:
        confidence = min(confidence, audible)

    reasons: list[str] = []
    if audible <= 0.0:
        reasons.append(f"effectively silent ({rel:.0f} dB below the rest)")
    elif s_energy < 0.35:
        reasons.append(f"very quiet ({rel:.0f} dB below the rest)")
    if s_band < 0.35:
        reasons.append(f"only {occ*100:.0f}% of its energy is in the range this instrument lives in")
    if s_onset < 0.35 and prof.min_onsets_per_min > 0:
        reasons.append(f"almost no note attacks ({rate:.0f}/min)")
    if s_harm < 0.35 and not prof.percussive:
        reasons.append("noise-like rather than harmonic")
    if vetoed:
        reasons.insert(0, "it moves exactly with the other tracks - assembled from them, not a source of its own")
    elif s_dist < 0.35:
        reasons.append("mostly a copy of the neighbouring stems")
    if not reasons:
        reasons.append("energy, note attacks and harmonic structure all consistent")

    hi, lo = 0.62 * strictness, 0.34 * strictness
    decision = "present" if confidence >= hi else ("absent" if confidence < lo else "uncertain")
    return Verdict(key, label, round(float(confidence), 3), decision, reasons, evidence)


def take_inventory(
    stems: dict[str, np.ndarray],
    parents: dict[str, str],
    sr: int,
    strictness: float = 1.0,
) -> dict[str, Verdict]:
    """Judge every gateable stem. `parents` maps stem key -> parent key."""
    verdicts: dict[str, Verdict] = {}
    for key, audio in stems.items():
        if key in NEVER_GATE or key not in PROFILES:
            continue
        parent_key = parents.get(key)
        parent = stems.get(parent_key) if parent_key else None
        if parent is None:
            parent = stems.get("mix")
        if parent is None:
            continue
        siblings = [
            v for k, v in stems.items()
            if k != key and parents.get(k) == parent_key and k in PROFILES
        ]
        try:
            verdicts[key] = judge_stem(key, audio, parent, siblings, sr, strictness)
        except Exception as exc:  # noqa: BLE001
            log.warning("inventory failed for %s: %s", key, exc)
    return verdicts


# --------------------------------------------------------------------------
# Optional: a zero-shot opinion from CLAP
# --------------------------------------------------------------------------

CLAP_PROMPTS = {
    "electric_guitar": ["an electric guitar playing", "a distorted electric guitar"],
    "keys": ["a piano playing", "an electric piano or synth keyboard"],
    "bass": ["a bass guitar playing", "a deep electric bass line"],
    "pads": ["a synthesiser pad", "sustained string section"],
    "lead_vocal": ["a person singing"],
    "bgv": ["a choir singing", "backing vocal harmonies"],
    "drums": ["a drum kit playing"],
}


def clap_available() -> bool:
    try:
        import laion_clap  # noqa: F401

        return True
    except Exception:
        return False


def clap_opinion(mix: np.ndarray, sr: int, keys: list[str]) -> dict[str, float]:
    """Zero-shot instrument likelihood from the original mix, 0..1 per key.

    This looks at the *mix*, never at a separated stem, so it cannot be
    fooled by a stem the separator invented.
    """
    try:
        import laion_clap
        import torch

        from .audio import resample

        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()

        y = resample(mix.mean(axis=0, keepdims=True), sr, 48000)[0]
        chunks = [y[i : i + 48000 * 10] for i in range(0, len(y), 48000 * 10)]
        chunks = [c for c in chunks if len(c) > 48000]
        if not chunks:
            chunks = [y]

        prompts, owners = [], []
        for k in keys:
            for p in CLAP_PROMPTS.get(k, []):
                prompts.append(p)
                owners.append(k)
        if not prompts:
            return {}

        text_emb = model.get_text_embedding(prompts, use_tensor=True)
        scores: dict[str, list[float]] = {k: [] for k in keys}
        for c in chunks:
            a = model.get_audio_embedding_from_data(x=c[None, :], use_tensor=False)
            a = torch.tensor(a)
            sim = (a @ text_emb.T).squeeze(0)
            sim = torch.softmax(sim * 10.0, dim=-1)
            for owner, val in zip(owners, sim.tolist()):
                scores[owner].append(val)
        return {k: float(np.max(v)) if v else 0.0 for k, v in scores.items()}
    except Exception as exc:  # noqa: BLE001
        log.warning("CLAP opinion unavailable: %s", exc)
        return {}
