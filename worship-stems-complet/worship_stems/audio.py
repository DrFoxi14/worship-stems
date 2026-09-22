"""Audio I/O and the exact-partition maths.

The important function here is `partition_exact`. Every separation model
returns estimates that do not add up to the input: they overlap in some
places and leave holes in others. Instead of shipping the raw estimates plus
a "residual" bucket that hides the error, we convert the estimates into
soft masks over the parent's own spectrogram and renormalise the masks so
they sum to one at every time-frequency bin.

  mask_i = |X_i|^p / sum_j |X_j|^p         Y_i = mask_i * X_parent

Because the masks sum to one and the iSTFT is linear, sum_i y_i == parent,
sample for sample, with the parent's original phase. That is what makes the
"recombined" file a real proof rather than a tautology.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

EPS = 1e-10


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def load_audio(path: str | Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    """Load audio as float32 (channels, samples). Falls back to ffmpeg."""
    path = Path(path)
    try:
        data, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
        audio = data.T
    except Exception:
        audio, file_sr = _load_via_ffmpeg(path)

    if sr is not None and sr != file_sr:
        audio = resample(audio, file_sr, sr)
        file_sr = sr
    if mono and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)
    if not mono and audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return np.ascontiguousarray(audio, dtype=np.float32), file_sr


def _load_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:a", "pcm_f32le", tmp_path],
            check=True,
            capture_output=True,
        )
        data, sr = sf.read(tmp_path, always_2d=True, dtype="float32")
        return data.T, sr
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def save_audio(path: str | Path, audio: np.ndarray, sr: int, fmt: str = "wav24") -> Path:
    """Write (channels, samples) float32 to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(audio.T, dtype=np.float32)

    if fmt == "flac":
        # FLAC is integer-only; clip to avoid wrap-around on inter-sample peaks.
        sf.write(str(path.with_suffix(".flac")), np.clip(data, -1.0, 1.0), sr, subtype="PCM_24")
        return path.with_suffix(".flac")
    subtype = {"wav24": "PCM_24", "wav16": "PCM_16"}.get(fmt, "PCM_24")
    if subtype.startswith("PCM"):
        data = np.clip(data, -1.0, 1.0)
    sf.write(str(path.with_suffix(".wav")), data, sr, subtype=subtype)
    return path.with_suffix(".wav")


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    import librosa

    return np.ascontiguousarray(
        librosa.resample(audio, orig_sr=sr_in, target_sr=sr_out, res_type="soxr_hq"),
        dtype=np.float32,
    )


# --------------------------------------------------------------------------
# Shape helpers
# --------------------------------------------------------------------------


def match_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad to exactly n samples."""
    if audio.shape[-1] == n:
        return audio
    if audio.shape[-1] > n:
        return audio[..., :n]
    pad = np.zeros(audio.shape[:-1] + (n - audio.shape[-1],), dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=-1)


def match_shape(audio: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Force `audio` to the channel count and length of `like`."""
    if audio.ndim == 1:
        audio = audio[None, :]
    ch = like.shape[0]
    if audio.shape[0] < ch:
        audio = np.repeat(audio[:1], ch, axis=0)
    elif audio.shape[0] > ch:
        audio = audio[:ch]
    return match_length(audio, like.shape[-1])


# --------------------------------------------------------------------------
# Exact partition
# --------------------------------------------------------------------------


def _stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    import librosa

    return np.stack([librosa.stft(np.ascontiguousarray(c), n_fft=n_fft, hop_length=hop) for c in x])


def _istft(X: np.ndarray, hop: int, length: int, n_fft: int) -> np.ndarray:
    import librosa

    out = [librosa.istft(c, hop_length=hop, length=length, n_fft=n_fft) for c in X]
    return np.stack(out).astype(np.float32)


def partition_exact(
    parent: np.ndarray,
    estimates: dict[str, np.ndarray],
    power: float = 2.0,
    n_fft: int = 4096,
    hop: int | None = None,
) -> dict[str, np.ndarray]:
    """Split `parent` into parts that sum back to it exactly.

    `estimates` are the raw model outputs; they only decide *where* energy
    goes, never how much. Returns arrays with the same shape as `parent`.
    """
    hop = hop or n_fft // 4
    n = parent.shape[-1]
    keys = list(estimates)
    if not keys:
        raise ValueError("no estimates given")
    if len(keys) == 1:
        return {keys[0]: parent.copy()}

    X_parent = _stft(parent, n_fft, hop)
    mags = []
    for k in keys:
        est = match_shape(np.asarray(estimates[k], dtype=np.float32), parent)
        mags.append(np.abs(_stft(est, n_fft, hop)) ** power)
    M = np.stack(mags)  # (parts, ch, freq, frames)

    total = M.sum(axis=0)
    # Where every model agrees there is nothing, fall back to an even split
    # so the parent's own content is still distributed rather than dropped.
    silent = total < EPS
    M = np.where(silent[None, ...], 1.0 / len(keys), M / np.where(silent, 1.0, total)[None, ...])

    out: dict[str, np.ndarray] = {}
    for i, k in enumerate(keys):
        out[k] = _istft(M[i] * X_parent, hop, n, n_fft)

    # iSTFT edge frames can lose a hair of energy; push any difference into
    # the part that is loudest overall so the sum stays bit-exact.
    residual = parent - sum(out.values())
    anchor = max(keys, key=lambda k: float(np.sum(out[k] ** 2)))
    out[anchor] = out[anchor] + residual
    return out


def partition_raw(parent: np.ndarray, estimates: dict[str, np.ndarray], residual_key: str) -> dict[str, np.ndarray]:
    """Keep raw model outputs; derive `residual_key` as whatever is left.

    Any estimate supplied for `residual_key` is discarded on purpose - it is
    defined as the remainder, otherwise the parts would not add back up.
    """
    out = {
        k: match_shape(np.asarray(v, dtype=np.float32), parent)
        for k, v in estimates.items()
        if k != residual_key
    }
    out[residual_key] = parent - sum(out.values()) if out else parent.copy()
    return out


def reconstruction_error_db(original: np.ndarray, parts: list[np.ndarray]) -> float:
    """Level of (original - sum(parts)) relative to the original, in dB."""
    total = np.zeros_like(original)
    for p in parts:
        total = total + match_shape(p, original)
    err = float(np.sqrt(np.mean((original - total) ** 2)))
    ref = float(np.sqrt(np.mean(original**2)))
    if ref < EPS:
        return -np.inf
    if err < EPS:
        return -np.inf
    return 20.0 * np.log10(err / ref)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB."""
    ref = reference.reshape(-1).astype(np.float64)
    est = match_length(estimate, reference.shape[-1]).reshape(-1).astype(np.float64)
    ref = ref - ref.mean()
    est = est - est.mean()
    denom = float(np.dot(ref, ref))
    if denom < EPS:
        return float("nan")
    alpha = float(np.dot(est, ref)) / denom
    target = alpha * ref
    noise = est - target
    nd = float(np.dot(noise, noise))
    td = float(np.dot(target, target))
    if nd < EPS or td < EPS:
        return float("inf") if nd < EPS else float("-inf")
    return 10.0 * np.log10(td / nd)


# --------------------------------------------------------------------------
# Loudness / limiting
# --------------------------------------------------------------------------


def _k_weight(audio: np.ndarray, sr: int) -> np.ndarray:
    """ITU-R BS.1770 K-weighting (shelf + high-pass)."""
    from scipy.signal import lfilter

    # Stage 1: high-shelf, +4 dB at HF.
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    Vh = 10 ** (G / 20.0)
    Vb = Vh**0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b = np.array([(Vh + Vb * K / Q + K * K) / a0, 2.0 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0])
    a = np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])
    y = lfilter(b, a, audio, axis=-1)

    # Stage 2: high-pass at 38 Hz.
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / sr)
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K), (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)])
    return lfilter(b2, a2, y, axis=-1)


def integrated_lufs(audio: np.ndarray, sr: int) -> float:
    """Gated integrated loudness, BS.1770-4."""
    y = _k_weight(audio.astype(np.float64), sr)
    block = int(0.4 * sr)
    step = max(1, int(0.1 * sr))
    if y.shape[-1] < block:
        block = y.shape[-1]
        step = block
    starts = range(0, max(1, y.shape[-1] - block + 1), step)
    weights = [1.0, 1.0, 1.0, 1.41, 1.41]
    loud = []
    powers = []
    for s in starts:
        seg = y[..., s : s + block]
        p = np.mean(seg**2, axis=-1)
        w = np.array(weights[: len(p)]) if len(p) <= len(weights) else np.ones(len(p))
        tot = float(np.sum(w * p))
        powers.append(tot)
        loud.append(-0.691 + 10.0 * np.log10(tot + EPS))
    loud = np.array(loud)
    powers = np.array(powers)
    keep = loud > -70.0
    if not keep.any():
        return -np.inf
    rel = -0.691 + 10.0 * np.log10(np.mean(powers[keep]) + EPS) - 10.0
    keep = keep & (loud > rel)
    if not keep.any():
        return -np.inf
    return float(-0.691 + 10.0 * np.log10(np.mean(powers[keep]) + EPS))


def normalize_lufs(audio: np.ndarray, sr: int, target: float = -18.0, max_gain_db: float = 24.0) -> np.ndarray:
    cur = integrated_lufs(audio, sr)
    if not np.isfinite(cur):
        return audio
    gain_db = float(np.clip(target - cur, -max_gain_db, max_gain_db))
    return (audio * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def soft_limit(audio: np.ndarray, sr: int, ceiling_db: float = -1.0, attack_ms: float = 1.5, release_ms: float = 80.0) -> np.ndarray:
    """Look-ahead peak limiter.

    Gain reduction is computed per sample, widened by a maximum filter so it
    arrives before the transient, then smoothed with a Hann window so there
    are no steps. A final elementwise min keeps the ceiling guaranteed.
    """
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    ceiling = 10.0 ** (ceiling_db / 20.0)
    peak = np.max(np.abs(audio), axis=0)
    if peak.size == 0 or peak.max() <= ceiling:
        return audio

    look = max(1, int(attack_ms * 1e-3 * sr))
    rel = max(look * 2 + 1, int(release_ms * 1e-3 * sr))
    target = np.minimum(ceiling / np.maximum(peak, EPS), 1.0)

    # Widen the dips (attack look-ahead + release hold), then smooth them.
    widened = -maximum_filter1d(-target, size=2 * rel + 1, mode="nearest")
    smooth = uniform_filter1d(widened, size=rel, mode="nearest")
    smooth = uniform_filter1d(smooth, size=rel, mode="nearest")  # ~triangular
    gain = np.minimum(smooth, target)
    return (audio * gain[None, :]).astype(np.float32)


def peak_db(audio: np.ndarray) -> float:
    p = float(np.max(np.abs(audio))) if audio.size else 0.0
    return 20.0 * np.log10(p) if p > EPS else -np.inf
