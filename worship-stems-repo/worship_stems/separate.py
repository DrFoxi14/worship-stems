"""Separation engine: a thin, cached wrapper over `audio-separator`.

Everything model-specific lives here. The rest of the app only ever sees
numpy arrays, so the pipeline can be tested without any model weights.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .audio import load_audio, save_audio

log = logging.getLogger(__name__)

_STEM_RE = re.compile(r"\(([^)]+)\)")

# Map whatever a model calls a stem onto our own keys.
STEM_ALIASES = {
    "vocals": "vocals",
    "vocal": "vocals",
    "lead vocals": "lead_vocal",
    "lead_vocals": "lead_vocal",
    "instrumental": "instrumental",
    "inst": "instrumental",
    "no vocals": "instrumental",
    "back vocals": "bgv",
    "backing vocals": "bgv",
    "bgv": "bgv",
    "drums": "drums",
    "bass": "bass",
    "other": "other",
    "guitar": "electric_guitar",
    "piano": "keys",
    "kick": "kick",
    "snare": "snare",
    "toms": "toms",
    "tom": "toms",
    "hh": "hihat",
    "hihat": "hihat",
    "hi-hat": "hihat",
    "ride": "cymbals",
    "crash": "cymbals",
    "cymbals": "cymbals",
}


def normalise_stem_name(raw: str) -> str:
    key = raw.strip().lower().replace("-", " ").replace("_", " ")
    key = re.sub(r"\s+", " ", key)
    return STEM_ALIASES.get(key, STEM_ALIASES.get(key.replace(" ", "_"), key.replace(" ", "_")))


class SeparationEngine:
    """Loads models on demand and keeps them warm between calls."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device_hint: str = "auto",
        segment_size: int = 256,
        overlap: float = 0.25,
        shifts: int = 2,
        log_level: int = logging.WARNING,
    ):
        self.model_dir = str(model_dir or default_model_dir())
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self.device_hint = device_hint
        self.segment_size = segment_size
        self.overlap = overlap
        self.shifts = shifts
        self.log_level = log_level
        self._cache: dict[str, object] = {}

    # -- separator construction -------------------------------------------

    def _make_separator(self, out_dir: str, ensemble_preset: str | None = None):
        from audio_separator.separator import Separator

        kwargs = dict(
            log_level=self.log_level,
            model_file_dir=self.model_dir,
            output_dir=out_dir,
            output_format="WAV",
            use_autocast=False,
            mdxc_params={
                "segment_size": self.segment_size,
                "override_model_segment_size": False,
                "batch_size": 1,
                "overlap": None,
                "pitch_shift": 0,
            },
            demucs_params={
                "segment_size": "Default",
                "shifts": self.shifts,
                "overlap": self.overlap,
                "segments_enabled": True,
            },
        )
        if ensemble_preset:
            kwargs["ensemble_preset"] = ensemble_preset
        return Separator(**kwargs)

    def _get(self, key: str, ensemble_preset: str | None, model: str | None):
        sep = self._cache.get(key)
        if sep is None:
            sep = self._make_separator(tempfile.mkdtemp(prefix="ws-sep-"), ensemble_preset)
            if ensemble_preset:
                sep.load_model()
            else:
                sep.load_model(model)
            self._cache[key] = sep
        return sep

    def release(self) -> None:
        """Drop every loaded model and free accelerator memory."""
        self._cache.clear()
        try:
            import gc

            import torch

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- the one call that matters ----------------------------------------

    def run(
        self,
        audio: np.ndarray,
        sr: int,
        model: str | None = None,
        ensemble_preset: str | None = None,
        wanted: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Separate an in-memory signal and return {stem_key: audio}."""
        if model is None and ensemble_preset is None:
            raise ValueError("need a model or an ensemble preset")
        key = f"ens:{ensemble_preset}" if ensemble_preset else f"mod:{model}"
        sep = self._get(key, ensemble_preset, model)

        work = Path(tempfile.mkdtemp(prefix="ws-run-"))
        try:
            in_path = work / "input.wav"
            save_audio(in_path, audio, sr, "wav24")

            out_dir = work / "out"
            out_dir.mkdir()
            sep.output_dir = str(out_dir)
            if getattr(sep, "model_instance", None) is not None:
                sep.model_instance.output_dir = str(out_dir)

            files = sep.separate(str(in_path))
            results: dict[str, np.ndarray] = {}
            for f in files:
                p = Path(f)
                if not p.is_absolute():
                    p = out_dir / p
                if not p.exists():
                    continue
                m = _STEM_RE.search(p.stem)
                if not m:
                    continue
                stem = normalise_stem_name(m.group(1))
                if wanted and stem not in wanted:
                    continue
                data, file_sr = load_audio(p, sr=sr)
                results[stem] = data
            if not results:
                raise RuntimeError(f"separation produced no recognisable stems (files: {files})")
            return results
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # -- availability ------------------------------------------------------

    @staticmethod
    def available() -> bool:
        return SeparationEngine.import_error() is None

    @staticmethod
    def import_error() -> str | None:
        """Import the real class, not just the package.

        `import audio_separator` succeeds even when a transitive dependency
        such as audioread or pydub is missing; the failure then only shows up
        once separation starts, after the user has already waited.
        """
        try:
            from audio_separator.separator import Separator  # noqa: F401

            return None
        except Exception as exc:  # noqa: BLE001
            missing = getattr(exc, "name", None)
            hint = f" (missing module: {missing})" if missing else ""
            return f"{type(exc).__name__}: {exc}{hint}"

    @staticmethod
    def describe_device() -> str:
        try:
            import torch

            if torch.backends.mps.is_available():
                return "Apple Silicon GPU (MPS)"
            if torch.cuda.is_available():
                return f"CUDA ({torch.cuda.get_device_name(0)})"
            return "CPU"
        except Exception:
            return "unknown"


def default_model_dir() -> Path:
    env = os.environ.get("WORSHIP_STEMS_MODEL_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "worship-stems" / "models"
