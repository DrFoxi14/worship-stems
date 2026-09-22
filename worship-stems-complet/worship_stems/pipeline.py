"""The pipeline: one MP3 in, a full multitrack package out.

Separation is a tree, not a flat list. Each split partitions its parent, so
whatever a model gets wrong stays inside that branch and the leaves still add
up to the original mix exactly:

    mix
    |-- vocals ------------- lead_vocal, bgv
    `-- instrumental ------- drums --- kick, snare, toms, hihat, cymbals
                             bass
                             electric_guitar
                             keys
                             pads          (whatever is left: pads, strings, synth)

Each branch is cut by the model that is measurably best at it - RoFormer for
voice, Demucs for the rhythm section, MDX23C for the drum kit - rather than
asking one model to do everything.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio as A
from . import difficulty as DIFF
from . import harmonic as HARM
from . import multitrack as MT
from . import inventory as INV
from . import quality as Q
from . import regrid as GRID
from . import render as REND
from . import resynth as RS_SYNTH
from . import restore as RS
from . import spatial as SP
from . import transpose as TR
from .analyze import Analysis, analyze, rebuild_grid
from .click import build_click, tempo_map
from .config import (
    DEMUCS_6S,
    DEMUCS_RHYTHM,
    DRUMSEP,
    INSTRUMENTAL_SINGLE,
    KARAOKE_PRESET,
    KARAOKE_SINGLE,
    STEMS_BY_KEY,
    VOCAL_PRESETS,
    VOCAL_SINGLE,
    Options,
    StemSpec,
)
from .export import (
    ExportedFile,
    song_folder,
    stem_filename,
    write_chart,
    write_cue_file,
    write_markers_csv,
    write_reaper_project,
    write_session_json,
)
from .guide import build_guide
from .pads import build_pad
from .refine import fold_back, refine_partition
from .separate import SeparationEngine

log = logging.getLogger(__name__)

Progress = Callable[[str, float, str], None]  # stage, 0..1, message


@dataclass
class Result:
    folder: Path
    analysis: Analysis
    files: list[ExportedFile] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    inventory: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    spatial: dict = field(default_factory=dict)
    restore: dict = field(default_factory=dict)
    difficulty: dict = field(default_factory=dict)


# Which parent each stem belongs to - used by the inventory and the QC pass.
PARENTS = {
    "vocals": "mix",
    "instrumental": "mix",
    "lead_vocal": "vocals",
    "bgv": "vocals",
    "drums": "instrumental",
    "bass": "instrumental",
    "electric_guitar": "instrumental",
    "keys": "instrumental",
    "pads": "instrumental",
    "kick": "drums",
    "snare": "drums",
    "toms": "drums",
    "hihat": "drums",
    "cymbals": "drums",
}


class Pipeline:
    def __init__(self, options: Options | None = None, engine: SeparationEngine | None = None):
        self.opt = options or Options()
        self.engine = engine
        self._progress: Progress = lambda stage, frac, msg: None

    # -- helpers -----------------------------------------------------------

    def _p(self, stage: str, frac: float, msg: str = "") -> None:
        try:
            self._progress(stage, max(0.0, min(1.0, frac)), msg)
        except Exception:  # noqa: BLE001
            pass

    def _need_engine(self) -> SeparationEngine:
        if self.engine is None:
            self.engine = SeparationEngine()
        return self.engine

    # -- separation tree ---------------------------------------------------

    def _split(
        self,
        parent: np.ndarray,
        sr: int,
        estimates: dict[str, np.ndarray],
        residual_key: str,
    ) -> dict[str, np.ndarray]:
        if self.opt.partition != "exact":
            return A.partition_raw(parent, estimates, residual_key)

        parts = A.partition_exact(parent, estimates, power=self.opt.mask_power)
        if self.opt.refine and len(parts) > 1:
            parts = refine_partition(
                parent,
                parts,
                sr,
                iterations=self.opt.refine_iterations,
                power_start=self.opt.mask_power,
                power_end=self.opt.refine_power_end,
                band_gating=self.opt.refine_band_gating,
            )
        return parts

    # -- inventory ---------------------------------------------------------

    def _apply_inventory(self, stems: dict[str, np.ndarray], sr: int, warnings: list[str]) -> dict[str, dict]:
        """Drop stems for instruments that are not in the song.

        Nothing is deleted: a rejected stem's audio is folded into its
        group's residual, so the leaves still sum to the mix exactly. What
        goes away is the false label, not the sound.
        """
        verdicts = INV.take_inventory(stems, PARENTS, sr, self.opt.gate_strictness)

        clap: dict[str, float] = {}
        if self.opt.use_clap and INV.clap_available() and "mix" in stems:
            clap = INV.clap_opinion(stems["mix"], sr, list(verdicts))
            for key, score in clap.items():
                if key in verdicts:
                    v = verdicts[key]
                    # A second, independent opinion taken from the original
                    # mix - it cannot be fooled by an invented stem.
                    blended = 0.65 * v.confidence + 0.35 * score
                    v.evidence["clap"] = round(score, 3)
                    v.confidence = round(float(blended), 3)
                    v.decision = "present" if blended >= 0.62 else ("absent" if blended < 0.34 else "uncertain")

        if not self.opt.gate_instruments:
            return {k: v.to_dict() for k, v in verdicts.items()}

        groups: list[tuple[str, list[str], str]] = [
            ("vocals", ["lead_vocal", "bgv"], "lead_vocal"),
            ("instrumental", ["drums", "bass", "electric_guitar", "keys", "pads"], "pads"),
            ("drums", ["kick", "snare", "toms", "hihat", "cymbals"], ""),
        ]

        for _parent, members, residual in groups:
            present = [k for k in members if k in stems]
            if len(present) < 2:
                continue
            drop = [
                k for k in present
                if k != residual
                and k in verdicts
                and (
                    verdicts[k].decision == "absent"
                    or (verdicts[k].decision == "uncertain" and not self.opt.keep_uncertain)
                )
            ]
            if not drop:
                continue

            kept = fold_back({k: stems[k] for k in present}, drop, residual)
            for k in drop:
                stems.pop(k, None)
                # A missing drum bus means its kit pieces are meaningless too.
                if k == "drums":
                    for child in ("kick", "snare", "toms", "hihat", "cymbals"):
                        stems.pop(child, None)
            stems.update(kept)

            for k in drop:
                v = verdicts[k]
                warnings.append(
                    f"No {v.label} found in this song - {v.reasons[0]}. Not exported; "
                    f"that audio stayed with the other tracks."
                )

        return {k: v.to_dict() for k, v in verdicts.items()}

    def separate_tree(self, mix: np.ndarray, sr: int) -> tuple[dict[str, np.ndarray], list[str], dict]:
        eng = self._need_engine()
        tier = self.opt.tier()
        warnings: list[str] = []
        stems: dict[str, np.ndarray] = {"mix": mix}

        # 1. vocals vs instrumental -----------------------------------------
        self._p("separate", 0.05, "Vocals vs instrumental")
        if tier["ensemble"]:
            preset = VOCAL_PRESETS.get(self.opt.vocal_preset, "vocal_balanced")
            out = eng.run(mix, sr, ensemble_preset=preset)
        else:
            out = eng.run(mix, sr, model=VOCAL_SINGLE)
        est = {}
        if "vocals" in out:
            est["vocals"] = out["vocals"]
        if "instrumental" in out:
            est["instrumental"] = out["instrumental"]
        elif "vocals" in out:
            est["instrumental"] = mix - A.match_shape(out["vocals"], mix)
        if "vocals" not in est:
            raise RuntimeError("the vocal model returned no vocal stem")
        top = self._split(mix, sr, est, "instrumental")
        stems["vocals"], stems["instrumental"] = top["vocals"], top["instrumental"]

        # 2. lead vs backing vocals -----------------------------------------
        if self.opt.split_lead_bgv:
            self._p("separate", 0.30, "Lead vocal vs BGVs")
            try:
                if tier["ensemble"]:
                    k = eng.run(stems["vocals"], sr, ensemble_preset=KARAOKE_PRESET)
                else:
                    k = eng.run(stems["vocals"], sr, model=KARAOKE_SINGLE)
                # Karaoke models label their outputs "Vocals" (the lead) and
                # "Instrumental" (everything else in the vocal bus = the BGVs).
                lead = k["lead_vocal"] if "lead_vocal" in k else k.get("vocals")
                bgv = k["bgv"] if "bgv" in k else k.get("instrumental")
                if lead is None or bgv is None:
                    raise RuntimeError(f"karaoke model returned {list(k)}")
                parts = self._split(stems["vocals"], sr, {"lead_vocal": lead, "bgv": bgv}, "bgv")
                stems.update(parts)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Lead/BGV split failed, keeping one vocal track ({exc}).")
                log.warning("lead/bgv failed: %s", traceback.format_exc())

        # 3. instrumental -> rhythm section + tonal -------------------------
        self._p("separate", 0.45, "Drums, bass, guitars, keys")
        inst = stems["instrumental"]
        est_inst: dict[str, np.ndarray] = {}

        rhythm = eng.run(inst, sr, model=DEMUCS_RHYTHM)
        for key in ("drums", "bass"):
            if key in rhythm:
                est_inst[key] = rhythm[key]

        if tier["bass_model"] != DEMUCS_RHYTHM:
            try:
                alt = eng.run(inst, sr, model=tier["bass_model"])
                if "bass" in alt:
                    est_inst["bass"] = alt["bass"]
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Second bass model unavailable ({exc}); using htdemucs_ft.")

        if self.opt.split_guitar_keys and tier["six_stem"]:
            self._p("separate", 0.60, "Guitars and keys")
            try:
                six = eng.run(inst, sr, model=DEMUCS_6S)
                if "electric_guitar" in six:
                    est_inst["electric_guitar"] = six["electric_guitar"]
                if "keys" in six:
                    est_inst["keys"] = six["keys"]
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Guitar/keys split unavailable ({exc}).")

        # "other" from demucs becomes the pads bucket; anything the models
        # could not claim lands here too, which is correct for worship:
        # pads, strings and synth are exactly the leftovers.
        est_inst["pads"] = rhythm.get("other", inst - sum(est_inst.values()))
        parts = self._split(inst, sr, est_inst, "pads")
        stems.update(parts)

        # 4. drum kit -------------------------------------------------------
        if self.opt.split_drums and tier["drumsep"] and "drums" in stems:
            self._p("separate", 0.78, "Kick, snare, toms, cymbals")
            try:
                d = eng.run(stems["drums"], sr, model=DRUMSEP)
                wanted = {k: v for k, v in d.items() if k in {"kick", "snare", "toms", "hihat", "cymbals"}}
                if len(wanted) >= 2:
                    stems.update(self._split(stems["drums"], sr, wanted, "cymbals"))
                else:
                    warnings.append(f"Drum split returned only {list(d)}; keeping one drum track.")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Drum kit split unavailable ({exc}).")

        # 5. which of these instruments are actually in the song? ----------
        self._p("separate", 0.88, "Checking which instruments are really there")
        verdicts = self._apply_inventory(stems, sr, warnings)

        self._p("separate", 0.95, "Separation done")
        return stems, warnings, verdicts

    # -- main ---------------------------------------------------------------

    def run_multitrack(
        self,
        folder: str | Path,
        output_root: str | Path,
        progress: Progress | None = None,
        analysis: Analysis | None = None,
        title: str | None = None,
    ) -> Result:
        """Build the package from real console stems instead of a mix.

        Nothing here has to be guessed: the stems are the recorded channels,
        so separation, instrument detection and hole repair are all skipped.
        What is left is the half that is still worth doing - tempo, key,
        structure, click, guide, MIDI and a session that opens in one click.
        """
        started = time.time()
        if progress:
            self._progress = progress
        folder = Path(folder)
        title = title or folder.name

        self._p("load", 0.0, f"Reading the channels in {folder.name}")
        stems, mix, sr, mt_report = MT.load_multitrack(folder, sr=self.opt.sample_rate)
        stems["mix"] = mix
        n = mix.shape[-1]
        self._p("load", 1.0, f"{len(mt_report.channels)} channels, {mt_report.duration:.0f}s")

        warnings: list[str] = []
        if mt_report.unmatched:
            warnings.append(
                "These channels were not recognised and kept as their own tracks: "
                + ", ".join(mt_report.unmatched)
                + ". Add a channels.json in the folder to name them."
            )
        if mt_report.click_channel:
            warnings.append(f"'{mt_report.click_channel}' looks like a click and was left out of the mix.")

        if analysis is None:
            self._p("analyze", 0.0, "Tempo, key and structure")
            analysis = analyze(mix, sr, source_path=None)
            self._p("analyze", 1.0, f"{analysis.key_display}, {analysis.bpm:.0f} BPM, {len(analysis.sections)} sections")
        if self.opt.tempo_mode == "fixed":
            analysis = rebuild_grid(analysis, self.opt.fixed_bpm or analysis.bpm)

        leaves = [k for k in stems if k != "mix"]
        metrics = {
            "Source": f"{len(mt_report.channels)} console channels (no separation needed)",
            "Leaf stems": ", ".join(STEMS_BY_KEY[k].label if k in STEMS_BY_KEY else k for k in leaves),
            "Recombined vs original": "exact by construction - these are the recorded channels",
        }

        notes_by_stem, midi_by_stem = self._transcribe(stems, MT.pitched_stems(stems), metrics, warnings)

        if self.opt.make_minus_one:
            vocal_keys = {"lead_vocal", "bgv"}
            instrumental = sum((v for k, v in stems.items() if k not in vocal_keys and k != "mix"), np.zeros_like(mix))
            stems["minus_one"] = instrumental
            acapella = sum((v for k, v in stems.items() if k in vocal_keys), np.zeros_like(mix))
            if np.any(acapella):
                stems["acapella"] = acapella

        stems["recombined"] = sum(stems[k] for k in leaves)

        stems, analysis, post = self._post_process(
            stems, analysis, sr, notes_by_stem, {}, metrics, warnings
        )
        n = max((v.shape[-1] for v in stems.values()), default=n)

        return self._finish(
            title=title, stems=stems, sr=sr, n=n, analysis=analysis, metrics=metrics,
            warnings=warnings, output_root=Path(output_root), started=started,
            verdicts={}, quality_report={}, spatial_report={}, restore_reports={},
            midi_by_stem=midi_by_stem,
            extra={"multitrack": mt_report.to_dict(), **post},
        )

    def _transcribe(
        self,
        stems: dict[str, np.ndarray],
        keys: list[str],
        metrics: dict,
        warnings: list[str],
    ) -> tuple[dict[str, list], dict[str, object]]:
        notes_by_stem: dict[str, list] = {}
        midi_by_stem: dict[str, object] = {}
        if not (self.opt.transcribe_midi or self.opt.note_informed_restore):
            return notes_by_stem, midi_by_stem
        if not HARM.transcription_available():
            warnings.append(
                f"Note transcription is unavailable - {HARM.transcription_error()}. "
                "Install it with: pip install --no-deps basic-pitch && "
                "pip install resampy mir-eval pretty_midi onnxruntime"
            )
            return notes_by_stem, midi_by_stem

        sr = self.opt.sample_rate or 44100
        self._p("analyze", 0.15, "Working out the notes")
        for key in keys:
            if key not in stems:
                continue
            found, midi_obj = HARM.transcribe(stems[key], sr, stem_key=key)
            if found:
                notes_by_stem[key] = found
                midi_by_stem[key] = midi_obj
        if notes_by_stem:
            metrics["Notes transcribed"] = ", ".join(
                f"{STEMS_BY_KEY[k].label if k in STEMS_BY_KEY else k} {len(v)}"
                for k, v in notes_by_stem.items()
            )
        return notes_by_stem, midi_by_stem

    def analyze_only(self, input_path: str | Path, progress: Progress | None = None) -> Analysis:
        """Fast first pass, so the section list can be corrected before the slow part."""
        if progress:
            self._progress = progress
        input_path = Path(input_path)
        self._p("analyze", 0.1, "Tempo, key and structure")
        mix, sr = A.load_audio(input_path)
        result = analyze(mix, sr, source_path=str(input_path))
        self._p("analyze", 1.0, f"{result.key_display}, {result.bpm:.0f} BPM, {len(result.sections)} sections")
        return result

    def run(
        self,
        input_path: str | Path,
        output_root: str | Path,
        progress: Progress | None = None,
        analysis: Analysis | None = None,
    ) -> Result:
        started = time.time()
        if progress:
            self._progress = progress
        input_path = Path(input_path)
        title = input_path.stem

        self._p("load", 0.0, f"Loading {input_path.name}")
        mix, sr = A.load_audio(input_path)
        if self.opt.sample_rate and self.opt.sample_rate != sr:
            mix = A.resample(mix, sr, self.opt.sample_rate)
            sr = self.opt.sample_rate
        n = mix.shape[-1]

        difficulty: dict = {}
        if self.opt.assess_difficulty:
            self._p("analyze", 0.0, "Judging how hard this one will be")
            try:
                verdict = DIFF.assess(mix, sr)
                difficulty = verdict.to_dict()
                difficulty["advice"] = DIFF.advice(verdict)
                self._p("analyze", 0.05, f"{verdict.verdict}: {verdict.summary}")
            except Exception as exc:  # noqa: BLE001
                log.warning("difficulty check failed: %s", exc)

        if analysis is None:
            self._p("analyze", 0.08, "Tempo, key and structure")
            analysis = analyze(mix, sr, source_path=str(input_path))
            self._p("analyze", 1.0, f"{analysis.key_display}, {analysis.bpm:.0f} BPM, {len(analysis.sections)} sections")

        if self.opt.tempo_mode == "fixed":
            target = self.opt.fixed_bpm or analysis.bpm
            analysis = rebuild_grid(analysis, target)
            self._p("analyze", 1.0, f"Grid rebuilt at a steady {analysis.bpm:.1f} BPM")

        warnings: list[str] = []
        stems: dict[str, np.ndarray] = {"mix": mix}
        verdicts: dict = {}
        # An engine handed in explicitly is used as given; otherwise fall back
        # to whatever is installed.
        if self.engine is not None or SeparationEngine.available():
            try:
                stems, warnings, verdicts = self.separate_tree(mix, sr)
            except Exception as exc:  # noqa: BLE001
                log.error("separation failed: %s", traceback.format_exc())
                warnings.append(f"Separation failed: {exc}. Only click, guide and analysis were produced.")
        else:
            detail = SeparationEngine.import_error() or "not installed"
            warnings.append(
                f"Separation is unavailable, so no instrument stems were produced - {detail}. "
                "Fix it with: pip install -r requirements.txt --upgrade"
            )

        # Optional: split a stem by where it sits in the stereo image.
        spatial_report: dict = {}
        if self.opt.spatial_split:
            for key in self.opt.spatial_targets:
                if key not in stems:
                    continue
                report = SP.pan_diversity(stems[key], sr)
                spatial_report[key] = report
                if not report["usable"]:
                    warnings.append(
                        f"{STEMS_BY_KEY[key].label if key in STEMS_BY_KEY else key}: cannot be split by "
                        f"position - {report['reason']}. Two sources in the same place cannot be told apart."
                    )
                    continue
                layers = SP.split_by_position(stems[key], sr)
                base = STEMS_BY_KEY.get(key)
                for i, (side, audio_layer) in enumerate(layers.items(), start=1):
                    new_key = f"{key}_{side}"
                    stems[new_key] = audio_layer
                    PARENTS[new_key] = key
                    STEMS_BY_KEY[new_key] = StemSpec(
                        key=new_key,
                        label=f"{base.label if base else key} ({side})",
                        filename=f"{(base.filename if base else key)}{chr(96 + i)}_{side}",
                        parent=key,
                        color=base.color if base else "#7c8aa5",
                    )

        # Which leaves must add back up to the mix.
        vocal_leaves = [k for k in ("lead_vocal", "bgv") if k in stems]
        drum_leaves = [k for k in ("kick", "snare", "toms", "hihat", "cymbals") if k in stems]
        leaves = vocal_leaves or [k for k in ("vocals",) if k in stems]
        leaves += drum_leaves or [k for k in ("drums",) if k in stems]
        leaves += [k for k in ("bass", "electric_guitar", "keys", "pads") if k in stems]

        metrics: dict[str, str] = {}
        if difficulty:
            metrics["Expected quality"] = f"{difficulty['verdict']} - {difficulty['summary']}"
            if difficulty["verdict"] in {"hard", "very hard"}:
                warnings.append(difficulty["summary"] + " " + difficulty["reasons"][0] + ".")
                warnings.extend(difficulty.get("advice", []))
        if leaves:
            recombined = sum(stems[k] for k in leaves)
            err = A.reconstruction_error_db(mix, [stems[k] for k in leaves])
            metrics["Leaf stems"] = ", ".join(STEMS_BY_KEY[k].label for k in leaves if k in STEMS_BY_KEY)
            metrics["Recombined vs original"] = ("below float precision" if err == -np.inf else f"{err:.1f} dB")
            metrics["Partition mode"] = self.opt.partition + (
                f" + {self.opt.refine_iterations}-pass refine" if self.opt.refine else ""
            )
            stems["recombined"] = recombined
        else:
            stems["recombined"] = mix.copy()

        # Reverse, then forward again: the cut tells us where each sound came
        # from; this puts back what the cut removed. The proof file above was
        # built from the exact chain and is not touched by any of it.
        # What notes was each instrument actually playing? The MIDI is worth
        # having on its own, and it is what lets a buried partial be rebuilt
        # from the other partials of the same note.
        notes_by_stem: dict[str, list] = {}
        midi_by_stem: dict[str, object] = {}
        if (self.opt.transcribe_midi or self.opt.note_informed_restore) and leaves:
            if HARM.transcription_available():
                self._p("analyze", 0.15, "Working out the notes")
                for key in self.opt.midi_stems:
                    if key not in stems:
                        continue
                    found, midi_obj = HARM.transcribe(stems[key], sr, stem_key=key)
                    if found:
                        notes_by_stem[key] = found
                        midi_by_stem[key] = midi_obj
                if notes_by_stem:
                    metrics["Notes transcribed"] = ", ".join(
                        f"{STEMS_BY_KEY[k].label if k in STEMS_BY_KEY else k} {len(v)}"
                        for k, v in notes_by_stem.items()
                    )
            else:
                warnings.append(
                    "Note transcription is unavailable, so the note-informed repair was "
                    f"skipped - {HARM.transcription_error()}. Install it with: "
                    "pip install --no-deps basic-pitch && pip install resampy mir-eval "
                    "pretty_midi onnxruntime"
                )

        restore_reports: dict = {}
        resynth_reports: dict = {}
        timbres: dict = {}
        if self.opt.restore and leaves and len(stems) > 1:
            self._p("analyze", 0.25, "Rebuilding what the cut removed")
            labels = {k: (STEMS_BY_KEY[k].label if k in STEMS_BY_KEY else k) for k in stems}
            exact_leaves = {k: stems[k].copy() for k in leaves if k in stems}

            # Rebuild each instrument from its own score and learned timbre.
            # The synthesis has no holes in it, so it is a clean source for
            # exactly the bins the cut destroyed - and the share of the stem
            # it explains is an honest measure of how well we understood it.
            syntheses: dict[str, np.ndarray] = {}
            if self.opt.resynthesize and notes_by_stem:
                self._p("analyze", 0.20, "Rebuilding each instrument from its score")
                for key, note_list in notes_by_stem.items():
                    if key not in stems:
                        continue
                    try:
                        synth, _resid, srep = RS_SYNTH.explain(
                            stems[key], sr, key,
                            STEMS_BY_KEY[key].label if key in STEMS_BY_KEY else key,
                            rounds=self.opt.resynth_rounds, notes=note_list,
                        )
                        resynth_reports[key] = srep.to_dict()
                        try:
                            timbres[key] = RS_SYNTH.learn_timbre(stems[key], note_list, sr)
                        except Exception:  # noqa: BLE001
                            pass
                        if self.opt.resynth_fill:
                            syntheses[key] = synth
                    except Exception as exc:  # noqa: BLE001
                        log.warning("resynthesis failed for %s: %s", key, exc)
                if resynth_reports:
                    metrics["Score explains"] = ", ".join(
                        f"{v['label']} {v['explained']*100:.0f}%" for v in resynth_reports.values()
                    )

            try:
                repaired, restore_reports = RS.restore_all(
                    stems, PARENTS, labels, leaves, sr,
                    max_gap_ms=self.opt.restore_max_gap_ms,
                    notes=notes_by_stem if self.opt.note_informed_restore else None,
                    analysis=analysis if self.opt.borrow_repeats else None,
                    offset=0.0,
                    min_agreement=self.opt.repeat_min_agreement,
                    syntheses=syntheses,
                )

                # If every instrument really played what we say it played,
                # does the record still come out the same? Anything that
                # cannot be reconciled gets trimmed back.
                if self.opt.mix_consistency and repaired:
                    self._p("analyze", 0.35, "Checking it still reconciles with the mix")
                    repaired, consistency = RS.enforce_mix_consistency(
                        repaired, exact_leaves, mix, leaves,
                        tolerance_db=self.opt.mix_tolerance_db,
                    )
                    restore_reports["_mix_consistency"] = consistency
                    if consistency["bins_trimmed"]:
                        metrics["Trimmed to fit the mix"] = f"{consistency['bins_trimmed']:,} bins"
                if self.opt.restore_chain == "both":
                    for key, audio_r in repaired.items():
                        new_key = f"{key}__restored"
                        stems[new_key] = audio_r
                        PARENTS[new_key] = PARENTS.get(key)
                        base = STEMS_BY_KEY.get(key)
                        STEMS_BY_KEY[new_key] = StemSpec(
                            key=new_key,
                            label=f"{base.label if base else key} (restored)",
                            filename=f"{(base.filename if base else key)}R",
                            parent=PARENTS.get(key),
                            color=base.color if base else "#7c8aa5",
                        )
                elif self.opt.restore_chain == "restored":
                    stems.update(repaired)
                per_stem = {k: v for k, v in restore_reports.items() if not k.startswith("_")}
                total_fixed = sum(r["bins_restored"] for r in per_stem.values())
                total_refused = sum(r["bins_refused"] for r in per_stem.values())
                total_partials = sum(
                    r.get("harmonic", {}).get("partials_rebuilt", 0) for r in per_stem.values()
                )
                if total_partials:
                    metrics["Partials rebuilt from the note"] = f"{total_partials:,}"
                total_borrowed = sum(
                    r.get("repeats", {}).get("bins_borrowed", 0) for r in per_stem.values()
                )
                if total_borrowed:
                    metrics["Borrowed from repeats"] = f"{total_borrowed:,} bins" 
                if total_fixed or total_refused:
                    metrics["Holes rebuilt"] = (
                        f"{total_fixed:,} bins repaired, {total_refused:,} left alone "
                        f"(nothing to read across)"
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("restore failed: %s", exc)
                warnings.append(f"Hole repair skipped ({exc}).")

        # How does each track actually sound - bleed and artefacts, not dB.
        quality_report: dict = {}
        if self.opt.quality_report and leaves:
            self._p("analyze", 0.5, "Checking bleed and artefacts")
            labels = {k: (STEMS_BY_KEY[k].label if k in STEMS_BY_KEY else k) for k in stems}
            try:
                assessed = Q.assess(stems, PARENTS, labels, sr, leaves)
                quality_report = {k: v.to_dict() for k, v in assessed.items()}
                rough = [v.label for v in assessed.values() if v.verdict == "rough"]
                if rough:
                    warnings.append("Rough tracks worth a listen before use: " + ", ".join(rough) + ".")
            except Exception as exc:  # noqa: BLE001
                log.warning("quality report failed: %s", exc)

        # Derived mixes.
        if self.opt.make_minus_one and "instrumental" in stems:
            stems["minus_one"] = stems["instrumental"].copy()
        if self.opt.make_acapella and "vocals" in stems:
            stems["acapella"] = stems["vocals"].copy()

        stems, analysis, post = self._post_process(
            stems, analysis, sr, notes_by_stem, timbres, metrics, warnings
        )
        n = max((v.shape[-1] for v in stems.values()), default=n)

        return self._finish(
            title=title, stems=stems, sr=sr, n=n, analysis=analysis, metrics=metrics,
            warnings=warnings, output_root=Path(output_root), started=started,
            verdicts=verdicts, quality_report=quality_report, spatial_report=spatial_report,
            restore_reports=restore_reports, midi_by_stem=midi_by_stem,
            extra={"resynthesis": resynth_reports, "difficulty": difficulty, **post},
            difficulty=difficulty,
        )

    # -- shared post-processing -------------------------------------------

    def _post_process(
        self,
        stems: dict,
        analysis: Analysis,
        sr: int,
        notes_by_stem: dict,
        timbres: dict,
        metrics: dict,
        warnings: list,
    ) -> tuple[dict, Analysis, dict]:
        """Rendered parts, re-grid and transposition, in that order.

        Order matters: rendering needs the original timing, the re-grid moves
        everything onto a steady tempo, and transposition comes last so the
        click, guide and pad built afterwards are all in the final key.
        """
        extra: dict = {}

        # --- play the part rather than extract it ------------------------
        if self.opt.render_parts and notes_by_stem:
            self._p("analyze", 0.55, "Playing the parts with a clean sound")
            rendered = 0
            for key, note_list in notes_by_stem.items():
                if not note_list:
                    continue
                patch = self.opt.render_patches.get(key)
                try:
                    audio_r, label = REND.render_stem(
                        key, note_list, analysis.duration, sr,
                        timbre=timbres.get(key), patch_name=patch,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("render failed for %s: %s", key, exc)
                    continue
                new_key = f"{key}__played"
                stems[new_key] = audio_r
                base = STEMS_BY_KEY.get(key)
                STEMS_BY_KEY[new_key] = StemSpec(
                    key=new_key,
                    label=f"{base.label if base else key} (played, {label})",
                    filename=f"{(base.filename if base else key)}P",
                    parent=None,
                    color=base.color if base else "#7c8aa5",
                )
                rendered += 1
            if rendered:
                metrics["Parts played back"] = f"{rendered} instruments rendered from their notes"

        # --- put it on a grid --------------------------------------------
        if self.opt.regrid:
            self._p("analyze", 0.65, "Putting the song on a steady grid")
            try:
                stems, analysis, report = GRID.regrid_all(
                    stems, analysis, sr, target_bpm=self.opt.regrid_bpm
                )
                if report:
                    extra["regrid"] = report.to_dict()
                    metrics["Re-gridded"] = (
                        f"{report.target_bpm:.1f} BPM steady; the record wandered "
                        f"{report.drift_before_ms:.0f} ms"
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("regrid failed: %s", exc)
                warnings.append(f"Could not put the song on a grid ({exc}); left as recorded.")

        # --- change key ---------------------------------------------------
        semitones = self.opt.transpose_semitones
        if self.opt.transpose_to_key:
            semitones = TR.suggest_for_singer(analysis, self.opt.transpose_to_key)
        if semitones:
            self._p("analyze", 0.75, f"Transposing {semitones:+d} semitones")
            try:
                stems, analysis, report = TR.transpose_stems(stems, analysis, sr, semitones)
                stems = TR.peak_safe(stems)
                extra["transpose"] = report.to_dict()
                metrics["Transposed"] = f"{report.from_key} to {report.to_key} ({semitones:+d})"
                if report.left_alone:
                    metrics["Left at pitch"] = ", ".join(report.left_alone) + " (percussion has no key)"
            except Exception as exc:  # noqa: BLE001
                log.warning("transpose failed: %s", exc)
                warnings.append(f"Could not transpose ({exc}); left in the original key.")

        return stems, analysis, extra

    # -- shared export -----------------------------------------------------

    def _finish(
        self,
        title: str,
        stems: dict,
        sr: int,
        n: int,
        analysis: Analysis,
        metrics: dict,
        warnings: list,
        output_root: Path,
        started: float,
        verdicts: dict,
        quality_report: dict,
        spatial_report: dict,
        restore_reports: dict,
        midi_by_stem: dict,
        extra: dict | None = None,
        difficulty: dict | None = None,
    ) -> Result:
        """Click, guide, pad and the whole export - shared by both entry points."""
        offset = 0.0
        cues: list[dict] = []
        click = guide = None
        if self.opt.make_click:
            self._p("click", 0.2, "Building the click")
            click, offset = build_click(
                analysis, sr, n,
                count_in_bars=self.opt.click_count_in_bars,
                subdivision=self.opt.click_subdivision,
            )
        if self.opt.make_guide:
            self._p("click", 0.6, "Speaking the guide cues")
            guide, cues = build_guide(
                analysis, sr, n,
                offset=offset,
                lang=self.opt.guide_language,
                lead_beats=self.opt.guide_lead_beats,
                count_in=self.opt.guide_count_in,
                count_in_bars=self.opt.click_count_in_bars,
            )

        # Everything shifts by the count-in so the package lines up in any DAW.
        pad_samples = int(round(offset * sr))
        if pad_samples > 0:
            for k, v in list(stems.items()):
                stems[k] = np.concatenate([np.zeros((v.shape[0], pad_samples), dtype=np.float32), v], axis=1)
        total = n + pad_samples

        if click is not None:
            stems["click"] = A.match_length(click, total)
        if guide is not None:
            stems["guide"] = A.match_length(guide, total)
        if click is not None and guide is not None and self.opt.make_click_guide_combined:
            stems["click_guide"] = np.clip(stems["click"] * 0.85 + stems["guide"] * 0.95, -1.0, 1.0)

        if self.opt.make_ambient_pad:
            self._p("click", 0.85, "Generating the ambient pad")
            stems["pad_key"] = A.match_length(build_pad(analysis.key, analysis.mode, total / sr, sr), total)

        self._p("export", 0.0, "Writing files")
        folder = song_folder(output_root, title, analysis)
        files: list[ExportedFile] = []
        order = [s.key for s in STEMS_BY_KEY.values()]
        keys = sorted(stems, key=lambda k: order.index(k) if k in order else 999)
        generated = {"click", "guide", "click_guide", "pad_key"}

        for i, key in enumerate(keys):
            data = stems[key]
            if self.opt.normalize_stems and key not in generated and key != "mix":
                data = A.normalize_lufs(data, sr, self.opt.target_lufs)
            if self.opt.limiter and key not in generated:
                data = A.soft_limit(data, sr, -1.0)
            spec = STEMS_BY_KEY.get(key)
            path = folder / f"{stem_filename(key, title)} - {title[:40]}"
            written = A.save_audio(path, data, sr, self.opt.output_format)
            files.append(
                ExportedFile(
                    key=key,
                    label=spec.label if spec else key.replace("_", " ").title(),
                    path=written,
                    peak_db=A.peak_db(data),
                    lufs=None,
                )
            )
            self._p("export", (i + 1) / max(1, len(keys)), written.name)

        quality = {
            "tier": self.opt.quality,
            "vocal_preset": self.opt.vocal_preset,
            "device": SeparationEngine.describe_device(),
            "tempo_mode": self.opt.tempo_mode,
            "refine": self.opt.refine,
            "instrument_gating": self.opt.gate_instruments,
            "inventory": verdicts,
            "stem_quality": quality_report,
            "spatial": spatial_report,
            "restore": restore_reports,
        }
        if extra:
            quality.update(extra)
        if self.opt.write_session_json:
            write_session_json(folder / "session.json", title, analysis, files, offset, cues, quality, metrics)
        if self.opt.write_markers_csv:
            write_markers_csv(folder / "markers.csv", analysis, offset)
            write_cue_file(folder / "sections.cue", analysis, files[0].path.name if files else "", offset)
        if self.opt.write_reaper_project:
            write_reaper_project(folder / f"{title[:40]}.RPP", title, analysis, files, offset)
        (folder / "tempo_map.json").write_text(
            __import__("json").dumps(tempo_map(analysis), indent=2), encoding="utf-8"
        )
        if self.opt.transcribe_midi and midi_by_stem:
            midi_dir = folder / "MIDI"
            written_midi = 0
            for key, midi_obj in midi_by_stem.items():
                spec = STEMS_BY_KEY.get(key)
                name = (spec.filename if spec else key)
                program = HARM.MIDI_PROGRAMS.get(key, 0)
                if HARM.write_midi(midi_obj, midi_dir / f"{name}.mid", program):
                    written_midi += 1
            if written_midi:
                metrics["MIDI written"] = f"{written_midi} files in MIDI/"

        write_chart(folder / "CHART.md", title, analysis, offset, metrics)

        self._p("done", 1.0, f"{len(files)} tracks in {folder.name}")
        return Result(
            folder=folder,
            analysis=analysis,
            files=files,
            metrics=metrics,
            warnings=warnings,
            seconds=time.time() - started,
            inventory=verdicts,
            quality=quality_report,
            spatial=spatial_report,
            restore=restore_reports,
            difficulty=difficulty or {},
        )

