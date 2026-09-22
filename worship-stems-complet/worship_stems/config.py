"""Configuration: model choices, stem tree, quality tiers.

Model rankings come from the SDR scores shipped with `audio-separator`
(MUSDB18-HQ style evaluation, 115 scored models). Summary of the picture
as of 2026-09:

  vocals / instrumental : RoFormer family wins clearly
                          (bs_roformer ~11.6 dB vocals vs htdemucs_ft ~8.x)
  drums / bass / other  : Demucs v4 still wins
                          (htdemucs_ft 9.68 drums, hdemucs_mmi 12.98 bass)
  lead vs backing       : MelBand "karaoke" models, best as a 3-model ensemble
  kick/snare/toms/hh    : MDX23C-DrumSep
  guitar / piano        : htdemucs_6s is the only local option

So the pipeline is deliberately heterogeneous: each stem is cut by the model
that is actually best at that stem, not by one model doing everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

# Ensemble presets built into audio-separator >= 0.47 (community curated).
VOCAL_PRESETS = {
    "balanced": "vocal_balanced",
    "clean": "vocal_clean",  # least instrumental bleed
    "full": "vocal_full",  # keeps breath / tails, more bleed
}

# Single models, used when ensembling is switched off (much faster).
VOCAL_SINGLE = "model_bs_roformer_ep_368_sdr_12.9628.ckpt"  # 11.63 dB vocals
INSTRUMENTAL_SINGLE = "bs_roformer_vocals_gabox.ckpt"  # 16.41 dB instrumental

# Lead vs backing vocals.
KARAOKE_PRESET = "karaoke"
KARAOKE_SINGLE = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"

# Rhythm section + tonal instruments.
DEMUCS_RHYTHM = "htdemucs_ft.yaml"  # best drums (9.68) and near-best bass
DEMUCS_BASS = "hdemucs_mmi.yaml"  # best bass (12.98)
DEMUCS_6S = "htdemucs_6s.yaml"  # adds guitar + piano stems

# Kick / snare / toms / hihat / cymbals out of the drum bus.
DRUMSEP = "MDX23C-DrumSep-aufr33-jarredou.ckpt"

# What each tier is *capable* of. Whether a capability is used is decided by
# Options - the tier must never override an explicit choice.
QUALITY_TIERS = {
    # name: (use ensembles, can split the kit, run 6-stem, bass model)
    "fast": dict(ensemble=False, drumsep=False, six_stem=False, bass_model=DEMUCS_RHYTHM),
    "standard": dict(ensemble=False, drumsep=False, six_stem=True, bass_model=DEMUCS_RHYTHM),
    "maximum": dict(ensemble=True, drumsep=True, six_stem=True, bass_model=DEMUCS_BASS),
}

# --------------------------------------------------------------------------
# Stem tree
# --------------------------------------------------------------------------


@dataclass
class StemSpec:
    """One output track."""

    key: str
    label: str  # MultiTracks-style display name
    filename: str  # export filename (without extension)
    parent: str | None = None
    color: str = "#7c8aa5"
    optional: bool = False


# The tree is the contract: every node's children partition the node exactly,
# so summing all leaves reproduces the original mix sample for sample.
STEM_TREE: list[StemSpec] = [
    StemSpec("mix", "Original Mix", "00_Original_Mix", None, "#444c5c"),
    # --- vocals -----------------------------------------------------------
    StemSpec("vocals", "All Vocals", "10_Vocals_All", "mix", "#e0685a"),
    StemSpec("lead_vocal", "Lead Vocal", "11_Lead_Vocal", "vocals", "#e0685a"),
    StemSpec("bgv", "Background Vocals", "12_BGVs", "vocals", "#e8917f"),
    # --- instrumental -----------------------------------------------------
    StemSpec("instrumental", "Instrumental", "20_Instrumental", "mix", "#4f8a76"),
    StemSpec("drums", "Drums", "30_Drums", "instrumental", "#c99a3e"),
    StemSpec("kick", "Kick", "31_Kick", "drums", "#c99a3e", optional=True),
    StemSpec("snare", "Snare", "32_Snare", "drums", "#c99a3e", optional=True),
    StemSpec("toms", "Toms", "33_Toms", "drums", "#c99a3e", optional=True),
    StemSpec("hihat", "Hi-Hat", "34_HiHat", "drums", "#c99a3e", optional=True),
    StemSpec("cymbals", "Cymbals / OH", "35_Cymbals", "drums", "#c99a3e", optional=True),
    StemSpec("bass", "Bass", "40_Bass", "instrumental", "#8a6fb0"),
    StemSpec("electric_guitar", "Guitars", "50_Guitars", "instrumental", "#5b8fc9"),
    StemSpec("keys", "Piano / Keys", "60_Keys", "instrumental", "#4aa3a3"),
    StemSpec("pads", "Pads / Synth / Strings", "70_Pads_Synth", "instrumental", "#9b7fb0"),
    # --- generated (not part of the partition) ----------------------------
    StemSpec("click", "Click", "01_Click", None, "#888888"),
    StemSpec("guide", "Guide", "02_Guide", None, "#888888"),
    StemSpec("click_guide", "Click + Guide", "03_Click_Guide", None, "#888888"),
    StemSpec("pad_key", "Ambient Pad (key)", "80_Ambient_Pad", None, "#9b7fb0"),
    StemSpec("minus_one", "Minus One (no vocals)", "90_Minus_One", None, "#4f8a76"),
    StemSpec("acapella", "Acapella (vocals only)", "91_Acapella", None, "#e0685a"),
    StemSpec("recombined", "Recombined (proof)", "99_Recombined", None, "#444c5c"),
]

STEMS_BY_KEY = {s.key: s for s in STEM_TREE}

# Leaves whose sum must equal the mix.
PARTITION_LEAVES_FULL = [
    "lead_vocal",
    "bgv",
    "kick",
    "snare",
    "toms",
    "hihat",
    "cymbals",
    "bass",
    "electric_guitar",
    "keys",
    "pads",
]
PARTITION_LEAVES_NO_DRUMSEP = [
    "lead_vocal",
    "bgv",
    "drums",
    "bass",
    "electric_guitar",
    "keys",
    "pads",
]
PARTITION_LEAVES_BASIC = ["vocals", "drums", "bass", "pads"]


# --------------------------------------------------------------------------
# Run options
# --------------------------------------------------------------------------


@dataclass
class Options:
    quality: Literal["fast", "standard", "maximum"] = "maximum"
    vocal_preset: str = "balanced"

    # Partition mode:
    #   "exact" - soft-mask renormalisation, leaves sum to the mix bit-exactly
    #   "raw"   - raw model outputs plus a residual track
    partition: Literal["exact", "raw"] = "exact"
    mask_power: float = 2.0  # 2.0 = Wiener-style, 1.0 = magnitude ratio

    split_lead_bgv: bool = True
    # The kit split is off by default: it is the slowest and most fragile
    # stage, and a worship band almost never needs kick and snare on separate
    # tracks - an electric kit arrives as one stereo pair anyway.
    split_drums: bool = False
    split_guitar_keys: bool = True

    # Instrument inventory: never export a track for something that is not
    # in the song. Rejected stems fold back into the residual, so the sum
    # stays exact and no audio is lost.
    gate_instruments: bool = True
    gate_strictness: float = 1.0  # >1 = fussier, <1 = keep more
    keep_uncertain: bool = True  # export borderline stems, flagged
    use_clap: bool = False  # optional zero-shot opinion on the original mix

    # Second pass: sharpen the split and cut each stem where its instrument
    # cannot physically be playing.
    refine: bool = True
    refine_iterations: int = 2
    refine_power_end: float = 5.0
    refine_band_gating: bool = True

    # Third pass: rebuild what the cut took out. The exported stems become
    # the repaired ones; the proof file is still built from the exact chain,
    # so the guarantee and the playable tracks both survive.
    restore: bool = True
    restore_max_gap_ms: float = 250.0
    restore_chain: Literal["restored", "both", "proof"] = "restored"

    # Transcribe each instrument to notes. The MIDI is useful on its own, and
    # knowing the notes lets a buried partial be rebuilt from the other
    # partials of the same note rather than only from its own past.
    transcribe_midi: bool = True
    note_informed_restore: bool = True
    midi_stems: tuple[str, ...] = ("lead_vocal", "bgv", "bass", "electric_guitar", "keys", "pads")

    # Take the clean version from another time the band played the same part.
    # This is the strongest evidence in the pipeline - not an inference at
    # all, but the same bar, genuinely recorded, with nothing on top of it.
    borrow_repeats: bool = True
    repeat_min_agreement: float = 0.45

    # Rebuild each instrument from its own score and learned timbre, feeding
    # the residual back to find the notes the first pass missed.
    resynthesize: bool = True
    resynth_rounds: int = 3
    resynth_fill: bool = True

    # Play the part instead of extracting it: render each instrument from
    # its transcribed notes with a clean sound, alongside the extracted stem.
    render_parts: bool = False
    render_patches: dict = field(default_factory=dict)  # stem key -> patch name

    # Put the song on a steady grid so the tracks and the click agree,
    # instead of the band chasing a drifting record.
    regrid: bool = False
    regrid_bpm: float | None = None  # None = the song's own median tempo

    # Say how well this song will separate before spending the time.
    assess_difficulty: bool = True

    # Change key. Percussion is left alone; the pad is re-rendered.
    transpose_semitones: int = 0
    transpose_to_key: str | None = None  # e.g. "E" - overrides the semitones

    # After rebuilding, check the stems still reconcile with the recording:
    # if every instrument played what we claim, does the mix come out the same?
    mix_consistency: bool = True
    mix_tolerance_db: float = 1.5

    # Split a stem by stereo position, when the material actually has any.
    spatial_split: bool = False
    spatial_targets: tuple[str, ...] = ("electric_guitar",)

    # Quality report
    quality_report: bool = True

    # Tempo: "detected" follows the recording's own grid (drift included);
    # "fixed" lays a constant grid, which is what a musician wants when they
    # intend to play to it rather than with the record.
    tempo_mode: Literal["detected", "fixed"] = "detected"
    fixed_bpm: float | None = None  # None = use the detected tempo

    # Click / guide
    make_click: bool = True
    make_guide: bool = True
    make_click_guide_combined: bool = True
    click_count_in_bars: int = 2
    click_subdivision: int = 1  # 1 = quarters, 2 = eighths
    guide_language: Literal["ro", "en"] = "ro"
    guide_lead_beats: float = 4.0  # announce this many beats before the section
    guide_count_in: bool = True

    # Extras
    make_ambient_pad: bool = True
    make_minus_one: bool = True
    make_acapella: bool = True

    # Post
    normalize_stems: bool = False
    target_lufs: float = -18.0
    limiter: bool = False
    output_format: Literal["wav24", "wav16", "flac"] = "wav24"
    sample_rate: int | None = None  # None = keep source rate

    # Export extras
    write_reaper_project: bool = True
    write_session_json: bool = True
    write_markers_csv: bool = True

    def tier(self) -> dict:
        return QUALITY_TIERS[self.quality]
