"""Export: folder layout, session metadata and DAW projects.

The folder is laid out the way a worship team actually receives tracks -
numbered so they sort in playback order, named so nobody has to guess.
A Reaper project is written alongside because it is a plain-text format,
which means the band gets every stem on its own track, at the right tempo,
with section markers already in place, from one double-click.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .analyze import Analysis
from .config import STEMS_BY_KEY


@dataclass
class ExportedFile:
    key: str
    label: str
    path: Path
    peak_db: float
    lufs: float | None = None


def safe_name(text: str) -> str:
    keep = "-_() "
    cleaned = "".join(c if c.isalnum() or c in keep else "_" for c in text).strip()
    return " ".join(cleaned.split()) or "Track"


def song_folder(root: Path, title: str, analysis: Analysis) -> Path:
    name = f"{safe_name(title)} [{analysis.key_display} {round(analysis.bpm)}bpm]"
    folder = root / name
    n = 2
    while folder.exists() and any(folder.iterdir()):
        folder = root / f"{name} ({n})"
        n += 1
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def stem_filename(key: str, title: str) -> str:
    spec = STEMS_BY_KEY.get(key)
    base = spec.filename if spec else key
    return f"{base}"


# --------------------------------------------------------------------------
# Markers
# --------------------------------------------------------------------------


def write_markers_csv(path: Path, analysis: Analysis, offset: float = 0.0) -> Path:
    """Marker list that Reaper, Audition and Ableton can all read."""
    lines = ["Name,Start,End,Length,Type"]
    for s in analysis.sections:
        start, end = s.start + offset, s.end + offset
        lines.append(f'"{s.display}",{start:.3f},{end:.3f},{end - start:.3f},Marker')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_cue_file(path: Path, analysis: Analysis, audio_filename: str, offset: float = 0.0) -> Path:
    """A .cue sheet, for players and for importing section points."""
    def stamp(t: float) -> str:
        t = max(0.0, t)
        m, s = divmod(t, 60)
        frames = int(round((s - math.floor(s)) * 75))
        return f"{int(m):02d}:{int(math.floor(s)):02d}:{min(frames, 74):02d}"

    lines = [f'FILE "{audio_filename}" WAVE']
    for i, s in enumerate(analysis.sections, start=1):
        lines += [f"  TRACK {i:02d} AUDIO", f'    TITLE "{s.display}"', f"    INDEX 01 {stamp(s.start + offset)}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Session metadata
# --------------------------------------------------------------------------


def write_session_json(
    path: Path,
    title: str,
    analysis: Analysis,
    files: list[ExportedFile],
    offset: float,
    cues: list[dict],
    quality: dict,
    metrics: dict,
) -> Path:
    payload = {
        "title": title,
        "generator": "Worship Stems",
        "count_in_offset_seconds": round(offset, 4),
        "analysis": analysis.to_dict(),
        "cues": cues,
        "quality": quality,
        "metrics": metrics,
        "files": [
            {"key": f.key, "label": f.label, "file": f.path.name, "peak_db": round(f.peak_db, 2), "lufs": (round(f.lufs, 2) if f.lufs is not None else None)}
            for f in files
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Reaper project
# --------------------------------------------------------------------------


def _rgb_to_reaper(hex_color: str) -> int:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b << 16) | (g << 8) | r | 0x1000000


def write_reaper_project(
    path: Path,
    title: str,
    analysis: Analysis,
    files: list[ExportedFile],
    offset: float = 0.0,
) -> Path:
    """Write an .RPP that loads every stem on its own track with markers."""
    bpm = analysis.bpm or 120.0
    lines = [
        "<REAPER_PROJECT 0.1 '7.0' 0",
        f"  TEMPO {bpm:.4f} {analysis.beats_per_bar} 4",
        "  SAMPLERATE 44100 0 0",
        "  PLAYRATE 1 0 0.25 4",
        f"  TITLE {json.dumps(title)}",
    ]

    for i, s in enumerate(analysis.sections, start=1):
        lines.append(f'  MARKER {i} {s.start + offset:.6f} "{s.display}" 0 0 1 B {{{i:08d}-0000-0000-0000-000000000000}}')

    for i, f in enumerate(files, start=1):
        spec = STEMS_BY_KEY.get(f.key)
        color = _rgb_to_reaper(spec.color if spec else "#7c8aa5")
        muted = 1 if f.key in {"mix", "recombined", "minus_one", "acapella", "instrumental", "vocals", "drums", "click_guide"} else 0
        lines += [
            "  <TRACK",
            f"    NAME {json.dumps(f.label)}",
            f"    PEAKCOL {color}",
            f"    TRACKID {{{i:08d}-1111-2222-3333-444444444444}}",
            f"    MUTESOLO {muted} 0 0",
            "    VOLPAN 1 0 -1 -1 1",
            "    <ITEM",
            "      POSITION 0",
            f"      LENGTH {analysis.duration + offset:.6f}",
            f"      NAME {json.dumps(f.path.name)}",
            "      <SOURCE WAVE",
            f"        FILE {json.dumps(f.path.name)}",
            "      >",
            "    >",
            "  >",
        ]
    lines.append(">")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Human-readable chart
# --------------------------------------------------------------------------


def write_chart(path: Path, title: str, analysis: Analysis, offset: float, metrics: dict) -> Path:
    def mmss(t: float) -> str:
        m, s = divmod(max(0.0, t), 60)
        return f"{int(m)}:{int(s):02d}"

    bars_per_section = []
    bpb = max(1, analysis.beats_per_bar)
    beat = 60.0 / (analysis.bpm or 120.0)
    for s in analysis.sections:
        bars = max(1, round((s.end - s.start) / (beat * bpb)))
        bars_per_section.append(bars)

    lines = [
        f"# {title}",
        "",
        f"**Key** {analysis.key_display} ({analysis.camelot})  |  **Tempo** {analysis.bpm:.1f} BPM  |  "
        f"**Meter** {bpb}/4  |  **Length** {mmss(analysis.duration)}",
        "",
        f"Count-in: {offset:.2f}s before bar 1. Structure detected by `{analysis.backend}`.",
        "",
        "| # | Section | Start | Bars | Length |",
        "|---|---------|-------|------|--------|",
    ]
    for i, (s, bars) in enumerate(zip(analysis.sections, bars_per_section), start=1):
        lines.append(f"| {i} | {s.display} | {mmss(s.start + offset)} | {bars} | {mmss(s.end - s.start)} |")

    lines += ["", "## Reconstruction check", ""]
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
