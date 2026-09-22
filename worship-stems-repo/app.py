#!/usr/bin/env python3
"""Worship Stems - start the local app, or run it from the command line.

    python app.py                          # open the UI in a browser
    python app.py song.mp3                 # separate one mix
    python app.py folder/ --batch          # every audio file in a folder
    python app.py channels/ --multitrack   # real console stems, no separation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".aiff", ".aif"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate worship multitracks from a stereo mix.")
    ap.add_argument("input", nargs="?", help="audio file or folder; omit to open the UI")
    ap.add_argument("-o", "--output", default=str(Path.home() / "WorshipStems" / "output"))
    ap.add_argument("--batch", action="store_true", help="process every audio file in the folder")
    ap.add_argument("--quality", choices=["fast", "standard", "maximum"], default="maximum")
    ap.add_argument("--vocal-preset", choices=["balanced", "clean", "full"], default="balanced")
    ap.add_argument("--lang", choices=["ro", "en"], default="ro", help="guide track language")
    ap.add_argument("--format", choices=["wav24", "wav16", "flac"], default="wav24")
    ap.add_argument("--partition", choices=["exact", "raw"], default="exact")
    ap.add_argument("--count-in", type=int, default=2, help="count-in bars")
    ap.add_argument("--no-click", action="store_true")
    ap.add_argument("--no-guide", action="store_true")
    ap.add_argument("--no-pad", action="store_true")
    ap.add_argument("--drumsep", action="store_true",
                    help="split the kit into kick/snare/toms/hats (slow, rarely needed)")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--tempo", type=float, default=None, help="force a steady BPM instead of following the recording")
    ap.add_argument("--no-gate", action="store_true", help="export every stem, even instruments that are not in the song")
    ap.add_argument("--no-refine", action="store_true", help="skip the second sharpening pass")
    ap.add_argument("--spatial", action="store_true", help="also split guitars by stereo position, where the mix allows it")
    ap.add_argument("--no-restore", action="store_true", help="skip rebuilding the holes the cut leaves")
    ap.add_argument("--both-chains", action="store_true", help="export the exact stems AND the rebuilt ones, for A/B")
    ap.add_argument("--no-midi", action="store_true", help="skip transcription and the note-informed repair")
    ap.add_argument("--multitrack", action="store_true",
                    help="the input is a folder of console channels, not a mix - no separation needed")
    ap.add_argument("--no-repeats", action="store_true", help="do not borrow from other instances of a section")
    ap.add_argument("--no-resynth", action="store_true", help="skip rebuilding each instrument from its score")
    ap.add_argument("--play-parts", action="store_true",
                    help="also render each instrument from its notes with a clean sound")
    ap.add_argument("--regrid", action="store_true",
                    help="stretch the song onto a steady tempo so the tracks lock to the click")
    ap.add_argument("--regrid-bpm", type=float, default=None, help="tempo to lock to (default: the song's own)")
    ap.add_argument("--transpose", type=int, default=0, help="move the key by N semitones")
    ap.add_argument("--to-key", default=None, help="move the song to this key, e.g. E")
    ap.add_argument("--no-forecast", action="store_true", help="skip the difficulty check")
    ap.add_argument("--doctor", action="store_true", help="check the install and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.doctor:
        return doctor()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    if not args.input:
        from worship_stems.server import serve

        print(f"Worship Stems  ->  http://{args.host}:{args.port}")
        serve(host=args.host, port=args.port, open_browser=not args.no_browser)
        return 0

    from worship_stems.config import Options
    from worship_stems.pipeline import Pipeline
    from worship_stems.separate import SeparationEngine

    source = Path(args.input)

    if args.multitrack:
        if not source.is_dir():
            print(f"{source} is not a folder. --multitrack expects a folder of console channels.")
            return 2
        folders = (
            sorted(d for d in source.iterdir() if d.is_dir()) if args.batch else [source]
        )
        if not folders:
            print("No channel folders found.")
            return 2
        return _run_multitrack(folders, args)

    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in AUDIO_EXT)
        if not args.batch:
            print(f"{source} is a folder; pass --batch to process all {len(files)} files.")
            return 2
    else:
        files = [source]
    if not files:
        print("No audio files found.")
        return 2

    opt = Options(
        quality=args.quality,
        vocal_preset=args.vocal_preset,
        guide_language=args.lang,
        output_format=args.format,
        partition=args.partition,
        click_count_in_bars=args.count_in,
        make_click=not args.no_click,
        make_guide=not args.no_guide,
        make_ambient_pad=not args.no_pad,
        split_drums=args.drumsep,
        normalize_stems=args.normalize,
        gate_instruments=not args.no_gate,
        refine=not args.no_refine,
        spatial_split=args.spatial,
        restore=not args.no_restore,
        restore_chain="both" if args.both_chains else "restored",
        transcribe_midi=not args.no_midi,
        note_informed_restore=not args.no_midi,
        borrow_repeats=not args.no_repeats,
        resynthesize=not args.no_resynth,
        render_parts=args.play_parts,
        regrid=args.regrid,
        regrid_bpm=args.regrid_bpm,
        transpose_semitones=args.transpose,
        transpose_to_key=args.to_key,
        assess_difficulty=not args.no_forecast,
        tempo_mode="fixed" if args.tempo else "detected",
        fixed_bpm=args.tempo,
    )

    engine = SeparationEngine() if SeparationEngine.available() else None
    pipe = Pipeline(opt, engine=engine)

    last = {"line": ""}

    def progress(stage: str, frac: float, msg: str) -> None:
        line = f"  {stage:9s} {frac*100:5.1f}%  {msg}"
        if line != last["line"]:
            print(line, flush=True)
            last["line"] = line

    failures = 0
    for i, f in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] {f.name}")
        try:
            result = pipe.run(f, args.output, progress=progress)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAILED: {exc}")
            continue
        print(f"  -> {result.folder}")
        if result.difficulty:
            d = result.difficulty
            print(f"     forecast: {d['verdict']} ({d['score']:.2f}) - {d['reasons'][0]}")
        for k, v in result.metrics.items():
            print(f"     {k}: {v}")
        for key, v in (result.inventory or {}).items():
            if v["decision"] != "present":
                print(f"     {v['decision']:9s} {v['label']}: {v['reasons'][0]}")
        for key, v in (result.quality or {}).items():
            if v["verdict"] != "good":
                print(f"     {v['verdict']:9s} {v['label']}: {v['notes'][0]}")
        for key, v in (result.restore or {}).items():
            if v["bins_restored"]:
                print(f"     rebuilt   {v['label']}: {v['bins_restored']:,} bins "
                      f"({v['share_restored']*100:.0f}% of the damage), {v['bins_refused']:,} left alone")
        for w in result.warnings:
            print(f"     note: {w}")

    if engine:
        engine.release()
    return 1 if failures else 0


def _run_multitrack(folders, args) -> int:
    from worship_stems.config import Options
    from worship_stems.pipeline import Pipeline

    opt = Options(
        guide_language=args.lang,
        output_format=args.format,
        click_count_in_bars=args.count_in,
        make_click=not args.no_click,
        make_guide=not args.no_guide,
        make_ambient_pad=not args.no_pad,
        normalize_stems=args.normalize,
        transcribe_midi=not args.no_midi,
        tempo_mode="fixed" if args.tempo else "detected",
        fixed_bpm=args.tempo,
        render_parts=args.play_parts,
        regrid=args.regrid,
        regrid_bpm=args.regrid_bpm,
        transpose_semitones=args.transpose,
        transpose_to_key=args.to_key,
        assess_difficulty=False,
    )
    pipe = Pipeline(opt)
    last = {"line": ""}

    def progress(stage: str, frac: float, msg: str) -> None:
        line = f"  {stage:9s} {frac*100:5.1f}%  {msg}"
        if line != last["line"]:
            print(line, flush=True)
            last["line"] = line

    failures = 0
    for i, folder in enumerate(folders, start=1):
        print(f"\n[{i}/{len(folders)}] {folder.name}  (console channels)")
        try:
            result = pipe.run_multitrack(folder, args.output, progress=progress)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAILED: {exc}")
            continue
        print(f"  -> {result.folder}")
        for k, v in result.metrics.items():
            print(f"     {k}: {v}")
        for w in result.warnings:
            print(f"     note: {w}")
    return 1 if failures else 0


def doctor() -> int:
    """Check the install and say exactly what is missing."""
    import shutil

    from worship_stems.guide import available_tts
    from worship_stems.separate import SeparationEngine

    print("Worship Stems - install check\n")
    rows: list[tuple[str, bool, str]] = []

    rows.append(("ffmpeg", shutil.which("ffmpeg") is not None, "needed for MP3"))
    for mod in ("numpy", "scipy", "soundfile", "librosa", "sklearn", "fastapi"):
        try:
            __import__(mod)
            rows.append((mod, True, ""))
        except Exception as exc:  # noqa: BLE001
            rows.append((mod, False, str(exc)))

    err = SeparationEngine.import_error()
    rows.append(("audio-separator", err is None, err or SeparationEngine.describe_device()))

    try:
        from worship_stems.harmonic import transcription_available

        import numpy as _np

        rows.append(("note transcription (MIDI)", transcription_available(),
                     f"numpy {_np.__version__}" if transcription_available() else "optional"))
        if _np.__version__ < "2":
            rows.append(("numpy version", False, "audio-separator needs numpy >= 2; TensorFlow pinned it down"))
    except Exception as exc:  # noqa: BLE001
        rows.append(("note transcription (MIDI)", False, f"optional - {exc}"))

    try:
        import allin1  # noqa: F401

        rows.append(("allin1 (structure)", True, "optional"))
    except Exception:
        rows.append(("allin1 (structure)", False, "optional - built-in analyser will be used"))

    tts = available_tts()
    rows.append(("guide voice", tts is not None, tts or "will fall back to tones"))

    width = max(len(r[0]) for r in rows)
    broken = False
    for name, good, note in rows:
        flag = "OK  " if good else "MISS"
        if not good and "optional" not in note and "fall back" not in note:
            broken = True
        print(f"  {flag}  {name.ljust(width)}  {note}")

    if broken:
        print("\nFix with:  pip install -r requirements.txt --upgrade")
        return 1
    print("\nEverything needed is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
