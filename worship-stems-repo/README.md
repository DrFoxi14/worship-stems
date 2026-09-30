# Worship Stems

Offline-first tooling for turning a stereo recording into a rehearsal-ready multitrack package.

The pipeline can produce separated stems, click and guide tracks, section markers, MIDI, charts and a Reaper project. The design goal is practical: preserve evidence from the recording where possible, make uncertainty explicit, and keep the workflow runnable on the user's machine.

## Highlights

- Local audio processing with no required upload workflow.
- Stem separation with different models selected by stem type.
- Click and spoken Guide tracks derived from the detected beat/section structure.
- Exact and restored output chains, making the difference between allocation and reconstruction explicit.
- Instrument-presence checks that can reject unsupported or uncertain stems instead of silently inventing content.
- MIDI transcription and project metadata for downstream editing.

## Quick start

### macOS / Apple Silicon

```bash
./install_macos.sh
./run.sh
```

Optional structure analysis:

```bash
./install_macos.sh --with-structure
```

Useful commands:

```bash
./run.sh song.mp3
./run.sh ~/Music/worship --batch
./run.sh --doctor
```

The first separation may download model weights. After installation, the core processing path is designed to run locally.

## Output

| Output | Purpose |
|---|---|
| Original mix | Reference source |
| Click / Click + Guide | Rehearsal and monitoring |
| Guide | Section cues |
| Vocal / instrumental stems | Main separation |
| Instrument stems | Bass, drums, guitars, keys, pads |
| Minus One / Acapella | Alternate rehearsal outputs |
| MIDI | Transcribed notes where available |
| CHART.md | Key, tempo and section summary |
| `sections.cue`, `markers.csv`, `tempo_map.json` | DAW/import metadata |
| `*.RPP` | Reaper session |
| `99_Recombined` | Reconstruction check for the exact chain |

## Design

```text
audio
  │
  ├── separation
  │      └── stem-specific models
  │
  ├── mask partitioning
  │      └── exact chain
  │
  ├── refinement
  │      └── sharper allocation / band constraints
  │
  ├── optional restoration
  │      └── reconstructed but separately reported output
  │
  ├── note-aware repair
  │      └── only when evidence supports the hypothesis
  │
  ├── instrument-presence checks
  │      └── reject / flag uncertain outputs
  │
  └── verification
         └── compare outputs against the source
```

A central idea is that **verification is part of the product, not an afterthought**. The exact chain exists specifically so the exported leaves can be recombined and compared with the parent recording.

## Reproducibility

The repository contains the runnable app, installation scripts, dependencies and tests.

```bash
pip install -r requirements.txt
pytest
```

The project currently targets local use on macOS, with the primary installation path tested on Apple Silicon.

## Benchmarks and technical notes

The repository also contains detailed experiments around separation quality, mask renormalisation, repair, note-aware reconstruction, instrument detection and spatial separation.

Those numbers are experimental results from this project, not universal guarantees. When citing a benchmark, use the exact test conditions and implementation version from the repository rather than treating a single score as a general property of the method.

## Scope

This is a research/prototyping project for music production and rehearsal workflows. It is not a claim that arbitrary recordings can be perfectly decomposed into their original multitrack sources.

In particular, the project intentionally refuses to fabricate information that is not recoverable from the stereo input. Repeated sections, spatial cues and note-level evidence can help, but they do not create information that was never recorded.

## License

No open-source license has been declared at the repository root yet. Until one is added, the code should not be assumed to be licensed for reuse.

Built by Paul.