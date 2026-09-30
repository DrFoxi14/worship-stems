# Worship Stems

Offline-first tooling for turning stereo recordings into rehearsal-ready multitrack packages.

This repository explores practical music-production tooling around stem separation, beat and section analysis, rehearsal clicks, Guide tracks, MIDI transcription and verification.

## What it does

- Separates a mix into instrument-oriented stems.
- Builds click and spoken Guide tracks from detected structure.
- Exports rehearsal assets and DAW metadata.
- Supports exact reconstruction checks alongside optional restored outputs.
- Tests whether proposed instruments are actually supported by evidence in the recording.
- Keeps the processing workflow local after required model weights are installed.

## Repository structure

```text
worship-stems-repo/   current runnable implementation
worship-stems-complet/ experimental / extended implementation
archive/              older snapshots
```

Start with [`worship-stems-repo/`](worship-stems-repo/) for the documented installation and usage path.

## Quick start

```bash
cd worship-stems-repo
./install_macos.sh
./run.sh
```

See the implementation README for CLI examples, outputs, design notes and benchmark methodology.

## Verification philosophy

The project distinguishes between three different things: what can be allocated directly from the recording, what can be reconstructed from additional evidence, and what should be rejected as unsupported.

That distinction is intentional. A system that produces a plausible stem is not necessarily a system that recovered a real source.

## Status

Research / prototype. The repository contains active experiments and older snapshots, so implementation details may change.

## License

No open-source license has been declared yet. Until one is added, the code should not be assumed to be licensed for reuse.

Built by Paul.