#!/usr/bin/env bash
# Worship Stems - installer for macOS (Apple Silicon).
set -euo pipefail

cd "$(dirname "$0")"
BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; OFF=$'\033[0m'
say(){ printf "%s\n" "${BOLD}$1${OFF}"; }
note(){ printf "%s\n" "${DIM}$1${OFF}"; }

WITH_STRUCTURE=0
for arg in "$@"; do
  case "$arg" in
    --with-structure) WITH_STRUCTURE=1 ;;
    -h|--help) echo "usage: ./install_macos.sh [--with-structure]"; exit 0 ;;
  esac
done

say "Worship Stems - installing"

# --- Python ----------------------------------------------------------------
PY=""
for c in python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
    major=${v%%.*}; minor=${v##*.}
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 13 ]; then PY="$c"; break; fi
  fi
done
if [ -z "$PY" ]; then
  printf "%s\n" "${RED}Need Python 3.10-3.13.${OFF}"
  note "Install it with:  brew install python@3.12"
  exit 1
fi
note "Using $PY ($($PY -c 'import sys;print(sys.version.split()[0])'))"

# --- ffmpeg ----------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  printf "%s\n" "${YELLOW}ffmpeg is missing - MP3 support needs it.${OFF}"
  if command -v brew >/dev/null 2>&1; then
    say "Installing ffmpeg with Homebrew"
    brew install ffmpeg
  else
    note "Install Homebrew from https://brew.sh then run: brew install ffmpeg"
    exit 1
  fi
fi

# --- venv ------------------------------------------------------------------
say "Creating the virtual environment"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel

say "Installing the app (this downloads PyTorch - a few minutes)"
pip install --quiet -r requirements.txt

# basic-pitch, without its dependency list. Left to itself it installs
# TensorFlow, which pins numpy below 2 and breaks audio-separator. The model
# it needs is bundled as ONNX and onnxruntime is already installed above.
say "Installing note transcription (MIDI)"
if ! pip install --quiet --no-deps basic-pitch; then
  printf "%s\n" "${YELLOW}basic-pitch did not install. MIDI and the note-informed repair will be skipped.${OFF}"
fi

if [ "$WITH_STRUCTURE" -eq 1 ]; then
  say "Installing the structure analyser (allin1)"
  note "This is optional. It gives much better verse/chorus detection for the Guide track."
  if ! pip install --quiet natten allin1; then
    printf "%s\n" "${YELLOW}allin1 did not install. The app still works with the built-in analyser.${OFF}"
  fi
fi

# --- check -----------------------------------------------------------------
say "Checking the setup"
python - <<'PY'
import importlib, shutil, sys
def ok(label, good, extra=""):
    print(f"  {'OK  ' if good else 'MISS'}  {label}{('  ' + extra) if extra else ''}")
    return good

ok("ffmpeg", shutil.which("ffmpeg") is not None)
ok("numpy / scipy / librosa", all(importlib.util.find_spec(m) for m in ("numpy", "scipy", "librosa")))
ok("web UI (fastapi)", importlib.util.find_spec("fastapi") is not None)

# Import it for real: find_spec passes even when a transitive dependency
# (audioread, pydub...) is missing, and then separation fails at run time.
sep = True
try:
    from audio_separator.separator import Separator  # noqa: F401
    ok("separation models (audio-separator)", True)
except Exception as exc:
    sep = False
    ok("separation models (audio-separator)", False, f"{type(exc).__name__}: {exc}")
    print("        fix:  pip install -r requirements.txt --upgrade")

if sep:
    try:
        import torch
        dev = "Apple Silicon GPU (MPS)" if torch.backends.mps.is_available() else "CPU"
        ok("acceleration", True, dev)
    except Exception as exc:
        ok("acceleration", False, str(exc))

try:
    from basic_pitch.inference import Model  # noqa: F401
    import onnxruntime  # noqa: F401
    import numpy as _np
    ok("note transcription (MIDI)", True, f"numpy {_np.__version__}")
    if _np.__version__ < "2":
        print("        warning: numpy < 2 - audio-separator needs numpy >= 2")
except Exception as exc:
    ok("note transcription (MIDI)", False, f"optional - {type(exc).__name__}")

ok("structure analyser (allin1)", importlib.util.find_spec("allin1") is not None, "optional")
ok("guide voice (macOS say)", shutil.which("say") is not None)
PY

cat <<EOF

${GREEN}Done.${OFF}

  Start the app:      ${BOLD}./run.sh${OFF}
  One file, no UI:    ${BOLD}.venv/bin/python app.py song.mp3${OFF}
  A whole folder:     ${BOLD}.venv/bin/python app.py ~/Music/worship --batch${OFF}

The first run downloads the separation models (about 2 GB). After that it
works with no internet at all.
EOF
