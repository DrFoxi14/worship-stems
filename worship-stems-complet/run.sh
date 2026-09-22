#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Not installed yet. Run ./install_macos.sh first."
  exit 1
fi
exec .venv/bin/python app.py "$@"
