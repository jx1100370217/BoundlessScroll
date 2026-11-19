#!/bin/sh
set -eu

project=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
core=${COMFYUI_DIR:-"$project/../comfyui-minimax-h3"}
if [ ! -f "$core/main.py" ]; then
  printf '%s\n' 'Set COMFYUI_DIR to a ComfyUI checkout that supports MiniMax H3.' >&2
  exit 1
fi
core=$(CDPATH= cd -- "$core" && pwd)
models=${COMFYUI_MODELS_DIR:-"$core/models"}
if [ ! -d "$models" ]; then
  printf 'Model directory not found: %s\n' "$models" >&2
  exit 1
fi
models=$(CDPATH= cd -- "$models" && pwd)

if [ -n "${COMFYUI_PYTHON:-}" ]; then
  python=$COMFYUI_PYTHON
elif [ -x "$project/runtime/node-venv/bin/python" ]; then
  python="$project/runtime/node-venv/bin/python"
elif [ -x "$core/.venv/bin/python" ]; then
  python="$core/.venv/bin/python"
else
  python=python3
fi

instance="$project/runtime/comfy"
mkdir -p "$instance"
cd "$instance"
export PYTHONDONTWRITEBYTECODE=1
exec "$python" "$core/main.py" \
  --base-directory "$instance" --models-directory "$models" \
  --listen 127.0.0.1 --port "${COMFYUI_PORT:-8189}" --disable-auto-launch "$@"
