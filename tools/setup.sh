#!/usr/bin/env bash
# One-time setup: Python venv (CPU torch), RVM source, RVM weights.
# Safe to re-run; skips anything already present.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found on PATH"; exit 1; }

if ! command -v uv >/dev/null; then
  echo "[setup] installing uv via brew"
  brew install uv
fi

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r tools/requirements.txt

if [ ! -d vendor/RobustVideoMatting ]; then
  git clone --depth 1 https://github.com/PeterL1n/RobustVideoMatting vendor/RobustVideoMatting
fi

REL=https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0
for m in mobilenetv3 resnet50; do
  f="models/rvm_${m}.pth"
  [ -s "$f" ] || curl -fL --retry 3 -o "$f" "$REL/rvm_${m}.pth"
done

.venv/bin/python - <<'PY'
import sys, torch
sys.path.insert(0, "vendor/RobustVideoMatting")
from model import MattingNetwork
for m in ("mobilenetv3", "resnet50"):
    net = MattingNetwork(m)
    net.load_state_dict(torch.load(f"models/rvm_{m}.pth", map_location="cpu"))
    print(f"[setup] {m} OK")
print("[setup] torch", torch.__version__, "threads", torch.get_num_threads())
PY
echo "[setup] done"
