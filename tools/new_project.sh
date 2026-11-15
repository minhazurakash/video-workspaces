#!/usr/bin/env bash
# Create projects/NNN-slug/ with input/output folders and a default project.json.
# Usage: tools/new_project.sh "Office interview"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TITLE="${1:?usage: new_project.sh \"title\"}"
SLUG="$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-|-$//g')"
LAST="$(ls "$ROOT/projects" | grep -oE '^[0-9]{3}' | sort -n | tail -1 || true)"
NUM="$(printf '%03d' $(( 10#${LAST:-0} + 1 )))"
DIR="$ROOT/projects/$NUM-${SLUG:-project}"

mkdir -p "$DIR"/{input/clips,input/bg,work,output}
"$ROOT/.venv/bin/python" - "$ROOT/tools/project_template.json" "$DIR/project.json" "$TITLE" <<'PY'
import json, sys, datetime
cfg = json.load(open(sys.argv[1]))
cfg["title"] = sys.argv[3]
cfg["created"] = datetime.date.today().isoformat()
open(sys.argv[2], "w").write(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
PY
"$ROOT/.venv/bin/python" "$ROOT/tools/run_project.py" --reindex
echo "$DIR"
