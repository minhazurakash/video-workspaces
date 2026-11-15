# Video workspace

Reusable workspace for video background replacement (RobustVideoMatting + ffmpeg).

- Common skill: `.claude/skills/bg-replace/SKILL.md` — follow it for every clip + background task.
- Shared code: `tools/` (`setup.sh`, `new_project.sh`, `run_project.py`, `matte.py`, `composite.py`, `vio.py`). Improve these instead of writing one-off scripts inside a project.
- Every task gets its own `projects/NNN-slug/` (`input/clips`, `input/bg`, `work`, `output`, `project.json`). `projects/INDEX.md` lists all of them and is auto-generated.
- Python: always `.venv/bin/python` (3.12, CPU torch). Do not use the system python 3.14.
- `vendor/`, `models/`, `.venv/` are created by `tools/setup.sh`; don't edit vendor code.
- Never overwrite or move the user's original files; copy them into `input/`.
- User talks in Banglish; reply the same way.
