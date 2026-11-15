---
name: bg-replace
description: Replace the background of raw video clips with a user-supplied image using RobustVideoMatting + ffmpeg, made to look real (light/colour match, light wrap, depth-of-field). Use whenever the user gives clips + a background image, asks to remove/replace/change a video background, or asks to re-render / tweak an existing project in projects/.
---

# Background replace (RVM + ffmpeg)

Everything runs from the workspace root (the folder that contains CLAUDE.md).
CPU only (no NVIDIA GPU): measured @1080p: mobilenetv3 ≈ 9 fps, resnet50 ≈ 4.7 fps matting, compositing ≈ 8 fps (1 min 1080p30 final ≈ 10–12 min). ProRes alpha output is big (~70 MB/s of footage). Run long renders with `run_in_background`.

## 0. Setup (once)
If `.venv/bin/python` or `models/rvm_resnet50.pth` is missing: `tools/setup.sh` (idempotent).

## 1. New project — one folder per task
```bash
tools/new_project.sh "Short title"      # prints projects/NNN-short-title
```
Copy the user's files in (never move/modify originals):
- clips → `projects/NNN-*/input/clips/` (several clips allowed; each renders separately)
- background image → `projects/NNN-*/input/bg/` (first image is used; set `"background"` in project.json to pick one)
Per-clip different BG/settings: `"clip_overrides": {"clip2.mp4": {"background": "input/bg/b.jpg", "bg": {"blur": 3}}}`.

## 2. Preview first (mobilenetv3, first `preview_seconds`)
```bash
.venv/bin/python tools/run_project.py projects/NNN-name --preview
```
Then **look at** `output/<clip>_preview_before_after.png` (Read tool) and judge: edges, hair, colour/brightness match, BG framing, scale/perspective of the subject against the BG. Show the user the PNG path and say what you'd tweak.

## 3. Tune `project.json` and re-run preview
Matte is cached in `work/`, so re-runs after a look change only redo compositing (fast).

| Symptom | Knob |
|---|---|
| Subject looks pasted / wrong brightness | `look.light_match` ↑ (0.35 → 0.5–0.7) |
| Skin/clothes colour doesn't fit BG light | `look.color_match` ↑ (keep ≤ 0.6 to protect skin) |
| Subject colours look weird/tinted | `look.color_match` ↓ (0.1–0.2) |
| Hard cut-out edge | `look.edge_soften` ↑ (1.5), `look.light_wrap` ↑ (0.35) |
| Glowing halo around subject | `look.light_wrap` ↓ (0.1), `look.light_wrap_radius` ↓ |
| BG too sharp vs subject (fake) | `bg.blur` ↑ (8–14); if subject is soft/out of focus go higher |
| BG too blurry | `bg.blur` ↓ (0–3) |
| Wrong part of BG visible | `bg.offset_x` / `bg.offset_y` (0..1), `bg.zoom` (>1 crops in) |
| Clean digital subject on noisy BG (or vice versa) | `look.grain` 0.3–0.6 |
| Edges shimmer/flicker frame to frame | default `temporal_smooth` r=3, σ=0.2. r=4/σ=0.25 is only slightly better and starts leaving faint ghost lines on fast hands. `radius: 0` disables |
| Matte misses thin parts / flickers on 4K | set `downsample_ratio` (default auto ≈ 512px long side; try 0.375 @1080p) |

`temporal_smooth` (tools/stabilize.py) is on by default: flow-aligned robust average of the raw RVM alpha over ±radius frames, ~5 fps @720p with r=3, cached in `work/`. Measured on the talking-head test: edge flicker −27%, no ghosting.

Tried and rejected (don't redo): recursive previous-output blend (ghost trails); changing `downsample_ratio` (0.25/0.4/0.6 gave identical flicker); bigger RVM model alone (resnet50 flickers as much as mobilenet); reverse-pass memory warm-up at clip start (no gain).

Fades from/to black: RVM can't matte near-black fade frames, and the BG is not faded, so the subject shows as a dark silhouette. Tell the user to trim fades from the raw clip (or re-add the fade after compositing).

Model note: MatAnyone 2 (github.com/pq-yang/MatAnyone2) has much better temporal stability than RVM but its NTU S-Lab license is **non-commercial only** and the code targets CUDA. Only add it if the user confirms the work is non-commercial.

Realism is also about the shot: if the camera height / perspective / light direction in the clip doesn't match the BG photo, tell the user — no grading fixes that; suggest a better-matching BG or `bg.zoom`/offset to align the horizon.

## 4. Final render (resnet50, full clip)
```bash
.venv/bin/python tools/run_project.py projects/NNN-name --final   # add --clip NAME for one clip
```
Outputs in `projects/NNN-*/output/`:
- `<clip>_final.mp4` — H.264 CRF18 BT.709, original audio (AAC)
- `<clip>_fg_alpha.mov` — ProRes 4444 with alpha (raw subject, no grading) for Premiere/DaVinci
- `<clip>_final_before_after.png` — still check

Verify with `ffprobe` that duration/fps/resolution match the source, then report output paths. `projects/INDEX.md` is regenerated automatically (`tools/run_project.py --reindex` to force).

## 5. Cleanup
`work/` holds lossless mattes (large). Delete `projects/NNN-*/work/` once the user is happy — ask first.
