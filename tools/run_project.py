"""Run the background-replace pipeline for one project folder.

  .venv/bin/python tools/run_project.py projects/001-name --preview   # mobilenet, first N sec
  .venv/bin/python tools/run_project.py projects/001-name --final     # resnet50, full clip
  .venv/bin/python tools/run_project.py --reindex                     # rebuild projects/INDEX.md
Options: --clip NAME (only that clip), --model NAME (override)
"""
import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composite import Compositor, fg_stats  # noqa: E402
from matte import run_matte  # noqa: E402
from stabilize import run_stabilize  # noqa: E402
from vio import H264, FrameReader, FrameWriter, probe  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "projects"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out


def load_cfg(pdir):
    return json.loads((pdir / "project.json").read_text())


def save_cfg(pdir, cfg):
    (pdir / "project.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")


def pick_background(pdir, cfg):
    if cfg.get("background"):
        return pdir / cfg["background"]
    imgs = sorted(p for p in (pdir / "input" / "bg").iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not imgs:
        sys.exit(f"no background image in {pdir / 'input' / 'bg'}")
    return imgs[0]


def render_clip(pdir, clip, cfg, mode, model_override=None):
    ccfg = deep_merge(cfg, cfg.get("clip_overrides", {}).get(clip.name, {}))
    info = probe(clip)
    preview = mode == "preview"
    seconds = ccfg.get("preview_seconds", 5) if preview else None
    model = model_override or ccfg["models"][mode]
    stem = clip.stem
    work = pdir / "work" / stem / (f"{model}_{seconds}s" if seconds else model)
    out_dir = pdir / "output"
    out_dir.mkdir(exist_ok=True)

    print(f"\n== {clip.name}  {info.width}x{info.height} @ {float(info.fps):.2f}fps  "
          f"{info.duration:.1f}s  [{mode}/{model}]")
    meta = run_matte(clip, work, model, seconds, ccfg.get("downsample_ratio"))
    n = meta["frames"]

    bg_path = pick_background(pdir, ccfg)
    bg = cv2.cvtColor(cv2.imread(str(bg_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    fgr_p, pha_p = work / "fgr.mkv", work / "pha.mkv"
    ts = ccfg.get("temporal_smooth", {})
    if ts.get("radius", 0) > 0:
        pha_p = run_stabilize(clip, pha_p, work / f"pha_stable_r{ts['radius']}_s{ts.get('range_sigma', 0.2)}.mkv",
                              n, ts["radius"], ts.get("range_sigma", 0.2), seconds)
    stats = fg_stats(fgr_p, pha_p, info.width, info.height, n)
    comp = Compositor(bg, info.width, info.height, ccfg["look"], ccfg["bg"], stats)

    final_mp4 = out_dir / f"{stem}_{'preview' if preview else 'final'}.mp4"
    audio_in = [["-i", str(clip)]] if info.has_audio else []
    audio_args = (["-map", "0:v", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
                  if info.has_audio else [])
    if seconds and info.has_audio:
        audio_args += ["-t", str(seconds)]
    enc = H264 + ["-preset", "medium" if preview else "slow"] + audio_args
    writer = FrameWriter(final_mp4, info.width, info.height, info.fps, enc, extra_inputs=audio_in)

    srcs = FrameReader(clip, info.width, info.height, seconds=seconds)
    still_at, still = n // 2, None
    pairs = zip(srcs, FrameReader(fgr_p, info.width, info.height), FrameReader(pha_p, info.width, info.height, "gray"))
    for i, (src, fgr, pha) in enumerate(tqdm(pairs, total=n, desc="composite", unit="f")):
        frame = comp(fgr, pha)
        writer.write(frame)
        if i == still_at:
            still = np.concatenate([src, frame], axis=1)
    writer.close()

    outputs = {"video": final_mp4.relative_to(pdir).as_posix()}
    if still is not None:
        png = out_dir / f"{stem}_{'preview' if preview else 'final'}_before_after.png"
        cv2.imwrite(str(png), cv2.cvtColor(still, cv2.COLOR_RGB2BGR))
        outputs["before_after"] = png.relative_to(pdir).as_posix()

    if not preview and ccfg.get("outputs", {}).get("alpha_mov", True):
        mov = out_dir / f"{stem}_fg_alpha.mov"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(fgr_p), "-i", str(pha_p),
                        "-filter_complex", "[1:v]format=gray[a];[0:v][a]alphamerge,format=yuva444p10le",
                        "-c:v", "prores_ks", "-profile:v", "4444", "-vendor", "apl0", str(mov)], check=True)
        outputs["alpha_mov"] = mov.relative_to(pdir).as_posix()

    return {"mode": mode, "model": model, "frames": n, "matte_fps": meta["fps_processed"],
            "rendered": datetime.now().isoformat(timespec="seconds"), "outputs": outputs}


def reindex():
    rows = []
    for pj in sorted(PROJECTS.glob("*/project.json")):
        cfg = json.loads(pj.read_text())
        pdir = pj.parent
        clips = cfg.get("clips", {})
        outs = [f"[{Path(o).name}]({pdir.name}/{o})" for c in clips.values()
                for k, o in c.get("outputs", {}).items() if k == "video"]
        rows.append(f"| {pdir.name} | {cfg.get('title', '')} | {cfg.get('created', '')} | "
                    f"{cfg.get('status', 'new')} | {len(clips)} | {', '.join(outs) or '-'} |")
    text = ("# Projects\n\nAuto-generated by `tools/run_project.py --reindex` — don't edit by hand.\n\n"
            "| ID | Title | Created | Status | Clips rendered | Outputs |\n|---|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n")
    (PROJECTS / "INDEX.md").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project", nargs="?")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--preview", action="store_true")
    g.add_argument("--final", action="store_true")
    ap.add_argument("--clip")
    ap.add_argument("--model", choices=["mobilenetv3", "resnet50"])
    ap.add_argument("--reindex", action="store_true")
    a = ap.parse_args()

    if a.reindex or not a.project:
        reindex()
        return

    pdir = Path(a.project).resolve()
    cfg = load_cfg(pdir)
    mode = "final" if a.final else "preview"
    clips = sorted(p for p in (pdir / "input" / "clips").iterdir() if p.suffix.lower() in VIDEO_EXT)
    if a.clip:
        clips = [c for c in clips if c.name == a.clip]
    if not clips:
        sys.exit(f"no clips in {pdir / 'input' / 'clips'}")

    cfg["status"] = f"{mode}-running"
    save_cfg(pdir, cfg)
    reindex()
    for clip in clips:
        result = render_clip(pdir, clip, cfg, mode, a.model)
        cfg = load_cfg(pdir)  # re-read: settings may be tweaked while rendering
        cfg.setdefault("clips", {}).setdefault(clip.name, {})[mode] = result
        cfg["clips"][clip.name]["outputs"] = {**cfg["clips"][clip.name].get("outputs", {}), **result["outputs"]}
        save_cfg(pdir, cfg)
        print(f"   -> {', '.join(result['outputs'].values())}")
    cfg["status"] = "done" if mode == "final" else "preview-ready"
    save_cfg(pdir, cfg)
    reindex()


if __name__ == "__main__":
    main()
