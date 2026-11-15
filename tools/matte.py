"""RVM matting: clip -> lossless foreground (fgr.mkv) + alpha (pha.mkv), cached in work/.

The matte is the slow step on CPU, so it is cached per (clip, model, length);
re-tuning the look in project.json only re-runs composite.py.
"""
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "RobustVideoMatting"))
from model import MattingNetwork  # noqa: E402

from vio import FFV1, FrameReader, FrameWriter, probe  # noqa: E402


def auto_downsample(w, h):
    # RVM works best when the internal resolution is ~512px on the long side.
    return min(1.0, 512 / max(w, h))


def run_matte(clip_path, out_dir, model_name="mobilenetv3", seconds=None, downsample_ratio=None):
    out_dir = Path(out_dir)
    done = out_dir / "matte.json"
    if done.exists():
        return json.loads(done.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe(clip_path)
    ratio = downsample_ratio or auto_downsample(info.width, info.height)

    net = MattingNetwork(model_name).eval()
    net.load_state_dict(torch.load(ROOT / "models" / f"rvm_{model_name}.pth", map_location="cpu"))

    total = round(min(seconds, info.duration) * info.fps) if seconds else info.frames
    reader = FrameReader(clip_path, info.width, info.height, seconds=seconds)
    fgr_w = FrameWriter(out_dir / "fgr.mkv", info.width, info.height, info.fps, FFV1)
    pha_w = FrameWriter(out_dir / "pha.mkv", info.width, info.height, info.fps, FFV1, pix_fmt="gray")

    def to_tensor(frame):
        return torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255)

    rec = [None] * 4
    n, t0 = 0, time.time()
    with torch.inference_mode():
        for frame in tqdm(reader, total=total, desc=f"matte[{model_name}]", unit="f"):
            fgr, pha, *rec = net(to_tensor(frame), *rec, downsample_ratio=ratio)
            fgr_w.write(fgr[0].permute(1, 2, 0).mul(255).round().clamp(0, 255).byte().numpy())
            pha_w.write(pha[0, 0].mul(255).round().clamp(0, 255).byte().numpy())
            n += 1
    fgr_w.close()
    pha_w.close()

    meta = {"clip": str(clip_path), "model": model_name, "seconds": seconds, "frames": n,
            "downsample_ratio": ratio, "fps_processed": round(n / (time.time() - t0), 2)}
    done.write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("out_dir")
    p.add_argument("--model", default="mobilenetv3", choices=["mobilenetv3", "resnet50"])
    p.add_argument("--seconds", type=float)
    p.add_argument("--ratio", type=float)
    a = p.parse_args()
    print(run_matte(a.clip, a.out_dir, a.model, a.seconds, a.ratio))
