"""Flow-guided temporal smoothing of an alpha matte (kills edge shimmer/flicker).

For frame t, the *raw* alphas of frames t-k..t+k are warped onto t with DIS optical
flow (computed on the source video) and averaged with robust per-pixel weights:
  - photometric weight: low where the warp is unreliable (occlusion, fast motion)
  - range weight:       low where the warped alpha differs a lot from frame t
Flicker is small frame-to-frame jitter, so it averages out; a real change or a bad
flow vector is a large difference and gets rejected. Non-recursive, so errors can't
accumulate into ghost trails.
"""
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from vio import FFV1, FrameReader, FrameWriter, probe


def run_stabilize(clip, pha_path, out_path, n_frames, radius=3, range_sigma=0.2, seconds=None):
    out_path = Path(out_path)
    done = out_path.with_suffix(".json")
    if done.exists():
        return out_path
    info = probe(clip)
    W, H = info.width, info.height
    # Flow at <=720p is plenty for a matte and keeps 1080p/4K fast.
    fs = min(1.0, 720 / H)
    fw, fh = round(W * fs), round(H * fs)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gx, gy = np.meshgrid(np.arange(fw, dtype=np.float32), np.arange(fh, dtype=np.float32))

    def smooth(i, frames):
        g_t, a_t = next((g, a) for j, g, a in frames if j == i)
        a_small = cv2.resize(a_t, (fw, fh), interpolation=cv2.INTER_AREA) if fs != 1 else a_t
        acc, wsum = a_small.copy(), np.ones_like(a_small)
        for j, g_s, a_s in frames:
            if j == i or abs(j - i) > radius:
                continue
            f = dis.calc(g_t, g_s, None)
            mx, my = gx + f[..., 0], gy + f[..., 1]
            a_s_small = cv2.resize(a_s, (fw, fh), interpolation=cv2.INTER_AREA) if fs != 1 else a_s
            warped = cv2.remap(a_s_small, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            err = np.abs(g_t.astype(np.float32) - cv2.remap(g_s, mx, my, cv2.INTER_LINEAR).astype(np.float32))
            w = np.exp(-(cv2.GaussianBlur(err, (0, 0), 1.5) / 12.0) ** 2)
            w *= np.exp(-((warped - a_small) / range_sigma) ** 2)
            w *= 1.0 / (1 + abs(j - i))  # nearer frames count more
            acc += w * warped
            wsum += w
        if fs == 1:
            return acc / wsum
        # Only the correction is upscaled, so full-res edge detail of frame t is kept.
        delta = cv2.resize(acc / wsum - a_small, (W, H), interpolation=cv2.INTER_LINEAR)
        return a_t + delta

    writer = FrameWriter(out_path, W, H, info.fps, FFV1, pix_fmt="gray")
    frames = deque()  # (index, gray, alpha) for a sliding window of 2*radius+1 frames
    nxt = 0

    def emit_ready(last_read, final=False):
        nonlocal nxt
        while nxt <= last_read and (final or nxt + radius <= last_read):
            out = smooth(nxt, frames)
            writer.write((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
            nxt += 1
            while frames and frames[0][0] < nxt - radius:
                frames.popleft()

    pairs = zip(FrameReader(clip, W, H, seconds=seconds), FrameReader(pha_path, W, H, "gray"))
    idx = -1
    for idx, (src, pha) in enumerate(tqdm(pairs, total=n_frames, desc="stabilize", unit="f")):
        small = cv2.resize(src, (fw, fh), interpolation=cv2.INTER_AREA) if fs != 1 else src
        frames.append((idx, cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), pha.astype(np.float32) / 255))
        emit_ready(idx)
    emit_ready(idx, final=True)
    writer.close()
    done.write_text(json.dumps({"radius": radius, "range_sigma": range_sigma}))
    return out_path
