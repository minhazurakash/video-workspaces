"""Small ffmpeg/ffprobe helpers: probe a clip, stream raw frames in and out."""
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass
class ClipInfo:
    path: str
    width: int    # display width (after rotation)
    height: int
    fps: Fraction
    duration: float
    frames: int   # estimate
    has_audio: bool


def probe(path) -> ClipInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    meta = json.loads(out)
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in meta["streams"])

    w, h = int(v["width"]), int(v["height"])
    rotation = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    if "rotate" in v.get("tags", {}):
        rotation = int(v["tags"]["rotate"])
    if abs(rotation) % 180 == 90:  # ffmpeg autorotates on decode
        w, h = h, w

    fps = Fraction(v.get("avg_frame_rate") or v["r_frame_rate"])
    if fps == 0:
        fps = Fraction(v["r_frame_rate"])
    duration = float(v.get("duration") or meta["format"]["duration"])
    nb = v.get("nb_frames", "")
    frames = int(nb) if nb.isdigit() and int(nb) else round(duration * fps)
    return ClipInfo(str(path), w, h, fps, duration, frames, has_audio)


class FrameReader:
    """Yields HxWxC uint8 frames (rgb24 or gray) from any ffmpeg-readable file."""

    def __init__(self, path, width, height, pix_fmt="rgb24", seconds=None):
        self.w, self.h = width, height
        self.ch = 1 if pix_fmt == "gray" else 3
        cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
        if seconds:
            cmd += ["-t", str(seconds)]
        cmd += ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self.size = width * height * self.ch

    def __iter__(self):
        while True:
            buf = self.proc.stdout.read(self.size)
            if len(buf) < self.size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, self.ch)
            yield frame[..., 0] if self.ch == 1 else frame
        self.proc.stdout.close()
        self.proc.wait()


class FrameWriter:
    """Pipes raw frames into ffmpeg. `out_args` are the encoder/output options."""

    def __init__(self, out_path, width, height, fps, out_args, pix_fmt="rgb24", extra_inputs=()):
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-"]
        for inp in extra_inputs:
            cmd += inp
        cmd += list(out_args) + [str(out_path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")


# Lossless intermediates for the matte cache.
FFV1 = ["-c:v", "ffv1", "-level", "3"]

# Final delivery: H.264, BT.709 tagged.
H264 = ["-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-movflags", "+faststart"]
