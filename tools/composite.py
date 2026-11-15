"""Realistic compositing of an RVM matte over a background image.

Steps per frame (all float32 RGB 0..1):
  bg      : cover-fit + depth-of-field blur (prepared once, the BG is static)
  fgr     : LAB light/colour match toward the BG (stats computed once per clip -> no flicker)
  alpha   : contrast cleanup + soft feather
  comp    : fgr*a + bg*(1-a)
  wrap    : wide-blurred BG screened onto the inner edge of the subject
  grain   : optional luma noise over the whole frame
"""
import cv2
import numpy as np

from vio import FrameReader


def fit_background(img_rgb, w, h, fit="cover", offset_x=0.5, offset_y=0.5, zoom=1.0):
    ih, iw = img_rgb.shape[:2]
    scale = (max if fit == "cover" else min)(w / iw, h / ih) * zoom
    nw, nh = round(iw * scale), round(ih * scale)
    if fit == "cover":
        nw, nh = max(w, nw), max(h, nh)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if fit == "cover":
        x = round((nw - w) * offset_x)
        y = round((nh - h) * offset_y)
        return resized[y:y + h, x:x + w]
    canvas = np.zeros((h, w, 3), resized.dtype)
    x, y = round((w - nw) * offset_x), round((h - nh) * offset_y)
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img


def lab_stats(rgb, mask):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask]
    return lab.mean(0), lab.std(0) + 1e-6


def fg_stats(fgr_path, pha_path, w, h, n_frames, samples=24):
    """Mean/std of the subject in LAB, sampled across the clip where alpha is solid."""
    labs = []
    step = max(1, n_frames // samples)
    pairs = zip(FrameReader(fgr_path, w, h), FrameReader(pha_path, w, h, "gray"))
    for i, (fgr, pha) in enumerate(pairs):
        if i % step:
            continue
        m = pha > 230
        if m.sum() > 500:
            labs.append(cv2.cvtColor(fgr.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[m])
    if not labs:
        return None
    lab = np.concatenate(labs)
    return lab.mean(0), lab.std(0) + 1e-6


class Compositor:
    def __init__(self, bg_rgb_u8, w, h, look, bg_cfg, fg_lab_stats):
        s = h / 1080  # scale pixel radii to resolution
        self.s, self.look = s, look
        bg = fit_background(bg_rgb_u8, w, h, bg_cfg.get("fit", "cover"), bg_cfg.get("offset_x", 0.5),
                            bg_cfg.get("offset_y", 0.5), bg_cfg.get("zoom", 1.0)).astype(np.float32) / 255
        self.bg = blur(bg, bg_cfg.get("blur", 6) * s)
        self.wrap_src = blur(self.bg, look.get("light_wrap_radius", 30) * s)

        self.color_xform = None
        if fg_lab_stats is not None and (look.get("color_match", 0) or look.get("light_match", 0)):
            mu_f, sd_f = fg_lab_stats
            mu_b, sd_b = lab_stats(self.bg, np.ones(self.bg.shape[:2], bool))
            k = np.array([look.get("light_match", 0), look.get("color_match", 0), look.get("color_match", 0)],
                         np.float32)
            # Partial transfer: shift means toward the BG and nudge contrast; ratios clamped so
            # a very flat/contrasty BG cannot wreck skin tones.
            ratio = np.clip(sd_b / sd_f, 0.75, 1.3)
            gain = 1 + k * (ratio - 1)
            gain[1:] = 1 + k[1:] * 0.5 * (ratio[1:] - 1)
            shift = k * (mu_b - mu_f)
            shift[1:] *= 0.5  # chroma moves less than luminance; keeps skin natural
            self.color_xform = (mu_f.astype(np.float32), gain.astype(np.float32), shift.astype(np.float32))
        self.rng = np.random.default_rng(0)

    def match(self, fgr):
        if self.color_xform is None:
            return fgr
        mu, gain, shift = self.color_xform
        lab = cv2.cvtColor(fgr, cv2.COLOR_RGB2LAB)
        lab = (lab - mu) * gain + mu + shift
        return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)

    def alpha(self, pha_u8):
        a = pha_u8.astype(np.float32) / 255
        a = np.clip((a - 0.03) / 0.94, 0, 1)  # kill faint haze, make the core solid
        soften = self.look.get("edge_soften", 1.0)
        return blur(a, 0.7 * soften * self.s) if soften > 0 else a

    def __call__(self, fgr_u8, pha_u8):
        fgr = self.match(fgr_u8.astype(np.float32) / 255)
        a = self.alpha(pha_u8)[..., None]
        out = fgr * a + self.bg * (1 - a)

        lw = self.look.get("light_wrap", 0)
        if lw > 0:
            r = self.look.get("light_wrap_radius", 30) * self.s * 0.4
            band = blur(1 - a[..., 0], r)[..., None] * a  # inner edge of the subject
            screen = 1 - (1 - out) * (1 - self.wrap_src)
            out = out + (screen - out) * band * lw

        g = self.look.get("grain", 0)
        if g > 0:
            noise = self.rng.standard_normal(out.shape[:2], np.float32)[..., None] * (0.03 * g)
            out = out + noise

        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
