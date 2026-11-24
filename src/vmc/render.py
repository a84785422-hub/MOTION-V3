"""Frame renderer: camera warp, mip-style prefiltering and temporal supersampled motion blur.

Pipeline for one output frame
-----------------------------
1. The decoded frame is converted to float32, optionally in linear light.
2. The camera is sampled at several instants spread over the shutter interval. The number of
   samples adapts to how many output pixels the picture moves during the shutter, so slow moves
   cost almost nothing and fast moves stay smooth.
3. Each sample is warped with a Lanczos kernel. When the picture is minified, an area-filtered
   copy of the source (exact target scale) is used first, which avoids aliasing/shimmer.
4. Samples are averaged, converted back to the display transfer curve and quantised to 8 or
   16 bits (with optional stochastic dithering to prevent banding).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .camera import CameraState, CameraTrack, matrix_scale, view_matrix

QUALITIES = ("draft", "high", "ultra")


@dataclass(frozen=True)
class QualityProfile:
    interpolation: int        # kernel for sharp (single or few sample) frames
    blur_interpolation: int   # cheaper kernel once many samples are averaged
    sharp_max_samples: int    # up to this many samples, ``interpolation`` is still used
    step_px: float            # max distance (output px) between two blur samples
    max_samples: int


# Averaging many sub-pixel shifted samples hides the difference between Lanczos and bicubic,
# and bicubic is about four times cheaper, so dense blur uses it.
QUALITY_PROFILES = {
    "draft": QualityProfile(cv2.INTER_LINEAR, cv2.INTER_LINEAR, 0, 2.0, 8),
    "high": QualityProfile(cv2.INTER_LANCZOS4, cv2.INTER_CUBIC, 3, 1.0, 32),
    "ultra": QualityProfile(cv2.INTER_LANCZOS4, cv2.INTER_CUBIC, 8, 0.5, 96),
}

_BORDER_MODES = {
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT_101,
    "black": cv2.BORDER_CONSTANT,
}


@dataclass
class RenderSettings:
    fit: str = "cover"
    quality: str = "high"
    linear_light: bool = True
    dither: bool = True
    motion_blur: bool = True
    shutter_angle: float = 180.0
    border: Optional[str] = None   # None = replicate for "cover", black for "contain"
    out_bits: int = 8              # 8 -> uint8 frames, 16 -> uint16 frames
    max_samples: Optional[int] = None


# ----------------------------------------------------------------------------- colour handling

_LUT_CACHE: dict[int, np.ndarray] = {}


def _srgb_decode_lut(bits: int) -> np.ndarray:
    lut = _LUT_CACHE.get(bits)
    if lut is None:
        n = 1 << bits
        v = np.arange(n, dtype=np.float64) / (n - 1)
        lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
        lut = lin.astype(np.float32)
        _LUT_CACHE[bits] = lut
    return lut


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Float array in [0, 1] -> linear light."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear light float32 array (already clipped to [0, 1]) -> sRGB-encoded."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    hi = cv2.pow(np.maximum(x, 0.0031308), 1.0 / 2.4)
    hi *= 1.055
    hi -= 0.055
    return np.where(x <= 0.0031308, x * 12.92, hi).astype(np.float32, copy=False)


# ----------------------------------------------------------------------------- source frames


class SourceFrame:
    """A decoded frame as float32 RGB, with lazily built area-filtered mip levels."""

    def __init__(self, data: np.ndarray, linear_light: bool = True) -> None:
        if data.ndim != 3 or data.shape[2] != 3:
            raise ValueError("expected an HxWx3 RGB array")
        self.h, self.w = int(data.shape[0]), int(data.shape[1])
        if data.dtype == np.uint8:
            bits = 8
        elif data.dtype == np.uint16:
            bits = 16
        else:
            raise ValueError(f"unsupported frame dtype {data.dtype}")
        if linear_light:
            base = _srgb_decode_lut(bits)[data]
        else:
            base = data.astype(np.float32) * np.float32(1.0 / ((1 << bits) - 1))
        self._levels: dict[int, np.ndarray] = {0: np.ascontiguousarray(base)}

    @property
    def size(self) -> tuple[int, int]:
        return self.w, self.h

    def _level(self, level: int) -> np.ndarray:
        if level in self._levels:
            return self._levels[level]
        finer = max(k for k in self._levels if k < level)
        img = self._levels[finer]
        for lv in range(finer + 1, level + 1):
            nh = max(1, int(round(self.h / (1 << lv))))
            nw = max(1, int(round(self.w / (1 << lv))))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            self._levels[lv] = img
        return self._levels[level]

    def prefiltered(self, scale: float) -> tuple[np.ndarray, float, float]:
        """Return (image, rx, ry): the source area-filtered for an output scale of ``scale``.

        ``rx``/``ry`` are the size ratios of the returned image relative to the source.
        """
        if scale >= 0.999:
            return self._levels[0], 1.0, 1.0
        level = max(0, int(math.floor(math.log2(1.0 / scale))))
        level = min(level, max(0, int(math.log2(max(1, min(self.w, self.h))))))
        base = self._level(level)
        tw = max(1, int(math.ceil(self.w * scale)))
        th = max(1, int(math.ceil(self.h * scale)))
        if base.shape[1] < tw or base.shape[0] < th:
            # Level already smaller than the target (only for extreme minification).
            return base, base.shape[1] / self.w, base.shape[0] / self.h
        if (tw, th) != (base.shape[1], base.shape[0]):
            base = cv2.resize(base, (tw, th), interpolation=cv2.INTER_AREA)
        return base, tw / self.w, th / self.h


# ----------------------------------------------------------------------------- renderer


def _level_transform(rx: float, ry: float) -> np.ndarray:
    """Maps source px coords to prefiltered-image px coords (pixel-centre convention)."""
    return np.array(
        [[rx, 0.0, 0.5 * rx - 0.5], [0.0, ry, 0.5 * ry - 0.5], [0.0, 0.0, 1.0]], dtype=np.float64
    )


class Renderer:
    """Renders output frames of a fixed size. Not thread-safe: use one instance per thread."""

    def __init__(self, out_w: int, out_h: int, settings: Optional[RenderSettings] = None, seed: int = 1234):
        self.settings = settings or RenderSettings()
        if self.settings.quality not in QUALITY_PROFILES:
            raise ValueError(f"unknown quality {self.settings.quality!r}")
        if self.settings.out_bits not in (8, 16):
            raise ValueError("out_bits must be 8 or 16")
        self.out_w, self.out_h = int(out_w), int(out_h)
        self.profile = QUALITY_PROFILES[self.settings.quality]
        border = self.settings.border or ("black" if self.settings.fit == "contain" else "replicate")
        if border not in _BORDER_MODES:
            raise ValueError(f"unknown border mode {border!r}")
        self._border = _BORDER_MODES[border]
        self._rng = np.random.default_rng(seed)
        self._tmp = np.empty((self.out_h, self.out_w, 3), np.float32)
        self._acc = np.empty((self.out_h, self.out_w, 3), np.float32)
        self.last_sample_count = 1

    # -- blur planning -------------------------------------------------------------------

    def _matrix(self, state: CameraState, src: SourceFrame) -> np.ndarray:
        return view_matrix(state, src.size, (self.out_w, self.out_h), self.settings.fit)

    def _displacement(self, m0: np.ndarray, m1: np.ndarray, mmid: np.ndarray) -> float:
        """Largest on-screen movement (output px) of the picture between two camera poses."""
        w, h = self.out_w, self.out_h
        pts = np.array(
            [[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5], [w / 2, h / 2],
             [w / 2, -0.5], [w / 2, h - 0.5], [-0.5, h / 2], [w - 0.5, h / 2]],
            dtype=np.float64,
        )
        hom = np.hstack([pts, np.ones((len(pts), 1))])
        src_pts = (np.linalg.inv(mmid) @ hom.T).T
        a = (m0 @ src_pts.T).T[:, :2]
        b = (m1 @ src_pts.T).T[:, :2]
        return float(np.max(np.hypot(*(a - b).T)))

    def plan_samples(self, track: CameraTrack, src: SourceFrame, t: float, frame_dt: float) -> list[float]:
        """Camera sample times for the frame at ``t`` (a single time when no blur is needed)."""
        s = self.settings
        if not s.motion_blur or s.shutter_angle <= 0.0 or len(track) < 2:
            return [t]
        window = frame_dt * min(s.shutter_angle, 360.0) / 360.0
        m0 = self._matrix(track.sample(t - window / 2), src)
        m1 = self._matrix(track.sample(t + window / 2), src)
        mm = self._matrix(track.sample(t), src)
        disp = self._displacement(m0, m1, mm)
        if disp < 0.35:
            return [t]
        cap = s.max_samples or self.profile.max_samples
        n = int(min(cap, max(2, math.ceil(disp / self.profile.step_px))))
        return [t + window * ((k + 0.5) / n - 0.5) for k in range(n)]

    # -- rendering -----------------------------------------------------------------------

    def render_linear(self, src: SourceFrame, track: CameraTrack, t: float, frame_dt: float) -> np.ndarray:
        """Render and return the float32 image (still in the working colour space, unclipped)."""
        times = self.plan_samples(track, src, t, frame_dt)
        self.last_sample_count = len(times)
        mats = [self._matrix(track.sample(tt), src) for tt in times]
        # One prefiltered image per frame, sized for the most minified sample.
        s_ref = min(matrix_scale(m) for m in mats)
        img, rx, ry = src.prefiltered(s_ref)
        inv_level = np.linalg.inv(_level_transform(rx, ry))
        interp = (
            self.profile.interpolation
            if len(mats) <= self.profile.sharp_max_samples or len(mats) == 1
            else self.profile.blur_interpolation
        )
        size = (self.out_w, self.out_h)
        for idx, m in enumerate(mats):
            m_level = (m @ inv_level)[:2]
            if len(mats) == 1:
                return cv2.warpAffine(
                    img, m_level, size, flags=interp, borderMode=self._border, borderValue=(0, 0, 0)
                )
            cv2.warpAffine(
                img, m_level, size, dst=self._tmp, flags=interp,
                borderMode=self._border, borderValue=(0, 0, 0),
            )
            if idx == 0:
                np.copyto(self._acc, self._tmp)
            else:
                self._acc += self._tmp
        self._acc *= np.float32(1.0 / len(mats))
        return self._acc

    def finish(self, img: np.ndarray) -> np.ndarray:
        """Clip, convert to display encoding and quantise to the output bit depth."""
        s = self.settings
        img = np.clip(img, 0.0, 1.0)
        if s.linear_light:
            img = linear_to_srgb(img)
        if s.out_bits == 16:
            return np.rint(img * 65535.0).astype(np.uint16)
        scaled = img * np.float32(255.0)
        if s.dither:
            noise = self._rng.random(scaled.shape, dtype=np.float32)
            scaled += noise
            np.floor(scaled, out=scaled)
        else:
            np.rint(scaled, out=scaled)
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def render(self, src: SourceFrame, track: CameraTrack, t: float, frame_dt: float) -> np.ndarray:
        return self.finish(self.render_linear(src, track, t, frame_dt))
