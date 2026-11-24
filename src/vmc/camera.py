"""Virtual camera: keyframed track and the geometry that maps source pixels to output pixels.

Conventions
-----------
* ``x`` / ``y``: camera centre in normalised source coordinates (0..1, origin top-left).
* ``zoom``: 1.0 frames the source according to the fit mode; 2.0 shows half of it.
  Zoom is interpolated in log space so that a zoom feels uniform over time.
* ``rotation``: degrees, positive rotates the picture clockwise on screen.
* Pixel centres sit on integer coordinates (OpenCV convention).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .curves import HOLD, SMOOTH, Interp, get_easing, hermite, normalize_interp, pchip_slopes

FIT_MODES = ("cover", "contain")
_PROPS = ("x", "y", "zoom", "rotation")


@dataclass(frozen=True)
class CameraState:
    x: float = 0.5
    y: float = 0.5
    zoom: float = 1.0
    rotation: float = 0.0


@dataclass
class Keyframe:
    """A camera pose at time ``t`` (seconds). ``interp`` shapes the segment *leaving* it."""

    t: float
    x: float = 0.5
    y: float = 0.5
    zoom: float = 1.0
    rotation: float = 0.0
    interp: Interp = "ease_in_out"

    def state(self) -> CameraState:
        return CameraState(self.x, self.y, self.zoom, self.rotation)

    def to_dict(self) -> dict[str, Any]:
        interp = self.interp if isinstance(self.interp, str) else [float(v) for v in self.interp]
        return {
            "t": float(self.t),
            "x": float(self.x),
            "y": float(self.y),
            "zoom": float(self.zoom),
            "rotation": float(self.rotation),
            "interp": interp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Keyframe":
        try:
            kf = cls(
                t=float(data["t"]),
                x=float(data.get("x", 0.5)),
                y=float(data.get("y", 0.5)),
                zoom=float(data.get("zoom", 1.0)),
                rotation=float(data.get("rotation", 0.0)),
                interp=normalize_interp(data.get("interp", "ease_in_out")),
            )
        except KeyError as exc:
            raise ValueError("a keyframe needs at least a 't' value") from exc
        if kf.zoom <= 0.0:
            raise ValueError("zoom must be greater than 0")
        return kf


class CameraTrack:
    """Samples a camera pose at any time from a list of keyframes."""

    def __init__(self, keyframes: Iterable[Keyframe] = ()) -> None:
        kfs = sorted(keyframes, key=lambda k: k.t)
        for a, b in zip(kfs, kfs[1:]):
            if b.t - a.t <= 1e-9:
                raise ValueError(f"two keyframes share the same time ({a.t} s)")
        for k in kfs:
            if k.zoom <= 0.0:
                raise ValueError("zoom must be greater than 0")
        self.keyframes: list[Keyframe] = kfs
        self._times = [k.t for k in kfs]
        self._values = {
            "x": [k.x for k in kfs],
            "y": [k.y for k in kfs],
            "zoom": [math.log(k.zoom) for k in kfs],
            "rotation": [k.rotation for k in kfs],
        }
        self._slopes = {name: pchip_slopes(self._times, vals) for name, vals in self._values.items()}
        self._modes: list[Any] = []
        for k in kfs:
            norm = normalize_interp(k.interp)
            self._modes.append(norm if isinstance(norm, str) and norm in (HOLD, SMOOTH) else get_easing(norm))

    def __len__(self) -> int:
        return len(self.keyframes)

    @property
    def end_time(self) -> float:
        return self._times[-1] if self._times else 0.0

    def sample(self, t: float) -> CameraState:
        n = len(self.keyframes)
        if n == 0:
            return CameraState()
        if n == 1 or t <= self._times[0]:
            return self.keyframes[0].state()
        if t >= self._times[-1]:
            return self.keyframes[-1].state()
        i = bisect_right(self._times, t) - 1
        t0, t1 = self._times[i], self._times[i + 1]
        mode = self._modes[i]
        out: dict[str, float] = {}
        u = (t - t0) / (t1 - t0)
        eased = None if isinstance(mode, str) else mode(u)
        for name in _PROPS:
            v0 = self._values[name][i]
            v1 = self._values[name][i + 1]
            if mode == HOLD:
                v = v0
            elif mode == SMOOTH:
                s = self._slopes[name]
                v = hermite(t0, t1, v0, v1, s[i], s[i + 1], t)
            else:
                v = v0 + (v1 - v0) * eased
            out[name] = v
        return CameraState(out["x"], out["y"], math.exp(out["zoom"]), out["rotation"])


def base_scale(src_size: tuple[int, int], out_size: tuple[int, int], fit: str = "cover") -> float:
    """Output pixels per source pixel at zoom 1."""
    sx = out_size[0] / src_size[0]
    sy = out_size[1] / src_size[1]
    if fit == "cover":
        return max(sx, sy)
    if fit == "contain":
        return min(sx, sy)
    raise ValueError(f"unknown fit mode {fit!r}")


def view_matrix(
    state: CameraState,
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    fit: str = "cover",
) -> np.ndarray:
    """3x3 matrix mapping source pixel coordinates to output pixel coordinates."""
    sw, sh = src_size
    ow, oh = out_size
    s = base_scale(src_size, out_size, fit) * state.zoom
    th = math.radians(state.rotation)
    c, sn = math.cos(th), math.sin(th)
    r00, r01, r10, r11 = s * c, -s * sn, s * sn, s * c
    cx = state.x * sw - 0.5
    cy = state.y * sh - 0.5
    ox = (ow - 1) / 2.0
    oy = (oh - 1) / 2.0
    tx = ox - (r00 * cx + r01 * cy)
    ty = oy - (r10 * cx + r11 * cy)
    return np.array([[r00, r01, tx], [r10, r11, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def matrix_scale(m: np.ndarray) -> float:
    """Isotropic scale factor of a similarity matrix."""
    return math.sqrt(abs(float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])))


def view_corners(
    state: CameraState,
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    fit: str = "cover",
) -> list[tuple[float, float]]:
    """Corners of the output frame in normalised source coordinates (TL, TR, BR, BL)."""
    sw, sh = src_size
    ow, oh = out_size
    inv = np.linalg.inv(view_matrix(state, src_size, out_size, fit))
    pts = np.array([[-0.5, -0.5, 1], [ow - 0.5, -0.5, 1], [ow - 0.5, oh - 0.5, 1], [-0.5, oh - 0.5, 1]], float)
    src = (inv @ pts.T).T
    return [((p[0] + 0.5) / sw, (p[1] + 0.5) / sh) for p in src]


def view_inside_source(
    state: CameraState,
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    fit: str = "cover",
    tolerance: float = 1e-4,
) -> bool:
    """True when every corner of the camera view lies inside the source image."""
    return all(
        -tolerance <= u <= 1.0 + tolerance and -tolerance <= v <= 1.0 + tolerance
        for u, v in view_corners(state, src_size, out_size, fit)
    )


def find_out_of_frame_times(
    track: CameraTrack,
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    fit: str,
    times: Sequence[float],
) -> list[float]:
    """Times (from ``times``) at which the view leaves the source (edge pixels get stretched)."""
    if fit == "contain":
        return []
    return [t for t in times if not view_inside_source(track.sample(t), src_size, out_size, fit)]
