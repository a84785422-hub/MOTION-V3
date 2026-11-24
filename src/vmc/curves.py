"""Easing curves and shape-preserving spline helpers for keyframe animation."""

from __future__ import annotations

from typing import Sequence, Union

Interp = Union[str, Sequence[float]]

# CSS-style cubic-bezier control points (x1, y1, x2, y2).
EASINGS: dict[str, tuple[float, float, float, float]] = {
    "linear": (0.0, 0.0, 1.0, 1.0),
    "ease_in": (0.42, 0.0, 1.0, 1.0),
    "ease_out": (0.0, 0.0, 0.58, 1.0),
    "ease_in_out": (0.42, 0.0, 0.58, 1.0),
    "ease_in_cubic": (0.32, 0.0, 0.67, 0.0),
    "ease_out_cubic": (0.33, 1.0, 0.68, 1.0),
    "ease_in_out_cubic": (0.65, 0.0, 0.35, 1.0),
    "ease_in_out_quint": (0.83, 0.0, 0.17, 1.0),
}

# Modes that are not a simple easing of a linear blend.
HOLD = "hold"      # keep the previous keyframe value until the next one
SMOOTH = "smooth"  # monotone cubic spline through neighbouring keyframes (no overshoot)
SPECIAL = (HOLD, SMOOTH)

ALL_INTERP_NAMES = tuple(EASINGS) + SPECIAL


def normalize_interp(spec: Interp) -> Union[str, tuple[float, float, float, float]]:
    """Validate an interpolation spec and return a canonical form.

    A spec is either a known name or a list of four numbers (cubic-bezier control points).
    """
    if isinstance(spec, str):
        if spec in EASINGS or spec in SPECIAL:
            return spec
        raise ValueError(
            f"unknown interpolation {spec!r}; expected one of {', '.join(ALL_INTERP_NAMES)} "
            "or [x1, y1, x2, y2]"
        )
    try:
        pts = tuple(float(v) for v in spec)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid interpolation {spec!r}") from exc
    if len(pts) != 4:
        raise ValueError("a custom interpolation needs exactly four numbers [x1, y1, x2, y2]")
    if not (0.0 <= pts[0] <= 1.0 and 0.0 <= pts[2] <= 1.0):
        raise ValueError("x1 and x2 of a custom bezier must be within [0, 1]")
    return pts  # type: ignore[return-value]


class CubicBezier:
    """CSS-style cubic-bezier easing: maps progress ``u`` in [0, 1] to eased progress."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if not (0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0):
            raise ValueError("x control points must be within [0, 1]")
        self.points = (x1, y1, x2, y2)
        self._cx = 3.0 * x1
        self._bx = 3.0 * (x2 - x1) - self._cx
        self._ax = 1.0 - self._cx - self._bx
        self._cy = 3.0 * y1
        self._by = 3.0 * (y2 - y1) - self._cy
        self._ay = 1.0 - self._cy - self._by

    def _x(self, s: float) -> float:
        return ((self._ax * s + self._bx) * s + self._cx) * s

    def _y(self, s: float) -> float:
        return ((self._ay * s + self._by) * s + self._cy) * s

    def _dx(self, s: float) -> float:
        return (3.0 * self._ax * s + 2.0 * self._bx) * s + self._cx

    def __call__(self, u: float) -> float:
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        # Newton-Raphson first (fast), bisection as a robust fallback.
        s = u
        for _ in range(8):
            err = self._x(s) - u
            if abs(err) < 1e-7:
                return self._y(s)
            d = self._dx(s)
            if abs(d) < 1e-6:
                break
            s -= err / d
        lo, hi, s = 0.0, 1.0, u
        for _ in range(64):
            err = self._x(s) - u
            if abs(err) < 1e-7:
                break
            if err > 0.0:
                hi = s
            else:
                lo = s
            s = 0.5 * (lo + hi)
        return self._y(s)


def get_easing(spec: Interp) -> CubicBezier:
    """Return the easing function for a named preset or custom bezier."""
    norm = normalize_interp(spec)
    if isinstance(norm, str):
        if norm in SPECIAL:
            raise ValueError(f"{norm!r} is not a bezier easing")
        return CubicBezier(*EASINGS[norm])
    return CubicBezier(*norm)


def _sign(v: float) -> int:
    return (v > 0.0) - (v < 0.0)


def _pchip_edge(h0: float, h1: float, m0: float, m1: float) -> float:
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if _sign(d) != _sign(m0):
        return 0.0
    if _sign(m0) != _sign(m1) and abs(d) > 3.0 * abs(m0):
        return 3.0 * m0
    return d


def pchip_slopes(x: Sequence[float], y: Sequence[float]) -> list[float]:
    """Fritsch-Carlson monotone cubic slopes (same definition as SciPy's PCHIP)."""
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must have the same length")
    if n < 2:
        return [0.0] * n
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    if any(v <= 0.0 for v in h):
        raise ValueError("x must be strictly increasing")
    delta = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        return [delta[0], delta[0]]
    d = [0.0] * n
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] <= 0.0:
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    d[0] = _pchip_edge(h[0], h[1], delta[0], delta[1])
    d[-1] = _pchip_edge(h[-1], h[-2], delta[-1], delta[-2])
    return d


def hermite(t0: float, t1: float, y0: float, y1: float, m0: float, m1: float, t: float) -> float:
    """Cubic Hermite interpolation between (t0, y0, slope m0) and (t1, y1, slope m1)."""
    h = t1 - t0
    s = (t - t0) / h
    s2 = s * s
    s3 = s2 * s
    return (
        (2.0 * s3 - 3.0 * s2 + 1.0) * y0
        + (s3 - 2.0 * s2 + s) * h * m0
        + (-2.0 * s3 + 3.0 * s2) * y1
        + (s3 - s2) * h * m1
    )
