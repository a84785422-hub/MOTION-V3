import math

import numpy as np
import pytest

from vmc.camera import (
    CameraState,
    CameraTrack,
    Keyframe,
    base_scale,
    matrix_scale,
    view_corners,
    view_inside_source,
    view_matrix,
)
from vmc.curves import EASINGS, CubicBezier, get_easing, normalize_interp, pchip_slopes


@pytest.mark.parametrize("name", list(EASINGS))
def test_easing_endpoints_and_range(name):
    e = get_easing(name)
    assert e(0.0) == 0.0 and e(1.0) == 1.0
    vals = [e(u / 50) for u in range(51)]
    # All presets have y control points inside [0, 1], so the curve stays inside it too.
    assert all(-1e-9 <= v <= 1.0 + 1e-9 for v in vals)


def test_linear_easing_is_identity():
    e = get_easing("linear")
    for u in (0.1, 0.25, 0.5, 0.9):
        assert e(u) == pytest.approx(u, abs=1e-6)


def test_ease_in_out_is_monotonic_and_symmetric():
    e = get_easing("ease_in_out")
    vals = [e(u / 100) for u in range(101)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
    assert e(0.5) == pytest.approx(0.5, abs=1e-6)
    assert e(0.25) == pytest.approx(1 - e(0.75), abs=1e-6)


def test_custom_bezier_and_validation():
    assert isinstance(get_easing([0.2, 0.0, 0.2, 1.0]), CubicBezier)
    with pytest.raises(ValueError):
        normalize_interp("bounce")
    with pytest.raises(ValueError):
        normalize_interp([0.2, 0.0, 1.5, 1.0])
    with pytest.raises(ValueError):
        normalize_interp([0.2, 0.0, 1.0])


def test_pchip_matches_scipy():
    scipy_interp = pytest.importorskip("scipy.interpolate")
    x = [0.0, 1.0, 2.5, 3.0, 6.0]
    y = [0.0, 2.0, 2.2, 5.0, 4.0]
    ours = pchip_slopes(x, y)
    ref = scipy_interp.PchipInterpolator(x, y).derivative()(x)
    assert np.allclose(ours, ref, atol=1e-9)


def test_track_holds_outside_range_and_linear_midpoint():
    track = CameraTrack([
        Keyframe(1.0, x=0.2, interp="linear"),
        Keyframe(3.0, x=0.8),
    ])
    assert track.sample(0.0).x == 0.2
    assert track.sample(5.0).x == 0.8
    assert track.sample(2.0).x == pytest.approx(0.5)


def test_zoom_is_interpolated_in_log_space():
    track = CameraTrack([Keyframe(0, zoom=1.0, interp="linear"), Keyframe(2, zoom=4.0)])
    assert track.sample(1.0).zoom == pytest.approx(2.0)


def test_hold_and_smooth_modes():
    hold = CameraTrack([Keyframe(0, x=0.1, interp="hold"), Keyframe(1, x=0.9)])
    assert hold.sample(0.99).x == 0.1
    kfs = [Keyframe(i, x=v, interp="smooth") for i, v in enumerate([0.0, 0.5, 0.5, 1.0])]
    smooth = CameraTrack(kfs)
    xs = [smooth.sample(t / 20).x for t in range(61)]
    assert min(xs) >= -1e-9 and max(xs) <= 1.0 + 1e-9  # monotone spline: no overshoot
    assert smooth.sample(1.0).x == pytest.approx(0.5)
    assert smooth.sample(1.5).x == pytest.approx(0.5, abs=1e-9)  # flat segment stays flat


def test_duplicate_times_rejected():
    with pytest.raises(ValueError):
        CameraTrack([Keyframe(1.0), Keyframe(1.0)])


def test_keyframe_roundtrip():
    kf = Keyframe(2.5, 0.3, 0.6, 1.5, -12.0, [0.1, 0.2, 0.3, 0.4])
    assert Keyframe.from_dict(kf.to_dict()) == Keyframe(2.5, 0.3, 0.6, 1.5, -12.0, (0.1, 0.2, 0.3, 0.4))


def test_view_matrix_identity_and_centre():
    m = view_matrix(CameraState(), (640, 360), (640, 360))
    assert np.allclose(m, np.eye(3))
    # Source centre always lands on the output centre, whatever zoom/rotation.
    st = CameraState(x=0.3, y=0.7, zoom=2.5, rotation=33.0)
    m = view_matrix(st, (1920, 1080), (1280, 720))
    c = np.array([0.3 * 1920 - 0.5, 0.7 * 1080 - 0.5, 1.0])
    assert np.allclose((m @ c)[:2], [(1280 - 1) / 2, (720 - 1) / 2])


def test_view_matrix_zoom_and_rotation():
    m1 = view_matrix(CameraState(zoom=1.0), (100, 100), (100, 100))
    m2 = view_matrix(CameraState(zoom=2.0), (100, 100), (100, 100))
    assert matrix_scale(m2) == pytest.approx(2 * matrix_scale(m1))
    # +90 degrees = clockwise on screen: a point to the right of centre moves to below it.
    m = view_matrix(CameraState(rotation=90.0), (101, 101), (101, 101))
    p = (m @ np.array([60.0, 50.0, 1.0]))[:2]
    assert p == pytest.approx([50.0, 60.0], abs=1e-9)


def test_fit_modes():
    assert base_scale((1920, 1080), (1080, 1920), "cover") == pytest.approx(1920 / 1080)
    assert base_scale((1920, 1080), (1080, 1920), "contain") == pytest.approx(1080 / 1920)


def test_view_corners_and_bounds():
    inside = CameraState(zoom=1.5)
    assert view_inside_source(inside, (1920, 1080), (1920, 1080))
    outside = CameraState(x=0.05, zoom=1.5)
    assert not view_inside_source(outside, (1920, 1080), (1920, 1080))
    corners = view_corners(CameraState(), (1920, 1080), (1920, 1080))
    assert corners[0] == pytest.approx((0.0, 0.0)) and corners[2] == pytest.approx((1.0, 1.0))
    # A rotated full-frame view must leave the source.
    assert not view_inside_source(CameraState(rotation=10.0), (1920, 1080), (1920, 1080))
    assert math.isclose(view_corners(CameraState(zoom=2.0), (100, 100), (100, 100))[0][0], 0.25)
