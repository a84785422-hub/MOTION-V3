import numpy as np
import pytest

from vmc.camera import CameraState, CameraTrack, Keyframe
from vmc.render import Renderer, RenderSettings, SourceFrame, linear_to_srgb, srgb_to_linear


def _gradient(w=320, h=180):
    xs = np.linspace(0, 255, w, dtype=np.float32)
    ys = np.linspace(0, 255, h, dtype=np.float32)
    r = np.tile(xs, (h, 1))
    g = np.tile(ys[:, None], (1, w))
    b = np.full((h, w), 90, np.float32)
    return np.dstack([r, g, b]).astype(np.uint8)


def _static(**kw):
    return RenderSettings(motion_blur=False, dither=False, **kw)


def test_color_roundtrip():
    x = np.linspace(0, 1, 1001, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=2e-5)


@pytest.mark.parametrize("linear", [True, False])
def test_identity_render_is_lossless(linear):
    img = _gradient()
    r = Renderer(320, 180, _static(linear_light=linear))
    out = r.render(SourceFrame(img, linear), CameraTrack([]), 0.0, 1 / 30)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_16bit_output_and_input():
    img16 = (_gradient().astype(np.uint16) * 257)
    r = Renderer(320, 180, _static(out_bits=16))
    out = r.render(SourceFrame(img16, True), CameraTrack([]), 0.0, 1 / 30)
    assert out.dtype == np.uint16
    assert np.abs(out.astype(int) - img16.astype(int)).max() <= 2


def test_zoom_in_crops_centre():
    img = _gradient()
    track = CameraTrack([Keyframe(0, zoom=2.0)])
    out = Renderer(320, 180, _static(linear_light=False)).render(SourceFrame(img, False), track, 0, 1 / 30)
    # Zoom 2 shows the middle half of the source: output centre column == source centre column.
    assert abs(int(out[90, 160, 0]) - int(img[90, 160, 0])) <= 2
    assert out[90, 0, 0] > img[90, 0, 0] + 50  # left edge now shows source x ~ 25 %


def test_downscale_is_antialiased():
    # One-pixel checkerboard minified 4x must average to mid-grey, not alias.
    h, w = 720, 1280
    yy, xx = np.mgrid[0:h, 0:w]
    img = (((xx + yy) % 2) * 255).astype(np.uint8)
    img = np.dstack([img, img, img])
    out = Renderer(320, 180, _static(linear_light=False)).render(SourceFrame(img, False), CameraTrack([]), 0, 1 / 30)
    centre = out[20:-20, 20:-20, 0].astype(float)
    assert abs(centre.mean() - 127.5) < 4
    assert centre.std() < 12


def test_linear_light_blends_brighter_than_gamma():
    # Averaging black and white in linear light gives sRGB ~188, not 128.
    h, w = 64, 64
    img = np.zeros((h, w, 3), np.uint8)
    img[:, ::2] = 255
    lin = Renderer(32, 32, _static(linear_light=True)).render(SourceFrame(img, True), CameraTrack([]), 0, 1 / 30)
    gam = Renderer(32, 32, _static(linear_light=False)).render(SourceFrame(img, False), CameraTrack([]), 0, 1 / 30)
    assert abs(float(lin[8:-8, 8:-8].mean()) - 188) < 6
    assert abs(float(gam[8:-8, 8:-8].mean()) - 128) < 6


def _line_image(w=400, h=100):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, 200:204] = 255
    return img


def _pan_track():
    # Moves the camera 40 source px (zoom 1, cover, same size) within one second, linearly.
    return CameraTrack([Keyframe(0, x=0.5, interp="linear"), Keyframe(1, x=0.6)])


def test_motion_blur_spreads_energy_and_adapts_sample_count():
    src = SourceFrame(_line_image(), True)
    track = _pan_track()
    sharp = Renderer(400, 100, RenderSettings(motion_blur=False, dither=False))
    blur = Renderer(400, 100, RenderSettings(motion_blur=True, shutter_angle=360, dither=False))
    a = sharp.render(src, track, 0.5, 1.0)
    b = blur.render(src, track, 0.5, 1.0)
    assert blur.last_sample_count > 8
    assert a.max() == 255 and b.max() < 255
    # Blur widens the streak: more columns carry visible light.
    assert (b[50, :, 0] > 20).sum() > 3 * (a[50, :, 0] > 20).sum()


def test_no_blur_when_nothing_moves():
    src = SourceFrame(_line_image(), True)
    r = Renderer(400, 100, RenderSettings(motion_blur=True, dither=False))
    r.render(src, CameraTrack([Keyframe(0), Keyframe(1)]), 0.5, 1 / 30)
    assert r.last_sample_count == 1


def test_blur_sample_count_is_capped_by_quality():
    src = SourceFrame(_line_image(), True)
    r = Renderer(400, 100, RenderSettings(quality="draft", motion_blur=True, shutter_angle=360, dither=False))
    r.render(src, _pan_track(), 0.5, 1.0)
    assert r.last_sample_count == 8


def test_dither_is_unbiased_and_reduces_banding():
    # A very flat gradient: dithering should keep the mean while mixing neighbouring levels.
    img = np.full((180, 320, 3), 100, np.uint8)
    settings = RenderSettings(motion_blur=False, dither=True, linear_light=True)
    out = Renderer(320, 180, settings).render(SourceFrame(img, True), CameraTrack([Keyframe(0, zoom=1.003)]), 0, 1 / 30)
    assert abs(float(out.mean()) - 100) < 0.6


def test_contain_uses_black_bars():
    img = np.full((100, 200, 3), 200, np.uint8)
    out = Renderer(100, 200, _static(fit="contain", linear_light=False)).render(
        SourceFrame(img, False), CameraTrack([]), 0, 1 / 30
    )
    assert out[0, 50, 0] == 0 and out[100, 50, 0] > 190
