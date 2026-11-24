"""Teste toute la chaîne IA avec de faux réseaux ONNX (mêmes formes d'entrée/sortie que les vrais)."""

import threading
from fractions import Fraction

import cv2
import numpy as np
import pytest

from vmc.ai import animate as anim
from vmc.ai import liveportrait as lp
from vmc.ai import models as store


class FakeNet:
    def __init__(self, path, threads=1):
        self.name = path.stem
        self.in_names, self.out_names = ["input"], ["o"]
        if self.name == "warping_spade":
            self.in_names = ["feature_3d", "kp_driving", "kp_source"]
        ang = np.linspace(0, 2 * np.pi, 203, endpoint=False)
        self.p203 = np.stack([0.5 + 0.3 * np.cos(ang), 0.5 + 0.35 * np.sin(ang)], 1).astype(np.float32)
        a106 = np.linspace(0, 2 * np.pi, 106, endpoint=False)
        self.p106 = np.stack([0.5 * np.cos(a106), 0.6 * np.sin(a106)], 1).astype(np.float32)
        self.calls = 0

    def run_named(self, feed):
        return self(*[feed[n] for n in self.in_names])

    def __call__(self, *arrays):
        self.calls += 1
        n = self.name
        if n == "retinaface_det_static":
            outs = []
            for size in (8192, 2048, 512):
                sc = np.zeros((size, 1), np.float32)
                if size == 2048:
                    sc[1056, 0] = 0.9
                outs.append(sc)
            for size in (8192, 2048, 512):
                bb = np.zeros((size, 4), np.float32)
                if size == 2048:
                    bb[1056] = [5, 5, 5, 5]
                outs.append(bb)
            for size in (8192, 2048, 512):
                outs.append(np.zeros((size, 10), np.float32))
            return outs
        if n == "face_2dpose_106_static":
            return [self.p106.reshape(1, -1)]
        if n == "landmark":
            return [np.zeros(1), np.zeros(1), self.p203.reshape(1, -1)]
        if n == "appearance_feature_extractor":
            return [np.zeros((1, 32, 16, 64, 64), np.float32)]
        if n == "motion_extractor":
            m = float(arrays[0].mean())
            rs = np.random.RandomState(0)
            return [np.zeros((1, 66)), np.zeros((1, 66)), np.zeros((1, 66)),
                    np.array([[m, 0, 0]], np.float32), np.zeros((1, 63), np.float32),
                    np.ones((1, 1), np.float32), rs.randn(1, 63).astype(np.float32) * 0.1]
        if n == "warping_spade":
            return [np.full((1, 3, 512, 512), 0.5, np.float32)]
        if n == "stitching":
            return [np.zeros((1, 65), np.float32)]
        if n == "stitching_lip":
            return [np.zeros((1, 63), np.float32)]
        raise AssertionError(n)


def _png(path, w=320, h=400):
    img = np.random.RandomState(1).randint(0, 255, (h, w, 3), np.uint8)
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "_Net", FakeNet)
    for name in store.MODEL_FILES:
        (tmp_path / name).write_bytes(b"0" * 20_000_000)
    return lp.LivePortrait(tmp_path, 1)


def test_models_missing_and_ready(tmp_path):
    assert set(store.missing(tmp_path)) == set(store.MODEL_FILES)
    for name in store.MODEL_FILES:
        (tmp_path / name).write_bytes(b"0" * 20_000_000)
    assert store.is_ready(tmp_path)


def test_rotation_identity():
    r = lp.rotation_matrix([[0]], [[0]], [[0]])
    assert np.allclose(r[0], np.eye(3))


def test_crop_roundtrip():
    img = np.zeros((300, 300, 3), np.uint8)
    ang = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    pts = np.stack([150 + 40 * np.cos(ang), 150 + 50 * np.sin(ang)], 1)
    crop, m_o2c, m_c2o = lp.crop_by_landmarks(img, pts, 256, 2.0)
    assert crop.shape == (256, 256, 3)
    assert np.allclose(m_o2c @ m_c2o, np.eye(3), atol=1e-4)


def test_source_and_render(engine):
    rgb = np.full((480, 360, 3), 90, np.uint8)
    src = engine.prepare_source(rgb)
    assert src.x_s.shape == (1, 21, 3)
    assert src.mask.shape == rgb.shape
    cur = engine._extract_motion(np.zeros((256, 256, 3), np.uint8))
    out = engine.render(src, cur, cur)
    assert out.shape == rgb.shape and out.dtype == np.uint8


def test_driving_motion_tracks(engine):
    frame = np.full((360, 640, 3), 100, np.uint8)
    assert engine.driving_motion(frame, redetect=True) is not None
    assert engine.driving_motion(frame) is not None


def test_end_to_end(engine, tmp_path, test_video):
    ref = _png(tmp_path / "ref.png")
    drv = test_video
    out = tmp_path / "out.mp4"
    seen = []
    res = anim.animate(anim.AnimateOptions(str(ref), str(drv), str(out), max_output_side=256),
                       progress=lambda *a: seen.append(a), engine=engine)
    assert res.frames >= 30 and out.is_file() and out.stat().st_size > 0
    assert seen


def test_cancel(engine, tmp_path, test_video):
    from vmc.export import ExportCancelled

    ref = _png(tmp_path / "ref.png")
    drv = test_video
    ev = threading.Event()
    ev.set()
    with pytest.raises(ExportCancelled):
        anim.animate(anim.AnimateOptions(str(ref), str(drv), str(tmp_path / "o.mp4")), cancel=ev, engine=engine)
    assert not (tmp_path / "o.mp4").exists()
