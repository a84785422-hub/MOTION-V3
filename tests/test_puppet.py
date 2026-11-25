import cv2
import numpy as np

from vmc.ai import puppet as pp


def _figure():
    lm = np.zeros((33, 2), np.float32)
    pts = {0: (200, 80), 7: (185, 80), 8: (215, 80), 11: (160, 170), 12: (240, 170), 13: (130, 260),
           14: (270, 260), 15: (120, 350), 16: (280, 350), 23: (180, 330), 24: (220, 330),
           25: (175, 450), 26: (225, 450), 27: (172, 560), 28: (228, 560), 31: (165, 580), 32: (235, 580)}
    for i, v in pts.items():
        lm[i] = v
    return lm


def test_identity_warp_keeps_image():
    lm = _figure()
    j = pp.joints_from_landmarks(lm)
    img = np.random.RandomState(0).randint(0, 255, (600, 400, 3), np.uint8)
    src = pp.make_source(img, j)
    out = pp.warp_image(src, pp.build_displacement(src, pp.retarget(j, j, j)))
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 0.5


def test_raised_arm_moves_pixels_and_keeps_bone_length():
    lm = _figure()
    j = pp.joints_from_landmarks(lm)
    lm2 = lm.copy()
    lm2[13], lm2[15] = (160, 80), (160, -10)  # bras gauche levé
    j2 = pp.joints_from_landmarks(lm2)
    new = pp.retarget(j, j2, j)
    for k in ("lel", "lwr"):
        p = pp.PARENT[k]
        assert abs(np.linalg.norm(new[k] - new[p]) - np.linalg.norm(j[k] - j[p])) < 1e-2
    assert np.allclose(new["rel"], j["rel"], atol=1e-3)  # l'autre bras ne bouge pas
    img = np.zeros((600, 400, 3), np.uint8)
    cv2.rectangle(img, (100, 240), (160, 360), (255, 255, 255), -1)
    src = pp.make_source(img, j)
    out = pp.warp_image(src, pp.build_displacement(src, new))
    assert np.abs(out.astype(int) - img.astype(int)).mean() > 0.3


def test_root_translation_is_scaled_to_reference():
    lm = _figure()
    j = pp.joints_from_landmarks(lm)
    lm2 = lm + np.array([30, 0], np.float32)
    new = pp.retarget(j, pp.joints_from_landmarks(lm2), j)
    assert np.allclose(new["pelvis"] - j["pelvis"], [30, 0], atol=1e-3)


def test_body_mode_end_to_end(tmp_path, test_video, monkeypatch):
    from vmc.ai import animate as anim

    class FakeDetector:
        def __init__(self, path):
            self.n = 0

        def reference(self, rgb):
            return _figure()

        def driving(self, rgb, fps):
            self.n += 1
            lm = _figure()
            lm[15] += (0, -min(self.n, 40))  # le poignet gauche monte
            return lm

        def reset(self):
            self.n = 0

    monkeypatch.setattr(pp, "PoseDetector", FakeDetector)
    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), np.random.RandomState(2).randint(0, 255, (600, 400, 3), np.uint8))
    out = tmp_path / "o.mp4"
    res = anim.animate(anim.AnimateOptions(str(ref), test_video, str(out), max_output_side=300, mode="body"),
                       engine=pp.PuppetEngine(tmp_path / "x.task"))
    assert res.frames >= 30 and out.stat().st_size > 0
