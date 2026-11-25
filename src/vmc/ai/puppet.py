"""Corps entier « marionnette » : squelette de la vidéo pilote → déformation de l'image de référence.

Léger (CPU) mais approximatif : l'image est déformée par un squelage (skinning) autour des os
détectés ; il n'y a aucune génération de pixels cachés (plis, dos, mains…). La pose est détectée
avec MediaPipe Pose Landmarker (modèle .task téléchargé au premier lancement).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# indices MediaPipe (33 points)
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
L_FT, R_FT = 31, 32

JOINTS = ["pelvis", "neck", "head", "lsh", "lel", "lwr", "rsh", "rel", "rwr",
          "lhip", "lkn", "lan", "lft", "rhip", "rkn", "ran", "rft"]
# (enfant, parent) ; le bassin est la racine
PARENT = {
    "neck": "pelvis", "head": "neck",
    "lsh": "neck", "lel": "lsh", "lwr": "lel",
    "rsh": "neck", "rel": "rsh", "rwr": "rel",
    "lhip": "pelvis", "lkn": "lhip", "lan": "lkn", "lft": "lan",
    "rhip": "pelvis", "rkn": "rhip", "ran": "rkn", "rft": "ran",
}
ORDER = list(PARENT)  # parents toujours avant enfants


class NoPoseError(RuntimeError):
    pass


def joints_from_landmarks(lm: np.ndarray, vis: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """33 points (pixels) → dict de joints 2D."""
    m = lambda a, b: (lm[a] + lm[b]) / 2.0  # noqa: E731
    head = m(L_EAR, R_EAR)
    # la tête est un point au-dessus des épaules : milieu des oreilles
    return {
        "pelvis": m(L_HIP, R_HIP), "neck": m(L_SH, R_SH), "head": head,
        "lsh": lm[L_SH], "lel": lm[L_EL], "lwr": lm[L_WR],
        "rsh": lm[R_SH], "rel": lm[R_EL], "rwr": lm[R_WR],
        "lhip": lm[L_HIP], "lkn": lm[L_KN], "lan": lm[L_AN], "lft": lm[L_FT],
        "rhip": lm[R_HIP], "rkn": lm[R_KN], "ran": lm[R_AN], "rft": lm[R_FT],
    }


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    d = b - a
    return float(np.arctan2(d[1], d[0]))


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], np.float32)


def retarget(ref: dict, drv: dict, drv0: dict, multiplier: float = 1.0) -> dict[str, np.ndarray]:
    """Cinématique directe : longueurs d'os de la référence, angles = référence + (pilote − pilote₀)."""
    torso_ref = max(1e-3, float(np.linalg.norm(ref["neck"] - ref["pelvis"])))
    torso_drv = max(1e-3, float(np.linalg.norm(drv0["neck"] - drv0["pelvis"])))
    k = torso_ref / torso_drv
    new = {"pelvis": ref["pelvis"] + (drv["pelvis"] - drv0["pelvis"]) * k * multiplier}
    for j in ORDER:
        p = PARENT[j]
        vec_ref = ref[j] - ref[p]
        length = float(np.linalg.norm(vec_ref))
        delta = (_angle(drv0[p], drv0[j]) * -1 + _angle(drv[p], drv[j]))
        delta = (delta + np.pi) % (2 * np.pi) - np.pi  # dans [-π, π]
        new[j] = new[p] + _rot(delta * multiplier) @ vec_ref
        # la longueur est conservée par la rotation ; ``length`` documente l'invariant
        assert length >= 0.0
    return new


@dataclass
class PuppetSource:
    image: np.ndarray
    joints: dict
    step: int
    grid_x: np.ndarray
    grid_y: np.ndarray
    sigma: float


def _segment_distance(px: np.ndarray, py: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = max(float(ab @ ab), 1e-6)
    t = np.clip(((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / denom, 0.0, 1.0)
    return np.hypot(px - (a[0] + t * ab[0]), py - (a[1] + t * ab[1]))


def build_displacement(src: PuppetSource, new: dict) -> np.ndarray:
    """Champ de déplacement (grille grossière) par skinning linéaire, poids gaussiens autour des os."""
    gx, gy = src.grid_x, src.grid_y
    logits = []
    targets = []
    for j in ORDER:
        p = PARENT[j]
        a, b = src.joints[p], src.joints[j]
        d = _segment_distance(gx, gy, a, b)
        logits.append(-(d / src.sigma) ** 2)
        theta = _angle(a, b)
        theta_new = _angle(new[p], new[j])
        r = _rot(theta_new - theta)
        rel = np.stack([gx - a[0], gy - a[1]], axis=-1) @ r.T
        targets.append(rel + new[p])
    logits.append(np.full_like(gx, -(1.6 ** 2)))  # fond : immobile
    targets.append(np.stack([gx, gy], axis=-1))
    lg = np.stack(logits, axis=0)
    lg -= lg.max(axis=0, keepdims=True)
    w = np.exp(lg)
    w /= w.sum(axis=0, keepdims=True)
    pos = sum(w[i][..., None] * targets[i] for i in range(len(targets)))
    return (pos - np.stack([gx, gy], axis=-1)).astype(np.float32)


def warp_image(src: PuppetSource, disp: np.ndarray) -> np.ndarray:
    h, w = src.image.shape[:2]
    d = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xs.copy(), ys.copy()
    for _ in range(6):  # inverse approchée : x = y − D(x)
        dx = cv2.remap(d[..., 0], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dy = cv2.remap(d[..., 1], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x, y = xs - dx, ys - dy
    return cv2.remap(src.image, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def make_source(image: np.ndarray, joints: dict, step: int = 8) -> PuppetSource:
    h, w = image.shape[:2]
    gx, gy = np.meshgrid(np.arange(0, w + step, step, dtype=np.float32),
                         np.arange(0, h + step, step, dtype=np.float32))
    torso = float(np.linalg.norm(joints["neck"] - joints["pelvis"]))
    return PuppetSource(image, joints, step, gx, gy, sigma=max(8.0, 0.22 * torso))


# ----------------------------------------------------------------------------- détecteur

class PoseDetector:
    """MediaPipe Pose Landmarker (mode image pour la référence, mode vidéo pour le pilote)."""

    def __init__(self, model_path: Path) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - dépend de l'installation
            raise NoPoseError("MediaPipe n'est pas installé : le mode corps entier est indisponible.") from exc
        self._mp = mp
        self._vision = vision
        self._mpt = mpt
        self.path = str(model_path)
        self._img = self._make(vision.RunningMode.IMAGE)
        self._vid = self._make(vision.RunningMode.VIDEO)
        self._ts = 0
        self._smooth: Optional[np.ndarray] = None

    def _make(self, mode):
        opts = self._vision.PoseLandmarkerOptions(
            base_options=self._mpt.BaseOptions(model_asset_path=self.path),
            running_mode=mode, num_poses=1,
            min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4, min_tracking_confidence=0.4,
        )
        return self._vision.PoseLandmarker.create_from_options(opts)

    def _to_px(self, result, w: int, h: int) -> Optional[np.ndarray]:
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in lm], np.float32)

    def reference(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return self._to_px(self._img.detect(img), rgb.shape[1], rgb.shape[0])

    def driving(self, rgb: np.ndarray, fps: float) -> Optional[np.ndarray]:
        self._ts += int(round(1000.0 / max(1.0, fps))) or 1
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        pts = self._to_px(self._vid.detect_for_video(img, self._ts), rgb.shape[1], rgb.shape[0])
        if pts is None:
            return None
        self._smooth = pts if self._smooth is None else 0.55 * pts + 0.45 * self._smooth
        return self._smooth.copy()

    def reset(self) -> None:
        self._ts = 0
        self._smooth = None

    def close(self) -> None:
        for lm in (self._img, self._vid):
            try:
                lm.close()
            except Exception:  # noqa: BLE001
                pass


class PuppetEngine:
    """Même interface que ``LivePortrait`` pour la chaîne vidéo."""

    def __init__(self, model_path: Path, fps: float = 25.0) -> None:
        self.detector = PoseDetector(model_path)
        self.fps = fps

    def prepare_source(self, rgb: np.ndarray) -> PuppetSource:
        lm = self.detector.reference(rgb)
        if lm is None:
            raise NoPoseError("Aucune personne détectée dans l'image de référence (il faut un corps visible, de face).")
        return make_source(rgb, joints_from_landmarks(lm))

    def reset_tracking(self) -> None:
        self.detector.reset()

    def driving_motion(self, rgb: np.ndarray, redetect: bool = False):
        lm = self.detector.driving(rgb, self.fps)
        return None if lm is None else joints_from_landmarks(lm)

    def render(self, src: PuppetSource, cur: dict, first: dict, multiplier: float = 1.0) -> np.ndarray:
        new = retarget(src.joints, cur, first, multiplier)
        return warp_image(src, build_displacement(src, new))
