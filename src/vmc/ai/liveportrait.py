"""Moteur LivePortrait en ONNX Runtime (CPU), en NumPy/OpenCV purs (sans PyTorch).

Porté depuis FasterLivePortrait (MIT, warmshao) et LivePortrait (MIT, KwaiVGI).
Une image de référence (visage) est animée avec les mouvements de tête et d'expression
d'une vidéo pilote (transfert de mouvement relatif).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from math import acos, cos, sin
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

NUM_KP = 21


class NoFaceError(RuntimeError):
    pass


def _asset(name: str) -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here / name, Path(getattr(sys, "_MEIPASS", here)) / "vmc" / "ai" / name):
        if cand.is_file():
            return cand
    raise FileNotFoundError(name)


# ----------------------------------------------------------------------------- géométrie

def _pts_apply(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    return pts @ m[:2, :2].T + m[:2, 2]


def _pt2_from_landmarks(pts: np.ndarray) -> np.ndarray:
    n = pts.shape[0]
    if n == 106:
        left = np.mean(pts[[33, 35, 40, 39]], axis=0)
        right = np.mean(pts[[87, 89, 94, 93]], axis=0)
        lip = (pts[52] + pts[61]) / 2
    elif n == 203:
        left = np.mean(pts[[0, 6, 12, 18]], axis=0)
        right = np.mean(pts[[24, 30, 36, 42]], axis=0)
        lip = (pts[48] + pts[66]) / 2
    else:
        raise ValueError(f"nombre de points inattendu : {n}")
    return np.stack([(left + right) / 2, lip], axis=0)


def _rect_from_landmarks(pts: np.ndarray, scale: float, vx_ratio: float, vy_ratio: float):
    pt2 = _pt2_from_landmarks(pts).astype(np.float32)
    uy = pt2[1] - pt2[0]
    length = float(np.linalg.norm(uy))
    uy = np.array([0, 1], dtype=np.float32) if length <= 1e-3 else uy / length
    ux = np.array((uy[1], -uy[0]), dtype=np.float32)
    angle = acos(float(np.clip(ux[0], -1, 1)))
    if ux[1] < 0:
        angle = -angle
    m = np.array([ux, uy])
    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ m.T
    lt, rb = np.min(rpts, axis=0), np.max(rpts, axis=0)
    center1 = (lt + rb) / 2
    size = rb - lt
    side = max(size[0], size[1])
    size = np.array([side, side], dtype=np.float32) * scale
    center = center0 + ux * center1[0] + uy * center1[1]
    center = center + ux * (vx_ratio * size) + uy * (vy_ratio * size)
    return center, size, angle


def crop_by_landmarks(img, pts, dsize, scale, vx_ratio=0.0, vy_ratio=-0.1, rotate=True):
    """Retourne (recadrage, M_origine→recadrage 3x3, M_recadrage→origine 3x3)."""
    center, size, angle = _rect_from_landmarks(np.asarray(pts, np.float32), scale, vx_ratio, vy_ratio)
    s = dsize / float(size[0])
    t = dsize / 2.0
    cx, cy = float(center[0]), float(center[1])
    if rotate:
        c, si = cos(angle), sin(angle)
        m = np.array([[s * c, s * si, t - s * (c * cx + si * cy)],
                      [-s * si, s * c, t - s * (-si * cx + c * cy)]], dtype=np.float32)
    else:
        m = np.array([[s, 0, t - s * cx], [0, s, t - s * cy]], dtype=np.float32)
    crop = cv2.warpAffine(img, m, (dsize, dsize), flags=cv2.INTER_LINEAR)
    m3 = np.vstack([m, [0, 0, 1]]).astype(np.float64)
    return crop, m3, np.linalg.inv(m3)


def rotation_matrix(pitch, yaw, roll) -> np.ndarray:
    """Angles en degrés, formes (B,1) → (B,3,3) (convention LivePortrait)."""
    x, y, z = (np.asarray(a, np.float64).reshape(-1, 1) * np.pi / 180 for a in (pitch, yaw, roll))
    bs = x.shape[0]
    o, zr = np.ones((bs, 1)), np.zeros((bs, 1))
    rx = np.concatenate([o, zr, zr, zr, np.cos(x), -np.sin(x), zr, np.sin(x), np.cos(x)], 1).reshape(bs, 3, 3)
    ry = np.concatenate([np.cos(y), zr, np.sin(y), zr, o, zr, -np.sin(y), zr, np.cos(y)], 1).reshape(bs, 3, 3)
    rz = np.concatenate([np.cos(z), -np.sin(z), zr, np.sin(z), np.cos(z), zr, zr, zr, o], 1).reshape(bs, 3, 3)
    return np.transpose(np.matmul(rz, np.matmul(ry, rx)), (0, 2, 1))


def transform_keypoints(pitch, yaw, roll, t, exp, scale, kp) -> np.ndarray:
    bs = kp.shape[0]
    rot = rotation_matrix(pitch, yaw, roll)
    out = kp.reshape(bs, -1, 3) @ rot + exp.reshape(bs, -1, 3)
    out = out * scale.reshape(bs, 1, 1)
    out[:, :, 0:2] += t.reshape(bs, 1, -1)[:, :, 0:2]
    return out.astype(np.float32)


def _ratio(lmk, a, b, c, d, eps=1e-6):
    return (np.linalg.norm(lmk[:, a] - lmk[:, b], axis=1, keepdims=True)
            / (np.linalg.norm(lmk[:, c] - lmk[:, d], axis=1, keepdims=True) + eps))


def lip_close_ratio(lmk203: np.ndarray) -> np.ndarray:
    return _ratio(lmk203, 90, 102, 48, 66)


def _headpose_to_degree(pred: np.ndarray) -> np.ndarray:
    if pred.ndim > 1 and pred.shape[1] == 66:
        e = np.exp(pred - pred.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return np.sum(p * np.arange(66), axis=1) * 3 - 97.5
    return pred


def _concat(x, y) -> np.ndarray:
    bs = x.shape[0]
    return np.concatenate([x.reshape(bs, -1), y.reshape(bs, -1)], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- ONNX

def _session(path: Path, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False  # peu de RAM : pas de pré-allocation agressive
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class _Net:
    def __init__(self, path: Path, threads: int) -> None:
        self.s = _session(path, threads)
        self.in_names = [i.name for i in self.s.get_inputs()]
        self.out_names = [o.name for o in self.s.get_outputs()]

    def __call__(self, *arrays):
        return self.run_named({n: a for n, a in zip(self.in_names, arrays)})

    def run_named(self, feed):
        return self.s.run(None, {n: np.ascontiguousarray(a, dtype=np.float32) for n, a in feed.items()})


class FaceFinder:
    """RetinaFace (SCRFD) + points 106 : trouve le visage principal."""

    def __init__(self, det: _Net, pose: _Net) -> None:
        self.det, self.pose = det, pose
        self.size = 512
        self._centers: dict = {}

    def _centers_for(self, stride: int):
        key = (self.size, stride)
        if key not in self._centers:
            n = self.size // stride
            c = np.stack(np.mgrid[:n, :n][::-1], axis=-1).astype(np.float32) * stride
            self._centers[key] = np.stack([c.reshape(-1, 2)] * 2, axis=1).reshape(-1, 2)
        return self._centers[key]

    def detect(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        ratio = h / w
        if ratio > 1:
            nh, nw = self.size, int(self.size / ratio)
        else:
            nw, nh = self.size, int(self.size * ratio)
        det_scale = nh / h
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32)
        blob = (blob - 127.5) / 128.0
        outs = self.det(blob)
        groups: dict[int, list[np.ndarray]] = {1: [], 4: [], 10: []}
        for o in outs:
            o = np.asarray(o).reshape(-1, np.asarray(o).shape[-1])
            if o.shape[1] in groups:
                groups[o.shape[1]].append(o)
        for k in groups:  # stride 8 (le plus de lignes) en premier
            groups[k].sort(key=lambda a: -a.shape[0])
        scores_l, boxes_l = [], []
        for idx, stride in enumerate((8, 16, 32)):
            sc, bb = groups[1][idx], groups[4][idx] * stride
            c = self._centers_for(stride)
            keep = np.where(sc[:, 0] >= 0.5)[0]
            box = np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1], c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], -1)
            scores_l.append(sc[keep])
            boxes_l.append(box[keep])
        scores = np.vstack(scores_l)
        if scores.shape[0] == 0:
            return None
        boxes = np.vstack(boxes_l) / det_scale
        dets = np.hstack([boxes, scores]).astype(np.float32)
        order = dets[:, 4].argsort()[::-1]
        dets = dets[order]
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        return dets[int(np.argmax(areas))]  # plus grand visage

    def landmarks106(self, bgr: np.ndarray, bbox) -> np.ndarray:
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        s = 192 / (max(w, h) * 1.5)
        m = np.array([[s, 0, 96 - cx * s], [0, s, 96 - cy * s]], np.float32)
        aimg = cv2.warpAffine(bgr, m, (192, 192), borderValue=0.0)
        blob = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32)
        pred = self.pose(blob)[0].reshape(-1, 2)
        if pred.shape[0] > 106:
            pred = pred[-106:]
        pred = (pred + 1) * 96
        return _pts_apply(pred, cv2.invertAffineTransform(m)).astype(np.float32)

    def find(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        box = self.detect(bgr)
        return None if box is None else self.landmarks106(bgr, box[:4])


# ----------------------------------------------------------------------------- moteur

@dataclass
class Motion:
    pitch: np.ndarray
    yaw: np.ndarray
    roll: np.ndarray
    t: np.ndarray
    exp: np.ndarray
    scale: np.ndarray
    kp: np.ndarray
    R: np.ndarray


@dataclass
class Source:
    image: np.ndarray       # RGB uint8, taille de sortie
    motion: Motion
    f_s: np.ndarray
    x_s: np.ndarray
    lip_delta: Optional[np.ndarray]
    mask: np.ndarray        # float32 HxWx3, masque de recollage dans l'image d'origine
    m_c2o: np.ndarray


class LivePortrait:
    SRC_SIZE, SRC_SCALE, SRC_VY = 512, 2.3, -0.125
    DRI_SCALE, DRI_VY = 2.2, -0.1
    LIP_THRESHOLD = 0.03

    def __init__(self, model_dir: Path, threads: Optional[int] = None) -> None:
        threads = threads or max(1, (os.cpu_count() or 2))
        d = Path(model_dir)
        self.finder = FaceFinder(_Net(d / "retinaface_det_static.onnx", threads),
                                 _Net(d / "face_2dpose_106_static.onnx", threads))
        self.landmark = _Net(d / "landmark.onnx", threads)
        self.app = _Net(d / "appearance_feature_extractor.onnx", threads)
        self.motion = _Net(d / "motion_extractor.onnx", threads)
        self.warp = _Net(d / "warping_spade.onnx", threads)
        self.stitch = _Net(d / "stitching.onnx", threads)
        self.stitch_lip = _Net(d / "stitching_lip.onnx", threads)
        self.mask_crop = cv2.imread(str(_asset("mask_template.png")), cv2.IMREAD_COLOR)
        if self.mask_crop is None:
            raise FileNotFoundError("mask_template.png")
        self._track: Optional[np.ndarray] = None

    # -- étapes élémentaires -------------------------------------------------
    def _landmarks203(self, rgb: np.ndarray, lmk: np.ndarray) -> np.ndarray:
        crop, _m, m_c2o = crop_by_landmarks(rgb, lmk, 224, 1.5, 0.0, -0.1)
        blob = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.landmark(blob)
        pts = np.asarray(out[2]).reshape(-1, 2) * 224
        return _pts_apply(pts, m_c2o).astype(np.float32)

    def _extract_motion(self, crop256_rgb: np.ndarray) -> Motion:
        blob = (crop256_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        outs = self.motion(blob)
        named = dict(zip(self.motion.out_names, outs))
        if {"pitch", "yaw", "roll", "t", "exp", "scale", "kp"} <= set(named):
            pitch, yaw, roll, t, exp, scale, kp = (named[k] for k in ("pitch", "yaw", "roll", "t", "exp", "scale", "kp"))
        else:  # ordre du modèle exporté : pitch, yaw, roll, t, exp, scale, kp
            pitch, yaw, roll, t, exp, scale, kp = outs
        pitch = _headpose_to_degree(pitch)[:, None]
        yaw = _headpose_to_degree(yaw)[:, None]
        roll = _headpose_to_degree(roll)[:, None]
        kp = kp.reshape(1, -1, 3)
        exp = exp.reshape(1, -1, 3)
        return Motion(pitch, yaw, roll, t, exp, scale, kp, rotation_matrix(pitch, yaw, roll))

    def _stitching(self, x_s, x_d):
        delta = self.stitch(_concat(x_s, x_d))[0]
        n = x_s.shape[1]
        out = x_d.copy()
        out += delta[..., : 3 * n].reshape(1, n, 3)
        out[..., :2] += delta[..., 3 * n: 3 * n + 2].reshape(1, 1, 2)
        return out

    # -- référence -----------------------------------------------------------
    def prepare_source(self, rgb: np.ndarray) -> Source:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lmk106 = self.finder.find(bgr)
        if lmk106 is None:
            raise NoFaceError("Aucun visage détecté dans l'image de référence.")
        crop, _m_o2c, m_c2o = crop_by_landmarks(rgb, lmk106, self.SRC_SIZE, self.SRC_SCALE, 0.0, self.SRC_VY)
        lmk203 = self._landmarks203(rgb, lmk106)
        crop256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
        mo = self._extract_motion(crop256)
        f_s = self.app((crop256.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])[0]
        x_s = transform_keypoints(mo.pitch, mo.yaw, mo.roll, mo.t, mo.exp, mo.scale, mo.kp)
        lip_delta = None
        ratio = np.concatenate([lip_close_ratio(lmk203[None]), np.array([[0.05]])], axis=1)
        if ratio[0, 0] >= self.LIP_THRESHOLD:
            lip_delta = self.stitch_lip(_concat(x_s, ratio))[0].reshape(-1, x_s.shape[1], 3)
        h, w = rgb.shape[:2]
        mask = cv2.warpAffine(self.mask_crop, m_c2o[:2], (w, h)).astype(np.float32) / 255.0
        return Source(rgb, mo, f_s, x_s, lip_delta, mask, m_c2o)

    # -- vidéo pilote --------------------------------------------------------
    def reset_tracking(self) -> None:
        self._track = None

    def driving_motion(self, rgb: np.ndarray, redetect: bool = False) -> Optional[Motion]:
        """Mouvement d'une image pilote ; ``None`` si aucun visage n'est trouvé."""
        lmk = self._track
        if lmk is None or redetect:
            found = self.finder.find(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if found is not None:
                lmk = found
        if lmk is None:
            return None
        lmk203 = self._landmarks203(rgb, lmk)
        self._track = lmk203
        crop, _a, _b = crop_by_landmarks(rgb, lmk203, 256, self.DRI_SCALE, 0.0, self.DRI_VY, rotate=False)
        return self._extract_motion(crop)

    # -- synthèse ------------------------------------------------------------
    def _warp(self, f_s, x_d, x_s) -> np.ndarray:
        names = self.warp.in_names
        wanted = {"feature_3d": f_s, "kp_driving": x_d, "kp_source": x_s}
        if set(names) == set(wanted):
            return self.warp.run_named(wanted)[0]
        return self.warp(f_s, x_d, x_s)[0]  # ordre du modèle exporté

    def render(self, src: Source, cur: Motion, first: Motion, multiplier: float = 1.0) -> np.ndarray:
        s = src.motion
        r_new = (cur.R @ np.transpose(first.R, (0, 2, 1))) @ s.R
        delta = s.exp + (cur.exp - first.exp)
        scale = s.scale * (cur.scale / first.scale)
        t = s.t + (cur.t - first.t)
        t = t.copy()
        t[..., 2] = 0
        x_d = scale.reshape(1, 1, 1) * (s.kp @ r_new + delta) + t.reshape(1, 1, -1)
        x_d = x_d.astype(np.float32)
        x_d = self._stitching(src.x_s, x_d)
        if src.lip_delta is not None:
            x_d = x_d + src.lip_delta
        x_d = src.x_s + (x_d - src.x_s) * multiplier
        out = self._warp(src.f_s, x_d, src.x_s)
        crop = np.clip(out[0].transpose(1, 2, 0), 0, 1) * 255.0
        h, w = src.image.shape[:2]
        back = cv2.warpAffine(crop.astype(np.float32), src.m_c2o[:2], (w, h), flags=cv2.INTER_LINEAR)
        blended = src.mask * back + (1.0 - src.mask) * src.image.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)
