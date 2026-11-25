"""Modèles ONNX de LivePortrait : emplacement, téléchargement (reprise possible) et vérification.

Les poids ne sont PAS embarqués dans le .exe : ils sont téléchargés une seule fois, au premier
lancement, dans ``%LOCALAPPDATA%\\VideoMotionControl\\models``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

REPO = "warmshao/FasterLivePortrait"
SUBDIR = "liveportrait_onnx"
# Miroir principal puis secours.
BASES = (
    f"https://huggingface.co/{REPO}/resolve/main/{SUBDIR}/",
    f"https://hf-mirror.com/{REPO}/resolve/main/{SUBDIR}/",
)

# nom de fichier -> (rôle, taille minimale plausible en octets)
MODEL_FILES: dict[str, tuple[str, int]] = {
    "retinaface_det_static.onnx": ("détection du visage", 1_000_000),
    "face_2dpose_106_static.onnx": ("points du visage (106)", 1_000_000),
    "landmark.onnx": ("points du visage (203)", 1_000_000),
    "appearance_feature_extractor.onnx": ("apparence", 1_000_000),
    "motion_extractor.onnx": ("extraction du mouvement", 1_000_000),
    "warping_spade.onnx": ("génération de l'image", 10_000_000),
    "stitching.onnx": ("raccord", 10_000),
    "stitching_lip.onnx": ("raccord des lèvres", 10_000),
}


class ModelError(RuntimeError):
    pass


class Cancelled(Exception):
    pass


def models_dir() -> Path:
    override = os.environ.get("VMC_MODELS")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "VideoMotionControl" / "models" / "liveportrait"


def missing(directory: Optional[Path] = None) -> list[str]:
    d = directory or models_dir()
    out = []
    for name, (_, min_size) in MODEL_FILES.items():
        p = d / name
        if not p.is_file() or p.stat().st_size < min_size:
            out.append(name)
    return out


def is_ready(directory: Optional[Path] = None) -> bool:
    return not missing(directory)


ProgressFn = Callable[[str, int, int, int, int], None]  # name, file_idx, n_files, done, total


def _download(url: str, dest: Path, on_bytes: Callable[[int, int], None], cancel: Optional[threading.Event]) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "VideoMotionControl"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        status = getattr(resp, "status", 200)
        if have and status != 206:  # le serveur ignore Range : on repart de zéro
            have = 0
        length = resp.headers.get("Content-Length")
        total = (int(length) + have) if length else 0
        mode = "ab" if have else "wb"
        with open(part, mode) as fh:
            done = have
            while True:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                on_bytes(done, total)
    if total and part.stat().st_size != total:
        raise ModelError(f"téléchargement incomplet : {dest.name}")
    part.replace(dest)


def ensure_models(
    directory: Optional[Path] = None,
    progress: Optional[ProgressFn] = None,
    cancel: Optional[threading.Event] = None,
) -> Path:
    """Télécharge les modèles manquants. Lève ``ModelError`` si le réseau est indisponible."""
    d = directory or models_dir()
    d.mkdir(parents=True, exist_ok=True)
    todo = missing(d)
    for i, name in enumerate(todo, 1):
        last_error: Optional[BaseException] = None
        for base in BASES:
            for _attempt in range(3):
                try:
                    def on_bytes(done: int, total: int, _n=name, _i=i) -> None:
                        if progress:
                            progress(_n, _i, len(todo), done, total)

                    _download(base + name, d / name, on_bytes, cancel)
                    last_error = None
                    break
                except Cancelled:
                    raise
                except (urllib.error.URLError, OSError, ModelError) as exc:
                    last_error = exc
            if last_error is None:
                break
        if last_error is not None:
            raise ModelError(
                f"Impossible de télécharger {name} ({last_error}).\n"
                f"Vérifiez la connexion Internet, ou copiez les fichiers .onnx à la main dans :\n{d}"
            )
    still = missing(d)
    if still:
        raise ModelError("Modèles invalides ou manquants : " + ", ".join(still))
    return d


def import_from_folder(source: Path, directory: Optional[Path] = None) -> list[str]:
    """Copie des .onnx déjà téléchargés (installation hors ligne)."""
    d = directory or models_dir()
    d.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in MODEL_FILES:
        for cand in (source / name, source / SUBDIR / name):
            if cand.is_file():
                shutil.copy2(cand, d / name)
                copied.append(name)
                break
    return copied


POSE_NAME = "pose_landmarker_full.task"
POSE_URLS = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/"
    + POSE_NAME,
)


def pose_model_path() -> Path:
    return models_dir().parent / "pose" / POSE_NAME


def ensure_pose_model(progress: Optional[ProgressFn] = None, cancel: Optional[threading.Event] = None) -> Path:
    """Modèle de pose (mode corps entier), téléchargé une seule fois."""
    dest = pose_model_path()
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Optional[BaseException] = None
    for url in POSE_URLS:
        for _ in range(3):
            try:
                _download(url, dest, lambda d, t: progress and progress(POSE_NAME, 1, 1, d, t), cancel)
                return dest
            except Cancelled:
                raise
            except (urllib.error.URLError, OSError, ModelError) as exc:
                last = exc
    raise ModelError(f"Impossible de télécharger le modèle de pose ({last}).\nCopiez {POSE_NAME} dans :\n{dest.parent}")
