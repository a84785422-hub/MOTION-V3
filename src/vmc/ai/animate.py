"""Chaîne complète : image de référence + vidéo pilote → vidéo animée (MP4, audio conservé)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ..export import ExportCancelled, ExportError
from ..presets import normalize_fps
from ..video_io import MediaError, VideoDecoder, VideoEncoder, load_image, probe, require_ffmpeg
from . import models as model_store
from .liveportrait import LivePortrait, NoFaceError

ProgressCallback = Callable[[int, int, float, float], None]  # done, total, fps, eta_seconds


@dataclass
class AnimateOptions:
    reference: str
    driving: str
    output: str
    max_output_side: int = 1024     # plus grand côté de la vidéo de sortie (la RAM et le temps en dépendent)
    max_fps: float = 25.0           # cadence maximale traitée
    driving_height: int = 540       # hauteur de décodage de la vidéo pilote
    start: float = 0.0
    duration: Optional[float] = None
    multiplier: float = 1.0         # intensité du mouvement (1 = fidèle)
    audio: bool = True
    crf: int = 16
    threads: Optional[int] = None
    mode: str = "face"             # "face" (LivePortrait, visage/tête) ou "body" (marionnette corps entier)


@dataclass
class AnimateResult:
    output: str
    frames: int
    seconds: float


def _fit_reference(img: np.ndarray, max_side: int) -> np.ndarray:
    if img.dtype != np.uint8:
        img = (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    if k < 1.0:
        img = cv2.resize(img, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    return np.ascontiguousarray(img[: h - h % 2, : w - w % 2])


def _encode_command(opts: AnimateOptions, w: int, h: int, fps: Fraction, audio_from: Optional[str]) -> list[str]:
    cmd = [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "-"]
    if audio_from:
        if opts.start > 0:
            cmd += ["-ss", f"{opts.start:.6f}"]
        if opts.duration is not None:
            cmd += ["-t", f"{opts.duration:.6f}"]
        cmd += ["-i", audio_from]
    cmd += ["-map", "0:v:0"]
    if audio_from:
        cmd += ["-map", "1:a:0"]
    cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(opts.crf), "-profile:v", "high"]
    cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    if audio_from:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-movflags", "+faststart", opts.output]
    return cmd


def animate(
    opts: AnimateOptions,
    progress: Optional[ProgressCallback] = None,
    cancel: Optional[threading.Event] = None,
    engine: Optional[LivePortrait] = None,
) -> AnimateResult:
    """Anime la référence avec le mouvement de la vidéo pilote. Lève ``ExportError`` / ``ExportCancelled``."""
    try:
        ref_info = probe(opts.reference)
        dri_info = probe(opts.driving)
    except MediaError as exc:
        raise ExportError(str(exc)) from exc
    if dri_info.is_image:
        raise ExportError("La vidéo pilote doit être une vidéo, pas une image.")
    try:
        reference = _fit_reference(load_image(opts.reference), opts.max_output_side)
    except MediaError as exc:
        raise ExportError(str(exc)) from exc
    del ref_info

    if engine is None:
        try:
            if opts.mode == "body":
                from .puppet import NoPoseError, PuppetEngine

                engine = PuppetEngine(model_store.ensure_pose_model(cancel=cancel))
            else:
                directory = model_store.ensure_models(cancel=cancel)
                engine = LivePortrait(directory, opts.threads)
        except model_store.Cancelled:
            raise ExportCancelled()
        except model_store.ModelError as exc:
            raise ExportError(str(exc)) from exc

    try:
        source = engine.prepare_source(reference)
    except Exception as exc:  # NoFaceError / NoPoseError
        if type(exc).__name__ not in ("NoFaceError", "NoPoseError"):
            raise
        raise ExportError(str(exc)) from exc

    src_fps = dri_info.fps or Fraction(25)
    fps = src_fps if float(src_fps) <= opts.max_fps else normalize_fps(opts.max_fps)
    if hasattr(engine, "fps"):
        engine.fps = float(fps)
    duration = dri_info.duration or 0.0
    span = max(0.0, duration - opts.start)
    if opts.duration is not None:
        span = min(span, opts.duration)
    total = max(1, int(round(span * float(fps)))) if span else (dri_info.frames or 1)

    h, w = reference.shape[:2]
    decoder = VideoDecoder(dri_info, start=opts.start, duration=opts.duration, fps=fps,
                           max_height=opts.driving_height)
    audio_from = opts.driving if (opts.audio and dri_info.has_audio) else None
    encoder = VideoEncoder(_encode_command(opts, w, h, fps, audio_from))

    engine.reset_tracking()
    first = None
    last_motion = None
    done = 0
    t0 = time.monotonic()
    try:
        for i, frame in enumerate(decoder):
            if cancel is not None and cancel.is_set():
                raise ExportCancelled()
            if frame.dtype != np.uint8:
                frame = (frame.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
            cur = engine.driving_motion(frame, redetect=(i % 30 == 0))
            if cur is None:
                if first is None:
                    continue  # on attend un premier visage
                cur = last_motion
            if first is None:
                first = cur
            last_motion = cur
            encoder.write(engine.render(source, cur, first, opts.multiplier))
            done += 1
            if progress is not None:
                el = max(1e-6, time.monotonic() - t0)
                rate = done / el
                progress(done, max(total, done), rate, max(0, total - done) / rate)
        if first is None:
            raise ExportError("Aucune personne détectée dans la vidéo pilote." if opts.mode == "body" else "Aucun visage détecté dans la vidéo pilote.")
        encoder.close()
    except BaseException:
        encoder.abort()
        try:
            Path(opts.output).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        decoder.close()
    return AnimateResult(opts.output, done, time.monotonic() - t0)
