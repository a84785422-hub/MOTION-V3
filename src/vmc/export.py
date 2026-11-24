"""Render pipeline: decode -> camera/motion blur -> encode, shared by export and previews."""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from .camera import CameraTrack
from .presets import PRESETS, Preset, normalize_fps
from .project import Project, ProjectError
from .render import Renderer, RenderSettings, SourceFrame
from .video_io import (
    MediaError,
    MediaInfo,
    VideoDecoder,
    VideoEncoder,
    available_encoders,
    load_image,
    prefetch,
    probe,
    require_ffmpeg,
)

ProgressCallback = Callable[[int, int, float, float], None]  # done, total, fps, eta_seconds

DEFAULT_STILL_FPS = 30


class ExportError(RuntimeError):
    """Raised when an export cannot start or fails."""


class ExportCancelled(Exception):
    """Raised internally when the user cancels an export."""


@dataclass(frozen=True)
class Geometry:
    """Resolved output geometry and timing for a project."""

    out_w: int
    out_h: int
    fps: Fraction
    start: float
    duration: float
    total_frames: int


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def resolve_geometry(project: Project, info: MediaInfo) -> Geometry:
    o = project.output
    out_w = _even(o.width) if o.width else _even(info.width)
    out_h = _even(o.height) if o.height else _even(info.height)
    if o.fps:
        fps = normalize_fps(o.fps)
    elif info.fps:
        fps = normalize_fps(info.fps)
    else:
        fps = Fraction(DEFAULT_STILL_FPS)
    if info.is_image:
        start = 0.0
        duration = project.still_duration
    else:
        start = project.start
        if info.duration is None:
            if project.end is None:
                raise ExportError("durée de la vidéo inconnue : définissez un temps de fin (trim.end)")
            end = project.end
        else:
            end = min(project.end, info.duration) if project.end is not None else info.duration
        duration = end - start
        if duration <= 0:
            raise ExportError("la plage à exporter est vide (début >= fin)")
    total = max(1, int(round(duration * float(fps))))
    return Geometry(out_w, out_h, fps, start, duration, total)


def render_settings_for(project: Project, out_bits: int, quality: Optional[str] = None) -> RenderSettings:
    return RenderSettings(
        fit=project.output.fit,
        quality=quality or project.output.quality,
        linear_light=project.linear_light,
        dither=project.dither,
        motion_blur=project.motion_blur.enabled,
        shutter_angle=project.motion_blur.shutter_angle,
        out_bits=out_bits,
    )


def iter_rendered_frames(
    project: Project,
    info: MediaInfo,
    geometry: Geometry,
    settings: RenderSettings,
    out_size: Optional[tuple[int, int]] = None,
    decode_height: Optional[int] = None,
    cancel: Optional[threading.Event] = None,
    shutter_dt: Optional[float] = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, rendered_frame)`` for the whole range of the project.

    ``shutter_dt`` is the frame duration used to size the motion-blur window. Previews that
    run at a lower frame rate pass the *final* frame duration so the blur matches the export.
    """
    track = CameraTrack(project.keyframes)
    ow, oh = out_size or (geometry.out_w, geometry.out_h)
    renderer = Renderer(ow, oh, settings)
    frame_step = 1.0 / float(geometry.fps)
    blur_dt = shutter_dt or frame_step
    linear = settings.linear_light

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise ExportCancelled()

    if info.is_image:
        src = SourceFrame(load_image(info.path), linear)
        for i in range(geometry.total_frames):
            check_cancel()
            yield i, renderer.render(src, track, i * frame_step, blur_dt)
        return

    decoder = VideoDecoder(
        info,
        start=geometry.start,
        duration=geometry.duration,
        fps=geometry.fps,
        max_height=decode_height,
    )
    try:
        for i, frame in enumerate(prefetch(decoder, depth=3)):
            if i >= geometry.total_frames:
                break
            check_cancel()
            yield i, renderer.render(SourceFrame(frame, linear), track, i * frame_step, blur_dt)
    finally:
        decoder.close()


def build_encode_command(
    preset: Preset,
    project: Project,
    info: MediaInfo,
    geometry: Geometry,
    output: str,
) -> list[str]:
    fps = f"{geometry.fps.numerator}/{geometry.fps.denominator}"
    cmd = [
        require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", preset.pipe_pix_fmt,
        "-s", f"{geometry.out_w}x{geometry.out_h}", "-framerate", fps, "-i", "-",
    ]
    with_audio = project.audio and info.has_audio and not info.is_image
    if with_audio:
        cmd += ["-ss", f"{geometry.start:.6f}", "-t", f"{geometry.duration:.6f}", "-i", info.path]
    cmd += ["-map", "0:v:0"]
    if with_audio:
        cmd += ["-map", "1:a:0"]
    vf = (
        "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,"
        f"format={preset.pix_fmt}"
    )
    cmd += ["-vf", vf, *preset.codec_args]
    cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    if with_audio:
        cmd += [*preset.audio_args, "-shortest"]
    if preset.extension in (".mp4", ".mov"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(output)
    return cmd


def check_project(project: Project, info: MediaInfo) -> list[str]:
    """Human-readable warnings about a project (does not raise)."""
    from .camera import find_out_of_frame_times

    warnings: list[str] = []
    try:
        geo = resolve_geometry(project, info)
    except ExportError as exc:
        return [str(exc)]
    if not project.keyframes:
        warnings.append("Aucune image clé : la caméra restera immobile.")
    else:
        track = CameraTrack(project.keyframes)
        times = np.linspace(0.0, max(geo.duration, 1e-6), 41)
        bad = find_out_of_frame_times(track, info.size, (geo.out_w, geo.out_h), project.output.fit, times)
        if bad:
            warnings.append(
                f"La vue sort de l'image source entre {bad[0]:.2f}s et {bad[-1]:.2f}s : "
                "les bords seront étirés. Augmentez le zoom ou recentrez la caméra."
            )
    preset = PRESETS[project.output.preset]
    if preset.hardware and not (set(preset.codec_args) & available_encoders()):
        warnings.append(f"L'encodeur matériel du préréglage « {preset.label} » semble indisponible.")
    return warnings


@dataclass
class ExportResult:
    output: str
    frames: int
    seconds: float


def export_project(
    project: Project,
    progress: Optional[ProgressCallback] = None,
    cancel: Optional[threading.Event] = None,
    info: Optional[MediaInfo] = None,
) -> ExportResult:
    """Render a project to a video file. Raises ``ExportError`` / ``ExportCancelled``."""
    try:
        project.validate()
    except ProjectError as exc:
        raise ExportError(str(exc)) from exc
    if not project.source:
        raise ExportError("aucun média source n'est défini")
    if not project.output.path:
        raise ExportError("aucun fichier de sortie n'est défini")
    try:
        info = info or probe(project.source)
    except MediaError as exc:
        raise ExportError(str(exc)) from exc

    preset = PRESETS[project.output.preset]
    out_path = project.output.path
    if Path(out_path).suffix.lower() != preset.extension:
        out_path = str(Path(out_path).with_suffix(preset.extension))
    if os.path.abspath(out_path) == os.path.abspath(project.source):
        raise ExportError("le fichier de sortie ne peut pas être le fichier source")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    geo = resolve_geometry(project, info)
    settings = render_settings_for(project, preset.pipe_bits)
    cmd = build_encode_command(preset, project, info, geo, out_path)

    started = time.monotonic()
    try:
        encoder = VideoEncoder(cmd)
    except OSError as exc:
        raise ExportError(f"impossible de lancer FFmpeg : {exc}") from exc

    done = 0
    try:
        for i, frame in iter_rendered_frames(project, info, geo, settings, cancel=cancel):
            encoder.write(frame)
            done = i + 1
            if progress is not None and (done == geo.total_frames or done % 2 == 0 or done <= 2):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = done / elapsed
                progress(done, geo.total_frames, rate, (geo.total_frames - done) / rate if rate else math.inf)
        if done == 0:
            raise ExportError("aucune image n'a pu être décodée depuis la source")
        encoder.close()
    except ExportCancelled:
        encoder.abort()
        _remove_quietly(out_path)
        raise
    except MediaError as exc:
        encoder.abort()
        _remove_quietly(out_path)
        raise ExportError(str(exc)) from exc
    except BaseException:
        encoder.abort()
        _remove_quietly(out_path)
        raise
    return ExportResult(out_path, done, time.monotonic() - started)


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
