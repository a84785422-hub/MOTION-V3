"""Background workers (run in QThreads) for exports and animated previews."""

from __future__ import annotations

import dataclasses
import threading
from fractions import Fraction

from PySide6 import QtCore
from PySide6.QtCore import Signal, Slot

from ..export import (
    ExportCancelled,
    export_project,
    iter_rendered_frames,
    render_settings_for,
    resolve_geometry,
)
from ..project import Project
from ..video_io import MediaInfo
from .widgets import to_qimage

PREVIEW_FPS = 12
PREVIEW_MAX_SECONDS = 25.0


class ExportWorker(QtCore.QObject):
    progress = Signal(int, int, float, float)   # done, total, frames/s, eta seconds
    finished = Signal(str)                      # output path
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, project: Project) -> None:
        super().__init__()
        self.project = project
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            result = export_project(
                self.project,
                progress=lambda done, total, fps, eta: self.progress.emit(done, total, fps, eta),
                cancel=self.cancel_event,
            )
        except ExportCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(str(exc))
            return
        self.finished.emit(result.output)

    def cancel(self) -> None:
        self.cancel_event.set()


class PreviewWorker(QtCore.QObject):
    """Renders a low-resolution, draft-quality version of the whole move for playback."""

    progress = Signal(int, int)
    ready = Signal(object, float)   # list[QImage], playback fps
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, project: Project, info: MediaInfo, height: int = 360) -> None:
        super().__init__()
        self.project = project
        self.info = info
        self.height = height
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            geo = resolve_geometry(self.project, self.info)
            fps = Fraction(PREVIEW_FPS)
            duration = min(geo.duration, PREVIEW_MAX_SECONDS)
            total = max(1, int(round(duration * PREVIEW_FPS)))
            geo = dataclasses.replace(geo, fps=fps, duration=duration, total_frames=total)
            ph = min(self.height, geo.out_h)
            pw = max(2, int(round(ph * geo.out_w / geo.out_h / 2)) * 2)
            settings = render_settings_for(self.project, 8, quality="draft")
            settings.dither = False
            frames = []
            final_fps = self.project.output.fps or (float(self.info.fps) if self.info.fps else 30.0)
            for i, frame in iter_rendered_frames(
                self.project, self.info, geo, settings,
                out_size=(pw, ph),
                decode_height=ph * 2,
                cancel=self.cancel_event,
                shutter_dt=1.0 / final_fps,
            ):
                frames.append(to_qimage(frame))
                self.progress.emit(i + 1, total)
        except ExportCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(str(exc))
            return
        self.ready.emit(frames, float(PREVIEW_FPS))

    def cancel(self) -> None:
        self.cancel_event.set()


class AnimateWorker(QtCore.QObject):
    """Télécharge les modèles IA si besoin, puis anime l'image de référence."""

    status = Signal(str)
    progress = Signal(int, int, float, float)
    finished = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, options) -> None:
        super().__init__()
        self.options = options
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        from ..ai import models as store
        from ..ai.animate import animate
        from ..ai.liveportrait import LivePortrait

        try:
            def on_dl(name, idx, n, done, total):
                self.progress.emit(done, max(total, done, 1), 0.0, 0.0)
                self.status.emit(f"Téléchargement {idx}/{n} : {name}")

            if self.options.mode == "body":
                from ..ai.puppet import PuppetEngine

                self.status.emit("Préparation du modèle de pose…")
                engine = PuppetEngine(store.ensure_pose_model(progress=on_dl, cancel=self.cancel_event))
            else:
                if not store.is_ready():
                    self.status.emit("Téléchargement des modèles IA (une seule fois)…")
                    store.ensure_models(progress=on_dl, cancel=self.cancel_event)
                self.status.emit("Chargement des modèles…")
                engine = LivePortrait(store.models_dir(), self.options.threads)
            self.status.emit("Animation en cours…")
            result = animate(
                self.options,
                progress=lambda d, t, f, e: self.progress.emit(d, t, f, e),
                cancel=self.cancel_event,
                engine=engine,
            )
        except (ExportCancelled, store.Cancelled):
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(str(exc))
            return
        self.finished.emit(result.output)

    def cancel(self) -> None:
        self.cancel_event.set()
