"""Custom widgets: image preview with framing overlay, and the keyframe timeline."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal


def to_qimage(arr: np.ndarray) -> QtGui.QImage:
    """Copy an HxWx3 uint8 RGB array into a QImage that owns its memory."""
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    return QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()


class PreviewWidget(QtWidgets.QWidget):
    """Shows an image scaled to fit, optionally with the camera framing drawn on top."""

    def __init__(self, title: str = "", parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
        self._image: Optional[QtGui.QImage] = None
        self._overlay: Optional[list[tuple[float, float]]] = None
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def set_image(self, image: Optional[QtGui.QImage]) -> None:
        self._image = image
        self.update()

    def set_overlay(self, quad: Optional[Sequence[tuple[float, float]]]) -> None:
        """Quad in normalised image coordinates (TL, TR, BR, BL), or None to hide it."""
        self._overlay = list(quad) if quad else None
        self.update()

    def _target_rect(self) -> QtCore.QRectF:
        area = QtCore.QRectF(self.rect()).adjusted(6, 24, -6, -6)
        if self._image is None or self._image.isNull():
            return area
        iw, ih = self._image.width(), self._image.height()
        scale = min(area.width() / iw, area.height() / ih)
        w, h = iw * scale, ih * scale
        return QtCore.QRectF(area.x() + (area.width() - w) / 2, area.y() + (area.height() - h) / 2, w, h)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt API
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(22, 22, 26))
        p.setPen(QtGui.QColor(160, 160, 170))
        p.drawText(8, 16, self._title)
        if self._image is None or self._image.isNull():
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Aucune image")
            return
        target = self._target_rect()
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(target, self._image)
        if self._overlay:
            poly = QtGui.QPolygonF(
                [QtCore.QPointF(target.x() + u * target.width(), target.y() + v * target.height())
                 for u, v in self._overlay]
            )
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            outside = QtGui.QPainterPath()
            outside.addRect(target)
            inner = QtGui.QPainterPath()
            inner.addPolygon(poly)
            inner.closeSubpath()
            p.setClipRect(target)
            p.fillPath(outside.subtracted(inner), QtGui.QColor(0, 0, 0, 120))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 190, 40), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(poly)


class TimelineWidget(QtWidgets.QWidget):
    """Time ruler with keyframe markers and a draggable playhead."""

    seek_requested = Signal(float)

    MARGIN = 16

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._duration = 1.0
        self._time = 0.0
        self._keys: list[float] = []
        self.setMinimumHeight(58)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_duration(self, duration: float) -> None:
        self._duration = max(duration, 1e-3)
        self.update()

    def set_time(self, t: float) -> None:
        self._time = t
        self.update()

    def set_keys(self, times: Sequence[float]) -> None:
        self._keys = list(times)
        self.update()

    def _x(self, t: float) -> float:
        span = max(1.0, self.width() - 2 * self.MARGIN)
        return self.MARGIN + span * t / self._duration

    def _t(self, x: float) -> float:
        span = max(1.0, self.width() - 2 * self.MARGIN)
        return min(max((x - self.MARGIN) / span * self._duration, 0.0), self._duration)

    def _tick_step(self) -> float:
        span = max(1.0, self.width() - 2 * self.MARGIN)
        for step in (0.04, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300):
            if span * step / self._duration >= 70:
                return step
        return 600

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt API
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QtGui.QColor(30, 30, 36))
        track_top, track_bottom = 30, self.height() - 8
        p.fillRect(
            QtCore.QRectF(self.MARGIN, track_top, self.width() - 2 * self.MARGIN, track_bottom - track_top),
            QtGui.QColor(44, 44, 52),
        )
        p.setPen(QtGui.QColor(140, 140, 150))
        step = self._tick_step()
        n = int(self._duration / step) + 1
        for i in range(n + 1):
            t = i * step
            if t > self._duration + 1e-9:
                break
            x = self._x(t)
            p.drawLine(QtCore.QPointF(x, 18), QtCore.QPointF(x, track_top))
            label = f"{t:g}s"
            p.drawText(QtCore.QPointF(x + 3, 14), label)
        # keyframe diamonds
        p.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 1))
        p.setBrush(QtGui.QColor(255, 190, 40))
        cy = (track_top + track_bottom) / 2
        for t in self._keys:
            x = self._x(t)
            d = 7
            p.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(x, cy - d), QtCore.QPointF(x + d, cy),
                QtCore.QPointF(x, cy + d), QtCore.QPointF(x - d, cy),
            ]))
        # playhead
        x = self._x(min(max(self._time, 0.0), self._duration))
        p.setPen(QtGui.QPen(QtGui.QColor(235, 70, 70), 2))
        p.drawLine(QtCore.QPointF(x, 18), QtCore.QPointF(x, self.height() - 2))
        p.setBrush(QtGui.QColor(235, 70, 70))
        p.drawPolygon(QtGui.QPolygonF([
            QtCore.QPointF(x - 6, 18), QtCore.QPointF(x + 6, 18), QtCore.QPointF(x, 27),
        ]))

    def _emit_seek(self, event: QtGui.QMouseEvent) -> None:
        self.seek_requested.emit(self._t(event.position().x()))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._emit_seek(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._emit_seek(event)


def format_time(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--:--"
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    return f"{m:02d}:{seconds - 60 * m:06.3f}"
