"""GUI entry point: ``VideoMotionControl.exe`` and ``python -m vmc gui``."""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Sequence


def _log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "state")
    d = Path(base) / "VideoMotionControl"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = Path(tempfile.gettempdir())
    return d


def _ensure_std_streams() -> Path:
    """Windowed (no console) builds have no stdout/stderr: send them to a log file."""
    log = _log_dir() / "log.txt"
    if sys.stdout is None or sys.stderr is None:
        handle = open(log, "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = handle
        if sys.stderr is None:
            sys.stderr = handle
    return log


def _apply_dark_theme(app) -> None:
    from PySide6 import QtGui

    app.setStyle("Fusion")
    role, group = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup
    c = QtGui.QColor
    pal = QtGui.QPalette()
    colors = {
        role.Window: c(37, 37, 43),
        role.WindowText: c(225, 225, 230),
        role.Base: c(26, 26, 31),
        role.AlternateBase: c(37, 37, 43),
        role.ToolTipBase: c(50, 50, 58),
        role.ToolTipText: c(235, 235, 240),
        role.Text: c(225, 225, 230),
        role.Button: c(52, 52, 60),
        role.ButtonText: c(225, 225, 230),
        role.BrightText: c(255, 90, 90),
        role.Link: c(255, 190, 40),
        role.Highlight: c(255, 170, 20),
        role.HighlightedText: c(20, 20, 20),
    }
    for r, color in colors.items():
        pal.setColor(r, color)
    for r in (role.Text, role.ButtonText, role.WindowText):
        pal.setColor(group.Disabled, r, c(110, 110, 118))
    app.setPalette(pal)


def _make_selftest_clip(directory: str) -> Optional[str]:
    import subprocess

    from ..video_io import find_tool

    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        return None
    out = str(Path(directory) / "selftest.mp4")
    flags = 0x08000000 if os.name == "nt" else 0
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=640x360:rate=30", "-t", "1", "-pix_fmt", "yuv420p", "-c:v", "libx264", out],
        check=True, creationflags=flags,
    )
    return out


def _finish(code: int, hard_exit: bool) -> int:
    """Return ``code`` or, for the packaged app, end the process right away.

    Letting Python tear down Qt objects at interpreter shutdown can crash on Windows
    (QApplication destroyed before its widgets, threads still alive...), so the executable
    flushes its streams and exits directly once everything the user cares about is done.
    """
    if not hard_exit:
        return code
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except (OSError, ValueError):
            pass
    os._exit(code)


def run(argv: Optional[Sequence[str]] = None, *, hard_exit: bool = False) -> int:
    args = list(argv if argv is not None else sys.argv)
    selftest = "--selftest" in args
    args = [a for a in args if a != "--selftest"]
    if selftest:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    log = _ensure_std_streams()

    from PySide6 import QtWidgets

    from .main_window import MainWindow

    app = QtWidgets.QApplication(args)
    app.setApplicationName("Video Motion Control")
    _apply_dark_theme(app)

    def excepthook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError:
            pass
        if not selftest:
            QtWidgets.QMessageBox.critical(None, "Erreur inattendue", f"{exc}\n\nDétails dans :\n{log}")

    sys.excepthook = excepthook
    window = MainWindow()

    if selftest:
        result_file = os.environ.get("VMC_SELFTEST_LOG")
        message = "OK"
        code = 0
        try:
            with tempfile.TemporaryDirectory() as tmp:
                clip = _make_selftest_clip(tmp)
                if clip is None:
                    message = "FAIL: ffmpeg not found"
                    code = 1
                else:
                    window.show()
                    app.processEvents()
                    err = window.selftest(clip)
                    app.processEvents()
                    if not err:
                        from .animate_dialog import AnimateDialog

                        dlg = AnimateDialog(window)
                        dlg.show()
                        app.processEvents()
                        dlg.close()
                    if err:
                        message, code = f"FAIL: {err}", 1
                    window.close()
        except Exception:  # noqa: BLE001 - reported through the log file
            message, code = "FAIL:\n" + traceback.format_exc(), 1
        if result_file:
            Path(result_file).write_text(message + "\n", encoding="utf-8")
        print(message)
        return _finish(code, hard_exit)

    window.show()
    if len(args) > 1 and os.path.isfile(args[1]):
        if args[1].lower().endswith(".json"):
            window.open_project(args[1])
        else:
            window.open_media(args[1])
    return _finish(app.exec(), hard_exit)
