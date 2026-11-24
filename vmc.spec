# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: builds dist/VideoMotionControl/ for Windows x64.

Two executables share one folder (and therefore one copy of numpy / OpenCV / Qt):
  * VideoMotionControl.exe : graphical interface (no console window)
  * vmc.exe                : command line interface

Build with:  pyinstaller --noconfirm --clean vmc.spec
FFmpeg must be present in vendor/ffmpeg (run scripts/fetch_ffmpeg.ps1).
"""
import os
from pathlib import Path

ROOT = Path(SPECPATH).resolve()  # noqa: F821 - provided by PyInstaller
FFMPEG_DIR = ROOT / "vendor" / "ffmpeg"
ICON = ROOT / "assets" / "icon.ico"

# --- bundled data --------------------------------------------------------------------------
datas = []
missing = []
for exe in ("ffmpeg.exe", "ffprobe.exe"):
    path = FFMPEG_DIR / exe
    if path.is_file():
        datas.append((str(path), "ffmpeg"))
    else:
        missing.append(exe)
for extra in ("LICENSE", "THIRD_PARTY_NOTICES.md", "README.md"):
    if (ROOT / extra).is_file():
        datas.append((str(ROOT / extra), "."))
MASK = ROOT / "src" / "vmc" / "ai" / "mask_template.png"
if MASK.is_file():
    datas.append((str(MASK), "vmc/ai"))
else:
    raise SystemExit("src/vmc/ai/mask_template.png est manquant")
license_txt = FFMPEG_DIR / "LICENSE.txt"
if license_txt.is_file():
    datas.append((str(license_txt), "ffmpeg"))

if missing and os.environ.get("VMC_ALLOW_NO_FFMPEG") != "1":
    raise SystemExit(
        "FFmpeg is missing from vendor/ffmpeg (" + ", ".join(missing) + "). "
        "Run scripts/fetch_ffmpeg.ps1 first, or set VMC_ALLOW_NO_FFMPEG=1 to build without it."
    )

# --- exclusions (keeps the bundle small) ---------------------------------------------------
COMMON_EXCLUDES = [
    "tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest", "imageio_ffmpeg",
]
QT_EXCLUDES = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets", "PySide6.Qt3DCore", "PySide6.Qt3DRender",
    "PySide6.Qt3DInput", "PySide6.Qt3DLogic", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtQuick3D",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtSql", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtSensors", "PySide6.QtSerialPort",
    "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtLocation",
    "PySide6.QtPositioning", "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtSpatialAudio",
    "PySide6.QtTextToSpeech", "PySide6.QtHttpServer", "PySide6.QtGraphs",
]

# --- analyses ------------------------------------------------------------------------------
gui_a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "gui_entry.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    hiddenimports=["onnxruntime"],
    excludes=COMMON_EXCLUDES + QT_EXCLUDES,
    noarchive=False,
)
# The console tool never imports Qt, so keep it (and its pure-python archive) Qt-free.
cli_a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "cli_entry.py")],
    pathex=[str(ROOT / "src")],
    datas=[(str(MASK), "vmc/ai")],
    hiddenimports=["onnxruntime"],
    excludes=COMMON_EXCLUDES + ["PySide6", "shiboken6", "vmc.gui"],
    noarchive=False,
)

gui_pyz = PYZ(gui_a.pure)  # noqa: F821
cli_pyz = PYZ(cli_a.pure)  # noqa: F821

icon = str(ICON) if ICON.is_file() else None

gui_exe = EXE(  # noqa: F821
    gui_pyz,
    gui_a.scripts,
    [],
    exclude_binaries=True,
    name="VideoMotionControl",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon,
)
cli_exe = EXE(  # noqa: F821
    cli_pyz,
    cli_a.scripts,
    [],
    exclude_binaries=True,
    name="vmc",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    icon=icon,
)

coll = COLLECT(  # noqa: F821
    gui_exe,
    cli_exe,
    gui_a.binaries,
    gui_a.datas,
    cli_a.binaries,
    cli_a.datas,
    strip=False,
    upx=False,
    name="VideoMotionControl",
)
