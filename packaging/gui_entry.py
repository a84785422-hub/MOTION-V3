"""PyInstaller entry point for VideoMotionControl.exe (windowed, no console)."""

import sys

from vmc.gui.app import run

if __name__ == "__main__":
    sys.exit(run(hard_exit=True))
