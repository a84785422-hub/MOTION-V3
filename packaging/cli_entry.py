"""PyInstaller entry point for vmc.exe (console)."""

import sys

from vmc.cli import main

if __name__ == "__main__":
    sys.exit(main())
