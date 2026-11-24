import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vmc.video_io import find_tool  # noqa: E402


@pytest.fixture(scope="session")
def ffmpeg() -> str:
    path = find_tool("ffmpeg")
    if not path:
        pytest.skip("FFmpeg not available")
    return path


@pytest.fixture(scope="session")
def test_video(tmp_path_factory, ffmpeg) -> str:
    """A short synthetic clip with audio (640x360, 30 fps, 1.5 s)."""
    import subprocess

    out = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "1.5", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "12", "-c:a", "aac", str(out),
        ],
        check=True,
    )
    return str(out)
