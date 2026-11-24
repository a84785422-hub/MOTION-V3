"""End-to-end tests: real FFmpeg encode/decode of small synthetic clips."""

import json
import subprocess
import threading

import numpy as np
import pytest

from vmc.camera import Keyframe
from vmc.export import ExportCancelled, ExportError, export_project
from vmc.project import Project
from vmc.video_io import VideoDecoder, available_encoders, find_tool, grab_frame, probe


def _ffprobe(path):
    tool = find_tool("ffprobe")
    if not tool:
        pytest.skip("ffprobe not available")
    out = subprocess.run(
        [tool, "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, check=True,
    ).stdout
    return json.loads(out)["streams"]


def _project(source, out, **kw):
    p = Project(source=source)
    p.output.path = str(out)
    p.output.quality = "draft"
    p.keyframes = [Keyframe(0, interp="linear"), Keyframe(1.0, x=0.55, zoom=1.2)]
    for k, v in kw.items():
        setattr(p.output, k, v)
    return p


def test_probe_video(test_video):
    info = probe(test_video)
    assert (info.width, info.height) == (640, 360)
    assert float(info.fps) == 30.0
    assert info.has_audio and not info.is_image
    assert info.duration == pytest.approx(1.5, abs=0.1)


def test_probe_falls_back_to_ffmpeg_when_ffprobe_missing(test_video, monkeypatch):
    import vmc.video_io as vio

    real = vio.find_tool
    monkeypatch.setattr(vio, "find_tool", lambda name: None if name == "ffprobe" else real(name))
    info = vio.probe(test_video)
    assert (info.width, info.height) == (640, 360)
    assert float(info.fps) == 30.0 and info.has_audio
    assert info.duration == pytest.approx(1.5, abs=0.1)
    assert info.pix_fmt == "yuv420p"


def test_decoder_returns_expected_frames(test_video):
    info = probe(test_video)
    with VideoDecoder(info, start=0.5, duration=0.5) as dec:
        frames = list(dec)
    assert len(frames) == 15
    assert frames[0].shape == (360, 640, 3) and frames[0].dtype == np.uint8
    assert grab_frame(info, 1.0, max_height=180).shape == (180, 320, 3)


def test_h264_export_with_audio(test_video, tmp_path):
    out = tmp_path / "out.mp4"
    seen = []
    res = export_project(_project(test_video, out), progress=lambda *a: seen.append(a))
    assert res.frames == 45 and seen[-1][0] == seen[-1][1] == 45
    streams = _ffprobe(out)
    v = next(s for s in streams if s["codec_type"] == "video")
    assert (v["codec_name"], v["width"], v["height"], v["pix_fmt"]) == ("h264", 640, 360, "yuv420p")
    assert v["nb_frames"] == "45" and v["r_frame_rate"] == "30/1"
    assert v.get("color_space") == "bt709"
    assert any(s["codec_type"] == "audio" for s in streams)


def test_roundtrip_colours_stay_close_with_identity_camera(test_video, tmp_path):
    out = tmp_path / "id.mp4"
    p = _project(test_video, out)
    p.keyframes = []
    p.motion_blur.enabled = False
    export_project(p)
    a = grab_frame(probe(test_video), 0.5)
    b = grab_frame(probe(str(out)), 0.5)
    err = np.abs(a.astype(float) - b.astype(float))
    assert err.mean() < 3.0, "colour matrix mismatch between decode and encode"


def test_resize_trim_and_fps_override(test_video, tmp_path):
    out = tmp_path / "small.mp4"
    p = _project(test_video, out, width=320, height=180, fps=24.0)
    p.start, p.end = 0.25, 1.25
    p.audio = False
    export_project(p)
    streams = _ffprobe(out)
    assert len(streams) == 1
    v = streams[0]
    assert (v["width"], v["height"], v["r_frame_rate"], v["nb_frames"]) == (320, 180, "24/1", "24")


@pytest.mark.parametrize(
    "preset,codec,pix_fmt",
    [("prores_422hq", "prores", "yuv422p10le"), ("h265_10bit", "hevc", "yuv420p10le")],
)
def test_ten_bit_presets(test_video, tmp_path, preset, codec, pix_fmt):
    needed = {"prores_422hq": "prores_ks", "h265_10bit": "libx265"}[preset]
    if needed not in available_encoders():
        pytest.skip(f"{needed} encoder not available")
    out = tmp_path / ("o.mov" if preset.startswith("prores") else "o.mp4")
    p = _project(test_video, out, preset=preset, width=320, height=180)
    p.end = 0.5
    export_project(p)
    v = next(s for s in _ffprobe(out) if s["codec_type"] == "video")
    assert (v["codec_name"], v["pix_fmt"]) == (codec, pix_fmt)


def test_still_image_source(tmp_path, ffmpeg):
    import cv2

    img = np.zeros((300, 500, 3), np.uint8)
    img[:, :250] = (255, 0, 0)
    png = tmp_path / "still.png"
    cv2.imwrite(str(png), img)
    out = tmp_path / "still.mp4"
    p = _project(str(png), out, width=640, height=360, fps=25.0)
    p.still_duration = 1.0
    export_project(p)
    v = next(s for s in _ffprobe(out) if s["codec_type"] == "video")
    assert (v["width"], v["height"], v["nb_frames"]) == (640, 360, "25")


def test_cancel_removes_partial_output(test_video, tmp_path):
    out = tmp_path / "cancel.mp4"
    cancel = threading.Event()

    def on_progress(done, total, fps, eta):
        if done >= 4:
            cancel.set()

    with pytest.raises(ExportCancelled):
        export_project(_project(test_video, out), progress=on_progress, cancel=cancel)
    assert not out.exists()


def test_errors_are_clear(test_video, tmp_path):
    with pytest.raises(ExportError):
        export_project(_project(test_video, test_video))  # would overwrite the source
    with pytest.raises(ExportError):
        export_project(_project(str(tmp_path / "missing.mp4"), tmp_path / "x.mp4"))


def _make_clip(ffmpeg, path, *, rate="30", pix_fmt="yuv420p", codec=("libx264",), seconds="1", extra=()):
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc2=size=640x360:rate={rate}", "-t", seconds, "-pix_fmt", pix_fmt,
         "-c:v", *codec, *extra, str(path)],
        check=True,
    )
    return str(path)


def test_ten_bit_ntsc_source_keeps_exact_frame_rate(tmp_path, ffmpeg):
    if "libx265" not in available_encoders():
        pytest.skip("libx265 not available")
    src = _make_clip(ffmpeg, tmp_path / "ten.mp4", rate="30000/1001", pix_fmt="yuv420p10le",
                     codec=("libx265",), extra=("-x265-params", "log-level=error"))
    info = probe(src)
    assert info.bit_depth == 10 and str(info.fps) == "30000/1001"
    with VideoDecoder(info, max_frames=1) as dec:
        assert next(iter(dec)).dtype == np.uint16   # high bit depth is decoded as 16-bit
    out = tmp_path / "ten_out.mp4"
    export_project(_project(src, out))
    v = next(s for s in _ffprobe(out) if s["codec_type"] == "video")
    assert v["r_frame_rate"] == "30000/1001" and v["nb_frames"] == "30"


def test_rotated_phone_footage_is_displayed_upright(tmp_path, ffmpeg):
    plain = _make_clip(ffmpeg, tmp_path / "plain.mp4")
    rotated = tmp_path / "rot.mp4"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90",
                    "-i", plain, "-c", "copy", str(rotated)], check=True)
    info = probe(str(rotated))
    assert (info.width, info.height) == (360, 640)
    out = tmp_path / "rot_out.mp4"
    p = _project(str(rotated), out)
    p.keyframes = []
    export_project(p)
    v = next(s for s in _ffprobe(out) if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (360, 640)


def test_sixteen_bit_png_to_prores(tmp_path, ffmpeg):
    import cv2

    if "prores_ks" not in available_encoders():
        pytest.skip("prores_ks not available")
    ramp = np.tile(np.linspace(0, 65535, 800, dtype=np.float32), (450, 1))
    png = tmp_path / "ramp.png"
    cv2.imwrite(str(png), np.dstack([ramp] * 3).astype(np.uint16))
    assert probe(str(png)).bit_depth == 16
    out = tmp_path / "ramp.mov"
    p = _project(str(png), out, preset="prores_422hq", width=640, height=360, fps=25.0)
    p.still_duration = 0.4
    export_project(p)
    v = next(s for s in _ffprobe(out) if s["codec_type"] == "video")
    assert (v["codec_name"], v["pix_fmt"], v["nb_frames"]) == ("prores", "yuv422p10le", "10")
