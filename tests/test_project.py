import json

import pytest

from vmc.camera import Keyframe
from vmc.presets import normalize_fps
from vmc.project import Project, ProjectError


def test_roundtrip_with_relative_paths(tmp_path):
    media = tmp_path / "media" / "clip.mp4"
    media.parent.mkdir()
    proj = Project(source=str(media), keyframes=[Keyframe(0), Keyframe(2, zoom=1.4, interp=[0.2, 0, 0.2, 1])])
    proj.output.path = str(tmp_path / "out" / "result.mp4")
    proj.output.fps = 29.97
    path = tmp_path / "p.json"
    proj.save(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["source"] == "media/clip.mp4"      # stored relative, portable between machines
    assert raw["output"]["path"] == "out/result.mp4"

    loaded = Project.load(path)
    assert loaded.source == str(media.resolve())
    assert loaded.output.fps == 29.97
    assert loaded.keyframes[1].interp == (0.2, 0.0, 0.2, 1.0)
    assert loaded.keyframes[1].zoom == 1.4


@pytest.mark.parametrize(
    "mutation",
    [
        {"output": {"preset": "nope"}},
        {"output": {"fit": "stretch"}},
        {"output": {"quality": "insane"}},
        {"output": {"width": 4}},
        {"motion_blur": {"shutter_angle": 400}},
        {"trim": {"start": 5, "end": 2}},
        {"camera": {"keyframes": [{"t": 0, "zoom": -1}]}},
        {"camera": {"keyframes": [{"x": 0.5}]}},
        {"camera": {"keyframes": [{"t": 0, "interp": "bounce"}]}},
        {"version": 99},
    ],
)
def test_invalid_projects_are_rejected(mutation):
    with pytest.raises(ProjectError):
        Project.from_dict({"source": "a.mp4", **mutation})


def test_bad_files(tmp_path):
    with pytest.raises(ProjectError):
        Project.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(ProjectError):
        Project.load(bad)


def test_normalize_fps_snaps_ntsc():
    assert str(normalize_fps(29.97)) == "30000/1001"
    assert str(normalize_fps(23.976)) == "24000/1001"
    assert str(normalize_fps(59.94)) == "60000/1001"
    assert str(normalize_fps(25)) == "25"
    assert str(normalize_fps("30000/1001")) == "30000/1001"
