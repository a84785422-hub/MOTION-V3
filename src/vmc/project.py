"""Project model: everything needed to reproduce a render, stored as a small JSON file."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .camera import FIT_MODES, Keyframe
from .presets import DEFAULT_PRESET, PRESETS
from .render import QUALITIES

PROJECT_VERSION = 1


class ProjectError(ValueError):
    """Raised for invalid project files or settings."""


@dataclass
class OutputSettings:
    path: str = ""
    width: Optional[int] = None      # None = source width
    height: Optional[int] = None     # None = source height
    fps: Optional[float] = None      # None = source frame rate (30 for stills)
    preset: str = DEFAULT_PRESET
    fit: str = "cover"
    quality: str = "high"


@dataclass
class MotionBlurSettings:
    enabled: bool = True
    shutter_angle: float = 180.0


@dataclass
class Project:
    source: str = ""
    start: float = 0.0
    end: Optional[float] = None       # None = until the end of the source
    still_duration: float = 5.0       # used when the source is a still image
    output: OutputSettings = field(default_factory=OutputSettings)
    motion_blur: MotionBlurSettings = field(default_factory=MotionBlurSettings)
    keyframes: list[Keyframe] = field(default_factory=list)
    linear_light: bool = True
    dither: bool = True
    audio: bool = True

    # ------------------------------------------------------------------ validation

    def validate(self) -> None:
        o = self.output
        if o.preset not in PRESETS:
            raise ProjectError(f"unknown preset {o.preset!r} (available: {', '.join(PRESETS)})")
        if o.fit not in FIT_MODES:
            raise ProjectError(f"fit must be one of {FIT_MODES}")
        if o.quality not in QUALITIES:
            raise ProjectError(f"quality must be one of {QUALITIES}")
        for name, val in (("width", o.width), ("height", o.height)):
            if val is not None and (not isinstance(val, int) or val < 16 or val > 16384):
                raise ProjectError(f"output {name} must be an integer between 16 and 16384")
        if o.fps is not None and not (0.1 <= float(o.fps) <= 480):
            raise ProjectError("output fps must be between 0.1 and 480")
        if not (0.0 <= self.motion_blur.shutter_angle <= 360.0):
            raise ProjectError("shutter angle must be between 0 and 360 degrees")
        if self.start < 0:
            raise ProjectError("start must be >= 0")
        if self.end is not None and self.end <= self.start:
            raise ProjectError("end must be greater than start")
        if self.still_duration <= 0:
            raise ProjectError("still_duration must be > 0")

    # ------------------------------------------------------------------ (de)serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROJECT_VERSION,
            "source": self.source,
            "trim": {"start": self.start, "end": self.end},
            "still_duration": self.still_duration,
            "output": {
                "path": self.output.path,
                "width": self.output.width,
                "height": self.output.height,
                "fps": self.output.fps,
                "preset": self.output.preset,
                "fit": self.output.fit,
                "quality": self.output.quality,
            },
            "motion_blur": {
                "enabled": self.motion_blur.enabled,
                "shutter_angle": self.motion_blur.shutter_angle,
            },
            "color": {"linear_light": self.linear_light, "dither": self.dither},
            "audio": self.audio,
            "camera": {"keyframes": [k.to_dict() for k in sorted(self.keyframes, key=lambda k: k.t)]},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        if not isinstance(data, dict):
            raise ProjectError("project file must contain a JSON object")
        version = data.get("version", PROJECT_VERSION)
        if version > PROJECT_VERSION:
            raise ProjectError(f"project version {version} is newer than this program supports")
        try:
            trim = data.get("trim", {}) or {}
            out = data.get("output", {}) or {}
            mb = data.get("motion_blur", {}) or {}
            color = data.get("color", {}) or {}
            kfs = [Keyframe.from_dict(k) for k in (data.get("camera", {}) or {}).get("keyframes", [])]
            project = cls(
                source=str(data.get("source", "")),
                start=float(trim.get("start", 0.0) or 0.0),
                end=None if trim.get("end") is None else float(trim["end"]),
                still_duration=float(data.get("still_duration", 5.0)),
                output=OutputSettings(
                    path=str(out.get("path", "")),
                    width=None if out.get("width") is None else int(out["width"]),
                    height=None if out.get("height") is None else int(out["height"]),
                    fps=None if out.get("fps") is None else float(out["fps"]),
                    preset=str(out.get("preset", DEFAULT_PRESET)),
                    fit=str(out.get("fit", "cover")),
                    quality=str(out.get("quality", "high")),
                ),
                motion_blur=MotionBlurSettings(
                    enabled=bool(mb.get("enabled", True)),
                    shutter_angle=float(mb.get("shutter_angle", 180.0)),
                ),
                keyframes=kfs,
                linear_light=bool(color.get("linear_light", True)),
                dither=bool(color.get("dither", True)),
                audio=bool(data.get("audio", True)),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ProjectError(f"invalid project: {exc}") from exc
        project.validate()
        return project

    # ------------------------------------------------------------------ files

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Project":
        """Load a project; relative paths inside it are resolved against the file's folder."""
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ProjectError(f"cannot read {p}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProjectError(f"{p} is not valid JSON: {exc}") from exc
        project = cls.from_dict(data)
        base = p.resolve().parent
        project.source = _resolve(project.source, base)
        project.output.path = _resolve(project.output.path, base)
        return project

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the project; paths are stored relative to the project file when possible."""
        p = Path(path)
        base = p.resolve().parent
        data = self.to_dict()
        data["source"] = _relativize(self.source, base)
        data["output"]["path"] = _relativize(self.output.path, base)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve(value: str, base: Path) -> str:
    if not value:
        return value
    p = Path(value)
    return str(p if p.is_absolute() else (base / p).resolve())


def _relativize(value: str, base: Path) -> str:
    if not value:
        return value
    try:
        rel = os.path.relpath(value, base)
    except ValueError:  # different drive on Windows
        return value
    if rel.startswith(".."):
        return value
    return rel.replace(os.sep, "/")
