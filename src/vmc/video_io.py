"""Media I/O: locating FFmpeg, probing files, streaming raw frames in and out."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

from .presets import normalize_fps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class MediaError(RuntimeError):
    """Raised when a media file cannot be read or an FFmpeg process fails."""


# ----------------------------------------------------------------------------- tool discovery


def _exe(name: str) -> str:
    return name + (".exe" if os.name == "nt" else "")


def _search_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_dir = os.environ.get("VMC_FFMPEG_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass) / "ffmpeg")
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        dirs += [exe_dir / "ffmpeg", exe_dir]
    try:
        dirs.append(Path(__file__).resolve().parents[2] / "vendor" / "ffmpeg")
    except IndexError:
        pass
    return dirs


@lru_cache(maxsize=None)
def find_tool(name: str) -> Optional[str]:
    """Locate ``ffmpeg`` or ``ffprobe``: env override, bundled copy, PATH, then imageio-ffmpeg."""
    override = os.environ.get("VMC_" + name.upper())
    if override and Path(override).is_file():
        return override
    for d in _search_dirs():
        candidate = d / _exe(name)
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg  # type: ignore

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001 - optional dependency, any failure means "not there"
            return None
    return None


def require_ffmpeg() -> str:
    path = find_tool("ffmpeg")
    if not path:
        raise MediaError(
            "FFmpeg introuvable. Placez ffmpeg.exe dans le dossier 'ffmpeg' à côté du programme, "
            "ajoutez-le au PATH, ou définissez la variable VMC_FFMPEG."
        )
    return path


def _run(cmd: Sequence[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        capture_output=True,
        timeout=timeout,
        creationflags=_CREATE_NO_WINDOW,
        stdin=subprocess.DEVNULL,
    )


def available_encoders() -> set[str]:
    """Names of the video encoders compiled into the FFmpeg binary."""
    try:
        out = _run([require_ffmpeg(), "-hide_banner", "-encoders"]).stdout.decode("utf-8", "replace")
    except (MediaError, subprocess.SubprocessError):
        return set()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return names


# ----------------------------------------------------------------------------- probing


@dataclass(frozen=True)
class MediaInfo:
    path: str
    width: int                 # display size (rotation applied)
    height: int
    fps: Optional[Fraction]    # None for still images
    duration: Optional[float]  # seconds; None for still images
    frames: Optional[int]
    pix_fmt: Optional[str]
    bit_depth: int
    has_audio: bool
    is_image: bool
    color_space: Optional[str] = None
    color_range: Optional[str] = None

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def is_image_path(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def load_image(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a still as RGB uint8/uint16 (works with non-ASCII paths on Windows)."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise MediaError(f"cannot read {path}: {exc}") from exc
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise MediaError(f"cannot decode image {path}")
    if img.dtype not in (np.uint8, np.uint16):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8) if img.dtype.kind == "f" else img.astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(img)


def _parse_fraction(text: Optional[str]) -> Optional[Fraction]:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        frac = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return frac if frac > 0 else None


def _bit_depth(pix_fmt: Optional[str], raw_bits: Optional[str]) -> int:
    if raw_bits and raw_bits.isdigit():
        return int(raw_bits)
    if pix_fmt and re.search(r"(10|12|14|16)(le|be)?$", pix_fmt):
        return int(re.search(r"(10|12|14|16)", pix_fmt).group(1))  # type: ignore[union-attr]
    return 8


def probe(path: str | os.PathLike[str]) -> MediaInfo:
    """Read size, frame rate, duration, pixel format and audio presence of a media file."""
    p = str(path)
    if not Path(p).is_file():
        raise MediaError(f"fichier introuvable : {p}")
    if is_image_path(p):
        img = load_image(p)
        return MediaInfo(
            p, img.shape[1], img.shape[0], None, None, None, None,
            16 if img.dtype == np.uint16 else 8, False, True,
        )
    ffprobe = find_tool("ffprobe")
    info = _probe_ffprobe(p, ffprobe) if ffprobe else None
    return info or _probe_ffmpeg_stderr(p)


def _probe_ffprobe(path: str, ffprobe: str) -> Optional[MediaInfo]:
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    try:
        res = _run(cmd)
        data = json.loads(res.stdout.decode("utf-8", "replace") or "{}")
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return None
    streams = data.get("streams", [])
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    if video is None:
        raise MediaError(f"aucun flux vidéo dans {path}")
    width, height = int(video["width"]), int(video["height"])
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    rotation = int(round(float((video.get("tags") or {}).get("rotate", rotation)))) % 360
    if rotation in (90, 270):
        width, height = height, width
    fps = _parse_fraction(video.get("avg_frame_rate")) or _parse_fraction(video.get("r_frame_rate"))
    fmt = data.get("format", {})
    dur_text = video.get("duration") if video.get("duration") not in (None, "N/A") else fmt.get("duration")
    duration = float(dur_text) if dur_text not in (None, "N/A") else None
    nb = video.get("nb_frames")
    frames = int(nb) if isinstance(nb, str) and nb.isdigit() else None
    pix_fmt = video.get("pix_fmt")
    return MediaInfo(
        path, width, height, fps, duration, frames, pix_fmt,
        _bit_depth(pix_fmt, video.get("bits_per_raw_sample")),
        any(s.get("codec_type") == "audio" for s in streams),
        False,
        video.get("color_space"),
        video.get("color_range"),
    )


def _probe_ffmpeg_stderr(path: str) -> MediaInfo:
    res = _run([require_ffmpeg(), "-hide_banner", "-i", path])
    text = res.stderr.decode("utf-8", "replace")
    vline = re.search(r"Stream #\d+:\d+[^\n]*: Video: ([^\n]*)", text)
    if not vline:
        raise MediaError(f"impossible de lire {path} :\n{text.strip()[-400:]}")
    vtext = vline.group(1)
    size = re.search(r",\s*(\d{2,5})x(\d{2,5})[\s,\[]", vtext + " ")
    if not size:
        raise MediaError(f"taille vidéo introuvable pour {path}")
    width, height = int(size.group(1)), int(size.group(2))
    rot = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", text)
    if rot and int(round(abs(float(rot.group(1))))) % 180 == 90:
        width, height = height, width
    fps_m = re.search(r"(\d+(?:\.\d+)?) fps", vtext) or re.search(r"(\d+(?:\.\d+)?) tbr", vtext)
    fps = normalize_fps(float(fps_m.group(1))) if fps_m else None
    dur_m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    duration = (
        int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60 + float(dur_m.group(3)) if dur_m else None
    )
    pix_m = re.search(r",\s*((?:yuv|yuvj|rgb|bgr|gbr|gray|nv|p0)[a-z0-9]*)", vtext)
    pix_fmt = pix_m.group(1) if pix_m else None
    cs_m = re.search(r"\((?:tv|pc|limited|full)?,?\s*(bt709|bt470bg|smpte170m|bt2020nc)", vtext)
    return MediaInfo(
        path, width, height, fps, duration, None, pix_fmt, _bit_depth(pix_fmt, None),
        bool(re.search(r"Stream #\d+:\d+[^\n]*: Audio:", text)), False,
        cs_m.group(1) if cs_m else None,
        "pc" if re.search(r"\(pc[,)]", vtext) else None,
    )


# ----------------------------------------------------------------------------- helpers


def _color_matrix(info: MediaInfo) -> str:
    cs = (info.color_space or "").lower()
    if cs == "bt709":
        return "bt709"
    if cs in ("bt470bg", "smpte170m"):
        return "bt601"
    if cs.startswith("bt2020"):
        return "bt2020"
    # Untagged: follow the usual convention (HD = Rec.709, SD = Rec.601).
    return "bt709" if info.height >= 720 else "bt601"


def _is_yuv_like(pix_fmt: Optional[str]) -> bool:
    return bool(pix_fmt) and pix_fmt.startswith(("yuv", "nv", "p0", "p2", "y2", "uyvy", "yuyv"))  # type: ignore[union-attr]


class _StderrTail:
    """Drains a process's stderr on a thread and keeps the last lines for error messages."""

    def __init__(self, stream) -> None:
        self._lines: deque[str] = deque(maxlen=40)
        self._thread = threading.Thread(target=self._pump, args=(stream,), daemon=True)
        self._thread.start()

    def _pump(self, stream) -> None:
        try:
            for raw in iter(stream.readline, b""):
                self._lines.append(raw.decode("utf-8", "replace").rstrip())
        except (OSError, ValueError):
            pass

    def text(self) -> str:
        self._thread.join(timeout=1.0)
        return "\n".join(self._lines)


def _read_exact(stream, buf: memoryview) -> bool:
    total = len(buf)
    got = 0
    while got < total:
        n = stream.readinto(buf[got:])
        if not n:
            return False
        got += n
    return True


# ----------------------------------------------------------------------------- decoding


class VideoDecoder:
    """Streams decoded RGB frames from a video file through an FFmpeg pipe.

    Frames are ``uint8`` (8-bit sources or scaled previews) or ``uint16`` (high bit depth).
    Colour conversion is done by FFmpeg with an explicit matrix so untagged HD footage is not
    misread as Rec.601.
    """

    def __init__(
        self,
        info: MediaInfo,
        start: float = 0.0,
        duration: Optional[float] = None,
        fps: Optional[Fraction] = None,
        max_height: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        self.info = info
        self.start = max(0.0, start)
        self.duration = duration
        self.fps = fps
        self.max_frames = max_frames
        self.out_w, self.out_h = info.width, info.height
        if max_height and info.height > max_height:
            self.out_h = int(max_height)
            self.out_w = max(2, int(round(info.width * self.out_h / info.height / 2)) * 2)
        self.high_depth = info.bit_depth > 8 and not max_height
        self.dtype = np.dtype("<u2") if self.high_depth else np.dtype(np.uint8)
        self.frame_bytes = self.out_w * self.out_h * 3 * self.dtype.itemsize
        self._proc: Optional[subprocess.Popen] = None
        self._err: Optional[_StderrTail] = None
        self.frames_read = 0

    def command(self) -> list[str]:
        info = self.info
        cmd = [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self.start > 0:
            cmd += ["-ss", f"{self.start:.6f}"]
        cmd += ["-i", info.path]
        if self.duration is not None:
            cmd += ["-t", f"{self.duration:.6f}"]
        cmd += ["-an", "-sn", "-dn"]
        filters = []
        if self.fps is not None:
            filters.append(f"fps={self.fps.numerator}/{self.fps.denominator}")
        scale = []
        if (self.out_w, self.out_h) != (info.width, info.height):
            scale += [f"w={self.out_w}", f"h={self.out_h}"]
        if _is_yuv_like(info.pix_fmt):
            scale.append(f"in_color_matrix={_color_matrix(info)}")
            rng = (info.color_range or "").lower()
            scale.append("in_range=full" if rng == "pc" else "in_range=limited" if rng == "tv" else "in_range=auto")
        scale.append("flags=accurate_rnd+full_chroma_int+full_chroma_inp+bicubic")
        filters.append("scale=" + ":".join(scale))
        filters.append("format=rgb48le" if self.high_depth else "format=rgb24")
        cmd += ["-vf", ",".join(filters)]
        if self.max_frames:
            cmd += ["-frames:v", str(self.max_frames)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb48le" if self.high_depth else "rgb24", "-"]
        return cmd

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        proc.wait()

    def __enter__(self) -> "VideoDecoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        cmd = self.command()
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, creationflags=_CREATE_NO_WINDOW,
        )
        self._err = _StderrTail(self._proc.stderr)
        proc = self._proc
        try:
            while True:
                buf = np.empty(self.frame_bytes, np.uint8)
                if not _read_exact(proc.stdout, memoryview(buf)):
                    break
                self.frames_read += 1
                yield buf.view(self.dtype).reshape(self.out_h, self.out_w, 3)
            rc = proc.wait()
            if rc != 0 and self.frames_read == 0:
                raise MediaError(f"échec du décodage de {self.info.path} :\n{self._err.text()}")
        finally:
            self.close()


def grab_frame(info: MediaInfo, t: float = 0.0, max_height: Optional[int] = None) -> np.ndarray:
    """Decode a single RGB uint8 frame at time ``t`` (used for previews)."""
    if info.is_image:
        img = load_image(info.path)
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        if max_height and img.shape[0] > max_height:
            h = int(max_height)
            w = max(2, int(round(img.shape[1] * h / img.shape[0])))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img
    # A max_height always selects the 8-bit path, even when it equals the source height.
    dec = VideoDecoder(info, start=t, max_height=max_height or info.height, max_frames=1)
    with dec:
        for frame in dec:
            return frame
    # Seeking past the end: fall back to the first frame.
    if t > 0:
        return grab_frame(info, 0.0, max_height)
    raise MediaError(f"impossible de lire une image de {info.path}")


# ----------------------------------------------------------------------------- encoding


class VideoEncoder:
    """Feeds raw RGB frames to an FFmpeg encoder process from a background writer thread."""

    def __init__(self, cmd: Sequence[str], queue_size: int = 4) -> None:
        self.cmd = list(cmd)
        self._proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            bufsize=0, creationflags=_CREATE_NO_WINDOW,
        )
        self._err = _StderrTail(self._proc.stderr)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._failure: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def _writer(self) -> None:
        stdin = self._proc.stdin
        while True:
            item = self._queue.get()
            if item is None:
                break
            if self._failure is not None:
                continue  # keep draining so the producer never blocks
            try:
                stdin.write(memoryview(item).cast("B"))
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._failure = exc

    def _raise_if_failed(self) -> None:
        if self._failure is not None or self._proc.poll() not in (None, 0):
            raise MediaError("l'encodeur FFmpeg s'est arrêté :\n" + self._err.text())

    def write(self, frame: np.ndarray) -> None:
        self._raise_if_failed()
        self._queue.put(np.ascontiguousarray(frame))

    def close(self) -> None:
        """Flush, wait for FFmpeg to finish writing the file and check its exit status."""
        self._queue.put(None)
        self._thread.join()
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        rc = self._proc.wait()
        if rc != 0 or self._failure is not None:
            raise MediaError(f"l'encodeur FFmpeg a échoué (code {rc}) :\n{self._err.text()}")

    def abort(self) -> None:
        """Stop immediately (used on cancel or error); the output file is left to the caller."""
        self._failure = self._failure or RuntimeError("aborted")
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._proc.poll() is None:
            self._proc.kill()
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        self._proc.wait()
        self._thread.join(timeout=2.0)


def prefetch(iterable, depth: int = 3):
    """Iterate ``iterable`` on a background thread, keeping up to ``depth`` items ready."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()
    stop = threading.Event()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def worker() -> None:
        try:
            for item in iterable:
                if not put(item):
                    return
            put(sentinel)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            put(exc)
        finally:
            close = getattr(iterable, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        t.join(timeout=5.0)
