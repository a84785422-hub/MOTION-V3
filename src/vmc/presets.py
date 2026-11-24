"""Export presets: FFmpeg codec settings for the supported delivery formats."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    extension: str                 # output container extension, with dot
    codec_args: tuple[str, ...]
    pix_fmt: str                   # pixel format written to the file
    pipe_bits: int                 # 8 -> rgb24 pipe, 16 -> rgb48le pipe
    audio_args: tuple[str, ...]
    hardware: bool = False         # needs a GPU encoder
    description: str = ""

    @property
    def pipe_pix_fmt(self) -> str:
        return "rgb48le" if self.pipe_bits == 16 else "rgb24"


PRESETS: dict[str, Preset] = {
    p.key: p
    for p in (
        Preset(
            "h264_high",
            "H.264 haute qualité (MP4, CRF 14)",
            ".mp4",
            ("-c:v", "libx264", "-preset", "slow", "-crf", "14", "-profile:v", "high"),
            "yuv420p",
            8,
            ("-c:a", "aac", "-b:a", "320k"),
            description="Compatible partout (web, réseaux sociaux, lecteurs).",
        ),
        Preset(
            "h265_10bit",
            "H.265 10 bits (MP4, CRF 16)",
            ".mp4",
            ("-c:v", "libx265", "-preset", "medium", "-crf", "16", "-tag:v", "hvc1",
             "-x265-params", "log-level=error"),
            "yuv420p10le",
            16,
            ("-c:a", "aac", "-b:a", "320k"),
            description="Fichiers plus légers, dégradés sans banding grâce au 10 bits.",
        ),
        Preset(
            "prores_422hq",
            "Apple ProRes 422 HQ (MOV, 10 bits)",
            ".mov",
            ("-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-bits_per_mb", "8000"),
            "yuv422p10le",
            16,
            ("-c:a", "pcm_s24le"),
            description="Master de montage (DaVinci Resolve, Premiere, Final Cut).",
        ),
        Preset(
            "prores_4444",
            "Apple ProRes 4444 (MOV, 10 bits)",
            ".mov",
            ("-c:v", "prores_ks", "-profile:v", "4", "-vendor", "apl0", "-bits_per_mb", "8000"),
            "yuv444p10le",
            16,
            ("-c:a", "pcm_s24le"),
            description="Chroma pleine résolution, idéal pour l'étalonnage et les incrustations.",
        ),
        Preset(
            "ffv1_lossless",
            "FFV1 sans perte (MKV, 10 bits)",
            ".mkv",
            ("-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1", "-slicecrc", "1"),
            "yuv444p10le",
            16,
            ("-c:a", "flac"),
            description="Archivage sans perte (fichiers volumineux).",
        ),
        Preset(
            "h264_nvenc",
            "H.264 NVENC (GPU NVIDIA, MP4)",
            ".mp4",
            ("-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "16",
             "-b:v", "0", "-profile:v", "high"),
            "yuv420p",
            8,
            ("-c:a", "aac", "-b:a", "320k"),
            hardware=True,
            description="Encodage rapide sur carte graphique NVIDIA.",
        ),
        Preset(
            "hevc_nvenc",
            "H.265 NVENC 10 bits (GPU NVIDIA, MP4)",
            ".mp4",
            ("-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "18",
             "-b:v", "0", "-profile:v", "main10", "-tag:v", "hvc1"),
            "p010le",
            16,
            ("-c:a", "aac", "-b:a", "320k"),
            hardware=True,
            description="Encodage 10 bits rapide sur carte graphique NVIDIA récente.",
        ),
    )
}

DEFAULT_PRESET = "h264_high"

_NTSC = (
    Fraction(24000, 1001),
    Fraction(30000, 1001),
    Fraction(60000, 1001),
    Fraction(120000, 1001),
)


def normalize_fps(value: float | Fraction | str) -> Fraction:
    """Convert a frame rate to an exact fraction, snapping NTSC rates (23.976, 29.97, 59.94...)."""
    if isinstance(value, str):
        value = Fraction(value)
    frac = value if isinstance(value, Fraction) else Fraction(float(value))
    approx = float(frac)
    if approx <= 0:
        raise ValueError("frame rate must be positive")
    for ntsc in _NTSC:
        if abs(approx - float(ntsc)) < 0.01:
            return ntsc
    return frac.limit_denominator(1000)
