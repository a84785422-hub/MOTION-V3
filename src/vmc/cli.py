"""Command line interface: ``vmc render|init|probe|presets``."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .camera import Keyframe
from .export import ExportCancelled, ExportError, check_project, export_project
from .presets import DEFAULT_PRESET, PRESETS
from .project import MotionBlurSettings, OutputSettings, Project, ProjectError
from .render import QUALITIES
from .video_io import MediaError, probe

MOVES = ("push", "pull", "pan-right", "pan-left", "tilt-up", "tilt-down", "none")


def _fmt_time(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().replace("×", "x").split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("la taille doit être de la forme 1920x1080") from exc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vmc",
        description="Video Motion Control : mouvements de caméra virtuels avec flou de mouvement haute qualité.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", metavar="commande")

    r = sub.add_parser("render", help="exporter un projet (.json) en vidéo")
    r.add_argument("project", help="fichier projet JSON")
    r.add_argument("-o", "--output", help="fichier de sortie (remplace celui du projet)")
    r.add_argument("--preset", choices=list(PRESETS), help="préréglage d'export")
    r.add_argument("--quality", choices=QUALITIES, help="qualité du rendu (draft, high, ultra)")
    r.add_argument("--size", type=_parse_size, metavar="LxH", help="taille de sortie, ex. 3840x2160")
    r.add_argument("--fps", type=float, help="cadence de sortie")
    r.add_argument("--no-blur", action="store_true", help="désactiver le flou de mouvement")
    r.add_argument("--shutter", type=float, help="angle d'obturation en degrés (180 par défaut)")
    r.add_argument("--start", type=float, help="début de la plage en secondes")
    r.add_argument("--end", type=float, help="fin de la plage en secondes")
    r.add_argument("--no-audio", action="store_true", help="ne pas copier l'audio")
    r.add_argument("-q", "--quiet", action="store_true", help="pas de barre de progression")

    i = sub.add_parser("init", help="créer un projet de départ pour un média")
    i.add_argument("media", help="vidéo ou image source")
    i.add_argument("-o", "--output", default="project.json", help="fichier projet à écrire")
    i.add_argument("--move", choices=MOVES, default="push", help="mouvement de caméra de départ")
    i.add_argument("--amount", type=float, default=1.2, help="zoom final (push) ou zoom de travail (pan/tilt)")
    i.add_argument("--duration", type=float, help="durée en secondes (obligatoire pour une image)")
    i.add_argument("--render-to", help="fichier vidéo de sortie à inscrire dans le projet")

    pr = sub.add_parser("probe", help="afficher les informations d'un média")
    pr.add_argument("media")

    a = sub.add_parser("animate", help="IA : animer une image de référence avec les mouvements d'une vidéo")
    a.add_argument("reference", help="image de référence (visage)")
    a.add_argument("driving", help="vidéo dont les mouvements sont copiés")
    a.add_argument("-o", "--output", help="vidéo de sortie (.mp4)")
    a.add_argument("--max-side", type=int, default=1024, help="plus grand côté de la sortie (1024 par défaut)")
    a.add_argument("--max-fps", type=float, default=25.0, help="cadence maximale traitée (25 par défaut)")
    a.add_argument("--start", type=float, default=0.0, help="début dans la vidéo pilote, en secondes")
    a.add_argument("--duration", type=float, help="durée traitée en secondes")
    a.add_argument("--intensity", type=float, default=1.0, help="intensité du mouvement (1 = fidèle)")
    a.add_argument("--no-audio", action="store_true", help="ne pas copier l'audio de la vidéo pilote")
    a.add_argument("--body", action="store_true", help="corps entier (marionnette, approximatif) au lieu du visage")
    a.add_argument("--download-models", action="store_true", help="télécharger les modèles IA puis quitter")
    a.add_argument("--models-from", help="dossier contenant déjà les .onnx (installation hors ligne)")
    a.add_argument("-q", "--quiet", action="store_true", help="pas de barre de progression")

    sub.add_parser("presets", help="lister les préréglages d'export")
    sub.add_parser("gui", help="ouvrir l'interface graphique")
    return p


def _starter_keyframes(move: str, amount: float, duration: float) -> list[Keyframe]:
    z = max(1.0, amount)
    lim = 0.5 / z  # how far the centre can travel while staying inside the frame
    a = Keyframe(0.0)
    b = Keyframe(duration)
    if move == "push":
        b.zoom = z
    elif move == "pull":
        a.zoom, b.zoom = z, 1.0
    elif move in ("pan-right", "pan-left"):
        a.zoom = b.zoom = z
        left, right = lim, 1.0 - lim
        a.x, b.x = (left, right) if move == "pan-right" else (right, left)
    elif move in ("tilt-up", "tilt-down"):
        a.zoom = b.zoom = z
        top, bottom = lim, 1.0 - lim
        a.y, b.y = (bottom, top) if move == "tilt-up" else (top, bottom)
    return [a, b]


def cmd_init(args: argparse.Namespace) -> int:
    info = probe(args.media)
    if info.is_image:
        if not args.duration:
            raise ProjectError("pour une image, précisez la durée avec --duration")
        duration = args.duration
    else:
        duration = args.duration or info.duration
        if not duration:
            raise ProjectError("durée inconnue : précisez --duration")
    out_path = args.render_to or str(Path(args.media).with_name(Path(args.media).stem + "_motion.mp4"))
    project = Project(
        source=str(Path(args.media).resolve()),
        still_duration=duration,
        end=None if info.is_image else duration,
        output=OutputSettings(path=str(Path(out_path).resolve()), preset=DEFAULT_PRESET),
        motion_blur=MotionBlurSettings(),
        keyframes=_starter_keyframes(args.move, args.amount, duration),
    )
    project.save(args.output)
    print(f"Projet créé : {args.output}")
    print(f"Source : {info.width}x{info.height}" + (f" @ {float(info.fps):.3f} fps" if info.fps else " (image)"))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    info = probe(args.media)
    print(f"Fichier      : {info.path}")
    print(f"Taille       : {info.width}x{info.height}")
    if info.is_image:
        print("Type         : image fixe")
    else:
        print(f"Cadence      : {float(info.fps):.3f} fps" if info.fps else "Cadence      : inconnue")
        print(f"Durée        : {info.duration:.3f} s" if info.duration else "Durée        : inconnue")
        print(f"Audio        : {'oui' if info.has_audio else 'non'}")
    print(f"Format pixel : {info.pix_fmt or '-'} ({info.bit_depth} bits)")
    return 0


def cmd_presets(_args: argparse.Namespace) -> int:
    width = max(len(k) for k in PRESETS)
    for key, p in PRESETS.items():
        print(f"{key:<{width}}  {p.label}{'  [GPU]' if p.hardware else ''}")
        if p.description:
            print(f"{'':<{width}}  {p.description}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    project = Project.load(args.project)
    if args.output:
        project.output.path = str(Path(args.output).resolve())
    if args.preset:
        project.output.preset = args.preset
    if args.quality:
        project.output.quality = args.quality
    if args.size:
        project.output.width, project.output.height = args.size
    if args.fps:
        project.output.fps = args.fps
    if args.no_blur:
        project.motion_blur.enabled = False
    if args.shutter is not None:
        project.motion_blur.shutter_angle = args.shutter
    if args.start is not None:
        project.start = args.start
    if args.end is not None:
        project.end = args.end
    if args.no_audio:
        project.audio = False
    project.validate()

    info = probe(project.source)
    for warning in check_project(project, info):
        print(f"Attention : {warning}", file=sys.stderr)

    cancel = threading.Event()

    def on_progress(done: int, total: int, fps: float, eta: float) -> None:
        if args.quiet:
            return
        pct = 100.0 * done / total
        sys.stdout.write(f"\r  {done}/{total} images  {pct:5.1f} %  {fps:5.1f} img/s  reste {_fmt_time(eta)}   ")
        sys.stdout.flush()

    try:
        result = export_project(project, progress=on_progress, cancel=cancel, info=info)
    except KeyboardInterrupt:
        cancel.set()
        print("\nExport interrompu.", file=sys.stderr)
        return 130
    except ExportCancelled:
        print("\nExport annulé.", file=sys.stderr)
        return 130
    if not args.quiet:
        sys.stdout.write("\n")
    print(f"Terminé : {result.output}  ({result.frames} images en {_fmt_time(result.seconds)})")
    return 0


def cmd_animate(args: argparse.Namespace) -> int:
    from .ai import models as store
    from .ai.animate import AnimateOptions, animate

    if args.models_from:
        copied = store.import_from_folder(Path(args.models_from))
        print(f"{len(copied)} modèle(s) copié(s) dans {store.models_dir()}")

    def on_download(name: str, idx: int, n: int, done: int, total: int) -> None:
        if args.quiet:
            return
        pct = f"{100.0 * done / total:5.1f} %" if total else f"{done / 1e6:6.1f} Mo"
        sys.stdout.write(f"\r  modèle {idx}/{n}  {name}  {pct}   ")
        sys.stdout.flush()

    if args.download_models:
        try:
            d = store.ensure_models(progress=on_download)
            if args.body:
                store.ensure_pose_model(progress=on_download)
        except store.ModelError as exc:
            print(f"Erreur : {exc}", file=sys.stderr)
            return 1
        print(f"\nModèles prêts : {d}")
        return 0

    out = args.output or str(Path(args.driving).with_name(Path(args.reference).stem + "_anime.mp4"))
    try:
        if args.body:
            store.ensure_pose_model(progress=on_download)
        else:
            store.ensure_models(progress=on_download)
    except store.ModelError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    cancel = threading.Event()

    def on_progress(done: int, total: int, fps: float, eta: float) -> None:
        if args.quiet:
            return
        sys.stdout.write(f"\r  {done}/{total} images  {100.0 * done / total:5.1f} %  {fps:5.2f} img/s  reste {_fmt_time(eta)}   ")
        sys.stdout.flush()

    opts = AnimateOptions(
        reference=args.reference, driving=args.driving, output=out, max_output_side=args.max_side,
        max_fps=args.max_fps, start=args.start, duration=args.duration, multiplier=args.intensity,
        audio=not args.no_audio, mode="body" if args.body else "face",
    )
    try:
        result = animate(opts, progress=on_progress, cancel=cancel)
    except KeyboardInterrupt:
        cancel.set()
        print("\nInterrompu.", file=sys.stderr)
        return 130
    except ExportCancelled:
        print("\nAnnulé.", file=sys.stderr)
        return 130
    if not args.quiet:
        sys.stdout.write("\n")
    print(f"Terminé : {result.output}  ({result.frames} images en {_fmt_time(result.seconds)})")
    return 0


def _make_streams_tolerant() -> None:
    """Never crash on characters a legacy Windows console code page cannot encode."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (OSError, ValueError):
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _make_streams_tolerant()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        if args.command == "gui":
            try:
                import PySide6  # noqa: F401 - only checking that the GUI toolkit is present

                from .gui.app import run
            except ImportError:
                print(
                    "L'interface graphique n'est pas incluse dans cet exécutable : "
                    "lancez VideoMotionControl.exe (ou installez PySide6).",
                    file=sys.stderr,
                )
                return 1
            return run([sys.argv[0]])
        return {"render": cmd_render, "animate": cmd_animate, "init": cmd_init, "probe": cmd_probe, "presets": cmd_presets}[args.command](args)
    except (ProjectError, MediaError, ExportError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
