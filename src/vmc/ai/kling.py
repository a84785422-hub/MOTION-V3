"""Kling Motion Control 3.0 client.

The reference image and motion video are sent to Kling as base64 JSON fields. The
API key is supplied by the caller and is never written to a project or log file.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import ssl
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote, urlsplit

import cv2

from ..export import ExportCancelled, ExportError
from ..video_io import MediaError, load_image, probe

API_BASE = "https://api-singapore.klingai.com"
_CREATE_PATH = "/motion-control/kling-3.0"
_MAX_IMAGE_BYTES = 50 * 1024 * 1024
_MAX_VIDEO_BYTES = 100 * 1024 * 1024
_UPLOAD_CHUNK = 3 * 1024 * 1024  # divisible by 3, so base64 blocks can be concatenated
_POLL_SECONDS = 5
_MAX_WAIT_SECONDS = 60 * 60
StatusCallback = Callable[[str], None]


def _say(status: Optional[StatusCallback], text: str) -> None:
    if status is not None:
        status(text)


def _base64_length(size: int) -> int:
    return 4 * ((size + 2) // 3)


def _stream_base64(conn: http.client.HTTPSConnection, path: Path, cancel) -> None:
    with path.open("rb") as stream:
        while True:
            if cancel is not None and cancel.is_set():
                raise ExportCancelled()
            chunk = stream.read(_UPLOAD_CHUNK)
            if not chunk:
                return
            conn.send(base64.b64encode(chunk))


def _prepare_image(path: Path) -> tuple[Path, Optional[Path]]:
    """Keep supported files untouched and convert the other GUI image formats to PNG/JPEG."""
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return path, None
    try:
        image = load_image(path)
    except MediaError as exc:
        raise ExportError(str(exc)) from exc
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    extension = ".png" if image.dtype.name == "uint16" else ".jpg"
    params = [] if extension == ".png" else [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise ExportError("Impossible de convertir l'image de référence pour Kling.")
    fd, temporary = tempfile.mkstemp(prefix="vmc-kling-image-", suffix=extension)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded.tobytes())
    return Path(temporary), Path(temporary)


def _api_parts() -> tuple[str, int, str]:
    base = API_BASE
    parsed = urlsplit(base)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ExportError("L'adresse de l'API Kling doit utiliser HTTPS.")
    port = parsed.port or 443
    prefix = parsed.path.rstrip("/")
    return parsed.hostname, port, prefix


def _connection(host: str, port: int) -> http.client.HTTPSConnection:
    return http.client.HTTPSConnection(host, port, timeout=120, context=ssl.create_default_context())


def _decode_response(response: http.client.HTTPResponse) -> dict:
    raw = response.read(4 * 1024 * 1024)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        detail = raw[:800].decode("utf-8", "replace")
        raise ExportError(f"Réponse Kling illisible (HTTP {response.status}) : {detail}") from exc
    if response.status < 200 or response.status >= 300:
        message = payload.get("message") or payload.get("error") or str(payload)
        raise ExportError(f"Kling a refusé la demande (HTTP {response.status}) : {message}")
    if not isinstance(payload, dict):
        raise ExportError("Réponse Kling inattendue.")
    code = payload.get("code")
    if code not in (None, 0, "0"):
        raise ExportError(f"Erreur Kling {code} : {payload.get('message', 'réponse inconnue')}")
    return payload


def _submit(
    api_key: str,
    prompt: str,
    image_path: Path,
    image_size: int,
    video_path: Path,
    video_size: int,
    orientation: str,
    resolution: str,
    audio: bool,
    cancel,
) -> str:
    host, port, prefix = _api_parts()
    prompt_json = json.dumps(prompt, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    settings = json.dumps(
        {
            "character_orientation": orientation,
            "resolution": resolution,
            "audio": "original" if audio else "off",
        },
        separators=(",", ":"),
    ).encode("utf-8")
    start = b'{"contents":[{"type":"prompt","text":' + prompt_json + b'},{"type":"image","url":"'
    middle = b'"},{"type":"video","url":"'
    end = b'"}],"settings":' + settings + b"}"
    content_length = (
        len(start) + _base64_length(image_size) + len(middle)
        + _base64_length(video_size) + len(end)
    )
    conn = _connection(host, port)
    try:
        conn.putrequest("POST", prefix + _CREATE_PATH)
        conn.putheader("Authorization", "Bearer " + api_key)
        conn.putheader("Content-Type", "application/json; charset=utf-8")
        conn.putheader("Content-Length", str(content_length))
        conn.putheader("Accept", "application/json")
        conn.endheaders()
        conn.send(start)
        _stream_base64(conn, image_path, cancel)
        conn.send(middle)
        _stream_base64(conn, video_path, cancel)
        conn.send(end)
        response = conn.getresponse()
        payload = _decode_response(response)
    except ExportCancelled:
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise ExportError(f"Connexion à Kling interrompue pendant l'envoi : {exc}") from exc
    finally:
        conn.close()

    data = payload.get("data") or {}
    task_id = data.get("id") if isinstance(data, dict) else None
    if not task_id:
        raise ExportError(f"Kling n'a pas retourné de numéro de tâche : {payload.get('message', payload)}")
    return str(task_id)


def _query_task(api_key: str, task_id: str) -> Optional[dict]:
    host, port, prefix = _api_parts()
    conn = _connection(host, port)
    try:
        path = prefix + "/tasks?task_ids=" + quote(task_id, safe="")
        conn.request("GET", path, headers={"Authorization": "Bearer " + api_key, "Accept": "application/json"})
        payload = _decode_response(conn.getresponse())
    except ExportError:
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise ExportError(f"Impossible de consulter la tâche Kling {task_id} : {exc}") from exc
    finally:
        conn.close()

    data = payload.get("data")
    if isinstance(data, list):
        return data[0] if data else None
    if isinstance(data, dict):
        result = data.get("result") or data.get("tasks") or []
        if isinstance(result, list):
            return result[0] if result else None
    return None


def _result_url(task: dict) -> Optional[str]:
    outputs = task.get("outputs") or []
    if isinstance(outputs, list):
        for output in outputs:
            if isinstance(output, dict) and output.get("type") == "video" and output.get("url"):
                return str(output["url"])
    return None


def _download(url: str, output: Path, cancel, status: Optional[StatusCallback]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + output.stem + "-", suffix=".part", dir=str(output.parent))
    os.close(fd)
    temp_path = Path(temporary)
    try:
        _say(status, "Téléchargement du résultat Kling…")
        with urllib.request.urlopen(url, timeout=120) as response, temp_path.open("wb") as target:
            while True:
                if cancel is not None and cancel.is_set():
                    raise ExportCancelled()
                block = response.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
        if temp_path.stat().st_size == 0:
            raise ExportError("Kling a retourné un fichier vidéo vide.")
        os.replace(temp_path, output)
    except ExportCancelled:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise ExportError(f"Impossible de télécharger le résultat Kling : {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def animate_kling(opts, api_key: str, progress=None, cancel=None, status=None):
    """Generate a motion-controlled video with the official Kling 3.0 API."""
    from .animate import AnimateResult

    key = (api_key or "").strip()
    if not key:
        raise ExportError("Clé API Kling manquante. Saisissez-la dans la fenêtre ou définissez KLING_API_KEY.")

    prompt = (getattr(opts, "kling_prompt", "") or "").strip()
    if not prompt:
        prompt = "Reproduire le mouvement et les expressions de la vidéo pilote en préservant l'apparence de l'image de référence."
    if len(prompt) > 2500:
        raise ExportError("La description Kling ne peut pas dépasser 2 500 caractères.")

    image_path = Path(opts.reference)
    video_path = Path(opts.driving)
    output = Path(opts.output)
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        raise ExportError("L'image de référence doit être au format JPG, PNG, WebP, BMP ou TIFF.")
    if video_path.suffix.lower() not in {".mp4", ".mov"}:
        raise ExportError("La vidéo pilote Kling doit être au format MP4 ou MOV.")
    if output.suffix.lower() != ".mp4":
        raise ExportError("Le fichier de sortie Kling doit être un MP4.")
    if not output.parent.is_dir():
        raise ExportError("Le dossier du fichier de sortie n'existe pas.")

    try:
        image_info = probe(image_path)
        video_info = probe(video_path)
    except MediaError as exc:
        raise ExportError(str(exc)) from exc
    if not image_info.is_image:
        raise ExportError("Le fichier de référence n'est pas une image.")
    if image_info.width < 300 or image_info.height < 300:
        raise ExportError("L'image de référence doit mesurer au moins 300 × 300 pixels.")
    ratio = image_info.width / image_info.height
    if not 0.4 <= ratio <= 2.5:
        raise ExportError("Le rapport largeur/hauteur de l'image doit être compris entre 0,4 et 2,5.")

    orientation = getattr(opts, "kling_orientation", "video")
    if orientation not in {"image", "video"}:
        raise ExportError("Orientation Kling invalide.")
    max_duration = 10.0 if orientation == "image" else 30.0
    if video_info.is_image or video_info.duration is None:
        raise ExportError("La vidéo pilote ne peut pas être lue.")
    if video_info.duration < 3.0:
        raise ExportError("Kling Motion Control exige au moins 3 secondes de vidéo pilote.")
    if video_info.duration > max_duration:
        raise ExportError(
            f"Avec cette orientation, la vidéo pilote ne peut pas dépasser {int(max_duration)} secondes."
        )
    if min(video_info.width, video_info.height) < 340 or max(video_info.width, video_info.height) > 3850:
        raise ExportError("La vidéo pilote doit mesurer entre 340 et 3 850 pixels de côté.")
    image_send_path, temporary_image = _prepare_image(image_path)
    image_size = image_send_path.stat().st_size
    video_size = video_path.stat().st_size
    if image_size > _MAX_IMAGE_BYTES:
        if temporary_image is not None:
            temporary_image.unlink(missing_ok=True)
        raise ExportError("L'image de référence dépasse la limite Kling de 50 Mo.")
    if video_size > _MAX_VIDEO_BYTES:
        if temporary_image is not None:
            temporary_image.unlink(missing_ok=True)
        raise ExportError("La vidéo pilote dépasse la limite Kling de 100 Mo.")

    started = time.monotonic()
    if cancel is not None and cancel.is_set():
        if temporary_image is not None:
            temporary_image.unlink(missing_ok=True)
        raise ExportCancelled()
    _say(status, "Envoi sécurisé de l'image et de la vidéo à Kling Motion Control 3.0…")
    try:
        task_id = _submit(
            key, prompt, image_send_path, image_size, video_path, video_size,
            orientation, getattr(opts, "kling_resolution", "1080p"),
            bool(opts.audio), cancel,
        )
    finally:
        if temporary_image is not None:
            temporary_image.unlink(missing_ok=True)
    _say(status, f"Tâche Kling {task_id} envoyée. Génération en cours…")

    deadline = time.monotonic() + _MAX_WAIT_SECONDS
    while True:
        if cancel is not None and cancel.is_set():
            raise ExportCancelled()
        if time.monotonic() > deadline:
            raise ExportError(
                f"Le délai d'attente est dépassé pour la tâche Kling {task_id}. "
                "La tâche peut continuer sur votre compte."
            )
        task = _query_task(key, task_id)
        if task is not None:
            task_status = str(task.get("status", "")).lower()
            if task_status == "failed":
                message = task.get("message") or "la génération a échoué."
                raise ExportError(f"Génération Kling échouée (tâche {task_id}) : {message}")
            if task_status in {"succeeded", "success", "completed"}:
                url = _result_url(task)
                if not url:
                    raise ExportError(f"La tâche Kling {task_id} est terminée, mais aucun lien vidéo n'a été retourné.")
                _download(url, output, cancel, status)
                _say(status, "Vidéo Kling téléchargée.")
                return AnimateResult(str(output), 0, time.monotonic() - started)
        elapsed = int(time.monotonic() - started)
        _say(status, f"Génération Kling 3.0 en cours (tâche {task_id}, {elapsed // 60} min {elapsed % 60:02d} s)…")
        if cancel is not None:
            if cancel.wait(_POLL_SECONDS):
                raise ExportCancelled()
        else:
            time.sleep(_POLL_SECONDS)
