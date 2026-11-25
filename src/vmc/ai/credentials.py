"""Secure storage for third-party API credentials."""

from __future__ import annotations

import os

_SERVICE = "VideoMotionControl"
_USERNAME = "kling-api-key"


def get_kling_api_key() -> str:
    """Read a key from the environment first, then the operating-system key store."""
    value = os.environ.get("KLING_API_KEY", "").strip()
    if value:
        return value
    try:
        import keyring
        return (keyring.get_password(_SERVICE, _USERNAME) or "").strip()
    except Exception:  # an unavailable keyring must not prevent local animation
        return ""


def save_kling_api_key(value: str) -> None:
    """Store a key in the operating-system credential vault, never in a project file."""
    try:
        import keyring
        keyring.set_password(_SERVICE, _USERNAME, value.strip())
    except Exception as exc:
        raise RuntimeError(f"Impossible d'enregistrer la clé dans le coffre sécurisé : {exc}") from exc


def forget_kling_api_key() -> None:
    """Remove a saved key. An environment variable, if set, is left untouched."""
    try:
        import keyring
        keyring.delete_password(_SERVICE, _USERNAME)
    except Exception as exc:
        # Forgetting an already absent key is a normal state.
        if type(exc).__name__ not in {"PasswordDeleteError", "CredentialNotFoundError"}:
            raise RuntimeError(f"Impossible d'effacer la clé mémorisée : {exc}") from exc
