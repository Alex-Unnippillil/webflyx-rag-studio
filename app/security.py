import os

import keyring
from keyring.errors import KeyringError

SERVICE = "WebflyxRAGStudio.OpenRouter"
ACCOUNT = "default"


class SecretStoreError(RuntimeError):
    pass


def validate_api_key(value: str) -> str:
    value = value.strip()

    if len(value) < 20:
        raise ValueError("The API key appears to be too short.")

    if any(char.isspace() for char in value):
        raise ValueError("API keys cannot contain whitespace.")

    return value


def save_openrouter_key(value: str) -> None:
    key = validate_api_key(value)

    try:
        keyring.set_password(SERVICE, ACCOUNT, key)
    except KeyringError as exc:
        raise SecretStoreError(
            "Secure operating-system credential storage is unavailable."
        ) from exc


def get_openrouter_key() -> str | None:
    development_key = os.getenv("OPENROUTER_API_KEY")

    if development_key:
        return development_key.strip()

    try:
        return keyring.get_password(SERVICE, ACCOUNT)
    except KeyringError as exc:
        raise SecretStoreError(
            "Could not access secure operating-system credential storage."
        ) from exc


def delete_openrouter_key() -> None:
    try:
        keyring.delete_password(SERVICE, ACCOUNT)
    except keyring.errors.PasswordDeleteError:
        pass
    except KeyringError as exc:
        raise SecretStoreError("Could not update credential storage.") from exc


def has_openrouter_key() -> bool:
    return bool(get_openrouter_key())
