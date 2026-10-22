"""Errors shared by the Forge core. The HTTP layer maps them to ``{"error": {code, message, details?}}``."""

from __future__ import annotations

from typing import Any


class ForgeError(Exception):
    """An error with a stable machine-readable ``code`` and the HTTP ``status`` it maps to."""

    code = "error"
    status = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            body["details"] = self.details
        return {"error": body}


class BadRequest(ForgeError):
    code = "bad_request"
    status = 400


class NotFound(ForgeError):
    code = "not_found"
    status = 404


class Conflict(ForgeError):
    code = "conflict"
    status = 409


class ValidationFailed(ForgeError):
    """Invalid request. ``details`` is a list of ``{"path": "params.bars", "message": ...}``."""

    code = "validation_error"
    status = 422

    def __init__(self, errors: list[dict[str, str]], message: str = "invalid request") -> None:
        super().__init__(message, details=errors)
        self.errors = errors


class ConfigError(ValueError):
    """The Forge configuration file is invalid."""
