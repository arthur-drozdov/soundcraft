"""Client for SoundCraft: the desktop control channel and headless ``soundcraft-cli mcp``."""

from .control import ControlChannel
from .errors import CommandError, StudioError
from .headless import HeadlessMcp
from .studio import Studio

__all__ = ["CommandError", "ControlChannel", "HeadlessMcp", "Studio", "StudioError"]
