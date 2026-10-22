from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from forge.service import ForgeService

from fcore_helpers import make_config, make_service


@pytest.fixture
async def service(tmp_path: Path) -> AsyncIterator[ForgeService]:
    """A service with the real testtone and yin worker subprocesses."""
    svc = make_service(make_config(tmp_path))
    await svc.start()
    try:
        yield svc
    finally:
        await svc.stop()
