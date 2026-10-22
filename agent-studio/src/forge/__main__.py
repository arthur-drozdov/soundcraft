"""``python -m forge`` is the ``forge`` command."""

from __future__ import annotations

import sys

from .cli import main

sys.exit(main())
