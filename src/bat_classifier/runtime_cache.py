"""Keep JIT cache writes inside the application workspace and fail promptly."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from numba.core.caching import UserWideCacheLocator


class WorkspaceCacheLocator(UserWideCacheLocator):
    """Support frozen source paths without probing a user-profile cache."""

    def __init__(self, py_func, py_file):
        self._py_file = py_file
        self._lineno = py_func.__code__.co_firstlineno
        root = Path(os.environ["BAT_WORKSPACE"]) / ".cache" / "numba"
        self._cache_path = str(root / self.get_suitable_cache_subpath(py_file))

    def ensure_cache_path(self):
        path = Path(self.get_cache_path())
        path.mkdir(parents=True, exist_ok=True)
        # tempfile on Windows retries PermissionError millions of times when
        # os.access reports writable. A single exclusive open fails promptly.
        probe = path / f".write-check-{uuid4().hex}"
        descriptor = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.close(descriptor)
        finally:
            probe.unlink()
