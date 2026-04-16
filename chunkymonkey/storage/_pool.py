# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 74e45bce-4997-41c7-b404-6c11f6ebfd91
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Thread-safe DuckDB connection pool for chunkymonkey."""

from __future__ import annotations

import atexit
import logging
import os
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Generator

try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False
    duckdb = None  # type: ignore

logger = logging.getLogger(__name__)

# Track all live pools for atexit cleanup
_all_pools: weakref.WeakSet = weakref.WeakSet()


def _close_all_pools() -> None:
    for pool in list(_all_pools):
        try:
            pool.close()
        except Exception:
            pass


atexit.register(_close_all_pools)


def _try_kill_orphan_lock_holder(error_msg: str) -> None:
    """Parse PID from DuckDB lock error and kill if it's an orphaned child process."""
    import re
    import signal

    match = re.search(r"\(PID (\d+)\)", error_msg)
    if not match:
        return
    pid = int(match.group(1))
    if pid == os.getpid():
        return
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return

    try:
        with open(f"/proc/{pid}/cmdline", "r") as f:
            cmdline = f.read()
    except FileNotFoundError:
        import subprocess
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        )
        cmdline = result.stdout.strip()

    if "multiprocessing" in cmdline or "resource_tracker" in cmdline:
        logger.warning(f"Killing orphaned multiprocessing child PID {pid}")
        try:
            os.kill(pid, signal.SIGKILL)
            import time
            time.sleep(0.5)
        except OSError:
            pass


class _PendingResult:
    """Holds RLock from execute() until a terminal fetch completes."""

    __slots__ = ("_conn", "_lock", "_released")

    def __init__(self, conn, lock: threading.RLock):
        self._conn = conn
        self._lock = lock
        self._released = False

    def _release(self):
        if not self._released:
            self._released = True
            self._lock.release()

    def fetchall(self):
        try:
            return self._conn.fetchall()
        finally:
            self._release()

    def fetchone(self):
        try:
            return self._conn.fetchone()
        finally:
            self._release()

    def fetchdf(self):
        try:
            return self._conn.fetchdf()
        finally:
            self._release()

    def df(self):
        try:
            return self._conn.df()
        finally:
            self._release()

    @property
    def description(self):
        return self._conn.description

    @property
    def rowcount(self):
        return self._conn.rowcount

    def __del__(self):
        self._release()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _LockedConnection:
    """Proxy that serializes execute+fetch cycles through a lock."""

    __slots__ = ("_conn", "_lock")

    def __init__(self, conn, lock: threading.RLock):
        self._conn = conn
        self._lock = lock

    def execute(self, *args, **kwargs):
        self._lock.acquire()
        try:
            self._conn.execute(*args, **kwargs)
            return _PendingResult(self._conn, self._lock)
        except Exception:
            self._lock.release()
            raise

    def executemany(self, *args, **kwargs):
        with self._lock:
            self._conn.executemany(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


class ThreadLocalDuckDB:
    """Single shared DuckDB connection with optional init SQL.

    Despite the name (kept for API compatibility), uses a single shared
    connection protected by a reentrant lock — no per-thread cursors.

    Usage:
        db = ThreadLocalDuckDB("/path/to/db.duckdb")
        db.conn.execute("SELECT 1").fetchone()
    """

    def __init__(
        self,
        db_path: str | Path,
        read_only: bool = False,
        config: Optional[dict] = None,
        init_sql: Optional[list[str]] = None,
    ):
        if not _DUCKDB_AVAILABLE:
            raise ImportError(
                "duckdb is required for storage. "
                "Install it with: pip install chunkymonkey[storage]"
            )
        self._db_path = str(db_path)
        self._read_only = read_only
        self._config = config or {}
        self._closed = False
        self._lock = threading.RLock()
        self._init_sql = init_sql or []
        self._init_lock = threading.Lock()

        import time
        last_err = None
        for attempt in range(5):
            try:
                self._shared_conn = duckdb.connect(
                    self._db_path,
                    read_only=self._read_only,
                    config=self._config,
                )
                last_err = None
                break
            except duckdb.IOException as e:
                if "Could not set lock" in str(e) and attempt < 4:
                    last_err = e
                    _try_kill_orphan_lock_holder(str(e))
                    logger.warning(f"DuckDB lock conflict, retrying in {attempt + 1}s...")
                    time.sleep(attempt + 1)
                else:
                    raise
        if last_err is not None:
            raise last_err

        _all_pools.add(self)

        locked = _LockedConnection(self._shared_conn, self._lock)
        for sql in self._init_sql:
            try:
                locked.execute(sql).fetchall()
            except Exception as e:
                logger.debug(f"Init SQL failed (may be expected): {e}")

    @property
    def conn(self) -> "_LockedConnection":
        """Get the shared connection wrapped in a locking proxy."""
        if self._closed:
            raise RuntimeError("ThreadLocalDuckDB has been closed")
        return _LockedConnection(self._shared_conn, self._lock)

    def close(self) -> None:
        """Close the shared connection."""
        if self._closed:
            return
        self._closed = True
        try:
            self._shared_conn.close()
        except Exception as e:
            logger.warning(f"Error closing DuckDB connection: {e}")

    def close_all(self) -> None:
        """Alias for close() — closes the shared connection."""
        self.close()

    def __enter__(self) -> "ThreadLocalDuckDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()
