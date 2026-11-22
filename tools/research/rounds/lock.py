"""One runner per round.

A round is driven by agents that take minutes, so two callers must not run the
same phase at the same time: the second one would pay for the same agents and
could write the same answer twice. The lock is a file the runner creates
atomically and removes when it is done; a lock whose owner is gone is stale and
is taken over.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class RoundBusy(RuntimeError):
    """Another runner holds this round."""


class RoundLock:
    """The lock of one round directory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.path = self.directory / "runner.lock"

    def _owner(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _alive(self, pid) -> bool:
        try:
            os.kill(int(pid), 0)
        except (OSError, ProcessLookupError, ValueError, TypeError):
            return False
        # A zombie still answers signal 0, but it is not working: it was killed
        # and its parent has not reaped it. A lock held by a zombie is stale,
        # and treating it as held would stop the round for good.
        try:
            stat = Path("/proc/" + str(int(pid)) + "/stat").read_text()
            state = stat.rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            return True
        return state != "Z"

    def acquire(self, *, owner: str = "") -> dict:
        """Take the lock, or refuse because another runner holds it."""
        self.directory.mkdir(parents=True, exist_ok=True)
        current = self._owner()
        if current and self._alive(current.get("pid")):
            raise RoundBusy("the round is being worked on by pid " +
                            str(current.get("pid")) + " (" + str(current.get("owner", "")) + ")")
        if current:
            # The owner is gone: its lock file stays behind, and it has to be
            # removed or the atomic create below would refuse for good. The
            # create is still atomic, so a runner that took the lock in the
            # meantime wins and this one is refused.
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        payload = {"pid": os.getpid(), "owner": owner, "at": time.time()}
        # Atomic: the file is created with O_EXCL, so only one caller wins.
        try:
            handle = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RoundBusy("the round is being worked on by pid " +
                            str(self._owner().get("pid")))
        with os.fdopen(handle, "w", encoding="utf-8") as sink:
            sink.write(json.dumps(payload))
        return payload

    def release(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass

    def holder(self) -> dict:
        """Who holds the lock, if anybody."""
        current = self._owner()
        if current and self._alive(current.get("pid")):
            return current
        return {}
