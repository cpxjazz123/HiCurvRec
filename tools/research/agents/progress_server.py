"""The append-only progress channel of one sub-agent.

The agent's only write is this channel. The tool takes the record's content and
nothing else: no path, no task identifier, no sequence number. The file it
writes to is bound when the server starts, so the agent cannot write anywhere
else, cannot write into another agent's file, and cannot rewrite a record that
is already there - the file is only ever opened for appending.

The record is one JSON object per line. A line is written, flushed and synced
before the tool answers, so a caller that received the confirmation knows the
record is on disk; a reader that finds an incomplete tail line ignores it.

The deadline is the server's own: after it, the tool refuses and the file stays
as it was.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

#: How long one record may be. A stage record is a summary, not a transcript.
MAX_RECORD_CHARS = 4000

#: The tool the agent calls. The name is fixed; the server offers nothing else.
TOOL_NAME = "append_progress"

_LOCK = threading.Lock()


class ProgressFile:
    """One agent's own file, opened for appending and never for rewriting."""

    def __init__(self, path: Path, task_id: str, deadline: float | None = None,
                 wrapup_margin: float = 60.0):
        self.path = Path(path)
        self.task_id = task_id
        self.deadline = deadline
        self.wrapup_margin = wrapup_margin
        self._sequence = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The file is created by the framework, which writes the task's start
        # record; the server only appends to it.
        self.path.touch(exist_ok=True)
        self._sequence = self._count_records()

    def _count_records(self) -> int:
        """How many whole records the file holds.

        A line that is not a whole JSON object was never confirmed, so it does
        not count: the sequence number must not skip a number because a
        previous run was killed mid-write.
        """
        count = 0
        for record in read_records(self.path):
            count += 1
        return count

    def _isolate_fragment(self) -> None:
        """Separate a truncated tail from the next record.

        A run that was killed while writing leaves a half line. Appending
        straight after it would glue the fragment and the new record into one
        line, and the reader would ignore both. The fragment gets a line of its
        own, where it is ignored, and the new record stands alone.
        """
        try:
            if not self.path.exists() or self.path.stat().st_size == 0:
                return
            with self.path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                last = handle.read(1)
        except OSError:
            return
        if last == b"\n":
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def remaining_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())

    def append(self, content: str) -> dict:
        """Append one record, or refuse and say why."""
        text = str(content or "").strip()
        if not text:
            raise ValueError("the record is empty")
        if len(text) > MAX_RECORD_CHARS:
            raise ValueError("the record is longer than " + str(MAX_RECORD_CHARS) +
                             " characters; summarise the stage instead")
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("the task's deadline has passed; the record was not saved")
        with _LOCK:
            self._isolate_fragment()
            size_before = self.path.stat().st_size if self.path.exists() else 0
            self._sequence += 1
            record = {
                "seq": self._sequence,
                "task_id": self.task_id,
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kind": "stage",
                "content": text,
            }
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            size_after = self.path.stat().st_size
            if size_after <= size_before:
                # The file only grows. If it did not, the record was not appended
                # and the sequence number must not be spent.
                self._sequence -= 1
                raise RuntimeError("the progress file did not grow; nothing was appended")
        return {"seq": record["seq"], "recorded_at": record["recorded_at"],
                "remaining_seconds": remaining}


def read_records(path: Path) -> list:
    """The complete records of a progress file.

    A tail line that is not a whole JSON object was never confirmed, so it is
    ignored: a reader never has to guess whether the last record landed.
    """
    records = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _tool_definition() -> dict:
    return {
        "name": TOOL_NAME,
        "description": (
            "Append one stage record to this task's own progress file. The file is "
            "yours alone and is only ever appended to. Send the content of the "
            "record only: no path, no task id. Record what you have checked so "
            "far (the goal of the check, what you read, what you observed, where "
            "the evidence is, your short reasoning, what is still open). Append "
            "again after every meaningful step, and append a correction when you "
            "overturn an earlier conclusion."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string",
                            "description": "The stage record. No file path, no task id."},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    }


def _reply(message_id, result) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result}) + "\n")
    sys.stdout.flush()


def _error(message_id, message: str) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message_id,
                                  "error": {"code": -32000, "message": message}}) + "\n")
    sys.stdout.flush()


def serve(progress: ProgressFile) -> int:
    """The stdio loop: one JSON-RPC message per line."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        method = message.get("method")
        message_id = message.get("id")
        if method == "initialize":
            version = message.get("params", {}).get("protocolVersion")
            _reply(message_id, {
                "protocolVersion": version if isinstance(version, str) else "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "progress", "version": "1"},
            })
            continue
        if method in ("notifications/initialized", "notifications/cancelled"):
            continue
        if method == "tools/list":
            _reply(message_id, {"tools": [_tool_definition()]})
            continue
        if method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name != TOOL_NAME:
                _reply(message_id, {"content": [{"type": "text",
                                                 "text": "unknown tool: " + str(name)}],
                                    "isError": True})
                continue
            try:
                saved = progress.append(arguments.get("content"))
            except (ValueError, TimeoutError, RuntimeError) as exc:
                _reply(message_id, {"content": [{"type": "text", "text": str(exc)}],
                                    "isError": True})
                continue
            text = ("saved as record " + str(saved["seq"]) + " of this task's progress file")
            remaining = saved.get("remaining_seconds")
            if remaining is not None:
                text += ("; " + str(int(remaining)) +
                         " seconds left before the deadline")
                if remaining <= progress.wrapup_margin:
                    text += (" - the deadline is near: record your final stage now, "
                             "a task that ends without a final record leaves its "
                             "last questions open")
            _reply(message_id, {"content": [{"type": "text", "text": text}],
                                "isError": False})
            continue
        if message_id is not None:
            _error(message_id, "unknown method: " + str(method))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True,
                        help="the progress file of this task; the only file written")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--deadline-epoch", type=float, default=0.0,
                        help="the unix time after which records are refused")
    parser.add_argument("--wrapup-margin", type=float, default=60.0,
                        help="seconds before the deadline in which the tool asks "
                             "the agent to record its final stage")
    args = parser.parse_args(argv)
    deadline = args.deadline_epoch or None
    return serve(ProgressFile(Path(args.file), args.task_id, deadline,
                              args.wrapup_margin))


if __name__ == "__main__":
    sys.exit(main())
