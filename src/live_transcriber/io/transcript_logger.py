"""
Transcript logger: writes each run to a new timestamped file inside a folder.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class TranscriptLogger:
    """
    Writes transcribed text to a timestamped file inside *transcript_dir*.

    A new file is created for every run, named ``YYYY-MM-DD_HH-MM-SS.txt``,
    so transcripts are never overwritten.

    Parameters
    ----------
    transcript_dir:
        Directory where transcript files are stored. Created if it doesn't exist.
    context:
        Optional free-text note about the session (speaker, topic, etc.).
        Written as a header block at the top of the file before any transcription.
    """

    def __init__(self, transcript_dir: str, context: Optional[str] = None) -> None:
        os.makedirs(transcript_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
        self._file_path = os.path.join(transcript_dir, filename)
        logger.info("Transcript file: %s", self._file_path)

        if context:
            self._write_header(context)

    @property
    def file_path(self) -> str:
        """Absolute path to the current transcript file."""
        return self._file_path

    def _write_header(self, context: str) -> None:
        """Write a context header block at the top of the transcript file."""
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            "=" * 60 + "\n"
            f"  Session started : {started_at}\n"
            f"  Context         :\n"
        )
        # Indent each line of the context block
        for line in context.strip().splitlines():
            header += f"    {line}\n"
        header += "=" * 60 + "\n\n"

        with open(self._file_path, "w", encoding="utf-8") as fh:
            fh.write(header)

        logger.debug("Wrote context header to transcript file.")

    def log(self, text: str) -> None:
        """Append *text* to the transcript file with a timestamp prefix."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {text}\n"

        with open(self._file_path, "a", encoding="utf-8") as fh:
            fh.write(line)

        logger.debug("Logged: %s", line.rstrip())
