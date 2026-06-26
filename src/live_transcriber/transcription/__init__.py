"""
Transcription package.

Use :func:`get_transcriber` to obtain the appropriate backend at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from live_transcriber.config import AppConfig


class TranscriberProtocol(Protocol):
    """Structural type shared by all transcriber backends."""

    def transcribe(self, audio_chunk: np.ndarray) -> str: ...


def get_transcriber(config: "AppConfig") -> TranscriberProtocol:
    """
    Return the transcriber backend selected by ``config.model.backend``.

    Supported backends
    ------------------
    ``mlx-whisper``
        Apple Silicon native (MLX / Apple Neural Engine).
        Requires ``pip install mlx-whisper``.  Best accuracy and speed on
        M1/M2/M3/M4 Macs.
    ``faster-whisper``
        ctranslate2-based, cross-platform fallback.
        Works on Linux, Windows, and non-Apple Macs.
    """
    backend = (config.model.backend or "faster-whisper").lower()

    if backend == "mlx-whisper":
        from live_transcriber.transcription.mlx_transcriber import MlxTranscriber
        return MlxTranscriber(config)

    if backend == "faster-whisper":
        from live_transcriber.transcription.transcriber import LiveTranscriber
        return LiveTranscriber(config)

    raise ValueError(
        f"Unknown transcription backend '{backend}'. "
        "Choose 'mlx-whisper' (Apple Silicon) or 'faster-whisper' (cross-platform)."
    )
