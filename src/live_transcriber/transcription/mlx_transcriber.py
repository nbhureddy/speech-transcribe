"""
MLX-native Whisper transcriber for Apple Silicon (M1/M2/M3/M4).

Uses the ``mlx-whisper`` library which runs the Whisper model via Apple's
MLX framework, leveraging the GPU / Apple Neural Engine for fast, accurate
inference without needing CUDA or ctranslate2.
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np

from live_transcriber.config import AppConfig
from live_transcriber.io.wav_writer import save_wav

logger = logging.getLogger(__name__)


class MlxTranscriber:
    """
    Whisper transcriber backed by Apple's MLX framework.

    Requires ``mlx-whisper`` (``pip install mlx-whisper``).  Only works on
    macOS with Apple Silicon.  Falls back gracefully with a clear error if
    the library is not installed.

    Parameters
    ----------
    config:
        Application configuration — uses ``model.size``, ``model.language``,
        and ``transcription.*`` fields.
    """

    def __init__(self, config: AppConfig) -> None:
        try:
            import mlx_whisper  # noqa: F401 — imported to validate install
        except ImportError as exc:
            raise ImportError(
                "mlx-whisper is required for backend='mlx-whisper'. "
                "Install it with: pip install mlx-whisper"
            ) from exc

        self._cfg = config
        # mlx_whisper uses HuggingFace-style repo IDs for large models
        self._model_id = self._resolve_model_id(config.model.size)
        logger.info(
            "MLX Whisper backend — model '%s' (repo: %s)…",
            config.model.size,
            self._model_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_id(size: str) -> str:
        """Map config model size names to mlx-whisper HuggingFace repo IDs."""
        _mapping = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "large": "mlx-community/whisper-large-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "distil-large-v3": "mlx-community/distil-whisper-large-v3",
        }
        resolved = _mapping.get(size)
        if resolved is None:
            # Allow passing a raw HuggingFace repo ID directly
            if "/" in size:
                return size
            raise ValueError(
                f"Unknown model size '{size}' for mlx-whisper backend. "
                f"Known sizes: {list(_mapping.keys())} — or pass a full HF repo ID."
            )
        return resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribe a float32 mono audio array using Apple MLX.

        Parameters
        ----------
        audio_chunk:
            1-D float32 numpy array sampled at ``config.audio.sample_rate``.

        Returns
        -------
        str
            Transcribed text, or an empty string if nothing was detected.
        """
        import mlx_whisper

        tc = self._cfg.transcription

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            save_wav(wav_path, audio_chunk, self._cfg.audio.sample_rate)

            # mlx-whisper does not support beam search yet; force greedy decoding.
            if tc.beam_size and tc.beam_size > 1:
                logger.warning(
                    "mlx-whisper does not support beam_size > 1 (beam search is not "
                    "implemented). Falling back to greedy decoding (beam_size=1)."
                )
            result = mlx_whisper.transcribe(
                wav_path,
                path_or_hf_repo=self._model_id,
                language=self._cfg.model.language or None,
                # beam_size intentionally omitted — mlx_whisper raises
                # NotImplementedError for beam_size > 1.
                best_of=tc.best_of,
                temperature=tc.temperature,
                # VAD is not built into mlx-whisper; silence is handled by
                # the audio pipeline's chunk/overlap logic
                verbose=False,
            )

            # mlx_whisper.transcribe returns a dict with a "text" key
            text = result.get("text", "").strip()

            if text and self._cfg.debug:
                logger.debug("Transcribed (MLX): %s", text)

            return text

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def transcribe_file(self, audio_file_path: str) -> str:
        """
        Transcribe an audio file path using mlx-whisper.

        Parameters
        ----------
        audio_file_path:
            Path to an audio file supported by mlx-whisper, such as WAV or MP3.

        Returns
        -------
        str
            Full transcribed text, or an empty string if nothing was detected.
        """
        import mlx_whisper

        tc = self._cfg.transcription

        if tc.beam_size and tc.beam_size > 1:
            logger.warning(
                "mlx-whisper does not support beam_size > 1 (beam search is not "
                "implemented). Falling back to greedy decoding (beam_size=1)."
            )

        result = mlx_whisper.transcribe(
            audio_file_path,
            path_or_hf_repo=self._model_id,
            language=self._cfg.model.language or None,
            best_of=tc.best_of,
            temperature=tc.temperature,
            verbose=False,
        )

        text = result.get("text", "").strip()

        if text and self._cfg.debug:
            logger.debug("Transcribed file (MLX): %s", audio_file_path)

        return text
