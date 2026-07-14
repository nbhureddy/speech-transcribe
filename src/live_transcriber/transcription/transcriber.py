"""
Whisper-based live transcriber.
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
from faster_whisper import WhisperModel

from live_transcriber.config import AppConfig
from live_transcriber.io.wav_writer import save_wav

logger = logging.getLogger(__name__)


class LiveTranscriber:
    """
    Loads a Whisper model and transcribes audio chunks on demand.

    Parameters
    ----------
    config:
        Application configuration (model section is used for loading;
        transcription section for inference parameters).
    """

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        logger.info(
            "Loading Whisper model '%s' (compute_type=%s)…",
            config.model.size,
            config.model.compute_type,
        )
        self._model = WhisperModel(
            config.model.size,
            compute_type=config.model.compute_type,
        )
        logger.info("Whisper model loaded.")

    def transcribe(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribe a float32 mono audio array.

        Parameters
        ----------
        audio_chunk:
            1-D float32 numpy array sampled at ``config.audio.sample_rate``.

        Returns
        -------
        str
            Transcribed text, or an empty string if nothing was detected.
        """
        tc = self._cfg.transcription

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            save_wav(wav_path, audio_chunk, self._cfg.audio.sample_rate)

            segments, _info = self._model.transcribe(
                wav_path,
                language=self._cfg.model.language or None,
                vad_filter=tc.vad_filter,
                vad_parameters={"min_silence_duration_ms": tc.min_silence_duration_ms},
                beam_size=tc.beam_size,
                best_of=tc.best_of,
                temperature=tc.temperature,
            )

            texts = [seg.text.strip() for seg in segments if seg.text.strip()]
            result = " ".join(texts).strip()

            if result and self._cfg.debug:
                logger.debug("Transcribed: %s", result)

            return result

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def transcribe_file(self, audio_file_path: str) -> str:
        """
        Transcribe an audio file using faster-whisper's native file support.

        Parameters
        ----------
        audio_file_path:
            Path to an audio file supported by the backend, such as WAV or MP3.

        Returns
        -------
        str
            Full transcribed text, or an empty string if nothing was detected.
        """
        tc = self._cfg.transcription

        segments, _info = self._model.transcribe(
            audio_file_path,
            language=self._cfg.model.language or None,
            vad_filter=tc.vad_filter,
            vad_parameters={"min_silence_duration_ms": tc.min_silence_duration_ms},
            beam_size=tc.beam_size,
            best_of=tc.best_of,
            temperature=tc.temperature,
        )

        texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        result = " ".join(texts).strip()

        if result and self._cfg.debug:
            logger.debug("Transcribed file: %s", audio_file_path)

        return result
