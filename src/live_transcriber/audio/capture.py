"""
Audio capture: wraps sounddevice.InputStream and an internal queue.
"""

from __future__ import annotations

import logging
import queue
import sys

import numpy as np
import sounddevice as sd

from live_transcriber.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Streams audio from a device into a thread-safe queue.

    Usage::

        capture = AudioCapture(config.audio, device_index=2)
        capture.start()
        try:
            while True:
                chunk = capture.get_block()   # blocks until data arrives
                process(chunk)
        finally:
            capture.stop()
    """

    def __init__(self, config: AudioConfig, device_index: int) -> None:
        self._config = config
        self._device_index = device_index
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    # -------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------- #

    def start(self) -> None:
        """Open and start the InputStream."""
        self._stream = sd.InputStream(
            device=self._device_index,
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=self._config.block_size,
        )
        self._stream.start()
        logger.debug(
            "Audio stream started: device=%d, sr=%d, blocksize=%d",
            self._device_index,
            self._config.sample_rate,
            self._config.block_size,
        )

    def stop(self) -> None:
        """Stop and close the InputStream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.debug("Audio stream stopped")

    def get_block(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        Return the next audio block from the queue.

        Returns ``None`` if *timeout* expires with no data.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # -------------------------------------------------------------- #
    # Internal callback (called from sounddevice audio thread)
    # -------------------------------------------------------------- #

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio stream status: %s", status)
            print(status, file=sys.stderr)
        self._queue.put(indata[:, 0].copy())
