"""
WAV file writer utility.
"""

from __future__ import annotations

import wave

import numpy as np


def save_wav(path: str, audio_data: np.ndarray, sample_rate: int) -> None:
    """
    Save a float32 mono audio array to a 16-bit PCM WAV file.

    Parameters
    ----------
    path:
        Destination file path.
    audio_data:
        1-D float32 numpy array in the range [-1.0, 1.0].
    sample_rate:
        Sample rate in Hz (e.g. 16000).
    """
    audio_data = np.clip(audio_data, -1.0, 1.0)
    pcm_audio = (audio_data * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_audio.tobytes())
