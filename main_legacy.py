import os
import sys
import time
import queue
import wave
import tempfile
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# =========================================================
# CONFIG
# =========================================================

MODEL_SIZE = "small"          # tiny / base / small / medium / large-v3
LANGUAGE = "en"

SAMPLE_RATE = 16000
CHANNELS = 1

CHUNK_SECONDS = 6
OVERLAP_SECONDS = 1

OUTPUT_TRANSCRIPT_FILE = "live_transcript.txt"

# Set True if you want partial debugging info
DEBUG = False

# =========================================================
# GLOBALS
# =========================================================

audio_queue = queue.Queue()

# =========================================================
# AUDIO DEVICE DETECTION
# =========================================================


def find_blackhole_device():
    """
    Auto-detect BlackHole input device on macOS.
    """

    devices = sd.query_devices()

    for idx, dev in enumerate(devices):
        name = dev["name"].lower()

        if (
            "blackhole" in name
            and dev["max_input_channels"] > 0
        ):
            return idx, dev["name"]

    return None, None


def print_input_devices():
    print("\nAvailable Input Devices:\n")

    devices = sd.query_devices()

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"{idx}: {dev['name']}")

# =========================================================
# WAV WRITER
# =========================================================


def save_wav(path, audio_data):
    """
    Save float32 mono audio to 16-bit PCM WAV.
    """

    audio_data = np.clip(audio_data, -1.0, 1.0)
    pcm_audio = (audio_data * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_audio.tobytes())

# =========================================================
# AUDIO CALLBACK
# =========================================================


def audio_callback(indata, frames, time_info, status):

    if status:
        print(status, file=sys.stderr)

    mono_audio = indata[:, 0].copy()

    audio_queue.put(mono_audio)

# =========================================================
# TRANSCRIPTION
# =========================================================


class LiveTranscriber:

    def __init__(self):

        print("\nLoading Whisper model...")
        self.model = WhisperModel(
            MODEL_SIZE,
            compute_type="int8"
        )

        print("Model loaded.\n")

    def transcribe(self, audio_chunk):

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_wav:

            wav_path = temp_wav.name

        try:

            save_wav(wav_path, audio_chunk)

            segments, info = self.model.transcribe(
                wav_path,
                language=LANGUAGE,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500
                ),
                beam_size=5,
                best_of=5,
                temperature=0.0
            )

            texts = []

            for segment in segments:
                txt = segment.text.strip()

                if txt:
                    texts.append(txt)

            final_text = " ".join(texts).strip()

            return final_text

        finally:

            if os.path.exists(wav_path):
                os.remove(wav_path)

# =========================================================
# FILE LOGGER
# =========================================================


def append_to_file(text):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(
        OUTPUT_TRANSCRIPT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(f"[{timestamp}] {text}\n")

# =========================================================
# MAIN LOOP
# =========================================================


def run_transcription(device_index):

    transcriber = LiveTranscriber()

    target_samples = CHUNK_SECONDS * SAMPLE_RATE
    overlap_samples = OVERLAP_SECONDS * SAMPLE_RATE

    pending_audio = np.array([], dtype=np.float32)

    print("Starting live transcription...\n")
    print("Press CTRL+C to stop.\n")

    with sd.InputStream(
        device=device_index,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
        blocksize=4096,
    ):

        try:

            while True:

                audio_block = audio_queue.get()

                pending_audio = np.concatenate(
                    [pending_audio, audio_block]
                )

                if len(pending_audio) >= target_samples:

                    chunk = pending_audio[:target_samples]

                    pending_audio = pending_audio[
                        target_samples - overlap_samples:
                    ]

                    if DEBUG:
                        print(
                            f"Processing {len(chunk)/SAMPLE_RATE:.2f}s audio..."
                        )

                    text = transcriber.transcribe(chunk)

                    if text:

                        print(text)

                        append_to_file(text)

        except KeyboardInterrupt:

            print("\nStopping transcription...\n")

# =========================================================
# MAIN
# =========================================================


def main():

    print("\n===================================")
    print(" Live System Audio Transcription")
    print("===================================\n")

    device_index, device_name = find_blackhole_device()

    if device_index is not None:

        print(f"Detected BlackHole device:")
        print(f"{device_index}: {device_name}\n")

        use_auto = input(
            "Use this device? (Y/n): "
        ).strip().lower()

        if use_auto in ["", "y", "yes"]:

            selected_device = device_index

        else:

            print_input_devices()

            selected_device = int(
                input("\nEnter device index: ")
            )

    else:

        print("BlackHole device not detected.\n")

        print_input_devices()

        selected_device = int(
            input("\nEnter device index: ")
        )

    run_transcription(selected_device)


if __name__ == "__main__":
    main()