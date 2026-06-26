"""
Command-line entry point for live-transcriber.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np

from live_transcriber.audio.capture import AudioCapture
from live_transcriber.audio.devices import find_default_input_device
from live_transcriber.config import AppConfig, load_config
from live_transcriber.io.transcript_logger import TranscriptLogger
from live_transcriber.llm.refiner import TranscriptRefiner
from live_transcriber.transcription import get_transcriber

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Context resolution
# ------------------------------------------------------------------ #


def _resolve_context(
    inline: Optional[str],
    file_path: Optional[str],
) -> Optional[str]:
    """
    Resolve session context from CLI flags or an interactive prompt.

    Priority: --context-file > --context > interactive prompt (Enter to skip).
    """
    if file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Context file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None

    if inline:
        return inline.strip() or None

    # Interactive fallback — one-line or multi-line input
    print("Enter session context (speaker, topic, background…)")
    print("  • Single line : just type and press Enter")
    print("  • Multi-line  : end with a blank line")
    print("  • Skip        : press Enter on an empty line\n")

    lines: list[str] = []
    while True:
        try:
            line = input("> " if not lines else "  ")
        except EOFError:
            break
        if not line and not lines:
            break          # first line empty → skip
        if not line:
            break          # blank line ends multi-line input
        lines.append(line)

    return "\n".join(lines).strip() or None


# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #


def _setup_logging(config: AppConfig) -> None:
    """Configure the root logger based on AppConfig."""
    level = logging.DEBUG if config.debug else getattr(logging, config.logging.level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if config.logging.log_file:
        import os
        os.makedirs(os.path.dirname(config.logging.log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(config.logging.log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Silence noisy third-party loggers (faster_whisper, ctranslate2) unless debugging
    if config.logging.suppress_third_party and not config.debug:
        for noisy in ("faster_whisper", "ctranslate2"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# ------------------------------------------------------------------ #
# Main transcription loop
# ------------------------------------------------------------------ #


def run_transcription(config: AppConfig, device_index: int, context: Optional[str] = None) -> None:
    """Start the live transcription loop until the user presses CTRL+C."""
    transcriber = get_transcriber(config)
    transcript_logger = TranscriptLogger(config.output.transcript_dir, context=context)

    sample_rate = config.audio.sample_rate
    target_samples = config.audio.chunk_seconds * sample_rate
    overlap_samples = config.audio.overlap_seconds * sample_rate

    pending_audio = np.array([], dtype=np.float32)
    raw_chunks: list[str] = []   # accumulates all transcribed text for LLM refinement

    logger.info("Starting live transcription — press CTRL+C to stop.")

    with AudioCapture(config.audio, device_index) as capture:
        try:
            while True:
                block = capture.get_block()
                if block is None:
                    continue

                pending_audio = np.concatenate([pending_audio, block])

                if len(pending_audio) < target_samples:
                    continue

                chunk = pending_audio[:target_samples]
                pending_audio = pending_audio[target_samples - overlap_samples:]

                if config.debug:
                    logger.debug("Processing %.2fs audio…", len(chunk) / sample_rate)

                text = transcriber.transcribe(chunk)

                if text:
                    print(text)
                    transcript_logger.log(text)
                    raw_chunks.append(text)

        except KeyboardInterrupt:
            logger.info("Transcription stopped by user.")
            print("\nStopping transcription…\n")

    # ── LLM refinement (runs after the audio stream is closed) ──────
    if config.llm.enabled and raw_chunks:
        print("Refining transcript with LLM…")
        try:
            refiner = TranscriptRefiner(config.llm)
            full_raw = "\n".join(raw_chunks)
            refined_path = refiner.refine_to_file(full_raw, transcript_logger.file_path)
            print(f"Refined transcript saved → {refined_path}\n")
        except Exception as exc:
            logger.error("LLM refinement failed — raw transcript is still saved. Error: %s", exc)


# ------------------------------------------------------------------ #
# CLI argument parsing
# ------------------------------------------------------------------ #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live-transcriber",
        description="Live system-audio transcription powered by faster-whisper.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a config.yaml file (default: ./config.yaml).",
    )
    parser.add_argument(
        "--device",
        metavar="INDEX",
        type=int,
        default=None,
        help="Audio input device index (skips interactive prompt).",
    )
    parser.add_argument(
        "--context",
        metavar="TEXT",
        default=None,
        help="Inline context note about the session (speaker, topic, etc.).",
    )
    parser.add_argument(
        "--context-file",
        metavar="PATH",
        default=None,
        dest="context_file",
        help="Path to a plain-text file containing the session context.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug-level logging (overrides config).",
    )
    return parser


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    if args.debug:
        config.debug = True
        config.logging.level = "DEBUG"

    _setup_logging(config)

    print("\n===================================")
    print(" Live System Audio Transcription")
    print("===================================\n")

    device_index = args.device if args.device is not None else find_default_input_device()[0]
    context = _resolve_context(args.context, args.context_file)
    run_transcription(config, device_index, context=context)


if __name__ == "__main__":
    main()
