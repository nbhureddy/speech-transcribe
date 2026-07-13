"""
Configuration loading.

Priority (highest to lowest):
  1. CLI --config <path>
  2. ./config.yaml  (project root)
  3. Built-in defaults (works without any YAML file)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Nested config dataclasses
# ------------------------------------------------------------------ #


@dataclass
class ModelConfig:
    size: str = "large-v3-turbo"
    language: str = "en"
    compute_type: str = "float16"
    # "mlx-whisper" for Apple Silicon native; "faster-whisper" for cross-platform
    backend: str = "mlx-whisper"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_seconds: int = 6
    overlap_seconds: int = 1
    block_size: int = 4096


@dataclass
class TranscriptionConfig:
    vad_filter: bool = True
    min_silence_duration_ms: int = 500
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0


@dataclass
class OutputConfig:
    transcript_dir: str = "transcripts"  # folder; a new timestamped file is created per run


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[str] = None
    suppress_third_party: bool = True  # silence faster_whisper, ctranslate2, etc.


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "anthropic"        # openai | ollama | anthropic
    model: str = "claude-sonnet-4-6" #"gpt-4o-mini"      # provider-specific model name
    api_key: Optional[str] = None   # falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY env vars
    base_url: Optional[str] = None  # override endpoint (e.g. http://localhost:11434/v1 for Ollama)
    temperature: float = 0.1        # low = faithful to source
    max_tokens: int = 4096
    chunk_tokens: int = 0           # split transcript into chunks of this size (large Ollama inputs auto-chunk when 0)
    timeout_seconds: Optional[int] = None  # provider default when null; use larger values for slow local models


# ------------------------------------------------------------------ #
# Root config
# ------------------------------------------------------------------ #


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    debug: bool = False


# ------------------------------------------------------------------ #
# Loader
# ------------------------------------------------------------------ #

_DEFAULT_CONFIG_PATH = "config.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns new dict)."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from a YAML file.

    Falls back to built-in defaults when the file is not found.
    """
    try:
        import yaml  # imported lazily so the rest of the app runs without PyYAML
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required for config loading. Run: pip install pyyaml"
        ) from exc

    resolved = path or _DEFAULT_CONFIG_PATH

    raw: dict = {}
    if os.path.exists(resolved):
        with open(resolved, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                raw = loaded
        logger.debug("Loaded config from %s", resolved)
    else:
        if path:  # user explicitly provided a path that doesn't exist
            raise FileNotFoundError(f"Config file not found: {path}")
        logger.debug("No config.yaml found; using defaults")

    def _get(section: str) -> dict:
        return raw.get(section) or {}

    model_raw = _get("model")
    model = ModelConfig(
        size=model_raw.get("size", "large-v3-turbo"),
        language=model_raw.get("language", "en"),
        compute_type=model_raw.get("compute_type", "float16"),
        backend=model_raw.get("backend", "mlx-whisper"),
    )
    audio = AudioConfig(**_get("audio"))
    transcription = TranscriptionConfig(**_get("transcription"))
    output = OutputConfig(**_get("output"))

    log_raw = _get("logging")
    logging_cfg = LoggingConfig(
        level=log_raw.get("level", "INFO"),
        log_file=log_raw.get("log_file"),
        suppress_third_party=bool(log_raw.get("suppress_third_party", True)),
    )

    llm_raw = _get("llm")
    llm_cfg = LLMConfig(
        enabled=bool(llm_raw.get("enabled", False)),
        provider=llm_raw.get("provider", "openai"),
        model=llm_raw.get("model", "gpt-4o-mini"),
        api_key=llm_raw.get("api_key"),
        base_url=llm_raw.get("base_url"),
        temperature=float(llm_raw.get("temperature", 0.1)),
        max_tokens=int(llm_raw.get("max_tokens", 4096)),
        chunk_tokens=int(llm_raw.get("chunk_tokens", 0)),
        timeout_seconds=(
            int(llm_raw["timeout_seconds"])
            if llm_raw.get("timeout_seconds") is not None
            else None
        ),
    )

    return AppConfig(
        model=model,
        audio=audio,
        transcription=transcription,
        output=output,
        logging=logging_cfg,
        llm=llm_cfg,
        debug=bool(raw.get("debug", False)),
    )
