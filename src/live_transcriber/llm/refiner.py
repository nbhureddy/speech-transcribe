"""
LLM-based transcript refinement.

Takes raw transcribed text (which may contain grammar errors, filler words,
repetitions, etc.) and produces a clean, readable version — without adding
any information that wasn't in the original speech.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen
from typing import Optional

from live_transcriber.config import LLMConfig

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Prompt
# ------------------------------------------------------------------ #

_SYSTEM_PROMPT = """\
You are a transcript editor. Clean up raw speech-to-text output.

Instructions:
- Fix grammar, punctuation, and sentence structure.
- Remove filler words (um, uh, like, you know, right, so, basically, literally).
- Remove word repetitions and false starts ("I I I was" → "I was").
- Keep ALL facts, names, numbers, and opinions exactly as spoken.
- Do NOT add information, summaries, or commentary.
- Do NOT paraphrase — only fix language.

Output ONLY the cleaned text. No introduction. No explanation. No markdown.
"""

_USER_TEMPLATE = """\
Clean this transcript:

{raw_text}
"""

# Approximate chars per token for rough chunking estimates
_CHARS_PER_TOKEN = 4
_DEFAULT_OLLAMA_CHUNK_TOKENS = 3000
_AUTO_OLLAMA_CHUNK_TRIGGER_TOKENS = 3000
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
_DEFAULT_OLLAMA_TIMEOUT_SECONDS = 1800
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ------------------------------------------------------------------ #
# Refiner
# ------------------------------------------------------------------ #


class TranscriptRefiner:
    """
    Sends accumulated transcript text through an LLM for cleanup.

    Supports OpenAI, Ollama (OpenAI-compatible), and Anthropic.

    Parameters
    ----------
    config:
        LLM configuration section from AppConfig.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._cfg = config
        self._provider = config.provider.lower().strip()
        self._base_url = self._resolve_base_url()
        self._client = self._build_client()
        self._resolved_model: Optional[str] = None

    # -------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------- #

    def refine(self, raw_text: str) -> str:
        """
        Refine *raw_text* and return the cleaned version.

        Parameters
        ----------
        raw_text:
            The full raw transcript accumulated during the session.

        Returns
        -------
        str
            Cleaned transcript text from the LLM.
        """
        if not raw_text.strip():
            logger.warning("No transcript text to refine.")
            return ""

        prepared_text = raw_text.strip()
        chunk_tokens = self._effective_chunk_tokens(prepared_text)
        estimated_tokens = self._estimate_tokens(prepared_text)

        logger.info("Sending transcript to %s (%s) for refinement…", self._cfg.provider, self._cfg.model)
        logger.info(
            "Refinement payload: %d chars (~%d tokens)%s",
            len(prepared_text),
            estimated_tokens,
            f", chunk_tokens={chunk_tokens}" if chunk_tokens > 0 else "",
        )

        try:
            if chunk_tokens > 0:
                return self._refine_chunked(prepared_text, chunk_tokens)
            if self._provider == "anthropic":
                return self._call_anthropic(prepared_text)
            else:
                return self._call_openai_compat(prepared_text)
        except Exception as exc:
            if self._provider == "ollama":
                logger.error(
                    "Ollama refinement failed after %ss timeout. "
                    "Increase llm.timeout_seconds or reduce chunk size/model size.",
                    self._request_timeout_seconds(),
                )
            logger.error("LLM refinement failed: %s", exc)
            raise

    def refine_to_file(self, raw_text: str, raw_file_path: str) -> str:
        """
        Refine *raw_text* and write the result to ``<raw_stem>_refined.txt``.

        Returns the path of the refined file.
        """
        refined = self.refine(raw_text)

        stem, _ = os.path.splitext(raw_file_path)
        refined_path = f"{stem}_refined.txt"

        with open(refined_path, "w", encoding="utf-8") as fh:
            fh.write(refined)
            if not refined.endswith("\n"):
                fh.write("\n")

        logger.info("Refined transcript saved: %s", refined_path)
        return refined_path

    # -------------------------------------------------------------- #
    # Client builders
    # -------------------------------------------------------------- #

    def _build_client(self):
        provider = self._provider

        if provider == "anthropic":
            try:
                import anthropic  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "anthropic package is required for the 'anthropic' provider. "
                    "Run: pip install anthropic"
                ) from exc
            return None  # client built per-call to avoid import at module level

        # OpenAI or Ollama (OpenAI-compatible)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Run: pip install openai"
            ) from exc

        api_key = (
            self._cfg.api_key
            or os.environ.get("OPENAI_API_KEY", "")
        )
        kwargs: dict = {"api_key": api_key or "no-key"}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        kwargs["timeout"] = self._request_timeout_seconds()
        if self._provider == "ollama":
            kwargs["max_retries"] = 0

        return OpenAI(**kwargs)

    def _resolve_base_url(self) -> Optional[str]:
        base_url = self._cfg.base_url.strip() if self._cfg.base_url else None
        if base_url or self._provider != "ollama":
            return base_url
        # Ollama runs an OpenAI-compatible API at /v1 on localhost by default.
        return "http://localhost:11434/v1"

    def _ollama_tags_url(self) -> str:
        if not self._base_url:
            raise ValueError("Ollama base URL is not configured.")
        parts = urlsplit(self._base_url)
        path = parts.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parts.scheme, parts.netloc, f"{path}/api/tags", "", ""))

    def _resolve_ollama_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model

        requested = self._cfg.model.strip()
        try:
            import json

            with urlopen(self._ollama_tags_url(), timeout=10) as response:
                payload = json.load(response)
        except Exception as exc:
            raise RuntimeError(
                f"Could not query Ollama models from {self._ollama_tags_url()}. "
                "Verify Ollama is running and base_url points to correct host."
            ) from exc

        available = sorted(
            model.get("name", "").strip()
            for model in payload.get("models", [])
            if model.get("name")
        )
        available_set = set(available)

        if requested in available_set:
            self._resolved_model = requested
            return self._resolved_model

        if ":" not in requested:
            latest = f"{requested}:latest"
            if latest in available_set:
                logger.info("Using Ollama model %s for requested %s", latest, requested)
                self._resolved_model = latest
                return self._resolved_model

        available_text = ", ".join(available) if available else "(none)"
        raise ValueError(
            f"Ollama model {requested!r} not found at {self._base_url}. "
            f"Available models: {available_text}"
        )

    def _estimate_tokens(self, raw_text: str) -> int:
        return max(1, (len(raw_text) + (_CHARS_PER_TOKEN - 1)) // _CHARS_PER_TOKEN)

    def _request_timeout_seconds(self) -> int:
        if self._cfg.timeout_seconds is not None:
            return self._cfg.timeout_seconds
        if self._provider == "ollama":
            return _DEFAULT_OLLAMA_TIMEOUT_SECONDS
        return _DEFAULT_REQUEST_TIMEOUT_SECONDS

    def _effective_chunk_tokens(self, raw_text: str) -> int:
        if self._cfg.chunk_tokens > 0:
            return self._cfg.chunk_tokens

        if self._provider != "ollama":
            return 0

        estimated_tokens = self._estimate_tokens(raw_text)
        if estimated_tokens < _AUTO_OLLAMA_CHUNK_TRIGGER_TOKENS:
            return 0

        logger.info(
            "Large Ollama refinement payload detected (~%d tokens); auto-chunking at %d tokens",
            estimated_tokens,
            _DEFAULT_OLLAMA_CHUNK_TOKENS,
        )
        return _DEFAULT_OLLAMA_CHUNK_TOKENS

    def _resolve_model_name(self) -> str:
        if self._provider == "ollama":
            return self._resolve_ollama_model()
        return self._cfg.model

    def _split_text_into_chunks(self, raw_text: str, chunk_tokens: int) -> list[str]:
        max_chars = chunk_tokens * _CHARS_PER_TOKEN
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush_current() -> None:
            nonlocal current, current_len
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

        def append_piece(piece: str) -> None:
            nonlocal current_len
            if current and current_len + len(piece) > max_chars:
                flush_current()
            current.append(piece)
            current_len += len(piece) + 1

        segments = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(raw_text.strip()) if segment.strip()]

        for segment in segments:
            if len(segment) <= max_chars:
                append_piece(segment)
                continue

            words = segment.split()
            oversized: list[str] = []
            oversized_len = 0
            for word in words:
                if oversized and oversized_len + len(word) + 1 > max_chars:
                    append_piece(" ".join(oversized))
                    oversized = []
                    oversized_len = 0
                oversized.append(word)
                oversized_len += len(word) + 1
            if oversized:
                append_piece(" ".join(oversized))

        flush_current()
        return chunks

    # -------------------------------------------------------------- #
    # Provider implementations
    # -------------------------------------------------------------- #

    def _call_openai_compat(self, raw_text: str) -> str:
        response = self._client.chat.completions.create(
            model=self._resolve_model_name(),
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(raw_text=raw_text)},
            ],
        )
        return response.choices[0].message.content.strip()

    def _refine_chunked(self, raw_text: str, chunk_tokens: int) -> str:
        """Split *raw_text* into chunks and refine each independently."""
        chunks = self._split_text_into_chunks(raw_text, chunk_tokens)
        logger.info("Refining transcript in %d chunk(s) (chunk_tokens=%d)", len(chunks), chunk_tokens)

        refined_parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            logger.debug("Refining chunk %d/%d…", i, len(chunks))
            if self._provider == "anthropic":
                refined_parts.append(self._call_anthropic(chunk))
            else:
                refined_parts.append(self._call_openai_compat(chunk))

        return " ".join(refined_parts)

    def _call_anthropic(self, raw_text: str) -> str:
        import anthropic

        api_key = self._cfg.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY env var or api_key in config."
            )
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=self._cfg.model,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _USER_TEMPLATE.format(raw_text=raw_text)},
            ],
        )
        return message.content[0].text.strip()
