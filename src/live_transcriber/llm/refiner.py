"""
LLM-based transcript refinement.

Takes raw transcribed text (which may contain grammar errors, filler words,
repetitions, etc.) and produces a clean, readable version — without adding
any information that wasn't in the original speech.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from live_transcriber.config import LLMConfig

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Prompt
# ------------------------------------------------------------------ #

_SYSTEM_PROMPT = """\
You are a transcript editor. Your only job is to clean up raw speech-to-text output.

Rules (follow strictly):
1. Fix grammar, punctuation, and sentence structure.
2. Remove filler words and sounds (um, uh, like, you know, right, so, basically, literally, etc.).
3. Remove exact word repetitions and false starts (e.g. "I I I was" → "I was").
4. Preserve ALL facts, names, numbers, and opinions exactly as spoken.
5. Do NOT add any information, commentary, summaries, or opinions of your own.
6. Do NOT remove or paraphrase content — only clean up the language.
7. Output ONLY the cleaned transcript text. No preamble, no explanation.
"""

_USER_TEMPLATE = """\
Clean up the following raw transcript. Follow the rules exactly.

--- RAW TRANSCRIPT ---
{raw_text}
--- END ---
"""


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
        self._client = self._build_client()

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

        logger.info("Sending transcript to %s (%s) for refinement…", self._cfg.provider, self._cfg.model)

        try:
            if self._cfg.provider == "anthropic":
                return self._call_anthropic(raw_text)
            else:
                return self._call_openai_compat(raw_text)
        except Exception as exc:
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
        provider = self._cfg.provider.lower()

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
        if self._cfg.base_url:
            kwargs["base_url"] = self._cfg.base_url

        return OpenAI(**kwargs)

    # -------------------------------------------------------------- #
    # Provider implementations
    # -------------------------------------------------------------- #

    def _call_openai_compat(self, raw_text: str) -> str:
        response = self._client.chat.completions.create(
            model=self._cfg.model,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(raw_text=raw_text)},
            ],
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, raw_text: str) -> str:
        import anthropic

        api_key = self._cfg.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
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
