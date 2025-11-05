"""Remote LLM API utilities for WIMHF."""

from __future__ import annotations

import os
import time
from typing import Any

import openai

_CACHED_CLIENT: openai.OpenAI | None = None

MODEL_ABBREV_TO_ID = {
    "gpt4o": "gpt-4o-2024-11-20",
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-5": "gpt-5",
    "gpt5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt5-mini": "gpt-5-mini",
}

DEFAULT_MODEL = "gpt-4.1-mini"


def get_client() -> openai.OpenAI:
    """Return a cached OpenAI client, initialising it if necessary."""
    global _CACHED_CLIENT
    if _CACHED_CLIENT is not None:
        return _CACHED_CLIENT

    api_key = os.environ.get("OAI_WIMHF")
    if not api_key:
        raise ValueError(
            "Set OAI_WIMHF in the environment before calling WIMHF LLM utilities."
        )

    _CACHED_CLIENT = openai.OpenAI(api_key=api_key)
    return _CACHED_CLIENT


def get_completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 15.0,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> str:
    """Fetch a chat completion with retry-and-backoff handling."""
    client = get_client()
    model_id = MODEL_ABBREV_TO_ID.get(model, model)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
                **kwargs,
            )
            return response.choices[0].message.content
        except (openai.RateLimitError, openai.APITimeoutError) as exc:
            if attempt == max_retries - 1:
                raise exc

            wait_time = timeout * (backoff_factor**attempt)
            if attempt > 0:
                print(
                    f"OpenAI API error ({exc}); retrying in {wait_time:.1f}s "
                    f"({attempt + 1}/{max_retries})"
                )
            time.sleep(wait_time)

    raise RuntimeError("Failed to obtain a completion after retries.")
