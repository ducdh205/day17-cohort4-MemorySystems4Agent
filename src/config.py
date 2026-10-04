from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load the shared lab configuration from the environment and ``.env``.

    The defaults deliberately permit the deterministic offline agents to run
    without credentials. A live agent is opt-in: it is only built when its
    caller requests one and the selected provider is configured.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
    except ImportError:
        # ``python-dotenv`` is an optional live-mode convenience. Retaining
        # this fallback keeps the required offline benchmark dependency-free.
        pass
    else:
        load_dotenv(root / ".env")

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    judge_provider = normalize_provider(os.getenv("JUDGE_LLM_PROVIDER", provider))
    judge_model_name = os.getenv("JUDGE_LLM_MODEL", model_name)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        # Short conversations remain intact, while the ~3k-token stress
        # fixture crosses this boundary several times.
        compact_threshold_tokens=_positive_int("COMPACT_THRESHOLD_TOKENS", 800),
        compact_keep_messages=_positive_int("COMPACT_KEEP_MESSAGES", 6),
        model=_provider_config(provider, model_name),
        judge_model=_provider_config(judge_provider, judge_model_name),
    )


def _positive_int(env_name: str, default: int) -> int:
    """Read a positive integer setting and fail early for invalid input."""

    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{env_name} must be a positive integer, got {raw_value!r}") from error
    if value < 1:
        raise ValueError(f"{env_name} must be a positive integer, got {raw_value!r}")
    return value


def _provider_config(provider: str, model_name: str) -> ProviderConfig:
    """Build one provider config using the environment convention in Analysis.md."""

    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "custom": os.getenv("CUSTOM_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "ollama": None,
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
    }
    base_urls = {
        "openai": os.getenv("OPENAI_BASE_URL"),
        "custom": os.getenv("CUSTOM_BASE_URL"),
        "gemini": None,
        "anthropic": None,
        "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "openrouter": os.getenv("OPENROUTER_BASE_URL"),
    }
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        api_key=api_keys[provider],
        base_url=base_urls[provider],
    )
