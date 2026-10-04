from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return a supported provider name or raise a clear configuration error."""

    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "openai": "openai",
        "open-ai": "openai",
        "custom": "custom",
        "openai-compatible": "custom",
        "openai-compat": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "local": "ollama",
        "openrouter": "openrouter",
        "open-router": "openrouter",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        supported = ", ".join(("openai", "custom", "gemini", "anthropic", "ollama", "openrouter"))
        raise ValueError(f"Unsupported LLM provider {value!r}. Choose one of: {supported}.") from error


def build_chat_model(config: ProviderConfig):
    """Instantiate the LangChain integration configured for one live provider.

    Imports are intentionally local so the deterministic, offline lab can run
    before optional provider packages and API keys are installed.
    """

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    if provider in {"openai", "custom"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise RuntimeError("Install langchain-openai to use the openai or custom provider.") from error
        if config.api_key:
            common["api_key"] = config.api_key
        if config.base_url:
            common["base_url"] = config.base_url
        return ChatOpenAI(**common)

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as error:
            raise RuntimeError("Install langchain-google-genai to use the gemini provider.") from error
        if config.api_key:
            common["api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**common)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as error:
            raise RuntimeError("Install langchain-anthropic to use the anthropic provider.") from error
        if config.api_key:
            common["api_key"] = config.api_key
        return ChatAnthropic(**common)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as error:
            raise RuntimeError("Install langchain-ollama to use the ollama provider.") from error
        if config.base_url:
            common["base_url"] = config.base_url
        return ChatOllama(**common)

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as error:
            raise RuntimeError("Install langchain-openrouter to use the openrouter provider.") from error
        if config.api_key:
            common["api_key"] = config.api_key
        if config.base_url:
            common["base_url"] = config.base_url
        return ChatOpenRouter(**common)

    # Kept as a defensive guard if a new normalize_provider implementation is
    # ever relaxed without updating this factory.
    raise ValueError(f"Unsupported LLM provider {config.provider!r}")
