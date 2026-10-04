from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None
        if not force_offline and self._live_model_is_configured():
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Reply with only the state belonging to ``thread_id``."""

        # User identity intentionally does not select or restore state here.
        # Doing so would give this comparison agent cross-session memory.
        del user_id
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Use deterministic within-thread recall for repeatable benchmarks."""

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        answer = self._offline_response(session.messages, message)
        output_tokens = estimate_tokens(answer)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += output_tokens
        session.prompt_tokens_processed += prompt_tokens
        return self._turn_result(answer, output_tokens, prompt_tokens)

    def _maybe_build_langchain_agent(self):
        """Build the configured chat model only for explicitly live configurations."""

        return build_chat_model(self.config.model)

    def _live_model_is_configured(self) -> bool:
        return self.config.model.provider == "ollama" or bool(self.config.model.api_key)

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        result = self.langchain_agent.invoke(session.messages)
        answer = result.content if isinstance(result.content, str) else str(result.content)
        output_tokens = estimate_tokens(answer)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += output_tokens
        session.prompt_tokens_processed += prompt_tokens
        return self._turn_result(answer, output_tokens, prompt_tokens)

    @staticmethod
    def _offline_response(messages: list[dict[str, str]], message: str) -> str:
        facts: dict[str, str] = {}
        for item in messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))

        question = message.lower()
        if "tên" in question and facts.get("name"):
            return f"Trong thread này, bạn cho biết tên là {facts['name']}."
        if ("nghề" in question or "làm gì" in question) and facts.get("profession"):
            return f"Trong thread này, bạn cho biết bạn là {facts['profession']}."
        if ("ở đâu" in question or "nơi ở" in question) and facts.get("location"):
            return f"Trong thread này, bạn cho biết bạn ở {facts['location']}."
        if message.strip().endswith("?"):
            return "Mình chưa có thông tin đó trong thread này."
        return "Mình đã ghi nhận thông tin trong thread hiện tại."

    @staticmethod
    def _turn_result(answer: str, output_tokens: int, prompt_tokens: int) -> dict[str, Any]:
        return {
            "answer": answer,
            "response": answer,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
        }
