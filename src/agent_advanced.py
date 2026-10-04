from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None
        if not force_offline and self._live_model_is_configured():
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:    
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        prompt_tokens = self._record_user_turn(user_id, thread_id, message)
        answer = self._offline_response(user_id, thread_id, message)
        return self._record_assistant_turn(thread_id, answer, prompt_tokens)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        context = self.compact_memory.context(thread_id)
        messages = context["messages"]
        assert isinstance(messages, list)
        recent_tokens = sum(estimate_tokens(str(item.get("content", ""))) for item in messages)
        return (
            estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(str(context["summary"]))
            + recent_tokens
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        del thread_id, message
        facts = self.profile_store.facts(user_id)
        if not facts:
            return "Mình chưa có thông tin ổn định nào để nhắc lại."

        identity = _format_fact_group(facts, ("name", "profession", "interests"))
        details = _format_fact_group(facts, ("location", "favorite_drink", "favorite_food", "pet"))
        style = _format_fact_group(facts, ("response_style",))
        groups = [group for group in (identity, details, style) if group]
        return "\n".join(f"- {group}" for group in groups)

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        return build_chat_model(self.config.model)

    def _live_model_is_configured(self) -> bool:
        return self.config.model.provider == "ollama" or bool(self.config.model.api_key)

    def _record_user_turn(self, user_id: str, thread_id: str, message: str) -> int:
        """Persist stable facts and append the user turn before building context."""

        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        return self._estimate_prompt_context_tokens(user_id, thread_id)

    def _record_assistant_turn(self, thread_id: str, answer: str, prompt_tokens: int) -> dict[str, Any]:
        self.compact_memory.append(thread_id, "assistant", answer)
        output_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return {
            "answer": answer,
            "response": answer,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Use a configured chat model while retaining the same memory accounting."""

        prompt_tokens = self._record_user_turn(user_id, thread_id, message)
        context = self.compact_memory.context(thread_id)
        messages = context["messages"]
        assert isinstance(messages, list)
        memory_prompt = "\n".join(
            [
                "Persistent user profile:",
                self.profile_store.read_text(user_id),
                "Conversation summary:",
                str(context["summary"]),
                "Recent conversation:",
                *[f"{item.get('role', 'message')}: {item.get('content', '')}" for item in messages],
            ]
        )
        result = self.langchain_agent.invoke(memory_prompt)
        answer = result.content if isinstance(result.content, str) else str(result.content)
        return self._record_assistant_turn(thread_id, answer, prompt_tokens)


def _format_fact_group(facts: dict[str, str], keys: tuple[str, ...]) -> str:
    labels = {
        "name": "Tên",
        "profession": "Nghề nghiệp",
        "interests": "Mối quan tâm",
        "location": "Nơi ở hiện tại",
        "favorite_drink": "Đồ uống yêu thích",
        "favorite_food": "Món ăn yêu thích",
        "pet": "Thú cưng",
        "response_style": "Style trả lời",
    }
    return "; ".join(f"{labels[key]}: {facts[key]}" for key in keys if facts.get(key))
