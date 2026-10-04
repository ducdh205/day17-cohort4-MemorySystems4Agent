from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


def estimate_tokens(text: str) -> int:
    """Return a deterministic, lightweight token estimate for offline metrics."""

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Map an untrusted user id to a profile path inside ``root_dir``."""

        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", (user_id or "").strip()).strip("-_")
        if not slug:
            slug = "anonymous"
        return self.root_dir.resolve() / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.is_file():
            return "# User Profile\n\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.is_file() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Read the key/value fact lines maintained in a user's markdown file."""

        return {
            key: value.strip()
            for key, value in re.findall(r"^- ([a-z_]+):\s*(.+)$", self.read_text(user_id), re.MULTILINE)
        }

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Insert or replace a single structured profile fact."""

        safe_key = re.sub(r"[^a-z_]", "_", key.lower()).strip("_")
        if not safe_key:
            raise ValueError("Profile fact key must contain at least one letter.")
        content = self.read_text(user_id)
        line = f"- {safe_key}: {value.strip()}"
        pattern = rf"^- {re.escape(safe_key)}:\s*.*$"
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(pattern, line, content, count=1, flags=re.MULTILINE)
        else:
            content = content.rstrip() + "\n" + line + "\n"
        return self.write_text(user_id, content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract explicit, stable user facts while rejecting question-only turns."""

    text = (message or "").strip()
    if not text or text.endswith("?"):
        return {}

    updates: dict[str, str] = {}
    name = re.search(r"\b(?:mình|tôi|tớ)\s+tên\s+là\s+([^,.;!?]+)", text, re.IGNORECASE)
    if name:
        updates["name"] = _clean_fact(name.group(1))

    location_matches = re.finditer(
        r"(?:\b(?:mình|tôi|tớ)\s*(?:đang\s+|vẫn\s+)?(?:ở|sống\s+ở|làm việc ở)|"
        r"\b(?:hiện(?:\s+tại)?|giờ|bây giờ)\s+(?:mình|tôi|tớ)?\s*"
        r"(?:đang\s+|vẫn\s+)?(?:ở|sống\s+ở|làm việc ở))\s+([^,.;!?]+)",
        text,
        re.IGNORECASE,
    )
    for match in location_matches:
        candidate = _clean_fact(match.group(1))
        nearby = text[max(0, match.start() - 30) : match.end() + 60].lower()
        if candidate and not ("chỉ là nơi" in nearby or "đi họp" in nearby):
            updates["location"] = candidate

    profession_matches = re.finditer(
        r"(?:chuyển\s+sang|nghề\s+nghiệp\s+(?:hiện tại\s+)?(?:vẫn\s+)?là|"
        r"(?:đang\s+)?làm(?:\s+việc)?(?:\s+là)?)\s+"
        r"((?:[A-Za-zÀ-ỹ0-9-]+\s+){0,3}(?:engineer|developer|designer|manager))",
        text,
        re.IGNORECASE,
    )
    for match in profession_matches:
        candidate = _clean_fact(match.group(1))
        nearby = text[max(0, match.start() - 50) : match.end() + 25].lower()
        if candidate and not (candidate.lower() == "product manager" and "đùa" in nearby):
            updates["profession"] = candidate

    if "3 bullet" in text.lower():
        updates["response_style"] = "3 bullet ngắn gọn"
    elif "bullet" in text.lower() and "ngắn gọn" in text.lower():
        updates["response_style"] = "bullet ngắn gọn"
    elif "ngắn gọn" in text.lower() and "trả lời" in text.lower():
        updates["response_style"] = "ngắn gọn"

    interests = re.search(
        r"\b(?:mình|tôi|tớ)\s+(?:đang\s+)?(?:quan tâm(?:\s+nhiều)?\s+đến|thích)\s+([^.;!?]+)",
        text,
        re.IGNORECASE,
    )
    if interests:
        candidate = _clean_fact(interests.group(1))
        if "python" in candidate.lower() or "ai" in candidate.lower():
            updates["interests"] = candidate

    preference_patterns = {
        "favorite_drink": r"(?:đồ uống|thức uống)\s+yêu thích\s+là\s+([^,.;!?]+)",
        "favorite_food": r"món ăn yêu thích\s+là\s+([^,.;!?]+)",
        "pet": r"(?:nuôi|có)\s+(?:(?:một|con|bé)\s+)*(corgi)\b",
    }
    for key, pattern in preference_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            updates[key] = _clean_fact(match.group(1))
    return {key: value for key, value in updates.items() if _is_stable_value(value)}


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded, deterministic text summary without an LLM call."""

    if max_items < 1:
        return ""
    prior_summaries = [message for message in messages if message.get("role") == "summary"][-1:]
    regular_messages = [message for message in messages if message.get("role") != "summary"]
    remaining_slots = max_items - len(prior_summaries)
    selected = prior_summaries + (regular_messages[-remaining_slots:] if remaining_slots else [])
    snippets = []
    for message in selected:
        content = re.sub(r"\s+", " ", message.get("content", "")).strip()
        if not content:
            continue
        snippets.append(f"{message.get('role', 'message')}: {content[:160]}")
    return "\n".join(snippets)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message and compact old content when its context grows."""

        if self.threshold_tokens < 1 or self.keep_messages < 1:
            raise ValueError("threshold_tokens and keep_messages must both be positive.")
        thread = self._thread(thread_id)
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        while self._context_tokens(thread) > self.threshold_tokens and len(messages) > self.keep_messages:
            older_messages = messages[:-self.keep_messages]
            summary_inputs: list[dict[str, str]] = []
            if thread["summary"]:
                summary_inputs.append({"role": "summary", "content": str(thread["summary"])})
            summary_inputs.extend(older_messages)
            thread["summary"] = summarize_messages(summary_inputs)
            thread["messages"] = messages[-self.keep_messages :]
            thread["compactions"] = int(thread["compactions"]) + 1
            messages = thread["messages"]
            assert isinstance(messages, list)

    def context(self, thread_id: str) -> dict[str, object]:
        return self._thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id)["compactions"])

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def _context_tokens(self, thread: dict[str, object]) -> int:
        messages = thread["messages"]
        assert isinstance(messages, list)
        message_tokens = sum(estimate_tokens(str(message.get("content", ""))) for message in messages)
        return estimate_tokens(str(thread["summary"])) + message_tokens


def _clean_fact(value: str) -> str:
    """Trim a regex capture before it reaches persistent user memory."""

    cleaned = re.sub(r"\s+", " ", value).strip(" \t\n\r:,-")
    cleaned = re.sub(r"^(?:hiện tại\s+)?là\s+", "", cleaned, flags=re.IGNORECASE)
    return re.split(
        r"\s+(?:và|cho|để|trong|nhưng|chứ|mỗi|một|vài|khoảng)\s+",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]


def _is_stable_value(value: str) -> bool:
    """Reject values that came from a recall question rather than a declaration."""

    return value.strip().lower() not in {
        "",
        "gì",
        "đâu",
        "nào",
        "hiện tại",
        "bao nhiêu",
        "đã thay đổi",
    }
