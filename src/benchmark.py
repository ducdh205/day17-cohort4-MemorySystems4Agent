from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read and minimally validate one benchmark fixture."""

    try:
        conversations = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid benchmark JSON in {path}") from error
    if not isinstance(conversations, list):
        raise ValueError(f"Benchmark fixture {path} must contain a JSON list.")
    required_fields = {"id", "user_id", "turns", "recall_questions"}
    for index, conversation in enumerate(conversations):
        if not isinstance(conversation, dict) or not required_fields.issubset(conversation):
            raise ValueError(f"Conversation {index} in {path} is missing required fields.")
        if not isinstance(conversation["turns"], list) or not isinstance(conversation["recall_questions"], list):
            raise ValueError(f"Conversation {conversation['id']!r} has invalid turns or recall questions.")
    return conversations


def recall_points(answer: str, expected: list[str]) -> float:
    """Score no facts, some facts, or all expected facts as 0 / 0.5 / 1."""

    if not expected:
        return 1.0
    normalized_answer = answer.casefold()
    matches = sum(expected_fact.casefold() in normalized_answer for expected_fact in expected)
    if matches == 0:
        return 0.0
    if matches == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Apply the same deterministic relevance-and-nonempty score to both agents."""

    if not answer.strip():
        return 0.0
    if not expected:
        return 1.0
    matched_fraction = sum(item.casefold() in answer.casefold() for item in expected) / len(expected)
    # A non-empty, but factually irrelevant reply earns a small base score;
    # factual coverage dominates the score and is provider-independent.
    return round(0.25 + (0.75 * matched_fraction), 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate an agent on identical turns and fresh-thread recall questions."""

    del config  # Agents own token accounting; this parameter preserves the lab API.
    thread_ids: list[str] = []
    user_ids: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        thread_id = str(conversation["id"])
        user_id = str(conversation["user_id"])
        thread_ids.append(thread_id)
        user_ids.add(user_id)
        for turn in conversation["turns"]:
            agent.reply(user_id, thread_id, str(turn))

        for question_index, recall in enumerate(conversation["recall_questions"]):
            recall_thread_id = f"{thread_id}:recall:{question_index}"
            thread_ids.append(recall_thread_id)
            response = agent.reply(user_id, recall_thread_id, str(recall["question"]))
            answer = str(response.get("answer", response.get("response", "")))
            expected = [str(item) for item in recall["expected_contains"]]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    memory_file_size = getattr(agent, "memory_file_size", None)
    memory_growth = sum(memory_file_size(user_id) for user_id in user_ids) if callable(memory_file_size) else 0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in thread_ids),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids),
        recall_score=round(sum(recall_scores) / len(recall_scores), 2) if recall_scores else 0.0,
        response_quality=round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0.0,
        memory_growth_bytes=memory_growth,
        compactions=sum(agent.compaction_count(thread_id) for thread_id in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format the required metrics as a dependency-free Markdown table."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            str(row.memory_growth_bytes),
            str(row.compactions),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    """Run the standard and long-context comparisons in deterministic offline mode."""

    config = load_config(Path(__file__).resolve().parent.parent)
    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")
    _print_suite("Standard Benchmark", standard, config)
    print()
    _print_suite("Long-Context Stress Benchmark", stress, config)


def _print_suite(title: str, conversations: list[dict[str, Any]], config) -> None:
    """Run one suite with ephemeral Advanced state, avoiding prior-run pollution."""

    with tempfile.TemporaryDirectory(prefix="benchmark-", dir=config.state_dir) as state_directory:
        suite_config = replace(config, state_dir=Path(state_directory))
        rows = [
            run_agent_benchmark(
                "Baseline",
                BaselineAgent(config=suite_config, force_offline=True),
                conversations,
                suite_config,
            ),
            run_agent_benchmark(
                "Advanced",
                AdvancedAgent(config=suite_config, force_offline=True),
                conversations,
                suite_config,
            ),
        ]
    print(title)
    print(format_rows(rows))


if __name__ == "__main__":
    main()
