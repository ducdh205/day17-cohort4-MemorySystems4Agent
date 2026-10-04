from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build a complete, isolated configuration that always exercises compact memory."""

    root = tmp_path.resolve()
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=root / "state",
        compact_threshold_tokens=80,
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Persistent User.md can be created, read, and corrected in isolated state."""

    agent = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    store = agent.profile_store
    store.write_text("dungct", "# User Profile\n\n- location: Huế\n")

    assert "Huế" in store.read_text("dungct")
    assert store.edit_text("dungct", "Huế", "Đà Nẵng") is True
    assert "Đà Nẵng" in store.read_text("dungct")
    assert store.file_size("dungct") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """A low test threshold compacts old messages while retaining a recent tail."""

    config = make_config(tmp_path)
    memory = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)
    for index in range(8):
        memory.append("long-thread", "user", f"turn {index}: " + ("context " * 20))

    assert memory.compaction_count("long-thread") > 0
    assert len(memory.context("long-thread")["messages"]) <= config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced recalls a saved fact in a fresh thread; Baseline deliberately cannot."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    advanced.reply("dungct", "first-thread", "Mình tên là DũngCT.")
    baseline.reply("dungct", "first-thread", "Mình tên là DũngCT.")

    advanced_answer = advanced.reply("dungct", "fresh-thread", "Mình tên gì?")["answer"]
    baseline_answer = baseline.reply("dungct", "fresh-thread", "Mình tên gì?")["answer"]

    assert "DũngCT" in advanced_answer
    assert "DũngCT" not in baseline_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compact memory lowers cumulative prompt context versus unbounded Baseline history."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)
    for index in range(10):
        message = f"Turn {index}: " + ("long context " * 25)
        advanced.reply("dungct", "long-thread", message)
        baseline.reply("dungct", "long-thread", message)

    assert advanced.compaction_count("long-thread") > 0
    assert advanced.prompt_token_usage("long-thread") < baseline.prompt_token_usage("long-thread")
