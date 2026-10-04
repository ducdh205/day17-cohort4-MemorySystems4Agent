from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import BenchmarkRow, format_rows, load_conversations, recall_points, run_agent_benchmark
from config import load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)


class MemoryStoreTests(unittest.TestCase):
    def test_token_estimate_is_deterministic_and_monotonic(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("   \n"), 0)
        self.assertEqual(estimate_tokens("abcd"), estimate_tokens("abcd"))
        self.assertGreaterEqual(estimate_tokens("abcdefgh"), estimate_tokens("abcd"))

    def test_profile_store_sanitizes_path_and_edits_one_fact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = UserProfileStore(Path(directory) / "profiles")

            path = store.write_text("../dungct", "# User Profile\n\n- location: Huế\n")
            self.assertTrue(path.is_relative_to(store.root_dir.resolve()))
            self.assertEqual(store.read_text("../dungct"), "# User Profile\n\n- location: Huế\n")
            self.assertTrue(store.edit_text("../dungct", "Huế", "Đà Nẵng"))
            self.assertFalse(store.edit_text("../dungct", "Hà Nội", "Huế"))
            self.assertIn("Đà Nẵng", store.read_text("../dungct"))
            self.assertGreater(store.file_size("../dungct"), 0)

    def test_profile_extractor_keeps_explicit_facts_and_skips_questions(self) -> None:
        facts = extract_profile_updates(
            "Mình tên là DũngCT, hiện ở Huế và đang làm MLOps engineer. "
            "Mình muốn câu trả lời ngắn gọn theo 3 bullet."
        )

        self.assertEqual(facts["name"], "DũngCT")
        self.assertEqual(facts["location"], "Huế")
        self.assertEqual(facts["profession"], "MLOps engineer")
        self.assertIn("3 bullet", facts["response_style"])
        self.assertEqual(extract_profile_updates("Mình tên gì?"), {})

    def test_profile_extractor_keeps_the_new_location_in_a_correction(self) -> None:
        facts = extract_profile_updates(
            "Mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa."
        )

        self.assertEqual(facts["location"], "Huế")

    def test_compaction_summarizes_old_messages_and_keeps_recent_tail(self) -> None:
        memory = CompactMemoryManager(threshold_tokens=12, keep_messages=2)
        for index in range(5):
            memory.append("thread", "user", f"message {index} with enough text")

        context = memory.context("thread")
        self.assertGreater(memory.compaction_count("thread"), 0)
        self.assertLessEqual(len(context["messages"]), 2)
        self.assertTrue(context["summary"])
        self.assertEqual(context["messages"][-1]["content"], "message 4 with enough text")


class BaselineAgentTests(unittest.TestCase):
    def test_baseline_recalls_only_within_the_same_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory))
            agent = BaselineAgent(config=config, force_offline=True)

            agent.reply("dungct", "first-thread", "Mình tên là DũngCT.")
            same_thread = agent.reply("dungct", "first-thread", "Mình tên gì?")
            new_thread = agent.reply("dungct", "second-thread", "Mình tên gì?")

            self.assertIn("DũngCT", same_thread["answer"])
            self.assertNotIn("DũngCT", new_thread["answer"])
            self.assertGreater(agent.token_usage("first-thread"), 0)
            self.assertGreater(agent.prompt_token_usage("first-thread"), 0)
            self.assertEqual(agent.compaction_count("first-thread"), 0)


class AdvancedAgentTests(unittest.TestCase):
    def test_advanced_persists_corrected_facts_and_recalls_them_in_a_new_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory))
            agent = AdvancedAgent(config=config, force_offline=True)

            agent.reply(
                "dungct",
                "setup-thread",
                "Mình tên là DũngCT, hiện ở Huế và đang làm MLOps engineer. "
                "Mình muốn câu trả lời ngắn gọn theo 3 bullet.",
            )
            agent.reply("dungct", "setup-thread", "Từ tuần này mình đang làm việc ở Đà Nẵng.")
            recalled = agent.reply(
                "dungct",
                "recall-thread",
                "Nhắc lại tên, nghề nghiệp, nơi ở hiện tại và style trả lời mình thích.",
            )

            self.assertIn("DũngCT", recalled["answer"])
            self.assertIn("MLOps engineer", recalled["answer"])
            self.assertIn("Đà Nẵng", recalled["answer"])
            self.assertIn("3 bullet", recalled["answer"])
            self.assertEqual(agent.profile_store.facts("dungct")["location"], "Đà Nẵng")
            self.assertGreater(agent.memory_file_size("dungct"), 0)

    def test_advanced_compaction_reduces_prompt_load_against_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory))
            config.compact_threshold_tokens = 80
            config.compact_keep_messages = 2
            advanced = AdvancedAgent(config=config, force_offline=True)
            baseline = BaselineAgent(config=config, force_offline=True)

            for index in range(8):
                message = f"Long turn {index}: " + ("memory context " * 20)
                advanced.reply("dungct", "long-thread", message)
                baseline.reply("dungct", "long-thread", message)

            self.assertGreater(advanced.compaction_count("long-thread"), 0)
            self.assertLess(
                advanced.prompt_token_usage("long-thread"),
                baseline.prompt_token_usage("long-thread"),
            )
            self.assertGreater(advanced.token_usage("long-thread"), 0)

    def test_advanced_recall_matches_the_long_context_fixture(self) -> None:
        fixture_path = Path(__file__).parent.parent / "data" / "advanced_long_context.json"
        conversation = json.loads(fixture_path.read_text(encoding="utf-8"))[0]

        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory))
            agent = AdvancedAgent(config=config, force_offline=True)
            for turn in conversation["turns"]:
                agent.reply(conversation["user_id"], conversation["id"], turn)

            for recall in conversation["recall_questions"]:
                answer = agent.reply(conversation["user_id"], "fresh-recall", recall["question"])["answer"]
                for expected in recall["expected_contains"]:
                    self.assertIn(expected, answer)

            self.assertGreater(agent.compaction_count(conversation["id"]), 0)

    def test_advanced_recall_matches_the_standard_fixture(self) -> None:
        fixture_path = Path(__file__).parent.parent / "data" / "conversations.json"
        conversations = json.loads(fixture_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory))
            agent = AdvancedAgent(config=config, force_offline=True)
            for conversation in conversations:
                for turn in conversation["turns"]:
                    agent.reply(conversation["user_id"], conversation["id"], turn)
                for index, recall in enumerate(conversation["recall_questions"]):
                    answer = agent.reply(
                        conversation["user_id"],
                        f"{conversation['id']}:recall:{index}",
                        recall["question"],
                    )["answer"]
                    for expected in recall["expected_contains"]:
                        self.assertIn(expected, answer)


class BenchmarkTests(unittest.TestCase):
    def test_loader_scoring_and_formatter_follow_the_benchmark_contract(self) -> None:
        conversations = load_conversations(Path(__file__).parent.parent / "data" / "conversations.json")

        self.assertEqual(len(conversations), 10)
        self.assertEqual(recall_points("DũngCT", ["DũngCT", "cà phê sữa đá"]), 0.5)
        self.assertEqual(recall_points("DũngCT và cà phê sữa đá", ["DũngCT", "cà phê sữa đá"]), 1.0)
        self.assertEqual(recall_points("Không rõ", ["DũngCT"]), 0.0)

        table = format_rows(
            [
                BenchmarkRow("Baseline", 10, 20, 0.0, 0.25, 0, 0),
                BenchmarkRow("Advanced", 12, 18, 1.0, 1.0, 42, 3),
            ]
        )
        self.assertIn("Agent tokens only", table)
        self.assertIn("Memory growth (bytes)", table)
        self.assertIn("Advanced", table)

    def test_benchmark_uses_a_fresh_thread_for_cross_session_recall(self) -> None:
        conversations = [
            {
                "id": "mini",
                "user_id": "dungct",
                "turns": ["Mình tên là DũngCT."],
                "recall_questions": [{"question": "Mình tên gì?", "expected_contains": ["DũngCT"]}],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory))
            baseline = BaselineAgent(config=config, force_offline=True)
            advanced = AdvancedAgent(config=config, force_offline=True)

            baseline_row = run_agent_benchmark("Baseline", baseline, conversations, config)
            advanced_row = run_agent_benchmark("Advanced", advanced, conversations, config)

            self.assertEqual(baseline_row.recall_score, 0.0)
            self.assertEqual(advanced_row.recall_score, 1.0)
            self.assertGreater(advanced_row.memory_growth_bytes, 0)


if __name__ == "__main__":
    unittest.main()
