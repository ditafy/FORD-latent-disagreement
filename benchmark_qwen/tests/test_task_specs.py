#!/usr/bin/env python3
"""Checks for ED2D-style task role specs."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_qwen.pipeline.task_specs import create_role_configs, get_task_spec  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_raises(fn, expected_error: type[Exception], label: str) -> None:
    try:
        fn()
    except expected_error:
        return
    except Exception as exc:
        raise AssertionError(f"{label}: expected {expected_error.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"{label}: expected {expected_error.__name__}, but no error was raised")


def test_task_specs() -> None:
    fakenews = get_task_spec("fakenews")
    require(fakenews.affirmative_label == "legitimate", "FakeNews affirmative label mismatch")
    require(fakenews.negative_label == "fake", "FakeNews negative label mismatch")

    strategyqa = get_task_spec("strategyqa")
    require(strategyqa.affirmative_label == "yes", "StrategyQA affirmative label mismatch")
    require(strategyqa.negative_label == "no", "StrategyQA negative label mismatch")

    pubmedqa = get_task_spec("pubmedqa")
    require(pubmedqa.judge_labels == ("yes", "no", "maybe"), "PubMedQA judge labels should include maybe")


def test_create_role_configs() -> None:
    record = {
        "id": "strategyqa_dummy_001",
        "task_type": "strategyqa",
        "choices": ["yes", "no"],
    }
    affirmative, negative = create_role_configs(record)

    require(affirmative.name == "Affirmative", "affirmative role name mismatch")
    require(affirmative.side == "affirmative", "affirmative side mismatch")
    require(affirmative.target_label == "yes", "affirmative target label mismatch")
    require("answer is YES" in affirmative.meta_prompt, "affirmative meta prompt should state the stance")

    require(negative.name == "Negative", "negative role name mismatch")
    require(negative.side == "negative", "negative side mismatch")
    require(negative.target_label == "no", "negative target label mismatch")
    require("answer is NO" in negative.meta_prompt, "negative meta prompt should state the stance")


def test_validation_errors() -> None:
    require_raises(lambda: get_task_spec("unknown"), ValueError, "unknown task")
    require_raises(
        lambda: create_role_configs({"id": "bad", "task_type": "strategyqa", "choices": ["yes"]}),
        ValueError,
        "missing negative label",
    )


def main() -> None:
    test_task_specs()
    test_create_role_configs()
    test_validation_errors()
    print("[task specs tests passed]")


if __name__ == "__main__":
    main()
