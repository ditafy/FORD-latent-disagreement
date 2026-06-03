#!/usr/bin/env python3
"""Validate converted benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "split",
    "task_type",
    "answer_type",
    "text",
    "label",
    "choices",
    "metadata",
}

VALID_LABELS = {
    "fakenews": {"fake", "legitimate"},
    "strategyqa": {"yes", "no"},
    "pubmedqa": {"yes", "no", "maybe"},
}

VALID_ANSWER_TYPES = {
    "fakenews": "binary",
    "strategyqa": "binary",
    "pubmedqa": "ternary",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_no}: record must be an object")
            records.append(value)
    return records


def validate_record(record: dict[str, Any], path: Path, line_no: int) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        errors.append(f"missing fields {sorted(missing)}")

    task_type = record.get("task_type")
    if task_type not in VALID_LABELS:
        errors.append(f"invalid task_type {task_type!r}")
        return errors

    label = record.get("label")
    if label not in VALID_LABELS[task_type]:
        errors.append(f"invalid label {label!r} for task_type {task_type!r}")

    answer_type = record.get("answer_type")
    expected_answer_type = VALID_ANSWER_TYPES[task_type]
    if answer_type != expected_answer_type:
        errors.append(f"answer_type {answer_type!r} should be {expected_answer_type!r}")

    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        errors.append("text must be a non-empty string")

    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        errors.append("id must be a non-empty string")

    choices = record.get("choices")
    if not isinstance(choices, list) or len(choices) < 2:
        errors.append("choices must be a list with at least two values")
    elif set(choices) != VALID_LABELS[task_type]:
        errors.append(
            f"choices {choices!r} do not match valid labels {sorted(VALID_LABELS[task_type])}"
        )

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")

    if errors:
        return [f"{path} line {line_no}: {error}" for error in errors]
    return []


def validate_file(path: Path) -> tuple[int, Counter[str], Counter[str], list[str]]:
    records = read_jsonl(path)
    errors: list[str] = []
    ids_by_task: dict[str, set[str]] = defaultdict(set)
    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for line_no, record in enumerate(records, start=1):
        errors.extend(validate_record(record, path, line_no))
        task_type = str(record.get("task_type", ""))
        record_id = str(record.get("id", ""))
        if task_type and record_id:
            if record_id in ids_by_task[task_type]:
                errors.append(f"{path} line {line_no}: duplicate id within {task_type}: {record_id}")
            ids_by_task[task_type].add(record_id)
        task_counts[task_type] += 1
        label_counts[f"{task_type}:{record.get('label')}"] += 1

    return len(records), task_counts, label_counts, errors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("benchmark_qwen/data/processed/fakenews.jsonl"),
            Path("benchmark_qwen/data/processed/strategyqa.jsonl"),
            Path("benchmark_qwen/data/processed/pubmedqa.jsonl"),
            Path("benchmark_qwen/data/processed/all_benchmarks.jsonl"),
        ],
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    all_errors: list[str] = []

    for path in args.paths:
        if not path.exists():
            all_errors.append(f"{path}: file does not exist")
            continue
        count, task_counts, label_counts, errors = validate_file(path)
        all_errors.extend(errors)
        print(f"[{path}] records={count}")
        print(f"  tasks: {dict(sorted(task_counts.items()))}")
        print(f"  labels: {dict(sorted(label_counts.items()))}")

    if all_errors:
        print("\n[validation failed]")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\n[validation passed]")


if __name__ == "__main__":
    main()
