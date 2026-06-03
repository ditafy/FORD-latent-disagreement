#!/usr/bin/env python3
"""Convert benchmark datasets into a shared JSONL schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


LABELS = {
    "fakenews": {"fake", "legitimate"},
    "strategyqa": {"yes", "no"},
    "pubmedqa": {"yes", "no", "maybe"},
}

CHOICES = {
    "fakenews": ["fake", "legitimate"],
    "strategyqa": ["yes", "no"],
    "pubmedqa": ["yes", "no", "maybe"],
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_label(value: Any) -> str:
    return str(value).strip().lower()


def convert_fakenews(root: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label in ("fake", "legit"):
        label_dir = root / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing FakeNews label directory: {label_dir}")

        normalized_label = "legitimate" if label == "legit" else "fake"
        for source_path in sorted(label_dir.glob("*.txt")):
            text = source_path.read_text(encoding="utf-8", errors="replace").strip()
            record_id = source_path.stem.replace(".fake", "").replace(".legit", "")
            records.append(
                {
                    "id": f"fakenews_{source_path.stem}",
                    "split": split,
                    "task_type": "fakenews",
                    "answer_type": "binary",
                    "text": (
                        "Article:\n"
                        f"{text}\n\n"
                        "Decide whether this article is FAKE or LEGITIMATE."
                    ),
                    "label": normalized_label,
                    "choices": CHOICES["fakenews"],
                    "metadata": {
                        "source_path": str(source_path),
                        "source_file": source_path.name,
                        "source_id": record_id,
                        "raw_label": label,
                    },
                }
            )
    return records


def convert_processed_jsonl(
    path: Path,
    task_type: str,
    split: str,
    answer_type: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    allowed = LABELS[task_type]
    for index, source in enumerate(read_jsonl(path)):
        label = normalize_label(source.get("label"))
        if label not in allowed:
            raise ValueError(f"{path} record {index} has invalid {task_type} label: {label!r}")

        source_id = str(source.get("id") or f"{task_type}_{index:06d}")
        metadata = source.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        records.append(
            {
                "id": source_id,
                "split": split,
                "task_type": task_type,
                "answer_type": answer_type,
                "text": str(source.get("text", "")).strip(),
                "label": label,
                "choices": CHOICES[task_type],
                "metadata": metadata,
            }
        )
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fakenews-dir", type=Path, default=Path("fakeNewsDataset"))
    parser.add_argument(
        "--strategyqa-path",
        type=Path,
        default=Path("data/jsonlines/strategyqa_processed.jsonl"),
    )
    parser.add_argument(
        "--pubmedqa-path",
        type=Path,
        default=Path("data/jsonlines/pubmedqa_test_processed.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_qwen/data/processed"))
    parser.add_argument("--split", default="test")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    conversions = {
        "fakenews": convert_fakenews(args.fakenews_dir, args.split),
        "strategyqa": convert_processed_jsonl(
            args.strategyqa_path,
            task_type="strategyqa",
            split=args.split,
            answer_type="binary",
        ),
        "pubmedqa": convert_processed_jsonl(
            args.pubmedqa_path,
            task_type="pubmedqa",
            split=args.split,
            answer_type="ternary",
        ),
    }

    total = 0
    for task_type, records in conversions.items():
        output_path = args.output_dir / f"{task_type}.jsonl"
        count = write_jsonl(records, output_path)
        total += count
        print(f"[{task_type}] wrote {count} records -> {output_path}")

    combined_path = args.output_dir / "all_benchmarks.jsonl"
    combined_count = write_jsonl(
        (record for records in conversions.values() for record in records),
        combined_path,
    )
    print(f"[all] wrote {combined_count} records -> {combined_path}")
    print(f"[done] converted {total} records")


if __name__ == "__main__":
    main()
