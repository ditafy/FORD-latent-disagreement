#!/usr/bin/env python3
"""Analyze round-level disagreement and convergence from a summary JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the average disagreement for each debate round and the "
            "average final convergence_delta from a summary JSONL file."
        )
    )
    parser.add_argument(
        "summary_path",
        nargs="?",
        default="outputs/summary.jsonl",
        help="Path to the summary JSONL file. Default: outputs/summary.jsonl",
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=100,
        help="Warn if the number of valid samples differs from this value. Default: 100",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text table.",
    )
    return parser.parse_args()


def as_float(value: Any, field_name: str, line_number: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: {field_name} is not numeric: {value!r}") from exc


def load_summary(path: Path) -> tuple[dict[int, list[float]], list[float], int, list[str]]:
    round_disagreements: dict[int, list[float]] = defaultdict(list)
    convergence_deltas: list[float] = []
    valid_samples = 0
    warnings: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"line {line_number}: skipped invalid JSON ({exc})")
                continue

            valid_samples += 1

            if "convergence_delta" in sample and sample["convergence_delta"] is not None:
                convergence_deltas.append(
                    as_float(sample["convergence_delta"], "convergence_delta", line_number)
                )
            else:
                warnings.append(f"line {line_number}: missing convergence_delta")

            rounds = sample.get("rounds")
            if isinstance(rounds, list):
                for index, round_summary in enumerate(rounds):
                    if not isinstance(round_summary, dict):
                        warnings.append(f"line {line_number}: skipped non-object round at index {index}")
                        continue
                    round_number = int(round_summary.get("round", index))
                    if "disagreement" not in round_summary:
                        warnings.append(
                            f"line {line_number}: missing disagreement for round {round_number}"
                        )
                        continue
                    round_disagreements[round_number].append(
                        as_float(
                            round_summary["disagreement"],
                            f"round {round_number} disagreement",
                            line_number,
                        )
                    )
            else:
                warnings.append(f"line {line_number}: missing rounds list")

    return round_disagreements, convergence_deltas, valid_samples, warnings


def build_result(
    round_disagreements: dict[int, list[float]],
    convergence_deltas: list[float],
    valid_samples: int,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "sample_count": valid_samples,
        "rounds": [
            {
                "round": round_number,
                "count": len(values),
                "average_disagreement": mean(values),
            }
            for round_number, values in sorted(round_disagreements.items())
        ],
        "convergence_delta_count": len(convergence_deltas),
        "average_convergence_delta": mean(convergence_deltas) if convergence_deltas else None,
        "warnings": warnings,
    }


def print_text_result(result: dict[str, Any], expected_samples: int | None) -> None:
    print(f"Samples: {result['sample_count']}")
    if expected_samples is not None and result["sample_count"] != expected_samples:
        print(f"Warning: expected {expected_samples} samples")

    print("\nRound average disagreement:")
    print("round\tcount\taverage_disagreement")
    for round_result in result["rounds"]:
        print(
            f"{round_result['round']}\t"
            f"{round_result['count']}\t"
            f"{round_result['average_disagreement']:.10f}"
        )

    average_delta = result["average_convergence_delta"]
    if average_delta is None:
        print("\nAverage convergence_delta: unavailable")
    else:
        print(
            "\nAverage convergence_delta: "
            f"{average_delta:.10f} "
            f"(count={result['convergence_delta_count']})"
        )

    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    round_disagreements, convergence_deltas, valid_samples, warnings = load_summary(summary_path)
    result = build_result(round_disagreements, convergence_deltas, valid_samples, warnings)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_text_result(result, args.expected_samples)


if __name__ == "__main__":
    main()
