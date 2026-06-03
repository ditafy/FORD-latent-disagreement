#!/usr/bin/env python3
"""Checks for hidden-state disagreement metrics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_qwen.metrics.hidden_state_metrics import (  # noqa: E402
    convergence_delta,
    cosine_similarity,
    mean_pairwise_disagreement,
    pairwise_disagreement,
    pairwise_disagreement_details,
)


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-6) -> None:
    if not math.isclose(actual, expected, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def assert_raises(fn, expected_error: type[Exception], label: str) -> None:
    try:
        fn()
    except expected_error:
        return
    except Exception as exc:
        raise AssertionError(f"{label}: expected {expected_error.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"{label}: expected {expected_error.__name__}, but no error was raised")


def test_pairwise_reference_cases() -> None:
    same_left = torch.tensor([[1.0, 0.0]])
    same_right = torch.tensor([[1.0, 0.0]])
    opposite = torch.tensor([[-1.0, 0.0]])
    orthogonal = torch.tensor([[0.0, 1.0]])

    assert_close(cosine_similarity(same_left, same_right), 1.0, "same cosine")
    assert_close(pairwise_disagreement(same_left, same_right), 0.0, "same disagreement")
    assert_close(cosine_similarity(same_left, opposite), -1.0, "opposite cosine")
    assert_close(pairwise_disagreement(same_left, opposite), 2.0, "opposite disagreement")
    assert_close(cosine_similarity(same_left, orthogonal), 0.0, "orthogonal cosine")
    assert_close(pairwise_disagreement(same_left, orthogonal), 1.0, "orthogonal disagreement")


def test_accepts_flat_or_batched_vectors() -> None:
    flat = torch.tensor([1.0, 1.0, 0.0])
    batched = torch.tensor([[1.0, 1.0, 0.0]])
    assert_close(pairwise_disagreement(flat, batched), 0.0, "flat/batched disagreement")


def test_mean_pairwise_disagreement() -> None:
    vectors = {
        "agent_a": torch.tensor([[1.0, 0.0]]),
        "agent_b": torch.tensor([[0.0, 1.0]]),
        "agent_c": torch.tensor([[1.0, 0.0]]),
    }
    # AB = 1, AC = 0, BC = 1, mean = 2/3.
    assert_close(mean_pairwise_disagreement(vectors), 2.0 / 3.0, "mean pairwise")

    details = pairwise_disagreement_details(vectors)
    assert len(details) == 3
    assert {(detail.left, detail.right) for detail in details} == {
        ("agent_a", "agent_b"),
        ("agent_a", "agent_c"),
        ("agent_b", "agent_c"),
    }


def test_convergence_delta() -> None:
    assert_close(convergence_delta(0.42, 0.10), 0.32, "positive convergence")
    assert_close(convergence_delta(0.10, 0.42), -0.32, "negative convergence")


def test_invalid_inputs() -> None:
    assert_raises(
        lambda: pairwise_disagreement(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0])),
        ValueError,
        "zero vector",
    )
    assert_raises(
        lambda: pairwise_disagreement(torch.tensor([float("nan"), 1.0]), torch.tensor([1.0, 0.0])),
        ValueError,
        "nan vector",
    )
    assert_raises(
        lambda: pairwise_disagreement(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0, 0.0])),
        ValueError,
        "shape mismatch",
    )
    assert_raises(
        lambda: mean_pairwise_disagreement({"agent_a": torch.tensor([1.0, 0.0])}),
        ValueError,
        "too few vectors",
    )


def main() -> None:
    test_pairwise_reference_cases()
    test_accepts_flat_or_batched_vectors()
    test_mean_pairwise_disagreement()
    test_convergence_delta()
    test_invalid_inputs()
    print("[disagreement metric tests passed]")


if __name__ == "__main__":
    main()
