#!/usr/bin/env python3
"""Checks for generated-token hidden-state extraction.

This test uses a tiny dummy model output so it can run without downloading
Qwen2.5-14B-Instruct. It validates the extraction contract used by the real
QwenAgent.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_qwen.models.qwen_agent import (  # noqa: E402
    AgentGeneration,
    extract_prediction,
    pool_generated_final_layer_hidden_states,
)


@dataclass
class DummyGenerateOutput:
    sequences: torch.Tensor
    hidden_states: tuple[tuple[torch.Tensor, ...], ...]


def build_hidden_states(generated_tokens: int = 3, hidden_dim: int = 4):
    steps = []
    expected_final_token_states = []
    for step in range(generated_tokens):
        lower_layer = torch.zeros(1, 2 + step, hidden_dim)
        final_layer = torch.full((1, 2 + step, hidden_dim), float(step))
        final_layer[:, -1, :] = torch.arange(
            step + 1,
            step + 1 + hidden_dim,
            dtype=final_layer.dtype,
        ).unsqueeze(0)
        expected_final_token_states.append(final_layer[:, -1, :])
        steps.append((lower_layer, final_layer))
    return tuple(steps), torch.stack(expected_final_token_states, dim=1).mean(dim=1)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.allclose(actual, expected):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def test_pooling_contract() -> None:
    hidden_states, expected = build_hidden_states()
    pooled = pool_generated_final_layer_hidden_states(hidden_states, generated_token_count=3)
    assert list(pooled.shape) == [1, 4], pooled.shape
    assert_close(pooled, expected, "pooled hidden state")
    assert not torch.isnan(pooled).any()
    assert not torch.all(pooled == 0)


def test_uses_only_requested_generated_steps() -> None:
    hidden_states, _ = build_hidden_states(generated_tokens=5, hidden_dim=2)
    pooled = pool_generated_final_layer_hidden_states(hidden_states, generated_token_count=2)
    expected = torch.tensor([[4.5, 5.5]])
    assert_close(pooled, expected, "last generated steps only")


def test_agent_generation_metadata() -> None:
    hidden_states, _ = build_hidden_states(generated_tokens=3, hidden_dim=4)
    output = DummyGenerateOutput(
        sequences=torch.tensor([[1, 2, 3, 4, 5]]),
        hidden_states=hidden_states,
    )
    pooled = pool_generated_final_layer_hidden_states(output.hidden_states, generated_token_count=3)
    generation = AgentGeneration(
        response_text="Answer: (A) yes.\nExplanation: direct evidence supports it.",
        prediction=extract_prediction("Answer: (A) yes.", ["yes", "no"]),
        pooled_hidden_state=pooled,
        generated_token_count=3,
        sequence_length=output.sequences.shape[-1],
    )

    assert generation.prediction == "yes"
    assert generation.generated_token_count == 3
    assert generation.sequence_length == 5
    assert generation.pooled_vector_shape == [1, 4]


def test_prediction_extraction() -> None:
    assert extract_prediction("Answer: (B) is more plausible.", ["fake", "legitimate"]) == "legitimate"
    assert extract_prediction("The answer is maybe because evidence is mixed.", ["yes", "no", "maybe"]) == "maybe"
    assert extract_prediction("I cannot determine this.", ["yes", "no"]) == "unknown"


def main() -> None:
    test_pooling_contract()
    test_uses_only_requested_generated_steps()
    test_agent_generation_metadata()
    test_prediction_extraction()
    print("[hidden-state extraction tests passed]")


if __name__ == "__main__":
    main()
