#!/usr/bin/env python3
"""Smoke test for the 2-agent FORD/Qwen pipeline without loading Qwen."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_qwen.models.qwen_agent import AgentGeneration  # noqa: E402
from benchmark_qwen.pipeline.ford_2agent_qwen import run_pipeline, write_jsonl  # noqa: E402


class DummyAgent:
    def __init__(self, name: str, vectors: list[torch.Tensor], prediction: str = "yes") -> None:
        self.name = name
        self.vectors = vectors
        self.prediction = prediction
        self.calls = 0

    def generate(
        self,
        messages,
        *,
        choices,
        max_new_tokens=1024,
        temperature=0.3,
    ) -> AgentGeneration:
        if not messages or "Previous debate transcript:" not in messages[0]["content"]:
            raise AssertionError("pipeline did not pass the expected FORD debate prompt")

        vector = self.vectors[min(self.calls, len(self.vectors) - 1)]
        self.calls += 1
        return AgentGeneration(
            response_text=(
                f"Answer: (A) {self.prediction}\n"
                f"Explanation: {self.name} dummy response {self.calls}."
            ),
            prediction=self.prediction,
            pooled_hidden_state=vector,
            generated_token_count=5 + self.calls,
            sequence_length=20 + self.calls,
        )


def sample_record():
    return {
        "id": "strategyqa_dummy_001",
        "split": "test",
        "task_type": "strategyqa",
        "answer_type": "binary",
        "text": "Question: Is water wet?\n\nDecide whether the answer is YES or NO.",
        "label": "yes",
        "choices": ["yes", "no"],
        "metadata": {},
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_pipeline_smoke() -> None:
    agent_a = DummyAgent(
        "Agent A",
        vectors=[torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]])],
    )
    agent_b = DummyAgent(
        "Agent B",
        vectors=[torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]])],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        summaries = run_pipeline(
            [sample_record()],
            agent_a=agent_a,
            agent_b=agent_b,
            rounds=2,
            max_new_tokens=32,
            temperature=0.0,
            output_dir=output_dir,
            save_hidden_states=True,
        )
        output_path = output_dir / "summary.jsonl"
        write_jsonl(summaries, output_path)

        require(output_path.exists(), "summary file was not written")
        require(len(summaries) == 1, "expected one summary")
        summary = summaries[0]
        require(summary["id"] == "strategyqa_dummy_001", "id mismatch")
        require(summary["verdict"] == "yes", "verdict should be consensus yes")
        require(summary["is_correct"] is True, "verdict should be correct")
        require(summary["error"] is False, "error should be false")
        require(summary["consensus"] is True, "agents should be in consensus")
        require(summary["false_consensus"] is False, "false consensus should be false")
        require(len(summary["rounds"]) == 2, "expected two round blocks")
        require(summary["round0_disagreement"] == 1.0, "round0 disagreement should be orthogonal")
        require(summary["round1_disagreement"] == 0.0, "round1 disagreement should converge")
        require(summary["convergence_delta"] == 1.0, "convergence delta should be 1.0")
        require(summary["agent_a_predictions"] == ["yes", "yes"], "agent A predictions mismatch")
        require(summary["agent_b_predictions"] == ["yes", "yes"], "agent B predictions mismatch")

        for round_summary in summary["rounds"]:
            require(round_summary["agent_a_generated_token_count"] > 0, "missing A token count")
            require(round_summary["agent_b_generated_token_count"] > 0, "missing B token count")
            require(round_summary["agent_a_pooled_vector_shape"] == [1, 2], "A vector shape mismatch")
            require(round_summary["agent_b_pooled_vector_shape"] == [1, 2], "B vector shape mismatch")
            require(Path(round_summary["agent_a_hidden_state_path"]).exists(), "A hidden state file missing")
            require(Path(round_summary["agent_b_hidden_state_path"]).exists(), "B hidden state file missing")


def main() -> None:
    test_pipeline_smoke()
    print("[2-agent pipeline smoke test passed]")


if __name__ == "__main__":
    main()
