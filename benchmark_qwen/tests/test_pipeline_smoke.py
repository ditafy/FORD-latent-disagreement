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
        self.seen_messages = []

    def generate(
        self,
        messages,
        *,
        choices,
        max_new_tokens=1024,
        temperature=0.3,
    ) -> AgentGeneration:
        if not messages:
            raise AssertionError("pipeline did not pass any messages")
        content = messages[0]["content"]
        if "Previous debate transcript:" not in content and "Full debate transcript:" not in content:
            raise AssertionError("pipeline did not pass the expected FORD debate prompt")
        self.seen_messages.append(messages)

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
        require(summary["verdict_source"] == "consensus", "persona verdict should come from consensus")
        require(summary["resolved_verdict_mode"] == "consensus", "persona auto mode should resolve to consensus")
        require(summary["judge_prediction"] is None, "persona consensus mode should not run a judge")
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


def test_stance_role_mode_consensus_ablation() -> None:
    agent_a = DummyAgent(
        "Affirmative",
        vectors=[torch.tensor([[1.0, 0.0]])],
        prediction="yes",
    )
    agent_b = DummyAgent(
        "Negative",
        vectors=[torch.tensor([[0.0, 1.0]])],
        prediction="no",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        summaries = run_pipeline(
            [sample_record()],
            agent_a=agent_a,
            agent_b=agent_b,
            rounds=1,
            max_new_tokens=32,
            temperature=0.0,
            output_dir=output_dir,
            role_mode="stance",
            verdict_mode="consensus",
        )

        require(len(summaries) == 1, "expected one stance summary")
        summary = summaries[0]
        require(summary["role_mode"] == "stance", "role mode should be recorded")
        require(summary["agent_a_role"] == "Affirmative", "agent A should be affirmative in stance mode")
        require(summary["agent_b_role"] == "Negative", "agent B should be negative in stance mode")
        require(summary["agent_a_target_label"] == "yes", "StrategyQA affirmative target should be yes")
        require(summary["agent_b_target_label"] == "no", "StrategyQA negative target should be no")
        require(summary["verdict"] == "unresolved", "opposing stance predictions should not be consensus")
        require(summary["verdict_source"] == "consensus", "consensus ablation should use consensus verdict")
        require(summary["consensus"] is False, "opposing stance predictions should not be consensus")

        round_summary = summary["rounds"][0]
        require(round_summary["agent_a_role"] == "Affirmative", "round A role mismatch")
        require(round_summary["agent_b_role"] == "Negative", "round B role mismatch")
        require(round_summary["agent_a_target_label"] == "yes", "round A target mismatch")
        require(round_summary["agent_b_target_label"] == "no", "round B target mismatch")

        affirmative_prompt = agent_a.seen_messages[0][0]["content"]
        negative_prompt = agent_b.seen_messages[0][0]["content"]
        require("Role instructions:" in affirmative_prompt, "affirmative prompt should include role instructions")
        require("assigned side is affirmative" in affirmative_prompt, "affirmative prompt should state assigned side")
        require("target answer label is 'yes'" in affirmative_prompt, "affirmative prompt should state target")
        require("Do not switch sides" in affirmative_prompt, "affirmative prompt should prevent side switching")
        require("Role instructions:" in negative_prompt, "negative prompt should include role instructions")
        require("assigned side is negative" in negative_prompt, "negative prompt should state assigned side")
        require("target answer label is 'no'" in negative_prompt, "negative prompt should state target")


def test_stance_judge_verdict_mode() -> None:
    agent_a = DummyAgent(
        "Affirmative",
        vectors=[torch.tensor([[1.0, 0.0]])],
        prediction="yes",
    )
    agent_b = DummyAgent(
        "Negative",
        vectors=[torch.tensor([[0.0, 1.0]])],
        prediction="no",
    )
    judge = DummyAgent(
        "Judge",
        vectors=[torch.tensor([[0.9, 0.1]])],
        prediction="yes",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        summaries = run_pipeline(
            [sample_record()],
            agent_a=agent_a,
            agent_b=agent_b,
            judge_agent=judge,
            rounds=1,
            max_new_tokens=32,
            temperature=0.0,
            output_dir=output_dir,
            save_hidden_states=True,
            role_mode="stance",
            verdict_mode="auto",
        )

        require(len(summaries) == 1, "expected one judge summary")
        summary = summaries[0]
        require(summary["role_mode"] == "stance", "role mode should be stance")
        require(summary["verdict_mode"] == "auto", "verdict mode should record requested value")
        require(summary["resolved_verdict_mode"] == "judge", "stance auto mode should resolve to judge")
        require(summary["verdict_source"] == "judge", "stance auto verdict should come from judge")
        require(summary["judge_prediction"] == "yes", "judge prediction mismatch")
        require(summary["verdict"] == "yes", "judge should set the final verdict")
        require(summary["consensus_verdict"] == "unresolved", "side consensus should remain unresolved")
        require(summary["is_correct"] is True, "judge verdict should be correct")
        require(summary["error"] is False, "judge error should be false")
        require(summary["judge_pooled_vector_shape"] == [1, 2], "judge vector shape mismatch")
        require(summary["judge_generated_token_count"] > 0, "judge token count missing")
        require(summary["judge_affirmative_disagreement"] < summary["judge_negative_disagreement"], "judge should be closer to affirmative")
        require(summary["judge_closer_to"] == "affirmative", "judge alignment mismatch")
        require(Path(summary["judge_hidden_state_path"]).exists(), "judge hidden state file missing")

        judge_prompt = judge.seen_messages[0][0]["content"]
        require("Full debate transcript:" in judge_prompt, "judge prompt should include full transcript")
        require("neutral judge" in judge_prompt, "judge prompt should include neutral role instructions")
        require("The debaters were assigned opposing positions" in judge_prompt, "judge prompt should warn about assigned sides")
        require("Answer: (<choice letter>) <choice label>" in judge_prompt, "judge prompt should require answer format")


def main() -> None:
    test_pipeline_smoke()
    test_stance_role_mode_consensus_ablation()
    test_stance_judge_verdict_mode()
    print("[2-agent pipeline smoke test passed]")


if __name__ == "__main__":
    main()
