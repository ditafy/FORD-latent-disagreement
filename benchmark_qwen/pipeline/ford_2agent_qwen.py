#!/usr/bin/env python3
"""Two-agent FORD-style debate pipeline for Qwen hidden-state benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Protocol

import torch

from benchmark_qwen.metrics.hidden_state_metrics import convergence_delta, pairwise_disagreement
from benchmark_qwen.models.qwen_agent import AgentGeneration, QwenAgent


DEFAULT_AGENT_A_PERSONA = (
    "You are Agent A, an evidence-focused analyst. Base your answer on concrete "
    "evidence in the input and explain briefly."
)
DEFAULT_AGENT_B_PERSONA = (
    "You are Agent B, a skeptical fact-checker. Look for weaknesses, alternative "
    "interpretations, and misleading surface cues before answering briefly."
)


class DebateAgent(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        choices: list[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AgentGeneration:
        ...


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid JSON: {exc}") from exc


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def choice_text(choices: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return " ".join(f"({letters[index]}) {choice}" for index, choice in enumerate(choices))


def safe_file_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:120] or "sample"


def normalize_label(value: Any) -> str:
    return str(value).strip().lower()


def render_transcript(transcript: list[dict[str, Any]]) -> str:
    if not transcript:
        return "No previous debate turns."
    lines = []
    for turn in transcript:
        lines.append(
            f"Round {turn['round']} {turn['agent_id']} "
            f"(prediction: {turn['prediction']}): {turn['response_text']}"
        )
    return "\n".join(lines)


def build_messages(
    record: dict[str, Any],
    *,
    agent_id: str,
    round_index: int,
    transcript: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if round_index == 0:
        instruction = (
            f"You are {agent_id}. Give your initial answer and a short explanation. "
            "Use exactly this format:\n"
            "Answer: (<choice letter>) <choice label>\n"
            "Explanation: <brief reason>"
        )
    else:
        instruction = (
            f"You are {agent_id}. Read the question and previous debate turns. "
            "You may defend your previous view or revise it if the other agent is more convincing. "
            "Use exactly this format:\n"
            "Answer: (<choice letter>) <choice label>\n"
            "Explanation: <brief reason>"
        )

    content = (
        f"Task type: {record['task_type']}\n"
        f"Input:\n{record['text']}\n\n"
        f"Choices: {choice_text(record['choices'])}\n\n"
        f"Previous debate transcript:\n{render_transcript(transcript)}\n\n"
        f"{instruction}"
    )
    return [{"role": "user", "content": content}]


def maybe_save_hidden_state(
    hidden_state: torch.Tensor,
    *,
    output_dir: Path,
    sample_id: str,
    round_index: int,
    agent_key: str,
    save_hidden_states: bool,
) -> str | None:
    if not save_hidden_states:
        return None
    path = output_dir / "hidden_states" / f"{safe_file_stem(sample_id)}_round{round_index}_{agent_key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hidden_state.cpu(), path)
    return str(path)


def run_sample(
    record: dict[str, Any],
    *,
    agent_a: DebateAgent,
    agent_b: DebateAgent,
    rounds: int,
    max_new_tokens: int,
    temperature: float,
    output_dir: Path,
    save_hidden_states: bool = False,
) -> dict[str, Any]:
    transcript: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    agent_a_predictions: list[str] = []
    agent_b_predictions: list[str] = []

    for round_index in range(rounds):
        messages_a = build_messages(record, agent_id="Agent A", round_index=round_index, transcript=transcript)
        result_a = agent_a.generate(
            messages_a,
            choices=record["choices"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        hidden_a_path = maybe_save_hidden_state(
            result_a.pooled_hidden_state,
            output_dir=output_dir,
            sample_id=record["id"],
            round_index=round_index,
            agent_key="agent_a",
            save_hidden_states=save_hidden_states,
        )
        turn_a = {
            "round": round_index,
            "agent_id": "Agent A",
            "prediction": result_a.prediction,
            "response_text": result_a.response_text,
        }
        transcript.append(turn_a)

        messages_b = build_messages(record, agent_id="Agent B", round_index=round_index, transcript=transcript)
        result_b = agent_b.generate(
            messages_b,
            choices=record["choices"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        hidden_b_path = maybe_save_hidden_state(
            result_b.pooled_hidden_state,
            output_dir=output_dir,
            sample_id=record["id"],
            round_index=round_index,
            agent_key="agent_b",
            save_hidden_states=save_hidden_states,
        )
        turn_b = {
            "round": round_index,
            "agent_id": "Agent B",
            "prediction": result_b.prediction,
            "response_text": result_b.response_text,
        }
        transcript.append(turn_b)

        disagreement = pairwise_disagreement(result_a.pooled_hidden_state, result_b.pooled_hidden_state)
        agent_a_predictions.append(result_a.prediction)
        agent_b_predictions.append(result_b.prediction)
        round_summaries.append(
            {
                "round": round_index,
                "disagreement": disagreement,
                "agent_a_prediction": result_a.prediction,
                "agent_b_prediction": result_b.prediction,
                "agent_a_response": result_a.response_text,
                "agent_b_response": result_b.response_text,
                "agent_a_generated_token_count": result_a.generated_token_count,
                "agent_b_generated_token_count": result_b.generated_token_count,
                "agent_a_sequence_length": result_a.sequence_length,
                "agent_b_sequence_length": result_b.sequence_length,
                "agent_a_pooled_vector_shape": result_a.pooled_vector_shape,
                "agent_b_pooled_vector_shape": result_b.pooled_vector_shape,
                "agent_a_hidden_state_path": hidden_a_path,
                "agent_b_hidden_state_path": hidden_b_path,
            }
        )

    final_a = agent_a_predictions[-1]
    final_b = agent_b_predictions[-1]
    consensus = final_a == final_b and final_a != "unknown"
    verdict = final_a if consensus else "unresolved"
    label = normalize_label(record["label"])
    is_correct = None if verdict == "unresolved" else verdict == label
    error = None if is_correct is None else not is_correct
    false_consensus = bool(consensus and verdict != label)
    initial_disagreement = round_summaries[0]["disagreement"]
    final_disagreement = round_summaries[-1]["disagreement"]

    summary: dict[str, Any] = {
        "id": record["id"],
        "split": record["split"],
        "task_type": record["task_type"],
        "answer_type": record["answer_type"],
        "label": label,
        "verdict": verdict,
        "is_correct": is_correct,
        "error": error,
        "consensus": consensus,
        "false_consensus": false_consensus,
        "agent_a_predictions": agent_a_predictions,
        "agent_b_predictions": agent_b_predictions,
        "final_agent_a_prediction": final_a,
        "final_agent_b_prediction": final_b,
        "initial_disagreement": initial_disagreement,
        "final_disagreement": final_disagreement,
        "convergence_delta": convergence_delta(initial_disagreement, final_disagreement),
        "rounds": round_summaries,
    }
    for round_summary in round_summaries:
        summary[f"round{round_summary['round']}_disagreement"] = round_summary["disagreement"]
    return summary


def run_pipeline(
    records: Iterable[dict[str, Any]],
    *,
    agent_a: DebateAgent,
    agent_b: DebateAgent,
    rounds: int,
    max_new_tokens: int,
    temperature: float,
    output_dir: Path,
    save_hidden_states: bool = False,
) -> list[dict[str, Any]]:
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    summaries = []
    for record in records:
        summaries.append(
            run_sample(
                record,
                agent_a=agent_a,
                agent_b=agent_b,
                rounds=rounds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                output_dir=output_dir,
                save_hidden_states=save_hidden_states,
            )
        )
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("benchmark_qwen/data/processed/all_benchmarks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_qwen/outputs/ford_2agent_qwen"))
    parser.add_argument("--output-name", default="summary.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--rounds", type=int, default=4, help="Round blocks, including round0 opening.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--agent-a-persona", default=DEFAULT_AGENT_A_PERSONA)
    parser.add_argument("--agent-b-persona", default=DEFAULT_AGENT_B_PERSONA)
    parser.add_argument("--save-hidden-states", action="store_true")
    parser.add_argument("--dtype", default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    records = list(read_jsonl(args.input))
    if args.max_samples is not None:
        records = records[: args.max_samples]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent_a = QwenAgent(args.model, persona=args.agent_a_persona, dtype=args.dtype)
    agent_b = QwenAgent(args.model, persona=args.agent_b_persona, dtype=args.dtype)
    summaries = run_pipeline(
        records,
        agent_a=agent_a,
        agent_b=agent_b,
        rounds=args.rounds,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
        save_hidden_states=args.save_hidden_states,
    )
    output_path = args.output_dir / args.output_name
    count = write_jsonl(summaries, output_path)
    print(f"[done] wrote {count} summaries -> {output_path}")


if __name__ == "__main__":
    main()
