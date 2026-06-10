#!/usr/bin/env python3
"""Task-specific role specs for ED2D-style stance debate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DebateTaskSpec:
    task_type: str
    affirmative_label: str
    negative_label: str
    judge_labels: tuple[str, ...]
    affirmative_description: str
    negative_description: str


@dataclass(frozen=True)
class RoleConfig:
    name: str
    side: str
    target_label: str
    meta_prompt: str


TASK_SPECS: dict[str, DebateTaskSpec] = {
    "fakenews": DebateTaskSpec(
        task_type="fakenews",
        affirmative_label="legitimate",
        negative_label="fake",
        judge_labels=("legitimate", "fake"),
        affirmative_description="the news is legitimate/true",
        negative_description="the news is fake/false",
    ),
    "strategyqa": DebateTaskSpec(
        task_type="strategyqa",
        affirmative_label="yes",
        negative_label="no",
        judge_labels=("yes", "no"),
        affirmative_description="the answer is YES",
        negative_description="the answer is NO",
    ),
    "pubmedqa": DebateTaskSpec(
        task_type="pubmedqa",
        affirmative_label="yes",
        negative_label="no",
        judge_labels=("yes", "no", "maybe"),
        affirmative_description="the answer is YES",
        negative_description="the answer is NO",
    ),
}


def normalize_task_type(value: Any) -> str:
    return str(value).strip().lower()


def get_task_spec(task_type: Any) -> DebateTaskSpec:
    normalized = normalize_task_type(task_type)
    try:
        return TASK_SPECS[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"Unsupported task_type {task_type!r}; expected one of: {known}") from exc


def create_role_configs(record: dict[str, Any]) -> tuple[RoleConfig, RoleConfig]:
    spec = get_task_spec(record.get("task_type"))
    choices = {str(choice).strip().lower() for choice in record.get("choices", [])}
    expected_labels = {spec.affirmative_label, spec.negative_label}
    missing = sorted(expected_labels - choices)
    if missing:
        raise ValueError(
            f"Record {record.get('id', '<unknown>')!r} choices are missing stance labels: {missing}"
        )

    affirmative = RoleConfig(
        name="Affirmative",
        side="affirmative",
        target_label=spec.affirmative_label,
        meta_prompt=(
            "You are the affirmative-side debate agent. "
            f"You believe {spec.affirmative_description}. "
            f"Argue in favor of the label '{spec.affirmative_label}' using evidence from the input."
        ),
    )
    negative = RoleConfig(
        name="Negative",
        side="negative",
        target_label=spec.negative_label,
        meta_prompt=(
            "You are the negative-side debate agent. "
            f"You believe {spec.negative_description}. "
            f"Argue in favor of the label '{spec.negative_label}' using evidence from the input."
        ),
    )
    return affirmative, negative


def create_judge_role_config(record: dict[str, Any]) -> RoleConfig:
    spec = get_task_spec(record.get("task_type"))
    allowed_labels = ", ".join(spec.judge_labels)
    maybe_guidance = ""
    if "maybe" in spec.judge_labels:
        maybe_guidance = " Use MAYBE when the evidence is mixed, insufficient, or inconclusive."

    return RoleConfig(
        name="Judge",
        side="neutral",
        target_label="",
        meta_prompt=(
            "You are the neutral judge in a debate. The affirmative and negative agents "
            "were assigned opposing positions, so do not assume either side is correct. "
            "Decide the best final answer based on the input evidence and the full debate "
            f"transcript. Valid final labels are: {allowed_labels}.{maybe_guidance}"
        ),
    )
