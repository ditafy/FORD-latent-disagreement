#!/usr/bin/env python3
"""Qwen agent wrapper with generated-token hidden-state pooling."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"


@dataclass
class AgentGeneration:
    response_text: str
    prediction: str
    pooled_hidden_state: torch.Tensor
    generated_token_count: int
    sequence_length: int

    @property
    def pooled_vector_shape(self) -> list[int]:
        return list(self.pooled_hidden_state.shape)


def _as_step_hidden_states(hidden_states: Any) -> list[Any]:
    """Normalize HF generation hidden_states into a per-token-step list."""
    if hidden_states is None:
        return []
    if isinstance(hidden_states, tuple):
        return list(hidden_states)
    if isinstance(hidden_states, list):
        return hidden_states
    raise TypeError(f"Unsupported hidden_states type: {type(hidden_states)!r}")


def pool_generated_final_layer_hidden_states(
    hidden_states: Any,
    generated_token_count: int | None = None,
) -> torch.Tensor:
    """Mean-pool final-layer hidden states for generated tokens.

    HuggingFace generate(..., output_hidden_states=True) returns hidden states
    per decoding step. Each step contains layer-wise tensors. This function
    mirrors the ED2D control variable:

        final_layer = step_hidden[-1]
        final_token_state = final_layer[:, -1, :]
        pooled = mean_t(final_token_state_t)
    """
    step_hidden_states = _as_step_hidden_states(hidden_states)
    if generated_token_count is not None:
        if generated_token_count < 1:
            raise ValueError("generated_token_count must be positive")
        step_hidden_states = step_hidden_states[-generated_token_count:]

    final_token_states: list[torch.Tensor] = []
    for step_index, step_hidden in enumerate(step_hidden_states):
        if not step_hidden:
            raise ValueError(f"hidden_states step {step_index} has no layers")
        final_layer = step_hidden[-1]
        if final_layer.ndim != 3:
            raise ValueError(
                f"final layer at step {step_index} should have shape [batch, seq, hidden], "
                f"got {tuple(final_layer.shape)}"
            )
        final_token_states.append(final_layer[:, -1, :].detach().float().cpu())

    if not final_token_states:
        raise ValueError("No generated-token hidden states were found")

    stacked_states = torch.stack(final_token_states, dim=1)
    return stacked_states.mean(dim=1)


def render_messages(tokenizer: Any, messages: Sequence[dict[str, str]], device: str) -> dict[str, torch.Tensor]:
    """Render chat messages with chat template, falling back to plain text."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return {key: value.to(device) for key, value in rendered.items()}

    prompt = "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)
    prompt += "\n\nASSISTANT:\n"
    rendered = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in rendered.items()}


def extract_prediction(response_text: str, choices: Sequence[str]) -> str:
    """Extract a normalized answer label from a model response."""
    normalized_choices = [choice.lower() for choice in choices]
    choice_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter_to_choice = {
        choice_letters[index]: choice
        for index, choice in enumerate(normalized_choices)
        if index < len(choice_letters)
    }

    answer_match = re.search(r"answer\s*:\s*\(?([A-Z])\)?", response_text, flags=re.IGNORECASE)
    if answer_match:
        letter = answer_match.group(1).upper()
        if letter in letter_to_choice:
            return letter_to_choice[letter]

    lowered = response_text.lower()
    for choice in normalized_choices:
        if re.search(rf"\b{re.escape(choice)}\b", lowered):
            return choice

    return "unknown"


class QwenAgent:
    """Single local Qwen persona used as a FORD debate agent."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        *,
        persona: str = "",
        device: str | None = None,
        dtype: str = "auto",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.persona = persona
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype: str | torch.dtype
        if dtype == "auto":
            torch_dtype = "auto"
        else:
            torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        choices: Sequence[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AgentGeneration:
        if self.persona:
            messages = [{"role": "system", "content": self.persona}, *messages]

        encoded = render_messages(self.tokenizer, messages, self.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": min(max_new_tokens, 1024),
            "do_sample": temperature > 0,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)

        sequences = outputs.sequences
        generated_token_ids = sequences[:, input_ids.shape[-1] :]
        generated_token_count = int(generated_token_ids.shape[-1])
        pooled = pool_generated_final_layer_hidden_states(
            outputs.hidden_states,
            generated_token_count=generated_token_count,
        )
        response_text = self.tokenizer.decode(generated_token_ids[0], skip_special_tokens=True).strip()

        return AgentGeneration(
            response_text=response_text,
            prediction=extract_prediction(response_text, choices),
            pooled_hidden_state=pooled,
            generated_token_count=generated_token_count,
            sequence_length=int(sequences.shape[-1]),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen hidden-state extraction.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt", default="Answer: (A) yes or (B) no. Is the sky blue?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    agent = QwenAgent(args.model)
    result = agent.generate(
        [{"role": "user", "content": args.prompt}],
        choices=["yes", "no"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"prediction={result.prediction}")
    print(f"generated_token_count={result.generated_token_count}")
    print(f"sequence_length={result.sequence_length}")
    print(f"pooled_vector_shape={result.pooled_vector_shape}")
    print(result.response_text)


if __name__ == "__main__":
    main()
