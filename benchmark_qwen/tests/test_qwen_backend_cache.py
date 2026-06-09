#!/usr/bin/env python3
"""Checks that Qwen role agents share one loaded backend.

This test uses a fake transformers module, so it does not download a model and
does not require GPU.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_qwen.models import qwen_agent  # noqa: E402


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = None
        self.eos_token_id = 0


class DummyModel:
    def __init__(self) -> None:
        self.to_calls: list[str] = []
        self.eval_calls = 0

    def to(self, device: str):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self


class DummyAutoTokenizer:
    load_calls: list[tuple[str, bool]] = []

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *, trust_remote_code: bool):
        cls.load_calls.append((model_name_or_path, trust_remote_code))
        return DummyTokenizer()


class DummyAutoModelForCausalLM:
    load_calls: list[dict[str, object]] = []
    loaded_models: list[DummyModel] = []

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        torch_dtype,
        device_map,
        trust_remote_code: bool,
    ):
        model = DummyModel()
        cls.load_calls.append(
            {
                "model_name_or_path": model_name_or_path,
                "torch_dtype": torch_dtype,
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
            }
        )
        cls.loaded_models.append(model)
        return model


def install_fake_transformers() -> object | None:
    previous = sys.modules.get("transformers")
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=DummyAutoTokenizer,
        AutoModelForCausalLM=DummyAutoModelForCausalLM,
    )
    sys.modules["transformers"] = fake_transformers
    return previous


def restore_transformers(previous: object | None) -> None:
    if previous is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = previous


def reset_dummy_loaders() -> None:
    DummyAutoTokenizer.load_calls.clear()
    DummyAutoModelForCausalLM.load_calls.clear()
    DummyAutoModelForCausalLM.loaded_models.clear()
    qwen_agent.clear_qwen_backend_cache()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_agents_share_backend_for_same_model_device_dtype() -> None:
    previous_transformers = install_fake_transformers()
    try:
        reset_dummy_loaders()

        agent_a = qwen_agent.QwenAgent(
            "dummy/qwen",
            persona="You are Agent A.",
            device="cpu",
            dtype="auto",
        )
        agent_b = qwen_agent.QwenAgent(
            "dummy/qwen",
            persona="You are Agent B.",
            device="cpu",
            dtype="auto",
        )

        require(agent_a.backend is agent_b.backend, "agents should share the same backend object")
        require(agent_a.model is agent_b.model, "agents should share the same model object")
        require(agent_a.tokenizer is agent_b.tokenizer, "agents should share the same tokenizer object")
        require(agent_a.persona != agent_b.persona, "personas should remain per-agent state")
        require(len(DummyAutoTokenizer.load_calls) == 1, "tokenizer should load once")
        require(len(DummyAutoModelForCausalLM.load_calls) == 1, "model should load once")

        loaded_model = DummyAutoModelForCausalLM.loaded_models[0]
        require(loaded_model.to_calls == ["cpu"], "CPU backend should move the model once")
        require(loaded_model.eval_calls == 1, "backend should call eval once")
        require(agent_a.tokenizer.pad_token_id == 0, "pad token should be normalized on the shared tokenizer")
    finally:
        reset_dummy_loaders()
        restore_transformers(previous_transformers)


def test_cache_key_keeps_dtype_separate() -> None:
    previous_transformers = install_fake_transformers()
    try:
        reset_dummy_loaders()

        auto_agent = qwen_agent.QwenAgent("dummy/qwen", device="cpu", dtype="auto")
        fp16_agent = qwen_agent.QwenAgent("dummy/qwen", device="cpu", dtype="float16")

        require(auto_agent.backend is not fp16_agent.backend, "different dtype settings need separate backends")
        require(len(DummyAutoTokenizer.load_calls) == 2, "different dtype should trigger a second tokenizer load")
        require(len(DummyAutoModelForCausalLM.load_calls) == 2, "different dtype should trigger a second model load")
    finally:
        reset_dummy_loaders()
        restore_transformers(previous_transformers)


def main() -> None:
    test_agents_share_backend_for_same_model_device_dtype()
    test_cache_key_keeps_dtype_separate()
    print("[qwen backend cache tests passed]")


if __name__ == "__main__":
    main()
