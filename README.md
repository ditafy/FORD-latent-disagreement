# FORD Hidden-State Debate Benchmark

This repository builds on the FORD debate framework and uses it as an interaction pipeline for studying process-level signals in multi-agent debate. The project focuses on whether hidden-state disagreement between agents is related to prediction error, and whether disagreement changes across debate rounds.

The current benchmark uses the FORD two-agent and three-agent debate forms as the code base. The target analysis compares debate behavior across misinformation detection, commonsense reasoning, and biomedical question answering.

## Research Goal

The project studies multi-agent debate reliability through round-level process signals rather than final accuracy alone.

The main questions are:

- Does hidden-state disagreement between agents correlate with final prediction error?
- Does disagreement decrease across debate rounds, suggesting convergence?
- Does convergence always indicate reliability, or can it produce false consensus?
- Do these patterns differ across FakeNews, StrategyQA, and PubMedQA?

The model setting uses the same local HuggingFace causal language model for all agents, such as Qwen2.5-14B-Instruct. Agents are differentiated by role prompts or personas, while hidden states remain in the same representation space.

## Benchmark Tasks

The repository keeps the datasets needed for the current benchmark:

- `fakeNewsDataset/`: fake vs legitimate news classification.
- `data/jsonlines/strategyqa_processed.jsonl`: yes/no commonsense reasoning.
- `data/jsonlines/pubmedqa_test_processed.jsonl`: biomedical question answering with yes/no/maybe labels.

These tasks represent three different forms of factual judgment:

- misinformation detection,
- commonsense reasoning,
- biomedical evidence-based QA.

## Code Structure

```text
.
├── README.md
├── tools.py
├── debate_chatgpt.py
├── debate_davinci.py
├── debate_llama_vicuna.py
├── debate_table.py
├── zero_shot_chatgpt.py
├── few_shot_cot.py
├── few_shot_vicuna.py
├── llama_official.py
├── data/
│   └── jsonlines/
│       ├── strategyqa_processed.jsonl
│       └── pubmedqa_test_processed.jsonl
└── fakeNewsDataset/
    ├── fake/
    └── legit/
```

### Current FORD Debate Files

- `debate_chatgpt.py`: original two-agent FORD debate logic for ChatGPT-style models.
- `debate_davinci.py`: original two-agent FORD debate logic for completion-style models.
- `debate_llama_vicuna.py`: original two-agent local-model debate reference.
- `debate_table.py`: original three-agent roundtable debate logic.
- `tools.py`: shared jsonlines loading, logging, and dataset utilities.

### Reference Generation Files

- `zero_shot_chatgpt.py`: original zero-shot generation script.
- `few_shot_cot.py`: original few-shot CoT generation script for completion models.
- `few_shot_vicuna.py`: original Vicuna few-shot generation reference.
- `llama_official.py`: original LLaMA few-shot generation reference.

These files are kept as implementation references while the benchmark code is adapted toward local HuggingFace models and hidden-state extraction.

## Experimental Design

The benchmark keeps FORD's iterative debate idea and extends it with hidden-state tracing.

For each sample:

1. Two or more agents produce initial answers and explanations.
2. Agents debate over multiple rounds using the previous debate history.
3. Each generated response is associated with a final-layer hidden-state representation.
4. Round-level disagreement is computed from the agent hidden states.
5. Final predictions are compared with gold labels.
6. Disagreement is analyzed against error and convergence behavior.

For the two-agent version, round-level disagreement is:

```text
D_round = 1 - cosine_similarity(h_agent_1_round, h_agent_2_round)
```

For the three-agent roundtable version, disagreement is computed as the mean pairwise distance:

```text
D_round = mean_{i < j}(1 - cosine_similarity(h_i_round, h_j_round))
```

## Hidden-State Extraction Knowledge

The hidden-state extraction follows the control setting from the previous ED2D-style experiments:

```text
h_agent_round = mean_t final_layer_hidden_state(generated_token_t)
```

The controlled extraction procedure is:

- use `AutoModelForCausalLM.generate(...)`;
- set `return_dict_in_generate=True`;
- set `output_hidden_states=True`;
- take only the final layer hidden state;
- for each decoding step, take the last token position;
- keep only generated-token hidden states;
- mean-pool across generated tokens;
- compute disagreement with `1 - cosine_similarity`.

This design avoids comparing prompt hidden states only. It represents each agent's actual generated debate response.

## Notes

- OpenAI API models do not expose hidden states, so hidden-state experiments use local HuggingFace causal language models.
- Using the same base model for all agents keeps hidden states in a shared representation space.
- Different agents are created through different system prompts, personas, or debate roles.


## Code Source

The debate pipeline is adapted from the FORD framework introduced in:

[Examining Inter-Consistency of Large Language Models Collaboration: An In-depth Analysis via Debate](https://aclanthology.org/2023.findings-emnlp.508/)

The original FORD code structure is retained where useful for two-agent debate and three-agent roundtable debate. This repository adapts that framework for hidden-state disagreement analysis across FakeNews, StrategyQA, and PubMedQA.

## Citation

```bibtex
@inproceedings{xiong2023examining,
  title={Examining Inter-Consistency of Large Language Models Collaboration: An In-depth Analysis via Debate},
  author={Xiong, Kai and Ding, Xiao and Cao, Yixin and Liu, Ting and Qin, Bing},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={7572--7590},
  year={2023}
}
```
