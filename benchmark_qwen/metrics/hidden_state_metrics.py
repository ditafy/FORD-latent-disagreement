#!/usr/bin/env python3
"""Hidden-state disagreement metrics for FORD/Qwen benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairwiseDisagreement:
    left: str
    right: str
    disagreement: float


def _to_vector(vector: torch.Tensor, name: str = "vector") -> torch.Tensor:
    if not isinstance(vector, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if vector.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if torch.isnan(vector).any():
        raise ValueError(f"{name} contains NaN")
    if torch.isinf(vector).any():
        raise ValueError(f"{name} contains Inf")

    if vector.ndim == 1:
        flattened = vector
    elif vector.ndim == 2 and vector.shape[0] == 1:
        flattened = vector[0]
    else:
        raise ValueError(f"{name} must have shape [hidden_dim] or [1, hidden_dim], got {tuple(vector.shape)}")

    if torch.linalg.vector_norm(flattened.float()).item() == 0.0:
        raise ValueError(f"{name} must not be the zero vector")
    return flattened.detach().float().cpu()


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return cosine similarity for two hidden-state vectors."""
    left_vec = _to_vector(left, "left")
    right_vec = _to_vector(right, "right")
    if left_vec.shape != right_vec.shape:
        raise ValueError(f"vector shapes must match, got {tuple(left_vec.shape)} and {tuple(right_vec.shape)}")
    return float(F.cosine_similarity(left_vec.unsqueeze(0), right_vec.unsqueeze(0), dim=1).item())


def pairwise_disagreement(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return 1 - cosine similarity for a pair of agent vectors."""
    return 1.0 - cosine_similarity(left, right)


def mean_pairwise_disagreement(vectors: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]]) -> float:
    """Return mean pairwise disagreement across two or more named vectors."""
    items = list(vectors.items() if isinstance(vectors, Mapping) else vectors)
    if len(items) < 2:
        raise ValueError("At least two vectors are required")

    disagreements = [
        pairwise_disagreement(left_vec, right_vec)
        for (_, left_vec), (_, right_vec) in combinations(items, 2)
    ]
    return float(sum(disagreements) / len(disagreements))


def pairwise_disagreement_details(
    vectors: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
) -> list[PairwiseDisagreement]:
    """Return all pairwise disagreements with agent labels."""
    items = list(vectors.items() if isinstance(vectors, Mapping) else vectors)
    if len(items) < 2:
        raise ValueError("At least two vectors are required")

    return [
        PairwiseDisagreement(
            left=left_name,
            right=right_name,
            disagreement=pairwise_disagreement(left_vec, right_vec),
        )
        for (left_name, left_vec), (right_name, right_vec) in combinations(items, 2)
    ]


def convergence_delta(initial_disagreement: float, final_disagreement: float) -> float:
    """Positive value means disagreement decreased from initial to final."""
    return float(initial_disagreement - final_disagreement)
