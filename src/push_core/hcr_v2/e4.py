"""HCR V2 E4 的 GPU-first Top-K action-selection 核心计算。"""

from __future__ import annotations

import torch

from push_core.hcr_v2.e3 import (
    YAW_TOLERANCE_RAD,
    interpolate_outcome_grid,
    tnpo_costs,
    wrap_to_pi,
)


PRIMARY_K = 100
METHOD_IDS = tuple(f"Method{index}" for index in range(1, 7))


def posterior_mean_hidden_parameters(
    nodes: torch.Tensor,
    posterior_weights: torch.Tensor,
) -> torch.Tensor:
    """计算每个 case 的 posterior-mean hidden condition。"""

    if posterior_weights.ndim != 2:
        raise ValueError("posterior_weights 必须具有 (B, N) 形状")
    if nodes.ndim != 2 or nodes.shape[0] != posterior_weights.shape[1]:
        raise ValueError("nodes 与 posterior_weights 的 node dimension 不一致")
    return torch.einsum("bn,nd->bd", posterior_weights.float(), nodes.float())


def select_minimum_actions(
    candidate_indices: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按最低 score 选择 action，平局时使用较小 manifest index。"""

    if candidate_indices.shape != scores.shape:
        raise ValueError("candidate_indices 与 scores shape 必须一致")
    minimum = scores.amin(dim=1, keepdim=True)
    tied_actions = torch.where(
        scores == minimum,
        candidate_indices,
        torch.full_like(candidate_indices, torch.iinfo(candidate_indices.dtype).max),
    )
    selected_actions = tied_actions.amin(dim=1)
    selected_mask = candidate_indices == selected_actions[:, None]
    selected_scores = torch.where(
        selected_mask,
        scores,
        torch.full_like(scores, torch.inf),
    ).amin(dim=1)
    return selected_actions, selected_scores


@torch.inference_mode()
def point_condition_candidate_scores(
    outcome_grid: torch.Tensor,
    hidden_parameters: torch.Tensor,
    candidate_indices: torch.Tensor,
    task_queries: torch.Tensor,
    case_chunk_size: int = 64,
) -> torch.Tensor:
    """在每个 case 的单一 hidden-condition point 上计算 Top-K TNPO costs。"""

    case_count, candidate_count = candidate_indices.shape
    if hidden_parameters.shape[0] != case_count or task_queries.shape[0] != case_count:
        raise ValueError("point scoring 的 case dimensions 不一致")
    outputs = torch.empty(
        (case_count, candidate_count),
        dtype=torch.float32,
        device=outcome_grid.device,
    )
    for start in range(0, case_count, case_chunk_size):
        stop = min(case_count, start + case_chunk_size)
        point_outcomes = interpolate_outcome_grid(
            outcome_grid,
            hidden_parameters[start:stop],
        )
        gather_indices = candidate_indices[start:stop, :, None].expand(-1, -1, 3)
        candidate_outcomes = torch.gather(point_outcomes, 1, gather_indices)
        outputs[start:stop] = tnpo_costs(
            candidate_outcomes,
            task_queries[start:stop],
        )
    return outputs


@torch.inference_mode()
def posterior_expected_candidate_scores(
    node_outcomes: torch.Tensor,
    posterior_weights: torch.Tensor,
    candidate_indices: torch.Tensor,
    task_queries: torch.Tensor,
    case_chunk_size: int = 64,
) -> torch.Tensor:
    """在完整 fixed-node posterior 上计算 Top-K expected TNPO costs。"""

    case_count, candidate_count = candidate_indices.shape
    node_count = node_outcomes.shape[0]
    if node_outcomes.ndim != 3 or node_outcomes.shape[-1] != 3:
        raise ValueError("node_outcomes 必须具有 (N, A, 3) 形状")
    if posterior_weights.shape != (case_count, node_count):
        raise ValueError("posterior_weights 与 candidate/node dimensions 不一致")
    if task_queries.shape[0] != case_count:
        raise ValueError("task_queries 的 case dimension 不一致")

    outputs = torch.empty(
        (case_count, candidate_count),
        dtype=torch.float32,
        device=node_outcomes.device,
    )
    for start in range(0, case_count, case_chunk_size):
        stop = min(case_count, start + case_chunk_size)
        indices = candidate_indices[start:stop]
        candidate_outcomes = node_outcomes[:, indices].permute(1, 0, 2, 3)
        tasks = task_queries[start:stop]
        position_error = torch.linalg.vector_norm(
            candidate_outcomes[..., :2] - tasks[:, None, None, :2],
            dim=-1,
        )
        desired_yaw = torch.atan2(tasks[:, 2], tasks[:, 3])
        yaw_error = torch.abs(
            wrap_to_pi(candidate_outcomes[..., 2] - desired_yaw[:, None, None])
        )
        costs = 0.5 * position_error / 0.010 + 0.5 * yaw_error / YAW_TOLERANCE_RAD
        outputs[start:stop] = torch.einsum(
            "bn,bnk->bk",
            posterior_weights[start:stop].float(),
            costs.float(),
        )
    return outputs


def deterministic_ranks(
    scores: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    """按 score 排序，并使用较小 action index 处理平局。"""

    action_order = torch.argsort(candidate_indices, dim=1, stable=True)
    action_sorted_scores = torch.gather(scores, 1, action_order)
    score_order = torch.argsort(action_sorted_scores, dim=1, stable=True)
    ordered_slots = torch.gather(action_order, 1, score_order)
    ranks = torch.empty_like(ordered_slots)
    rank_values = torch.arange(
        scores.shape[1], dtype=ordered_slots.dtype, device=scores.device
    )[None, :].expand_as(ordered_slots)
    ranks.scatter_(1, ordered_slots, rank_values)
    return ranks


@torch.inference_mode()
def ranking_diagnostics(
    predicted_scores: torch.Tensor,
    true_costs: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算确定性 tie-break 下的 Spearman 与 pairwise ordering accuracy。"""

    predicted_ranks = deterministic_ranks(predicted_scores, candidate_indices).float()
    true_ranks = deterministic_ranks(true_costs, candidate_indices).float()
    candidate_count = predicted_scores.shape[1]
    squared_difference = torch.square(predicted_ranks - true_ranks).sum(dim=1)
    denominator = float(candidate_count * (candidate_count**2 - 1))
    spearman = 1.0 - 6.0 * squared_difference / denominator

    upper = torch.triu_indices(
        candidate_count,
        candidate_count,
        offset=1,
        device=predicted_scores.device,
    )
    predicted_order = (
        predicted_ranks[:, upper[0]] < predicted_ranks[:, upper[1]]
    )
    true_order = true_ranks[:, upper[0]] < true_ranks[:, upper[1]]
    pairwise_accuracy = (predicted_order == true_order).float().mean(dim=1)
    return spearman, pairwise_accuracy

