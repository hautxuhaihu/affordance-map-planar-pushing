"""HCR V2 E3 的 GPU-first candidate-proposal 核心计算。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


SCENARIOS = ("friction", "com", "joint")
REGION_COUNT = 12
ACTIONS_PER_REGION = 378
ACTION_COUNT = REGION_COUNT * ACTIONS_PER_REGION
TASK_DIMENSION = 4
LABEL_TEMPERATURE = 0.10
POSITION_TOLERANCE_M = 0.010
YAW_TOLERANCE_RAD = math.radians(5.0)


def wrap_to_pi(values: torch.Tensor) -> torch.Tensor:
    """把角度归一化到 [-pi, pi)。"""

    return torch.remainder(values + math.pi, 2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class TaskNormaliser:
    """只标准化 current-local remaining translation。"""

    position_mean: torch.Tensor
    position_std: torch.Tensor

    def transform(self, task_queries: torch.Tensor) -> torch.Tensor:
        """将四维 task query 转换为 proposal model 输入。"""

        if task_queries.ndim != 2 or task_queries.shape[1] != TASK_DIMENSION:
            raise ValueError("task_queries 必须具有 (B, 4) 形状")
        position = (
            task_queries[:, :2] - self.position_mean.to(task_queries.device)
        ) / self.position_std.to(task_queries.device)
        return torch.cat([position, task_queries[:, 2:]], dim=1)


def make_task_queries(
    positions_xy: torch.Tensor,
    yaw_corrections_rad: torch.Tensor,
) -> torch.Tensor:
    """构造 position 加 sine/cosine yaw correction 的 task queries。"""

    if positions_xy.ndim != 2 or positions_xy.shape[1] != 2:
        raise ValueError("positions_xy 必须具有 (N, 2) 形状")
    if yaw_corrections_rad.ndim != 1:
        raise ValueError("yaw_corrections_rad 必须是一维数组")
    repeated_positions = positions_xy.repeat_interleave(
        yaw_corrections_rad.numel(), dim=0
    )
    repeated_yaw = yaw_corrections_rad.repeat(positions_xy.shape[0])
    return torch.cat(
        [
            repeated_positions,
            torch.sin(repeated_yaw)[:, None],
            torch.cos(repeated_yaw)[:, None],
        ],
        dim=1,
    )


def yaw_correction_from_query(task_queries: torch.Tensor) -> torch.Tensor:
    """从 sine/cosine encoding 恢复 desired yaw correction。"""

    return torch.atan2(task_queries[:, 2], task_queries[:, 3])


def tnpo_costs(
    outcomes: torch.Tensor,
    task_queries: torch.Tensor,
) -> torch.Tensor:
    """批量计算 query-outcome TNPO costs。

    outcomes 的尾部形状必须为 ``(..., A, 3)``，task_queries 为 ``(..., 4)``。
    """

    if outcomes.shape[-1] != 3:
        raise ValueError("outcomes 最后一维必须为 3")
    if task_queries.shape[-1] != TASK_DIMENSION:
        raise ValueError("task_queries 最后一维必须为 4")
    position_error = torch.linalg.vector_norm(
        outcomes[..., :2] - task_queries[..., None, :2],
        dim=-1,
    )
    desired_yaw = torch.atan2(
        task_queries[..., 2], task_queries[..., 3]
    )
    yaw_error = torch.abs(
        wrap_to_pi(outcomes[..., 2] - desired_yaw[..., None])
    )
    return (
        0.5 * position_error / POSITION_TOLERANCE_M
        + 0.5 * yaw_error / YAW_TOLERANCE_RAD
    )


def action_soft_labels(
    costs: torch.Tensor,
    temperature: float = LABEL_TEMPERATURE,
) -> torch.Tensor:
    """把 TNPO costs 转换为数值稳定的 soft labels。"""

    if temperature <= 0.0:
        raise ValueError("temperature 必须大于 0")
    shifted = costs.float() - costs.float().amin(dim=-1, keepdim=True)
    return torch.softmax(-shifted / temperature, dim=-1)


def region_soft_labels_from_costs(
    costs: torch.Tensor,
    temperature: float = LABEL_TEMPERATURE,
) -> torch.Tensor:
    """由 4,536-action costs 计算 12-class Region labels。"""

    if costs.shape[-1] != ACTION_COUNT:
        raise ValueError(f"costs 最后一维必须为 {ACTION_COUNT}")
    weights = action_soft_labels(costs, temperature)
    return weights.reshape(*weights.shape[:-1], REGION_COUNT, ACTIONS_PER_REGION).sum(
        dim=-1
    )


def soft_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算 soft-label cross-entropy。"""

    losses = -(labels.float() * torch.log_softmax(logits.float(), dim=-1)).sum(
        dim=-1
    )
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.float()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


class RegionProposalMLP(nn.Module):
    """输出 12 个 contact-region logits。"""

    def __init__(self, hidden_dimension: int, input_dimension: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, REGION_COUNT),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ActionProposalMLP(nn.Module):
    """输出给定 contact region 下的 378 个 action logits。"""

    def __init__(self, hidden_dimension: int, input_dimension: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension + REGION_COUNT, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, ACTIONS_PER_REGION),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        region_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat([inputs, region_one_hot], dim=-1))


class TwoStageProposal(nn.Module):
    """共享 task/condition inputs 的 two-stage categorical proposal。"""

    def __init__(self, hidden_parameter_dimension: int):
        super().__init__()
        input_dimension = TASK_DIMENSION + hidden_parameter_dimension
        self.hidden_parameter_dimension = hidden_parameter_dimension
        self.region_model = RegionProposalMLP(128, input_dimension)
        self.action_model = ActionProposalMLP(256, input_dimension)

    def proposal_inputs(
        self,
        normalised_tasks: torch.Tensor,
        hidden_parameters: torch.Tensor,
    ) -> torch.Tensor:
        """拼接 task features 与 active hidden coordinates。"""

        if hidden_parameters.shape[-1] != self.hidden_parameter_dimension:
            raise ValueError("hidden parameter dimension 与模型不一致")
        return torch.cat([normalised_tasks, hidden_parameters], dim=-1)

    def point_action_probabilities(
        self,
        normalised_tasks: torch.Tensor,
        hidden_parameters: torch.Tensor,
    ) -> torch.Tensor:
        """为 point-conditioned cases 生成完整 4,536-action probabilities。"""

        inputs = self.proposal_inputs(normalised_tasks, hidden_parameters)
        region_probabilities = torch.softmax(
            self.region_model(inputs).float(), dim=-1
        )
        rows: list[torch.Tensor] = []
        for region_index in range(REGION_COUNT):
            one_hot = torch.zeros(
                (inputs.shape[0], REGION_COUNT),
                dtype=inputs.dtype,
                device=inputs.device,
            )
            one_hot[:, region_index] = 1.0
            conditional = torch.softmax(
                self.action_model(inputs, one_hot).float(), dim=-1
            )
            rows.append(
                region_probabilities[:, region_index, None] * conditional
            )
        return torch.stack(rows, dim=1).reshape(inputs.shape[0], ACTION_COUNT)


@torch.inference_mode()
def belief_marginalised_probabilities(
    model: TwoStageProposal,
    normalised_tasks: torch.Tensor,
    nodes: torch.Tensor,
    posterior_weights: torch.Tensor,
    node_query_chunk_size: int = 32_768,
) -> torch.Tensor:
    """在完整 fixed-node posterior 上精确边缘化 proposal probabilities。"""

    case_count, node_count = posterior_weights.shape
    if nodes.shape != (node_count, model.hidden_parameter_dimension):
        raise ValueError("nodes shape 与 posterior/model 不一致")
    if normalised_tasks.shape != (case_count, TASK_DIMENSION):
        raise ValueError("normalised_tasks shape 与 cases 不一致")
    if node_query_chunk_size < node_count:
        case_batch_size = 1
    else:
        case_batch_size = max(1, node_query_chunk_size // node_count)

    outputs = torch.empty(
        (case_count, ACTION_COUNT),
        dtype=torch.float32,
        device=normalised_tasks.device,
    )
    for case_start in range(0, case_count, case_batch_size):
        case_stop = min(case_count, case_start + case_batch_size)
        tasks = normalised_tasks[case_start:case_stop]
        weights = posterior_weights[case_start:case_stop].float()
        batch_count = tasks.shape[0]
        accumulated = torch.zeros(
            (batch_count, REGION_COUNT, ACTIONS_PER_REGION),
            dtype=torch.float32,
            device=tasks.device,
        )
        nodes_per_chunk = max(1, node_query_chunk_size // batch_count)
        for node_start in range(0, node_count, nodes_per_chunk):
            node_stop = min(node_count, node_start + nodes_per_chunk)
            node_chunk = nodes[node_start:node_stop]
            chunk_count = node_chunk.shape[0]
            repeated_tasks = tasks[:, None, :].expand(
                batch_count, chunk_count, TASK_DIMENSION
            ).reshape(-1, TASK_DIMENSION)
            repeated_nodes = node_chunk[None, :, :].expand(
                batch_count, chunk_count, model.hidden_parameter_dimension
            ).reshape(-1, model.hidden_parameter_dimension)
            inputs = model.proposal_inputs(repeated_tasks, repeated_nodes)
            region_probabilities = torch.softmax(
                model.region_model(inputs).float(), dim=-1
            ).reshape(batch_count, chunk_count, REGION_COUNT)
            chunk_weights = weights[:, node_start:node_stop]
            for region_index in range(REGION_COUNT):
                one_hot = torch.zeros(
                    (inputs.shape[0], REGION_COUNT),
                    dtype=inputs.dtype,
                    device=inputs.device,
                )
                one_hot[:, region_index] = 1.0
                conditional = torch.softmax(
                    model.action_model(inputs, one_hot).float(), dim=-1
                ).reshape(batch_count, chunk_count, ACTIONS_PER_REGION)
                joint_weight = (
                    chunk_weights
                    * region_probabilities[:, :, region_index]
                )
                accumulated[:, region_index, :] += torch.einsum(
                    "bn,bnk->bk", joint_weight, conditional
                )
        flat = accumulated.reshape(batch_count, ACTION_COUNT)
        outputs[case_start:case_stop] = flat / flat.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
    return outputs


def interpolate_outcome_grid(
    outcome_grid: torch.Tensor,
    nodes: torch.Tensor,
) -> torch.Tensor:
    """在规则 5-point tensor grid 上为全部 nodes 插值 action outcomes。"""

    dimension = nodes.shape[1]
    if outcome_grid.ndim != dimension + 2:
        raise ValueError("outcome_grid dimension 与 nodes 不一致")
    if tuple(outcome_grid.shape[:dimension]) != (5,) * dimension:
        raise ValueError("outcome_grid 每个 hidden dimension 必须有 5 个 anchors")
    if outcome_grid.shape[-2:] != (ACTION_COUNT, 3):
        raise ValueError("outcome_grid action/outcome shape 错误")

    scaled = (nodes + 1.0) * 2.0
    lower = torch.floor(scaled).to(torch.long).clamp(0, 4)
    upper = (lower + 1).clamp(0, 4)
    upper_weight = (scaled - lower.float()).clamp(0.0, 1.0)
    upper_weight = torch.where(lower == upper, torch.zeros_like(upper_weight), upper_weight)
    result = torch.zeros(
        (nodes.shape[0], ACTION_COUNT, 3),
        dtype=torch.float32,
        device=outcome_grid.device,
    )
    for corner in range(1 << dimension):
        indices: list[torch.Tensor] = []
        weight = torch.ones(nodes.shape[0], dtype=torch.float32, device=nodes.device)
        for axis in range(dimension):
            use_upper = bool(corner & (1 << axis))
            indices.append(upper[:, axis] if use_upper else lower[:, axis])
            weight *= upper_weight[:, axis] if use_upper else 1.0 - upper_weight[:, axis]
        result += weight[:, None, None] * outcome_grid[tuple(indices)]
    return result


@torch.inference_mode()
def posterior_weights_from_histories(
    node_means_standardised: torch.Tensor,
    prior_weights: torch.Tensor,
    action_indices: torch.Tensor,
    observations_standardised: torch.Tensor,
    observation_mask: torch.Tensor,
    residual_bias_standardised: torch.Tensor,
    precision: torch.Tensor,
    degrees_of_freedom: float,
    case_batch_size: int = 256,
) -> torch.Tensor:
    """使用 E2 Student-t likelihood 在 GPU 上重放有限更新时域 posterior。"""

    case_count, update_count = action_indices.shape
    node_count = node_means_standardised.shape[1]
    outputs = torch.empty(
        (case_count, node_count),
        dtype=torch.float32,
        device=node_means_standardised.device,
    )
    log_prior = torch.log(prior_weights.float())
    exponent = -0.5 * (degrees_of_freedom + 3.0)
    for start in range(0, case_count, case_batch_size):
        stop = min(case_count, start + case_batch_size)
        selected_means = node_means_standardised[
            action_indices[start:stop]
        ]
        selected_means = selected_means + residual_bias_standardised[
            None, None, None, :
        ]
        residual = (
            observations_standardised[start:stop, :, None, :]
            - selected_means
        )
        mahalanobis = torch.einsum(
            "buni,ij,bunj->bun",
            residual,
            precision,
            residual,
        )
        log_likelihoods = exponent * torch.log1p(
            mahalanobis / degrees_of_freedom
        )
        log_likelihoods *= observation_mask[start:stop, :, None]
        log_posterior = log_prior[None, :] + log_likelihoods.sum(dim=1)
        outputs[start:stop] = torch.softmax(log_posterior, dim=1)
    return outputs


def topk_probabilities(
    probabilities: torch.Tensor,
    k: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 probability 降序提取 Top-K；精确平局优先较小 action index。"""

    if not 0 < k <= probabilities.shape[1]:
        raise ValueError("k 超出 action probability dimension")
    indices = torch.argsort(
        probabilities,
        dim=1,
        descending=True,
        stable=True,
    )[:, :k]
    values = torch.gather(probabilities, 1, indices)
    return indices, values


def model_parameter_count(modules: Iterable[nn.Module]) -> int:
    """返回多个模块的总可训练参数量。"""

    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
