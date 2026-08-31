"""HCR V2 E5 的信念条件闭环动作决策核心。"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from push_core.hcr_v2.e3 import (
    TaskNormaliser,
    TwoStageProposal,
    belief_marginalised_probabilities,
    topk_probabilities,
)
from push_core.hcr_v2.e4 import (
    point_condition_candidate_scores,
    posterior_expected_candidate_scores,
    posterior_mean_hidden_parameters,
    select_minimum_actions,
)


PROTOCOL_VERSION = "hcr_v2_e5_v1"
PRIMARY_K = 100
BELIEF_UPDATE_HORIZON = 4
STUDENT_T_DEGREES_OF_FREEDOM = 3.0
OBSERVATION_SCALE = np.asarray(
    [0.010, 0.010, math.radians(5.0)], dtype=np.float32
)

NOMINAL_CONTROLLER_ID = "nominal_condition_state_feedback"
CERTAINTY_EQUIVALENT_CONTROLLER_ID = (
    "certainty_equivalent_belief_conditioned"
)
BELIEF_MARGINALISED_CONTROLLER_ID = "belief_marginalised_closed_loop"
FULL_INFORMATION_CONTROLLER_ID = (
    "full_information_tensor_interpolation_state_feedback"
)

CONTROLLER_NAMES = {
    NOMINAL_CONTROLLER_ID: "Nominal-Condition State-Feedback Controller",
    CERTAINTY_EQUIVALENT_CONTROLLER_ID: (
        "Certainty-Equivalent Belief-Conditioned Controller"
    ),
    BELIEF_MARGINALISED_CONTROLLER_ID: (
        "Belief-Marginalised Closed-Loop Controller"
    ),
    FULL_INFORMATION_CONTROLLER_ID: (
        "Full-Information Tensor-Interpolation State-Feedback Diagnostic"
    ),
}
CONTROLLER_IDS = tuple(CONTROLLER_NAMES)


def wrap_to_pi(value: float) -> float:
    """把单个角度归一化到 [-pi, pi)。"""

    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def make_task_query(
    current_position_xy: np.ndarray,
    current_yaw_rad: float,
    target_position_xy: np.ndarray,
    initial_yaw_rad: float,
) -> np.ndarray:
    """构造 E3 使用的 current-box local 四维 task query。"""

    current = np.asarray(current_position_xy, dtype=np.float64)
    target = np.asarray(target_position_xy, dtype=np.float64)
    world_delta = target - current
    cosine = math.cos(current_yaw_rad)
    sine = math.sin(current_yaw_rad)
    local_x = cosine * world_delta[0] + sine * world_delta[1]
    local_y = -sine * world_delta[0] + cosine * world_delta[1]
    yaw_correction = wrap_to_pi(initial_yaw_rad - current_yaw_rad)
    return np.asarray(
        [local_x, local_y, math.sin(yaw_correction), math.cos(yaw_correction)],
        dtype=np.float32,
    )


@dataclass
class BeliefState:
    """单条 controller trajectory 的 fixed-node posterior state。"""

    log_weights: torch.Tensor
    update_count: int = 0

    @property
    def weights(self) -> torch.Tensor:
        """返回归一化 posterior weights。"""

        return torch.softmax(self.log_weights, dim=0)


@dataclass(frozen=True)
class ActionDecision:
    """一次 E5 controller action decision。"""

    action_index: int
    decision_score: float
    proposal_probability: float | None
    candidate_count: int
    candidate_source: str
    proposal_latency_s: float
    selection_latency_s: float

    @property
    def total_latency_s(self) -> float:
        """返回 proposal 与 selection 的总耗时。"""

        return self.proposal_latency_s + self.selection_latency_s


class ClosedLoopDecisionEngine:
    """复用 E1–E4 artifacts 的单场景 GPU closed-loop decision engine。"""

    def __init__(
        self,
        scenario: str,
        action_ids: list[str],
        outcome_grid: torch.Tensor,
        nodes: torch.Tensor,
        prior_weights: torch.Tensor,
        node_outcomes: torch.Tensor,
        proposal_model: TwoStageProposal,
        task_normaliser: TaskNormaliser,
        residual_bias_standardised: torch.Tensor,
        precision: torch.Tensor,
        device: torch.device,
        node_query_chunk_size: int = 65_536,
    ):
        self.scenario = scenario
        self.action_ids = tuple(action_ids)
        self.outcome_grid = outcome_grid
        self.nodes = nodes.float()
        self.prior_weights = prior_weights.float()
        self.node_outcomes = node_outcomes.float()
        self.proposal_model = proposal_model
        self.task_normaliser = task_normaliser
        self.residual_bias_standardised = residual_bias_standardised.float()
        self.precision = precision.float()
        self.device = device
        self.node_query_chunk_size = int(node_query_chunk_size)
        self.observation_scale_tensor = torch.tensor(
            OBSERVATION_SCALE, dtype=torch.float32, device=device
        )
        self._log_prior_weights = torch.log(self.prior_weights)

    def new_belief(self) -> BeliefState:
        """为一个新 episode 建立 bounded-uniform prior。"""

        return BeliefState(self._log_prior_weights.clone())

    def _synchronise(self) -> None:
        """在计时边界同步当前 CUDA stream。"""

        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _autocast(self):
        """只为 proposal MLP 启用 BF16 Tensor Core。"""

        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def belief_summary(self, belief: BeliefState) -> dict[str, object]:
        """返回 active normalised mean、covariance trace 与 update count。"""

        weights = belief.weights
        mean = torch.einsum("n,nd->d", weights, self.nodes)
        centered = self.nodes - mean[None, :]
        covariance = torch.einsum("n,ni,nj->ij", weights, centered, centered)
        return {
            "mean_normalised": mean.detach().cpu().numpy(),
            "covariance_trace_normalised": float(torch.trace(covariance)),
            "update_count": belief.update_count,
        }

    @torch.inference_mode()
    def update_belief(
        self,
        belief: BeliefState,
        action_index: int,
        observation: np.ndarray,
    ) -> bool:
        """使用一次有效 observation 更新 posterior，最多更新四次。"""

        if belief.update_count >= BELIEF_UPDATE_HORIZON:
            return False
        observation_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32) / OBSERVATION_SCALE,
            dtype=torch.float32,
            device=self.device,
        )
        means = (
            self.node_outcomes[:, action_index, :]
            / self.observation_scale_tensor[None, :]
            + self.residual_bias_standardised[None, :]
        )
        residual = observation_tensor[None, :] - means
        mahalanobis = torch.einsum(
            "ni,ij,nj->n", residual, self.precision, residual
        )
        exponent = -0.5 * (STUDENT_T_DEGREES_OF_FREEDOM + 3.0)
        belief.log_weights = belief.log_weights + exponent * torch.log1p(
            mahalanobis / STUDENT_T_DEGREES_OF_FREEDOM
        )
        belief.log_weights = torch.log_softmax(belief.log_weights, dim=0)
        belief.update_count += 1
        return True

    @torch.inference_mode()
    def select_action(
        self,
        controller_id: str,
        task_query: np.ndarray,
        belief: BeliefState,
        true_hidden_parameters: np.ndarray,
    ) -> ActionDecision:
        """按正式 E5 controller 定义选择下一次 atomic push。"""

        if controller_id not in CONTROLLER_NAMES:
            raise ValueError(f"未知 E5 controller: {controller_id}")
        raw_task = torch.as_tensor(
            np.asarray(task_query, dtype=np.float32)[None, :],
            device=self.device,
        )
        normalised_task = self.task_normaliser.transform(raw_task)
        posterior_weights = belief.weights[None, :]
        true_hidden = torch.as_tensor(
            np.asarray(true_hidden_parameters, dtype=np.float32)[None, :],
            device=self.device,
        )

        self._synchronise()
        proposal_start = time.perf_counter()
        proposal_probability: float | None = None
        if controller_id == NOMINAL_CONTROLLER_ID:
            hidden = torch.zeros_like(true_hidden)
            with self._autocast():
                probabilities = self.proposal_model.point_action_probabilities(
                    normalised_task, hidden
                )
            candidates, candidate_probabilities = topk_probabilities(
                probabilities, PRIMARY_K
            )
            candidate_source = "nominal_condition_top100"
        elif controller_id == CERTAINTY_EQUIVALENT_CONTROLLER_ID:
            hidden = posterior_mean_hidden_parameters(
                self.nodes, posterior_weights
            )
            with self._autocast():
                probabilities = self.proposal_model.point_action_probabilities(
                    normalised_task, hidden
                )
            candidates, candidate_probabilities = topk_probabilities(
                probabilities, PRIMARY_K
            )
            candidate_source = "posterior_mean_top100"
        elif controller_id == BELIEF_MARGINALISED_CONTROLLER_ID:
            with self._autocast():
                probabilities = belief_marginalised_probabilities(
                    self.proposal_model,
                    normalised_task,
                    self.nodes,
                    posterior_weights,
                    node_query_chunk_size=self.node_query_chunk_size,
                )
            candidates, candidate_probabilities = topk_probabilities(
                probabilities, PRIMARY_K
            )
            candidate_source = "belief_marginalised_top100"
        else:
            candidates = torch.arange(
                len(self.action_ids), dtype=torch.long, device=self.device
            )[None, :]
            candidate_probabilities = None
            hidden = true_hidden
            candidate_source = "full_action_library"
        self._synchronise()
        proposal_latency = time.perf_counter() - proposal_start

        selection_start = time.perf_counter()
        if controller_id == BELIEF_MARGINALISED_CONTROLLER_ID:
            scores = posterior_expected_candidate_scores(
                self.node_outcomes,
                posterior_weights,
                candidates,
                raw_task,
                case_chunk_size=1,
            )
        else:
            scores = point_condition_candidate_scores(
                self.outcome_grid,
                hidden,
                candidates,
                raw_task,
                case_chunk_size=1,
            )
        selected, selected_scores = select_minimum_actions(candidates, scores)
        self._synchronise()
        selection_latency = time.perf_counter() - selection_start

        action_index = int(selected.item())
        if candidate_probabilities is not None:
            slot = torch.nonzero(
                candidates[0] == action_index, as_tuple=False
            )[0, 0]
            proposal_probability = float(candidate_probabilities[0, slot])
        return ActionDecision(
            action_index=action_index,
            decision_score=float(selected_scores.item()),
            proposal_probability=proposal_probability,
            candidate_count=int(candidates.shape[1]),
            candidate_source=candidate_source,
            proposal_latency_s=proposal_latency,
            selection_latency_s=selection_latency,
        )
