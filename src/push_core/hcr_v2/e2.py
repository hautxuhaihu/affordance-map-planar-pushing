"""HCR V2 E2 的统一序贯贝叶斯估计方法。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from push_core.hcr_v2.e1 import (
    OUTCOME_FIELDS,
    SCENARIO_ACTIVE_COORDINATES,
    TensorOutcomeInterpolator,
)
from push_core.project_paths import HCR_V2_RESULTS_DIR


SCENARIOS = ("friction", "com", "joint")
ESTIMATOR_NAMES = ("prior_only", "single_observation", "sequential_bayesian")
QUADRATURE_POINT_COUNTS = (5, 9, 17)
OBSERVATION_SCALE = np.asarray(
    [0.010, 0.010, math.radians(5.0)],
    dtype=np.float64,
)
NUMERICAL_JITTER = 1e-6


def logsumexp(values: np.ndarray) -> float:
    """稳定计算一维数组的 log-sum-exp。"""

    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(array))
    if not math.isfinite(maximum):
        raise ValueError("log-sum-exp 输入没有有限最大值")
    return maximum + math.log(float(np.exp(array - maximum).sum()))


@dataclass(frozen=True)
class QuadratureRule:
    """一个场景在归一化 hidden-parameter support 上的积分规则。"""

    scenario: str
    active_coordinates: tuple[str, ...]
    points_per_dimension: int
    nodes: np.ndarray
    prior_weights: np.ndarray

    @property
    def dimension(self) -> int:
        """返回 active hidden-parameter dimension。"""

        return len(self.active_coordinates)

    @property
    def node_count(self) -> int:
        """返回 tensor-product node 数量。"""

        return len(self.nodes)


def make_quadrature_rule(
    scenario: str,
    points_per_dimension: int = 9,
) -> QuadratureRule:
    """构造 5、9 或 17 点 Gauss–Legendre tensor-product rule。"""

    if scenario not in SCENARIOS:
        raise ValueError(f"未知 E2 scenario: {scenario}")
    if points_per_dimension not in QUADRATURE_POINT_COUNTS:
        raise ValueError(
            f"points_per_dimension 必须属于 {QUADRATURE_POINT_COUNTS}"
        )

    active_coordinates = SCENARIO_ACTIVE_COORDINATES[scenario]
    dimension = len(active_coordinates)
    points, integration_weights = np.polynomial.legendre.leggauss(
        points_per_dimension
    )
    one_dimensional_prior_weights = integration_weights / 2.0

    node_meshes = np.meshgrid(*([points] * dimension), indexing="ij")
    nodes = np.stack(
        [mesh.reshape(-1) for mesh in node_meshes],
        axis=1,
    ).astype(np.float64)
    weight_meshes = np.meshgrid(
        *([one_dimensional_prior_weights] * dimension),
        indexing="ij",
    )
    prior_weights = np.ones_like(weight_meshes[0], dtype=np.float64)
    for mesh in weight_meshes:
        prior_weights *= mesh
    prior_weights = prior_weights.reshape(-1)
    prior_weights /= prior_weights.sum()

    return QuadratureRule(
        scenario=scenario,
        active_coordinates=active_coordinates,
        points_per_dimension=points_per_dimension,
        nodes=nodes,
        prior_weights=prior_weights,
    )


def condition_row_from_normalised(
    scenario: str,
    coordinates: np.ndarray,
) -> dict[str, float]:
    """把 active normalised coordinates 转换为 P1 condition row。"""

    if scenario not in SCENARIOS:
        raise ValueError(f"未知 E2 scenario: {scenario}")
    active_coordinates = SCENARIO_ACTIVE_COORDINATES[scenario]
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (len(active_coordinates),):
        raise ValueError(
            f"{scenario} coordinates shape 错误: {values.shape}"
        )
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("normalised hidden coordinates 必须位于 [-1, 1]")

    row = {
        "hidden_u_friction": 0.0,
        "hidden_u_com_x": 0.0,
        "hidden_u_com_y": 0.0,
    }
    for field, value in zip(active_coordinates, values):
        row[field] = float(value)
    return row


def local_motion_observation(
    pre_position_xy: np.ndarray,
    pre_yaw_rad: float,
    post_position_xy: np.ndarray,
    post_yaw_rad: float,
) -> np.ndarray:
    """把一次 push 前后的 world pose 转换为起始 box frame observation。"""

    pre_position = np.asarray(pre_position_xy, dtype=np.float64)
    post_position = np.asarray(post_position_xy, dtype=np.float64)
    if pre_position.shape != (2,) or post_position.shape != (2,):
        raise ValueError("planar position 必须是 shape=(2,) 的数组")
    world_delta = post_position - pre_position
    cosine = math.cos(pre_yaw_rad)
    sine = math.sin(pre_yaw_rad)
    local_delta = np.asarray(
        [
            cosine * world_delta[0] + sine * world_delta[1],
            -sine * world_delta[0] + cosine * world_delta[1],
        ],
        dtype=np.float64,
    )
    return np.asarray(
        [local_delta[0], local_delta[1], post_yaw_rad - pre_yaw_rad],
        dtype=np.float64,
    )


def normalise_observation(observation: np.ndarray) -> np.ndarray:
    """使用 E2 固定尺度标准化三维 motion observation。"""

    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("observation 必须是有限的 shape=(3,) 数组")
    return values / OBSERVATION_SCALE


@dataclass(frozen=True)
class ResidualStatistics:
    """E2 Training 阶段拟合的场景级 likelihood residual statistics。"""

    scenario: str
    observation_fields: tuple[str, ...]
    observation_scale: np.ndarray
    residual_bias: np.ndarray
    base_covariance: np.ndarray
    sample_count: int

    @classmethod
    def load(cls, path: str | Path) -> "ResidualStatistics":
        """读取并核对 residual-statistics artifact。"""

        with np.load(Path(path), allow_pickle=False) as payload:
            statistics = cls(
                scenario=str(payload["scenario"].item()),
                observation_fields=tuple(
                    str(value) for value in payload["observation_fields"]
                ),
                observation_scale=np.asarray(
                    payload["observation_scale"], dtype=np.float64
                ),
                residual_bias=np.asarray(
                    payload["residual_bias"], dtype=np.float64
                ),
                base_covariance=np.asarray(
                    payload["base_covariance"], dtype=np.float64
                ),
                sample_count=int(payload["sample_count"].item()),
            )
        statistics.validate()
        return statistics

    def validate(self) -> None:
        """核对统计量的 observation semantics 与数值结构。"""

        if self.scenario not in SCENARIOS:
            raise ValueError(f"residual statistics scenario 错误: {self.scenario}")
        if self.observation_fields != OUTCOME_FIELDS:
            raise ValueError("residual statistics 的 observation fields 不匹配")
        if self.observation_scale.shape != (3,) or not np.allclose(
            self.observation_scale,
            OBSERVATION_SCALE,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("residual statistics 的 observation scale 不匹配")
        if self.residual_bias.shape != (3,):
            raise ValueError("residual bias shape 必须为 (3,)")
        if self.base_covariance.shape != (3, 3):
            raise ValueError("base covariance shape 必须为 (3, 3)")
        if not np.isfinite(self.residual_bias).all() or not np.isfinite(
            self.base_covariance
        ).all():
            raise ValueError("residual statistics 含非有限值")
        if not np.allclose(
            self.base_covariance,
            self.base_covariance.T,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("base covariance 不对称")
        if self.sample_count < 2:
            raise ValueError("residual statistics 至少需要两个 samples")


class GaussianOutcomeLikelihood:
    """在 fixed quadrature nodes 上评价 P1 action-conditioned likelihood。"""

    def __init__(
        self,
        scenario: str,
        p1: TensorOutcomeInterpolator,
        residual_statistics: ResidualStatistics,
        covariance_inflation: float = 1.0,
        points_per_dimension: int = 9,
    ):
        if scenario not in SCENARIOS:
            raise ValueError(f"未知 E2 scenario: {scenario}")
        if p1.scenario != scenario or residual_statistics.scenario != scenario:
            raise ValueError("P1、residual statistics 与 scenario 不一致")
        if covariance_inflation <= 0.0:
            raise ValueError("covariance inflation 必须大于 0")

        self.scenario = scenario
        self.p1 = p1
        self.residual_statistics = residual_statistics
        self.covariance_inflation = float(covariance_inflation)
        self.rule = make_quadrature_rule(scenario, points_per_dimension)
        self.action_ids = tuple(str(action_id) for action_id in p1.action_ids)
        self._action_index = {
            action_id: index for index, action_id in enumerate(self.action_ids)
        }
        self.set_covariance_inflation(covariance_inflation)
        self.node_means_standardised = self._precompute_node_means()

    def set_covariance_inflation(self, covariance_inflation: float) -> None:
        """更新 likelihood covariance，同时复用与 inflation 无关的 P1 means。"""

        if covariance_inflation <= 0.0:
            raise ValueError("covariance inflation 必须大于 0")
        self.covariance_inflation = float(covariance_inflation)
        self.covariance = (
            self.covariance_inflation
            * self.residual_statistics.base_covariance
            + NUMERICAL_JITTER * np.eye(3, dtype=np.float64)
        )
        cholesky = np.linalg.cholesky(self.covariance)
        self.precision = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, np.eye(3, dtype=np.float64)),
        )
        self.log_normalisation = 0.5 * (
            3.0 * math.log(2.0 * math.pi)
            + 2.0 * float(np.log(np.diag(cholesky)).sum())
        )

    @classmethod
    def from_project_artifacts(
        cls,
        scenario: str,
        covariance_inflation: float = 1.0,
        points_per_dimension: int = 9,
    ) -> "GaussianOutcomeLikelihood":
        """从当前正式 E1 P1 与 E2 residual artifacts 构造 likelihood。"""

        p1_path = (
            HCR_V2_RESULTS_DIR
            / "e1"
            / "p1"
            / scenario
            / "tensor_outcome_interpolator.npz"
        )
        residual_path = (
            HCR_V2_RESULTS_DIR
            / "e2"
            / "residual_statistics"
            / scenario
            / "residual_statistics.npz"
        )
        return cls(
            scenario=scenario,
            p1=TensorOutcomeInterpolator.load(p1_path),
            residual_statistics=ResidualStatistics.load(residual_path),
            covariance_inflation=covariance_inflation,
            points_per_dimension=points_per_dimension,
        )

    def _precompute_node_means(self) -> np.ndarray:
        """预计算所有 action 与 quadrature node 的标准化 likelihood means。"""

        means = np.empty(
            (len(self.action_ids), self.rule.node_count, 3),
            dtype=np.float32,
        )
        for node_index, coordinates in enumerate(self.rule.nodes):
            prediction = self.p1.predict(
                condition_row_from_normalised(self.scenario, coordinates)
            ).astype(np.float64)
            means[:, node_index, :] = (
                prediction / OBSERVATION_SCALE
                + self.residual_statistics.residual_bias
            )
        return means

    def _log_likelihood_from_mean(
        self,
        observation_standardised: np.ndarray,
        mean_standardised: np.ndarray,
    ) -> np.ndarray:
        """计算一批三维 Gaussian log likelihood。"""

        residual = observation_standardised - mean_standardised
        mahalanobis = np.einsum(
            "...i,ij,...j->...",
            residual,
            self.precision,
            residual,
        )
        return -0.5 * mahalanobis - self.log_normalisation

    def node_log_likelihoods(
        self,
        action_id: str,
        observation: np.ndarray,
    ) -> np.ndarray:
        """返回一个 action-observation token 对所有 nodes 的 log likelihood。"""

        if action_id not in self._action_index:
            raise KeyError(f"P1 artifact 中不存在 action: {action_id}")
        action_index = self._action_index[action_id]
        values = self._log_likelihood_from_mean(
            normalise_observation(observation),
            self.node_means_standardised[action_index],
        )
        if not np.isfinite(values).all():
            raise ValueError("node log likelihoods 含非有限值")
        return np.asarray(values, dtype=np.float64)

    def true_log_likelihood(
        self,
        action_id: str,
        observation: np.ndarray,
        true_normalised_coordinates: np.ndarray,
    ) -> float:
        """为离线 NLL/coverage 评价计算真实 continuous condition 的 likelihood。"""

        condition_row = condition_row_from_normalised(
            self.scenario,
            true_normalised_coordinates,
        )
        prediction = self.p1.predict_action(condition_row, action_id).astype(
            np.float64
        )
        mean = (
            prediction / OBSERVATION_SCALE
            + self.residual_statistics.residual_bias
        )
        value = self._log_likelihood_from_mean(
            normalise_observation(observation),
            mean,
        )
        return float(value)


class StudentTOutcomeLikelihood(GaussianOutcomeLikelihood):
    """使用重尾 multivariate Student-t residual model 的稳健 likelihood。"""

    def __init__(
        self,
        scenario: str,
        p1: TensorOutcomeInterpolator,
        residual_statistics: ResidualStatistics,
        covariance_inflation: float = 1.0,
        points_per_dimension: int = 9,
        degrees_of_freedom: float = 3.0,
    ):
        if degrees_of_freedom <= 2.0:
            raise ValueError("Student-t degrees of freedom 必须大于 2")
        self.degrees_of_freedom = float(degrees_of_freedom)
        super().__init__(
            scenario=scenario,
            p1=p1,
            residual_statistics=residual_statistics,
            covariance_inflation=covariance_inflation,
            points_per_dimension=points_per_dimension,
        )

    @classmethod
    def from_project_artifacts(
        cls,
        scenario: str,
        covariance_inflation: float = 1.0,
        points_per_dimension: int = 9,
        degrees_of_freedom: float = 3.0,
    ) -> "StudentTOutcomeLikelihood":
        """从正式 P1 与 residual artifacts 构造稳健 likelihood。"""

        p1_path = (
            HCR_V2_RESULTS_DIR
            / "e1"
            / "p1"
            / scenario
            / "tensor_outcome_interpolator.npz"
        )
        residual_path = (
            HCR_V2_RESULTS_DIR
            / "e2"
            / "residual_statistics"
            / scenario
            / "residual_statistics.npz"
        )
        return cls(
            scenario=scenario,
            p1=TensorOutcomeInterpolator.load(p1_path),
            residual_statistics=ResidualStatistics.load(residual_path),
            covariance_inflation=covariance_inflation,
            points_per_dimension=points_per_dimension,
            degrees_of_freedom=degrees_of_freedom,
        )

    def set_covariance_inflation(self, covariance_inflation: float) -> None:
        """设置 Student-t scale matrix 的 inflation。"""

        super().set_covariance_inflation(covariance_inflation)
        dimension = 3
        degrees = self.degrees_of_freedom
        self.log_normalisation = (
            math.lgamma((degrees + dimension) / 2.0)
            - math.lgamma(degrees / 2.0)
            - 0.5 * dimension * math.log(degrees * math.pi)
            - 0.5 * float(np.linalg.slogdet(self.covariance)[1])
        )

    def _log_likelihood_from_mean(
        self,
        observation_standardised: np.ndarray,
        mean_standardised: np.ndarray,
    ) -> np.ndarray:
        """计算一批三维 multivariate Student-t log likelihood。"""

        residual = observation_standardised - mean_standardised
        mahalanobis = np.einsum(
            "...i,ij,...j->...",
            residual,
            self.precision,
            residual,
        )
        return self.log_normalisation - 0.5 * (
            self.degrees_of_freedom + 3.0
        ) * np.log1p(mahalanobis / self.degrees_of_freedom)


@dataclass(frozen=True)
class PosteriorSummary:
    """Fixed-node posterior 的常用矩摘要。"""

    mean_normalised: np.ndarray
    covariance_normalised: np.ndarray
    uncertainty_contraction: float
    update_count: int


class FixedNodePosterior:
    """在 bounded support 上维护完整 quadrature posterior weights。"""

    def __init__(self, rule: QuadratureRule):
        self.rule = rule
        self.log_prior_weights = np.log(rule.prior_weights)
        self._prior_covariance = self._weighted_covariance(rule.prior_weights)
        self.reset()

    def reset(self) -> None:
        """恢复 bounded-uniform initial prior。"""

        self.cumulative_node_log_likelihoods = np.zeros(
            self.rule.node_count,
            dtype=np.float64,
        )
        self.log_weights = self.log_prior_weights.copy()
        self.update_count = 0

    @property
    def weights(self) -> np.ndarray:
        """返回当前归一化 posterior weights。"""

        return np.exp(self.log_weights)

    def update(self, node_log_likelihoods: np.ndarray) -> PosteriorSummary:
        """使用一个有效 atomic push 的 node likelihoods 更新一次 belief。"""

        values = np.asarray(node_log_likelihoods, dtype=np.float64)
        if values.shape != (self.rule.node_count,) or not np.isfinite(values).all():
            raise ValueError("node log likelihoods shape 或数值错误")
        self.cumulative_node_log_likelihoods += values
        unnormalised = (
            self.log_prior_weights + self.cumulative_node_log_likelihoods
        )
        self.log_weights = unnormalised - logsumexp(unnormalised)
        self.update_count += 1
        return self.summary()

    def _weighted_covariance(self, weights: np.ndarray) -> np.ndarray:
        """计算给定 node weights 的归一化坐标 covariance。"""

        mean = np.sum(weights[:, None] * self.rule.nodes, axis=0)
        centered = self.rule.nodes - mean
        return np.einsum("n,ni,nj->ij", weights, centered, centered)

    def summary(self) -> PosteriorSummary:
        """计算 posterior mean、covariance 与 uncertainty contraction。"""

        weights = self.weights
        mean = np.sum(weights[:, None] * self.rule.nodes, axis=0)
        covariance = self._weighted_covariance(weights)
        prior_trace = float(np.trace(self._prior_covariance))
        contraction = 1.0 - float(np.trace(covariance)) / prior_trace
        return PosteriorSummary(
            mean_normalised=mean,
            covariance_normalised=covariance,
            uncertainty_contraction=contraction,
            update_count=self.update_count,
        )

    def log_evidence(self) -> float:
        """返回当前 history 在 quadrature prior 下的 log evidence。"""

        return logsumexp(
            self.log_prior_weights + self.cumulative_node_log_likelihoods
        )

    def continuous_log_density(
        self,
        cumulative_true_log_likelihood: float,
    ) -> float:
        """计算一个 continuous true condition 的 posterior log density。"""

        if not math.isfinite(cumulative_true_log_likelihood):
            raise ValueError("true-condition cumulative log likelihood 必须有限")
        return (
            -self.rule.dimension * math.log(2.0)
            + cumulative_true_log_likelihood
            - self.log_evidence()
        )

    def hpd_log_density_threshold(self, mass: float = 0.95) -> float:
        """返回 quadrature 近似 HPD region 的 log-density threshold。"""

        if not 0.0 < mass <= 1.0:
            raise ValueError("HPD mass 必须位于 (0, 1]")
        log_node_density = (
            -self.rule.dimension * math.log(2.0)
            + self.log_weights
            - self.log_prior_weights
        )
        order = np.argsort(-log_node_density, kind="stable")
        cumulative_mass = np.cumsum(self.weights[order])
        last_index = int(np.searchsorted(cumulative_mass, mass, side="left"))
        last_index = min(last_index, len(order) - 1)
        return float(log_node_density[order[last_index]])

    def probabilistic_metrics(
        self,
        cumulative_true_log_likelihood: float,
        hpd_mass: float = 0.95,
    ) -> dict[str, float | int]:
        """计算 posterior NLL、HPD coverage 与 uncertainty contraction。"""

        log_density = self.continuous_log_density(
            cumulative_true_log_likelihood
        )
        threshold = self.hpd_log_density_threshold(hpd_mass)
        return {
            "posterior_nll": -log_density,
            "hpd_covered": int(log_density >= threshold - 1e-12),
            "hpd_log_density_threshold": threshold,
            "true_log_density": log_density,
            "uncertainty_contraction": self.summary().uncertainty_contraction,
        }


class PairedEstimatorSuite:
    """在同一批 node likelihoods 上维护 E2 的三个 compared estimators。"""

    def __init__(self, rule: QuadratureRule):
        self.rule = rule
        self.posteriors = {
            name: FixedNodePosterior(rule) for name in ESTIMATOR_NAMES
        }

    def reset(self) -> None:
        """重置一个 episode 的全部 compared estimators。"""

        for posterior in self.posteriors.values():
            posterior.reset()

    def update_from_log_likelihoods(
        self,
        node_log_likelihoods: np.ndarray,
    ) -> dict[str, PosteriorSummary]:
        """用同一个 observation 更新 single 与 sequential estimators。"""

        self.posteriors["single_observation"].reset()
        single_summary = self.posteriors["single_observation"].update(
            node_log_likelihoods
        )
        sequential_summary = self.posteriors["sequential_bayesian"].update(
            node_log_likelihoods
        )
        return {
            "prior_only": self.posteriors["prior_only"].summary(),
            "single_observation": single_summary,
            "sequential_bayesian": sequential_summary,
        }

    def update_observation(
        self,
        likelihood: GaussianOutcomeLikelihood,
        action_id: str,
        observation: np.ndarray,
    ) -> dict[str, PosteriorSummary]:
        """由 action-observation token 计算 likelihood 并更新 paired estimators。"""

        if (
            likelihood.rule.scenario != self.rule.scenario
            or likelihood.rule.points_per_dimension
            != self.rule.points_per_dimension
        ):
            raise ValueError("likelihood 与 estimator suite 的 quadrature rule 不一致")
        return self.update_from_log_likelihoods(
            likelihood.node_log_likelihoods(action_id, observation)
        )


def normalised_to_physical(
    scenario: str,
    normalised_coordinates: np.ndarray,
) -> dict[str, float]:
    """把 posterior point estimate 映射回 hidden parameters 的物理单位。"""

    if scenario not in SCENARIOS:
        raise ValueError(f"未知 E2 scenario: {scenario}")
    values = np.asarray(normalised_coordinates, dtype=np.float64)
    expected_dimension = len(SCENARIO_ACTIVE_COORDINATES[scenario])
    if values.shape != (expected_dimension,):
        raise ValueError(f"{scenario} normalised coordinates shape 错误")

    if scenario == "friction":
        return {"friction_sliding_mu": 0.40 + 0.20 * float(values[0])}
    if scenario == "com":
        return {
            "com_offset_x_m": 0.03 * float(values[0]),
            "com_offset_y_m": 0.03 * float(values[1]),
        }
    if scenario == "joint":
        return {
            "friction_sliding_mu": 0.40 + 0.20 * float(values[0]),
            "com_offset_x_m": 0.03 * float(values[1]),
            "com_offset_y_m": 0.03 * float(values[2]),
        }
    raise AssertionError("已验证的 scenario 应在前述分支中返回")


def parameter_error_metrics(
    scenario: str,
    estimate_normalised: np.ndarray,
    true_normalised: np.ndarray,
) -> dict[str, float]:
    """计算 E2 预声明的 physical-unit point-estimation metrics。"""

    estimate = normalised_to_physical(scenario, estimate_normalised)
    truth = normalised_to_physical(scenario, true_normalised)
    metrics: dict[str, float] = {}
    if scenario in {"friction", "joint"}:
        metrics["friction_absolute_error"] = abs(
            estimate["friction_sliding_mu"] - truth["friction_sliding_mu"]
        )
    if scenario in {"com", "joint"}:
        delta_x = estimate["com_offset_x_m"] - truth["com_offset_x_m"]
        delta_y = estimate["com_offset_y_m"] - truth["com_offset_y_m"]
        metrics["com_euclidean_error_mm"] = (
            math.hypot(delta_x, delta_y) * 1000.0
        )
    return metrics
