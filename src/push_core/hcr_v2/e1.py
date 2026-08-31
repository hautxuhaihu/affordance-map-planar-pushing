"""HCR V2 E1 的 cost、outcome predictor 与评估方法。"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ACTION_FEATURE_FIELDS: tuple[str, ...] = (
    "contact_point_local_x",
    "contact_point_local_y",
    "contact_normal_local_x",
    "contact_normal_local_y",
    "force_direction_local_x",
    "force_direction_local_y",
    "commanded_force_N",
    "ramp_up_s",
    "hold_s",
    "ramp_down_s",
)

OUTCOME_FIELDS: tuple[str, ...] = (
    "real_delta_x",
    "real_delta_y",
    "real_delta_yaw",
)

SCENARIO_ACTIVE_COORDINATES: dict[str, tuple[str, ...]] = {
    "friction": ("hidden_u_friction",),
    "com": ("hidden_u_com_x", "hidden_u_com_y"),
    "joint": ("hidden_u_friction", "hidden_u_com_x", "hidden_u_com_y"),
}

NEAR_OPTIMAL_EPSILON = 0.10


@dataclass(frozen=True)
class TNPOCostConfig:
    """TNPO cost 的容差与权重配置。"""

    position_tolerance_m: float
    yaw_tolerance_rad: float
    position_weight: float = 0.5
    yaw_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m 必须大于 0")
        if self.yaw_tolerance_rad <= 0.0:
            raise ValueError("yaw_tolerance_rad 必须大于 0")
        if not math.isclose(
            self.position_weight + self.yaw_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("position_weight 与 yaw_weight 之和必须为 1")


PRIMARY_TNPO_COST = TNPOCostConfig(
    position_tolerance_m=0.010,
    yaw_tolerance_rad=math.radians(5.0),
)

SENSITIVITY_TNPO_COST = TNPOCostConfig(
    position_tolerance_m=0.010,
    yaw_tolerance_rad=math.radians(10.0),
)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """以 UTF-8 读取 CSV。"""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_csv_rows(paths: Iterable[str | Path]) -> Iterable[dict[str, str]]:
    """按文件顺序流式读取多份 CSV。"""

    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)


def wrap_to_pi(values: np.ndarray | float) -> np.ndarray:
    """把 yaw 归一化到 [-pi, pi)。"""

    array = np.asarray(values, dtype=np.float64)
    return (array + math.pi) % (2.0 * math.pi) - math.pi


def pose_error_matrices(
    outcomes: np.ndarray,
    target_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """计算全部 target-action pairs 的 position 与 yaw 误差。"""

    outcome_array = np.asarray(outcomes, dtype=np.float64)
    target_array = np.asarray(target_positions, dtype=np.float64)
    if outcome_array.ndim != 2 or outcome_array.shape[1] != 3:
        raise ValueError("outcomes 必须具有 (A, 3) 形状")
    if target_array.ndim != 2 or target_array.shape[1] != 2:
        raise ValueError("target_positions 必须具有 (T, 2) 形状")

    displacement_error = (
        outcome_array[None, :, :2] - target_array[:, None, :]
    )
    position_error = np.linalg.norm(displacement_error, axis=2)
    yaw_error = np.abs(wrap_to_pi(outcome_array[:, 2]))
    return position_error, np.broadcast_to(yaw_error[None, :], position_error.shape)


def cost_matrix(
    objective: str,
    position_error: np.ndarray,
    yaw_error: np.ndarray,
    tnpo_config: TNPOCostConfig = PRIMARY_TNPO_COST,
    unwrapped_yaw: np.ndarray | None = None,
) -> np.ndarray:
    """计算 V1、position-only 或 V2 objective。"""

    if objective == "v1":
        yaw_values = yaw_error if unwrapped_yaw is None else np.abs(unwrapped_yaw)
        return position_error + yaw_values / math.radians(10.0)
    if objective == "position_only":
        return position_error.copy()
    if objective in {"v2", "v2_yaw10_sensitivity"}:
        return (
            tnpo_config.position_weight
            * position_error
            / tnpo_config.position_tolerance_m
            + tnpo_config.yaw_weight
            * yaw_error
            / tnpo_config.yaw_tolerance_rad
        )
    raise ValueError(f"未知 objective: {objective}")


def success_matrix(
    position_error: np.ndarray,
    yaw_error: np.ndarray,
    tnpo_config: TNPOCostConfig = PRIMARY_TNPO_COST,
) -> np.ndarray:
    """按 primary position/yaw tolerances 判断 one-step success。"""

    return (
        (position_error <= tnpo_config.position_tolerance_m)
        & (yaw_error <= tnpo_config.yaw_tolerance_rad)
    )


def evaluate_selector(
    actual_outcomes: np.ndarray,
    predicted_outcomes: np.ndarray,
    target_positions: np.ndarray,
    valid_action_mask: np.ndarray,
    objective: str = "v2",
    tnpo_config: TNPOCostConfig = PRIMARY_TNPO_COST,
    near_optimal_epsilon: float = NEAR_OPTIMAL_EPSILON,
) -> dict[str, np.ndarray]:
    """在完整 action library 上比较 predicted selector 与 MuJoCo oracle。"""

    actual = np.asarray(actual_outcomes, dtype=np.float64)
    predicted = np.asarray(predicted_outcomes, dtype=np.float64)
    valid = np.asarray(valid_action_mask, dtype=bool)
    if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != 3:
        raise ValueError("actual_outcomes 与 predicted_outcomes 必须同为 (A, 3)")
    if valid.shape != (actual.shape[0],):
        raise ValueError("valid_action_mask 必须具有 (A,) 形状")
    if not np.any(valid):
        raise ValueError("当前 condition 没有有效 MuJoCo actions")

    actual_position, actual_yaw = pose_error_matrices(actual, target_positions)
    predicted_position, predicted_yaw = pose_error_matrices(predicted, target_positions)
    actual_unwrapped = np.broadcast_to(
        np.abs(actual[None, :, 2]), actual_position.shape
    )
    predicted_unwrapped = np.broadcast_to(
        np.abs(predicted[None, :, 2]), predicted_position.shape
    )
    actual_cost = cost_matrix(
        objective,
        actual_position,
        actual_yaw,
        tnpo_config,
        unwrapped_yaw=actual_unwrapped,
    )
    predicted_cost = cost_matrix(
        objective,
        predicted_position,
        predicted_yaw,
        tnpo_config,
        unwrapped_yaw=predicted_unwrapped,
    )

    oracle_cost_search = actual_cost.copy()
    oracle_cost_search[:, ~valid] = np.inf
    oracle_indices = np.argmin(oracle_cost_search, axis=1)
    selected_indices = np.argmin(predicted_cost, axis=1)
    row_indices = np.arange(target_positions.shape[0])

    oracle_cost = oracle_cost_search[row_indices, oracle_indices]
    selected_valid = valid[selected_indices]
    selected_actual_cost = actual_cost[row_indices, selected_indices]
    selected_actual_cost = np.where(selected_valid, selected_actual_cost, np.nan)
    selected_predicted_cost = predicted_cost[row_indices, selected_indices]
    selection_gap = selected_actual_cost - oracle_cost
    selected_position_error = actual_position[row_indices, selected_indices]
    selected_yaw_error = actual_yaw[row_indices, selected_indices]
    selected_success = (
        selected_valid
        & (selected_position_error <= PRIMARY_TNPO_COST.position_tolerance_m)
        & (selected_yaw_error <= PRIMARY_TNPO_COST.yaw_tolerance_rad)
    )
    near_optimal = (
        selected_valid
        & np.isfinite(selection_gap)
        & (selection_gap <= near_optimal_epsilon)
    )

    return {
        "oracle_indices": oracle_indices,
        "selected_indices": selected_indices,
        "oracle_cost": oracle_cost,
        "selected_actual_cost": selected_actual_cost,
        "selected_predicted_cost": selected_predicted_cost,
        "selection_gap": selection_gap,
        "selected_position_error": selected_position_error,
        "selected_yaw_error": selected_yaw_error,
        "selected_valid": selected_valid,
        "selected_success": selected_success,
        "near_optimal": near_optimal,
        "optimism": selected_actual_cost - selected_predicted_cost,
    }


def outcome_error_summary(
    actual_outcomes: np.ndarray,
    predicted_outcomes: np.ndarray,
    valid_action_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """汇总 planar 与 yaw outcome prediction error。"""

    actual = np.asarray(actual_outcomes, dtype=np.float64)
    predicted = np.asarray(predicted_outcomes, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != 3:
        raise ValueError("actual_outcomes 与 predicted_outcomes 必须同为 (A, 3)")
    valid = (
        np.ones(actual.shape[0], dtype=bool)
        if valid_action_mask is None
        else np.asarray(valid_action_mask, dtype=bool)
    )
    actual_valid = actual[valid]
    predicted_valid = predicted[valid]
    planar_mm = (
        np.linalg.norm(predicted_valid[:, :2] - actual_valid[:, :2], axis=1)
        * 1000.0
    )
    yaw_deg = np.degrees(
        np.abs(wrap_to_pi(predicted_valid[:, 2] - actual_valid[:, 2]))
    )
    true_abs_yaw = np.abs(wrap_to_pi(actual_valid[:, 2]))
    high_yaw_threshold = float(np.quantile(true_abs_yaw, 0.90))
    high_yaw_mask = true_abs_yaw >= high_yaw_threshold

    return {
        "n_actions": int(actual_valid.shape[0]),
        "planar_error_mean_mm": float(np.mean(planar_mm)),
        "planar_error_median_mm": float(np.median(planar_mm)),
        "planar_error_p90_mm": float(np.quantile(planar_mm, 0.90)),
        "yaw_error_mean_deg": float(np.mean(yaw_deg)),
        "yaw_error_median_deg": float(np.median(yaw_deg)),
        "yaw_error_p90_deg": float(np.quantile(yaw_deg, 0.90)),
        "high_yaw_threshold_deg": float(math.degrees(high_yaw_threshold)),
        "high_yaw_planar_error_mean_mm": float(np.mean(planar_mm[high_yaw_mask])),
        "high_yaw_yaw_error_mean_deg": float(np.mean(yaw_deg[high_yaw_mask])),
    }


def summary_statistics(values: np.ndarray) -> dict[str, float | int]:
    """计算有限值的 count、mean、median 与 P90。"""

    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "p90": math.nan}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
    }


def paired_two_way_bootstrap_mean_ci(
    matrix: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 20260810,
) -> tuple[float, float]:
    """对 target-condition crossed matrix 计算 mean 的 95% bootstrap CI。"""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("matrix 必须具有 (targets, conditions) 形状")
    if n_resamples <= 0:
        raise ValueError("n_resamples 必须大于 0")
    rng = np.random.default_rng(seed)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    finite_float = finite.astype(np.float64)
    estimates = np.full(n_resamples, np.nan, dtype=np.float64)
    batch_size = 256
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        size = stop - start
        target_weights = rng.multinomial(
            values.shape[0],
            np.full(values.shape[0], 1.0 / values.shape[0]),
            size=size,
        ).astype(np.float64)
        condition_weights = rng.multinomial(
            values.shape[1],
            np.full(values.shape[1], 1.0 / values.shape[1]),
            size=size,
        ).astype(np.float64)
        weighted_values = np.sum(
            target_weights[:, :, None] * filled[None, :, :], axis=1
        )
        weighted_counts = np.sum(
            target_weights[:, :, None] * finite_float[None, :, :], axis=1
        )
        numerators = np.sum(weighted_values * condition_weights, axis=1)
        denominators = np.sum(weighted_counts * condition_weights, axis=1)
        estimates[start:stop] = np.divide(
            numerators,
            denominators,
            out=np.full(size, np.nan, dtype=np.float64),
            where=denominators > 0.0,
        )
    return (
        float(np.nanquantile(estimates, 0.025)),
        float(np.nanquantile(estimates, 0.975)),
    )


def condition_vector(
    condition_row: dict[str, Any],
    scenario: str,
) -> np.ndarray:
    """按 scenario 提取归一化 hidden-parameter vector。"""

    try:
        fields = SCENARIO_ACTIVE_COORDINATES[scenario]
    except KeyError as exc:
        raise ValueError(f"未知 scenario: {scenario}") from exc
    return np.asarray([float(condition_row[field]) for field in fields], dtype=np.float32)


def action_feature_matrix(action_rows: list[dict[str, Any]]) -> np.ndarray:
    """把 action manifest 转换为固定顺序的 10 维 feature matrix。"""

    return np.asarray(
        [
            [float(row[field]) for field in ACTION_FEATURE_FIELDS]
            for row in action_rows
        ],
        dtype=np.float32,
    )


def conditioned_feature_matrix(
    action_rows: list[dict[str, Any]],
    condition_row: dict[str, Any],
    scenario: str,
) -> np.ndarray:
    """构造 action features 与 active hidden coordinates 的拼接输入。"""

    actions = action_feature_matrix(action_rows)
    hidden = condition_vector(condition_row, scenario)
    tiled_hidden = np.broadcast_to(hidden[None, :], (actions.shape[0], hidden.size))
    return np.concatenate([actions, tiled_hidden], axis=1).astype(np.float32)


class TensorOutcomeInterpolator:
    """在每个 action 的 5-point hidden-parameter grid 上做 multilinear interpolation。"""

    def __init__(
        self,
        scenario: str,
        action_ids: np.ndarray,
        axes: tuple[np.ndarray, ...],
        outcome_grid: np.ndarray,
    ):
        self.scenario = scenario
        self.action_ids = np.asarray(action_ids, dtype=str)
        self._action_index = {
            action_id: index for index, action_id in enumerate(self.action_ids)
        }
        self.axes = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
        self.outcome_grid = np.asarray(outcome_grid, dtype=np.float32)
        expected_shape = tuple(len(axis) for axis in self.axes) + (
            len(self.action_ids),
            3,
        )
        if self.outcome_grid.shape != expected_shape:
            raise ValueError(
                f"outcome_grid 形状错误: expected={expected_shape}, "
                f"observed={self.outcome_grid.shape}"
            )

    @classmethod
    def fit(
        cls,
        scenario: str,
        action_rows: list[dict[str, Any]],
        outcome_rows: Iterable[dict[str, Any]],
    ) -> "TensorOutcomeInterpolator":
        """从完整 training-anchor outcomes 建立插值张量。"""

        coordinate_fields = SCENARIO_ACTIVE_COORDINATES[scenario]
        action_ids = np.asarray([str(row["v2_action_id"]) for row in action_rows])
        action_index = {action_id: index for index, action_id in enumerate(action_ids)}
        axes = tuple(
            np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
            for _ in coordinate_fields
        )
        axis_indices = [
            {round(float(value), 9): index for index, value in enumerate(axis)}
            for axis in axes
        ]
        grid = np.full(
            tuple(len(axis) for axis in axes) + (len(action_ids), 3),
            np.nan,
            dtype=np.float32,
        )
        for row in outcome_rows:
            grid_index = tuple(
                axis_indices[dimension][round(float(row[field]), 9)]
                for dimension, field in enumerate(coordinate_fields)
            )
            action_id = str(row["v2_action_id"])
            grid[grid_index + (action_index[action_id], slice(None))] = [
                float(row[field]) for field in OUTCOME_FIELDS
            ]
        if np.isnan(grid).any():
            missing = int(np.isnan(grid[..., 0]).sum())
            raise ValueError(f"P1 training grid 不完整，缺少 {missing} 个 condition-action outcomes")
        return cls(scenario, action_ids, axes, grid)

    def _interpolation_terms(
        self,
        condition_row: dict[str, Any],
    ) -> list[tuple[tuple[int, ...], float]]:
        """返回一个 hidden condition 对应的插值网格角点与权重。"""

        query = condition_vector(condition_row, self.scenario).astype(np.float64)
        brackets: list[tuple[int, int, float]] = []
        for value, axis in zip(query, self.axes):
            if value < axis[0] - 1e-12 or value > axis[-1] + 1e-12:
                raise ValueError(f"P1 不允许 extrapolation: value={value}, axis={axis}")
            upper = int(np.searchsorted(axis, value, side="right"))
            if upper == 0:
                brackets.append((0, 0, 0.0))
                continue
            if upper >= len(axis):
                last = len(axis) - 1
                brackets.append((last, last, 0.0))
                continue
            lower = upper - 1
            weight = float((value - axis[lower]) / (axis[upper] - axis[lower]))
            brackets.append((lower, upper, weight))

        corner_choices = [
            (0,) if lower == upper else (0, 1)
            for lower, upper, _ in brackets
        ]
        terms: list[tuple[tuple[int, ...], float]] = []
        for corner in product(*corner_choices):
            indices = []
            weight = 1.0
            for use_upper, (lower, upper, upper_weight) in zip(corner, brackets):
                if lower == upper:
                    indices.append(lower)
                    continue
                indices.append(upper if use_upper else lower)
                weight *= upper_weight if use_upper else 1.0 - upper_weight
            terms.append((tuple(indices), weight))
        return terms

    def predict(self, condition_row: dict[str, Any]) -> np.ndarray:
        """对一个 hidden condition 预测完整 action library outcomes。"""

        prediction = np.zeros((len(self.action_ids), 3), dtype=np.float64)
        for indices, weight in self._interpolation_terms(condition_row):
            prediction += weight * self.outcome_grid[indices]
        return prediction.astype(np.float32)

    def predict_action(
        self,
        condition_row: dict[str, Any],
        action_id: str,
    ) -> np.ndarray:
        """只预测一个 action 的三维 outcome。"""

        if action_id not in self._action_index:
            raise KeyError(f"P1 artifact 中不存在 action: {action_id}")
        action_index = self._action_index[action_id]
        prediction = np.zeros(3, dtype=np.float64)
        for indices, weight in self._interpolation_terms(condition_row):
            prediction += weight * self.outcome_grid[indices + (action_index,)]
        return prediction.astype(np.float32)

    def save(self, path: str | Path) -> None:
        """保存 P1 interpolation artifact。"""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "scenario": np.asarray(self.scenario),
            "action_ids": self.action_ids,
            "outcome_grid": self.outcome_grid,
            "n_axes": np.asarray(len(self.axes), dtype=np.int64),
        }
        for index, axis in enumerate(self.axes):
            payload[f"axis_{index}"] = axis
        np.savez_compressed(output_path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "TensorOutcomeInterpolator":
        """读取 P1 interpolation artifact。"""

        with np.load(Path(path), allow_pickle=False) as payload:
            n_axes = int(payload["n_axes"])
            axes = tuple(payload[f"axis_{index}"] for index in range(n_axes))
            return cls(
                scenario=str(payload["scenario"]),
                action_ids=payload["action_ids"],
                axes=axes,
                outcome_grid=payload["outcome_grid"],
            )


class ConditionedOutcomePredictor:
    """加载 P2 condition-conditioned MLP 并预测完整 action library outcomes。"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        normaliser_path: str | Path,
        device: Any = "cpu",
    ):
        import torch

        from push_core.forward_model.models import OutcomeMLP
        from push_core.forward_model.normaliser import StandardNormaliser

        self._torch = torch
        self.checkpoint_path = Path(checkpoint_path)
        self.normaliser_path = Path(normaliser_path)
        self.device = torch.device(device)
        with self.normaliser_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.scenario = str(metadata["scenario"])
        self.input_fields = tuple(metadata["input_fields"])
        expected_fields = ACTION_FEATURE_FIELDS + SCENARIO_ACTIVE_COORDINATES[self.scenario]
        if self.input_fields != expected_fields:
            raise ValueError("P2 normaliser 的 input_fields 与 scenario 不一致")
        self.normaliser = StandardNormaliser.from_json_dict(metadata)
        hidden_dim = int(metadata.get("hidden_dim", 64))
        self.model = OutcomeMLP(
            input_dim=len(self.input_fields),
            hidden_dim=hidden_dim,
        )
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        action_rows: list[dict[str, Any]],
        condition_row: dict[str, Any],
    ) -> np.ndarray:
        """预测一个 condition 下全部 actions 的三维 outcome。"""

        raw = conditioned_feature_matrix(action_rows, condition_row, self.scenario)
        transformed = self.normaliser.transform(raw).astype(np.float32)
        with self._torch.no_grad():
            values = self.model(
                self._torch.from_numpy(transformed).to(self.device)
            ).detach().cpu().numpy()
        return values.astype(np.float32)


def action_distance_components(
    selected_action: dict[str, Any],
    reference_action: dict[str, Any],
    action_rows: list[dict[str, Any]],
    box_size_x_m: float = 0.090,
    box_size_y_m: float = 0.060,
) -> dict[str, float]:
    """分别计算 contact、direction、force 与 timing 的归一化距离。"""

    contact_selected = np.asarray(
        [
            float(selected_action["contact_point_local_x"]),
            float(selected_action["contact_point_local_y"]),
        ]
    )
    contact_reference = np.asarray(
        [
            float(reference_action["contact_point_local_x"]),
            float(reference_action["contact_point_local_y"]),
        ]
    )
    contact_scale = math.hypot(box_size_x_m, box_size_y_m)

    direction_selected = np.asarray(
        [
            float(selected_action["force_direction_local_x"]),
            float(selected_action["force_direction_local_y"]),
        ]
    )
    direction_reference = np.asarray(
        [
            float(reference_action["force_direction_local_x"]),
            float(reference_action["force_direction_local_y"]),
        ]
    )
    direction_dot = float(
        np.clip(np.dot(direction_selected, direction_reference), -1.0, 1.0)
    )

    force_values = np.asarray(
        [float(row["commanded_force_N"]) for row in action_rows], dtype=np.float64
    )
    force_range = float(np.max(force_values) - np.min(force_values))

    timing_fields = ("ramp_up_s", "hold_s", "ramp_down_s")
    timing_differences = []
    for field in timing_fields:
        values = np.asarray([float(row[field]) for row in action_rows], dtype=np.float64)
        value_range = float(np.max(values) - np.min(values))
        timing_differences.append(
            abs(float(selected_action[field]) - float(reference_action[field]))
            / value_range
        )

    return {
        "contact_distance": float(
            np.linalg.norm(contact_selected - contact_reference) / contact_scale
        ),
        "direction_distance": float(math.acos(direction_dot) / math.pi),
        "force_distance": float(
            abs(
                float(selected_action["commanded_force_N"])
                - float(reference_action["commanded_force_N"])
            )
            / force_range
        ),
        "timing_distance": float(
            np.linalg.norm(np.asarray(timing_differences, dtype=np.float64))
            / math.sqrt(3.0)
        ),
    }
