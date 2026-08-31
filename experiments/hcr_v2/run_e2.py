"""HCR V2 E2 的统一分析与评估入口。

当前提供以下子命令：

- prepare-targets：分析 Validation long-distance candidates 并生成正式 target manifest。
- fit-residuals：使用 E2 Training outcomes 拟合 P1 likelihood residual statistics。
- diagnose-validation：检查 residual calibration 与 quadrature sensitivity。
- calibrate-validation：比较 Gaussian inflation 与 update horizon。
- calibrate-student-t-validation：比较 Student-t inflation 与 update horizon。
- compare-final-quadrature：在最终 Student-t 配置下比较 9-point 与 17-point。
- evaluate-validation：使用固定 calibration rule 生成正式 Validation 结果。
- bootstrap-validation：生成正式 Validation paired bootstrap 与结果表。
- evaluate-test：使用固定 Validation 配置评价 Independent Test。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from push_core.hcr_v2.e1 import (
    OUTCOME_FIELDS,
    TensorOutcomeInterpolator,
    read_csv_rows,
)
from push_core.hcr_v2.e2 import (
    ESTIMATOR_NAMES,
    FixedNodePosterior,
    GaussianOutcomeLikelihood,
    PairedEstimatorSuite,
    QUADRATURE_POINT_COUNTS,
    StudentTOutcomeLikelihood,
    condition_row_from_normalised,
    normalise_observation,
    parameter_error_metrics,
)
from push_core.project_paths import HCR_V2_DATA_DIR, HCR_V2_RESULTS_DIR


PROTOCOL_VERSION = "hcr_v2_e2_v1"
SCENARIOS = ("friction", "com", "joint")
E2_DATA_ROOT = HCR_V2_DATA_DIR / "e2"
SELECTION_RULE_VERSION = "hcr_v2_e2_long_distance_targets_v1"
FRICTION_CONE = "elliptic"
VALIDATION_CANDIDATE_SEED = 2026081201
TEST_CANDIDATE_SEED = 2026081202
EXPECTED_TRAJECTORIES = 5000
TRAJECTORY_LENGTH = 5
DISTANCE_THRESHOLD_M = 0.18
ABS_YAW_THRESHOLD_DEG = 5.0
TARGETS_PER_WITNESS_STEP = 32
WITNESS_STEPS = (2, 3, 4, 5)
VALIDATION_PUSH_CEILING = 20
FINAL_TEST_MAXIMUM_PUSH_BUDGET = 20
INFLATION_CANDIDATES = (1.0, 2.0, 4.0, 8.0, 16.0)
DIAGNOSTIC_INFLATION_CANDIDATES = (
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
)
DIAGNOSTIC_UPDATE_HORIZONS = (3, 4, None)
FULL_CALIBRATION_INFLATION_CANDIDATES = (4.0, 8.0, 16.0, 32.0, 64.0)
FULL_CALIBRATION_QUADRATURE_POINTS = 17
FINAL_LIKELIHOOD_FAMILY = "student_t"
FINAL_STUDENT_T_DEGREES_OF_FREEDOM = 3.0
FINAL_QUADRATURE_POINTS = 17
FINAL_BELIEF_UPDATE_HORIZON = 4
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 2026081301
DEFAULT_TEST_BOOTSTRAP_SEED = 2026081304
CALIBRATION_COVERAGE_LOWER = 0.93
CALIBRATION_COVERAGE_UPPER = 0.97
STUDENT_T_DEGREES_CANDIDATES = (3.0, 5.0, 10.0)
STUDENT_T_INFLATION_CANDIDATES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
STUDENT_T_FULL_CANDIDATES = {
    scenario: (3.0, STUDENT_T_INFLATION_CANDIDATES)
    for scenario in SCENARIOS
}
PRIMARY_QUADRATURE_POINTS = 9
OBSERVATION_SCALE = np.asarray(
    [0.010, 0.010, math.radians(5.0)],
    dtype=np.float64,
)

CANDIDATE_PATH = (
    E2_DATA_ROOT / "target_candidates" / "validation" / "candidate_steps.csv"
)
TEST_CANDIDATE_PATH = (
    E2_DATA_ROOT / "target_candidates" / "test" / "candidate_steps.csv"
)
MANIFEST_DIR = REPOSITORY_ROOT / "manifests" / "hcr_v2"
TARGET_MANIFEST_PATH = MANIFEST_DIR / "hcr_v2_core_target_manifest_v1.csv"
BASE_XML_PATH = PROJECT_ROOT / "assets" / "xml" / "msc_rod_pusher_box_hcr_v2.xml"
OUTPUT_MANIFEST_PATH = (
    MANIFEST_DIR
    / "hcr_v2_e2_long_distance_validation_target_manifest_v1.csv"
)
TEST_OUTPUT_MANIFEST_PATH = (
    MANIFEST_DIR / "hcr_v2_e2_long_distance_test_target_manifest_v1.csv"
)
CORE_SUBSET_MANIFEST_PATH = (
    MANIFEST_DIR / "hcr_v2_e2_core_validation_target_manifest_v1.csv"
)
TEST_CORE_SUBSET_MANIFEST_PATH = (
    MANIFEST_DIR / "hcr_v2_e2_core_test_target_manifest_v1.csv"
)
SUMMARY_PATH = (
    HCR_V2_RESULTS_DIR
    / "e2"
    / "target_candidates"
    / "validation"
    / "long_distance_target_selection_summary.json"
)
TEST_SUMMARY_PATH = (
    HCR_V2_RESULTS_DIR
    / "e2"
    / "target_candidates"
    / "test"
    / "long_distance_target_selection_summary.json"
)
TRAINING_OUTCOME_ROOT = E2_DATA_ROOT / "training_outcomes"
RESIDUAL_STATISTICS_ROOT = HCR_V2_RESULTS_DIR / "e2" / "residual_statistics"
VALIDATION_HISTORY_ROOT = E2_DATA_ROOT / "histories" / "validation"
VALIDATION_RESULT_ROOT = HCR_V2_RESULTS_DIR / "e2" / "validation"
TEST_HISTORY_ROOT = E2_DATA_ROOT / "histories" / "test"
TEST_RESULT_ROOT = HCR_V2_RESULTS_DIR / "e2" / "test"
VALIDATION_DIAGNOSTIC_ROOT = VALIDATION_RESULT_ROOT / "diagnostics"
FINAL_QUADRATURE_COMPARISON_PATH = (
    VALIDATION_DIAGNOSTIC_ROOT / "final_quadrature_comparison.json"
)

TARGET_FIELDS = [
    "v2_target_id",
    "split_role",
    "target_group",
    "target_stratum",
    "target_delta_x_m",
    "target_delta_y_m",
    "target_yaw_offset_rad",
    "canonical_position_key",
    "radial_distance_m",
    "witness_push_count",
    "witness_trajectory_id",
    "witness_action_ids",
    "witness_relative_yaw_rad",
    "witness_abs_yaw_deg",
    "witness_distance_m",
    "selection_rank_within_stratum",
    "source_base_random_seed",
    "source_protocol_version",
    "friction_cone",
    "environment_xml",
    "selection_rule_version",
]
CORE_SUBSET_FIELDS = [
    "v2_target_id",
    "split_role",
    "target_group",
    "target_stratum",
    "target_delta_x_m",
    "target_delta_y_m",
    "target_yaw_offset_rad",
    "canonical_position_key",
    "radial_distance_m",
    "selection_rank_within_stratum",
    "source_core_rank_within_stratum",
    "source_core_target_manifest",
    "selection_rule_version",
]

EVALUATION_ROW_FIELDS = [
    "protocol_version",
    "friction_cone",
    "scenario",
    "condition_id",
    "target_id",
    "target_group",
    "target_stratum",
    "episode_key",
    "episode_success",
    "terminal_reason",
    "update_index",
    "episode_update_count",
    "terminal_update_count",
    "is_terminal_update",
    "estimator",
    "likelihood_family",
    "student_t_degrees_of_freedom",
    "covariance_inflation",
    "belief_update_horizon",
    "points_per_dimension",
    "node_count",
    "posterior_nll",
    "hpd_covered",
    "true_log_density",
    "hpd_log_density_threshold",
    "uncertainty_contraction",
    "friction_absolute_error",
    "com_euclidean_error_mm",
]

BOOTSTRAP_METHOD_FIELDS = [
    "scenario",
    "estimator",
    "subset",
    "metric",
    "episodes",
    "point_estimate",
    "ci95_low",
    "ci95_high",
    "coverage_deviation",
    "bootstrap_resamples",
    "bootstrap_seed",
]

BOOTSTRAP_COMPARISON_FIELDS = [
    "scenario",
    "hypothesis",
    "baseline",
    "minimum_updates",
    "metric",
    "paired_episodes",
    "baseline_mean",
    "baseline_ci95_low",
    "baseline_ci95_high",
    "sequential_mean",
    "sequential_ci95_low",
    "sequential_ci95_high",
    "mean_sequential_minus_baseline",
    "effect_ci95_low",
    "effect_ci95_high",
    "sequential_better_rate",
    "better_rate_ci95_low",
    "better_rate_ci95_high",
    "supports_lower_sequential",
    "bootstrap_resamples",
    "bootstrap_seed",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 写出严格 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    """以稳定字段顺序写出 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_scenarios(value: str) -> tuple[str, ...]:
    """解析单一场景或全部场景。"""

    return SCENARIOS if value == "all" else (value,)


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    """把数值矩阵转换为可写入 JSON 的列表。"""

    return [[float(value) for value in row] for row in matrix]


def fit_scenario_residuals(scenario: str) -> dict[str, Any]:
    """拟合一个场景的 P1 residual bias 与 centered base covariance。"""

    outcome_paths = sorted((TRAINING_OUTCOME_ROOT / scenario).glob("*.csv"))
    if len(outcome_paths) != 16:
        raise RuntimeError(
            f"{scenario} E2 Training condition 数量错误: {len(outcome_paths)}"
        )

    p1_path = (
        HCR_V2_RESULTS_DIR
        / "e1"
        / "p1"
        / scenario
        / "tensor_outcome_interpolator.npz"
    )
    p1 = TensorOutcomeInterpolator.load(p1_path)
    expected_action_ids = [str(action_id) for action_id in p1.action_ids]
    expected_action_set = set(expected_action_ids)

    residual_parts: list[np.ndarray] = []
    condition_ids: list[str] = []
    excluded_invalid_rows = 0
    for path in outcome_paths:
        rows = read_csv_rows(path)
        row_by_action = {row["v2_action_id"]: row for row in rows}
        if len(rows) != len(expected_action_ids) or set(row_by_action) != expected_action_set:
            raise RuntimeError(f"{path} 的 action coverage 与 P1 artifact 不一致")

        condition_ids_in_file = {row["condition_id"] for row in rows}
        if len(condition_ids_in_file) != 1:
            raise RuntimeError(f"{path} 包含多个 hidden conditions")
        condition_ids.append(next(iter(condition_ids_in_file)))

        first_row = rows[0]
        expected_metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "friction_cone": FRICTION_CONE,
            "environment_xml": str(BASE_XML_PATH.resolve()),
            "scenario": scenario,
            "e2_data_role": "training",
        }
        for field, expected in expected_metadata.items():
            observed = {row[field] for row in rows}
            if observed != {expected}:
                raise RuntimeError(f"{path} 的 {field} metadata 不匹配: {observed}")

        prediction = p1.predict(first_row).astype(np.float64)
        observation = np.asarray(
            [
                [float(row_by_action[action_id][field]) for field in OUTCOME_FIELDS]
                for action_id in expected_action_ids
            ],
            dtype=np.float64,
        )
        valid = np.asarray(
            [
                int(row_by_action[action_id]["quality_pass"]) == 1
                and int(row_by_action[action_id]["simulation_unstable"]) == 0
                for action_id in expected_action_ids
            ],
            dtype=bool,
        )
        excluded_invalid_rows += int((~valid).sum())
        residual_parts.append((observation[valid] - prediction[valid]) / OBSERVATION_SCALE)

    residuals = np.concatenate(residual_parts, axis=0)
    if not np.isfinite(residuals).all():
        raise RuntimeError(f"{scenario} standardized residuals 含非有限值")

    bias = residuals.mean(axis=0)
    centered = residuals - bias
    base_covariance = centered.T @ centered / (len(centered) - 1)
    centered_std = np.sqrt(np.diag(base_covariance))
    correlation = base_covariance / np.outer(centered_std, centered_std)
    eigenvalues = np.linalg.eigvalsh(base_covariance)

    raw_residuals = residuals * OBSERVATION_SCALE
    raw_bias = bias * OBSERVATION_SCALE
    output_dir = RESIDUAL_STATISTICS_ROOT / scenario
    artifact_path = output_dir / "residual_statistics.npz"
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        scenario=np.asarray(scenario),
        observation_fields=np.asarray(OUTCOME_FIELDS),
        observation_scale=OBSERVATION_SCALE,
        residual_bias=bias,
        base_covariance=base_covariance,
        base_correlation=correlation,
        sample_count=np.asarray(len(residuals), dtype=np.int64),
    )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "environment_xml": str(BASE_XML_PATH.resolve()),
        "scenario": scenario,
        "source_data_role": "training",
        "source_training_outcome_directory": str(
            (TRAINING_OUTCOME_ROOT / scenario).resolve()
        ),
        "source_p1_artifact": str(p1_path.resolve()),
        "condition_count": len(condition_ids),
        "condition_ids": condition_ids,
        "actions_per_condition": len(expected_action_ids),
        "sample_count": len(residuals),
        "excluded_invalid_rows": excluded_invalid_rows,
        "observation_fields": list(OUTCOME_FIELDS),
        "observation_scale": {
            "delta_x_m": float(OBSERVATION_SCALE[0]),
            "delta_y_m": float(OBSERVATION_SCALE[1]),
            "delta_yaw_rad": float(OBSERVATION_SCALE[2]),
            "delta_yaw_deg": 5.0,
        },
        "residual_bias_standardized": [float(value) for value in bias],
        "residual_bias_physical": {
            "delta_x_mm": float(raw_bias[0] * 1000.0),
            "delta_y_mm": float(raw_bias[1] * 1000.0),
            "delta_yaw_deg": float(math.degrees(raw_bias[2])),
        },
        "centered_base_covariance_standardized": matrix_to_list(base_covariance),
        "centered_residual_std_standardized": [
            float(value) for value in centered_std
        ],
        "centered_residual_std_physical": {
            "delta_x_mm": float(centered_std[0] * OBSERVATION_SCALE[0] * 1000.0),
            "delta_y_mm": float(centered_std[1] * OBSERVATION_SCALE[1] * 1000.0),
            "delta_yaw_deg": float(
                math.degrees(centered_std[2] * OBSERVATION_SCALE[2])
            ),
        },
        "centered_residual_correlation": matrix_to_list(correlation),
        "base_covariance_eigenvalues": [float(value) for value in eigenvalues],
        "base_covariance_condition_number": float(np.linalg.cond(base_covariance)),
        "outcome_error_without_bias_correction": {
            "delta_x_mae_mm": float(np.mean(np.abs(raw_residuals[:, 0])) * 1000.0),
            "delta_y_mae_mm": float(np.mean(np.abs(raw_residuals[:, 1])) * 1000.0),
            "planar_mae_mm": float(
                np.mean(np.linalg.norm(raw_residuals[:, :2], axis=1)) * 1000.0
            ),
            "yaw_mae_deg": float(
                np.mean(np.abs(np.degrees(raw_residuals[:, 2])))
            ),
        },
        "artifact_path": str(artifact_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }
    write_json(summary_path, summary)
    return summary


def fit_residuals(args: argparse.Namespace) -> dict[str, Any]:
    """拟合所选场景的 E2 likelihood residual statistics。"""

    scenario_summaries = {
        scenario: fit_scenario_residuals(scenario)
        for scenario in select_scenarios(args.scenario)
    }
    combined_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "scenarios": list(scenario_summaries),
        "total_samples": sum(
            summary["sample_count"] for summary in scenario_summaries.values()
        ),
        "scenario_summaries": scenario_summaries,
    }
    combined_path = RESIDUAL_STATISTICS_ROOT / "combined_summary.json"
    write_json(combined_path, combined_summary)
    for scenario, summary in scenario_summaries.items():
        print(
            f"{scenario}: samples={summary['sample_count']}, "
            f"artifact={summary['artifact_path']}"
        )
    print(f"Combined summary: {combined_path.resolve()}")
    return combined_summary


def distribution_summary(values: list[float]) -> dict[str, float]:
    """计算用于记录 target 分布的简洁统计量。"""

    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    names = ("min", "p25", "median", "p75", "p90", "p95", "max")
    return {name: round(float(value), 8) for name, value in zip(names, quantiles)}


def validate_candidate_rows(
    rows: list[dict[str, str]],
    split_role: str,
    base_seed: int,
) -> dict[str, Any]:
    """核对正式 candidate collection 的完整性。"""

    if len(rows) != EXPECTED_TRAJECTORIES * TRAJECTORY_LENGTH:
        raise RuntimeError(f"candidate row 数量错误: {len(rows)}")

    expected_metadata = {
        "experiment_id": "E2",
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "environment_xml": str(BASE_XML_PATH.resolve()),
        "split_role": split_role,
        "base_random_seed": str(base_seed),
    }
    for field, expected in expected_metadata.items():
        observed = {row[field] for row in rows}
        if observed != {expected}:
            raise RuntimeError(f"candidate {field} metadata 不匹配: {observed}")

    steps_by_trajectory: dict[int, set[int]] = defaultdict(set)
    seen_pairs: set[tuple[int, int]] = set()
    invalid_trajectory_ids: set[int] = set()
    step_counts: Counter[int] = Counter()
    for row in rows:
        trajectory_id = int(row["trajectory_id"])
        step_index = int(row["step_index"])
        pair = (trajectory_id, step_index)
        if pair in seen_pairs:
            raise RuntimeError(f"重复 trajectory-step: {pair}")
        seen_pairs.add(pair)
        steps_by_trajectory[trajectory_id].add(step_index)
        step_counts[step_index] += 1

        push_valid = int(row["push_valid"]) == 1
        if not push_valid:
            invalid_trajectory_ids.add(trajectory_id)
        if (
            int(row["quality_pass"]) != 1
            or int(row["simulation_unstable"]) != 0
            or int(row["contact_success"]) != 1
            or int(row["stopped_by_threshold"]) != 1
        ):
            raise RuntimeError(f"trajectory {trajectory_id} step {step_index} 无效")

        numeric_values = [
            float(row["target_delta_x_m"]),
            float(row["target_delta_y_m"]),
            float(row["target_relative_yaw_rad"]),
            float(row["target_distance_m"]),
        ]
        if not all(math.isfinite(value) for value in numeric_values):
            raise RuntimeError(f"trajectory {trajectory_id} step {step_index} 含非有限值")

        expected_prefix_length = step_index
        observed_prefix_length = len(row["prefix_action_ids"].split("|"))
        if observed_prefix_length != expected_prefix_length:
            raise RuntimeError(f"trajectory {trajectory_id} 的 action prefix 长度错误")

    expected_steps = set(range(1, TRAJECTORY_LENGTH + 1))
    if len(steps_by_trajectory) != EXPECTED_TRAJECTORIES:
        raise RuntimeError(f"trajectory 数量错误: {len(steps_by_trajectory)}")
    if any(steps != expected_steps for steps in steps_by_trajectory.values()):
        raise RuntimeError("存在不完整的五步 trajectory")
    if any(step_counts[step] != EXPECTED_TRAJECTORIES for step in expected_steps):
        raise RuntimeError(f"step rows 数量错误: {dict(step_counts)}")

    return {
        "recorded_step_rows": len(rows),
        "completed_trajectories": len(steps_by_trajectory),
        "invalid_trajectories": len(invalid_trajectory_ids),
        "step_row_counts": {
            str(step): int(step_counts[step]) for step in sorted(step_counts)
        },
    }


def load_core_keys(split_role: str) -> tuple[set[str], float]:
    """读取指定 split 的 core keys 并计算 radial-distance P90。"""

    rows = [
        row
        for row in read_csv_rows(TARGET_MANIFEST_PATH)
        if row["split_role"] == split_role
    ]
    keys = {row["canonical_position_key"] for row in rows}
    if len(rows) != 408 or len(keys) != 408:
        raise RuntimeError(
            f"core {split_role} target manifest 数量或 key 唯一性错误"
        )
    distances = [
        math.hypot(float(row["target_delta_x_m"]), float(row["target_delta_y_m"]))
        for row in rows
    ]
    return keys, float(np.quantile(np.asarray(distances), 0.9))


def prepare_core_subset(
    split_role: str,
    output_path: Path,
) -> list[dict[str, Any]]:
    """按 radial-distance quartiles 选择 128 个 core targets。"""

    rows = [
        row
        for row in read_csv_rows(TARGET_MANIFEST_PATH)
        if row["split_role"] == split_role
    ]
    if len(rows) != 408:
        raise RuntimeError(f"core {split_role} target 数量错误: {len(rows)}")
    rows.sort(
        key=lambda row: (
            math.hypot(
                float(row["target_delta_x_m"]),
                float(row["target_delta_y_m"]),
            ),
            row["v2_target_id"],
        )
    )

    selected: list[dict[str, Any]] = []
    indices = [math.floor(101 * rank / 31) for rank in range(32)]
    for quartile_index in range(4):
        stratum = rows[quartile_index * 102 : (quartile_index + 1) * 102]
        for selection_rank, source_rank in enumerate(indices):
            row = stratum[source_rank]
            distance = math.hypot(
                float(row["target_delta_x_m"]),
                float(row["target_delta_y_m"]),
            )
            selected.append(
                {
                    "v2_target_id": row["v2_target_id"],
                    "split_role": split_role,
                    "target_group": "core",
                    "target_stratum": f"Q{quartile_index + 1}",
                    "target_delta_x_m": row["target_delta_x_m"],
                    "target_delta_y_m": row["target_delta_y_m"],
                    "target_yaw_offset_rad": row["target_yaw_offset_rad"],
                    "canonical_position_key": row["canonical_position_key"],
                    "radial_distance_m": f"{distance:.8f}",
                    "selection_rank_within_stratum": selection_rank,
                    "source_core_rank_within_stratum": source_rank,
                    "source_core_target_manifest": str(
                        TARGET_MANIFEST_PATH.resolve()
                    ),
                    "selection_rule_version": "hcr_v2_e2_core_subset_v1",
                }
            )

    if len(selected) != 128:
        raise RuntimeError(f"core {split_role} subset 数量错误: {len(selected)}")
    if len({row["v2_target_id"] for row in selected}) != 128:
        raise RuntimeError(f"core {split_role} subset target IDs 不唯一")
    write_csv(output_path, selected, CORE_SUBSET_FIELDS)
    return selected


def prepare_candidate_pool(
    rows: list[dict[str, str]],
    core_keys: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """应用固定 Validation long-distance eligibility rule。"""

    eligible_rows = [row for row in rows if int(row["candidate_eligible"]) == 1]
    unique_eligible_keys = {row["target_key_4dp"] for row in eligible_rows}
    core_overlap_before_filter = len(unique_eligible_keys & core_keys)

    filtered: list[dict[str, Any]] = []
    for row in eligible_rows:
        distance = float(row["target_distance_m"])
        abs_yaw_deg = math.degrees(abs(float(row["target_relative_yaw_rad"])))
        key = row["target_key_4dp"]
        if (
            distance >= DISTANCE_THRESHOLD_M
            and abs_yaw_deg <= ABS_YAW_THRESHOLD_DEG
            and key not in core_keys
        ):
            filtered.append(
                {
                    **row,
                    "distance": distance,
                    "abs_yaw_deg": abs_yaw_deg,
                }
            )

    filtered.sort(
        key=lambda row: (
            int(row["step_index"]),
            float(row["distance"]),
            row["target_key_4dp"],
            int(row["trajectory_id"]),
        )
    )
    unique_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for row in filtered:
        key = row["target_key_4dp"]
        if key not in seen_keys:
            seen_keys.add(key)
            unique_rows.append(row)

    return unique_rows, {
        "eligible_candidate_rows": len(eligible_rows),
        "unique_eligible_target_keys_4dp": len(unique_eligible_keys),
        "duplicate_eligible_keys": len(eligible_rows) - len(unique_eligible_keys),
        "core_overlap_before_long_distance_filter": core_overlap_before_filter,
    }


def select_targets(candidate_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 witness push count 分层确定性选择 128 个 targets。"""

    rows_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_pool:
        rows_by_step[int(row["step_index"])].append(row)

    selected: list[dict[str, Any]] = []
    for step in WITNESS_STEPS:
        stratum = rows_by_step[step]
        if len(stratum) < TARGETS_PER_WITNESS_STEP:
            raise RuntimeError(
                f"witness step {step} 只有 {len(stratum)} 个 eligible candidates"
            )
        indices = np.floor(
            np.linspace(0, len(stratum) - 1, TARGETS_PER_WITNESS_STEP)
        ).astype(int)
        if len(set(int(index) for index in indices)) != TARGETS_PER_WITNESS_STEP:
            raise RuntimeError(f"witness step {step} 的 selection ranks 不唯一")
        for rank, index in enumerate(indices):
            selected.append(
                {
                    **stratum[int(index)],
                    "selection_rank_within_stratum": rank,
                }
            )

    if len(selected) != TARGETS_PER_WITNESS_STEP * len(WITNESS_STEPS):
        raise RuntimeError(f"selected target 数量错误: {len(selected)}")
    selected_keys = {row["target_key_4dp"] for row in selected}
    if len(selected_keys) != len(selected):
        raise RuntimeError("selected target keys 不唯一")
    return selected


def build_manifest_rows(
    selected: list[dict[str, Any]],
    split_role: str,
    base_seed: int,
) -> list[dict[str, Any]]:
    """把 selected candidates 转换为正式 target manifest rows。"""

    manifest_rows: list[dict[str, Any]] = []
    for target_index, row in enumerate(selected):
        key_x, key_y = row["target_key_4dp"].split("|")
        target_x = float(key_x)
        target_y = float(key_y)
        manifest_rows.append(
            {
                "v2_target_id": (
                    f"E2L_{'VAL' if split_role == 'validation' else 'TEST'}_"
                    f"{target_index:03d}"
                ),
                "split_role": split_role,
                "target_group": "long_distance",
                "target_stratum": f"witness_{int(row['step_index'])}_push",
                "target_delta_x_m": f"{target_x:.4f}",
                "target_delta_y_m": f"{target_y:.4f}",
                "target_yaw_offset_rad": "0.00000000",
                "canonical_position_key": row["target_key_4dp"],
                "radial_distance_m": f"{math.hypot(target_x, target_y):.8f}",
                "witness_push_count": int(row["step_index"]),
                "witness_trajectory_id": int(row["trajectory_id"]),
                "witness_action_ids": row["prefix_action_ids"],
                "witness_relative_yaw_rad": (
                    f"{float(row['target_relative_yaw_rad']):.8f}"
                ),
                "witness_abs_yaw_deg": f"{float(row['abs_yaw_deg']):.6f}",
                "witness_distance_m": f"{float(row['distance']):.8f}",
                "selection_rank_within_stratum": row[
                    "selection_rank_within_stratum"
                ],
                "source_base_random_seed": base_seed,
                "source_protocol_version": PROTOCOL_VERSION,
                "friction_cone": FRICTION_CONE,
                "environment_xml": str(BASE_XML_PATH.resolve()),
                "selection_rule_version": SELECTION_RULE_VERSION,
            }
        )
    return manifest_rows


def prepare_targets(args: argparse.Namespace) -> dict[str, Any]:
    """生成正式 E2 Validation 或 Test target manifests。"""

    split_role = args.role
    if split_role == "validation":
        candidate_path = CANDIDATE_PATH
        core_subset_path = CORE_SUBSET_MANIFEST_PATH
        output_manifest_path = OUTPUT_MANIFEST_PATH
        summary_path = SUMMARY_PATH
        base_seed = VALIDATION_CANDIDATE_SEED
    else:
        candidate_path = TEST_CANDIDATE_PATH
        core_subset_path = TEST_CORE_SUBSET_MANIFEST_PATH
        output_manifest_path = TEST_OUTPUT_MANIFEST_PATH
        summary_path = TEST_SUMMARY_PATH
        base_seed = TEST_CANDIDATE_SEED

    core_subset_rows = prepare_core_subset(split_role, core_subset_path)
    candidate_rows = read_csv_rows(candidate_path)
    integrity = validate_candidate_rows(candidate_rows, split_role, base_seed)
    core_keys, core_distance_p90 = load_core_keys(split_role)
    candidate_pool, source_counts = prepare_candidate_pool(candidate_rows, core_keys)
    selected = select_targets(candidate_pool)
    manifest_rows = build_manifest_rows(selected, split_role, base_seed)

    pool_counts = Counter(int(row["step_index"]) for row in candidate_pool)
    selected_counts = Counter(int(row["step_index"]) for row in selected)
    selected_distances = [float(row["radial_distance_m"]) for row in manifest_rows]
    selected_yaws = [float(row["witness_abs_yaw_deg"]) for row in manifest_rows]

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_rule_version": SELECTION_RULE_VERSION,
        "friction_cone": FRICTION_CONE,
        "environment_xml": str(BASE_XML_PATH.resolve()),
        "split_role": split_role,
        "source_candidate_path": str(candidate_path.resolve()),
        "source_core_target_manifest": str(TARGET_MANIFEST_PATH.resolve()),
        "output_core_subset_manifest_path": str(
            core_subset_path.resolve()
        ),
        "output_manifest_path": str(output_manifest_path.resolve()),
        "source_base_random_seed": base_seed,
        **integrity,
        **source_counts,
        "core_distance_p90_m": round(core_distance_p90, 8),
        "selection_rule": {
            "minimum_distance_m": DISTANCE_THRESHOLD_M,
            "maximum_absolute_witness_yaw_deg": ABS_YAW_THRESHOLD_DEG,
            "position_key_decimal_places": 4,
            "exclude_core_split_keys": True,
            "witness_push_counts": list(WITNESS_STEPS),
            "targets_per_witness_push_count": TARGETS_PER_WITNESS_STEP,
            "within_stratum_selection": "distance_sorted_equal_interval_ranks",
        },
        "eligible_pool_after_rule": len(candidate_pool),
        "eligible_pool_by_witness_push_count": {
            str(step): int(pool_counts[step]) for step in WITNESS_STEPS
        },
        "selected_targets": len(manifest_rows),
        "selected_targets_by_witness_push_count": {
            str(step): int(selected_counts[step]) for step in WITNESS_STEPS
        },
        "selected_target_distance_m": distribution_summary(selected_distances),
        "selected_witness_abs_yaw_deg": distribution_summary(selected_yaws),
        "selected_core_key_overlap": len(
            {row["canonical_position_key"] for row in manifest_rows} & core_keys
        ),
        "selected_core_targets": len(core_subset_rows),
        "selected_core_targets_by_quartile": {
            str(quartile): sum(
                row["target_stratum"] == f"Q{quartile}"
                for row in core_subset_rows
            )
            for quartile in range(1, 5)
        },
        "maximum_push_budget": FINAL_TEST_MAXIMUM_PUSH_BUDGET,
        "test_candidate_pool_generated": split_role == "test",
    }

    write_csv(output_manifest_path, manifest_rows, TARGET_FIELDS)
    write_json(summary_path, summary)
    print(f"Core target subset: {core_subset_path.resolve()}")
    print(f"Target manifest: {output_manifest_path.resolve()}")
    print(f"Selection summary: {summary_path.resolve()}")
    print(
        "Selected targets: "
        f"{len(manifest_rows)} "
        f"({dict(sorted(selected_counts.items()))})"
    )
    return summary


def load_history_episodes(
    scenario: str,
    data_role: str,
) -> list[list[dict[str, str]]]:
    """读取并按 episode 组织一个场景的正式 histories。"""

    history_root = (
        VALIDATION_HISTORY_ROOT if data_role == "validation" else TEST_HISTORY_ROOT
    )
    expected_conditions = 16 if data_role == "validation" else 32
    expected_episodes = 4096 if data_role == "validation" else 8192
    paths = sorted((history_root / scenario).glob("*.csv"))
    if len(paths) != expected_conditions:
        raise RuntimeError(
            f"{scenario} {data_role} history condition 数量错误: {len(paths)}"
        )

    episodes: list[list[dict[str, str]]] = []
    seen_episode_keys: set[str] = set()
    for path in paths:
        rows = read_csv_rows(path)
        current_rows: list[dict[str, str]] = []
        current_key = ""
        for row in rows:
            episode_key = row["episode_key"]
            if current_rows and episode_key != current_key:
                episodes.append(current_rows)
                current_rows = []
            current_key = episode_key
            current_rows.append(row)
        if current_rows:
            episodes.append(current_rows)

    for episode_rows in episodes:
        first_row = episode_rows[0]
        last_row = episode_rows[-1]
        episode_key = first_row["episode_key"]
        if episode_key in seen_episode_keys:
            raise RuntimeError(f"重复 Validation episode: {episode_key}")
        seen_episode_keys.add(episode_key)

        expected_metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "friction_cone": FRICTION_CONE,
            "e2_data_role": data_role,
            "scenario": scenario,
            "maximum_push_budget": str(VALIDATION_PUSH_CEILING),
        }
        for field, expected in expected_metadata.items():
            observed = {row[field] for row in episode_rows}
            if observed != {expected}:
                raise RuntimeError(
                    f"{episode_key} 的 {field} metadata 不匹配: {observed}"
                )

        push_indices = [int(row["push_index"]) for row in episode_rows]
        if push_indices != list(range(1, len(episode_rows) + 1)):
            raise RuntimeError(f"{episode_key} 的 push index 不连续")
        terminal_flags = [int(row["is_terminal_push"]) for row in episode_rows]
        if sum(terminal_flags) != 1 or terminal_flags[-1] != 1:
            raise RuntimeError(f"{episode_key} 的 terminal row 不唯一")
        if int(last_row["terminal_push_count"]) != len(episode_rows):
            raise RuntimeError(f"{episode_key} 的 terminal push count 不匹配")

        valid_rows = [
            row for row in episode_rows if int(row["valid_observation"]) == 1
        ]
        update_indices = [int(row["valid_update_index"]) for row in valid_rows]
        if update_indices != list(range(1, len(valid_rows) + 1)):
            raise RuntimeError(f"{episode_key} 的 valid update index 不连续")

    if len(episodes) != expected_episodes:
        raise RuntimeError(
            f"{scenario} {data_role} episode 数量错误: {len(episodes)}"
        )
    return episodes


def load_validation_episodes(scenario: str) -> list[list[dict[str, str]]]:
    """读取一个场景的正式 Validation histories。"""

    return load_history_episodes(scenario, "validation")


def history_observation(row: dict[str, str]) -> np.ndarray:
    """从 history row 读取三维 local-frame observation。"""

    return np.asarray(
        [
            float(row["observation_local_delta_x_m"]),
            float(row["observation_local_delta_y_m"]),
            float(row["observation_delta_yaw_rad"]),
        ],
        dtype=np.float64,
    )


def true_normalised_coordinates(
    first_row: dict[str, str],
    likelihood: GaussianOutcomeLikelihood,
) -> np.ndarray:
    """读取仅用于离线评价的真实 hidden-parameter coordinates。"""

    return np.asarray(
        [float(first_row[field]) for field in likelihood.rule.active_coordinates],
        dtype=np.float64,
    )


def select_quadrature_diagnostic_episodes(
    episodes: list[list[dict[str, str]]],
) -> list[list[dict[str, str]]]:
    """从每个 Validation condition 确定性选择四条分层 histories。"""

    by_condition: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for episode_rows in episodes:
        by_condition[episode_rows[0]["condition_id"]].append(episode_rows)

    selected: list[list[dict[str, str]]] = []
    for condition_index, condition_id in enumerate(sorted(by_condition)):
        by_stratum: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
        for episode_rows in by_condition[condition_id]:
            first_row = episode_rows[0]
            stratum_key = (
                f"{first_row['target_group']}::{first_row['target_stratum']}"
            )
            by_stratum[stratum_key].append(episode_rows)

        strata = sorted(by_stratum)
        if len(strata) != 8:
            raise RuntimeError(
                f"{condition_id} 的 target strata 数量错误: {len(strata)}"
            )
        selected_strata = strata[condition_index % 2 :: 2]
        for stratum in selected_strata:
            candidates = sorted(
                by_stratum[stratum],
                key=lambda rows: rows[0]["target_id"],
            )
            selected.append(candidates[0])

    if len(selected) != 64:
        raise RuntimeError(f"quadrature diagnostic episode 数量错误: {len(selected)}")
    return selected


def evaluate_quadrature_diagnostic(
    likelihood: GaussianOutcomeLikelihood,
    episodes: list[list[dict[str, str]]],
    update_horizon: int | None = None,
) -> list[dict[str, Any]]:
    """在固定 histories 上计算一个 quadrature rule 的 terminal posterior。"""

    rows: list[dict[str, Any]] = []
    for episode_rows in episodes:
        posterior = FixedNodePosterior(likelihood.rule)
        first_row = episode_rows[0]
        true_coordinates = true_normalised_coordinates(first_row, likelihood)
        cumulative_true_log_likelihood = 0.0
        used_rows = (
            episode_rows
            if update_horizon is None
            else episode_rows[:update_horizon]
        )
        for row in used_rows:
            observation = history_observation(row)
            action_id = row["v2_action_id"]
            posterior.update(
                likelihood.node_log_likelihoods(action_id, observation)
            )
            cumulative_true_log_likelihood += likelihood.true_log_likelihood(
                action_id,
                observation,
                true_coordinates,
            )

        summary = posterior.summary()
        probabilistic = posterior.probabilistic_metrics(
            cumulative_true_log_likelihood
        )
        point_metrics = parameter_error_metrics(
            likelihood.scenario,
            summary.mean_normalised,
            true_coordinates,
        )
        rows.append(
            {
                "scenario": likelihood.scenario,
                "condition_id": first_row["condition_id"],
                "target_id": first_row["target_id"],
                "target_group": first_row["target_group"],
                "target_stratum": first_row["target_stratum"],
                "episode_key": first_row["episode_key"],
                "episode_update_count": len(episode_rows),
                "used_update_count": len(used_rows),
                "update_horizon": (
                    "full" if update_horizon is None else update_horizon
                ),
                "points_per_dimension": (
                    likelihood.rule.points_per_dimension
                ),
                "node_count": likelihood.rule.node_count,
                "posterior_mean_normalised": [
                    float(value) for value in summary.mean_normalised
                ],
                "posterior_covariance_trace": float(
                    np.trace(summary.covariance_normalised)
                ),
                "posterior_nll": float(probabilistic["posterior_nll"]),
                "hpd_covered": int(probabilistic["hpd_covered"]),
                "uncertainty_contraction": float(
                    probabilistic["uncertainty_contraction"]
                ),
                "friction_absolute_error": point_metrics.get(
                    "friction_absolute_error",
                    None,
                ),
                "com_euclidean_error_mm": point_metrics.get(
                    "com_euclidean_error_mm",
                    None,
                ),
            }
        )
    return rows


def summarise_quadrature_diagnostic(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总一个 quadrature rule 的 64-episode sensitivity 结果。"""

    nll = np.asarray([row["posterior_nll"] for row in rows], dtype=np.float64)
    covariance_trace = np.asarray(
        [row["posterior_covariance_trace"] for row in rows],
        dtype=np.float64,
    )
    result: dict[str, Any] = {
        "episodes": len(rows),
        "points_per_dimension": rows[0]["points_per_dimension"],
        "node_count": rows[0]["node_count"],
        "update_horizon": rows[0]["update_horizon"],
        "mean_used_update_count": float(
            np.mean([row["used_update_count"] for row in rows])
        ),
        "mean_terminal_nll": float(np.mean(nll)),
        "median_terminal_nll": float(np.median(nll)),
        "terminal_hpd_coverage": float(
            np.mean([row["hpd_covered"] for row in rows])
        ),
        "mean_posterior_covariance_trace": float(np.mean(covariance_trace)),
        "median_posterior_covariance_trace": float(
            np.median(covariance_trace)
        ),
    }
    for field in ("friction_absolute_error", "com_euclidean_error_mm"):
        values = [row[field] for row in rows if row[field] is not None]
        if values:
            result[f"mean_{field}"] = float(np.mean(values))
            result[f"median_{field}"] = float(np.median(values))
    return result


def evaluate_diagnostic_configuration_grid(
    scenario: str,
    episodes: list[list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """联合比较 quadrature、inflation 与有限 update horizon。"""

    results: list[dict[str, Any]] = []
    for points in (9, 17):
        for inflation in DIAGNOSTIC_INFLATION_CANDIDATES:
            likelihood = GaussianOutcomeLikelihood.from_project_artifacts(
                scenario,
                covariance_inflation=inflation,
                points_per_dimension=points,
            )
            for update_horizon in DIAGNOSTIC_UPDATE_HORIZONS:
                rows = evaluate_quadrature_diagnostic(
                    likelihood,
                    episodes,
                    update_horizon=update_horizon,
                )
                summary = summarise_quadrature_diagnostic(rows)
                summary["covariance_inflation"] = inflation
                results.append(summary)
    return results


def evaluate_student_t_diagnostic_grid(
    scenario: str,
    episodes: list[list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """在固定 subset 上比较稳健 Student-t likelihood 候选。"""

    results: list[dict[str, Any]] = []
    for degrees in STUDENT_T_DEGREES_CANDIDATES:
        likelihood = StudentTOutcomeLikelihood.from_project_artifacts(
            scenario,
            covariance_inflation=STUDENT_T_INFLATION_CANDIDATES[0],
            points_per_dimension=17,
            degrees_of_freedom=degrees,
        )
        for inflation in STUDENT_T_INFLATION_CANDIDATES:
            likelihood.set_covariance_inflation(inflation)
            for update_horizon in DIAGNOSTIC_UPDATE_HORIZONS:
                rows = evaluate_quadrature_diagnostic(
                    likelihood,
                    episodes,
                    update_horizon=update_horizon,
                )
                summary = summarise_quadrature_diagnostic(rows)
                summary["likelihood_family"] = "student_t"
                summary["degrees_of_freedom"] = degrees
                summary["covariance_inflation"] = inflation
                results.append(summary)
    return results


def compare_quadrature_rows(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """以相同 episode 上的高分辨率结果为 reference 比较数值差异。"""

    reference_by_episode = {
        row["episode_key"]: row for row in reference_rows
    }
    mean_shifts: list[float] = []
    nll_differences: list[float] = []
    covariance_log_ratios: list[float] = []
    coverage_disagreements: list[int] = []
    for row in candidate_rows:
        reference = reference_by_episode[row["episode_key"]]
        mean_shifts.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(row["posterior_mean_normalised"])
                        - np.asarray(reference["posterior_mean_normalised"])
                    )
                )
            )
        )
        nll_differences.append(
            abs(float(row["posterior_nll"]) - float(reference["posterior_nll"]))
        )
        candidate_trace = max(float(row["posterior_covariance_trace"]), 1e-300)
        reference_trace = max(
            float(reference["posterior_covariance_trace"]),
            1e-300,
        )
        covariance_log_ratios.append(
            abs(math.log(candidate_trace / reference_trace))
        )
        coverage_disagreements.append(
            int(row["hpd_covered"] != reference["hpd_covered"])
        )

    return {
        "reference_points_per_dimension": reference_rows[0][
            "points_per_dimension"
        ],
        "candidate_points_per_dimension": candidate_rows[0][
            "points_per_dimension"
        ],
        "mean_max_absolute_posterior_mean_shift_normalised": float(
            np.mean(mean_shifts)
        ),
        "p95_max_absolute_posterior_mean_shift_normalised": float(
            np.quantile(mean_shifts, 0.95)
        ),
        "maximum_absolute_posterior_mean_shift_normalised": float(
            np.max(mean_shifts)
        ),
        "mean_absolute_terminal_nll_difference": float(
            np.mean(nll_differences)
        ),
        "median_absolute_terminal_nll_difference": float(
            np.median(nll_differences)
        ),
        "mean_absolute_log_covariance_trace_ratio": float(
            np.mean(covariance_log_ratios)
        ),
        "coverage_disagreement_rate": float(
            np.mean(coverage_disagreements)
        ),
    }


def compare_final_quadrature(args: argparse.Namespace) -> dict[str, Any]:
    """在最终 Student-t 配置下比较 9-point 与 17-point。"""

    scenario_results: dict[str, dict[str, Any]] = {}
    for scenario in select_scenarios(args.scenario):
        episodes = [
            episode_rows
            for episode_rows in load_validation_episodes(scenario)
            if int(episode_rows[-1]["episode_valid"]) == 1
        ]
        inflation_selection = select_calibrated_student_t_configuration(
            scenario
        )
        covariance_inflation = float(
            inflation_selection["selected_covariance_inflation"]
        )

        rows_by_points: dict[int, list[dict[str, Any]]] = {}
        summaries_by_points: dict[str, dict[str, Any]] = {}
        runtimes_by_points: dict[str, dict[str, float | int]] = {}
        for points in (9, 17):
            likelihood = StudentTOutcomeLikelihood.from_project_artifacts(
                scenario,
                covariance_inflation=covariance_inflation,
                points_per_dimension=points,
                degrees_of_freedom=FINAL_STUDENT_T_DEGREES_OF_FREEDOM,
            )
            started = time.perf_counter()
            rows = evaluate_quadrature_diagnostic(
                likelihood,
                episodes,
                update_horizon=FINAL_BELIEF_UPDATE_HORIZON,
            )
            elapsed_seconds = time.perf_counter() - started
            rows_by_points[points] = rows
            summaries_by_points[str(points)] = (
                summarise_quadrature_diagnostic(rows)
            )
            runtimes_by_points[str(points)] = {
                "seconds": elapsed_seconds,
                "mean_seconds_per_episode": elapsed_seconds / len(rows),
                "node_count": likelihood.rule.node_count,
                "node_mean_cache_mib": (
                    likelihood.node_means_standardised.nbytes / (1024.0**2)
                ),
            }

        comparison = compare_quadrature_rows(
            rows_by_points[17],
            rows_by_points[9],
        )
        summary_9 = summaries_by_points["9"]
        summary_17 = summaries_by_points["17"]
        aggregate_differences: dict[str, float] = {}
        for field in (
            "mean_terminal_nll",
            "median_terminal_nll",
            "terminal_hpd_coverage",
            "mean_posterior_covariance_trace",
            "mean_friction_absolute_error",
            "mean_com_euclidean_error_mm",
        ):
            if field in summary_9 and field in summary_17:
                aggregate_differences[f"{field}_9_minus_17"] = float(
                    summary_9[field] - summary_17[field]
                )

        scenario_results[scenario] = {
            "scenario": scenario,
            "episodes": len(episodes),
            "likelihood_family": FINAL_LIKELIHOOD_FAMILY,
            "student_t_degrees_of_freedom": (
                FINAL_STUDENT_T_DEGREES_OF_FREEDOM
            ),
            "belief_update_horizon": FINAL_BELIEF_UPDATE_HORIZON,
            "covariance_inflation": covariance_inflation,
            "summaries_by_points_per_dimension": summaries_by_points,
            "comparison_9_to_17": comparison,
            "aggregate_differences_9_minus_17": aggregate_differences,
            "runtime_by_points_per_dimension": runtimes_by_points,
        }
        print(
            f"{scenario}: coverage 9/17="
            f"{summary_9['terminal_hpd_coverage']:.4f}/"
            f"{summary_17['terminal_hpd_coverage']:.4f}, "
            f"mean_shift={comparison['mean_max_absolute_posterior_mean_shift_normalised']:.6f}, "
            f"coverage_disagreement={comparison['coverage_disagreement_rate']:.4f}"
        )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "data_role": "validation",
        "purpose": "final_student_t_quadrature_numerical_comparison",
        "test_histories_viewed": False,
        "scenario_results": scenario_results,
    }
    write_json(FINAL_QUADRATURE_COMPARISON_PATH, payload)
    print(
        "Final quadrature comparison: "
        f"{FINAL_QUADRATURE_COMPARISON_PATH.resolve()}"
    )
    return payload


def correlation_by_dimension(
    left: np.ndarray,
    right: np.ndarray,
) -> list[float]:
    """计算两组三维 residual 的逐维相关系数。"""

    correlations: list[float] = []
    for dimension in range(3):
        correlations.append(
            float(np.corrcoef(left[:, dimension], right[:, dimension])[0, 1])
        )
    return correlations


def build_residual_calibration_diagnostic(
    likelihood: GaussianOutcomeLikelihood,
    episodes: list[list[dict[str, str]]],
) -> dict[str, Any]:
    """计算 continuous histories 的 NIS、尺度迁移与 lag-1 correlation。"""

    residuals: list[np.ndarray] = []
    lag_left: list[np.ndarray] = []
    lag_right: list[np.ndarray] = []
    nis_values: list[float] = []
    nis_by_push: dict[int, list[float]] = defaultdict(list)
    condition_nis: dict[str, list[float]] = defaultdict(list)
    statistics = likelihood.residual_statistics
    covariance = statistics.base_covariance + 1e-6 * np.eye(3)
    precision = np.linalg.inv(covariance)

    for episode_rows in episodes:
        first_row = episode_rows[0]
        true_coordinates = true_normalised_coordinates(first_row, likelihood)
        true_condition = condition_row_from_normalised(
            likelihood.scenario,
            true_coordinates,
        )
        episode_residuals: list[np.ndarray] = []
        for push_index, row in enumerate(episode_rows, start=1):
            observation = normalise_observation(history_observation(row))
            prediction = likelihood.p1.predict_action(
                true_condition,
                row["v2_action_id"],
            ).astype(np.float64) / OBSERVATION_SCALE
            centered_residual = (
                observation - prediction - statistics.residual_bias
            )
            nis = float(centered_residual @ precision @ centered_residual)
            residuals.append(centered_residual)
            episode_residuals.append(centered_residual)
            nis_values.append(nis)
            nis_by_push[push_index].append(nis)
            condition_nis[first_row["condition_id"]].append(nis)
        if len(episode_residuals) >= 2:
            lag_left.extend(episode_residuals[:-1])
            lag_right.extend(episode_residuals[1:])

    residual_array = np.asarray(residuals, dtype=np.float64)
    lag_left_array = np.asarray(lag_left, dtype=np.float64)
    lag_right_array = np.asarray(lag_right, dtype=np.float64)
    validation_covariance = np.cov(residual_array, rowvar=False, ddof=1)
    validation_std = np.sqrt(np.diag(validation_covariance))
    training_std = np.sqrt(np.diag(statistics.base_covariance))
    nis_array = np.asarray(nis_values, dtype=np.float64)
    return {
        "sample_count": len(residual_array),
        "observation_dimension": 3,
        "expected_mean_nis": 3.0,
        "mean_nis": float(np.mean(nis_array)),
        "median_nis": float(np.median(nis_array)),
        "p95_nis": float(np.quantile(nis_array, 0.95)),
        "nis_implied_scalar_inflation": float(np.mean(nis_array) / 3.0),
        "centered_residual_mean_standardised": [
            float(value) for value in residual_array.mean(axis=0)
        ],
        "validation_to_training_std_ratio": [
            float(value) for value in validation_std / training_std
        ],
        "validation_centered_covariance": matrix_to_list(
            validation_covariance
        ),
        "training_centered_covariance": matrix_to_list(
            statistics.base_covariance
        ),
        "lag1_pair_count": len(lag_left_array),
        "lag1_residual_correlation": correlation_by_dimension(
            lag_left_array,
            lag_right_array,
        ),
        "mean_nis_by_push_index": {
            str(push_index): float(np.mean(values))
            for push_index, values in sorted(nis_by_push.items())
        },
        "mean_nis_by_condition": {
            condition_id: float(np.mean(values))
            for condition_id, values in sorted(condition_nis.items())
        },
    }


def diagnose_validation(args: argparse.Namespace) -> dict[str, Any]:
    """运行不改动正式配置的 Validation calibration diagnostics。"""

    scenario_results: dict[str, Any] = {}
    for scenario in select_scenarios(args.scenario):
        all_episodes = load_validation_episodes(scenario)
        valid_episodes = [
            rows for rows in all_episodes if int(rows[-1]["episode_valid"]) == 1
        ]
        diagnostic_episodes = select_quadrature_diagnostic_episodes(
            valid_episodes
        )
        rows_by_points: dict[int, list[dict[str, Any]]] = {}
        summaries_by_points: dict[str, dict[str, Any]] = {}
        primary_likelihood: GaussianOutcomeLikelihood | None = None
        for points in QUADRATURE_POINT_COUNTS:
            likelihood = GaussianOutcomeLikelihood.from_project_artifacts(
                scenario,
                covariance_inflation=1.0,
                points_per_dimension=points,
            )
            rows = evaluate_quadrature_diagnostic(
                likelihood,
                diagnostic_episodes,
            )
            rows_by_points[points] = rows
            summaries_by_points[str(points)] = summarise_quadrature_diagnostic(
                rows
            )
            if points == PRIMARY_QUADRATURE_POINTS:
                primary_likelihood = likelihood
            print(
                f"{scenario} quadrature={points}: "
                f"coverage={summaries_by_points[str(points)]['terminal_hpd_coverage']:.4f}"
            )

        if primary_likelihood is None:
            raise RuntimeError("没有构造 primary quadrature likelihood")
        comparisons = {
            "5_vs_17": compare_quadrature_rows(
                rows_by_points[17],
                rows_by_points[5],
            ),
            "9_vs_17": compare_quadrature_rows(
                rows_by_points[17],
                rows_by_points[9],
            ),
        }
        residual_diagnostic = build_residual_calibration_diagnostic(
            primary_likelihood,
            valid_episodes,
        )
        configuration_grid = evaluate_diagnostic_configuration_grid(
            scenario,
            diagnostic_episodes,
        )
        student_t_grid = evaluate_student_t_diagnostic_grid(
            scenario,
            diagnostic_episodes,
        )
        scenario_output_dir = VALIDATION_DIAGNOSTIC_ROOT / scenario
        write_json(
            scenario_output_dir / "quadrature_sensitivity.json",
            {
                "scenario": scenario,
                "covariance_inflation": 1.0,
                "episode_selection": (
                    "16 conditions x 4 deterministic stratified targets"
                ),
                "summaries": summaries_by_points,
                "comparisons": comparisons,
            },
        )
        write_json(
            scenario_output_dir / "residual_calibration.json",
            residual_diagnostic,
        )
        write_json(
            scenario_output_dir / "configuration_grid.json",
            {
                "scenario": scenario,
                "episode_selection": (
                    "16 conditions x 4 deterministic stratified targets"
                ),
                "candidate_results": configuration_grid,
            },
        )
        write_json(
            scenario_output_dir / "student_t_configuration_grid.json",
            {
                "scenario": scenario,
                "candidate_results": student_t_grid,
            },
        )
        scenario_results[scenario] = {
            "quadrature_sensitivity": {
                "summaries": summaries_by_points,
                "comparisons": comparisons,
            },
            "residual_calibration": residual_diagnostic,
            "configuration_grid": configuration_grid,
            "student_t_configuration_grid": student_t_grid,
        }
        print(
            f"{scenario} mean NIS={residual_diagnostic['mean_nis']:.4f}, "
            f"implied inflation="
            f"{residual_diagnostic['nis_implied_scalar_inflation']:.4f}"
        )

    combined_scenarios = dict(scenario_results)
    for scenario in SCENARIOS:
        if scenario in combined_scenarios:
            continue
        quadrature_path = (
            VALIDATION_DIAGNOSTIC_ROOT
            / scenario
            / "quadrature_sensitivity.json"
        )
        residual_path = (
            VALIDATION_DIAGNOSTIC_ROOT
            / scenario
            / "residual_calibration.json"
        )
        student_t_path = (
            VALIDATION_DIAGNOSTIC_ROOT
            / scenario
            / "student_t_configuration_grid.json"
        )
        if quadrature_path.exists() and residual_path.exists() and student_t_path.exists():
            with quadrature_path.open("r", encoding="utf-8") as handle:
                quadrature_payload = json.load(handle)
            with residual_path.open("r", encoding="utf-8") as handle:
                residual_payload = json.load(handle)
            with student_t_path.open("r", encoding="utf-8") as handle:
                student_t_payload = json.load(handle)
            combined_scenarios[scenario] = {
                "quadrature_sensitivity": {
                    "summaries": quadrature_payload["summaries"],
                    "comparisons": quadrature_payload["comparisons"],
                },
                "residual_calibration": residual_payload,
                "student_t_configuration_grid": student_t_payload[
                    "candidate_results"
                ],
            }
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "data_role": "validation",
        "formal_configuration_changed": False,
        "scenario_results": combined_scenarios,
    }
    output_path = VALIDATION_DIAGNOSTIC_ROOT / "combined_diagnostics.json"
    write_json(output_path, combined)
    print(f"Validation diagnostics: {output_path.resolve()}")
    return combined


def add_condition_balanced_metrics(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """补充等权 hidden-condition terminal NLL 与 coverage。"""

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition_id"]].append(row)
    condition_mean_nll = [
        float(np.mean([row["posterior_nll"] for row in condition_rows]))
        for condition_rows in by_condition.values()
    ]
    condition_coverage = [
        float(np.mean([row["hpd_covered"] for row in condition_rows]))
        for condition_rows in by_condition.values()
    ]
    summary["condition_balanced_mean_terminal_nll"] = float(
        np.mean(condition_mean_nll)
    )
    summary["condition_balanced_terminal_hpd_coverage"] = float(
        np.mean(condition_coverage)
    )
    summary["minimum_condition_terminal_hpd_coverage"] = float(
        np.min(condition_coverage)
    )
    summary["maximum_condition_terminal_hpd_coverage"] = float(
        np.max(condition_coverage)
    )
    return summary


def evaluate_full_update_horizon_grid(
    likelihood: GaussianOutcomeLikelihood,
    episodes: list[list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """一次回放同时评价 3、4 和完整 sequential update horizons。"""

    rows_by_horizon: dict[str, list[dict[str, Any]]] = {
        "3": [],
        "4": [],
        "full": [],
    }
    for episode_rows in episodes:
        posterior = FixedNodePosterior(likelihood.rule)
        first_row = episode_rows[0]
        true_coordinates = true_normalised_coordinates(first_row, likelihood)
        cumulative_true_log_likelihood = 0.0
        requested_indices = {
            "3": min(3, len(episode_rows)),
            "4": min(4, len(episode_rows)),
            "full": len(episode_rows),
        }
        snapshots: dict[int, dict[str, Any]] = {}

        for update_index, row in enumerate(episode_rows, start=1):
            observation = history_observation(row)
            action_id = row["v2_action_id"]
            posterior.update(
                likelihood.node_log_likelihoods(action_id, observation)
            )
            cumulative_true_log_likelihood += likelihood.true_log_likelihood(
                action_id,
                observation,
                true_coordinates,
            )
            if update_index in requested_indices.values():
                posterior_summary = posterior.summary()
                probabilistic = posterior.probabilistic_metrics(
                    cumulative_true_log_likelihood
                )
                point_metrics = parameter_error_metrics(
                    likelihood.scenario,
                    posterior_summary.mean_normalised,
                    true_coordinates,
                )
                snapshots[update_index] = {
                    "used_update_count": update_index,
                    "posterior_nll": float(probabilistic["posterior_nll"]),
                    "hpd_covered": int(probabilistic["hpd_covered"]),
                    "uncertainty_contraction": float(
                        probabilistic["uncertainty_contraction"]
                    ),
                    "posterior_covariance_trace": float(
                        np.trace(posterior_summary.covariance_normalised)
                    ),
                    "friction_absolute_error": point_metrics.get(
                        "friction_absolute_error",
                        None,
                    ),
                    "com_euclidean_error_mm": point_metrics.get(
                        "com_euclidean_error_mm",
                        None,
                    ),
                }

        for horizon, snapshot_index in requested_indices.items():
            rows_by_horizon[horizon].append(
                {
                    "scenario": likelihood.scenario,
                    "condition_id": first_row["condition_id"],
                    "target_id": first_row["target_id"],
                    "episode_key": first_row["episode_key"],
                    "episode_update_count": len(episode_rows),
                    "update_horizon": horizon,
                    "points_per_dimension": (
                        likelihood.rule.points_per_dimension
                    ),
                    "node_count": likelihood.rule.node_count,
                    **snapshots[snapshot_index],
                }
            )

    results: list[dict[str, Any]] = []
    for horizon, rows in rows_by_horizon.items():
        summary = summarise_quadrature_diagnostic(rows)
        add_condition_balanced_metrics(rows, summary)
        summary["covariance_inflation"] = likelihood.covariance_inflation
        results.append(summary)
    return results


def calibrate_validation(args: argparse.Namespace) -> dict[str, Any]:
    """在完整 Validation histories 上评价 calibration 与 update horizon。"""

    scenario_results: dict[str, Any] = {}
    for scenario in select_scenarios(args.scenario):
        all_episodes = load_validation_episodes(scenario)
        valid_episodes = [
            rows for rows in all_episodes if int(rows[-1]["episode_valid"]) == 1
        ]
        likelihood = GaussianOutcomeLikelihood.from_project_artifacts(
            scenario,
            covariance_inflation=FULL_CALIBRATION_INFLATION_CANDIDATES[0],
            points_per_dimension=FULL_CALIBRATION_QUADRATURE_POINTS,
        )
        candidate_results: list[dict[str, Any]] = []
        for inflation in FULL_CALIBRATION_INFLATION_CANDIDATES:
            likelihood.set_covariance_inflation(inflation)
            inflation_results = evaluate_full_update_horizon_grid(
                likelihood,
                valid_episodes,
            )
            candidate_results.extend(inflation_results)
            for result in inflation_results:
                print(
                    f"{scenario} points=17 inflation={inflation:g} "
                    f"horizon={result['update_horizon']}: "
                    f"coverage={result['terminal_hpd_coverage']:.6f}, "
                    f"nll={result['mean_terminal_nll']:.6f}"
                )

        scenario_payload = {
            "scenario": scenario,
            "episodes": len(valid_episodes),
            "points_per_dimension": FULL_CALIBRATION_QUADRATURE_POINTS,
            "inflation_candidates": list(
                FULL_CALIBRATION_INFLATION_CANDIDATES
            ),
            "update_horizon_candidates": [3, 4, "full"],
            "candidate_results": candidate_results,
        }
        scenario_path = (
            VALIDATION_DIAGNOSTIC_ROOT
            / scenario
            / "full_configuration_grid.json"
        )
        write_json(scenario_path, scenario_payload)
        scenario_results[scenario] = scenario_payload

    combined_scenarios = dict(scenario_results)
    for scenario in SCENARIOS:
        if scenario in combined_scenarios:
            continue
        scenario_path = (
            VALIDATION_DIAGNOSTIC_ROOT
            / scenario
            / "full_configuration_grid.json"
        )
        if scenario_path.exists():
            with scenario_path.open("r", encoding="utf-8") as handle:
                combined_scenarios[scenario] = json.load(handle)
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "data_role": "validation",
        "formal_configuration_changed": False,
        "scenario_results": combined_scenarios,
    }
    output_path = VALIDATION_DIAGNOSTIC_ROOT / "full_configuration_grid.json"
    write_json(output_path, combined)
    print(f"Full Validation calibration grid: {output_path.resolve()}")
    return combined


def calibrate_student_t_validation(args: argparse.Namespace) -> dict[str, Any]:
    """在完整 Validation histories 上评价稳健 Student-t likelihood。"""

    scenario_results: dict[str, Any] = {}
    for scenario in select_scenarios(args.scenario):
        degrees, inflation_candidates = STUDENT_T_FULL_CANDIDATES[scenario]
        all_episodes = load_validation_episodes(scenario)
        valid_episodes = [
            rows for rows in all_episodes if int(rows[-1]["episode_valid"]) == 1
        ]
        likelihood = StudentTOutcomeLikelihood.from_project_artifacts(
            scenario,
            covariance_inflation=inflation_candidates[0],
            points_per_dimension=17,
            degrees_of_freedom=degrees,
        )
        candidate_results: list[dict[str, Any]] = []
        for inflation in inflation_candidates:
            likelihood.set_covariance_inflation(inflation)
            inflation_results = evaluate_full_update_horizon_grid(
                likelihood,
                valid_episodes,
            )
            for result in inflation_results:
                result["likelihood_family"] = "student_t"
                result["degrees_of_freedom"] = degrees
            candidate_results.extend(inflation_results)
            for result in inflation_results:
                print(
                    f"{scenario} Student-t df={degrees:g} "
                    f"inflation={inflation:g} horizon={result['update_horizon']}: "
                    f"coverage={result['terminal_hpd_coverage']:.6f}, "
                    f"nll={result['mean_terminal_nll']:.6f}"
                )

        scenario_payload = {
            "scenario": scenario,
            "episodes": len(valid_episodes),
            "points_per_dimension": 17,
            "likelihood_family": "student_t",
            "degrees_of_freedom": degrees,
            "inflation_candidates": list(inflation_candidates),
            "update_horizon_candidates": [3, 4, "full"],
            "candidate_results": candidate_results,
        }
        output_path = (
            VALIDATION_DIAGNOSTIC_ROOT
            / scenario
            / "student_t_full_configuration_grid.json"
        )
        write_json(output_path, scenario_payload)
        scenario_results[scenario] = scenario_payload

    combined_scenarios = dict(scenario_results)
    for scenario in SCENARIOS:
        if scenario in combined_scenarios:
            continue
        scenario_path = (
            VALIDATION_DIAGNOSTIC_ROOT
            / scenario
            / "student_t_full_configuration_grid.json"
        )
        if scenario_path.exists():
            with scenario_path.open("r", encoding="utf-8") as handle:
                combined_scenarios[scenario] = json.load(handle)
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "data_role": "validation",
        "formal_configuration_changed": False,
        "scenario_results": combined_scenarios,
    }
    output_path = (
        VALIDATION_DIAGNOSTIC_ROOT
        / "student_t_full_configuration_grid.json"
    )
    write_json(output_path, combined)
    print(f"Student-t Validation calibration grid: {output_path.resolve()}")
    return combined


def evaluate_inflation_candidate(
    scenario: str,
    episodes: list[list[dict[str, str]]],
    covariance_inflation: float,
) -> dict[str, Any]:
    """计算一个 covariance inflation 的 Sequential Validation NLL。"""

    started = time.perf_counter()
    likelihood = GaussianOutcomeLikelihood.from_project_artifacts(
        scenario,
        covariance_inflation=covariance_inflation,
        points_per_dimension=PRIMARY_QUADRATURE_POINTS,
    )
    update_nll_values: list[float] = []
    terminal_nll_values: list[float] = []
    terminal_covered: list[int] = []

    for episode_rows in episodes:
        posterior = FixedNodePosterior(likelihood.rule)
        true_coordinates = true_normalised_coordinates(
            episode_rows[0],
            likelihood,
        )
        cumulative_true_log_likelihood = 0.0
        for row in episode_rows:
            observation = history_observation(row)
            posterior.update(
                likelihood.node_log_likelihoods(
                    row["v2_action_id"],
                    observation,
                )
            )
            cumulative_true_log_likelihood += likelihood.true_log_likelihood(
                row["v2_action_id"],
                observation,
                true_coordinates,
            )
            update_nll_values.append(
                -posterior.continuous_log_density(
                    cumulative_true_log_likelihood
                )
            )

        terminal_metrics = posterior.probabilistic_metrics(
            cumulative_true_log_likelihood
        )
        terminal_nll_values.append(float(terminal_metrics["posterior_nll"]))
        terminal_covered.append(int(terminal_metrics["hpd_covered"]))

    return {
        "covariance_inflation": float(covariance_inflation),
        "points_per_dimension": PRIMARY_QUADRATURE_POINTS,
        "node_count": likelihood.rule.node_count,
        "episodes": len(episodes),
        "valid_updates": len(update_nll_values),
        "mean_sequential_posterior_nll_all_updates": float(
            np.mean(update_nll_values)
        ),
        "mean_terminal_sequential_posterior_nll": float(
            np.mean(terminal_nll_values)
        ),
        "terminal_hpd_coverage": float(np.mean(terminal_covered)),
        "runtime_seconds": time.perf_counter() - started,
    }


def select_covariance_inflation(
    scenario: str,
    episodes: list[list[dict[str, str]]],
) -> dict[str, Any]:
    """按预声明的平均 update-level posterior NLL 选择 inflation。"""

    candidate_results: list[dict[str, Any]] = []
    for inflation in INFLATION_CANDIDATES:
        result = evaluate_inflation_candidate(
            scenario,
            episodes,
            inflation,
        )
        candidate_results.append(result)
        print(
            f"{scenario} inflation={inflation:g}: "
            f"mean_update_nll="
            f"{result['mean_sequential_posterior_nll_all_updates']:.6f}"
        )

    selected = min(
        candidate_results,
        key=lambda row: (
            row["mean_sequential_posterior_nll_all_updates"],
            row["covariance_inflation"],
        ),
    )
    return {
        "selection_metric": "mean_sequential_posterior_nll_all_updates",
        "candidate_values": list(INFLATION_CANDIDATES),
        "candidate_results": candidate_results,
        "selected_covariance_inflation": selected["covariance_inflation"],
    }


def summarise_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总同一 estimator 的 terminal 或 update-level metrics。"""

    summary: dict[str, Any] = {"rows": len(rows)}
    for field in (
        "posterior_nll",
        "uncertainty_contraction",
        "friction_absolute_error",
        "com_euclidean_error_mm",
    ):
        values = [float(row[field]) for row in rows if row[field] != ""]
        if values:
            summary[f"mean_{field}"] = float(np.mean(values))
            summary[f"median_{field}"] = float(np.median(values))
    coverage = [int(row["hpd_covered"]) for row in rows]
    if coverage:
        empirical_coverage = float(np.mean(coverage))
        summary["hpd_coverage"] = empirical_coverage
        summary["coverage_deviation"] = empirical_coverage - 0.95
    return summary


def paired_terminal_effect(
    terminal_rows: list[dict[str, Any]],
    baseline: str,
    minimum_updates: int,
) -> dict[str, Any]:
    """计算 Sequential 相对于一个 baseline 的 terminal paired effect。"""

    by_episode: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in terminal_rows:
        if int(row["terminal_update_count"]) >= minimum_updates:
            by_episode[row["episode_key"]][row["estimator"]] = row

    paired = [
        methods
        for methods in by_episode.values()
        if baseline in methods and "sequential_bayesian" in methods
    ]
    result: dict[str, Any] = {
        "baseline": baseline,
        "minimum_updates": minimum_updates,
        "paired_episodes": len(paired),
    }
    for field in (
        "posterior_nll",
        "friction_absolute_error",
        "com_euclidean_error_mm",
    ):
        differences = [
            float(methods["sequential_bayesian"][field])
            - float(methods[baseline][field])
            for methods in paired
            if methods["sequential_bayesian"][field] != ""
            and methods[baseline][field] != ""
        ]
        if differences:
            result[field] = {
                "mean_sequential_minus_baseline": float(np.mean(differences)),
                "median_sequential_minus_baseline": float(
                    np.median(differences)
                ),
                "sequential_better_rate": float(
                    np.mean(np.asarray(differences) < 0.0)
                ),
            }
    return result


def build_terminal_metric_matrices(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    """将 terminal rows 组织为 condition-target crossed matrices。"""

    condition_ids = sorted({row["condition_id"] for row in rows})
    target_metadata: dict[str, tuple[str, str]] = {}
    for row in rows:
        target_metadata[row["target_id"]] = (
            row["target_group"],
            row["target_stratum"],
        )

    targets_by_stratum: dict[str, list[str]] = defaultdict(list)
    for target_id, (target_group, target_stratum) in target_metadata.items():
        targets_by_stratum[
            f"{target_group}::{target_stratum}"
        ].append(target_id)
    target_ids: list[str] = []
    target_indices_by_stratum: dict[str, list[int]] = {}
    for stratum in sorted(targets_by_stratum):
        stratum_targets = sorted(targets_by_stratum[stratum])
        start = len(target_ids)
        target_ids.extend(stratum_targets)
        target_indices_by_stratum[stratum] = list(
            range(start, start + len(stratum_targets))
        )

    condition_index = {
        condition_id: index
        for index, condition_id in enumerate(condition_ids)
    }
    target_index = {
        target_id: index for index, target_id in enumerate(target_ids)
    }
    metrics = (
        "posterior_nll",
        "hpd_covered",
        "uncertainty_contraction",
        "friction_absolute_error",
        "com_euclidean_error_mm",
    )
    matrices = {
        (estimator, metric): np.full(
            (len(condition_ids), len(target_ids)),
            np.nan,
            dtype=np.float64,
        )
        for estimator in ESTIMATOR_NAMES
        for metric in metrics
    }
    update_counts = np.full(
        (len(condition_ids), len(target_ids)),
        np.nan,
        dtype=np.float64,
    )

    for row in rows:
        row_index = condition_index[row["condition_id"]]
        column_index = target_index[row["target_id"]]
        estimator = row["estimator"]
        for metric in metrics:
            value = row[metric]
            if value != "":
                matrices[(estimator, metric)][row_index, column_index] = float(
                    value
                )
        if estimator == "sequential_bayesian":
            update_counts[row_index, column_index] = float(
                row["terminal_update_count"]
            )

    return {
        "condition_ids": condition_ids,
        "target_ids": target_ids,
        "target_indices_by_stratum": target_indices_by_stratum,
        "matrices": matrices,
        "update_counts": update_counts,
    }


def make_stratified_bootstrap_weights(
    condition_count: int,
    target_count: int,
    target_indices_by_stratum: dict[str, list[int]],
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 condition 与分层 target 的 two-way bootstrap 权重。"""

    rng = np.random.default_rng(seed)
    condition_weights = rng.multinomial(
        condition_count,
        np.full(condition_count, 1.0 / condition_count),
        size=n_resamples,
    ).astype(np.float64)
    target_weights = np.zeros(
        (n_resamples, target_count),
        dtype=np.float64,
    )
    for indices in target_indices_by_stratum.values():
        stratum_size = len(indices)
        target_weights[:, indices] = rng.multinomial(
            stratum_size,
            np.full(stratum_size, 1.0 / stratum_size),
            size=n_resamples,
        )
    return condition_weights, target_weights


def bootstrap_matrix_means(
    matrices: dict[str, np.ndarray],
    condition_weights: np.ndarray,
    target_weights: np.ndarray,
) -> dict[str, dict[str, float]]:
    """使用同一组 two-way 权重计算多个 matrix mean 的区间。"""

    names = list(matrices)
    values = np.stack([matrices[name] for name in names], axis=0)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    finite_float = finite.astype(np.float64)
    n_resamples = condition_weights.shape[0]
    estimates = np.full(
        (n_resamples, len(names)),
        np.nan,
        dtype=np.float64,
    )
    batch_size = 256
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        current_condition_weights = condition_weights[start:stop]
        current_target_weights = target_weights[start:stop]
        weighted_values = np.einsum(
            "bi,mij->bmj",
            current_condition_weights,
            filled,
            optimize=True,
        )
        weighted_counts = np.einsum(
            "bi,mij->bmj",
            current_condition_weights,
            finite_float,
            optimize=True,
        )
        numerators = np.einsum(
            "bmj,bj->bm",
            weighted_values,
            current_target_weights,
            optimize=True,
        )
        denominators = np.einsum(
            "bmj,bj->bm",
            weighted_counts,
            current_target_weights,
            optimize=True,
        )
        estimates[start:stop] = np.divide(
            numerators,
            denominators,
            out=np.full_like(numerators, np.nan),
            where=denominators > 0.0,
        )

    summaries: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        distribution = estimates[:, index]
        summaries[name] = {
            "point_estimate": float(np.nanmean(values[index])),
            "ci95_low": float(np.nanquantile(distribution, 0.025)),
            "ci95_high": float(np.nanquantile(distribution, 0.975)),
        }
    return summaries


def bootstrap_result_scenario(
    scenario: str,
    data_role: str,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """生成一个场景的 terminal bootstrap summaries 与正式结果表。"""

    result_root = (
        VALIDATION_RESULT_ROOT if data_role == "validation" else TEST_RESULT_ROOT
    )
    terminal_path = result_root / scenario / "terminal_metrics.csv"
    rows = read_csv_rows(terminal_path)
    bundle = build_terminal_metric_matrices(rows)
    matrices: dict[tuple[str, str], np.ndarray] = bundle["matrices"]
    update_counts: np.ndarray = bundle["update_counts"]
    condition_weights, target_weights = make_stratified_bootstrap_weights(
        len(bundle["condition_ids"]),
        len(bundle["target_ids"]),
        bundle["target_indices_by_stratum"],
        n_resamples,
        seed,
    )

    statistic_matrices: dict[str, np.ndarray] = {}
    method_keys: dict[tuple[str, str], str] = {}
    for estimator in ESTIMATOR_NAMES:
        for metric in (
            "posterior_nll",
            "hpd_covered",
            "uncertainty_contraction",
            "friction_absolute_error",
            "com_euclidean_error_mm",
        ):
            matrix = matrices[(estimator, metric)]
            if not np.isfinite(matrix).any():
                continue
            key = f"method::{estimator}::{metric}"
            statistic_matrices[key] = matrix
            method_keys[(estimator, metric)] = key

    comparison_definitions = (
        ("h1_sequential_vs_prior", "prior_only", 1),
        ("h2_sequential_vs_single", "single_observation", 2),
    )
    comparison_keys: dict[tuple[str, str], dict[str, str]] = {}
    for hypothesis, baseline, minimum_updates in comparison_definitions:
        eligible = update_counts >= minimum_updates
        for metric in (
            "posterior_nll",
            "friction_absolute_error",
            "com_euclidean_error_mm",
        ):
            sequential = matrices[("sequential_bayesian", metric)]
            baseline_values = matrices[(baseline, metric)]
            valid = eligible & np.isfinite(sequential) & np.isfinite(
                baseline_values
            )
            if not valid.any():
                continue
            sequential_subset = np.where(valid, sequential, np.nan)
            baseline_subset = np.where(valid, baseline_values, np.nan)
            difference = sequential_subset - baseline_subset
            better = np.where(valid, difference < 0.0, np.nan)
            prefix = f"comparison::{hypothesis}::{metric}"
            keys = {
                "sequential": f"{prefix}::sequential",
                "baseline": f"{prefix}::baseline",
                "difference": f"{prefix}::difference",
                "better": f"{prefix}::better",
            }
            statistic_matrices[keys["sequential"]] = sequential_subset
            statistic_matrices[keys["baseline"]] = baseline_subset
            statistic_matrices[keys["difference"]] = difference
            statistic_matrices[keys["better"]] = better
            comparison_keys[(hypothesis, metric)] = {
                **keys,
                "baseline_name": baseline,
                "minimum_updates": str(minimum_updates),
                "paired_episodes": str(int(np.count_nonzero(valid))),
            }

    bootstrap = bootstrap_matrix_means(
        statistic_matrices,
        condition_weights,
        target_weights,
    )

    method_rows: list[dict[str, Any]] = []
    for (estimator, metric), key in method_keys.items():
        result = bootstrap[key]
        method_rows.append(
            {
                "scenario": scenario,
                "estimator": estimator,
                "subset": "all_valid_episodes",
                "metric": metric,
                "episodes": int(
                    np.count_nonzero(
                        np.isfinite(matrices[(estimator, metric)])
                    )
                ),
                **result,
                "coverage_deviation": (
                    result["point_estimate"] - 0.95
                    if metric == "hpd_covered"
                    else ""
                ),
                "bootstrap_resamples": n_resamples,
                "bootstrap_seed": seed,
            }
        )

    comparison_rows: list[dict[str, Any]] = []
    for (hypothesis, metric), keys in comparison_keys.items():
        baseline_result = bootstrap[keys["baseline"]]
        sequential_result = bootstrap[keys["sequential"]]
        effect_result = bootstrap[keys["difference"]]
        better_result = bootstrap[keys["better"]]
        comparison_rows.append(
            {
                "scenario": scenario,
                "hypothesis": hypothesis,
                "baseline": keys["baseline_name"],
                "minimum_updates": int(keys["minimum_updates"]),
                "metric": metric,
                "paired_episodes": int(keys["paired_episodes"]),
                "baseline_mean": baseline_result["point_estimate"],
                "baseline_ci95_low": baseline_result["ci95_low"],
                "baseline_ci95_high": baseline_result["ci95_high"],
                "sequential_mean": sequential_result["point_estimate"],
                "sequential_ci95_low": sequential_result["ci95_low"],
                "sequential_ci95_high": sequential_result["ci95_high"],
                "mean_sequential_minus_baseline": effect_result[
                    "point_estimate"
                ],
                "effect_ci95_low": effect_result["ci95_low"],
                "effect_ci95_high": effect_result["ci95_high"],
                "sequential_better_rate": better_result["point_estimate"],
                "better_rate_ci95_low": better_result["ci95_low"],
                "better_rate_ci95_high": better_result["ci95_high"],
                "supports_lower_sequential": int(
                    effect_result["ci95_high"] < 0.0
                ),
                "bootstrap_resamples": n_resamples,
                "bootstrap_seed": seed,
            }
        )

    output_dir = result_root / scenario
    method_path = output_dir / "terminal_method_bootstrap.csv"
    comparison_path = output_dir / "primary_paired_bootstrap.csv"
    summary_path = output_dir / "bootstrap_summary.json"
    write_csv(method_path, method_rows, BOOTSTRAP_METHOD_FIELDS)
    write_csv(
        comparison_path,
        comparison_rows,
        BOOTSTRAP_COMPARISON_FIELDS,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "data_role": data_role,
        "scenario": scenario,
        "bootstrap_method": "stratified_two_way_paired_bootstrap",
        "condition_count": len(bundle["condition_ids"]),
        "target_count": len(bundle["target_ids"]),
        "target_strata": {
            stratum: len(indices)
            for stratum, indices in bundle[
                "target_indices_by_stratum"
            ].items()
        },
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": seed,
        "terminal_method_summary": method_rows,
        "primary_paired_comparisons": comparison_rows,
        "terminal_method_table_path": str(method_path.resolve()),
        "primary_paired_table_path": str(comparison_path.resolve()),
    }
    write_json(summary_path, payload)
    payload["summary_path"] = str(summary_path.resolve())
    return payload


def bootstrap_validation(args: argparse.Namespace) -> dict[str, Any]:
    """生成三场景 Validation bootstrap 结果。"""

    scenario_results: dict[str, dict[str, Any]] = {}
    for scenario in select_scenarios(args.scenario):
        scenario_seed = args.bootstrap_seed + SCENARIOS.index(scenario)
        scenario_result = bootstrap_result_scenario(
            scenario,
            "validation",
            args.bootstrap_resamples,
            scenario_seed,
        )
        scenario_results[scenario] = scenario_result
        print(
            f"{scenario}: bootstrap={args.bootstrap_resamples}, "
            f"summary={scenario_result['summary_path']}"
        )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "data_role": "validation",
        "bootstrap_method": "stratified_two_way_paired_bootstrap",
        "bootstrap_resamples": args.bootstrap_resamples,
        "base_bootstrap_seed": args.bootstrap_seed,
        "scenario_results": scenario_results,
    }
    combined_path = VALIDATION_RESULT_ROOT / "bootstrap_combined_summary.json"
    write_json(combined_path, payload)
    print(f"Combined bootstrap summary: {combined_path.resolve()}")
    return payload


def build_update_curves(
    update_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 estimator 与实际 update index 汇总 active-history curves。"""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in update_rows:
        grouped[(row["estimator"], int(row["update_index"]))].append(row)

    curves: list[dict[str, Any]] = []
    for (estimator, update_index), rows in sorted(grouped.items()):
        curves.append(
            {
                "estimator": estimator,
                "update_index": update_index,
                "active_history_count": len(rows),
                **summarise_metric_rows(rows),
            }
        )
    return curves


def build_transfer_diagnostics(
    likelihood: GaussianOutcomeLikelihood,
    residuals_standardised: list[np.ndarray],
) -> dict[str, Any]:
    """比较 continuous Validation 与 Training residual statistics。"""

    residuals = np.asarray(residuals_standardised, dtype=np.float64)
    validation_bias = residuals.mean(axis=0)
    centered = residuals - validation_bias
    validation_covariance = centered.T @ centered / (len(centered) - 1)
    validation_std = np.sqrt(np.diag(validation_covariance))
    validation_correlation = validation_covariance / np.outer(
        validation_std,
        validation_std,
    )

    statistics = likelihood.residual_statistics
    training_std = np.sqrt(np.diag(statistics.base_covariance))
    training_correlation = statistics.base_covariance / np.outer(
        training_std,
        training_std,
    )
    physical_bias = validation_bias * statistics.observation_scale
    physical_std = validation_std * statistics.observation_scale
    return {
        "sample_count": len(residuals),
        "validation_bias_standardised": [
            float(value) for value in validation_bias
        ],
        "training_bias_standardised": [
            float(value) for value in statistics.residual_bias
        ],
        "bias_shift_standardised": [
            float(value)
            for value in validation_bias - statistics.residual_bias
        ],
        "validation_bias_physical": {
            "delta_x_mm": float(physical_bias[0] * 1000.0),
            "delta_y_mm": float(physical_bias[1] * 1000.0),
            "delta_yaw_deg": float(np.degrees(physical_bias[2])),
        },
        "validation_centered_std_physical": {
            "delta_x_mm": float(physical_std[0] * 1000.0),
            "delta_y_mm": float(physical_std[1] * 1000.0),
            "delta_yaw_deg": float(np.degrees(physical_std[2])),
        },
        "validation_to_training_std_ratio": [
            float(value) for value in validation_std / training_std
        ],
        "validation_correlation": matrix_to_list(validation_correlation),
        "training_correlation": matrix_to_list(training_correlation),
        "maximum_absolute_correlation_change": float(
            np.max(np.abs(validation_correlation - training_correlation))
        ),
    }


def evaluate_selected_configuration(
    scenario: str,
    episodes: list[list[dict[str, str]]],
    covariance_inflation: float,
) -> dict[str, Any]:
    """使用校准后的 Student-t likelihood 回放三套 paired estimators。"""

    started = time.perf_counter()
    likelihood = StudentTOutcomeLikelihood.from_project_artifacts(
        scenario,
        covariance_inflation=covariance_inflation,
        points_per_dimension=FINAL_QUADRATURE_POINTS,
        degrees_of_freedom=FINAL_STUDENT_T_DEGREES_OF_FREEDOM,
    )
    suite = PairedEstimatorSuite(likelihood.rule)
    update_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    residuals_standardised: list[np.ndarray] = []

    for episode_rows in episodes:
        suite.reset()
        first_row = episode_rows[0]
        last_row = episode_rows[-1]
        used_episode_rows = episode_rows[:FINAL_BELIEF_UPDATE_HORIZON]
        true_coordinates = true_normalised_coordinates(first_row, likelihood)
        true_condition_row = condition_row_from_normalised(
            scenario,
            true_coordinates,
        )
        cumulative_true_log_likelihood = 0.0

        for update_index, row in enumerate(used_episode_rows, start=1):
            observation = history_observation(row)
            action_id = row["v2_action_id"]
            node_log_likelihoods = likelihood.node_log_likelihoods(
                action_id,
                observation,
            )
            summaries = suite.update_from_log_likelihoods(
                node_log_likelihoods
            )
            current_true_log_likelihood = likelihood.true_log_likelihood(
                action_id,
                observation,
                true_coordinates,
            )
            cumulative_true_log_likelihood += current_true_log_likelihood

            true_prediction = likelihood.p1.predict_action(
                true_condition_row,
                action_id,
            ).astype(np.float64)
            residuals_standardised.append(
                normalise_observation(observation)
                - true_prediction / OBSERVATION_SCALE
            )

            true_log_likelihoods = {
                "prior_only": 0.0,
                "single_observation": current_true_log_likelihood,
                "sequential_bayesian": cumulative_true_log_likelihood,
            }
            is_terminal_update = int(
                update_index == len(used_episode_rows)
            )
            for estimator in ESTIMATOR_NAMES:
                posterior = suite.posteriors[estimator]
                probabilistic = posterior.probabilistic_metrics(
                    true_log_likelihoods[estimator]
                )
                point_metrics = parameter_error_metrics(
                    scenario,
                    summaries[estimator].mean_normalised,
                    true_coordinates,
                )
                evaluation_row: dict[str, Any] = {
                    "protocol_version": PROTOCOL_VERSION,
                    "friction_cone": FRICTION_CONE,
                    "scenario": scenario,
                    "condition_id": first_row["condition_id"],
                    "target_id": first_row["target_id"],
                    "target_group": first_row["target_group"],
                    "target_stratum": first_row["target_stratum"],
                    "episode_key": first_row["episode_key"],
                    "episode_success": int(last_row["episode_success"]),
                    "terminal_reason": last_row["terminal_reason"],
                    "update_index": update_index,
                    "episode_update_count": len(episode_rows),
                    "terminal_update_count": len(used_episode_rows),
                    "is_terminal_update": is_terminal_update,
                    "estimator": estimator,
                    "likelihood_family": FINAL_LIKELIHOOD_FAMILY,
                    "student_t_degrees_of_freedom": (
                        FINAL_STUDENT_T_DEGREES_OF_FREEDOM
                    ),
                    "covariance_inflation": covariance_inflation,
                    "belief_update_horizon": FINAL_BELIEF_UPDATE_HORIZON,
                    "points_per_dimension": FINAL_QUADRATURE_POINTS,
                    "node_count": likelihood.rule.node_count,
                    "posterior_nll": probabilistic["posterior_nll"],
                    "hpd_covered": probabilistic["hpd_covered"],
                    "true_log_density": probabilistic["true_log_density"],
                    "hpd_log_density_threshold": probabilistic[
                        "hpd_log_density_threshold"
                    ],
                    "uncertainty_contraction": probabilistic[
                        "uncertainty_contraction"
                    ],
                    "friction_absolute_error": point_metrics.get(
                        "friction_absolute_error",
                        "",
                    ),
                    "com_euclidean_error_mm": point_metrics.get(
                        "com_euclidean_error_mm",
                        "",
                    ),
                }
                update_rows.append(evaluation_row)
                if is_terminal_update:
                    terminal_rows.append(evaluation_row.copy())

    terminal_summary = {
        estimator: summarise_metric_rows(
            [row for row in terminal_rows if row["estimator"] == estimator]
        )
        for estimator in ESTIMATOR_NAMES
    }
    elapsed_seconds = time.perf_counter() - started
    return {
        "update_rows": update_rows,
        "terminal_rows": terminal_rows,
        "terminal_summary": terminal_summary,
        "update_curves": build_update_curves(update_rows),
        "paired_effects": {
            "h1_sequential_vs_prior_all_valid_episodes": (
                paired_terminal_effect(terminal_rows, "prior_only", 1)
            ),
            "h2_sequential_vs_single_multi_update_episodes": (
                paired_terminal_effect(
                    terminal_rows,
                    "single_observation",
                    2,
                )
            ),
        },
        "transfer_diagnostics": build_transfer_diagnostics(
            likelihood,
            residuals_standardised,
        ),
        "runtime": {
            "seconds": elapsed_seconds,
            "evaluated_updates": len(residuals_standardised),
            "mean_seconds_per_update": (
                elapsed_seconds / len(residuals_standardised)
            ),
            "node_mean_cache_mib": (
                likelihood.node_means_standardised.nbytes / (1024.0**2)
            ),
        },
    }


def select_calibrated_student_t_configuration(
    scenario: str,
) -> dict[str, Any]:
    """按 coverage band 与 terminal NLL 选择场景级 inflation。"""

    grid_path = (
        VALIDATION_DIAGNOSTIC_ROOT
        / scenario
        / "student_t_full_configuration_grid.json"
    )
    if not grid_path.exists():
        raise FileNotFoundError(
            f"缺少完整 Student-t calibration grid: {grid_path}"
        )
    with grid_path.open("r", encoding="utf-8") as handle:
        grid = json.load(handle)
    candidates = [
        row
        for row in grid["candidate_results"]
        if str(row["update_horizon"]) == str(FINAL_BELIEF_UPDATE_HORIZON)
        and int(row["points_per_dimension"]) == FINAL_QUADRATURE_POINTS
        and float(row["degrees_of_freedom"])
        == FINAL_STUDENT_T_DEGREES_OF_FREEDOM
    ]
    calibrated = [
        row
        for row in candidates
        if CALIBRATION_COVERAGE_LOWER
        <= float(row["terminal_hpd_coverage"])
        <= CALIBRATION_COVERAGE_UPPER
    ]
    if not calibrated:
        raise RuntimeError(
            f"{scenario} 没有满足 coverage band 的 Student-t configuration"
        )
    selected = min(
        calibrated,
        key=lambda row: (
            float(row["condition_balanced_mean_terminal_nll"]),
            float(row["covariance_inflation"]),
        ),
    )
    selected_inflation = float(selected["covariance_inflation"])
    update_horizon_ablation = [
        row
        for row in grid["candidate_results"]
        if float(row["covariance_inflation"]) == selected_inflation
        and int(row["points_per_dimension"]) == FINAL_QUADRATURE_POINTS
        and float(row["degrees_of_freedom"])
        == FINAL_STUDENT_T_DEGREES_OF_FREEDOM
    ]
    return {
        "selection_metric": (
            "condition_balanced_mean_terminal_nll_within_93_to_97_percent_coverage"
        ),
        "coverage_band": [
            CALIBRATION_COVERAGE_LOWER,
            CALIBRATION_COVERAGE_UPPER,
        ],
        "candidate_values": [
            float(row["covariance_inflation"]) for row in candidates
        ],
        "candidate_results": candidates,
        "selected_covariance_inflation": selected_inflation,
        "selected_validation_coverage": float(
            selected["terminal_hpd_coverage"]
        ),
        "selected_condition_balanced_mean_terminal_nll": float(
            selected["condition_balanced_mean_terminal_nll"]
        ),
        "update_horizon_ablation": update_horizon_ablation,
        "source_calibration_grid": str(grid_path.resolve()),
    }


def evaluate_validation(args: argparse.Namespace) -> dict[str, Any]:
    """固定校准配置并写出 paired estimator Validation evaluation。"""

    scenario_summaries: dict[str, dict[str, Any]] = {}
    selected_inflations: dict[str, float] = {}
    for scenario in select_scenarios(args.scenario):
        all_episodes = load_validation_episodes(scenario)
        valid_episodes = [
            episode_rows
            for episode_rows in all_episodes
            if int(episode_rows[-1]["episode_valid"]) == 1
        ]
        invalid_episode_count = len(all_episodes) - len(valid_episodes)
        inflation_selection = select_calibrated_student_t_configuration(
            scenario
        )
        selected_inflation = float(
            inflation_selection["selected_covariance_inflation"]
        )
        selected_inflations[scenario] = selected_inflation
        evaluation = evaluate_selected_configuration(
            scenario,
            valid_episodes,
            selected_inflation,
        )

        output_dir = VALIDATION_RESULT_ROOT / scenario
        update_path = output_dir / "update_metrics.csv"
        terminal_path = output_dir / "terminal_metrics.csv"
        inflation_path = output_dir / "inflation_selection.json"
        summary_path = output_dir / "summary.json"
        write_csv(update_path, evaluation["update_rows"], EVALUATION_ROW_FIELDS)
        write_csv(
            terminal_path,
            evaluation["terminal_rows"],
            EVALUATION_ROW_FIELDS,
        )
        write_json(inflation_path, inflation_selection)

        scenario_summary = {
            "protocol_version": PROTOCOL_VERSION,
            "friction_cone": FRICTION_CONE,
            "data_role": "validation",
            "scenario": scenario,
            "source_history_directory": str(
                (VALIDATION_HISTORY_ROOT / scenario).resolve()
            ),
            "maximum_push_budget": VALIDATION_PUSH_CEILING,
            "final_test_maximum_push_budget": (
                FINAL_TEST_MAXIMUM_PUSH_BUDGET
            ),
            "episodes": len(all_episodes),
            "valid_episodes": len(valid_episodes),
            "invalid_episodes": invalid_episode_count,
            "history_valid_updates": sum(
                len(rows) for rows in valid_episodes
            ),
            "belief_updates_used": sum(
                min(len(rows), FINAL_BELIEF_UPDATE_HORIZON)
                for rows in valid_episodes
            ),
            "likelihood_family": FINAL_LIKELIHOOD_FAMILY,
            "student_t_degrees_of_freedom": (
                FINAL_STUDENT_T_DEGREES_OF_FREEDOM
            ),
            "belief_update_horizon": FINAL_BELIEF_UPDATE_HORIZON,
            "points_per_dimension": FINAL_QUADRATURE_POINTS,
            "selected_covariance_inflation": selected_inflation,
            "inflation_selection": inflation_selection,
            "terminal_summary": evaluation["terminal_summary"],
            "paired_effects": evaluation["paired_effects"],
            "update_curves": evaluation["update_curves"],
            "transfer_diagnostics": evaluation["transfer_diagnostics"],
            "runtime": evaluation["runtime"],
            "update_metrics_path": str(update_path.resolve()),
            "terminal_metrics_path": str(terminal_path.resolve()),
            "inflation_selection_path": str(inflation_path.resolve()),
            "summary_path": str(summary_path.resolve()),
        }
        write_json(summary_path, scenario_summary)
        scenario_summaries[scenario] = scenario_summary
        print(
            f"{scenario}: selected inflation={selected_inflation:g}, "
            f"summary={summary_path.resolve()}"
        )

    combined_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "data_role": "validation",
        "scenarios": list(scenario_summaries),
        "final_test_maximum_push_budget": FINAL_TEST_MAXIMUM_PUSH_BUDGET,
        "likelihood_family": FINAL_LIKELIHOOD_FAMILY,
        "student_t_degrees_of_freedom": (
            FINAL_STUDENT_T_DEGREES_OF_FREEDOM
        ),
        "belief_update_horizon": FINAL_BELIEF_UPDATE_HORIZON,
        "primary_points_per_dimension": FINAL_QUADRATURE_POINTS,
        "selected_covariance_inflation": selected_inflations,
        "scenario_summaries": scenario_summaries,
    }
    combined_path = VALIDATION_RESULT_ROOT / "combined_summary.json"
    configuration_path = VALIDATION_RESULT_ROOT / "selected_configuration.json"
    write_json(combined_path, combined_summary)
    write_json(
        configuration_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "friction_cone": FRICTION_CONE,
            "source_data_role": "validation",
            "maximum_push_budget": FINAL_TEST_MAXIMUM_PUSH_BUDGET,
            "likelihood_family": FINAL_LIKELIHOOD_FAMILY,
            "student_t_degrees_of_freedom": (
                FINAL_STUDENT_T_DEGREES_OF_FREEDOM
            ),
            "belief_update_horizon": FINAL_BELIEF_UPDATE_HORIZON,
            "points_per_dimension": FINAL_QUADRATURE_POINTS,
            "covariance_inflation": selected_inflations,
            "inflation_selection_metric": (
                "condition_balanced_mean_terminal_nll_within_93_to_97_percent_coverage"
            ),
            "test_histories_viewed": False,
        },
    )
    print(f"Combined Validation summary: {combined_path.resolve()}")
    print(f"Selected configuration: {configuration_path.resolve()}")
    return combined_summary


def load_selected_validation_configuration() -> dict[str, Any]:
    """读取并核对 Independent Test 使用的固定 Validation 配置。"""

    path = VALIDATION_RESULT_ROOT / "selected_configuration.json"
    with path.open("r", encoding="utf-8") as handle:
        configuration = json.load(handle)
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "maximum_push_budget": FINAL_TEST_MAXIMUM_PUSH_BUDGET,
        "likelihood_family": FINAL_LIKELIHOOD_FAMILY,
        "student_t_degrees_of_freedom": (
            FINAL_STUDENT_T_DEGREES_OF_FREEDOM
        ),
        "belief_update_horizon": FINAL_BELIEF_UPDATE_HORIZON,
        "points_per_dimension": FINAL_QUADRATURE_POINTS,
    }
    for field, value in expected.items():
        if configuration[field] != value:
            raise RuntimeError(
                f"Test configuration 的 {field} 不匹配: "
                f"{configuration[field]} != {value}"
            )
    if set(configuration["covariance_inflation"]) != set(SCENARIOS):
        raise RuntimeError("Test configuration 缺少场景级 covariance inflation")
    configuration["source_configuration_path"] = str(path.resolve())
    return configuration


def evaluate_test(args: argparse.Namespace) -> dict[str, Any]:
    """使用固定 Validation 配置评价 Independent Test。"""

    configuration = load_selected_validation_configuration()
    scenario_summaries: dict[str, dict[str, Any]] = {}
    for scenario in select_scenarios(args.scenario):
        all_episodes = load_history_episodes(scenario, "test")
        valid_episodes = [
            episode_rows
            for episode_rows in all_episodes
            if int(episode_rows[-1]["episode_valid"]) == 1
        ]
        selected_inflation = float(
            configuration["covariance_inflation"][scenario]
        )
        evaluation = evaluate_selected_configuration(
            scenario,
            valid_episodes,
            selected_inflation,
        )

        output_dir = TEST_RESULT_ROOT / scenario
        update_path = output_dir / "update_metrics.csv"
        terminal_path = output_dir / "terminal_metrics.csv"
        summary_path = output_dir / "summary.json"
        write_csv(update_path, evaluation["update_rows"], EVALUATION_ROW_FIELDS)
        write_csv(
            terminal_path,
            evaluation["terminal_rows"],
            EVALUATION_ROW_FIELDS,
        )
        bootstrap_seed = (
            args.bootstrap_seed + SCENARIOS.index(scenario)
        )
        bootstrap = bootstrap_result_scenario(
            scenario,
            "test",
            args.bootstrap_resamples,
            bootstrap_seed,
        )
        terminal_reasons = Counter(
            episode_rows[-1]["terminal_reason"]
            for episode_rows in all_episodes
        )
        scenario_summary = {
            "protocol_version": PROTOCOL_VERSION,
            "friction_cone": FRICTION_CONE,
            "data_role": "test",
            "scenario": scenario,
            "source_history_directory": str(
                (TEST_HISTORY_ROOT / scenario).resolve()
            ),
            "source_configuration_path": configuration[
                "source_configuration_path"
            ],
            "maximum_push_budget": FINAL_TEST_MAXIMUM_PUSH_BUDGET,
            "episodes": len(all_episodes),
            "valid_episodes": len(valid_episodes),
            "invalid_episodes": len(all_episodes) - len(valid_episodes),
            "successful_episodes": int(terminal_reasons["success"]),
            "maximum_budget_episodes": int(
                terminal_reasons["maximum_push_budget"]
            ),
            "history_valid_observations": sum(
                len(rows) for rows in valid_episodes
            ),
            "belief_updates_used": sum(
                min(len(rows), FINAL_BELIEF_UPDATE_HORIZON)
                for rows in valid_episodes
            ),
            "likelihood_family": FINAL_LIKELIHOOD_FAMILY,
            "student_t_degrees_of_freedom": (
                FINAL_STUDENT_T_DEGREES_OF_FREEDOM
            ),
            "belief_update_horizon": FINAL_BELIEF_UPDATE_HORIZON,
            "points_per_dimension": FINAL_QUADRATURE_POINTS,
            "covariance_inflation": selected_inflation,
            "terminal_summary": evaluation["terminal_summary"],
            "paired_effects": evaluation["paired_effects"],
            "update_curves": evaluation["update_curves"],
            "runtime": evaluation["runtime"],
            "bootstrap": bootstrap,
            "update_metrics_path": str(update_path.resolve()),
            "terminal_metrics_path": str(terminal_path.resolve()),
            "summary_path": str(summary_path.resolve()),
        }
        write_json(summary_path, scenario_summary)
        scenario_summaries[scenario] = scenario_summary
        print(
            f"{scenario}: Test episodes={len(all_episodes)}, "
            f"summary={summary_path.resolve()}"
        )

    combined_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "data_role": "test",
        "configuration_selected_on": "validation",
        "configuration_changed_after_test": False,
        "likelihood_family": FINAL_LIKELIHOOD_FAMILY,
        "student_t_degrees_of_freedom": (
            FINAL_STUDENT_T_DEGREES_OF_FREEDOM
        ),
        "belief_update_horizon": FINAL_BELIEF_UPDATE_HORIZON,
        "points_per_dimension": FINAL_QUADRATURE_POINTS,
        "covariance_inflation": configuration["covariance_inflation"],
        "bootstrap_resamples": args.bootstrap_resamples,
        "base_bootstrap_seed": args.bootstrap_seed,
        "scenario_summaries": scenario_summaries,
    }
    combined_path = TEST_RESULT_ROOT / "combined_summary.json"
    evaluated_configuration_path = TEST_RESULT_ROOT / "evaluated_configuration.json"
    write_json(combined_path, combined_summary)
    write_json(
        evaluated_configuration_path,
        {
            **configuration,
            "data_role": "test",
            "test_histories_viewed": True,
            "test_histories_evaluated": True,
            "configuration_changed_after_test": False,
        },
    )
    print(f"Combined Test summary: {combined_path.resolve()}")
    return combined_summary


def build_parser() -> argparse.ArgumentParser:
    """构建 E2 命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-targets",
        help="分析 Validation 或 Test candidates 并生成 target manifests",
    )
    prepare.add_argument(
        "--role",
        choices=("validation", "test"),
        default="validation",
    )
    prepare.set_defaults(handler=prepare_targets)

    residuals = subparsers.add_parser(
        "fit-residuals",
        help="使用 E2 Training outcomes 拟合 P1 likelihood residual statistics",
    )
    residuals.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="all",
    )
    residuals.set_defaults(handler=fit_residuals)

    validation = subparsers.add_parser(
        "evaluate-validation",
        help="选择 covariance inflation 并评价 paired sequential estimators",
    )
    validation.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="all",
    )
    validation.set_defaults(handler=evaluate_validation)

    diagnostics = subparsers.add_parser(
        "diagnose-validation",
        help="检查 quadrature sensitivity 与 likelihood calibration",
    )
    diagnostics.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="all",
    )
    diagnostics.set_defaults(handler=diagnose_validation)

    calibration = subparsers.add_parser(
        "calibrate-validation",
        help="在完整 histories 上比较 inflation 与有限 update horizon",
    )
    calibration.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="all",
    )
    calibration.set_defaults(handler=calibrate_validation)

    student_t_calibration = subparsers.add_parser(
        "calibrate-student-t-validation",
        help="在完整 histories 上评价稳健 Student-t likelihood",
    )
    student_t_calibration.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="all",
    )
    student_t_calibration.set_defaults(
        handler=calibrate_student_t_validation
    )

    quadrature_comparison = subparsers.add_parser(
        "compare-final-quadrature",
        help="在最终 Student-t 配置下比较 9-point 与 17-point",
    )
    quadrature_comparison.add_argument(
        "--scenario",
        choices=("all", *SCENARIOS),
        default="all",
    )
    quadrature_comparison.set_defaults(handler=compare_final_quadrature)

    validation_bootstrap = subparsers.add_parser(
        "bootstrap-validation",
        help="生成正式 Validation paired bootstrap 与结果表",
    )
    validation_bootstrap.add_argument(
        "--scenario",
        choices=("all", *SCENARIOS),
        default="all",
    )
    validation_bootstrap.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    validation_bootstrap.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    validation_bootstrap.set_defaults(handler=bootstrap_validation)

    test_evaluation = subparsers.add_parser(
        "evaluate-test",
        help="使用固定 Validation 配置评价 Independent Test",
    )
    test_evaluation.add_argument(
        "--scenario",
        choices=("all", *SCENARIOS),
        default="all",
    )
    test_evaluation.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    test_evaluation.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_TEST_BOOTSTRAP_SEED,
    )
    test_evaluation.set_defaults(handler=evaluate_test)
    return parser


def main() -> None:
    """解析参数并执行对应的 E2 任务。"""

    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
