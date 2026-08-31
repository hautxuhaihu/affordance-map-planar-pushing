"""CAR（Continuous Action Refinement）Experiment 1 统一入口。

该脚本实现 Continuous Action Oracle Headroom：

- plan：读取现有 HCR V2 artifacts，打印正式实验规模；
- prepare：生成 128 个 Validation targets、oracle anchors 与连续候选；
- collect：并行执行 unique continuous actions 的 MuJoCo rollouts；
- evaluate：计算 oracle headroom、bootstrap confidence intervals 与 Go/No-Go。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import mujoco
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from push_core.hcr_v2.e1 import (  # noqa: E402
    PRIMARY_TNPO_COST,
    cost_matrix,
    pose_error_matrices,
    read_csv_rows,
    success_matrix,
    wrap_to_pi,
)
from push_core.schema import level1_schema as schema  # noqa: E402
from push_core.simulation.physical_pusher_rollout import (  # noqa: E402
    load_model,
    run_physical_pusher_rollout,
)


PROTOCOL_VERSION = "continuous_action_refinement_e1_v1"
CAR_FULL_NAME = "Continuous Action Refinement"
FRICTION_CONE = "elliptic"
DEFAULT_SEED = 0

MANIFEST_ROOT = REPOSITORY_ROOT / "manifests" / "car"
HCR_MANIFEST_ROOT = (
    REPOSITORY_ROOT / "manifests" / "hcr_v2"
)
ACTION_MANIFEST_PATH = HCR_MANIFEST_ROOT / "hcr_v2_action_core_manifest_v1.csv"
TARGET_SOURCE_PATH = HCR_MANIFEST_ROOT / "hcr_v2_core_target_manifest_v1.csv"
NOMINAL_OUTCOME_PATH = (
    PROJECT_ROOT
    / "data"
    / "hcr_v2"
    / "e1"
    / "outcomes"
    / "joint"
    / "training"
    / "J_TRAIN_062.csv"
)
BASE_XML_PATH = PROJECT_ROOT / "assets" / "xml" / "msc_rod_pusher_box_hcr_v2.xml"

TARGET_MANIFEST_PATH = (
    MANIFEST_ROOT / "continuous_action_refinement_validation_targets.csv"
)
DATA_ROOT = PROJECT_ROOT / "data" / "car" / "experiment_1"
RESULTS_ROOT = PROJECT_ROOT / "results" / "car" / "experiment_1"
ANCHOR_PATH = DATA_ROOT / "continuous_action_refinement_anchors.csv"
CANDIDATE_PATH = DATA_ROOT / "continuous_action_refinement_candidates.csv"
UNIQUE_ACTION_PATH = DATA_ROOT / "continuous_action_refinement_unique_actions.csv"
OUTCOME_PATH = DATA_ROOT / "continuous_action_refinement_unique_outcomes.csv"
PREPARATION_SUMMARY_PATH = RESULTS_ROOT / "preparation_summary.json"
CASE_METRICS_PATH = RESULTS_ROOT / "continuous_action_refinement_case_metrics.csv"
SUMMARY_PATH = RESULTS_ROOT / "continuous_action_refinement_summary.json"

ANCHOR_BUDGETS = (20, 50, 100)
SAMPLE_BUDGETS = (128, 256, 512)
PRIMARY_ANCHOR_BUDGET = 20
PRIMARY_SAMPLE_BUDGET = 512
PRIMARY_IMPROVEMENT_THRESHOLD = 0.05
SENSITIVITY_IMPROVEMENT_THRESHOLD = 0.02
PRACTICAL_EQUIVALENCE_TOLERANCE = 0.05

TARGET_FIELDS = [
    "car_target_index",
    "v2_target_id",
    "source_split_role",
    "target_delta_x_m",
    "target_delta_y_m",
    "canonical_position_key",
    "radial_distance_m",
    "polar_angle_rad",
    "radial_quartile_index",
    "angular_quadrant_index",
    "selection_seed",
]

ANCHOR_FIELDS = [
    "car_target_index",
    "v2_target_id",
    "target_delta_x_m",
    "target_delta_y_m",
    "radial_quartile_index",
    "angular_quadrant_index",
    "anchor_rank",
    "v2_action_id",
    "discrete_tnpo_cost",
    "discrete_position_error_m",
    "discrete_yaw_error_rad",
    "discrete_pose_success",
    "real_delta_x",
    "real_delta_y",
    "real_delta_yaw",
    "surface_id",
    "contact_region_id",
    "contact_region_row",
    "contact_region_col",
    "contact_point_local_x",
    "contact_point_local_y",
    "force_angle_relative_to_normal_deg",
    "commanded_force_N",
    "ramp_up_s",
    "hold_s",
    "ramp_down_s",
]

CANDIDATE_FIELDS = [
    "source_v2_action_id",
    "source_action_index",
    "source_contact_region_id",
    "sobol_sample_index",
    "minimum_candidate_budget",
    "physical_action_id",
    "physical_action_key",
]

UNIQUE_ACTION_FIELDS = [
    "physical_action_id",
    "physical_action_key",
    "source_v2_action_id",
    "source_action_index",
    "source_candidate_id",
    "source_action_param_index",
    "source_contact_region_id",
    "sobol_sample_index",
    "surface_id",
    "contact_region_row",
    "contact_region_col",
    "contact_point_local_x",
    "contact_point_local_y",
    "contact_normal_local_x",
    "contact_normal_local_y",
    "force_angle_relative_to_normal_deg",
    "force_direction_local_x",
    "force_direction_local_y",
    "commanded_force_N",
    "ramp_up_s",
    "hold_s",
    "ramp_down_s",
    "execution_duration_s",
    "sampling_seed",
]

OUTCOME_FIELDS = [
    "experiment_id",
    "protocol_version",
    "car_full_name",
    "friction_cone",
    "condition_id",
    *UNIQUE_ACTION_FIELDS,
    "real_delta_x",
    "real_delta_y",
    "real_delta_yaw",
    "real_delta_z",
    "quality_pass",
    "simulation_unstable",
    "contact_success",
    "num_contacts",
    "stopped_by_threshold",
]

CASE_METRIC_FIELDS = [
    "car_target_index",
    "v2_target_id",
    "radial_quartile_index",
    "angular_quadrant_index",
    "anchor_budget",
    "candidates_per_anchor",
    "target_candidate_relationships",
    "valid_candidate_relationships",
    "discrete_oracle_action_id",
    "discrete_oracle_cost",
    "discrete_oracle_pose_success",
    "continuous_oracle_physical_action_id",
    "continuous_oracle_physical_action_key",
    "continuous_oracle_cost",
    "continuous_oracle_pose_success",
    "union_oracle_cost",
    "union_oracle_pose_success",
    "union_oracle_improvement",
    "continuous_only_improvement",
    "effective_improvement_0p05",
    "effective_improvement_0p02",
    "continuous_only_result",
]

_GRID_SIGMA = np.asarray(
    [0.015, 7.5, 0.15, 0.010, 0.025, 0.010], dtype=np.float64
)
_CONTACT_INTERVALS = {
    0: (-0.045, -0.015),
    1: (-0.015, 0.015),
    2: (0.015, 0.045),
}

_WORKER_MODEL: Any | None = None
_WORKER_DATA: Any | None = None


def make_json_compatible(value: Any) -> Any:
    """将非有限浮点值转换为严格 JSON 支持的空值。"""

    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {key: make_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_compatible(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写出 UTF-8 严格 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            make_json_compatible(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    """按固定字段顺序写出 UTF-8 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_source_targets() -> list[dict[str, str]]:
    """读取 HCR V2 training-role core targets。"""

    rows = [
        row
        for row in read_csv_rows(TARGET_SOURCE_PATH)
        if row["split_role"] == "training"
    ]
    if len(rows) != 3_268:
        raise RuntimeError(f"training target 数量错误: {len(rows)}")
    return rows


def angular_quadrant(delta_x: float, delta_y: float) -> int:
    """按固定坐标轴边界返回 angular quadrant。"""

    if delta_x >= 0.0 and delta_y >= 0.0:
        return 0
    if delta_x < 0.0 and delta_y >= 0.0:
        return 1
    if delta_x < 0.0 and delta_y < 0.0:
        return 2
    return 3


def select_validation_targets() -> list[dict[str, Any]]:
    """按四个 radial quartiles 与四个 angular quadrants 选择 targets。"""

    enriched: list[dict[str, Any]] = []
    for row in load_source_targets():
        delta_x = float(row["target_delta_x_m"])
        delta_y = float(row["target_delta_y_m"])
        enriched.append(
            {
                **row,
                "radial_distance_m": math.hypot(delta_x, delta_y),
                "polar_angle_rad": math.atan2(delta_y, delta_x),
                "angular_quadrant_index": angular_quadrant(delta_x, delta_y),
            }
        )

    enriched.sort(
        key=lambda row: (
            float(row["radial_distance_m"]),
            float(row["polar_angle_rad"]),
            row["v2_target_id"],
        )
    )
    quartile_size = len(enriched) // 4
    for rank, row in enumerate(enriched):
        row["radial_quartile_index"] = rank // quartile_size

    strata: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        key = (
            int(row["radial_quartile_index"]),
            int(row["angular_quadrant_index"]),
        )
        strata[key].append(row)

    selected: list[dict[str, Any]] = []
    for radial_index in range(4):
        for angular_index in range(4):
            stratum = sorted(
                strata[(radial_index, angular_index)],
                key=lambda row: row["v2_target_id"],
            )
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [DEFAULT_SEED, radial_index, angular_index]
                )
            )
            chosen_indices = rng.choice(len(stratum), size=8, replace=False)
            selected.extend(stratum[int(index)] for index in chosen_indices)

    selected.sort(key=lambda row: row["v2_target_id"])
    output: list[dict[str, Any]] = []
    for target_index, row in enumerate(selected):
        output.append(
            {
                "car_target_index": target_index,
                "v2_target_id": row["v2_target_id"],
                "source_split_role": row["split_role"],
                "target_delta_x_m": f"{float(row['target_delta_x_m']):.4f}",
                "target_delta_y_m": f"{float(row['target_delta_y_m']):.4f}",
                "canonical_position_key": row["canonical_position_key"],
                "radial_distance_m": f"{float(row['radial_distance_m']):.12f}",
                "polar_angle_rad": f"{float(row['polar_angle_rad']):.12f}",
                "radial_quartile_index": int(row["radial_quartile_index"]),
                "angular_quadrant_index": int(row["angular_quadrant_index"]),
                "selection_seed": DEFAULT_SEED,
            }
        )
    return output


def load_nominal_action_outcomes() -> list[dict[str, str]]:
    """合并 action manifest 与固定 nominal MuJoCo outcomes。"""

    actions = sorted(read_csv_rows(ACTION_MANIFEST_PATH), key=lambda row: row["v2_action_id"])
    outcomes = {row["v2_action_id"]: row for row in read_csv_rows(NOMINAL_OUTCOME_PATH)}
    if len(actions) != 4_536 or len(outcomes) != 4_536:
        raise RuntimeError(
            f"nominal action/outcome 数量错误: actions={len(actions)}, outcomes={len(outcomes)}"
        )

    merged: list[dict[str, str]] = []
    for action in actions:
        action_id = action["v2_action_id"]
        if action_id not in outcomes:
            raise RuntimeError(f"nominal outcome 缺少 action: {action_id}")
        merged.append({**action, **outcomes[action_id]})
    return merged


def build_anchor_rows(
    targets: list[dict[str, Any]],
    actions: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """计算每个 target 的真实 MuJoCo Top-100 oracle anchors。"""

    outcomes = np.asarray(
        [
            [
                float(row["real_delta_x"]),
                float(row["real_delta_y"]),
                float(row["real_delta_yaw"]),
            ]
            for row in actions
        ],
        dtype=np.float64,
    )
    valid = np.asarray(
        [
            int(row["quality_pass"]) == 1
            and int(row["simulation_unstable"]) == 0
            and int(row["contact_success"]) == 1
            and int(row["stopped_by_threshold"]) == 1
            for row in actions
        ],
        dtype=bool,
    )
    target_positions = np.asarray(
        [
            [float(row["target_delta_x_m"]), float(row["target_delta_y_m"])]
            for row in targets
        ],
        dtype=np.float64,
    )
    position_error, yaw_error = pose_error_matrices(outcomes, target_positions)
    costs = cost_matrix("v2", position_error, yaw_error, PRIMARY_TNPO_COST)
    costs[:, ~valid] = np.inf
    successes = success_matrix(position_error, yaw_error, PRIMARY_TNPO_COST)

    anchor_rows: list[dict[str, Any]] = []
    unique_counts = {budget: set() for budget in ANCHOR_BUDGETS}
    for target_index, target in enumerate(targets):
        order = np.argsort(costs[target_index], kind="stable")[: max(ANCHOR_BUDGETS)]
        for zero_rank, action_index in enumerate(order):
            rank = zero_rank + 1
            action = actions[int(action_index)]
            for budget in ANCHOR_BUDGETS:
                if rank <= budget:
                    unique_counts[budget].add(action["v2_action_id"])
            anchor_rows.append(
                {
                    "car_target_index": target_index,
                    "v2_target_id": target["v2_target_id"],
                    "target_delta_x_m": target["target_delta_x_m"],
                    "target_delta_y_m": target["target_delta_y_m"],
                    "radial_quartile_index": target["radial_quartile_index"],
                    "angular_quadrant_index": target["angular_quadrant_index"],
                    "anchor_rank": rank,
                    "v2_action_id": action["v2_action_id"],
                    "discrete_tnpo_cost": f"{costs[target_index, action_index]:.12f}",
                    "discrete_position_error_m": (
                        f"{position_error[target_index, action_index]:.12f}"
                    ),
                    "discrete_yaw_error_rad": (
                        f"{yaw_error[target_index, action_index]:.12f}"
                    ),
                    "discrete_pose_success": int(
                        successes[target_index, action_index]
                    ),
                    "real_delta_x": action["real_delta_x"],
                    "real_delta_y": action["real_delta_y"],
                    "real_delta_yaw": action["real_delta_yaw"],
                    "surface_id": action["surface_id"],
                    "contact_region_id": action["contact_region_id"],
                    "contact_region_row": action["contact_region_row"],
                    "contact_region_col": action["contact_region_col"],
                    "contact_point_local_x": action["contact_point_local_x"],
                    "contact_point_local_y": action["contact_point_local_y"],
                    "force_angle_relative_to_normal_deg": action[
                        "force_angle_relative_to_normal_deg"
                    ],
                    "commanded_force_N": action["commanded_force_N"],
                    "ramp_up_s": action["ramp_up_s"],
                    "hold_s": action["hold_s"],
                    "ramp_down_s": action["ramp_down_s"],
                }
            )
    return anchor_rows, {budget: len(values) for budget, values in unique_counts.items()}


def stable_anchor_seed(action_index: int) -> int:
    """由默认 seed 与稳定 action index 派生 Sobol seed。"""

    return int(
        np.random.SeedSequence([DEFAULT_SEED, action_index]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def rotate_inward_direction(normal_x: float, normal_y: float, angle_deg: float) -> tuple[float, float]:
    """由 surface outward normal 与相对角度计算局部施力方向。"""

    inward_x = -normal_x
    inward_y = -normal_y
    angle = math.radians(angle_deg)
    direction_x = math.cos(angle) * inward_x - math.sin(angle) * inward_y
    direction_y = math.sin(angle) * inward_x + math.cos(angle) * inward_y
    return direction_x, direction_y


def physical_action_key(row: dict[str, Any]) -> str:
    """根据正式字段顺序构造可读的 physical action key。"""

    return (
        f"surface={int(row['surface_id'])}|"
        f"x={float(row['contact_point_local_x']):.8f}|"
        f"y={float(row['contact_point_local_y']):.8f}|"
        f"angle={float(row['force_angle_relative_to_normal_deg']):.8f}|"
        f"force={float(row['commanded_force_N']):.8f}|"
        f"ramp_up={float(row['ramp_up_s']):.8f}|"
        f"hold={float(row['hold_s']):.8f}|"
        f"ramp_down={float(row['ramp_down_s']):.8f}"
    )


def generate_anchor_candidates(anchor: dict[str, str]) -> Iterator[dict[str, Any]]:
    """为一个离散 anchor 生成 512 个嵌套 Sobol 截断高斯 candidates。"""

    action_index = int(anchor["v2_action_id"][1:])
    surface_id = int(anchor["surface_id"])
    contact_col = int(anchor["contact_region_col"])
    tangent_is_y = surface_id in {0, 1}
    tangent_center = float(
        anchor["contact_point_local_y" if tangent_is_y else "contact_point_local_x"]
    )
    contact_low, contact_high = _CONTACT_INTERVALS[contact_col]

    centers = np.asarray(
        [
            tangent_center,
            float(anchor["force_angle_relative_to_normal_deg"]),
            float(anchor["commanded_force_N"]),
            float(anchor["ramp_up_s"]),
            float(anchor["hold_s"]),
            float(anchor["ramp_down_s"]),
        ],
        dtype=np.float64,
    )
    lower = np.asarray(
        [contact_low, -45.0, 0.50, 0.020, 0.050, 0.020], dtype=np.float64
    )
    upper = np.asarray(
        [contact_high, 45.0, 1.10, 0.060, 0.150, 0.040], dtype=np.float64
    )
    alpha = ndtr((lower - centers) / _GRID_SIGMA)
    beta = ndtr((upper - centers) / _GRID_SIGMA)
    sampling_seed = stable_anchor_seed(action_index)
    sobol = qmc.Sobol(d=6, scramble=True, seed=sampling_seed)
    unit_points = sobol.random_base2(m=9)
    probabilities = alpha[None, :] + unit_points * (beta - alpha)[None, :]
    samples = centers[None, :] + _GRID_SIGMA[None, :] * ndtri(probabilities)
    samples = np.round(samples, decimals=8)

    normal_x = float(anchor["contact_normal_local_x"])
    normal_y = float(anchor["contact_normal_local_y"])
    fixed_x = round(float(anchor["contact_point_local_x"]), 8)
    fixed_y = round(float(anchor["contact_point_local_y"]), 8)
    for sample_index, values in enumerate(samples):
        tangent, angle_deg, force, ramp_up, hold, ramp_down = [
            float(value) for value in values
        ]
        contact_x = fixed_x if tangent_is_y else tangent
        contact_y = tangent if tangent_is_y else fixed_y
        direction_x, direction_y = rotate_inward_direction(normal_x, normal_y, angle_deg)
        yield {
            "source_v2_action_id": anchor["v2_action_id"],
            "source_action_index": action_index,
            "source_candidate_id": anchor["candidate_id"],
            "source_action_param_index": anchor["action_param_index"],
            "source_contact_region_id": anchor["contact_region_id"],
            "sobol_sample_index": sample_index,
            "minimum_candidate_budget": (
                128 if sample_index < 128 else 256 if sample_index < 256 else 512
            ),
            "surface_id": surface_id,
            "contact_region_row": anchor["contact_region_row"],
            "contact_region_col": anchor["contact_region_col"],
            "contact_point_local_x": round(contact_x, 8),
            "contact_point_local_y": round(contact_y, 8),
            "contact_normal_local_x": normal_x,
            "contact_normal_local_y": normal_y,
            "force_angle_relative_to_normal_deg": round(angle_deg, 8),
            "force_direction_local_x": round(direction_x, 12),
            "force_direction_local_y": round(direction_y, 12),
            "commanded_force_N": round(force, 8),
            "ramp_up_s": round(ramp_up, 8),
            "hold_s": round(hold, 8),
            "ramp_down_s": round(ramp_down, 8),
            "execution_duration_s": round(ramp_up + hold + ramp_down, 8),
            "sampling_seed": sampling_seed,
        }


def prepare_candidates(
    unique_anchor_ids: list[str],
    action_by_id: dict[str, dict[str, str]],
) -> tuple[int, int]:
    """生成 candidate relationships 与 unique physical actions。"""

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    physical_id_by_key: dict[str, int] = {}
    relationship_count = 0
    with CANDIDATE_PATH.open("w", encoding="utf-8-sig", newline="") as candidate_handle, UNIQUE_ACTION_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as action_handle:
        candidate_writer = csv.DictWriter(
            candidate_handle,
            fieldnames=CANDIDATE_FIELDS,
            extrasaction="ignore",
        )
        action_writer = csv.DictWriter(
            action_handle,
            fieldnames=UNIQUE_ACTION_FIELDS,
            extrasaction="ignore",
        )
        candidate_writer.writeheader()
        action_writer.writeheader()

        for anchor_number, action_id in enumerate(unique_anchor_ids, start=1):
            for candidate in generate_anchor_candidates(action_by_id[action_id]):
                key = physical_action_key(candidate)
                physical_id = physical_id_by_key.get(key)
                if physical_id is None:
                    physical_id = len(physical_id_by_key)
                    physical_id_by_key[key] = physical_id
                    action_writer.writerow(
                        {
                            **candidate,
                            "physical_action_id": physical_id,
                            "physical_action_key": key,
                        }
                    )
                candidate_writer.writerow(
                    {
                        **candidate,
                        "physical_action_id": physical_id,
                        "physical_action_key": key,
                    }
                )
                relationship_count += 1
            if anchor_number % 100 == 0 or anchor_number == len(unique_anchor_ids):
                print(
                    f"prepared anchor {anchor_number}/{len(unique_anchor_ids)}; "
                    f"unique actions={len(physical_id_by_key)}"
                )
    return relationship_count, len(physical_id_by_key)


def prepare_experiment() -> dict[str, Any]:
    """生成正式 targets、anchors 与 continuous candidates。"""

    targets = select_validation_targets()
    actions = load_nominal_action_outcomes()
    anchor_rows, unique_anchor_counts = build_anchor_rows(targets, actions)
    write_csv(TARGET_MANIFEST_PATH, targets, TARGET_FIELDS)
    write_csv(ANCHOR_PATH, anchor_rows, ANCHOR_FIELDS)

    top100_ids = sorted(
        {
            row["v2_action_id"]
            for row in anchor_rows
            if int(row["anchor_rank"]) <= max(ANCHOR_BUDGETS)
        }
    )
    action_by_id = {row["v2_action_id"]: row for row in actions}
    relationship_count, unique_action_count = prepare_candidates(top100_ids, action_by_id)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "car_full_name": CAR_FULL_NAME,
        "friction_cone": FRICTION_CONE,
        "nominal_condition_id": "J_TRAIN_062",
        "default_seed": DEFAULT_SEED,
        "validation_targets": len(targets),
        "anchor_rows": len(anchor_rows),
        "unique_anchor_counts": {
            f"top_{budget}": unique_anchor_counts[budget]
            for budget in ANCHOR_BUDGETS
        },
        "candidate_relationships": relationship_count,
        "unique_continuous_actions": unique_action_count,
        "primary_target_candidate_relationships": (
            len(targets) * PRIMARY_ANCHOR_BUDGET * PRIMARY_SAMPLE_BUDGET
        ),
        "target_manifest": str(TARGET_MANIFEST_PATH.resolve()),
        "anchor_path": str(ANCHOR_PATH.resolve()),
        "candidate_path": str(CANDIDATE_PATH.resolve()),
        "unique_action_path": str(UNIQUE_ACTION_PATH.resolve()),
    }
    write_json(PREPARATION_SUMMARY_PATH, summary)
    return summary


def plan_experiment() -> dict[str, Any]:
    """只读计算 target 与 anchor 规模，不生成正式 artifacts。"""

    targets = select_validation_targets()
    actions = load_nominal_action_outcomes()
    _, unique_anchor_counts = build_anchor_rows(targets, actions)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "car_full_name": CAR_FULL_NAME,
        "friction_cone": FRICTION_CONE,
        "nominal_condition_id": "J_TRAIN_062",
        "default_seed": DEFAULT_SEED,
        "validation_targets": len(targets),
        "unique_anchor_counts": {
            f"top_{budget}": unique_anchor_counts[budget]
            for budget in ANCHOR_BUDGETS
        },
        "primary_target_candidate_relationships": (
            len(targets) * PRIMARY_ANCHOR_BUDGET * PRIMARY_SAMPLE_BUDGET
        ),
        "maximum_unique_rollouts_from_observed_top_100": (
            unique_anchor_counts[100] * PRIMARY_SAMPLE_BUDGET
        ),
    }


def named_id(model: Any, object_type: Any, name: str) -> int:
    """按名称读取 MuJoCo object id。"""

    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return int(object_id)


def set_sliding_friction(model: Any, friction_mu: float) -> None:
    """同步设置 table 与 object geom 的 sliding friction。"""

    table_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    object_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    model.geom_friction[table_id, 0] = float(friction_mu)
    model.geom_friction[object_id, 0] = float(friction_mu)


def build_rollout_input(action: dict[str, str], episode_id: int) -> dict[str, Any]:
    """把 continuous physical action 转换为 physical-pusher rollout row。"""

    ramp_up = float(action["ramp_up_s"])
    hold = float(action["hold_s"])
    ramp_down = float(action["ramp_down_s"])
    row: dict[str, Any] = dict(schema.LEVEL1_FIXED_VALUES)
    row.update(
        {
            "dataset_role": "car_experiment_1_validation",
            "episode_id": episode_id,
            "step_id": 0,
            "group_id": 0,
            "candidate_id": int(action["physical_action_id"]),
            "contact_region_id": int(action["source_contact_region_id"]),
            "goal_delta_local_x": 0.0,
            "goal_delta_local_y": 0.0,
            "goal_yaw": 0.0,
            "surface_id": int(action["surface_id"]),
            "contact_region_row": int(action["contact_region_row"]),
            "contact_region_col": int(action["contact_region_col"]),
            "contact_point_local_x": float(action["contact_point_local_x"]),
            "contact_point_local_y": float(action["contact_point_local_y"]),
            "contact_point_local_z": 0.0,
            "contact_normal_local_x": float(action["contact_normal_local_x"]),
            "contact_normal_local_y": float(action["contact_normal_local_y"]),
            "contact_normal_local_z": 0.0,
            "force_angle_relative_to_normal_deg": float(
                action["force_angle_relative_to_normal_deg"]
            ),
            "force_direction_local_x": float(action["force_direction_local_x"]),
            "force_direction_local_y": float(action["force_direction_local_y"]),
            "force_direction_local_z": 0.0,
            "commanded_force_N": float(action["commanded_force_N"]),
            "ramp_up_s": ramp_up,
            "hold_s": hold,
            "ramp_down_s": ramp_down,
            "command_duration_s": ramp_up + hold + ramp_down,
            "hidden_com_offset_x": 0.0,
            "hidden_com_offset_y": 0.0,
            "hidden_com_offset_z": 0.0,
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_yaw": 0.0,
            "delta_z": 0.0,
            "final_qpos_x": 0.0,
            "final_qpos_y": 0.0,
            "final_qpos_yaw": 0.0,
            "settle_time_s": 0.0,
            "simulation_unstable": 0,
            "quality_pass": 0,
            "contact_success": 0,
            "num_contacts": 0,
            "stopped_by_threshold": 0,
        }
    )
    return row


def initialise_rollout_worker() -> None:
    """在 collection worker 中加载一次 Elliptic MuJoCo model。"""

    global _WORKER_MODEL, _WORKER_DATA
    _WORKER_MODEL, _WORKER_DATA = load_model(BASE_XML_PATH)
    set_sliding_friction(_WORKER_MODEL, 0.40)


def process_action_batch(actions: list[dict[str, str]]) -> list[dict[str, Any]]:
    """在一个 worker 中顺序执行一批独立 continuous actions。"""

    if _WORKER_MODEL is None or _WORKER_DATA is None:
        initialise_rollout_worker()
    rows: list[dict[str, Any]] = []
    for action in actions:
        rollout_input = build_rollout_input(action, int(action["physical_action_id"]))
        result = run_physical_pusher_rollout(
            _WORKER_MODEL,
            _WORKER_DATA,
            rollout_input,
            validate_schema=False,
        )
        rows.append(
            {
                "experiment_id": "E1",
                "protocol_version": PROTOCOL_VERSION,
                "car_full_name": CAR_FULL_NAME,
                "friction_cone": FRICTION_CONE,
                "condition_id": "J_TRAIN_062",
                **action,
                "real_delta_x": result["delta_x"],
                "real_delta_y": result["delta_y"],
                "real_delta_yaw": result["delta_yaw"],
                "real_delta_z": result["delta_z"],
                "quality_pass": result["quality_pass"],
                "simulation_unstable": result["simulation_unstable"],
                "contact_success": result["contact_success"],
                "num_contacts": result["num_contacts"],
                "stopped_by_threshold": result["stopped_by_threshold"],
            }
        )
    return rows


def count_existing_outcomes(path: Path) -> tuple[int, int]:
    """统计已有 outcomes，并返回行数和最后一个 physical action id。"""

    if not path.exists():
        return 0, -1
    count = 0
    last_id = -1
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            count += 1
            last_id = int(row["physical_action_id"])
    return count, last_id


def iter_action_batches(
    path: Path,
    start_index: int,
    batch_size: int,
    maximum_actions: int,
) -> Iterator[list[dict[str, str]]]:
    """按 physical action id 顺序流式生成 collection batches。"""

    batch: list[dict[str, str]] = []
    yielded = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["physical_action_id"]) < start_index:
                continue
            if maximum_actions > 0 and yielded >= maximum_actions:
                break
            batch.append(row)
            yielded += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def collect_outcomes(
    num_workers: int,
    batch_size: int,
    resume: bool,
    smoke: bool,
    maximum_actions: int,
) -> dict[str, Any]:
    """并行采集 unique continuous-action MuJoCo outcomes。"""

    started_at = time.perf_counter()
    if not UNIQUE_ACTION_PATH.exists():
        raise FileNotFoundError("请先运行 prepare 生成 unique continuous actions")
    if smoke and maximum_actions <= 0:
        maximum_actions = 8
    if not smoke and maximum_actions > 0:
        raise ValueError("--max-actions 只允许与 --smoke 一起使用")

    output_path = DATA_ROOT / "smoke" / OUTCOME_PATH.name if smoke else OUTCOME_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_count, last_id = count_existing_outcomes(output_path) if resume else (0, -1)
    if not resume and output_path.exists():
        output_path.unlink()
    start_index = last_id + 1

    batches = iter_action_batches(
        UNIQUE_ACTION_PATH,
        start_index=start_index,
        batch_size=max(1, int(batch_size)),
        maximum_actions=maximum_actions,
    )
    mode = "a" if existing_count > 0 else "w"
    processed = 0
    valid = 0
    with output_path.open(mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTCOME_FIELDS, extrasaction="ignore")
        if existing_count == 0:
            writer.writeheader()

        worker_count = max(1, int(num_workers))
        if worker_count == 1:
            initialise_rollout_worker()
            result_batches = map(process_action_batch, batches)
            pool = None
        else:
            context = mp.get_context("spawn")
            pool = context.Pool(worker_count, initializer=initialise_rollout_worker)
            result_batches = pool.imap(process_action_batch, batches, chunksize=1)
        try:
            for result_batch in result_batches:
                writer.writerows(result_batch)
                handle.flush()
                processed += len(result_batch)
                valid += sum(
                    int(row["quality_pass"]) == 1
                    and int(row["simulation_unstable"]) == 0
                    and int(row["contact_success"]) == 1
                    and int(row["stopped_by_threshold"]) == 1
                    for row in result_batch
                )
                if processed % 1_024 == 0:
                    print(f"collected {existing_count + processed} unique actions")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    elapsed_seconds = time.perf_counter() - started_at
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "car_full_name": CAR_FULL_NAME,
        "friction_cone": FRICTION_CONE,
        "smoke": smoke,
        "num_workers": max(1, int(num_workers)),
        "batch_size": max(1, int(batch_size)),
        "resumed_rows": existing_count,
        "new_rollouts": processed,
        "new_valid_rollouts": valid,
        "elapsed_seconds": elapsed_seconds,
        "new_rollouts_per_second": (
            processed / elapsed_seconds if processed > 0 else 0.0
        ),
        "outcome_path": str(output_path.resolve()),
    }
    summary_path = (
        RESULTS_ROOT / "smoke" / "collection_summary.json"
        if smoke
        else RESULTS_ROOT / "collection_summary.json"
    )
    write_json(summary_path, summary)
    return summary


def load_evaluation_arrays() -> dict[str, Any]:
    """将正式 outcomes 与 candidate mappings 读取为紧凑数组。"""

    if not PREPARATION_SUMMARY_PATH.exists() or not OUTCOME_PATH.exists():
        raise FileNotFoundError("请先完成 prepare 与正式 collect")
    with PREPARATION_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        preparation = json.load(handle)
    action_count = int(preparation["unique_continuous_actions"])
    outcomes = np.empty((action_count, 3), dtype=np.float64)
    quality = np.zeros(action_count, dtype=bool)
    unstable = np.zeros(action_count, dtype=bool)
    contact = np.zeros(action_count, dtype=bool)
    stopped = np.zeros(action_count, dtype=bool)
    keys: list[str] = [""] * action_count
    observed = 0
    with OUTCOME_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            action_id = int(row["physical_action_id"])
            if action_id >= action_count:
                raise RuntimeError(f"physical_action_id 超出 preparation 范围: {action_id}")
            outcomes[action_id] = [
                float(row["real_delta_x"]),
                float(row["real_delta_y"]),
                float(row["real_delta_yaw"]),
            ]
            quality[action_id] = int(row["quality_pass"]) == 1
            unstable[action_id] = int(row["simulation_unstable"]) == 1
            contact[action_id] = int(row["contact_success"]) == 1
            stopped[action_id] = int(row["stopped_by_threshold"]) == 1
            keys[action_id] = row["physical_action_key"]
            observed += 1
    if observed != action_count:
        raise RuntimeError(
            f"formal outcomes 尚未完整: expected={action_count}, observed={observed}"
        )
    valid = quality & ~unstable & contact & stopped

    candidate_ids = np.full((4_536, PRIMARY_SAMPLE_BUDGET), -1, dtype=np.int32)
    with CANDIDATE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            action_index = int(row["source_action_index"])
            sample_index = int(row["sobol_sample_index"])
            candidate_ids[action_index, sample_index] = int(row["physical_action_id"])
    return {
        "outcomes": outcomes,
        "keys": keys,
        "quality": quality,
        "unstable": unstable,
        "contact": contact,
        "stopped": stopped,
        "valid": valid,
        "candidate_ids": candidate_ids,
    }


def choose_continuous_oracle(
    candidate_ids: np.ndarray,
    target_position: np.ndarray,
    arrays: dict[str, Any],
) -> tuple[int, str, float, bool, int]:
    """在一个 target-configuration 中选择真实 continuous oracle。"""

    valid_mask = arrays["valid"][candidate_ids]
    valid_ids = candidate_ids[valid_mask]
    if valid_ids.size == 0:
        return -1, "", math.nan, False, 0
    outcomes = arrays["outcomes"][valid_ids]
    position_error = np.linalg.norm(outcomes[:, :2] - target_position[None, :], axis=1)
    yaw_error = np.abs(wrap_to_pi(outcomes[:, 2]))
    costs = (
        0.5 * position_error / PRIMARY_TNPO_COST.position_tolerance_m
        + 0.5 * yaw_error / PRIMARY_TNPO_COST.yaw_tolerance_rad
    )
    minimum = float(np.min(costs))
    tied = np.flatnonzero(costs == minimum)
    if tied.size == 1:
        local_index = int(tied[0])
    else:
        local_index = min(
            (int(index) for index in tied),
            key=lambda index: arrays["keys"][int(valid_ids[index])],
        )
    physical_id = int(valid_ids[local_index])
    success = bool(
        position_error[local_index] <= PRIMARY_TNPO_COST.position_tolerance_m
        and yaw_error[local_index] <= PRIMARY_TNPO_COST.yaw_tolerance_rad
    )
    return (
        physical_id,
        arrays["keys"][physical_id],
        minimum,
        success,
        int(valid_ids.size),
    )


def build_case_metrics(arrays: dict[str, Any]) -> list[dict[str, Any]]:
    """计算 128 个 targets 与九个 nested configurations 的 case metrics。"""

    targets = sorted(
        read_csv_rows(TARGET_MANIFEST_PATH), key=lambda row: int(row["car_target_index"])
    )
    anchors_by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(ANCHOR_PATH):
        anchors_by_target[row["v2_target_id"]].append(row)
    for rows in anchors_by_target.values():
        rows.sort(key=lambda row: int(row["anchor_rank"]))

    case_rows: list[dict[str, Any]] = []
    for target in targets:
        target_id = target["v2_target_id"]
        target_position = np.asarray(
            [float(target["target_delta_x_m"]), float(target["target_delta_y_m"])],
            dtype=np.float64,
        )
        anchors = anchors_by_target[target_id]
        discrete = anchors[0]
        discrete_cost = float(discrete["discrete_tnpo_cost"])
        discrete_success = int(discrete["discrete_pose_success"]) == 1
        for anchor_budget in ANCHOR_BUDGETS:
            action_indices = np.asarray(
                [int(row["v2_action_id"][1:]) for row in anchors[:anchor_budget]],
                dtype=np.int32,
            )
            for sample_budget in SAMPLE_BUDGETS:
                candidate_ids = arrays["candidate_ids"][action_indices, :sample_budget].reshape(-1)
                if np.any(candidate_ids < 0):
                    raise RuntimeError(
                        f"candidate mapping 不完整: target={target_id}, "
                        f"anchors={anchor_budget}, samples={sample_budget}"
                    )
                (
                    continuous_id,
                    continuous_key,
                    continuous_cost,
                    continuous_success,
                    valid_count,
                ) = choose_continuous_oracle(candidate_ids, target_position, arrays)
                if math.isfinite(continuous_cost):
                    continuous_improvement = discrete_cost - continuous_cost
                    union_cost = min(discrete_cost, continuous_cost)
                    use_continuous = continuous_cost < discrete_cost
                else:
                    continuous_improvement = math.nan
                    union_cost = discrete_cost
                    use_continuous = False
                union_success = continuous_success if use_continuous else discrete_success
                union_improvement = discrete_cost - union_cost
                if not math.isfinite(continuous_improvement):
                    continuous_result = "unavailable"
                elif continuous_improvement >= PRACTICAL_EQUIVALENCE_TOLERANCE:
                    continuous_result = "win"
                elif continuous_improvement <= -PRACTICAL_EQUIVALENCE_TOLERANCE:
                    continuous_result = "loss"
                else:
                    continuous_result = "tie"
                case_rows.append(
                    {
                        "car_target_index": target["car_target_index"],
                        "v2_target_id": target_id,
                        "radial_quartile_index": target["radial_quartile_index"],
                        "angular_quadrant_index": target["angular_quadrant_index"],
                        "anchor_budget": anchor_budget,
                        "candidates_per_anchor": sample_budget,
                        "target_candidate_relationships": anchor_budget * sample_budget,
                        "valid_candidate_relationships": valid_count,
                        "discrete_oracle_action_id": discrete["v2_action_id"],
                        "discrete_oracle_cost": discrete_cost,
                        "discrete_oracle_pose_success": int(discrete_success),
                        "continuous_oracle_physical_action_id": continuous_id,
                        "continuous_oracle_physical_action_key": continuous_key,
                        "continuous_oracle_cost": continuous_cost,
                        "continuous_oracle_pose_success": int(continuous_success),
                        "union_oracle_cost": union_cost,
                        "union_oracle_pose_success": int(union_success),
                        "union_oracle_improvement": union_improvement,
                        "continuous_only_improvement": continuous_improvement,
                        "effective_improvement_0p05": int(
                            union_improvement >= PRIMARY_IMPROVEMENT_THRESHOLD
                        ),
                        "effective_improvement_0p02": int(
                            union_improvement >= SENSITIVITY_IMPROVEMENT_THRESHOLD
                        ),
                        "continuous_only_result": continuous_result,
                    }
                )
    return case_rows


def bootstrap_indices(case_rows: list[dict[str, Any]], resamples: int) -> np.ndarray:
    """生成一次并跨所有 configurations 复用的 stratified bootstrap indices。"""

    primary_rows = [
        row
        for row in case_rows
        if int(row["anchor_budget"]) == PRIMARY_ANCHOR_BUDGET
        and int(row["candidates_per_anchor"]) == PRIMARY_SAMPLE_BUDGET
    ]
    primary_rows.sort(key=lambda row: int(row["car_target_index"]))
    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(primary_rows):
        strata[
            (
                int(row["radial_quartile_index"]),
                int(row["angular_quadrant_index"]),
            )
        ].append(index)
    rng = np.random.default_rng(DEFAULT_SEED)
    sampled_parts: list[np.ndarray] = []
    for radial_index in range(4):
        for angular_index in range(4):
            indices = np.asarray(strata[(radial_index, angular_index)], dtype=np.int32)
            sampled_parts.append(
                rng.choice(indices, size=(resamples, len(indices)), replace=True)
            )
    return np.concatenate(sampled_parts, axis=1)


def percentile_mean_ci(values: np.ndarray, indices: np.ndarray) -> list[float]:
    """计算 target-level bootstrap mean 的 percentile 95% CI。"""

    means = np.mean(values[indices], axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def primary_rollout_ids(case_rows: list[dict[str, Any]], arrays: dict[str, Any]) -> np.ndarray:
    """返回 Primary configuration 使用的 unique physical action ids。"""

    anchors_by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(ANCHOR_PATH):
        anchors_by_target[row["v2_target_id"]].append(row)
    ids: list[np.ndarray] = []
    for target_id, anchors in anchors_by_target.items():
        del target_id
        anchors.sort(key=lambda row: int(row["anchor_rank"]))
        action_indices = np.asarray(
            [int(row["v2_action_id"][1:]) for row in anchors[:PRIMARY_ANCHOR_BUDGET]],
            dtype=np.int32,
        )
        ids.append(
            arrays["candidate_ids"][action_indices, :PRIMARY_SAMPLE_BUDGET].reshape(-1)
        )
    return np.unique(np.concatenate(ids))


def validity_summary(ids: np.ndarray, arrays: dict[str, Any]) -> dict[str, Any]:
    """汇总一组 unique rollouts 的有效性与失败原因。"""

    count = int(ids.size)
    valid_count = int(np.sum(arrays["valid"][ids]))
    return {
        "unique_rollouts": count,
        "valid_rollouts": valid_count,
        "validity_rate": valid_count / count,
        "contact_failure_rate": float(np.mean(~arrays["contact"][ids])),
        "not_stopped_rate": float(np.mean(~arrays["stopped"][ids])),
        "simulation_unstable_rate": float(np.mean(arrays["unstable"][ids])),
        "quality_failure_rate": float(np.mean(~arrays["quality"][ids])),
    }


def evaluate_experiment(bootstrap_resamples: int) -> dict[str, Any]:
    """计算正式 E1 Validation 结果与 Go/No-Go。"""

    arrays = load_evaluation_arrays()
    case_rows = build_case_metrics(arrays)
    write_csv(CASE_METRICS_PATH, case_rows, CASE_METRIC_FIELDS)
    bootstrap = bootstrap_indices(case_rows, bootstrap_resamples)

    configuration_summaries: list[dict[str, Any]] = []
    for anchor_budget in ANCHOR_BUDGETS:
        for sample_budget in SAMPLE_BUDGETS:
            rows = [
                row
                for row in case_rows
                if int(row["anchor_budget"]) == anchor_budget
                and int(row["candidates_per_anchor"]) == sample_budget
            ]
            rows.sort(key=lambda row: int(row["car_target_index"]))
            union_improvement = np.asarray(
                [float(row["union_oracle_improvement"]) for row in rows],
                dtype=np.float64,
            )
            continuous_improvement = np.asarray(
                [float(row["continuous_only_improvement"]) for row in rows],
                dtype=np.float64,
            )
            effective_0p05 = union_improvement >= PRIMARY_IMPROVEMENT_THRESHOLD
            effective_0p02 = union_improvement >= SENSITIVITY_IMPROVEMENT_THRESHOLD
            success_difference = np.asarray(
                [
                    int(row["union_oracle_pose_success"])
                    - int(row["discrete_oracle_pose_success"])
                    for row in rows
                ],
                dtype=np.float64,
            )
            results = [row["continuous_only_result"] for row in rows]
            configuration_summaries.append(
                {
                    "anchor_budget": anchor_budget,
                    "candidates_per_anchor": sample_budget,
                    "target_candidate_relationships": anchor_budget * sample_budget,
                    "mean_union_oracle_improvement": float(np.mean(union_improvement)),
                    "mean_union_oracle_improvement_95_ci": percentile_mean_ci(
                        union_improvement, bootstrap
                    ),
                    "median_union_oracle_improvement": float(
                        np.median(union_improvement)
                    ),
                    "p90_union_oracle_improvement": float(
                        np.percentile(union_improvement, 90.0)
                    ),
                    "minimum_union_oracle_improvement": float(
                        np.min(union_improvement)
                    ),
                    "maximum_union_oracle_improvement": float(
                        np.max(union_improvement)
                    ),
                    "effective_improvement_0p05_rate": float(
                        np.mean(effective_0p05)
                    ),
                    "effective_improvement_0p05_rate_95_ci": percentile_mean_ci(
                        effective_0p05.astype(np.float64), bootstrap
                    ),
                    "effective_improvement_0p02_rate": float(
                        np.mean(effective_0p02)
                    ),
                    "continuous_only_mean_improvement": float(
                        np.nanmean(continuous_improvement)
                    ),
                    "continuous_only_win_rate": results.count("win") / len(results),
                    "continuous_only_tie_rate": results.count("tie") / len(results),
                    "continuous_only_loss_rate": results.count("loss") / len(results),
                    "discrete_one_step_success_rate": float(
                        np.mean(
                            [int(row["discrete_oracle_pose_success"]) for row in rows]
                        )
                    ),
                    "union_one_step_success_rate": float(
                        np.mean([int(row["union_oracle_pose_success"]) for row in rows])
                    ),
                    "paired_success_rate_difference": float(
                        np.mean(success_difference)
                    ),
                    "paired_success_rate_difference_95_ci": percentile_mean_ci(
                        success_difference, bootstrap
                    ),
                    "mean_valid_candidate_relationships": float(
                        np.mean(
                            [int(row["valid_candidate_relationships"]) for row in rows]
                        )
                    ),
                }
            )

    primary = next(
        row
        for row in configuration_summaries
        if row["anchor_budget"] == PRIMARY_ANCHOR_BUDGET
        and row["candidates_per_anchor"] == PRIMARY_SAMPLE_BUDGET
    )
    primary_ids = primary_rollout_ids(case_rows, arrays)
    primary_validity = validity_summary(primary_ids, arrays)
    all_ids = np.arange(arrays["outcomes"].shape[0], dtype=np.int32)
    all_validity = validity_summary(all_ids, arrays)
    go_criteria = {
        "mean_improvement_ci_lower_above_zero": (
            primary["mean_union_oracle_improvement_95_ci"][0] > 0.0
        ),
        "effective_improvement_rate_at_least_0p10": (
            primary["effective_improvement_0p05_rate"] >= 0.10
        ),
        "primary_rollout_validity_at_least_0p99": (
            primary_validity["validity_rate"] >= 0.99
        ),
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "car_full_name": CAR_FULL_NAME,
        "friction_cone": FRICTION_CONE,
        "nominal_condition_id": "J_TRAIN_062",
        "default_seed": DEFAULT_SEED,
        "bootstrap_resamples": bootstrap_resamples,
        "validation_targets": 128,
        "primary_configuration": {
            "anchor_budget": PRIMARY_ANCHOR_BUDGET,
            "candidates_per_anchor": PRIMARY_SAMPLE_BUDGET,
            "effective_improvement_threshold": PRIMARY_IMPROVEMENT_THRESHOLD,
            "sensitivity_improvement_threshold": SENSITIVITY_IMPROVEMENT_THRESHOLD,
        },
        "configuration_summaries": configuration_summaries,
        "primary_rollout_validity": primary_validity,
        "all_sensitivity_rollout_validity": all_validity,
        "go_criteria": go_criteria,
        "go_decision": all(go_criteria.values()),
        "case_metrics_path": str(CASE_METRICS_PATH.resolve()),
    }
    write_json(SUMMARY_PATH, summary)
    return summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="CAR (Continuous Action Refinement) Experiment 1"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="只读计算正式实验规模")
    subparsers.add_parser("prepare", help="生成 targets、anchors 与 continuous candidates")

    collect_parser = subparsers.add_parser("collect", help="采集 continuous MuJoCo outcomes")
    collect_parser.add_argument("--num-workers", type=int, default=8)
    collect_parser.add_argument("--batch-size", type=int, default=32)
    collect_parser.add_argument("--resume", action="store_true")
    collect_parser.add_argument("--smoke", action="store_true")
    collect_parser.add_argument("--max-actions", type=int, default=0)

    evaluate_parser = subparsers.add_parser("evaluate", help="计算 Validation 统计结果")
    evaluate_parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    """执行 CAR Experiment 1 子命令。"""

    args = parse_args()
    print("CAR = Continuous Action Refinement")
    print("Experiment 1 = Continuous Action Oracle Headroom")
    if args.command == "plan":
        payload = plan_experiment()
    elif args.command == "prepare":
        payload = prepare_experiment()
    elif args.command == "collect":
        payload = collect_outcomes(
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            resume=args.resume,
            smoke=args.smoke,
            maximum_actions=args.max_actions,
        )
    elif args.command == "evaluate":
        payload = evaluate_experiment(args.bootstrap_resamples)
    else:
        raise ValueError(f"未知 command: {args.command}")
    print(json.dumps(make_json_compatible(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
