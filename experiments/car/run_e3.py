"""CAR（Continuous Action Refinement）Experiment 3 统一入口。

CAR 是 Continuous Action Refinement 的代码目录缩写。本脚本实现：

- prepare：生成互斥的 Validation/Test targets，并准备 Validation candidates；
- collect-validation / collect-test：并行采集 nominal MuJoCo outcomes；
- evaluate-validation：评价三档 candidate budget 并选择唯一 Test 配置；
- prepare-test：按 Validation 选择的 budget 生成独立 Test candidates；
- evaluate-test：一次性执行 Independent Test 与 Go/No-Go 判断。
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

import numpy as np
import torch
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_e1 as car_e1
import run_e2 as car_e2
from push_core.hcr_v2.e1 import read_csv_rows, wrap_to_pi


PROTOCOL_VERSION = "continuous_action_refinement_e3_v1"
CAR_FULL_NAME = "Continuous Action Refinement"
EXPERIMENT_NAME = "Bounded Continuous Action Selection"
FRICTION_CONE = "elliptic"
NOMINAL_CONDITION_ID = "J_TRAIN_062"
DEFAULT_SEED = 0

ACTION_COUNT = 4_536
TARGETS_PER_ROLE = 128
ANCHOR_BUDGET = 20
CANDIDATE_BUDGETS = (128, 256, 512)
MAXIMUM_CANDIDATE_BUDGET = 512
PROMOTION_MARGIN = 0.05
PRACTICAL_TOLERANCE = 0.05
SENSITIVITY_TOLERANCE = 0.02
VALIDITY_THRESHOLD = 0.99
EFFECTIVE_WIN_RATE_THRESHOLD = 0.10
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
SOBOL_ROLE_START = {"validation": 512, "test": 1_024}

MANIFEST_ROOT = REPOSITORY_ROOT / "manifests" / "car"
HCR_MANIFEST_ROOT = (
    REPOSITORY_ROOT / "manifests" / "hcr_v2"
)
ACTION_MANIFEST_PATH = HCR_MANIFEST_ROOT / "hcr_v2_action_core_manifest_v1.csv"
TARGET_SOURCE_PATH = HCR_MANIFEST_ROOT / "hcr_v2_core_target_manifest_v1.csv"
E1_TARGET_PATH = MANIFEST_ROOT / "continuous_action_refinement_validation_targets.csv"
E2_VALIDATION_TARGET_PATH = (
    MANIFEST_ROOT / "continuous_action_refinement_e2_validation_target_queries.csv"
)
E2_TEST_TARGET_PATH = (
    MANIFEST_ROOT / "continuous_action_refinement_e2_test_target_queries.csv"
)
VALIDATION_TARGET_PATH = (
    MANIFEST_ROOT / "continuous_action_refinement_e3_validation_target_queries.csv"
)
TEST_TARGET_PATH = (
    MANIFEST_ROOT / "continuous_action_refinement_e3_test_target_queries.csv"
)

DATA_ROOT = PROJECT_ROOT / "data" / "car" / "experiment_3"
RESULTS_ROOT = PROJECT_ROOT / "results" / "car" / "experiment_3"
DIRECT_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "car"
    / "experiment_2"
    / "models"
    / "direct_continuous_outcome_model.pt"
)
ANCHORED_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "car"
    / "experiment_2"
    / "models"
    / "p1_anchored_continuous_residual_model.pt"
)

TARGET_FIELDS = [
    "e3_target_index",
    "v2_target_id",
    "evaluation_role",
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
    "e3_target_index",
    "v2_target_id",
    "evaluation_role",
    "radial_quartile_index",
    "angular_quadrant_index",
    "anchor_rank",
    "v2_action_id",
    "source_action_index",
    "predicted_discrete_tnpo_cost",
]
CANDIDATE_FIELDS = [
    *car_e1.UNIQUE_ACTION_FIELDS,
    "evaluation_role",
    "role_candidate_rank",
    "minimum_candidate_budget",
]
OUTCOME_FIELDS = [
    "experiment_id",
    "protocol_version",
    "car_full_name",
    "experiment_name",
    "friction_cone",
    "condition_id",
    *CANDIDATE_FIELDS,
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
CASE_FIELDS = [
    "e3_target_index",
    "v2_target_id",
    "evaluation_role",
    "radial_quartile_index",
    "angular_quadrant_index",
    "candidates_per_anchor",
    "method_id",
    "method_name",
    "selected_action_id",
    "selected_physical_action_key",
    "selected_continuous",
    "selected_valid",
    "selected_predicted_tnpo_cost",
    "selected_real_tnpo_cost",
    "m0_real_tnpo_cost",
    "real_improvement_over_m0",
    "effective_result_over_m0",
    "continuous_selection_effective_win",
    "continuous_selection_false_promotion",
    "selected_action_optimism",
    "union_oracle_action_id",
    "union_oracle_physical_action_key",
    "union_oracle_cost",
    "union_oracle_improvement_over_m0",
    "selection_gap",
    "near_optimal_0p05",
    "near_optimal_0p02",
    "oracle_headroom_utilisation_raw",
    "oracle_headroom_utilisation_clipped_to_zero",
    "one_step_pose_success",
    "selected_position_error_mm",
    "selected_yaw_error_deg",
]

METHOD_NAMES = {
    "M0": "Full-Discrete P1 Selector",
    "M1": "Unprotected Direct Continuous Argmin",
    "M2": "Protected Direct Bounded Continuous Selector",
    "M3": "Protected P1-Anchored Continuous Selector",
}

_WORKER_ROLE = ""


def make_json_compatible(value: Any) -> Any:
    """把 NumPy 类型与非有限浮点数转换为严格 JSON 值。"""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): make_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
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


def role_paths(role: str) -> dict[str, Path]:
    """返回一个 evaluation role 的正式 artifact 路径。"""

    role_root = DATA_ROOT / role
    evaluation_root = RESULTS_ROOT / "evaluation" / role
    return {
        "target": VALIDATION_TARGET_PATH if role == "validation" else TEST_TARGET_PATH,
        "anchor": role_root / "continuous_action_refinement_anchors.csv",
        "candidate": role_root / "continuous_action_refinement_candidates.csv",
        "outcome": role_root / "continuous_action_refinement_outcomes.csv",
        "preparation_summary": RESULTS_ROOT / f"{role}_preparation_summary.json",
        "collection_summary": RESULTS_ROOT / f"{role}_collection_summary.json",
        "evaluation_root": evaluation_root,
        "case": evaluation_root / "target_cases.csv",
        "summary": evaluation_root / "summary.json",
    }


def angular_quadrant(delta_x: float, delta_y: float) -> int:
    """按固定坐标轴边界返回 angular quadrant。"""

    if delta_x >= 0.0 and delta_y >= 0.0:
        return 0
    if delta_x < 0.0 and delta_y >= 0.0:
        return 1
    if delta_x < 0.0 and delta_y < 0.0:
        return 2
    return 3


def build_targets() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """生成与 E1/E2 完全互斥的 E3 Validation/Test targets。"""

    excluded_ids: set[str] = set()
    for path in (E1_TARGET_PATH, E2_VALIDATION_TARGET_PATH, E2_TEST_TARGET_PATH):
        excluded_ids.update(row["v2_target_id"] for row in read_csv_rows(path))

    candidates: list[dict[str, Any]] = []
    for row in read_csv_rows(TARGET_SOURCE_PATH):
        if row["split_role"] != "training" or row["v2_target_id"] in excluded_ids:
            continue
        delta_x = float(row["target_delta_x_m"])
        delta_y = float(row["target_delta_y_m"])
        candidates.append(
            {
                **row,
                "radial_distance_m": math.hypot(delta_x, delta_y),
                "polar_angle_rad": math.atan2(delta_y, delta_x),
                "angular_quadrant_index": angular_quadrant(delta_x, delta_y),
            }
        )
    candidates.sort(
        key=lambda row: (
            float(row["radial_distance_m"]),
            float(row["polar_angle_rad"]),
            row["v2_target_id"],
        )
    )
    if len(candidates) != 2_884 or len(candidates) % 4:
        raise RuntimeError(f"E3 eligible target 数量错误: {len(candidates)}")
    quartile_size = len(candidates) // 4
    for rank, row in enumerate(candidates):
        row["radial_quartile_index"] = rank // quartile_size

    strata: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        strata[
            (
                int(row["radial_quartile_index"]),
                int(row["angular_quadrant_index"]),
            )
        ].append(row)

    role_rows: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
    stratum_counts: dict[str, int] = {}
    for radial_index in range(4):
        for angular_index in range(4):
            ordered = sorted(
                strata[(radial_index, angular_index)],
                key=lambda row: row["v2_target_id"],
            )
            stratum_counts[f"{radial_index}_{angular_index}"] = len(ordered)
            permutation = np.random.default_rng(
                np.random.SeedSequence(
                    [DEFAULT_SEED, 3, radial_index, angular_index]
                )
            ).permutation(len(ordered))
            for role, indices in (
                ("validation", permutation[:8]),
                ("test", permutation[8:16]),
            ):
                for selected_index in indices:
                    source = ordered[int(selected_index)]
                    role_rows[role].append(
                        {
                            "v2_target_id": source["v2_target_id"],
                            "evaluation_role": role,
                            "source_split_role": source["split_role"],
                            "target_delta_x_m": f"{float(source['target_delta_x_m']):.4f}",
                            "target_delta_y_m": f"{float(source['target_delta_y_m']):.4f}",
                            "canonical_position_key": source["canonical_position_key"],
                            "radial_distance_m": f"{float(source['radial_distance_m']):.12f}",
                            "polar_angle_rad": f"{float(source['polar_angle_rad']):.12f}",
                            "radial_quartile_index": radial_index,
                            "angular_quadrant_index": angular_index,
                            "selection_seed": DEFAULT_SEED,
                        }
                    )

    outputs: dict[str, list[dict[str, Any]]] = {}
    for role, rows in role_rows.items():
        rows.sort(key=lambda row: row["v2_target_id"])
        outputs[role] = [
            {**row, "e3_target_index": index} for index, row in enumerate(rows)
        ]
        if len(outputs[role]) != TARGETS_PER_ROLE:
            raise RuntimeError(f"{role} target 数量错误: {len(outputs[role])}")
    audit = {
        "excluded_unique_targets": len(excluded_ids),
        "eligible_targets": len(candidates),
        "stratum_counts": stratum_counts,
    }
    return outputs["validation"], outputs["test"], audit


def load_actions() -> list[dict[str, str]]:
    """读取固定 action_core manifest。"""

    actions = sorted(
        read_csv_rows(ACTION_MANIFEST_PATH), key=lambda row: row["v2_action_id"]
    )
    if len(actions) != ACTION_COUNT:
        raise RuntimeError(f"action_core 数量错误: {len(actions)}")
    return actions


def build_anchor_rows(
    role: str,
    targets: list[dict[str, str]],
    actions: list[dict[str, str]],
    p1_outcomes: np.ndarray,
) -> list[dict[str, Any]]:
    """使用完整离散 P1 预测为每个 target 选择 Top-20 anchors。"""

    action_ids = np.asarray([row["v2_action_id"] for row in actions], dtype=str)
    rows: list[dict[str, Any]] = []
    for target in targets:
        target_xy = np.asarray(
            [float(target["target_delta_x_m"]), float(target["target_delta_y_m"])],
            dtype=np.float32,
        )
        costs = car_e2.tnpo_cost(p1_outcomes, target_xy)
        order = np.lexsort((action_ids, costs))[:ANCHOR_BUDGET]
        for rank, action_index in enumerate(order, start=1):
            rows.append(
                {
                    "e3_target_index": int(target["e3_target_index"]),
                    "v2_target_id": target["v2_target_id"],
                    "evaluation_role": role,
                    "radial_quartile_index": int(target["radial_quartile_index"]),
                    "angular_quadrant_index": int(target["angular_quadrant_index"]),
                    "anchor_rank": rank,
                    "v2_action_id": actions[int(action_index)]["v2_action_id"],
                    "source_action_index": int(action_index),
                    "predicted_discrete_tnpo_cost": float(costs[action_index]),
                }
            )
    return rows


def generate_role_candidates(
    anchor: dict[str, str], role: str, candidate_budget: int
) -> Iterator[dict[str, Any]]:
    """从固定 2,048-point Sobol sequence 中提取一个 E3 role 区段。"""

    action_index = int(anchor["v2_action_id"][1:])
    surface_id = int(anchor["surface_id"])
    contact_col = int(anchor["contact_region_col"])
    tangent_is_y = surface_id in {0, 1}
    tangent_center = float(
        anchor["contact_point_local_y" if tangent_is_y else "contact_point_local_x"]
    )
    contact_low, contact_high = car_e1._CONTACT_INTERVALS[contact_col]
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
    alpha = ndtr((lower - centers) / car_e1._GRID_SIGMA)
    beta = ndtr((upper - centers) / car_e1._GRID_SIGMA)
    sampling_seed = car_e1.stable_anchor_seed(action_index)
    sobol = qmc.Sobol(d=6, scramble=True, seed=sampling_seed)
    all_points = sobol.random_base2(m=11)
    start = SOBOL_ROLE_START[role]
    unit_points = all_points[start : start + candidate_budget]
    probabilities = alpha[None, :] + unit_points * (beta - alpha)[None, :]
    samples = centers[None, :] + car_e1._GRID_SIGMA[None, :] * ndtri(probabilities)
    samples = np.round(samples, decimals=8)

    normal_x = float(anchor["contact_normal_local_x"])
    normal_y = float(anchor["contact_normal_local_y"])
    fixed_x = round(float(anchor["contact_point_local_x"]), 8)
    fixed_y = round(float(anchor["contact_point_local_y"]), 8)
    for role_rank, values in enumerate(samples):
        tangent, angle_deg, force, ramp_up, hold, ramp_down = [
            float(value) for value in values
        ]
        contact_x = fixed_x if tangent_is_y else tangent
        contact_y = tangent if tangent_is_y else fixed_y
        direction_x, direction_y = car_e1.rotate_inward_direction(
            normal_x, normal_y, angle_deg
        )
        yield {
            "source_v2_action_id": anchor["v2_action_id"],
            "source_action_index": action_index,
            "source_candidate_id": anchor["candidate_id"],
            "source_action_param_index": anchor["action_param_index"],
            "source_contact_region_id": anchor["contact_region_id"],
            "sobol_sample_index": start + role_rank,
            "evaluation_role": role,
            "role_candidate_rank": role_rank,
            "minimum_candidate_budget": (
                128 if role_rank < 128 else 256 if role_rank < 256 else 512
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


def prepare_role(
    role: str,
    targets: list[dict[str, str]],
    actions: list[dict[str, str]],
    p1_outcomes: np.ndarray,
    candidate_budget: int,
) -> dict[str, Any]:
    """生成一个 role 的 Top-20 anchors 与 unique continuous candidates。"""

    paths = role_paths(role)
    anchor_rows = build_anchor_rows(role, targets, actions, p1_outcomes)
    write_csv(paths["anchor"], anchor_rows, ANCHOR_FIELDS)
    unique_anchor_indices = sorted(
        {int(row["source_action_index"]) for row in anchor_rows}
    )
    paths["candidate"].parent.mkdir(parents=True, exist_ok=True)
    with paths["candidate"].open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CANDIDATE_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        physical_action_id = 0
        for anchor_number, action_index in enumerate(unique_anchor_indices, start=1):
            for candidate in generate_role_candidates(
                actions[action_index], role, candidate_budget
            ):
                writer.writerow(
                    {
                        **candidate,
                        "physical_action_id": physical_action_id,
                        "physical_action_key": car_e1.physical_action_key(candidate),
                    }
                )
                physical_action_id += 1
            if anchor_number % 100 == 0 or anchor_number == len(unique_anchor_indices):
                print(
                    f"prepared {role} anchor {anchor_number}/{len(unique_anchor_indices)}; "
                    f"candidates={physical_action_id}"
                )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "car_full_name": CAR_FULL_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "friction_cone": FRICTION_CONE,
        "nominal_condition_id": NOMINAL_CONDITION_ID,
        "default_seed": DEFAULT_SEED,
        "role": role,
        "targets": len(targets),
        "anchors_per_target": ANCHOR_BUDGET,
        "anchor_rows": len(anchor_rows),
        "unique_anchors": len(unique_anchor_indices),
        "candidates_per_anchor": candidate_budget,
        "unique_continuous_actions": len(unique_anchor_indices) * candidate_budget,
        "sobol_index_start": SOBOL_ROLE_START[role],
        "sobol_index_stop_exclusive": SOBOL_ROLE_START[role] + candidate_budget,
        "target_path": str(paths["target"].resolve()),
        "anchor_path": str(paths["anchor"].resolve()),
        "candidate_path": str(paths["candidate"].resolve()),
    }
    write_json(paths["preparation_summary"], summary)
    return summary


def prepare_validation() -> dict[str, Any]:
    """生成 E3 targets 和完整 Validation candidate-budget artifacts。"""

    validation_targets, test_targets, audit = build_targets()
    write_csv(VALIDATION_TARGET_PATH, validation_targets, TARGET_FIELDS)
    write_csv(TEST_TARGET_PATH, test_targets, TARGET_FIELDS)
    actions = load_actions()
    p1_outcomes = car_e2.nominal_p1_outcomes(actions)
    summary = prepare_role(
        "validation",
        validation_targets,
        actions,
        p1_outcomes,
        MAXIMUM_CANDIDATE_BUDGET,
    )
    summary["target_exclusion_and_stratification_audit"] = audit
    write_json(role_paths("validation")["preparation_summary"], summary)
    return summary


def prepare_test() -> dict[str, Any]:
    """读取 Validation 选定配置并生成 Independent Test candidates。"""

    selected_path = RESULTS_ROOT / "evaluation" / "selected_configuration.json"
    if not selected_path.exists():
        raise FileNotFoundError("请先完成 evaluate-validation 并选择 Test budget")
    with selected_path.open("r", encoding="utf-8") as handle:
        selected = json.load(handle)
    candidate_budget = selected.get("selected_candidates_per_anchor")
    if candidate_budget not in CANDIDATE_BUDGETS:
        raise RuntimeError("Validation 未产生可进入 Test 的 candidate budget")
    targets = read_csv_rows(TEST_TARGET_PATH)
    actions = load_actions()
    p1_outcomes = car_e2.nominal_p1_outcomes(actions)
    return prepare_role(
        "test", targets, actions, p1_outcomes, int(candidate_budget)
    )


def initialise_rollout_worker(role: str) -> None:
    """初始化 Elliptic nominal-condition MuJoCo worker。"""

    global _WORKER_ROLE
    _WORKER_ROLE = role
    car_e1.initialise_rollout_worker()


def process_action_batch(actions: list[dict[str, str]]) -> list[dict[str, Any]]:
    """执行一批 continuous actions 并写入 E3 正式元数据。"""

    rows = car_e1.process_action_batch(actions)
    for row in rows:
        row["experiment_id"] = "E3"
        row["protocol_version"] = PROTOCOL_VERSION
        row["experiment_name"] = EXPERIMENT_NAME
        row["evaluation_role"] = _WORKER_ROLE
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
    path: Path, start_id: int, batch_size: int, maximum_actions: int
) -> Iterator[list[dict[str, str]]]:
    """按 physical action id 流式生成 collection batches。"""

    batch: list[dict[str, str]] = []
    yielded = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["physical_action_id"]) < start_id:
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


def collect_role(
    role: str,
    num_workers: int,
    batch_size: int,
    resume: bool,
    smoke: bool,
    maximum_actions: int,
) -> dict[str, Any]:
    """并行采集一个 role 的 unique continuous-action outcomes。"""

    paths = role_paths(role)
    if not paths["candidate"].exists():
        raise FileNotFoundError(f"请先生成 {role} candidates")
    if smoke and maximum_actions <= 0:
        maximum_actions = 1_024
    if not smoke and maximum_actions > 0:
        raise ValueError("--max-actions 只允许与 --smoke 一起使用")
    output_path = (
        DATA_ROOT / "smoke" / role / paths["outcome"].name
        if smoke
        else paths["outcome"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_count, last_id = count_existing_outcomes(output_path) if resume else (0, -1)
    if not resume and output_path.exists():
        output_path.unlink()
    batches = iter_action_batches(
        paths["candidate"],
        last_id + 1,
        max(1, int(batch_size)),
        maximum_actions,
    )

    started_at = time.perf_counter()
    processed = 0
    valid = 0
    mode = "a" if existing_count else "w"
    with output_path.open(mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTCOME_FIELDS, extrasaction="ignore")
        if not existing_count:
            writer.writeheader()
        worker_count = max(1, int(num_workers))
        if worker_count == 1:
            initialise_rollout_worker(role)
            result_batches = map(process_action_batch, batches)
            pool = None
        else:
            context = mp.get_context("spawn")
            pool = context.Pool(
                worker_count, initializer=initialise_rollout_worker, initargs=(role,)
            )
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
                if processed % 2_048 == 0:
                    print(f"collected {role} {existing_count + processed} actions")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    elapsed = time.perf_counter() - started_at
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "car_full_name": CAR_FULL_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "friction_cone": FRICTION_CONE,
        "role": role,
        "smoke": smoke,
        "num_workers": max(1, int(num_workers)),
        "batch_size": max(1, int(batch_size)),
        "resumed_rows": existing_count,
        "new_rollouts": processed,
        "new_valid_rollouts": valid,
        "elapsed_seconds": elapsed,
        "new_rollouts_per_second": processed / elapsed if processed else 0.0,
        "outcome_path": str(output_path.resolve()),
    }
    summary_path = (
        RESULTS_ROOT / "smoke" / role / "collection_summary.json"
        if smoke
        else paths["collection_summary"]
    )
    write_json(summary_path, summary)
    return summary


def load_nominal_discrete() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取全部离散 actions 的 P1 predictions 与真实 nominal outcomes。"""

    actions = load_actions()
    p1 = car_e2.nominal_p1_outcomes(actions)
    rows = sorted(
        read_csv_rows(car_e2.NOMINAL_DISCRETE_OUTCOME_PATH),
        key=lambda row: row["v2_action_id"],
    )
    if len(rows) != ACTION_COUNT:
        raise RuntimeError(f"nominal discrete outcome 数量错误: {len(rows)}")
    truth = np.asarray(
        [
            [
                float(row["real_delta_x"]),
                float(row["real_delta_y"]),
                float(row["real_delta_yaw"]),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    valid = np.asarray(
        [
            int(row["quality_pass"]) == 1
            and int(row["simulation_unstable"]) == 0
            and int(row["contact_success"]) == 1
            and int(row["stopped_by_threshold"]) == 1
            for row in rows
        ],
        dtype=bool,
    )
    return p1, truth, valid


def load_continuous_arrays(role: str) -> dict[str, Any]:
    """将一个 role 的 candidates 与真实 outcomes 读取为紧凑数组。"""

    paths = role_paths(role)
    if not paths["outcome"].exists():
        raise FileNotFoundError(f"请先完成 {role} 正式 collection")
    rows = read_csv_rows(paths["outcome"])
    rows.sort(key=lambda row: int(row["physical_action_id"]))
    for expected_id, row in enumerate(rows):
        if int(row["physical_action_id"]) != expected_id:
            raise RuntimeError(f"{role} outcomes 不完整或顺序错误: row={expected_id}")
    return {
        "rows": rows,
        "source": np.asarray(
            [int(row["source_action_index"]) for row in rows], dtype=np.int32
        ),
        "role_rank": np.asarray(
            [int(row["role_candidate_rank"]) for row in rows], dtype=np.int16
        ),
        "region": np.asarray(
            [
                int(row["surface_id"]) * 3 + int(row["contact_region_col"])
                for row in rows
            ],
            dtype=np.uint8,
        ),
        "continuous_q": np.asarray(
            [car_e2.action_coordinate(row) for row in rows], dtype=np.float32
        ),
        "truth": np.asarray(
            [
                [
                    float(row["real_delta_x"]),
                    float(row["real_delta_y"]),
                    float(row["real_delta_yaw"]),
                ]
                for row in rows
            ],
            dtype=np.float32,
        ),
        "valid": np.asarray(
            [
                int(row["quality_pass"]) == 1
                and int(row["simulation_unstable"]) == 0
                and int(row["contact_success"]) == 1
                and int(row["stopped_by_threshold"]) == 1
                for row in rows
            ],
            dtype=bool,
        ),
        "keys": np.asarray([row["physical_action_key"] for row in rows], dtype=str),
    }


def predict_continuous_outcomes(
    arrays: dict[str, Any],
    p1_outcomes: np.ndarray,
    actions: list[dict[str, str]],
) -> tuple[dict[str, np.ndarray], str]:
    """在 CUDA 上批量执行 Direct 与 P1-Anchored outcome inference。"""

    device = car_e2.require_cuda()
    anchor_q = np.asarray(
        [car_e2.action_coordinate(row) for row in actions], dtype=np.float32
    )
    one_hot = np.eye(12, dtype=np.float32)[arrays["region"]]
    source = arrays["source"]
    continuous_q = arrays["continuous_q"]
    predictions: dict[str, np.ndarray] = {}
    for method, checkpoint_path in (
        ("direct", DIRECT_MODEL_PATH),
        ("anchored", ANCHORED_MODEL_PATH),
    ):
        model, normalisers, model_type = car_e2.load_checkpoint_model(
            checkpoint_path, device
        )
        if model_type == "direct":
            features = np.concatenate(
                [
                    one_hot,
                    (continuous_q - car_e2.ACTION_CENTRES)
                    / car_e2.ACTION_SCALES,
                ],
                axis=1,
            ).astype(np.float32)
        else:
            features = np.concatenate(
                [
                    one_hot,
                    (anchor_q[source] - car_e2.ACTION_CENTRES)
                    / car_e2.ACTION_SCALES,
                    (continuous_q - anchor_q[source]) / car_e2.OFFSET_SCALES,
                    (p1_outcomes[source] - normalisers["p1_mean"])
                    / normalisers["p1_scale"],
                ],
                axis=1,
            ).astype(np.float32)
        tensor = torch.from_numpy(features).to(device)
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(features), car_e2.EVALUATION_BATCH_SIZE):
                stop = min(len(features), start + car_e2.EVALUATION_BATCH_SIZE)
                with car_e2.autocast_context():
                    scaled = model(tensor[start:stop])
                batches.append(scaled.float().cpu().numpy())
        prediction = (
            np.concatenate(batches, axis=0) * normalisers["label_scale"]
            + normalisers["label_mean"]
        )
        if model_type == "anchored":
            prediction += p1_outcomes[source]
        predictions[method] = prediction.astype(np.float32)
        del model, tensor
        torch.cuda.empty_cache()
    return predictions, torch.cuda.get_device_name(device)


def first_by_cost(costs: np.ndarray, keys: np.ndarray) -> int:
    """按 cost 后 action key 的稳定顺序返回首个 index。"""

    return int(np.lexsort((keys, costs))[0])


def selected_case(
    method_id: str,
    target: dict[str, str],
    budget: int,
    target_xy: np.ndarray,
    m0_index: int,
    p1_costs: np.ndarray,
    discrete_truth: np.ndarray,
    discrete_valid: np.ndarray,
    candidate_indices: np.ndarray,
    arrays: dict[str, Any],
    prediction: np.ndarray | None,
    protected: bool,
    union: dict[str, Any],
) -> dict[str, Any]:
    """构造一个 selector method 的 target-level 真实评价记录。"""

    m0_real_outcome = discrete_truth[m0_index]
    m0_real_cost = float(car_e2.tnpo_cost(m0_real_outcome[None, :], target_xy)[0])
    selected_continuous = False
    selected_index = -1
    selected_predicted_cost = float(p1_costs[m0_index])
    selected_real_outcome = m0_real_outcome
    selected_valid = bool(discrete_valid[m0_index])
    selected_action_id = f"A{m0_index:04d}"
    selected_key = selected_action_id

    if method_id != "M0":
        assert prediction is not None
        predicted_costs = car_e2.tnpo_cost(prediction[candidate_indices], target_xy)
        candidate_keys = arrays["keys"][candidate_indices]
        best_local = first_by_cost(predicted_costs, candidate_keys)
        predicted_improvement = float(p1_costs[m0_index] - predicted_costs[best_local])
        promote = predicted_improvement >= PROMOTION_MARGIN if protected else predicted_improvement > 0.0
        if promote:
            selected_continuous = True
            selected_index = int(candidate_indices[best_local])
            selected_predicted_cost = float(predicted_costs[best_local])
            selected_real_outcome = arrays["truth"][selected_index]
            selected_valid = bool(arrays["valid"][selected_index])
            selected_action_id = f"continuous:{int(arrays['source'][selected_index])}:{int(arrays['role_rank'][selected_index])}"
            selected_key = str(arrays["keys"][selected_index])

    selected_real_cost = float(
        car_e2.tnpo_cost(selected_real_outcome[None, :], target_xy)[0]
    )
    improvement = m0_real_cost - selected_real_cost
    if improvement >= PRACTICAL_TOLERANCE:
        result = "win"
    elif improvement <= -PRACTICAL_TOLERANCE:
        result = "loss"
    else:
        result = "tie"
    selection_gap = selected_real_cost - float(union["cost"])
    headroom = float(union["improvement"])
    utilisation = improvement / headroom if headroom >= PRACTICAL_TOLERANCE else math.nan
    optimism = (
        selected_real_cost - selected_predicted_cost
        if selected_continuous
        else math.nan
    )
    return {
        "e3_target_index": int(target["e3_target_index"]),
        "v2_target_id": target["v2_target_id"],
        "evaluation_role": target["evaluation_role"],
        "radial_quartile_index": int(target["radial_quartile_index"]),
        "angular_quadrant_index": int(target["angular_quadrant_index"]),
        "candidates_per_anchor": budget,
        "method_id": method_id,
        "method_name": METHOD_NAMES[method_id],
        "selected_action_id": selected_action_id,
        "selected_physical_action_key": selected_key,
        "selected_continuous": int(selected_continuous),
        "selected_valid": int(selected_valid),
        "selected_predicted_tnpo_cost": selected_predicted_cost,
        "selected_real_tnpo_cost": selected_real_cost,
        "m0_real_tnpo_cost": m0_real_cost,
        "real_improvement_over_m0": improvement,
        "effective_result_over_m0": result,
        "continuous_selection_effective_win": (
            int(improvement >= PRACTICAL_TOLERANCE) if selected_continuous else ""
        ),
        "continuous_selection_false_promotion": (
            int(improvement <= -PRACTICAL_TOLERANCE) if selected_continuous else ""
        ),
        "selected_action_optimism": optimism,
        "union_oracle_action_id": union["action_id"],
        "union_oracle_physical_action_key": union["key"],
        "union_oracle_cost": union["cost"],
        "union_oracle_improvement_over_m0": headroom,
        "selection_gap": selection_gap,
        "near_optimal_0p05": int(selection_gap <= PRACTICAL_TOLERANCE),
        "near_optimal_0p02": int(selection_gap <= SENSITIVITY_TOLERANCE),
        "oracle_headroom_utilisation_raw": utilisation,
        "oracle_headroom_utilisation_clipped_to_zero": (
            max(0.0, utilisation) if math.isfinite(utilisation) else math.nan
        ),
        "one_step_pose_success": int(car_e2.pose_success(selected_real_outcome, target_xy)),
        "selected_position_error_mm": float(
            np.linalg.norm(selected_real_outcome[:2] - target_xy) * 1_000.0
        ),
        "selected_yaw_error_deg": float(
            np.degrees(abs(float(wrap_to_pi(selected_real_outcome[2]))))
        ),
    }


def build_case_rows(
    role: str,
    budgets: tuple[int, ...],
    arrays: dict[str, Any],
    predictions: dict[str, np.ndarray],
    p1_outcomes: np.ndarray,
    discrete_truth: np.ndarray,
    discrete_valid: np.ndarray,
) -> list[dict[str, Any]]:
    """执行 M0–M3 selection，并用真实 outcomes 评价每个 target。"""

    targets = read_csv_rows(role_paths(role)["target"])
    anchor_rows = read_csv_rows(role_paths(role)["anchor"])
    anchors_by_target: dict[int, list[int]] = defaultdict(list)
    for row in anchor_rows:
        anchors_by_target[int(row["e3_target_index"])].append(
            int(row["source_action_index"])
        )
    candidate_lookup: dict[tuple[int, int], int] = {
        (int(source), int(rank)): index
        for index, (source, rank) in enumerate(zip(arrays["source"], arrays["role_rank"]))
    }
    discrete_keys = np.asarray([f"A{index:04d}" for index in range(ACTION_COUNT)])
    rows: list[dict[str, Any]] = []
    for target_number, target in enumerate(targets, start=1):
        target_index = int(target["e3_target_index"])
        target_xy = np.asarray(
            [float(target["target_delta_x_m"]), float(target["target_delta_y_m"])],
            dtype=np.float32,
        )
        p1_costs = car_e2.tnpo_cost(p1_outcomes, target_xy)
        m0_index = first_by_cost(p1_costs, discrete_keys)
        discrete_real_costs = car_e2.tnpo_cost(discrete_truth, target_xy)
        discrete_real_costs[~discrete_valid] = np.inf
        discrete_oracle_index = first_by_cost(discrete_real_costs, discrete_keys)
        m0_real_cost = float(
            car_e2.tnpo_cost(discrete_truth[m0_index : m0_index + 1], target_xy)[0]
        )
        anchors = anchors_by_target[target_index]
        if len(anchors) != ANCHOR_BUDGET:
            raise RuntimeError(f"target {target_index} 的 Top-20 anchors 不完整")

        for budget in budgets:
            candidate_indices = np.asarray(
                [
                    candidate_lookup[(anchor, role_rank)]
                    for anchor in anchors
                    for role_rank in range(budget)
                ],
                dtype=np.int32,
            )
            candidate_real_costs = car_e2.tnpo_cost(
                arrays["truth"][candidate_indices], target_xy
            )
            candidate_real_costs[~arrays["valid"][candidate_indices]] = np.inf
            continuous_oracle_local = first_by_cost(
                candidate_real_costs, arrays["keys"][candidate_indices]
            )
            continuous_oracle_index = int(candidate_indices[continuous_oracle_local])
            if candidate_real_costs[continuous_oracle_local] < discrete_real_costs[discrete_oracle_index]:
                union_cost = float(candidate_real_costs[continuous_oracle_local])
                union_action_id = (
                    f"continuous:{int(arrays['source'][continuous_oracle_index])}:"
                    f"{int(arrays['role_rank'][continuous_oracle_index])}"
                )
                union_key = str(arrays["keys"][continuous_oracle_index])
            else:
                union_cost = float(discrete_real_costs[discrete_oracle_index])
                union_action_id = f"A{discrete_oracle_index:04d}"
                union_key = union_action_id
            union = {
                "cost": union_cost,
                "action_id": union_action_id,
                "key": union_key,
                "improvement": m0_real_cost - union_cost,
            }
            for method_id, prediction, protected in (
                ("M0", None, False),
                ("M1", predictions["direct"], False),
                ("M2", predictions["direct"], True),
                ("M3", predictions["anchored"], True),
            ):
                rows.append(
                    selected_case(
                        method_id,
                        target,
                        budget,
                        target_xy,
                        m0_index,
                        p1_costs,
                        discrete_truth,
                        discrete_valid,
                        candidate_indices,
                        arrays,
                        prediction,
                        protected,
                        union,
                    )
                )
        print(f"evaluated {role} target {target_number}/{len(targets)}")
    return rows


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    """计算 mean、median 与 P90。"""

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def bootstrap_mean_ci(
    values: np.ndarray, strata: np.ndarray, resamples: int
) -> list[float]:
    """返回 target-level stratified-bootstrap mean 的 95% 区间。"""

    bootstrap = car_e2.stratified_bootstrap_means(values, strata, resamples)
    return car_e2.bootstrap_ci(bootstrap)


def bootstrap_ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    strata: np.ndarray,
    resamples: int,
) -> list[float]:
    """对 target-level 分子和分母执行分层 bootstrap 比率统计。"""

    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    strata = np.asarray(strata, dtype=np.int64)
    rng = np.random.default_rng(DEFAULT_SEED)
    numerator_sums = np.zeros(resamples, dtype=np.float64)
    denominator_sums = np.zeros(resamples, dtype=np.float64)
    for stratum_id in sorted(np.unique(strata)):
        indices = np.flatnonzero(strata == stratum_id)
        draws = rng.integers(0, len(indices), size=(resamples, len(indices)))
        sampled = indices[draws]
        numerator_sums += numerator[sampled].sum(axis=1)
        denominator_sums += denominator[sampled].sum(axis=1)
    valid = denominator_sums > 0.0
    if not np.any(valid):
        return [math.nan, math.nan]
    return car_e2.bootstrap_ci(numerator_sums[valid] / denominator_sums[valid])


def method_summary(
    rows: list[dict[str, Any]], strata: np.ndarray, resamples: int
) -> dict[str, Any]:
    """汇总一个 method 与 candidate budget 的正式指标。"""

    real_cost = np.asarray([float(row["selected_real_tnpo_cost"]) for row in rows])
    improvement = np.asarray([float(row["real_improvement_over_m0"]) for row in rows])
    valid = np.asarray([int(row["selected_valid"]) for row in rows], dtype=bool)
    selected_continuous = np.asarray(
        [int(row["selected_continuous"]) for row in rows], dtype=bool
    )
    results = np.asarray([row["effective_result_over_m0"] for row in rows], dtype=str)
    gaps = np.asarray([float(row["selection_gap"]) for row in rows])
    optimism = np.asarray(
        [float(row["selected_action_optimism"]) for row in rows], dtype=np.float64
    )
    one_step_success = np.asarray(
        [int(row["one_step_pose_success"]) for row in rows], dtype=np.float64
    )
    utilisation = np.asarray(
        [float(row["oracle_headroom_utilisation_clipped_to_zero"]) for row in rows],
        dtype=np.float64,
    )
    promoted_count = int(selected_continuous.sum())
    return {
        "targets": len(rows),
        "selected_rollout_validity_rate": float(np.mean(valid)),
        "selected_rollout_validity_rate_95_ci": bootstrap_mean_ci(
            valid.astype(float), strata, resamples
        ),
        "selected_rollout_validity_gate_passed": bool(np.mean(valid) >= VALIDITY_THRESHOLD),
        "selected_real_tnpo_cost": percentile_summary(real_cost),
        "mean_selected_real_tnpo_cost_95_ci": bootstrap_mean_ci(
            real_cost, strata, resamples
        ),
        "mean_real_tnpo_improvement_over_m0": float(np.mean(improvement)),
        "mean_real_tnpo_improvement_95_ci": bootstrap_mean_ci(
            improvement, strata, resamples
        ),
        "median_real_tnpo_improvement_over_m0": float(np.median(improvement)),
        "p90_real_tnpo_improvement_over_m0": float(np.quantile(improvement, 0.90)),
        "effective_win_rate": float(np.mean(results == "win")),
        "effective_win_rate_95_ci": bootstrap_mean_ci(
            (results == "win").astype(float), strata, resamples
        ),
        "effective_tie_rate": float(np.mean(results == "tie")),
        "effective_tie_rate_95_ci": bootstrap_mean_ci(
            (results == "tie").astype(float), strata, resamples
        ),
        "effective_loss_rate": float(np.mean(results == "loss")),
        "effective_loss_rate_95_ci": bootstrap_mean_ci(
            (results == "loss").astype(float), strata, resamples
        ),
        "continuous_selection_rate": float(np.mean(selected_continuous)),
        "continuous_selection_rate_95_ci": bootstrap_mean_ci(
            selected_continuous.astype(float), strata, resamples
        ),
        "continuous_selection_count": promoted_count,
        "continuous_selection_precision": (
            float(np.mean(improvement[selected_continuous] >= PRACTICAL_TOLERANCE))
            if promoted_count
            else math.nan
        ),
        "continuous_selection_precision_95_ci": bootstrap_ratio_ci(
            selected_continuous.astype(float)
            * (improvement >= PRACTICAL_TOLERANCE).astype(float),
            selected_continuous.astype(float),
            strata,
            resamples,
        ),
        "false_promotion_rate": (
            float(np.mean(improvement[selected_continuous] <= -PRACTICAL_TOLERANCE))
            if promoted_count
            else math.nan
        ),
        "false_promotion_rate_95_ci": bootstrap_ratio_ci(
            selected_continuous.astype(float)
            * (improvement <= -PRACTICAL_TOLERANCE).astype(float),
            selected_continuous.astype(float),
            strata,
            resamples,
        ),
        "selected_action_optimism": (
            percentile_summary(optimism[selected_continuous])
            if promoted_count
            else {"mean": math.nan, "median": math.nan, "p90": math.nan}
        ),
        "selection_gap": percentile_summary(gaps),
        "near_optimal_rate_0p05": float(np.mean(gaps <= PRACTICAL_TOLERANCE)),
        "near_optimal_rate_0p02": float(np.mean(gaps <= SENSITIVITY_TOLERANCE)),
        "one_step_pose_success_rate": float(np.mean(one_step_success)),
        "one_step_pose_success_rate_95_ci": bootstrap_mean_ci(
            one_step_success, strata, resamples
        ),
        "selected_position_error_mm": percentile_summary(
            np.asarray([float(row["selected_position_error_mm"]) for row in rows])
        ),
        "selected_yaw_error_deg": percentile_summary(
            np.asarray([float(row["selected_yaw_error_deg"]) for row in rows])
        ),
        "oracle_headroom_targets": int(np.sum(np.isfinite(utilisation))),
        "oracle_headroom_negative_raw_count": int(
            np.sum(
                [
                    math.isfinite(float(row["oracle_headroom_utilisation_raw"]))
                    and float(row["oracle_headroom_utilisation_raw"]) < 0.0
                    for row in rows
                ]
            )
        ),
        "oracle_headroom_utilisation_clipped_to_zero": (
            percentile_summary(utilisation[np.isfinite(utilisation)])
            if np.any(np.isfinite(utilisation))
            else {"mean": math.nan, "median": math.nan, "p90": math.nan}
        ),
    }


def comparison_summary(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    strata: np.ndarray,
    resamples: int,
) -> dict[str, Any]:
    """计算两个 methods 的 paired target-level differences。"""

    left_cost = np.asarray([float(row["selected_real_tnpo_cost"]) for row in left_rows])
    right_cost = np.asarray([float(row["selected_real_tnpo_cost"]) for row in right_rows])
    left_loss = np.asarray(
        [row["effective_result_over_m0"] == "loss" for row in left_rows], dtype=float
    )
    right_loss = np.asarray(
        [row["effective_result_over_m0"] == "loss" for row in right_rows], dtype=float
    )
    cost_difference = left_cost - right_cost
    loss_difference = left_loss - right_loss
    return {
        "left_method": left_rows[0]["method_id"],
        "right_method": right_rows[0]["method_id"],
        "mean_selected_real_cost_difference": float(np.mean(cost_difference)),
        "mean_selected_real_cost_difference_95_ci": bootstrap_mean_ci(
            cost_difference, strata, resamples
        ),
        "effective_loss_rate_difference": float(np.mean(loss_difference)),
        "effective_loss_rate_difference_95_ci": bootstrap_mean_ci(
            loss_difference, strata, resamples
        ),
    }


def summarize_cases(
    role: str, case_rows: list[dict[str, Any]], budgets: tuple[int, ...], resamples: int
) -> dict[str, Any]:
    """汇总各预算、methods、paired comparisons 与 hypothesis evidence。"""

    summary: dict[str, Any] = {"by_candidate_budget": {}}
    for budget in budgets:
        budget_rows = [
            row for row in case_rows if int(row["candidates_per_anchor"]) == budget
        ]
        by_method = {
            method: sorted(
                [row for row in budget_rows if row["method_id"] == method],
                key=lambda row: int(row["e3_target_index"]),
            )
            for method in METHOD_NAMES
        }
        strata = np.asarray(
            [
                int(row["radial_quartile_index"]) * 4
                + int(row["angular_quadrant_index"])
                for row in by_method["M0"]
            ],
            dtype=np.int64,
        )
        win_minus_loss = np.asarray(
            [
                1.0
                if row["effective_result_over_m0"] == "win"
                else -1.0
                if row["effective_result_over_m0"] == "loss"
                else 0.0
                for row in by_method["M2"]
            ]
        )
        method_metrics = {
            method: method_summary(rows, strata, resamples)
            for method, rows in by_method.items()
        }
        union_headroom = np.asarray(
            [float(row["union_oracle_improvement_over_m0"]) for row in by_method["M0"]]
        )
        summary["by_candidate_budget"][str(budget)] = {
            "continuous_candidates_per_target": ANCHOR_BUDGET * budget,
            "methods": method_metrics,
            "union_oracle": {
                "real_tnpo_improvement_over_m0": percentile_summary(union_headroom),
                "effective_headroom_rate_0p05": float(
                    np.mean(union_headroom >= PRACTICAL_TOLERANCE)
                ),
            },
            "m2_effective_win_minus_loss_rate": float(np.mean(win_minus_loss)),
            "m2_effective_win_minus_loss_rate_95_ci": bootstrap_mean_ci(
                win_minus_loss, strata, resamples
            ),
            "m2_minus_m1": comparison_summary(
                by_method["M2"], by_method["M1"], strata, resamples
            ),
            "m2_minus_m3": comparison_summary(
                by_method["M2"], by_method["M3"], strata, resamples
            ),
        }
    return summary


def select_validation_budget(summary: dict[str, Any]) -> dict[str, Any]:
    """按预定义 validity、cost 与 practical-tie 规则选择 Test budget。"""

    eligible: list[int] = []
    for budget in CANDIDATE_BUDGETS:
        metrics = summary["by_candidate_budget"][str(budget)]["methods"]["M2"]
        if (
            metrics["selected_rollout_validity_rate"] >= VALIDITY_THRESHOLD
            and metrics["mean_real_tnpo_improvement_95_ci"][0] > 0.0
        ):
            eligible.append(budget)
    selected_budget: int | None = None
    if eligible:
        best_cost = min(
            summary["by_candidate_budget"][str(budget)]["methods"]["M2"]
            ["selected_real_tnpo_cost"]["mean"]
            for budget in eligible
        )
        practically_tied = [
            budget
            for budget in eligible
            if summary["by_candidate_budget"][str(budget)]["methods"]["M2"]
            ["selected_real_tnpo_cost"]["mean"]
            <= best_cost + SENSITIVITY_TOLERANCE
        ]
        selected_budget = min(practically_tied)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "selection_role": "validation",
        "eligible_candidate_budgets": eligible,
        "selected_candidates_per_anchor": selected_budget,
        "promotion_margin": PROMOTION_MARGIN,
        "anchors_per_target": ANCHOR_BUDGET,
        "validation_stop_before_test": selected_budget is None,
        "selection_rule": (
            "validity>=0.99 and positive improvement CI; choose lowest mean cost, "
            "then the smallest eligible budget within 0.02"
        ),
    }


def evaluate_role(role: str, bootstrap_resamples: int) -> dict[str, Any]:
    """执行一个 role 的 GPU inference、selection 与正式统计。"""

    paths = role_paths(role)
    actions = load_actions()
    p1_outcomes, discrete_truth, discrete_valid = load_nominal_discrete()
    arrays = load_continuous_arrays(role)
    predictions, device_name = predict_continuous_outcomes(
        arrays, p1_outcomes, actions
    )
    if role == "validation":
        budgets = CANDIDATE_BUDGETS
    else:
        selected_path = RESULTS_ROOT / "evaluation" / "selected_configuration.json"
        with selected_path.open("r", encoding="utf-8") as handle:
            selected = json.load(handle)
        budgets = (int(selected["selected_candidates_per_anchor"]),)
    case_rows = build_case_rows(
        role,
        budgets,
        arrays,
        predictions,
        p1_outcomes,
        discrete_truth,
        discrete_valid,
    )
    write_csv(paths["case"], case_rows, CASE_FIELDS)
    metrics = summarize_cases(role, case_rows, budgets, bootstrap_resamples)
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "car_full_name": CAR_FULL_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "friction_cone": FRICTION_CONE,
        "nominal_condition_id": NOMINAL_CONDITION_ID,
        "role": role,
        "device": device_name,
        "targets": TARGETS_PER_ROLE,
        "anchors_per_target": ANCHOR_BUDGET,
        "candidate_budgets": list(budgets),
        "promotion_margin": PROMOTION_MARGIN,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": DEFAULT_SEED,
        **metrics,
        "target_cases_path": str(paths["case"].resolve()),
    }

    if role == "validation":
        selected = select_validation_budget(summary)
        write_json(RESULTS_ROOT / "evaluation" / "selected_configuration.json", selected)
        summary["selected_configuration"] = selected
    else:
        budget = budgets[0]
        budget_summary = summary["by_candidate_budget"][str(budget)]
        m2 = budget_summary["methods"]["M2"]
        h1_supported = m2["mean_real_tnpo_improvement_95_ci"][0] > 0.0
        h2_comparison = budget_summary["m2_minus_m1"]
        h2_supported = (
            h2_comparison["effective_loss_rate_difference_95_ci"][1] < 0.0
            and h2_comparison["mean_selected_real_cost_difference_95_ci"][1]
            <= SENSITIVITY_TOLERANCE
        )
        h3_ci = budget_summary["m2_minus_m3"][
            "mean_selected_real_cost_difference_95_ci"
        ]
        h3_superiority = h3_ci[1] < 0.0
        h3_noninferiority = (not h3_superiority) and h3_ci[1] <= SENSITIVITY_TOLERANCE
        win_loss_ci = budget_summary["m2_effective_win_minus_loss_rate_95_ci"]
        go = (
            h1_supported
            and m2["effective_win_rate"] >= EFFECTIVE_WIN_RATE_THRESHOLD
            and win_loss_ci[0] > 0.0
            and m2["selected_rollout_validity_rate"] >= VALIDITY_THRESHOLD
        )
        summary["hypothesis_decisions"] = {
            "h1_primary_task_level_selection_benefit_supported": h1_supported,
            "h2_protected_promotion_supported": h2_supported,
            "h3_direct_superiority_supported": h3_superiority,
            "h3_direct_practical_noninferiority_supported": h3_noninferiority,
        }
        summary["go_no_go"] = {
            "decision": "Go" if go else "No-Go",
            "positive_mean_improvement_ci": h1_supported,
            "effective_win_rate_at_least_0p10": (
                m2["effective_win_rate"] >= EFFECTIVE_WIN_RATE_THRESHOLD
            ),
            "positive_win_minus_loss_ci": win_loss_ci[0] > 0.0,
            "selected_rollout_validity_at_least_0p99": (
                m2["selected_rollout_validity_rate"] >= VALIDITY_THRESHOLD
            ),
        }
    write_json(paths["summary"], summary)
    return summary


def add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    """为 collection 子命令加入一致的运行参数。"""

    parser.add_argument("--num-workers", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-actions", type=int, default=0)


def parse_args() -> argparse.Namespace:
    """解析 Experiment 3 命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "CAR (Continuous Action Refinement) Experiment 3: "
            "Bounded Continuous Action Selection"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="生成 E3 targets 与 Validation candidates")
    validation_collect = subparsers.add_parser(
        "collect-validation", help="采集 Validation continuous outcomes"
    )
    add_collection_arguments(validation_collect)
    validation_evaluate = subparsers.add_parser(
        "evaluate-validation", help="评价 Validation 并选择 Test candidate budget"
    )
    validation_evaluate.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    subparsers.add_parser(
        "prepare-test", help="按 Validation selected configuration 生成 Test candidates"
    )
    test_collect = subparsers.add_parser(
        "collect-test", help="采集 Independent Test continuous outcomes"
    )
    add_collection_arguments(test_collect)
    test_evaluate = subparsers.add_parser(
        "evaluate-test", help="执行一次 Independent Test 与 Go/No-Go"
    )
    test_evaluate.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    return parser.parse_args()


def main() -> None:
    """执行 CAR Experiment 3 子命令。"""

    args = parse_args()
    print("CAR = Continuous Action Refinement")
    print(f"Experiment 3 = {EXPERIMENT_NAME}")
    if args.command == "prepare":
        payload = prepare_validation()
    elif args.command == "collect-validation":
        payload = collect_role(
            "validation",
            args.num_workers,
            args.batch_size,
            args.resume,
            args.smoke,
            args.max_actions,
        )
    elif args.command == "evaluate-validation":
        payload = evaluate_role("validation", args.bootstrap_resamples)
    elif args.command == "prepare-test":
        payload = prepare_test()
    elif args.command == "collect-test":
        payload = collect_role(
            "test",
            args.num_workers,
            args.batch_size,
            args.resume,
            args.smoke,
            args.max_actions,
        )
    elif args.command == "evaluate-test":
        payload = evaluate_role("test", args.bootstrap_resamples)
    else:
        raise ValueError(f"未知 command: {args.command}")
    print(json.dumps(make_json_compatible(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
