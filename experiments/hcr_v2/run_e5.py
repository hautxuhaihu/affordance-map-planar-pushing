"""运行 HCR V2 E5 fixed-condition belief-space closed-loop pushing。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_e3 as e3_runner
from run_e1 import (
    ACTION_MANIFEST_PATH,
    BASE_XML_PATH,
    FRICTION_CONE,
    build_rollout_input,
    load_actions,
    load_conditions,
    prepare_environment_xmls,
    read_csv_rows,
    set_sliding_friction,
    write_csv,
    write_json,
)
from push_core.hcr_v2.e1 import (
    PRIMARY_TNPO_COST,
    SCENARIO_ACTIVE_COORDINATES,
)
from push_core.hcr_v2.e2 import local_motion_observation
from push_core.hcr_v2.e3 import interpolate_outcome_grid
from push_core.hcr_v2.e5 import (
    BELIEF_MARGINALISED_CONTROLLER_ID,
    CERTAINTY_EQUIVALENT_CONTROLLER_ID,
    CONTROLLER_IDS,
    CONTROLLER_NAMES,
    FULL_INFORMATION_CONTROLLER_ID,
    NOMINAL_CONTROLLER_ID,
    PROTOCOL_VERSION,
    BeliefState,
    ClosedLoopDecisionEngine,
    make_task_query,
    wrap_to_pi,
)
from push_core.project_paths import HCR_V2_DATA_DIR, HCR_V2_RESULTS_DIR
from push_core.simulation.physical_pusher_rollout import (
    get_body_id,
    get_object_yaw_qpos,
    load_model,
    reset_state,
    run_physical_pusher_atomic_push,
)


SCENARIOS = ("friction", "com", "joint")
MAXIMUM_PUSHES = 20
BOOTSTRAP_RESAMPLES = 10_000
MANIFEST_DIR = REPOSITORY_ROOT / "manifests" / "hcr_v2"
E2_CORE_TARGET_PATHS = {
    "validation": MANIFEST_DIR / "hcr_v2_e2_core_validation_target_manifest_v1.csv",
    "test": MANIFEST_DIR / "hcr_v2_e2_core_test_target_manifest_v1.csv",
}
E5_CORE_TARGET_PATHS = {
    "validation": MANIFEST_DIR / "hcr_v2_e5_core_validation_target_manifest_v1.csv",
    "test": MANIFEST_DIR / "hcr_v2_e5_core_test_target_manifest_v1.csv",
}
E5_SEQUENTIAL_TARGET_PATHS = {
    "validation": (
        MANIFEST_DIR
        / "hcr_v2_e5_sequential_extension_validation_target_manifest_v1.csv"
    ),
    "test": (
        MANIFEST_DIR
        / "hcr_v2_e5_sequential_extension_test_target_manifest_v1.csv"
    ),
}
E5_DATA_ROOT = HCR_V2_DATA_DIR / "e5"
E5_RESULTS_ROOT = HCR_V2_RESULTS_DIR / "e5"

TARGET_FIELDS = [
    "v2_target_id",
    "split_role",
    "target_group",
    "target_regime",
    "target_stratum",
    "target_delta_x_m",
    "target_delta_y_m",
    "target_yaw_offset_rad",
    "canonical_position_key",
    "radial_distance_m",
    "polar_angle_deg",
    "selection_rank_within_stratum",
    "source_target_id",
    "source_manifest",
    "selection_rule_version",
]

CANDIDATE_FIELDS = [
    "split_role",
    "candidate_id",
    "target_delta_x_m",
    "target_delta_y_m",
    "canonical_position_key",
    "radial_distance_m",
    "polar_angle_deg",
    "excluded_core_overlap",
    "excluded_cross_grid_overlap",
    "one_step_success_any",
    "feasible_all",
    "eligible",
    "worst_pushes_to_success",
    "maximum_yaw_deviation_rad",
    "friction_one_step_success_any",
    "com_one_step_success_any",
    "joint_one_step_success_any",
    "friction_feasible_all",
    "com_feasible_all",
    "joint_feasible_all",
]

STEP_FIELDS = [
    "experiment_id",
    "protocol_version",
    "friction_cone",
    "environment_xml",
    "role",
    "scenario",
    "condition_id",
    "condition_index_within_role",
    "hidden_parameter_dimension",
    "friction_sliding_mu",
    "com_offset_x_m",
    "com_offset_y_m",
    "hidden_u_friction",
    "hidden_u_com_x",
    "hidden_u_com_y",
    "target_id",
    "target_group",
    "target_stratum",
    "target_delta_x_m",
    "target_delta_y_m",
    "target_radial_distance_m",
    "controller_name",
    "controller_id",
    "episode_key",
    "push_index",
    "attempted_push_count",
    "maximum_push_budget",
    "valid_update_count_pre",
    "valid_update_count_post",
    "belief_updated",
    "belief_mean_hidden_u_friction_pre",
    "belief_mean_hidden_u_com_x_pre",
    "belief_mean_hidden_u_com_y_pre",
    "belief_covariance_trace_pre",
    "belief_mean_hidden_u_friction_post",
    "belief_mean_hidden_u_com_x_post",
    "belief_mean_hidden_u_com_y_post",
    "belief_covariance_trace_post",
    "task_local_x_m",
    "task_local_y_m",
    "task_yaw_sin",
    "task_yaw_cos",
    "target_world_x_m",
    "target_world_y_m",
    "target_world_yaw_rad",
    "pre_push_x_m",
    "pre_push_y_m",
    "pre_push_yaw_rad",
    "post_push_x_m",
    "post_push_y_m",
    "post_push_yaw_rad",
    "terminal_metric_pose_source",
    "observation_local_delta_x_m",
    "observation_local_delta_y_m",
    "observation_delta_yaw_rad",
    "v2_action_id",
    "candidate_id",
    "action_param_index",
    "candidate_source",
    "candidate_count",
    "proposal_probability",
    "predicted_selected_tnpo_cost",
    "actual_position_error_m",
    "actual_yaw_error_rad",
    "actual_tnpo_cost",
    "maximum_yaw_deviation_rad",
    "success_after_push",
    "valid_observation",
    "quality_pass",
    "simulation_unstable",
    "contact_success",
    "stopped_by_threshold",
    "num_contacts",
    "settle_time_s",
    "proposal_latency_s",
    "selection_latency_s",
    "belief_update_latency_s",
    "simulation_latency_s",
    "episode_success",
    "episode_invalid",
    "terminal_reason",
    "terminal_push_count",
    "is_terminal_push",
]

PROCESS_WORKER_CONTEXT: dict[str, Any] = {}


def select_scenarios(value: str) -> tuple[str, ...]:
    """解析单场景或全部场景。"""

    if value == "all":
        return SCENARIOS
    if value not in SCENARIOS:
        raise ValueError(f"未知 scenario: {value}")
    return (value,)


def parse_controller_ids(value: str) -> tuple[str, ...]:
    """解析完整语义 controller identifiers。"""

    if value == "all":
        return CONTROLLER_IDS
    identifiers = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [item for item in identifiers if item not in CONTROLLER_NAMES]
    if unknown:
        raise ValueError(f"未知 controller identifiers: {unknown}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("controller identifiers 不能重复")
    return identifiers


def require_cuda() -> torch.device:
    """E5 正式 inference 要求使用当前 CUDA GPU。"""

    if not torch.cuda.is_available():
        raise RuntimeError("E5 GPU-first implementation 需要可用的 CUDA device")
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def target_key(delta_x_m: float, delta_y_m: float) -> str:
    """构造四位米制 canonical target key。"""

    return f"{delta_x_m:.4f}|{delta_y_m:.4f}"


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    """从稳定排序后的序列选择等间隔 ranks。"""

    if length < count:
        raise RuntimeError(f"候选数量 {length} 小于需要的 {count}")
    return [int(value) for value in np.rint(np.linspace(0, length - 1, count))]


def generate_polar_candidates(role: str) -> list[dict[str, Any]]:
    """生成 Validation/Test 交错的确定性 polar candidate grid。"""

    if role == "validation":
        directions = np.arange(0.0, 360.0, 10.0)
        radii = np.arange(0.02, 0.5001, 0.02)
        prefix = "E5SEV_CAND"
    else:
        directions = np.arange(5.0, 360.0, 10.0)
        radii = np.arange(0.03, 0.4901, 0.02)
        prefix = "E5SET_CAND"
    rows: list[dict[str, Any]] = []
    for radius in radii:
        for angle in directions:
            radians = math.radians(float(angle))
            delta_x = float(radius * math.cos(radians))
            delta_y = float(radius * math.sin(radians))
            rows.append(
                {
                    "split_role": role,
                    "candidate_id": f"{prefix}_{len(rows):04d}",
                    "target_delta_x_m": delta_x,
                    "target_delta_y_m": delta_y,
                    "canonical_position_key": target_key(delta_x, delta_y),
                    "radial_distance_m": float(radius),
                    "polar_angle_deg": float(angle),
                }
            )
    keys = [row["canonical_position_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{role} polar grid 在四位 target key 下不唯一")
    return rows


def select_core_targets(role: str) -> list[dict[str, Any]]:
    """从 E2 core subset 中确定性选择 4×16 个 E5 Core targets。"""

    source_path = E2_CORE_TARGET_PATHS[role]
    source_rows = read_csv_rows(source_path)
    selected: list[dict[str, Any]] = []
    for stratum in ("Q1", "Q2", "Q3", "Q4"):
        eligible = [
            row
            for row in source_rows
            if row["target_stratum"] == stratum
            and math.hypot(
                float(row["target_delta_x_m"]),
                float(row["target_delta_y_m"]),
            )
            > PRIMARY_TNPO_COST.position_tolerance_m
        ]
        eligible.sort(
            key=lambda row: (
                float(row["radial_distance_m"]),
                row["canonical_position_key"],
                row["v2_target_id"],
            )
        )
        for selection_rank, index in enumerate(
            evenly_spaced_indices(len(eligible), 16)
        ):
            source = eligible[index]
            selected.append(
                {
                    "v2_target_id": source["v2_target_id"],
                    "split_role": role,
                    "target_group": "core",
                    "target_regime": "core",
                    "target_stratum": stratum,
                    "target_delta_x_m": source["target_delta_x_m"],
                    "target_delta_y_m": source["target_delta_y_m"],
                    "target_yaw_offset_rad": "0.00000000",
                    "canonical_position_key": source["canonical_position_key"],
                    "radial_distance_m": source["radial_distance_m"],
                    "polar_angle_deg": "",
                    "selection_rank_within_stratum": selection_rank,
                    "source_target_id": source["v2_target_id"],
                    "source_manifest": str(source_path.resolve()),
                    "selection_rule_version": "hcr_v2_e5_core_64_v1",
                }
            )
    return selected


@torch.inference_mode()
def evaluate_scenario_feasibility(
    outcomes: torch.Tensor,
    candidate_positions: torch.Tensor,
    maximum_pushes: int,
    candidate_batch_size: int,
) -> dict[str, np.ndarray]:
    """在一个场景的全部 Validation conditions 上执行 greedy feasibility。"""

    condition_count, action_count, _ = outcomes.shape
    candidate_count = candidate_positions.shape[0]
    one_step_any = torch.zeros(
        candidate_count, dtype=torch.bool, device=outcomes.device
    )
    feasible_all = torch.zeros_like(one_step_any)
    worst_pushes = torch.full(
        (candidate_count,), -1, dtype=torch.long, device=outcomes.device
    )
    maximum_yaw = torch.zeros(
        candidate_count, dtype=torch.float32, device=outcomes.device
    )
    condition_indices = torch.arange(
        condition_count, dtype=torch.long, device=outcomes.device
    )[:, None]

    for start in range(0, candidate_count, candidate_batch_size):
        stop = min(candidate_count, start + candidate_batch_size)
        targets = candidate_positions[start:stop]
        batch_count = targets.shape[0]

        initial_position_error = torch.linalg.vector_norm(
            outcomes[:, None, :, :2] - targets[None, :, None, :], dim=-1
        )
        initial_yaw_error = torch.abs(
            torch.remainder(
                outcomes[:, None, :, 2] + math.pi,
                2.0 * math.pi,
            )
            - math.pi
        )
        one_step = (
            (initial_position_error <= PRIMARY_TNPO_COST.position_tolerance_m)
            & (initial_yaw_error <= PRIMARY_TNPO_COST.yaw_tolerance_rad)
        ).any(dim=2)
        one_step_any[start:stop] = one_step.any(dim=0)

        current_x = torch.zeros(
            (condition_count, batch_count),
            dtype=torch.float32,
            device=outcomes.device,
        )
        current_y = torch.zeros_like(current_x)
        current_yaw = torch.zeros_like(current_x)
        success = torch.zeros_like(current_x, dtype=torch.bool)
        first_success = torch.zeros_like(current_x, dtype=torch.long)
        yaw_excursion = torch.zeros_like(current_x)

        for push_index in range(1, maximum_pushes + 1):
            cosine = torch.cos(current_yaw)[:, :, None]
            sine = torch.sin(current_yaw)[:, :, None]
            local_x = outcomes[:, None, :, 0]
            local_y = outcomes[:, None, :, 1]
            predicted_x = (
                current_x[:, :, None] + cosine * local_x - sine * local_y
            )
            predicted_y = (
                current_y[:, :, None] + sine * local_x + cosine * local_y
            )
            predicted_yaw = current_yaw[:, :, None] + outcomes[:, None, :, 2]
            position_error = torch.hypot(
                predicted_x - targets[None, :, None, 0],
                predicted_y - targets[None, :, None, 1],
            )
            yaw_error = torch.abs(
                torch.remainder(predicted_yaw + math.pi, 2.0 * math.pi)
                - math.pi
            )
            costs = (
                0.5
                * position_error
                / PRIMARY_TNPO_COST.position_tolerance_m
                + 0.5 * yaw_error / PRIMARY_TNPO_COST.yaw_tolerance_rad
            )
            selected_indices = torch.argmin(costs, dim=2)
            selected_outcomes = outcomes[
                condition_indices.expand(-1, batch_count), selected_indices
            ]
            next_x = (
                current_x
                + torch.cos(current_yaw) * selected_outcomes[:, :, 0]
                - torch.sin(current_yaw) * selected_outcomes[:, :, 1]
            )
            next_y = (
                current_y
                + torch.sin(current_yaw) * selected_outcomes[:, :, 0]
                + torch.cos(current_yaw) * selected_outcomes[:, :, 1]
            )
            next_yaw = current_yaw + selected_outcomes[:, :, 2]
            active = ~success
            current_x = torch.where(active, next_x, current_x)
            current_y = torch.where(active, next_y, current_y)
            current_yaw = torch.where(active, next_yaw, current_yaw)
            yaw_excursion = torch.maximum(
                yaw_excursion,
                torch.abs(
                    torch.remainder(current_yaw + math.pi, 2.0 * math.pi)
                    - math.pi
                ),
            )
            newly_successful = active & (
                torch.hypot(
                    current_x - targets[None, :, 0],
                    current_y - targets[None, :, 1],
                )
                <= PRIMARY_TNPO_COST.position_tolerance_m
            ) & (
                torch.abs(
                    torch.remainder(current_yaw + math.pi, 2.0 * math.pi)
                    - math.pi
                )
                <= PRIMARY_TNPO_COST.yaw_tolerance_rad
            )
            first_success = torch.where(
                newly_successful,
                torch.full_like(first_success, push_index),
                first_success,
            )
            success |= newly_successful
            if bool(success.all()):
                break

        feasible_all[start:stop] = success.all(dim=0)
        batch_worst = first_success.max(dim=0).values
        batch_worst = torch.where(
            success.all(dim=0), batch_worst, torch.full_like(batch_worst, -1)
        )
        worst_pushes[start:stop] = batch_worst
        maximum_yaw[start:stop] = yaw_excursion.max(dim=0).values

    return {
        "one_step_success_any": one_step_any.cpu().numpy(),
        "feasible_all": feasible_all.cpu().numpy(),
        "worst_pushes_to_success": worst_pushes.cpu().numpy(),
        "maximum_yaw_deviation_rad": maximum_yaw.cpu().numpy(),
    }


def evaluate_candidate_grid(
    candidates: list[dict[str, Any]],
    core_keys: set[str],
    cross_grid_keys: set[str],
    action_ids: list[str],
    device: torch.device,
    maximum_pushes: int,
    candidate_batch_size: int,
) -> list[dict[str, Any]]:
    """使用全部 Validation conditions 评价一个 polar candidate grid。"""

    positions = torch.tensor(
        [
            [row["target_delta_x_m"], row["target_delta_y_m"]]
            for row in candidates
        ],
        dtype=torch.float32,
        device=device,
    )
    scenario_results: dict[str, dict[str, np.ndarray]] = {}
    for scenario in SCENARIOS:
        conditions = e3_runner.load_conditions(scenario, "validation")
        outcomes = torch.from_numpy(
            e3_runner.load_outcome_array(
                scenario, "validation", conditions, action_ids
            )
        ).to(device)
        scenario_results[scenario] = evaluate_scenario_feasibility(
            outcomes,
            positions,
            maximum_pushes,
            candidate_batch_size,
        )
        print(f"evaluated {candidates[0]['split_role']} {scenario} feasibility")

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        one_step_any = any(
            bool(scenario_results[scenario]["one_step_success_any"][index])
            for scenario in SCENARIOS
        )
        feasible_all = all(
            bool(scenario_results[scenario]["feasible_all"][index])
            for scenario in SCENARIOS
        )
        core_overlap = candidate["canonical_position_key"] in core_keys
        cross_overlap = candidate["canonical_position_key"] in cross_grid_keys
        row = dict(candidate)
        aggregate_worst_pushes = (
            max(
                int(
                    scenario_results[scenario]["worst_pushes_to_success"][
                        index
                    ]
                )
                for scenario in SCENARIOS
            )
            if feasible_all
            else -1
        )
        row.update(
            {
                "excluded_core_overlap": int(core_overlap),
                "excluded_cross_grid_overlap": int(cross_overlap),
                "one_step_success_any": int(one_step_any),
                "feasible_all": int(feasible_all),
                "eligible": int(
                    not core_overlap
                    and not cross_overlap
                    and not one_step_any
                    and feasible_all
                ),
                "worst_pushes_to_success": aggregate_worst_pushes,
                "maximum_yaw_deviation_rad": max(
                    float(
                        scenario_results[scenario][
                            "maximum_yaw_deviation_rad"
                        ][index]
                    )
                    for scenario in SCENARIOS
                ),
            }
        )
        for scenario in SCENARIOS:
            row[f"{scenario}_one_step_success_any"] = int(
                scenario_results[scenario]["one_step_success_any"][index]
            )
            row[f"{scenario}_feasible_all"] = int(
                scenario_results[scenario]["feasible_all"][index]
            )
        rows.append(row)
    return rows


def radial_stratum(radius: float, boundaries: np.ndarray) -> str:
    """使用 Validation eligible-distance quartiles 分配 SE-Q1–SE-Q4。"""

    index = int(np.searchsorted(boundaries, radius, side="right"))
    return f"SE-Q{index + 1}"


def select_sequential_targets(
    rows: list[dict[str, Any]],
    role: str,
    boundaries: np.ndarray,
) -> list[dict[str, Any]]:
    """按固定 radial strata 和 angular ranks 选择 4×16 个 targets。"""

    eligible = [row for row in rows if int(row["eligible"]) == 1]
    selected: list[dict[str, Any]] = []
    for stratum in ("SE-Q1", "SE-Q2", "SE-Q3", "SE-Q4"):
        pool = [
            row
            for row in eligible
            if radial_stratum(float(row["radial_distance_m"]), boundaries)
            == stratum
        ]
        pool.sort(
            key=lambda row: (
                float(row["polar_angle_deg"]),
                float(row["radial_distance_m"]),
                row["canonical_position_key"],
            )
        )
        for selection_rank, index in enumerate(
            evenly_spaced_indices(len(pool), 16)
        ):
            source = pool[index]
            target_index = len(selected)
            prefix = "E5SEV" if role == "validation" else "E5SET"
            selected.append(
                {
                    "v2_target_id": f"{prefix}_{target_index:03d}",
                    "split_role": role,
                    "target_group": "sequential_extension",
                    "target_regime": "sequential_extension",
                    "target_stratum": stratum,
                    "target_delta_x_m": f"{float(source['target_delta_x_m']):.8f}",
                    "target_delta_y_m": f"{float(source['target_delta_y_m']):.8f}",
                    "target_yaw_offset_rad": "0.00000000",
                    "canonical_position_key": source["canonical_position_key"],
                    "radial_distance_m": f"{float(source['radial_distance_m']):.8f}",
                    "polar_angle_deg": f"{float(source['polar_angle_deg']):.4f}",
                    "selection_rank_within_stratum": selection_rank,
                    "source_target_id": source["candidate_id"],
                    "source_manifest": str(
                        (
                            E5_DATA_ROOT
                            / "target_candidates"
                            / role
                            / "candidate_analysis.csv"
                        ).resolve()
                    ),
                    "selection_rule_version": (
                        "hcr_v2_e5_validation_defined_sequential_64_v1"
                    ),
                }
            )
    return selected


def prepare_targets(args: argparse.Namespace) -> dict[str, Any]:
    """生成 E5 Core 与 Sequential-Extension target manifests。"""

    device = require_cuda()
    action_ids, _ = e3_runner.load_action_layout()
    core_targets = {
        role: select_core_targets(role) for role in ("validation", "test")
    }
    all_core_keys = {
        row["canonical_position_key"]
        for role_rows in core_targets.values()
        for row in role_rows
    }
    grids = {
        role: generate_polar_candidates(role)
        for role in ("validation", "test")
    }
    validation_keys = {
        row["canonical_position_key"] for row in grids["validation"]
    }
    test_keys = {row["canonical_position_key"] for row in grids["test"]}
    analyses = {
        "validation": evaluate_candidate_grid(
            grids["validation"],
            all_core_keys,
            test_keys,
            action_ids,
            device,
            MAXIMUM_PUSHES,
            args.candidate_batch_size,
        ),
        "test": evaluate_candidate_grid(
            grids["test"],
            all_core_keys,
            validation_keys,
            action_ids,
            device,
            MAXIMUM_PUSHES,
            args.candidate_batch_size,
        ),
    }
    for role, rows in analyses.items():
        write_csv(
            E5_DATA_ROOT
            / "target_candidates"
            / role
            / "candidate_analysis.csv",
            rows,
            CANDIDATE_FIELDS,
        )

    validation_distances = np.asarray(
        [
            float(row["radial_distance_m"])
            for row in analyses["validation"]
            if int(row["eligible"]) == 1
        ],
        dtype=np.float64,
    )
    if len(validation_distances) < 64:
        raise RuntimeError(
            "Validation eligible Sequential-Extension candidates 少于 64"
        )
    boundaries = np.quantile(validation_distances, [0.25, 0.50, 0.75])
    sequential_targets = {
        role: select_sequential_targets(analyses[role], role, boundaries)
        for role in ("validation", "test")
    }
    for role in ("validation", "test"):
        write_csv(E5_CORE_TARGET_PATHS[role], core_targets[role], TARGET_FIELDS)
        write_csv(
            E5_SEQUENTIAL_TARGET_PATHS[role],
            sequential_targets[role],
            TARGET_FIELDS,
        )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "maximum_attempted_pushes": MAXIMUM_PUSHES,
        "validation_radial_boundaries_m": boundaries.tolist(),
        "candidate_counts": {
            role: len(analyses[role]) for role in analyses
        },
        "eligible_counts": {
            role: sum(int(row["eligible"]) for row in analyses[role])
            for role in analyses
        },
        "core_target_counts": {
            role: len(core_targets[role]) for role in core_targets
        },
        "sequential_extension_target_counts": {
            role: len(sequential_targets[role])
            for role in sequential_targets
        },
        "core_manifests": {
            role: str(E5_CORE_TARGET_PATHS[role].resolve())
            for role in E5_CORE_TARGET_PATHS
        },
        "sequential_extension_manifests": {
            role: str(E5_SEQUENTIAL_TARGET_PATHS[role].resolve())
            for role in E5_SEQUENTIAL_TARGET_PATHS
        },
    }
    write_json(E5_RESULTS_ROOT / "target_preparation" / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def load_e5_targets(role: str) -> list[dict[str, str]]:
    """读取一个 split 的 64 Core 与 64 Sequential-Extension targets。"""

    core = read_csv_rows(E5_CORE_TARGET_PATHS[role])
    sequential = read_csv_rows(E5_SEQUENTIAL_TARGET_PATHS[role])
    if len(core) != 64 or len(sequential) != 64:
        raise RuntimeError(
            f"E5 {role} target manifests 必须分别包含 64 个 targets"
        )
    rows = core + sequential
    ids = [row["v2_target_id"] for row in rows]
    keys = [row["canonical_position_key"] for row in rows]
    if len(set(ids)) != 128 or len(set(keys)) != 128:
        raise RuntimeError(f"E5 {role} target IDs 或 keys 不唯一")
    return rows


def load_decision_engine(
    scenario: str,
    action_ids: list[str],
    device: torch.device,
    node_query_chunk_size: int,
) -> ClosedLoopDecisionEngine:
    """加载一个场景的 E1–E4 artifacts 并常驻 GPU。"""

    nodes_array, prior_array = e3_runner.make_fixed_quadrature(scenario)
    nodes = torch.from_numpy(nodes_array.astype(np.float32)).to(device)
    prior_weights = torch.from_numpy(prior_array.astype(np.float32)).to(device)
    outcome_grid = e3_runner.load_p1_grid(scenario, action_ids, device)
    node_outcomes = interpolate_outcome_grid(outcome_grid, nodes)
    proposal_model, normaliser = e3_runner.load_trained_proposal(
        scenario, device
    )
    residual_bias, precision = e3_runner.load_residual_parameters(
        scenario, device
    )
    return ClosedLoopDecisionEngine(
        scenario=scenario,
        action_ids=action_ids,
        outcome_grid=outcome_grid,
        nodes=nodes,
        prior_weights=prior_weights,
        node_outcomes=node_outcomes,
        proposal_model=proposal_model,
        task_normaliser=normaliser,
        residual_bias_standardised=residual_bias,
        precision=precision,
        device=device,
        node_query_chunk_size=node_query_chunk_size,
    )


def read_pose(model, data, object_body_id: int) -> tuple[float, float, float]:
    """读取物体的平面位置与 unwrapped yaw。"""

    return (
        float(data.xpos[object_body_id][0]),
        float(data.xpos[object_body_id][1]),
        float(get_object_yaw_qpos(model, data)),
    )


def tnpo_cost(position_error_m: float, yaw_error_rad: float) -> float:
    """计算 actual terminal pose 的 Target-Normalised Pose-Objective Cost。"""

    return (
        0.5 * position_error_m / PRIMARY_TNPO_COST.position_tolerance_m
        + 0.5 * yaw_error_rad / PRIMARY_TNPO_COST.yaw_tolerance_rad
    )


def active_hidden_parameters(
    scenario: str, condition: dict[str, str]
) -> np.ndarray:
    """提取当前场景的 active normalised hidden coordinates。"""

    return np.asarray(
        [float(condition[field]) for field in SCENARIO_ACTIVE_COORDINATES[scenario]],
        dtype=np.float32,
    )


def belief_log_fields(
    scenario: str, summary: dict[str, object], suffix: str
) -> dict[str, Any]:
    """把 active posterior mean 写入统一的三维日志字段。"""

    values = {
        "hidden_u_friction": 0.0,
        "hidden_u_com_x": 0.0,
        "hidden_u_com_y": 0.0,
    }
    mean = np.asarray(summary["mean_normalised"], dtype=np.float64)
    for field, value in zip(SCENARIO_ACTIVE_COORDINATES[scenario], mean):
        values[field] = float(value)
    return {
        f"belief_mean_hidden_u_friction_{suffix}": values[
            "hidden_u_friction"
        ],
        f"belief_mean_hidden_u_com_x_{suffix}": values["hidden_u_com_x"],
        f"belief_mean_hidden_u_com_y_{suffix}": values["hidden_u_com_y"],
        f"belief_covariance_trace_{suffix}": summary[
            "covariance_trace_normalised"
        ],
    }


def run_closed_loop_episode(
    model,
    data,
    object_body_id: int,
    engine: ClosedLoopDecisionEngine,
    controller_id: str,
    condition: dict[str, str],
    target: dict[str, str],
    action_rows_by_id: dict[str, dict[str, str]],
    episode_id: int,
    environment_xml: Path,
    role: str,
    maximum_pushes: int,
) -> list[dict[str, Any]]:
    """执行一个 condition-target-controller closed-loop episode。"""

    reset_action = action_rows_by_id[engine.action_ids[0]]
    reset_input = build_rollout_input(reset_action, condition, episode_id)
    reset_state(model, data, reset_input)
    initial_x, initial_y, initial_yaw = read_pose(model, data, object_body_id)
    target_world_x = initial_x + float(target["target_delta_x_m"])
    target_world_y = initial_y + float(target["target_delta_y_m"])
    target_world_yaw = initial_yaw
    initial_error = math.hypot(
        initial_x - target_world_x, initial_y - target_world_y
    )
    if initial_error <= PRIMARY_TNPO_COST.position_tolerance_m:
        raise RuntimeError(
            f"E5 正式 target 在初始化时已经成功: {target['v2_target_id']}"
        )

    belief = engine.new_belief()
    true_hidden = active_hidden_parameters(engine.scenario, condition)
    episode_key = (
        f"{engine.scenario}|{condition['condition_id']}|"
        f"{target['v2_target_id']}|{controller_id}"
    )
    rows: list[dict[str, Any]] = []
    maximum_yaw_deviation = 0.0
    terminal_reason = "maximum_push_budget"

    for push_index in range(1, maximum_pushes + 1):
        pre_x, pre_y, pre_yaw = read_pose(model, data, object_body_id)
        query = make_task_query(
            np.asarray([pre_x, pre_y]),
            pre_yaw,
            np.asarray([target_world_x, target_world_y]),
            initial_yaw,
        )
        pre_belief = engine.belief_summary(belief)
        decision = engine.select_action(
            controller_id, query, belief, true_hidden
        )
        action_id = engine.action_ids[decision.action_index]
        action = action_rows_by_id[action_id]
        rollout_input = build_rollout_input(action, condition, episode_id)
        rollout_input["dataset_role"] = (
            f"hcr_v2_e5_{role}_{engine.scenario}_{controller_id}"
        )
        rollout_input["step_id"] = push_index
        rollout_input["group_id"] = episode_id

        simulation_start = time.perf_counter()
        result = run_physical_pusher_atomic_push(model, data, rollout_input)
        simulation_latency = time.perf_counter() - simulation_start
        raw_post_x, raw_post_y, raw_post_yaw = read_pose(
            model, data, object_body_id
        )
        raw_pose_finite = all(
            math.isfinite(value)
            for value in (raw_post_x, raw_post_y, raw_post_yaw)
        )
        valid_observation = (
            int(result["quality_pass"]) == 1
            and int(result["simulation_unstable"]) == 0
            and int(result["contact_success"]) == 1
            and int(result["stopped_by_threshold"]) == 1
            and raw_pose_finite
        )
        reliable_post_pose = (
            int(result["simulation_unstable"]) == 0 and raw_pose_finite
        )
        if reliable_post_pose:
            metric_x, metric_y, metric_yaw = (
                raw_post_x,
                raw_post_y,
                raw_post_yaw,
            )
            metric_pose_source = "post_push_settled_pose"
        else:
            metric_x, metric_y, metric_yaw = pre_x, pre_y, pre_yaw
            metric_pose_source = "last_valid_pre_push_pose"

        observation = local_motion_observation(
            np.asarray([pre_x, pre_y]),
            pre_yaw,
            np.asarray([metric_x, metric_y]),
            metric_yaw,
        )
        belief_updated = False
        belief_update_latency = 0.0
        belief_conditioned_controller = controller_id in {
            CERTAINTY_EQUIVALENT_CONTROLLER_ID,
            BELIEF_MARGINALISED_CONTROLLER_ID,
        }
        if valid_observation and belief_conditioned_controller:
            if engine.device.type == "cuda":
                torch.cuda.current_stream(engine.device).synchronize()
            update_start = time.perf_counter()
            belief_updated = engine.update_belief(
                belief, decision.action_index, observation
            )
            if engine.device.type == "cuda":
                torch.cuda.current_stream(engine.device).synchronize()
            belief_update_latency = time.perf_counter() - update_start
        post_belief = engine.belief_summary(belief)

        position_error = math.hypot(
            metric_x - target_world_x, metric_y - target_world_y
        )
        yaw_error = abs(wrap_to_pi(metric_yaw - target_world_yaw))
        maximum_yaw_deviation = max(
            maximum_yaw_deviation,
            abs(wrap_to_pi(metric_yaw - initial_yaw)),
        )
        success = (
            valid_observation
            and position_error <= PRIMARY_TNPO_COST.position_tolerance_m
            and yaw_error <= PRIMARY_TNPO_COST.yaw_tolerance_rad
        )
        if not valid_observation:
            terminal_reason = "invalid_push"
        elif success:
            terminal_reason = "success"
        elif push_index == maximum_pushes:
            terminal_reason = "maximum_push_budget"

        row: dict[str, Any] = {
            "experiment_id": "E5",
            "protocol_version": PROTOCOL_VERSION,
            "friction_cone": FRICTION_CONE,
            "environment_xml": str(environment_xml.resolve()),
            "role": role,
            "scenario": engine.scenario,
            "condition_id": condition["condition_id"],
            "condition_index_within_role": condition[
                "condition_index_within_role"
            ],
            "hidden_parameter_dimension": condition[
                "hidden_parameter_dimension"
            ],
            "friction_sliding_mu": condition["friction_sliding_mu"],
            "com_offset_x_m": condition["com_offset_x_m"],
            "com_offset_y_m": condition["com_offset_y_m"],
            "hidden_u_friction": condition["hidden_u_friction"],
            "hidden_u_com_x": condition["hidden_u_com_x"],
            "hidden_u_com_y": condition["hidden_u_com_y"],
            "target_id": target["v2_target_id"],
            "target_group": target["target_group"],
            "target_stratum": target["target_stratum"],
            "target_delta_x_m": target["target_delta_x_m"],
            "target_delta_y_m": target["target_delta_y_m"],
            "target_radial_distance_m": target["radial_distance_m"],
            "controller_name": CONTROLLER_NAMES[controller_id],
            "controller_id": controller_id,
            "episode_key": episode_key,
            "push_index": push_index,
            "attempted_push_count": push_index,
            "maximum_push_budget": maximum_pushes,
            "valid_update_count_pre": pre_belief["update_count"],
            "valid_update_count_post": post_belief["update_count"],
            "belief_updated": int(belief_updated),
            "task_local_x_m": query[0],
            "task_local_y_m": query[1],
            "task_yaw_sin": query[2],
            "task_yaw_cos": query[3],
            "target_world_x_m": target_world_x,
            "target_world_y_m": target_world_y,
            "target_world_yaw_rad": target_world_yaw,
            "pre_push_x_m": pre_x,
            "pre_push_y_m": pre_y,
            "pre_push_yaw_rad": pre_yaw,
            "post_push_x_m": raw_post_x,
            "post_push_y_m": raw_post_y,
            "post_push_yaw_rad": raw_post_yaw,
            "terminal_metric_pose_source": metric_pose_source,
            "observation_local_delta_x_m": observation[0],
            "observation_local_delta_y_m": observation[1],
            "observation_delta_yaw_rad": observation[2],
            "v2_action_id": action_id,
            "candidate_id": action["candidate_id"],
            "action_param_index": action["action_param_index"],
            "candidate_source": decision.candidate_source,
            "candidate_count": decision.candidate_count,
            "proposal_probability": (
                ""
                if decision.proposal_probability is None
                else decision.proposal_probability
            ),
            "predicted_selected_tnpo_cost": decision.decision_score,
            "actual_position_error_m": position_error,
            "actual_yaw_error_rad": yaw_error,
            "actual_tnpo_cost": tnpo_cost(position_error, yaw_error),
            "maximum_yaw_deviation_rad": maximum_yaw_deviation,
            "success_after_push": int(success),
            "valid_observation": int(valid_observation),
            "quality_pass": result["quality_pass"],
            "simulation_unstable": result["simulation_unstable"],
            "contact_success": result["contact_success"],
            "stopped_by_threshold": result["stopped_by_threshold"],
            "num_contacts": result["num_contacts"],
            "settle_time_s": result["settle_time_s"],
            "proposal_latency_s": decision.proposal_latency_s,
            "selection_latency_s": decision.selection_latency_s,
            "belief_update_latency_s": belief_update_latency,
            "simulation_latency_s": simulation_latency,
        }
        row.update(belief_log_fields(engine.scenario, pre_belief, "pre"))
        row.update(belief_log_fields(engine.scenario, post_belief, "post"))
        rows.append(row)
        if terminal_reason in {"invalid_push", "success"}:
            break

    episode_success = terminal_reason == "success"
    episode_invalid = terminal_reason == "invalid_push"
    for row_index, row in enumerate(rows):
        row["episode_success"] = int(episode_success)
        row["episode_invalid"] = int(episode_invalid)
        row["terminal_reason"] = terminal_reason
        row["terminal_push_count"] = len(rows)
        row["is_terminal_push"] = int(row_index == len(rows) - 1)
    return rows


def inspect_complete_condition_shard(
    path: Path,
    target_ids: set[str],
    controller_ids: tuple[str, ...],
    maximum_pushes: int,
    role: str,
) -> bool:
    """检查一个 condition shard 是否包含完整 terminal episodes。"""

    if not path.exists():
        return False
    rows = read_csv_rows(path)
    terminal = [row for row in rows if int(row["is_terminal_push"]) == 1]
    observed = {
        (row["target_id"], row["controller_id"]) for row in terminal
    }
    expected = {
        (target_id, controller_id)
        for target_id in target_ids
        for controller_id in controller_ids
    }
    return (
        observed == expected
        and len(terminal) == len(expected)
        and {row["protocol_version"] for row in rows} == {PROTOCOL_VERSION}
        and {row["role"] for row in rows} == {role}
        and {int(row["maximum_push_budget"]) for row in rows}
        == {maximum_pushes}
    )


def summarise_condition_rows(
    rows: list[dict[str, Any]] | list[dict[str, str]],
    scenario: str,
    condition_id: str,
    path: Path,
    resumed: int,
    artifact_saving_latency_s: float,
) -> dict[str, Any]:
    """汇总一个 condition shard 的闭环 episodes。"""

    terminal = [row for row in rows if int(row["is_terminal_push"]) == 1]
    reasons = Counter(row["terminal_reason"] for row in terminal)
    return {
        "scenario": scenario,
        "condition_id": condition_id,
        "path": str(path.resolve()),
        "episodes": len(terminal),
        "step_rows": len(rows),
        "successful_episodes": int(reasons["success"]),
        "invalid_episodes": int(reasons["invalid_push"]),
        "maximum_budget_episodes": int(reasons["maximum_push_budget"]),
        "artifact_saving_latency_s": artifact_saving_latency_s,
        "resumed": resumed,
    }


def collect_condition_shard(
    scenario: str,
    condition: dict[str, str],
    condition_index: int,
    condition_count: int,
    engine: ClosedLoopDecisionEngine,
    environment_xml: Path,
    output_path: Path,
    targets: list[dict[str, str]],
    controller_ids: tuple[str, ...],
    action_rows_by_id: dict[str, dict[str, str]],
    role: str,
    maximum_pushes: int,
) -> dict[str, Any]:
    """在独立 MuJoCo state 和 CUDA stream 中采集一个 condition shard。"""

    cuda_stream = torch.cuda.Stream(device=engine.device)
    with torch.cuda.stream(cuda_stream):
        model, data = load_model(environment_xml)
        set_sliding_friction(model, float(condition["friction_sliding_mu"]))
        object_body_id = get_body_id(model)
        rows: list[dict[str, Any]] = []
        for target_index, target in enumerate(targets):
            for controller_index, controller_id in enumerate(controller_ids):
                episode_id = (
                    int(condition["condition_index_within_role"]) * 100_000
                    + target_index * 10
                    + controller_index
                )
                rows.extend(
                    run_closed_loop_episode(
                        model=model,
                        data=data,
                        object_body_id=object_body_id,
                        engine=engine,
                        controller_id=controller_id,
                        condition=condition,
                        target=target,
                        action_rows_by_id=action_rows_by_id,
                        episode_id=episode_id,
                        environment_xml=environment_xml,
                        role=role,
                        maximum_pushes=maximum_pushes,
                    )
                )
        cuda_stream.synchronize()
    saving_start = time.perf_counter()
    write_csv(output_path, rows, STEP_FIELDS)
    saving_latency = time.perf_counter() - saving_start
    result = summarise_condition_rows(
        rows,
        scenario,
        condition["condition_id"],
        output_path,
        resumed=0,
        artifact_saving_latency_s=saving_latency,
    )
    result["condition_index"] = condition_index
    result["condition_count"] = condition_count
    return result


def initialise_condition_process_worker(
    scenario: str,
    targets: list[dict[str, str]],
    controller_ids: tuple[str, ...],
    role: str,
    maximum_pushes: int,
    node_query_chunk_size: int,
) -> None:
    """为一个 Windows spawn worker 加载一次场景级 CUDA artifacts。"""

    torch.set_num_threads(1)
    device = require_cuda()
    action_rows = load_actions(ACTION_MANIFEST_PATH)
    action_ids, _ = e3_runner.load_action_layout()
    engine = load_decision_engine(
        scenario, action_ids, device, node_query_chunk_size
    )
    torch.cuda.reset_peak_memory_stats(device)
    PROCESS_WORKER_CONTEXT.clear()
    PROCESS_WORKER_CONTEXT.update(
        {
            "scenario": scenario,
            "targets": targets,
            "controller_ids": controller_ids,
            "role": role,
            "maximum_pushes": maximum_pushes,
            "engine": engine,
            "action_rows_by_id": {
                row["v2_action_id"]: row for row in action_rows
            },
        }
    )


def collect_condition_process_task(
    task: tuple[int, int, dict[str, str], str, str],
) -> dict[str, Any]:
    """在已初始化的独立进程中执行一个 condition shard。"""

    condition_index, condition_count, condition, xml_text, output_text = task
    context = PROCESS_WORKER_CONTEXT
    result = collect_condition_shard(
        scenario=context["scenario"],
        condition=condition,
        condition_index=condition_index,
        condition_count=condition_count,
        engine=context["engine"],
        environment_xml=Path(xml_text),
        output_path=Path(output_text),
        targets=context["targets"],
        controller_ids=context["controller_ids"],
        action_rows_by_id=context["action_rows_by_id"],
        role=context["role"],
        maximum_pushes=context["maximum_pushes"],
    )
    result["worker_process_id"] = os.getpid()
    result["worker_peak_cuda_memory_mib"] = (
        torch.cuda.max_memory_allocated(context["engine"].device)
        / (1024.0**2)
    )
    return result


def collect_closed_loop(args: argparse.Namespace) -> dict[str, Any]:
    """采集 E5 Validation 或 Test 的 controller-specific trajectories。"""

    if args.maximum_pushes != MAXIMUM_PUSHES and not (
        args.max_conditions > 0 or args.max_targets > 0
    ):
        raise ValueError("正式 E5 collection 的 maximum-pushes 必须为 20")
    device = require_cuda()
    selected_scenarios = select_scenarios(args.scenario)
    controller_ids = parse_controller_ids(args.controllers)
    all_targets = load_e5_targets(args.role)
    if args.target_group != "all":
        all_targets = [
            row
            for row in all_targets
            if row["target_group"] == args.target_group
        ]
    if args.max_targets > 0:
        all_targets = all_targets[: args.max_targets]
    formal_run = (
        controller_ids == CONTROLLER_IDS
        and args.target_group == "all"
        and args.max_conditions <= 0
        and args.max_targets <= 0
        and args.maximum_pushes == MAXIMUM_PUSHES
    )
    collection_root = (
        Path(args.data_root) / "closed_loop"
        if formal_run
        else Path(args.data_root) / "smoke"
    )

    action_rows = load_actions(ACTION_MANIFEST_PATH)
    action_rows_by_id = {row["v2_action_id"]: row for row in action_rows}
    action_ids, _ = e3_runner.load_action_layout()
    condition_results: list[dict[str, Any]] = []

    for scenario in selected_scenarios:
        engine = None
        if args.worker_mode == "thread":
            engine = load_decision_engine(
                scenario, action_ids, device, args.node_query_chunk_size
            )
        conditions = e3_runner.load_conditions(scenario, args.role)
        if args.max_conditions > 0:
            conditions = conditions[: args.max_conditions]
        generated_dir = (
            Path(args.data_root)
            / "generated_xml"
            / args.role
            / scenario
        )
        xml_by_com = prepare_environment_xmls(conditions, generated_dir)

        pending_conditions: list[tuple[int, dict[str, str], Path, Path]] = []
        for condition_index, condition in enumerate(conditions, start=1):
            output_path = (
                collection_root
                / args.role
                / scenario
                / f"{condition['condition_id']}.csv"
            )
            target_ids = {row["v2_target_id"] for row in all_targets}
            if args.resume and inspect_complete_condition_shard(
                output_path,
                target_ids,
                controller_ids,
                args.maximum_pushes,
                args.role,
            ):
                condition_results.append(
                    summarise_condition_rows(
                        read_csv_rows(output_path),
                        scenario,
                        condition["condition_id"],
                        output_path,
                        resumed=1,
                        artifact_saving_latency_s=0.0,
                    )
                )
                print(
                    f"resumed {scenario} condition "
                    f"{condition_index}/{len(conditions)}"
                )
                continue

            com_key = (
                round(float(condition["com_offset_x_m"]), 9),
                round(float(condition["com_offset_y_m"]), 9),
            )
            environment_xml = xml_by_com[com_key]
            pending_conditions.append(
                (condition_index, condition, environment_xml, output_path)
            )

        worker_count = min(args.num_workers, len(pending_conditions))
        if worker_count > 0:
            if args.worker_mode == "process":
                executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=mp.get_context("spawn"),
                    initializer=initialise_condition_process_worker,
                    initargs=(
                        scenario,
                        all_targets,
                        controller_ids,
                        args.role,
                        args.maximum_pushes,
                        args.node_query_chunk_size,
                    ),
                )
                futures = {
                    executor.submit(
                        collect_condition_process_task,
                        (
                            condition_index,
                            len(conditions),
                            condition,
                            str(environment_xml),
                            str(output_path),
                        ),
                    ): condition_index
                    for (
                        condition_index,
                        condition,
                        environment_xml,
                        output_path,
                    ) in pending_conditions
                }
            else:
                executor = ThreadPoolExecutor(max_workers=worker_count)
                futures = {
                    executor.submit(
                        collect_condition_shard,
                        scenario,
                        condition,
                        condition_index,
                        len(conditions),
                        engine,
                        environment_xml,
                        output_path,
                        all_targets,
                        controller_ids,
                        action_rows_by_id,
                        args.role,
                        args.maximum_pushes,
                    ): condition_index
                    for (
                        condition_index,
                        condition,
                        environment_xml,
                        output_path,
                    ) in pending_conditions
                }
            with executor:
                for future in as_completed(futures):
                    result = future.result()
                    condition_results.append(result)
                    print(
                        f"finished {scenario} condition "
                        f"{result['condition_index']}/{result['condition_count']}"
                    )
        if engine is not None:
            del engine
            torch.cuda.empty_cache()

    condition_results.sort(
        key=lambda row: (row["scenario"], row["condition_id"])
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "role": args.role,
        "scenario": args.scenario,
        "formal_run": formal_run,
        "single_cuda_process": args.worker_mode == "thread",
        "condition_worker_type": args.worker_mode,
        "num_workers": args.num_workers,
        "shared_cuda_engine_per_scenario": args.worker_mode == "thread",
        "device": torch.cuda.get_device_name(device),
        "controllers": list(controller_ids),
        "target_group": args.target_group,
        "conditions": len(condition_results),
        "targets_per_condition": len(all_targets),
        "maximum_attempted_pushes": args.maximum_pushes,
        "episodes": sum(row["episodes"] for row in condition_results),
        "step_rows": sum(row["step_rows"] for row in condition_results),
        "successful_episodes": sum(
            row["successful_episodes"] for row in condition_results
        ),
        "invalid_episodes": sum(
            row["invalid_episodes"] for row in condition_results
        ),
        "maximum_budget_episodes": sum(
            row["maximum_budget_episodes"] for row in condition_results
        ),
        "resumed_conditions": sum(row["resumed"] for row in condition_results),
        "collection_root": str(collection_root.resolve()),
        "condition_results": condition_results,
    }
    write_json(
        collection_root
        / args.role
        / f"collection_summary_{args.scenario}.json",
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def benchmark_joint(args: argparse.Namespace) -> dict[str, Any]:
    """运行包含一次真实 push、belief update 与再次规划的 Joint benchmark。"""

    device = require_cuda()
    torch.cuda.reset_peak_memory_stats(device)
    action_rows = load_actions(ACTION_MANIFEST_PATH)
    action_rows_by_id = {row["v2_action_id"]: row for row in action_rows}
    action_ids, _ = e3_runner.load_action_layout()
    engine = load_decision_engine(
        "joint", action_ids, device, args.node_query_chunk_size
    )
    condition = e3_runner.load_conditions("joint", "validation")[0]
    source_targets = read_csv_rows(E2_CORE_TARGET_PATHS["validation"])
    target = max(source_targets, key=lambda row: float(row["radial_distance_m"]))
    generated_dir = E5_DATA_ROOT / "generated_xml" / "benchmark" / "joint"
    xml_by_com = prepare_environment_xmls([condition], generated_dir)
    com_key = (
        round(float(condition["com_offset_x_m"]), 9),
        round(float(condition["com_offset_y_m"]), 9),
    )
    environment_xml = xml_by_com[com_key]
    model, data = load_model(environment_xml)
    set_sliding_friction(model, float(condition["friction_sliding_mu"]))
    object_body_id = get_body_id(model)
    reset_input = build_rollout_input(
        action_rows_by_id[action_ids[0]], condition, 0
    )
    reset_state(model, data, reset_input)
    initial_x, initial_y, initial_yaw = read_pose(model, data, object_body_id)
    target_xy = np.asarray(
        [
            initial_x + float(target["target_delta_x_m"]),
            initial_y + float(target["target_delta_y_m"]),
        ]
    )
    true_hidden = active_hidden_parameters("joint", condition)
    belief = engine.new_belief()

    total_start = time.perf_counter()
    query_start = time.perf_counter()
    first_query = make_task_query(
        np.asarray([initial_x, initial_y]),
        initial_yaw,
        target_xy,
        initial_yaw,
    )
    first_query_latency = time.perf_counter() - query_start
    first_decision = engine.select_action(
        BELIEF_MARGINALISED_CONTROLLER_ID,
        first_query,
        belief,
        true_hidden,
    )
    action_id = action_ids[first_decision.action_index]
    rollout_input = build_rollout_input(
        action_rows_by_id[action_id], condition, 0
    )
    rollout_input["dataset_role"] = "hcr_v2_e5_joint_benchmark"
    rollout_input["step_id"] = 1
    simulation_start = time.perf_counter()
    result = run_physical_pusher_atomic_push(model, data, rollout_input)
    simulation_latency = time.perf_counter() - simulation_start
    post_x, post_y, post_yaw = read_pose(model, data, object_body_id)
    valid_observation = (
        int(result["quality_pass"]) == 1
        and int(result["simulation_unstable"]) == 0
        and int(result["contact_success"]) == 1
        and int(result["stopped_by_threshold"]) == 1
    )
    observation = local_motion_observation(
        np.asarray([initial_x, initial_y]),
        initial_yaw,
        np.asarray([post_x, post_y]),
        post_yaw,
    )
    torch.cuda.synchronize(device)
    update_start = time.perf_counter()
    belief_updated = False
    if valid_observation:
        belief_updated = engine.update_belief(
            belief, first_decision.action_index, observation
        )
    torch.cuda.synchronize(device)
    belief_update_latency = time.perf_counter() - update_start
    next_query_start = time.perf_counter()
    next_query = make_task_query(
        np.asarray([post_x, post_y]),
        post_yaw,
        target_xy,
        initial_yaw,
    )
    next_query_latency = time.perf_counter() - next_query_start
    next_decision = engine.select_action(
        BELIEF_MARGINALISED_CONTROLLER_ID,
        next_query,
        belief,
        true_hidden,
    )
    total_latency = time.perf_counter() - total_start

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "scenario": "joint",
        "controller_name": CONTROLLER_NAMES[
            BELIEF_MARGINALISED_CONTROLLER_ID
        ],
        "controller_id": BELIEF_MARGINALISED_CONTROLLER_ID,
        "device": torch.cuda.get_device_name(device),
        "condition_id": condition["condition_id"],
        "target_id": target["v2_target_id"],
        "node_count": int(engine.nodes.shape[0]),
        "first_action_id": action_id,
        "next_action_id": action_ids[next_decision.action_index],
        "valid_observation": int(valid_observation),
        "belief_updated": int(belief_updated),
        "current_state_query_latency_s": first_query_latency,
        "proposal_latency_s": first_decision.proposal_latency_s,
        "selection_latency_s": first_decision.selection_latency_s,
        "simulation_latency_s": simulation_latency,
        "belief_update_latency_s": belief_update_latency,
        "next_state_query_latency_s": next_query_latency,
        "next_proposal_latency_s": next_decision.proposal_latency_s,
        "next_selection_latency_s": next_decision.selection_latency_s,
        "total_wall_time_s": total_latency,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device)
        / (1024.0**2),
        "environment_xml": str(environment_xml.resolve()),
    }
    output_path = E5_RESULTS_ROOT / "benchmark" / "joint_summary.json"
    saving_start = time.perf_counter()
    write_json(output_path, summary)
    summary["artifact_saving_latency_s"] = time.perf_counter() - saving_start
    write_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def descriptive(values: np.ndarray) -> dict[str, float | int | None]:
    """返回论文结果表常用的描述统计。"""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def load_scenario_rows(
    scenario: str, role: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """读取并检查一个场景的 E5 formal step 与 terminal rows。"""

    root = E5_DATA_ROOT / "closed_loop" / role / scenario
    conditions = e3_runner.load_conditions(scenario, role)
    all_rows: list[dict[str, str]] = []
    for condition in conditions:
        path = root / f"{condition['condition_id']}.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少 E5 condition shard: {path}")
        all_rows.extend(read_csv_rows(path))
    rows = [row for row in all_rows if int(row["is_terminal_push"]) == 1]
    expected = len(conditions) * 128 * len(CONTROLLER_IDS)
    if len(rows) != expected:
        raise RuntimeError(
            f"{scenario}/{role} terminal rows 错误: {len(rows)} != {expected}"
        )
    episode_keys = [row["episode_key"] for row in rows]
    if len(set(episode_keys)) != expected:
        raise RuntimeError(f"{scenario}/{role} episode keys 不唯一")
    if {row["protocol_version"] for row in rows} != {PROTOCOL_VERSION}:
        raise RuntimeError(f"{scenario}/{role} protocol version 不一致")
    return all_rows, rows


def controller_summary(
    rows: list[dict[str, str]],
    step_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """汇总一个 controller-regime 的 terminal task performance。"""

    success = np.asarray(
        [int(row["episode_success"]) for row in rows], dtype=np.float64
    )
    terminal_pushes = np.asarray(
        [int(row["terminal_push_count"]) for row in rows], dtype=np.int64
    )
    final_position = np.asarray(
        [float(row["actual_position_error_m"]) for row in rows]
    )
    final_yaw = np.asarray(
        [float(row["actual_yaw_error_rad"]) for row in rows]
    )
    final_cost = np.asarray([float(row["actual_tnpo_cost"]) for row in rows])
    maximum_yaw = np.asarray(
        [float(row["maximum_yaw_deviation_rad"]) for row in rows]
    )
    invalid = np.asarray(
        [int(row["episode_invalid"]) for row in rows], dtype=np.float64
    )
    success_curve = [
        float(np.mean(success * (terminal_pushes <= push_index)))
        for push_index in range(1, MAXIMUM_PUSHES + 1)
    ]
    successful_pushes = terminal_pushes[success.astype(bool)]
    one_push_success = float(np.mean(success * (terminal_pushes == 1)))
    invalid_reasons = Counter()
    for row in rows:
        if int(row["episode_invalid"]) != 1:
            continue
        if int(row["simulation_unstable"]) == 1:
            invalid_reasons["simulation_unstable"] += 1
        if int(row["quality_pass"]) == 0:
            invalid_reasons["quality_failure"] += 1
        if int(row["contact_success"]) == 0:
            invalid_reasons["contact_failure"] += 1
        if int(row["stopped_by_threshold"]) == 0:
            invalid_reasons["settling_failure"] += 1
    return {
        "episodes": len(rows),
        "episode_success_rate": float(np.mean(success)),
        "one_push_success_rate": one_push_success,
        "final_position_error_m": descriptive(final_position),
        "final_yaw_error_rad": descriptive(final_yaw),
        "final_tnpo_cost": descriptive(final_cost),
        "pushes_to_success": descriptive(successful_pushes),
        "success_by_push_curve": success_curve,
        "success_by_push_auc": float(np.mean(success_curve)),
        "maximum_yaw_deviation_rad": descriptive(maximum_yaw),
        "invalid_episode_rate": float(np.mean(invalid)),
        "invalid_reason_counts": dict(invalid_reasons),
        "runtime_s": {
            field: descriptive(
                np.asarray(
                    [float(row[field]) for row in step_rows], dtype=np.float64
                )
            )
            for field in (
                "proposal_latency_s",
                "selection_latency_s",
                "belief_update_latency_s",
                "simulation_latency_s",
            )
        },
    }


@torch.inference_mode()
def ground_truth_one_step_diagnostic(
    scenario: str,
    role: str,
    action_ids: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """用 E1 true outcomes 计算 Sequential-Extension 的单步可达率。"""

    conditions = e3_runner.load_conditions(scenario, role)
    targets = read_csv_rows(E5_SEQUENTIAL_TARGET_PATHS[role])
    outcomes = torch.from_numpy(
        e3_runner.load_outcome_array(
            scenario,
            role,
            conditions,
            action_ids,
        )
    ).to(device)
    target_positions = torch.tensor(
        [
            [float(row["target_delta_x_m"]), float(row["target_delta_y_m"])]
            for row in targets
        ],
        dtype=torch.float32,
        device=device,
    )
    one_step_success = torch.zeros(
        (len(conditions), len(targets)), dtype=torch.bool, device=device
    )
    target_batch_size = 16
    for start in range(0, len(targets), target_batch_size):
        stop = min(start + target_batch_size, len(targets))
        positions = target_positions[start:stop]
        position_error = torch.linalg.vector_norm(
            outcomes[:, None, :, :2] - positions[None, :, None, :], dim=-1
        )
        yaw_error = torch.abs(
            torch.remainder(outcomes[:, None, :, 2] + math.pi, 2.0 * math.pi)
            - math.pi
        )
        one_step_success[:, start:stop] = (
            (position_error <= PRIMARY_TNPO_COST.position_tolerance_m)
            & (yaw_error <= PRIMARY_TNPO_COST.yaw_tolerance_rad)
        ).any(dim=2)

    success_array = one_step_success.cpu().numpy()
    case_rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        for target_index, target in enumerate(targets):
            case_rows.append(
                {
                    "scenario": scenario,
                    "role": role,
                    "condition_id": condition["condition_id"],
                    "target_id": target["v2_target_id"],
                    "target_stratum": target["target_stratum"],
                    "ground_truth_one_step_success": int(
                        success_array[condition_index, target_index]
                    ),
                }
            )

    stratum_summary: dict[str, Any] = {}
    for stratum in sorted({row["target_stratum"] for row in targets}):
        indices = [
            index
            for index, row in enumerate(targets)
            if row["target_stratum"] == stratum
        ]
        values = success_array[:, indices]
        stratum_summary[stratum] = {
            "condition_target_cases": int(values.size),
            "successful_cases": int(values.sum()),
            "success_rate": float(values.mean()),
        }

    summary = {
        "definition": (
            "在当前 split 的真实 hidden condition 与 target 下，4,536 个 "
            "action_core actions 中是否至少存在一个满足 10 mm / 5 deg "
            "criterion 的单步 action"
        ),
        "conditions": len(conditions),
        "targets": len(targets),
        "actions_per_case": len(action_ids),
        "condition_target_cases": int(success_array.size),
        "successful_cases": int(success_array.sum()),
        "success_rate": float(success_array.mean()),
        "targets_with_any_success": int(success_array.any(axis=0).sum()),
        "conditions_with_any_success": int(success_array.any(axis=1).sum()),
        "by_target_stratum": stratum_summary,
    }
    return summary, case_rows


def paired_matrices(
    rows: list[dict[str, str]],
) -> tuple[
    list[str],
    list[str],
    dict[str, np.ndarray],
    dict[str, str],
]:
    """将 terminal rows 对齐为 condition×target paired matrices。"""

    condition_ids = sorted({row["condition_id"] for row in rows})
    target_ids = sorted({row["target_id"] for row in rows})
    condition_index = {value: index for index, value in enumerate(condition_ids)}
    target_index = {value: index for index, value in enumerate(target_ids)}
    shape = (len(condition_ids), len(target_ids))
    metrics: dict[str, np.ndarray] = {}
    for controller_id in CONTROLLER_IDS:
        for metric in (
            "success",
            "terminal_push_count",
            "final_cost",
            "success_auc",
        ):
            metrics[f"{controller_id}|{metric}"] = np.full(
                shape, np.nan, dtype=np.float64
            )
    target_strata: dict[str, str] = {}
    for row in rows:
        condition_slot = condition_index[row["condition_id"]]
        target_slot = target_index[row["target_id"]]
        controller_id = row["controller_id"]
        success = float(row["episode_success"])
        push_count = int(row["terminal_push_count"])
        metrics[f"{controller_id}|success"][condition_slot, target_slot] = success
        metrics[f"{controller_id}|terminal_push_count"][
            condition_slot, target_slot
        ] = push_count
        metrics[f"{controller_id}|final_cost"][condition_slot, target_slot] = float(
            row["actual_tnpo_cost"]
        )
        metrics[f"{controller_id}|success_auc"][condition_slot, target_slot] = (
            (MAXIMUM_PUSHES + 1 - push_count) / MAXIMUM_PUSHES
            if success == 1.0
            else 0.0
        )
        target_strata[row["target_id"]] = row["target_stratum"]
    if any(np.isnan(values).any() for values in metrics.values()):
        raise RuntimeError("E5 paired matrices 存在缺失 controller episodes")
    return condition_ids, target_ids, metrics, target_strata


def primary_effect_arrays(metrics: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """构造三项 Primary Hypotheses 的 paired episode effects。"""

    return {
        "primary_hypothesis_1_episode_success_rate": (
            metrics[f"{BELIEF_MARGINALISED_CONTROLLER_ID}|success"]
            - metrics[f"{NOMINAL_CONTROLLER_ID}|success"]
        ),
        "primary_hypothesis_2_mean_final_tnpo_cost": (
            metrics[f"{CERTAINTY_EQUIVALENT_CONTROLLER_ID}|final_cost"]
            - metrics[f"{BELIEF_MARGINALISED_CONTROLLER_ID}|final_cost"]
        ),
        "primary_hypothesis_3_success_by_push_auc": (
            metrics[f"{BELIEF_MARGINALISED_CONTROLLER_ID}|success_auc"]
            - metrics[f"{CERTAINTY_EQUIVALENT_CONTROLLER_ID}|success_auc"]
        ),
    }


def paired_two_way_bootstrap(
    rows: list[dict[str, str]],
    resamples: int,
    seed: int,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """执行 condition-target-stratified paired two-way bootstrap。"""

    condition_ids, target_ids, metrics, target_strata = paired_matrices(rows)
    effects = primary_effect_arrays(metrics)
    strata = sorted(set(target_strata.values()))
    stratum_indices = {
        stratum: np.asarray(
            [
                index
                for index, target_id in enumerate(target_ids)
                if target_strata[target_id] == stratum
            ],
            dtype=np.int64,
        )
        for stratum in strata
    }
    rng = np.random.default_rng(seed)
    samples = {
        name: np.empty(resamples, dtype=np.float64) for name in effects
    }
    curve_success: list[np.ndarray] = []
    curve_push_count: list[np.ndarray] = []
    for controller_id in CONTROLLER_IDS:
        curve_success.append(
            metrics[f"{controller_id}|success"].reshape(-1).astype(bool)
        )
        curve_push_count.append(
            metrics[f"{controller_id}|terminal_push_count"]
            .reshape(-1)
            .astype(np.int64)
        )
    curve_samples = np.empty(
        (resamples, len(CONTROLLER_IDS) * MAXIMUM_PUSHES),
        dtype=np.float64,
    )
    condition_count = len(condition_ids)
    for sample_index in range(resamples):
        condition_counts = np.bincount(
            rng.integers(0, condition_count, size=condition_count),
            minlength=condition_count,
        ).astype(np.float64)
        target_counts = np.zeros(len(target_ids), dtype=np.float64)
        for indices in stratum_indices.values():
            draws = rng.integers(0, len(indices), size=len(indices))
            target_counts[indices] = np.bincount(
                draws, minlength=len(indices)
            )
        weights = condition_counts[:, None] * target_counts[None, :]
        denominator = float(weights.sum())
        for name, values in effects.items():
            samples[name][sample_index] = float(
                np.sum(weights * values) / denominator
            )
        flat_weights = weights.reshape(-1)
        for controller_index in range(len(CONTROLLER_IDS)):
            success_mask = curve_success[controller_index]
            histogram = np.bincount(
                curve_push_count[controller_index][success_mask],
                weights=flat_weights[success_mask],
                minlength=MAXIMUM_PUSHES + 1,
            )
            start = controller_index * MAXIMUM_PUSHES
            stop = start + MAXIMUM_PUSHES
            curve_samples[sample_index, start:stop] = (
                np.cumsum(histogram[1 : MAXIMUM_PUSHES + 1]) / denominator
            )
    summary = {
        name: {
            "point_estimate": float(np.mean(values)),
            "ci_95_low": float(np.quantile(samples[name], 0.025)),
            "ci_95_high": float(np.quantile(samples[name], 0.975)),
            "positive_effect_probability": float(np.mean(samples[name] > 0.0)),
            "resamples": resamples,
            "seed": seed,
        }
        for name, values in effects.items()
    }
    curve_bands: dict[str, Any] = {}
    curve_points = []
    for controller_index in range(len(CONTROLLER_IDS)):
        histogram = np.bincount(
            curve_push_count[controller_index][curve_success[controller_index]],
            minlength=MAXIMUM_PUSHES + 1,
        )
        curve_points.extend(
            (
                np.cumsum(histogram[1 : MAXIMUM_PUSHES + 1])
                / curve_success[controller_index].size
            ).tolist()
        )
    curve_points_array = np.asarray(curve_points, dtype=np.float64)
    curve_low = np.quantile(curve_samples, 0.025, axis=0)
    curve_high = np.quantile(curve_samples, 0.975, axis=0)
    for controller_index, controller_id in enumerate(CONTROLLER_IDS):
        start = controller_index * MAXIMUM_PUSHES
        stop = start + MAXIMUM_PUSHES
        curve_bands[controller_id] = {
            "push_indices": list(range(1, MAXIMUM_PUSHES + 1)),
            "point_estimate": curve_points_array[start:stop].tolist(),
            "ci_95_low": curve_low[start:stop].tolist(),
            "ci_95_high": curve_high[start:stop].tolist(),
            "resamples": resamples,
            "seed": seed,
        }
    return summary, samples, curve_bands


def paired_win_tie_loss(rows: list[dict[str, str]]) -> dict[str, Any]:
    """按 actual final TNPO cost 汇总重要 controller pairs。"""

    by_case: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_case[(row["condition_id"], row["target_id"])][
            row["controller_id"]
        ] = float(row["actual_tnpo_cost"])
    pairs = (
        (BELIEF_MARGINALISED_CONTROLLER_ID, NOMINAL_CONTROLLER_ID),
        (
            BELIEF_MARGINALISED_CONTROLLER_ID,
            CERTAINTY_EQUIVALENT_CONTROLLER_ID,
        ),
        (
            CERTAINTY_EQUIVALENT_CONTROLLER_ID,
            NOMINAL_CONTROLLER_ID,
        ),
        (BELIEF_MARGINALISED_CONTROLLER_ID, FULL_INFORMATION_CONTROLLER_ID),
    )
    result: dict[str, Any] = {}
    for left, right in pairs:
        wins = ties = losses = 0
        for costs in by_case.values():
            difference = costs[left] - costs[right]
            if difference < -1e-12:
                wins += 1
            elif difference > 1e-12:
                losses += 1
            else:
                ties += 1
        result[f"{left}_versus_{right}"] = {
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """评价一个 E5 split 并执行正式 paired bootstrap。"""

    selected_scenarios = select_scenarios(args.scenario)
    device = require_cuda()
    action_ids, _ = e3_runner.load_action_layout()
    scenario_summaries: dict[str, Any] = {}
    effect_samples: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    seed_base = 2_026_081_300 if args.role == "validation" else 2_026_081_400

    for scenario_index, scenario in enumerate(selected_scenarios):
        all_step_rows, terminal_rows = load_scenario_rows(scenario, args.role)
        one_step_diagnostic, one_step_case_rows = (
            ground_truth_one_step_diagnostic(
                scenario,
                args.role,
                action_ids,
                device,
            )
        )
        scenario_summary: dict[str, Any] = {
            "scenario": scenario,
            "role": args.role,
            "ground_truth_one_step_diagnostic": one_step_diagnostic,
            "regimes": {},
        }
        effect_samples[scenario] = {}
        for regime_index, regime in enumerate(
            ("core", "sequential_extension")
        ):
            regime_rows = [
                row for row in terminal_rows if row["target_group"] == regime
            ]
            method_summaries = {
                controller_id: controller_summary(
                    [
                        row
                        for row in regime_rows
                        if row["controller_id"] == controller_id
                    ],
                    [
                        row
                        for row in all_step_rows
                        if row["target_group"] == regime
                        and row["controller_id"] == controller_id
                    ],
                )
                for controller_id in CONTROLLER_IDS
            }
            effects, samples, curve_bands = paired_two_way_bootstrap(
                regime_rows,
                args.bootstrap_resamples,
                seed_base + scenario_index * 10 + regime_index,
            )
            scenario_summary["regimes"][regime] = {
                "controllers": method_summaries,
                "primary_paired_effects": effects,
                "success_by_push_confidence_bands": curve_bands,
                "paired_win_tie_loss": paired_win_tie_loss(regime_rows),
            }
            effect_samples[scenario][regime] = samples
        scenario_summaries[scenario] = scenario_summary
        output_dir = E5_RESULTS_ROOT / "evaluation" / args.role / scenario
        write_json(output_dir / "summary.json", scenario_summary)
        write_csv(
            output_dir / "ground_truth_one_step_diagnostic.csv",
            one_step_case_rows,
            [
                "scenario",
                "role",
                "condition_id",
                "target_id",
                "target_stratum",
                "ground_truth_one_step_success",
            ],
        )
        effect_rows: list[dict[str, Any]] = []
        curve_rows: list[dict[str, Any]] = []
        for regime, payload in scenario_summary["regimes"].items():
            for hypothesis, values in payload["primary_paired_effects"].items():
                effect_rows.append(
                    {"target_regime": regime, "hypothesis": hypothesis, **values}
                )
            for controller_id, values in payload[
                "success_by_push_confidence_bands"
            ].items():
                for index, push_index in enumerate(values["push_indices"]):
                    curve_rows.append(
                        {
                            "target_regime": regime,
                            "controller_id": controller_id,
                            "push_index": push_index,
                            "point_estimate": values["point_estimate"][index],
                            "ci_95_low": values["ci_95_low"][index],
                            "ci_95_high": values["ci_95_high"][index],
                            "resamples": values["resamples"],
                            "seed": values["seed"],
                        }
                    )
        write_csv(
            output_dir / "primary_paired_effects.csv",
            effect_rows,
            [
                "target_regime",
                "hypothesis",
                "point_estimate",
                "ci_95_low",
                "ci_95_high",
                "positive_effect_probability",
                "resamples",
                "seed",
            ],
        )
        write_csv(
            output_dir / "success_by_push_confidence_bands.csv",
            curve_rows,
            [
                "target_regime",
                "controller_id",
                "push_index",
                "point_estimate",
                "ci_95_low",
                "ci_95_high",
                "resamples",
                "seed",
            ],
        )
        print(f"evaluated {scenario} {args.role}")

    macro: dict[str, Any] = {}
    for regime in ("core", "sequential_extension"):
        macro[regime] = {}
        hypothesis_names = next(iter(effect_samples.values()))[regime].keys()
        for hypothesis in hypothesis_names:
            arrays = [
                effect_samples[scenario][regime][hypothesis]
                for scenario in selected_scenarios
            ]
            macro_samples = np.mean(np.stack(arrays, axis=0), axis=0)
            points = [
                scenario_summaries[scenario]["regimes"][regime][
                    "primary_paired_effects"
                ][hypothesis]["point_estimate"]
                for scenario in selected_scenarios
            ]
            macro[regime][hypothesis] = {
                "point_estimate": float(np.mean(points)),
                "ci_95_low": float(np.quantile(macro_samples, 0.025)),
                "ci_95_high": float(np.quantile(macro_samples, 0.975)),
                "equal_weight_scenarios": list(selected_scenarios),
            }
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "role": args.role,
        "bootstrap_resamples": args.bootstrap_resamples,
        "scenarios": scenario_summaries,
        "equal_weight_macro_primary_effects": macro,
        "headline_regime": "sequential_extension",
    }
    output_path = E5_RESULTS_ROOT / "evaluation" / args.role / "combined_summary.json"
    write_json(output_path, combined)
    print(f"Combined summary: {output_path.resolve()}")
    return combined


def build_parser() -> argparse.ArgumentParser:
    """构建 E5 统一命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    targets = subparsers.add_parser(
        "prepare-targets",
        help="生成 64 Core + 64 Sequential-Extension Validation/Test manifests",
    )
    targets.add_argument("--candidate-batch-size", type=int, default=32)
    targets.set_defaults(handler=prepare_targets)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="运行 Joint 单 CUDA 进程端到端 benchmark",
    )
    benchmark_parser.add_argument(
        "--node-query-chunk-size", type=int, default=65_536
    )
    benchmark_parser.set_defaults(handler=benchmark_joint)

    collect_parser = subparsers.add_parser(
        "collect",
        help="采集 controller-specific closed-loop Validation 或 Test trajectories",
    )
    collect_parser.add_argument(
        "--scenario", choices=(*SCENARIOS, "all"), default="all"
    )
    collect_parser.add_argument(
        "--role", choices=("validation", "test"), required=True
    )
    collect_parser.add_argument(
        "--controllers",
        default="all",
        help=(
            "all 或逗号分隔的完整 controller identifiers；"
            "非 all 运行写入 smoke 目录"
        ),
    )
    collect_parser.add_argument(
        "--target-group",
        choices=("all", "core", "sequential_extension"),
        default="all",
        help="仅限制 partial/smoke 运行范围；正式 collection 使用 all",
    )
    collect_parser.add_argument(
        "--maximum-pushes", type=int, default=MAXIMUM_PUSHES
    )
    collect_parser.add_argument(
        "--node-query-chunk-size", type=int, default=65_536
    )
    collect_parser.add_argument("--max-conditions", type=int, default=0)
    collect_parser.add_argument("--max-targets", type=int, default=0)
    collect_parser.add_argument("--num-workers", type=int, default=8)
    collect_parser.add_argument(
        "--worker-mode", choices=("process", "thread"), default="process"
    )
    collect_parser.add_argument("--resume", action="store_true")
    collect_parser.add_argument(
        "--data-root", type=Path, default=E5_DATA_ROOT
    )
    collect_parser.set_defaults(handler=collect_closed_loop)

    evaluation = subparsers.add_parser(
        "evaluate",
        help="评价已采集的正式 closed-loop trajectories",
    )
    evaluation.add_argument(
        "--scenario", choices=(*SCENARIOS, "all"), default="all"
    )
    evaluation.add_argument(
        "--role", choices=("validation", "test"), required=True
    )
    evaluation.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    evaluation.set_defaults(handler=evaluate)
    return parser


def main() -> None:
    """解析命令并运行对应 E5 阶段。"""

    args = build_parser().parse_args()
    if hasattr(args, "candidate_batch_size") and args.candidate_batch_size <= 0:
        raise ValueError("candidate-batch-size 必须大于 0")
    if hasattr(args, "node_query_chunk_size") and args.node_query_chunk_size <= 0:
        raise ValueError("node-query-chunk-size 必须大于 0")
    if hasattr(args, "bootstrap_resamples") and args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples 必须大于 0")
    if hasattr(args, "num_workers") and args.num_workers <= 0:
        raise ValueError("num-workers 必须大于 0")
    args.handler(args)


if __name__ == "__main__":
    main()
