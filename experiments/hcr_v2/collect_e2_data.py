"""采集 HCR V2 E2 所需的 Training outcomes、targets 与连续 histories。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_e1 import (
    ACTION_FEATURE_FIELDS,
    ACTION_MANIFEST_PATH,
    BASE_XML_PATH,
    E1_RESULTS_ROOT,
    FRICTION_CONE,
    OUTCOME_FIELDS,
    PROTOCOL_VERSION as E1_PROTOCOL_VERSION,
    build_rollout_input,
    load_actions,
    load_conditions,
    prepare_environment_xmls,
    read_csv_rows,
    set_sliding_friction,
    write_json,
)
from push_core.project_paths import HCR_V2_DATA_DIR
from push_core.hcr_v2.e1 import (
    PRIMARY_TNPO_COST,
    TensorOutcomeInterpolator,
    wrap_to_pi,
)
from push_core.hcr_v2.e2 import local_motion_observation
from push_core.simulation.physical_pusher_rollout import (
    get_body_id,
    get_object_yaw_qpos,
    load_model,
    reset_state,
    run_physical_pusher_atomic_push,
    run_physical_pusher_rollout,
)


PROTOCOL_VERSION = "hcr_v2_e2_v1"
SCENARIOS = ("friction", "com", "joint")
E2_DATA_ROOT = HCR_V2_DATA_DIR / "e2"
MANIFEST_DIR = PROJECT_ROOT / "manifests" / "hcr_v2"
CORE_VALIDATION_TARGET_PATH = (
    MANIFEST_DIR / "hcr_v2_e2_core_validation_target_manifest_v1.csv"
)
LONG_VALIDATION_TARGET_PATH = (
    MANIFEST_DIR / "hcr_v2_e2_long_distance_validation_target_manifest_v1.csv"
)
CORE_TEST_TARGET_PATH = (
    MANIFEST_DIR / "hcr_v2_e2_core_test_target_manifest_v1.csv"
)
LONG_TEST_TARGET_PATH = (
    MANIFEST_DIR / "hcr_v2_e2_long_distance_test_target_manifest_v1.csv"
)
PROVISIONAL_VALIDATION_PUSH_CEILING = 20
FINAL_TEST_MAXIMUM_PUSH_BUDGET = 20

TRAINING_OUTCOME_FIELDS = [
    "experiment_id",
    "protocol_version",
    "friction_cone",
    "environment_xml",
    "scenario",
    "e2_data_role",
    "source_condition_role",
    "source_validation_partition",
    "hidden_parameter_dimension",
    "condition_id",
    "condition_index_within_role",
    "v2_action_id",
    "candidate_id",
    "action_param_index",
    "contact_region_id",
    "surface_id",
    "contact_region_row",
    "contact_region_col",
    "force_angle_relative_to_normal_deg",
    *ACTION_FEATURE_FIELDS,
    "execution_duration_s",
    "friction_sliding_mu",
    "com_offset_x_m",
    "com_offset_y_m",
    "hidden_u_friction",
    "hidden_u_com_x",
    "hidden_u_com_y",
    *OUTCOME_FIELDS,
    "real_delta_z",
    "quality_pass",
    "simulation_unstable",
    "contact_success",
    "num_contacts",
    "stopped_by_threshold",
]

TARGET_CANDIDATE_FIELDS = [
    "experiment_id",
    "protocol_version",
    "friction_cone",
    "environment_xml",
    "split_role",
    "base_random_seed",
    "trajectory_id",
    "step_index",
    "v2_action_id",
    "candidate_id",
    "prefix_action_ids",
    "initial_x_m",
    "initial_y_m",
    "initial_yaw_rad",
    "pre_push_x_m",
    "pre_push_y_m",
    "pre_push_yaw_rad",
    "final_x_m",
    "final_y_m",
    "final_yaw_rad",
    "step_delta_local_x_m",
    "step_delta_local_y_m",
    "step_delta_yaw_rad",
    "target_delta_x_m",
    "target_delta_y_m",
    "target_relative_yaw_rad",
    "target_distance_m",
    "target_key_4dp",
    "push_valid",
    "prefix_valid",
    "candidate_eligible",
    "quality_pass",
    "simulation_unstable",
    "contact_success",
    "stopped_by_threshold",
    "num_contacts",
    "settle_time_s",
]

HISTORY_FIELDS = [
    "experiment_id",
    "protocol_version",
    "friction_cone",
    "environment_xml",
    "e2_data_role",
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
    "target_yaw_offset_rad",
    "target_radial_distance_m",
    "episode_key",
    "episode_index_within_condition",
    "push_index",
    "v2_action_id",
    "candidate_id",
    "action_param_index",
    "pre_push_x_m",
    "pre_push_y_m",
    "pre_push_yaw_rad",
    "final_x_m",
    "final_y_m",
    "final_yaw_rad",
    "observation_local_delta_x_m",
    "observation_local_delta_y_m",
    "observation_delta_yaw_rad",
    "target_world_x_m",
    "target_world_y_m",
    "target_world_yaw_rad",
    "predicted_local_delta_x_m",
    "predicted_local_delta_y_m",
    "predicted_delta_yaw_rad",
    "predicted_final_x_m",
    "predicted_final_y_m",
    "predicted_final_yaw_rad",
    "predicted_position_error_m",
    "predicted_yaw_error_rad",
    "predicted_tnpo_cost",
    "actual_position_error_m",
    "actual_yaw_error_rad",
    "actual_tnpo_cost",
    "success_after_push",
    "valid_observation",
    "valid_update_index",
    "quality_pass",
    "simulation_unstable",
    "contact_success",
    "stopped_by_threshold",
    "num_contacts",
    "settle_time_s",
    "episode_valid",
    "episode_success",
    "terminal_reason",
    "terminal_push_count",
    "is_terminal_push",
    "maximum_push_budget",
    "behaviour_policy",
    "behaviour_p1_artifact",
]

_TRAINING_ACTIONS: list[dict[str, str]] = []
_TARGET_ACTIONS: list[dict[str, str]] = []
_TARGET_MODEL = None
_TARGET_DATA = None
_TARGET_OBJECT_BODY_ID = -1
_TARGET_BASE_SEED = 0
_TARGET_SPLIT_ROLE = "validation"
_HISTORY_ACTIONS: list[dict[str, str]] = []
_HISTORY_NOMINAL_OUTCOMES: dict[str, np.ndarray] = {}
_HISTORY_BEHAVIOUR_P1_PATHS: dict[str, str] = {}


def select_scenarios(value: str) -> tuple[str, ...]:
    """解析单一场景或全部场景。"""

    if value == "all":
        return SCENARIOS
    if value not in SCENARIOS:
        raise ValueError(f"未知 scenario: {value}")
    return (value,)


def load_e1_p1_metadata(scenarios: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """读取并核对 E2 后续使用的 Elliptic E1 P1 元数据。"""

    expected_environment = str(BASE_XML_PATH.resolve())
    p1_metadata: dict[str, dict[str, str]] = {}
    for scenario in scenarios:
        artifact_path = (
            E1_RESULTS_ROOT
            / "p1"
            / scenario
            / "tensor_outcome_interpolator.npz"
        )
        metadata_path = artifact_path.parent / "metadata.json"
        if not artifact_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"缺少 {scenario} 的 E1 P1 artifact 或 metadata")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_values = {
            "protocol_version": E1_PROTOCOL_VERSION,
            "friction_cone": FRICTION_CONE,
            "environment_xml": expected_environment,
            "scenario": scenario,
            "predictor": "p1",
        }
        for key, expected in expected_values.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"{scenario} E1 P1 metadata 的 {key} 不匹配: "
                    f"{metadata.get(key)!r} != {expected!r}"
                )
        if Path(metadata.get("artifact_path", "")).resolve() != artifact_path.resolve():
            raise RuntimeError(f"{scenario} E1 P1 metadata 的 artifact_path 不匹配")

        p1_metadata[scenario] = {
            **expected_values,
            "artifact_path": str(artifact_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
        }
    return p1_metadata


def select_e2_training_conditions(
    conditions: list[dict[str, str]],
    scenario: str,
) -> list[dict[str, str]]:
    """选择 E1 calibration partition 作为 E2 Training conditions。"""

    selected = [
        row
        for row in conditions
        if row["scenario"] == scenario
        and row["condition_role"] == "validation"
        and row["validation_partition"] == "calibration"
    ]
    selected.sort(key=lambda row: int(row["condition_index_within_role"]))
    if len(selected) != 16:
        raise RuntimeError(
            f"{scenario} E2 Training condition 数量错误: {len(selected)}"
        )
    return selected


def inspect_complete_training_shard(
    path: Path,
    expected_action_ids: set[str],
) -> bool:
    """判断现有 condition shard 是否已采集完整。"""

    if not path.exists():
        return False
    rows = read_csv_rows(path)
    observed_ids = {row["v2_action_id"] for row in rows}
    expected_environment = str(BASE_XML_PATH.resolve())
    metadata_matches = all(
        row.get("friction_cone") == FRICTION_CONE
        and row.get("environment_xml") == expected_environment
        for row in rows
    )
    return (
        len(rows) == len(expected_action_ids)
        and observed_ids == expected_action_ids
        and metadata_matches
    )


def initialise_training_worker(action_manifest_path: str, max_actions: int) -> None:
    """为 Training outcome worker 加载共享 action_core。"""

    global _TRAINING_ACTIONS
    actions = load_actions(Path(action_manifest_path))
    _TRAINING_ACTIONS = actions if max_actions <= 0 else actions[:max_actions]


def process_training_condition(task: dict[str, Any]) -> dict[str, Any]:
    """在一个 E2 Training hidden condition 下采集所有 action outcomes。"""

    condition = task["condition"]
    output_path = Path(task["output_path"])
    expected_action_ids = {row["v2_action_id"] for row in _TRAINING_ACTIONS}
    if bool(task["resume"]) and inspect_complete_training_shard(
        output_path,
        expected_action_ids,
    ):
        existing = read_csv_rows(output_path)
        return {
            "scenario": condition["scenario"],
            "condition_id": condition["condition_id"],
            "path": str(output_path.resolve()),
            "rollouts": len(existing),
            "quality_pass_count": sum(int(row["quality_pass"]) for row in existing),
            "simulation_unstable_count": sum(
                int(row["simulation_unstable"]) for row in existing
            ),
            "resumed": 1,
        }

    model, data = load_model(Path(task["xml_path"]))
    set_sliding_friction(model, float(condition["friction_sliding_mu"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality_count = 0
    unstable_count = 0
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_OUTCOME_FIELDS)
        writer.writeheader()
        for action_index, action in enumerate(_TRAINING_ACTIONS):
            episode_id = (
                int(condition["condition_index_within_role"]) * 10_000
                + action_index
            )
            rollout_input = build_rollout_input(action, condition, episode_id)
            rollout_input["dataset_role"] = (
                f"hcr_v2_e2_training_{condition['scenario']}"
            )
            result = run_physical_pusher_rollout(
                model,
                data,
                rollout_input,
                validate_schema=False,
            )
            output_row = {
                "experiment_id": "E2",
                "protocol_version": PROTOCOL_VERSION,
                "friction_cone": FRICTION_CONE,
                "environment_xml": str(BASE_XML_PATH.resolve()),
                "scenario": condition["scenario"],
                "e2_data_role": "training",
                "source_condition_role": condition["condition_role"],
                "source_validation_partition": condition["validation_partition"],
                "hidden_parameter_dimension": condition[
                    "hidden_parameter_dimension"
                ],
                "condition_id": condition["condition_id"],
                "condition_index_within_role": condition[
                    "condition_index_within_role"
                ],
                "v2_action_id": action["v2_action_id"],
                "candidate_id": action["candidate_id"],
                "action_param_index": action["action_param_index"],
                "contact_region_id": action["contact_region_id"],
                "surface_id": action["surface_id"],
                "contact_region_row": action["contact_region_row"],
                "contact_region_col": action["contact_region_col"],
                "force_angle_relative_to_normal_deg": action[
                    "force_angle_relative_to_normal_deg"
                ],
                **{field: action[field] for field in ACTION_FEATURE_FIELDS},
                "execution_duration_s": action["execution_duration_s"],
                "friction_sliding_mu": condition["friction_sliding_mu"],
                "com_offset_x_m": condition["com_offset_x_m"],
                "com_offset_y_m": condition["com_offset_y_m"],
                "hidden_u_friction": condition["hidden_u_friction"],
                "hidden_u_com_x": condition["hidden_u_com_x"],
                "hidden_u_com_y": condition["hidden_u_com_y"],
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
            writer.writerow(output_row)
            quality_count += int(result["quality_pass"])
            unstable_count += int(result["simulation_unstable"])

    return {
        "scenario": condition["scenario"],
        "condition_id": condition["condition_id"],
        "path": str(output_path.resolve()),
        "rollouts": len(_TRAINING_ACTIONS),
        "quality_pass_count": quality_count,
        "simulation_unstable_count": unstable_count,
        "resumed": 0,
    }


def collect_training_outcomes(args: argparse.Namespace) -> dict[str, Any]:
    """采集 E2 residual statistics 所需的 Training outcomes。"""

    actions = load_actions()
    if args.max_actions > 0:
        actions = actions[: args.max_actions]
    conditions = load_conditions()
    data_root = Path(args.data_root)
    selected_scenarios = select_scenarios(args.scenario)
    e1_p1_artifacts = load_e1_p1_metadata(selected_scenarios)
    tasks: list[dict[str, Any]] = []
    for scenario in selected_scenarios:
        selected = select_e2_training_conditions(conditions, scenario)
        generated_dir = data_root / "generated_xml" / "training" / scenario
        xml_by_com = prepare_environment_xmls(selected, generated_dir)
        for condition in selected:
            com_key = (
                round(float(condition["com_offset_x_m"]), 9),
                round(float(condition["com_offset_y_m"]), 9),
            )
            tasks.append(
                {
                    "condition": condition,
                    "xml_path": str(xml_by_com[com_key]),
                    "output_path": str(
                        data_root
                        / "training_outcomes"
                        / scenario
                        / f"{condition['condition_id']}.csv"
                    ),
                    "resume": args.resume,
                }
            )

    worker_count = min(max(1, int(args.num_workers)), len(tasks))
    results: list[dict[str, Any]] = []
    if worker_count == 1:
        initialise_training_worker(str(ACTION_MANIFEST_PATH), len(actions))
        for index, task in enumerate(tasks, start=1):
            results.append(process_training_condition(task))
            print(f"finished E2 Training condition {index}/{len(tasks)}")
    else:
        with mp.Pool(
            processes=worker_count,
            initializer=initialise_training_worker,
            initargs=(str(ACTION_MANIFEST_PATH), len(actions)),
            maxtasksperchild=4,
        ) as pool:
            for index, result in enumerate(
                pool.imap_unordered(process_training_condition, tasks),
                start=1,
            ):
                results.append(result)
                print(f"finished E2 Training condition {index}/{len(tasks)}")

    results.sort(key=lambda row: (row["scenario"], row["condition_id"]))
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "environment_xml": str(BASE_XML_PATH.resolve()),
        "e1_p1_artifacts": e1_p1_artifacts,
        "data_role": "training",
        "source_partition": "E1 validation/calibration",
        "scenario": args.scenario,
        "conditions": len(tasks),
        "actions_per_condition": len(actions),
        "rollouts": sum(int(row["rollouts"]) for row in results),
        "quality_pass_count": sum(
            int(row["quality_pass_count"]) for row in results
        ),
        "simulation_unstable_count": sum(
            int(row["simulation_unstable_count"]) for row in results
        ),
        "resumed_conditions": sum(int(row["resumed"]) for row in results),
        "num_workers": worker_count,
        "condition_results": results,
    }
    write_json(
        data_root
        / "training_outcomes"
        / f"collection_summary_{args.scenario}.json",
        summary,
    )
    print_json(summary)
    return summary


def nominal_condition(split_role: str) -> dict[str, str]:
    """构造长距离 target generation 使用的 nominal hidden condition。"""

    return {
        "scenario": "joint",
        "hidden_parameter_dimension": "3",
        "condition_role": split_role,
        "validation_partition": (
            "model_selection" if split_role == "validation" else "not_applicable"
        ),
        "condition_id": "NOMINAL",
        "condition_index_within_role": "0",
        "friction_sliding_mu": "0.40",
        "com_offset_x_m": "0.0",
        "com_offset_y_m": "0.0",
        "hidden_u_friction": "0.0",
        "hidden_u_com_x": "0.0",
        "hidden_u_com_y": "0.0",
    }


def initialise_target_worker(
    action_manifest_path: str,
    base_seed: int,
    split_role: str,
) -> None:
    """为长距离 target trajectory worker 加载模型和 action_core。"""

    global _TARGET_ACTIONS
    global _TARGET_MODEL
    global _TARGET_DATA
    global _TARGET_OBJECT_BODY_ID
    global _TARGET_BASE_SEED
    global _TARGET_SPLIT_ROLE
    _TARGET_ACTIONS = load_actions(Path(action_manifest_path))
    _TARGET_MODEL, _TARGET_DATA = load_model(BASE_XML_PATH)
    set_sliding_friction(_TARGET_MODEL, 0.40)
    _TARGET_OBJECT_BODY_ID = get_body_id(_TARGET_MODEL)
    _TARGET_BASE_SEED = int(base_seed)
    _TARGET_SPLIT_ROLE = split_role


def read_object_pose() -> tuple[float, float, float]:
    """读取 target worker 中物体的平面位姿。"""

    return (
        float(_TARGET_DATA.xpos[_TARGET_OBJECT_BODY_ID][0]),
        float(_TARGET_DATA.xpos[_TARGET_OBJECT_BODY_ID][1]),
        get_object_yaw_qpos(_TARGET_MODEL, _TARGET_DATA),
    )


def process_target_trajectory(trajectory_id: int) -> list[dict[str, Any]]:
    """连续执行五次随机 atomic pushes 并记录第 2 至 5 步候选。"""

    rng = np.random.default_rng(
        np.random.SeedSequence([_TARGET_BASE_SEED, int(trajectory_id)])
    )
    action_indices = rng.integers(0, len(_TARGET_ACTIONS), size=5)
    condition = nominal_condition(_TARGET_SPLIT_ROLE)
    first_action = _TARGET_ACTIONS[int(action_indices[0])]
    first_input = build_rollout_input(first_action, condition, trajectory_id * 10)
    reset_state(_TARGET_MODEL, _TARGET_DATA, first_input)
    initial_x, initial_y, initial_yaw = read_object_pose()

    rows: list[dict[str, Any]] = []
    prefix_action_ids: list[str] = []
    prefix_valid = True
    for step_index, action_index in enumerate(action_indices, start=1):
        action = _TARGET_ACTIONS[int(action_index)]
        prefix_action_ids.append(action["v2_action_id"])
        pre_x, pre_y, pre_yaw = read_object_pose()
        rollout_input = build_rollout_input(
            action,
            condition,
            trajectory_id * 10 + step_index,
        )
        rollout_input["dataset_role"] = (
            f"hcr_v2_e2_target_candidate_{_TARGET_SPLIT_ROLE}"
        )
        result = run_physical_pusher_atomic_push(
            _TARGET_MODEL,
            _TARGET_DATA,
            rollout_input,
        )
        final_x, final_y, final_yaw = read_object_pose()
        cosine = math.cos(pre_yaw)
        sine = math.sin(pre_yaw)
        world_dx = final_x - pre_x
        world_dy = final_y - pre_y
        local_dx = cosine * world_dx + sine * world_dy
        local_dy = -sine * world_dx + cosine * world_dy
        push_valid = (
            int(result["quality_pass"]) == 1
            and int(result["simulation_unstable"]) == 0
        )
        prefix_valid = prefix_valid and push_valid
        target_dx = final_x - initial_x
        target_dy = final_y - initial_y
        target_yaw = final_yaw - initial_yaw
        rows.append(
            {
                "experiment_id": "E2",
                "protocol_version": PROTOCOL_VERSION,
                "friction_cone": FRICTION_CONE,
                "environment_xml": str(BASE_XML_PATH.resolve()),
                "split_role": _TARGET_SPLIT_ROLE,
                "base_random_seed": _TARGET_BASE_SEED,
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "v2_action_id": action["v2_action_id"],
                "candidate_id": action["candidate_id"],
                "prefix_action_ids": "|".join(prefix_action_ids),
                "initial_x_m": initial_x,
                "initial_y_m": initial_y,
                "initial_yaw_rad": initial_yaw,
                "pre_push_x_m": pre_x,
                "pre_push_y_m": pre_y,
                "pre_push_yaw_rad": pre_yaw,
                "final_x_m": final_x,
                "final_y_m": final_y,
                "final_yaw_rad": final_yaw,
                "step_delta_local_x_m": local_dx,
                "step_delta_local_y_m": local_dy,
                "step_delta_yaw_rad": final_yaw - pre_yaw,
                "target_delta_x_m": target_dx,
                "target_delta_y_m": target_dy,
                "target_relative_yaw_rad": target_yaw,
                "target_distance_m": math.hypot(target_dx, target_dy),
                "target_key_4dp": f"{target_dx:.4f}|{target_dy:.4f}",
                "push_valid": int(push_valid),
                "prefix_valid": int(prefix_valid),
                "candidate_eligible": int(step_index >= 2 and prefix_valid),
                "quality_pass": result["quality_pass"],
                "simulation_unstable": result["simulation_unstable"],
                "contact_success": result["contact_success"],
                "stopped_by_threshold": result["stopped_by_threshold"],
                "num_contacts": result["num_contacts"],
                "settle_time_s": result["settle_time_s"],
            }
        )
        if not push_valid:
            break
    return rows


def completed_trajectory_ids(
    path: Path,
    base_seed: int,
    split_role: str,
) -> set[int]:
    """读取相同 random seed 下已完成写出的 trajectory IDs。"""

    if not path.exists():
        return set()
    rows = read_csv_rows(path)
    observed_seeds = {int(row["base_random_seed"]) for row in rows}
    if observed_seeds != {int(base_seed)}:
        raise RuntimeError(
            f"resume seed 与现有 candidate file 不一致: {sorted(observed_seeds)}"
        )
    observed_roles = {row["split_role"] for row in rows}
    if observed_roles != {split_role}:
        raise RuntimeError(
            f"resume role 与现有 candidate file 不一致: {sorted(observed_roles)}"
        )
    observed_cones = {row.get("friction_cone") for row in rows}
    observed_environments = {row.get("environment_xml") for row in rows}
    if observed_cones != {FRICTION_CONE} or observed_environments != {
        str(BASE_XML_PATH.resolve())
    }:
        raise RuntimeError("resume candidate file 不是当前 Elliptic HCR V2 环境数据")
    return {int(row["trajectory_id"]) for row in rows}


def summarise_target_candidates(
    output_path: Path,
    requested_trajectories: int,
    base_seed: int,
    worker_count: int,
    split_role: str,
) -> dict[str, Any]:
    """汇总长距离 target 原始候选数据。"""

    rows = read_csv_rows(output_path)
    trajectory_ids = {int(row["trajectory_id"]) for row in rows}
    eligible = [row for row in rows if int(row["candidate_eligible"]) == 1]
    invalid_ids = {
        int(row["trajectory_id"])
        for row in rows
        if int(row["push_valid"]) == 0
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "environment_xml": str(BASE_XML_PATH.resolve()),
        "split_role": split_role,
        "nominal_friction_sliding_mu": 0.40,
        "nominal_com_offset_x_m": 0.0,
        "nominal_com_offset_y_m": 0.0,
        "base_random_seed": base_seed,
        "requested_trajectories": requested_trajectories,
        "completed_trajectories": len(trajectory_ids),
        "trajectory_length": 5,
        "recorded_step_rows": len(rows),
        "eligible_candidate_rows": len(eligible),
        "unique_eligible_target_keys_4dp": len(
            {row["target_key_4dp"] for row in eligible}
        ),
        "invalid_trajectories": len(invalid_ids),
        "num_workers": worker_count,
        "output_path": str(output_path.resolve()),
    }


def collect_target_candidates(args: argparse.Namespace) -> dict[str, Any]:
    """采集 Validation 或 Test long-distance target 原始候选池。"""

    if args.num_trajectories <= 0:
        raise ValueError("num-trajectories 必须大于 0")
    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(args.data_root)
        / "target_candidates"
        / args.role
        / "candidate_steps.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        completed_trajectory_ids(output_path, args.seed, args.role)
        if args.resume
        else set()
    )
    trajectory_ids = [
        trajectory_id
        for trajectory_id in range(args.num_trajectories)
        if trajectory_id not in completed
    ]
    worker_count = min(max(1, int(args.num_workers)), max(1, len(trajectory_ids)))
    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_CANDIDATE_FIELDS)
        if mode == "w":
            writer.writeheader()
        if trajectory_ids:
            if worker_count == 1:
                initialise_target_worker(
                    str(ACTION_MANIFEST_PATH),
                    args.seed,
                    args.role,
                )
                iterator = map(process_target_trajectory, trajectory_ids)
                for index, rows in enumerate(iterator, start=1):
                    writer.writerows(rows)
                    print(
                        f"finished target trajectory {index}/{len(trajectory_ids)}"
                    )
            else:
                with mp.Pool(
                    processes=worker_count,
                    initializer=initialise_target_worker,
                    initargs=(str(ACTION_MANIFEST_PATH), args.seed, args.role),
                ) as pool:
                    for index, rows in enumerate(
                        pool.imap(process_target_trajectory, trajectory_ids),
                        start=1,
                    ):
                        writer.writerows(rows)
                        print(
                            f"finished target trajectory {index}/{len(trajectory_ids)}"
                        )

    summary = summarise_target_candidates(
        output_path,
        args.num_trajectories,
        args.seed,
        worker_count,
        args.role,
    )
    write_json(output_path.with_suffix(".summary.json"), summary)
    print_json(summary)
    return summary


def select_e2_history_conditions(
    conditions: list[dict[str, str]],
    scenario: str,
    data_role: str,
) -> list[dict[str, str]]:
    """选择 E2 Validation 或 Test hidden conditions。"""

    selected = [
        row
        for row in conditions
        if row["scenario"] == scenario
        and row["condition_role"] == data_role
        and (
            data_role == "test"
            or row["validation_partition"] == "model_selection"
        )
    ]
    selected.sort(key=lambda row: int(row["condition_index_within_role"]))
    expected = 16 if data_role == "validation" else 32
    if len(selected) != expected:
        raise RuntimeError(
            f"{scenario} E2 {data_role} condition 数量错误: {len(selected)}"
        )
    return selected


def load_history_targets(
    data_role: str,
    max_targets: int = 0,
) -> list[dict[str, str]]:
    """读取固定的 128 core 与 128 long-distance targets。"""

    if data_role == "validation":
        core_path = CORE_VALIDATION_TARGET_PATH
        long_path = LONG_VALIDATION_TARGET_PATH
    else:
        core_path = CORE_TEST_TARGET_PATH
        long_path = LONG_TEST_TARGET_PATH
    core = read_csv_rows(core_path)
    long_distance = read_csv_rows(long_path)
    if len(core) != 128 or len(long_distance) != 128:
        raise RuntimeError(
            f"E2 {data_role} target manifests 必须分别包含 128 个 targets"
        )
    targets = core + long_distance
    target_ids = [row["v2_target_id"] for row in targets]
    target_keys = [row["canonical_position_key"] for row in targets]
    if len(set(target_ids)) != 256 or len(set(target_keys)) != 256:
        raise RuntimeError(f"E2 {data_role} target IDs 或 position keys 不唯一")
    if max_targets > 0:
        targets = targets[:max_targets]
    return targets


def initialise_history_worker(
    action_manifest_path: str,
    behaviour_p1_paths_json: str,
) -> None:
    """为 continuous-history worker 加载 action_core 与各场景 nominal P1。"""

    global _HISTORY_ACTIONS
    global _HISTORY_NOMINAL_OUTCOMES
    global _HISTORY_BEHAVIOUR_P1_PATHS
    _HISTORY_ACTIONS = load_actions(Path(action_manifest_path))
    expected_action_ids = [row["v2_action_id"] for row in _HISTORY_ACTIONS]
    paths = json.loads(behaviour_p1_paths_json)
    _HISTORY_NOMINAL_OUTCOMES = {}
    _HISTORY_BEHAVIOUR_P1_PATHS = {}
    nominal_condition = {
        "hidden_u_friction": 0.0,
        "hidden_u_com_x": 0.0,
        "hidden_u_com_y": 0.0,
    }
    for scenario, path in paths.items():
        p1 = TensorOutcomeInterpolator.load(Path(path))
        if p1.scenario != scenario or list(p1.action_ids) != expected_action_ids:
            raise RuntimeError(
                f"{scenario} behaviour P1 与 scenario 或 action_core 不一致"
            )
        _HISTORY_NOMINAL_OUTCOMES[scenario] = p1.predict(nominal_condition)
        _HISTORY_BEHAVIOUR_P1_PATHS[scenario] = str(Path(path).resolve())


def read_pose(model, data, object_body_id: int) -> tuple[float, float, float]:
    """读取指定 MuJoCo state 中物体的 unwrapped planar pose。"""

    return (
        float(data.xpos[object_body_id][0]),
        float(data.xpos[object_body_id][1]),
        get_object_yaw_qpos(model, data),
    )


def tnpo_cost(position_error_m: float, yaw_error_rad: float) -> float:
    """计算 E2 history policy 使用的 primary TNPO cost。"""

    return (
        PRIMARY_TNPO_COST.position_weight
        * position_error_m
        / PRIMARY_TNPO_COST.position_tolerance_m
        + PRIMARY_TNPO_COST.yaw_weight
        * yaw_error_rad
        / PRIMARY_TNPO_COST.yaw_tolerance_rad
    )


def select_history_action(
    scenario: str,
    pre_x: float,
    pre_y: float,
    pre_yaw: float,
    target_world_x: float,
    target_world_y: float,
    target_world_yaw: float,
) -> dict[str, Any]:
    """使用 belief-independent nominal-condition P1 policy 选择 task action。"""

    nominal_outcomes = _HISTORY_NOMINAL_OUTCOMES[scenario]
    cosine = math.cos(pre_yaw)
    sine = math.sin(pre_yaw)
    local_dx = nominal_outcomes[:, 0].astype(np.float64)
    local_dy = nominal_outcomes[:, 1].astype(np.float64)
    predicted_x = pre_x + cosine * local_dx - sine * local_dy
    predicted_y = pre_y + sine * local_dx + cosine * local_dy
    predicted_yaw = pre_yaw + nominal_outcomes[:, 2]
    position_error = np.hypot(
        predicted_x - target_world_x,
        predicted_y - target_world_y,
    )
    yaw_error = np.abs(wrap_to_pi(predicted_yaw - target_world_yaw))
    costs = (
        PRIMARY_TNPO_COST.position_weight
        * position_error
        / PRIMARY_TNPO_COST.position_tolerance_m
        + PRIMARY_TNPO_COST.yaw_weight
        * yaw_error
        / PRIMARY_TNPO_COST.yaw_tolerance_rad
    )
    minimum = float(np.min(costs))
    tied_indices = np.flatnonzero(costs == minimum)
    selected_index = min(
        (int(index) for index in tied_indices),
        key=lambda index: _HISTORY_ACTIONS[index]["v2_action_id"],
    )
    return {
        "action_index": selected_index,
        "predicted_local_outcome": nominal_outcomes[selected_index].astype(
            np.float64
        ),
        "predicted_final_x": float(predicted_x[selected_index]),
        "predicted_final_y": float(predicted_y[selected_index]),
        "predicted_final_yaw": float(predicted_yaw[selected_index]),
        "predicted_position_error": float(position_error[selected_index]),
        "predicted_yaw_error": float(yaw_error[selected_index]),
        "predicted_tnpo_cost": float(costs[selected_index]),
    }


def summarise_history_rows(
    rows: list[dict[str, Any]],
    scenario: str,
    condition_id: str,
    output_path: Path,
    resumed: int,
) -> dict[str, Any]:
    """汇总一个 condition shard 的 episode lifecycle。"""

    terminal_rows = [row for row in rows if int(row["is_terminal_push"]) == 1]
    reasons = Counter(row["terminal_reason"] for row in terminal_rows)
    return {
        "scenario": scenario,
        "condition_id": condition_id,
        "path": str(output_path.resolve()),
        "episodes": len(terminal_rows),
        "step_rows": len(rows),
        "valid_observations": sum(int(row["valid_observation"]) for row in rows),
        "valid_episodes": sum(int(row["episode_valid"]) for row in terminal_rows),
        "invalid_episodes": int(reasons["invalid_push"]),
        "successful_episodes": int(reasons["success"]),
        "maximum_budget_episodes": int(reasons["maximum_push_budget"]),
        "resumed": resumed,
    }


def inspect_complete_history_shard(
    path: Path,
    expected_target_ids: set[str],
    maximum_pushes: int,
    data_role: str,
) -> bool:
    """判断一个 condition history shard 是否完整。"""

    if not path.exists():
        return False
    rows = read_csv_rows(path)
    if not rows:
        return False
    terminal_rows = [row for row in rows if int(row["is_terminal_push"]) == 1]
    terminal_target_ids = {row["target_id"] for row in terminal_rows}
    return (
        len(terminal_rows) == len(expected_target_ids)
        and terminal_target_ids == expected_target_ids
        and {row["protocol_version"] for row in rows} == {PROTOCOL_VERSION}
        and {row["friction_cone"] for row in rows} == {FRICTION_CONE}
        and {row["e2_data_role"] for row in rows} == {data_role}
        and {int(row["maximum_push_budget"]) for row in rows}
        == {maximum_pushes}
    )


def process_history_condition(task: dict[str, Any]) -> dict[str, Any]:
    """在一个 hidden condition 下采集全部 continuous target histories。"""

    condition = task["condition"]
    targets = task["targets"]
    maximum_pushes = int(task["maximum_pushes"])
    data_role = task["data_role"]
    output_path = Path(task["output_path"])
    expected_target_ids = {row["v2_target_id"] for row in targets}
    if bool(task["resume"]) and inspect_complete_history_shard(
        output_path,
        expected_target_ids,
        maximum_pushes,
        data_role,
    ):
        return summarise_history_rows(
            read_csv_rows(output_path),
            condition["scenario"],
            condition["condition_id"],
            output_path,
            resumed=1,
        )

    model, data = load_model(Path(task["xml_path"]))
    set_sliding_friction(model, float(condition["friction_sliding_mu"]))
    object_body_id = get_body_id(model)
    all_rows: list[dict[str, Any]] = []

    for episode_index, target in enumerate(targets):
        episode_id = (
            int(condition["condition_index_within_role"]) * 1000
            + episode_index
        )
        reset_input = build_rollout_input(
            _HISTORY_ACTIONS[0],
            condition,
            episode_id,
        )
        reset_state(model, data, reset_input)
        initial_x, initial_y, initial_yaw = read_pose(model, data, object_body_id)
        target_world_x = initial_x + float(target["target_delta_x_m"])
        target_world_y = initial_y + float(target["target_delta_y_m"])
        target_world_yaw = initial_yaw + float(target["target_yaw_offset_rad"])
        episode_key = (
            f"{condition['scenario']}|{condition['condition_id']}|"
            f"{target['v2_target_id']}"
        )
        episode_rows: list[dict[str, Any]] = []
        terminal_reason = "maximum_push_budget"

        for push_index in range(1, maximum_pushes + 1):
            pre_x, pre_y, pre_yaw = read_pose(model, data, object_body_id)
            policy = select_history_action(
                condition["scenario"],
                pre_x,
                pre_y,
                pre_yaw,
                target_world_x,
                target_world_y,
                target_world_yaw,
            )
            action_index = int(policy["action_index"])
            action = _HISTORY_ACTIONS[action_index]
            rollout_input = build_rollout_input(action, condition, episode_id)
            rollout_input["dataset_role"] = (
                f"hcr_v2_e2_history_{data_role}_{condition['scenario']}"
            )
            rollout_input["step_id"] = push_index
            rollout_input["group_id"] = episode_index
            result = run_physical_pusher_atomic_push(model, data, rollout_input)
            final_x, final_y, final_yaw = read_pose(model, data, object_body_id)
            observation = local_motion_observation(
                np.asarray([pre_x, pre_y]),
                pre_yaw,
                np.asarray([final_x, final_y]),
                final_yaw,
            )
            valid_observation = (
                int(result["quality_pass"]) == 1
                and int(result["simulation_unstable"]) == 0
                and int(result["contact_success"]) == 1
                and int(result["stopped_by_threshold"]) == 1
            )
            actual_position_error = math.hypot(
                final_x - target_world_x,
                final_y - target_world_y,
            )
            actual_yaw_error = abs(
                float(wrap_to_pi(final_yaw - target_world_yaw))
            )
            success_after_push = (
                valid_observation
                and actual_position_error
                <= PRIMARY_TNPO_COST.position_tolerance_m
                and actual_yaw_error <= PRIMARY_TNPO_COST.yaw_tolerance_rad
            )
            if not valid_observation:
                terminal_reason = "invalid_push"
            elif success_after_push:
                terminal_reason = "success"
            elif push_index == maximum_pushes:
                terminal_reason = "maximum_push_budget"

            predicted_local = policy["predicted_local_outcome"]
            episode_rows.append(
                {
                    "experiment_id": "E2",
                    "protocol_version": PROTOCOL_VERSION,
                    "friction_cone": FRICTION_CONE,
                    "environment_xml": str(Path(task["xml_path"]).resolve()),
                    "e2_data_role": data_role,
                    "scenario": condition["scenario"],
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
                    "target_yaw_offset_rad": target["target_yaw_offset_rad"],
                    "target_radial_distance_m": target["radial_distance_m"],
                    "episode_key": episode_key,
                    "episode_index_within_condition": episode_index,
                    "push_index": push_index,
                    "v2_action_id": action["v2_action_id"],
                    "candidate_id": action["candidate_id"],
                    "action_param_index": action["action_param_index"],
                    "pre_push_x_m": pre_x,
                    "pre_push_y_m": pre_y,
                    "pre_push_yaw_rad": pre_yaw,
                    "final_x_m": final_x,
                    "final_y_m": final_y,
                    "final_yaw_rad": final_yaw,
                    "observation_local_delta_x_m": observation[0],
                    "observation_local_delta_y_m": observation[1],
                    "observation_delta_yaw_rad": observation[2],
                    "target_world_x_m": target_world_x,
                    "target_world_y_m": target_world_y,
                    "target_world_yaw_rad": target_world_yaw,
                    "predicted_local_delta_x_m": predicted_local[0],
                    "predicted_local_delta_y_m": predicted_local[1],
                    "predicted_delta_yaw_rad": predicted_local[2],
                    "predicted_final_x_m": policy["predicted_final_x"],
                    "predicted_final_y_m": policy["predicted_final_y"],
                    "predicted_final_yaw_rad": policy["predicted_final_yaw"],
                    "predicted_position_error_m": policy[
                        "predicted_position_error"
                    ],
                    "predicted_yaw_error_rad": policy["predicted_yaw_error"],
                    "predicted_tnpo_cost": policy["predicted_tnpo_cost"],
                    "actual_position_error_m": actual_position_error,
                    "actual_yaw_error_rad": actual_yaw_error,
                    "actual_tnpo_cost": tnpo_cost(
                        actual_position_error,
                        actual_yaw_error,
                    ),
                    "success_after_push": int(success_after_push),
                    "valid_observation": int(valid_observation),
                    "valid_update_index": push_index if valid_observation else "",
                    "quality_pass": result["quality_pass"],
                    "simulation_unstable": result["simulation_unstable"],
                    "contact_success": result["contact_success"],
                    "stopped_by_threshold": result["stopped_by_threshold"],
                    "num_contacts": result["num_contacts"],
                    "settle_time_s": result["settle_time_s"],
                    "maximum_push_budget": maximum_pushes,
                    "behaviour_policy": "belief_independent_nominal_p1_tnpo",
                    "behaviour_p1_artifact": _HISTORY_BEHAVIOUR_P1_PATHS[
                        condition["scenario"]
                    ],
                }
            )
            if terminal_reason in {"invalid_push", "success"}:
                break

        episode_valid = terminal_reason != "invalid_push"
        episode_success = terminal_reason == "success"
        for row_index, row in enumerate(episode_rows):
            row["episode_valid"] = int(episode_valid)
            row["episode_success"] = int(episode_success)
            row["terminal_reason"] = terminal_reason
            row["terminal_push_count"] = len(episode_rows)
            row["is_terminal_push"] = int(row_index == len(episode_rows) - 1)
        all_rows.extend(episode_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    return summarise_history_rows(
        all_rows,
        condition["scenario"],
        condition["condition_id"],
        output_path,
        resumed=0,
    )


def collect_histories(args: argparse.Namespace) -> dict[str, Any]:
    """采集 E2 fixed-condition continuous Validation 或 Test histories。"""

    if args.maximum_pushes <= 0:
        raise ValueError("maximum-pushes 必须大于 0")
    data_role = args.role
    selected_scenarios = select_scenarios(args.scenario)
    targets = load_history_targets(data_role, args.max_targets)
    all_conditions = load_conditions()
    data_root = Path(args.data_root)
    p1_metadata = load_e1_p1_metadata(selected_scenarios)
    behaviour_p1_paths = {
        scenario: p1_metadata[scenario]["artifact_path"]
        for scenario in selected_scenarios
    }
    behaviour_p1_paths_json = json.dumps(behaviour_p1_paths)

    tasks: list[dict[str, Any]] = []
    for scenario in selected_scenarios:
        conditions = select_e2_history_conditions(
            all_conditions,
            scenario,
            data_role,
        )
        if args.max_conditions > 0:
            conditions = conditions[: args.max_conditions]
        generated_dir = (
            data_root / "generated_xml" / "histories" / data_role / scenario
        )
        xml_by_com = prepare_environment_xmls(conditions, generated_dir)
        for condition in conditions:
            com_key = (
                round(float(condition["com_offset_x_m"]), 9),
                round(float(condition["com_offset_y_m"]), 9),
            )
            tasks.append(
                {
                    "condition": condition,
                    "targets": targets,
                    "data_role": data_role,
                    "maximum_pushes": args.maximum_pushes,
                    "xml_path": str(xml_by_com[com_key]),
                    "output_path": str(
                        data_root
                        / "histories"
                        / data_role
                        / scenario
                        / f"{condition['condition_id']}.csv"
                    ),
                    "resume": args.resume,
                }
            )

    worker_count = min(max(1, int(args.num_workers)), len(tasks))
    results: list[dict[str, Any]] = []
    if worker_count == 1:
        initialise_history_worker(
            str(ACTION_MANIFEST_PATH),
            behaviour_p1_paths_json,
        )
        for index, task in enumerate(tasks, start=1):
            results.append(process_history_condition(task))
            print(
                f"finished {data_role} history condition {index}/{len(tasks)}"
            )
    else:
        with mp.Pool(
            processes=worker_count,
            initializer=initialise_history_worker,
            initargs=(str(ACTION_MANIFEST_PATH), behaviour_p1_paths_json),
            maxtasksperchild=2,
        ) as pool:
            for index, result in enumerate(
                pool.imap_unordered(process_history_condition, tasks),
                start=1,
            ):
                results.append(result)
                print(
                    f"finished {data_role} history condition {index}/{len(tasks)}"
                )

    results.sort(key=lambda row: (row["scenario"], row["condition_id"]))
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "data_role": data_role,
        "scenario": args.scenario,
        "smoke": bool(
            args.max_conditions > 0
            or args.max_targets > 0
            or args.maximum_pushes
            != (
                PROVISIONAL_VALIDATION_PUSH_CEILING
                if data_role == "validation"
                else FINAL_TEST_MAXIMUM_PUSH_BUDGET
            )
        ),
        "conditions": len(tasks),
        "targets_per_condition": len(targets),
        "core_targets_per_condition": sum(
            target["target_group"] == "core" for target in targets
        ),
        "long_distance_targets_per_condition": sum(
            target["target_group"] == "long_distance" for target in targets
        ),
        "episodes": sum(int(row["episodes"]) for row in results),
        "step_rows": sum(int(row["step_rows"]) for row in results),
        "valid_observations": sum(
            int(row["valid_observations"]) for row in results
        ),
        "valid_episodes": sum(int(row["valid_episodes"]) for row in results),
        "invalid_episodes": sum(
            int(row["invalid_episodes"]) for row in results
        ),
        "successful_episodes": sum(
            int(row["successful_episodes"]) for row in results
        ),
        "maximum_budget_episodes": sum(
            int(row["maximum_budget_episodes"]) for row in results
        ),
        "maximum_push_budget": args.maximum_pushes,
        "behaviour_policy": "belief_independent_nominal_p1_tnpo",
        "behaviour_nominal_friction": 0.40,
        "behaviour_nominal_com_x_m": 0.0,
        "behaviour_nominal_com_y_m": 0.0,
        "behaviour_p1_artifacts": behaviour_p1_paths,
        "core_target_manifest": str(
            (
                CORE_VALIDATION_TARGET_PATH
                if data_role == "validation"
                else CORE_TEST_TARGET_PATH
            ).resolve()
        ),
        "long_distance_target_manifest": str(
            (
                LONG_VALIDATION_TARGET_PATH
                if data_role == "validation"
                else LONG_TEST_TARGET_PATH
            ).resolve()
        ),
        "num_workers": worker_count,
        "resumed_conditions": sum(int(row["resumed"]) for row in results),
        "condition_results": results,
    }
    write_json(
        data_root
        / "histories"
        / data_role
        / f"collection_summary_{args.scenario}.json",
        summary,
    )
    print_json(summary)
    return summary


def print_json(payload: dict[str, Any]) -> None:
    """把摘要以 UTF-8 友好的 JSON 形式打印到终端。"""

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """构建 E2 数据采集命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    training = subparsers.add_parser(
        "training-outcomes",
        help="采集 E2 Training residual-statistics outcomes",
    )
    training.add_argument(
        "--scenario",
        choices=("all", *SCENARIOS),
        default="all",
    )
    training.add_argument("--num-workers", type=int, default=8)
    training.add_argument("--resume", action="store_true")
    training.add_argument("--max-actions", type=int, default=0)
    training.add_argument("--data-root", type=Path, default=E2_DATA_ROOT)
    training.set_defaults(handler=collect_training_outcomes)

    targets = subparsers.add_parser(
        "target-candidates",
        help="采集 Validation 或 Test long-distance target 原始候选池",
    )
    targets.add_argument(
        "--role",
        choices=("validation", "test"),
        default="validation",
    )
    targets.add_argument("--num-trajectories", type=int, required=True)
    targets.add_argument("--seed", type=int, required=True)
    targets.add_argument("--num-workers", type=int, default=8)
    targets.add_argument("--resume", action="store_true")
    targets.add_argument("--data-root", type=Path, default=E2_DATA_ROOT)
    targets.add_argument("--output-path", type=Path)
    targets.set_defaults(handler=collect_target_candidates)

    histories = subparsers.add_parser(
        "validation-histories",
        help="采集 E2 continuous Validation action-observation histories",
    )
    histories.add_argument(
        "--scenario",
        choices=("all", *SCENARIOS),
        default="all",
    )
    histories.add_argument("--num-workers", type=int, default=8)
    histories.add_argument("--resume", action="store_true")
    histories.add_argument(
        "--maximum-pushes",
        type=int,
        default=PROVISIONAL_VALIDATION_PUSH_CEILING,
    )
    histories.add_argument("--max-conditions", type=int, default=0)
    histories.add_argument("--max-targets", type=int, default=0)
    histories.add_argument("--data-root", type=Path, default=E2_DATA_ROOT)
    histories.set_defaults(handler=collect_histories, role="validation")

    test_histories = subparsers.add_parser(
        "test-histories",
        help="采集 E2 continuous Test action-observation histories",
    )
    test_histories.add_argument(
        "--scenario",
        choices=("all", *SCENARIOS),
        default="all",
    )
    test_histories.add_argument("--num-workers", type=int, default=8)
    test_histories.add_argument("--resume", action="store_true")
    test_histories.add_argument(
        "--maximum-pushes",
        type=int,
        default=FINAL_TEST_MAXIMUM_PUSH_BUDGET,
    )
    test_histories.add_argument("--max-conditions", type=int, default=0)
    test_histories.add_argument("--max-targets", type=int, default=0)
    test_histories.add_argument("--data-root", type=Path, default=E2_DATA_ROOT)
    test_histories.set_defaults(handler=collect_histories, role="test")
    return parser


def main() -> None:
    """解析参数并执行对应的数据采集任务。"""

    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
