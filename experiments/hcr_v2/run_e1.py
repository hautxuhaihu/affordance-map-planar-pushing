"""HCR V2 E1 的统一实验入口。

该入口提供以下子命令：

- plan：只读取 manifests 并打印实验规模；
- smoke：每个场景执行一个 condition 和少量 actions；
- collect：采集指定场景与数据角色的完整 condition-action outcomes；
- prepare-p1：由 training-anchor outcomes 建立 tensor interpolation artifact；
- train-p2：训练 condition-conditioned MLP；
- evaluate：执行 oracle、outcome prediction 与 TNPO-cost action selection 评估。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from push_core.forward_model.models import OutcomeMLP
from push_core.forward_model.normaliser import StandardNormaliser
from push_core.forward_model.predictor import Level1OutcomePredictor
from push_core.forward_model.trainer_v2 import (
    make_target_scale,
    train_model_with_early_stopping,
)
from push_core.hcr_v2.e1 import (
    ACTION_FEATURE_FIELDS,
    NEAR_OPTIMAL_EPSILON,
    OUTCOME_FIELDS,
    PRIMARY_TNPO_COST,
    SCENARIO_ACTIVE_COORDINATES,
    SENSITIVITY_TNPO_COST,
    ConditionedOutcomePredictor,
    TensorOutcomeInterpolator,
    cost_matrix,
    evaluate_selector,
    iter_csv_rows,
    outcome_error_summary,
    paired_two_way_bootstrap_mean_ci,
    pose_error_matrices,
    read_csv_rows,
    summary_statistics,
)
from push_core.project_paths import HCR_V2_DATA_DIR, HCR_V2_RESULTS_DIR
from push_core.schema import level1_schema as schema
from push_core.simulation.physical_pusher_rollout import (
    load_model,
    run_physical_pusher_rollout,
)


MANIFEST_DIR = REPOSITORY_ROOT / "manifests" / "hcr_v2"
ACTION_MANIFEST_PATH = MANIFEST_DIR / "hcr_v2_action_core_manifest_v1.csv"
CONDITION_MANIFEST_PATH = MANIFEST_DIR / "hcr_v2_hidden_condition_manifest_v1.csv"
TARGET_MANIFEST_PATH = MANIFEST_DIR / "hcr_v2_core_target_manifest_v1.csv"
BASE_XML_PATH = PROJECT_ROOT / "assets" / "xml" / "msc_rod_pusher_box_hcr_v2.xml"

E1_DATA_ROOT = HCR_V2_DATA_DIR / "e1"
E1_RESULTS_ROOT = HCR_V2_RESULTS_DIR / "e1"
P2_HIDDEN_DIM = 128

SCENARIOS = ("friction", "com", "joint")
FORMAL_ROLE_CONDITION_COUNTS = {
    "training": {"friction": 5, "com": 25, "joint": 125},
    "validation": {"friction": 16, "com": 16, "joint": 16},
    "test": {"friction": 32, "com": 32, "joint": 32},
}

ROLLOUT_FIELDS = [
    "experiment_id",
    "protocol_version",
    "friction_cone",
    "scenario",
    "hidden_parameter_dimension",
    "condition_role",
    "validation_partition",
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

PROTOCOL_VERSION = "hcr_v2_e1_v2"
FRICTION_CONE = "elliptic"
ENVIRONMENT_METADATA = {
    "friction_cone": FRICTION_CONE,
    "environment_xml": str(BASE_XML_PATH.resolve()),
}

_WORKER_ACTION_ROWS: list[dict[str, str]] = []


def make_json_compatible(value: Any) -> Any:
    """将非有限浮点值转换为严格 JSON 支持的空值。"""

    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        return {key: make_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_compatible(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 写出符合标准的可读 JSON。"""

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
    """以稳定字段顺序写出 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_scenarios(value: str) -> tuple[str, ...]:
    """解析单一场景或 all。"""

    if value == "all":
        return SCENARIOS
    if value not in SCENARIOS:
        raise ValueError(f"未知 scenario: {value}")
    return (value,)


def load_actions(path: Path = ACTION_MANIFEST_PATH) -> list[dict[str, str]]:
    """读取并检查共享 action_core。"""

    rows = read_csv_rows(path)
    if len(rows) != 4_536:
        raise RuntimeError(f"action_core 数量错误: {len(rows)}")
    action_ids = [row["v2_action_id"] for row in rows]
    candidate_ids = [row["candidate_id"] for row in rows]
    if len(set(action_ids)) != len(rows) or len(set(candidate_ids)) != len(rows):
        raise RuntimeError("action_core IDs 不唯一")
    return rows


def load_conditions(path: Path = CONDITION_MANIFEST_PATH) -> list[dict[str, str]]:
    """读取 hidden-condition manifest。"""

    rows = read_csv_rows(path)
    if len(rows) != 347:
        raise RuntimeError(f"hidden-condition 数量错误: {len(rows)}")
    if len({row["condition_id"] for row in rows}) != len(rows):
        raise RuntimeError("hidden-condition IDs 不唯一")
    return rows


def select_conditions(
    rows: list[dict[str, str]],
    scenario: str,
    role: str,
    validation_partition: str = "model_selection",
) -> list[dict[str, str]]:
    """按场景、数据角色和 validation partition 选择 conditions。"""

    selected = [
        row
        for row in rows
        if row["scenario"] == scenario and row["condition_role"] == role
    ]
    if role == "validation" and validation_partition != "all":
        selected = [
            row
            for row in selected
            if row["validation_partition"] == validation_partition
        ]
    selected.sort(key=lambda row: int(row["condition_index_within_role"]))
    return selected


def load_targets(role: str, path: Path = TARGET_MANIFEST_PATH) -> list[dict[str, str]]:
    """读取 validation 或 test core targets。"""

    rows = [row for row in read_csv_rows(path) if row["split_role"] == role]
    expected = 408 if role in {"validation", "test"} else 3_268
    if len(rows) != expected:
        raise RuntimeError(f"{role} target 数量错误: {len(rows)}")
    return rows


def plan_summary() -> dict[str, Any]:
    """读取 manifests 并生成不写文件的 E1 规模摘要。"""

    actions = load_actions()
    conditions = load_conditions()
    roles: dict[str, Any] = {}
    for role in ("training", "validation", "test"):
        role_rows: dict[str, Any] = {}
        for scenario in SCENARIOS:
            partition = "model_selection" if role == "validation" else "all"
            selected = select_conditions(conditions, scenario, role, partition)
            expected = FORMAL_ROLE_CONDITION_COUNTS[role][scenario]
            if len(selected) != expected:
                raise RuntimeError(
                    f"{scenario}/{role} condition 数量错误: "
                    f"expected={expected}, observed={len(selected)}"
                )
            role_rows[scenario] = {
                "conditions": len(selected),
                "actions": len(actions),
                "rollouts": len(selected) * len(actions),
            }
        role_rows["total_rollouts"] = sum(
            row["rollouts"] for row in role_rows.values() if isinstance(row, dict)
        )
        roles[role] = role_rows
    return {
        "protocol_version": PROTOCOL_VERSION,
        **ENVIRONMENT_METADATA,
        "action_count": len(actions),
        "validation_target_count": len(load_targets("validation")),
        "test_target_count": len(load_targets("test")),
        "roles": roles,
        "total_rollouts": sum(role["total_rollouts"] for role in roles.values()),
        "primary_position_tolerance_m": PRIMARY_TNPO_COST.position_tolerance_m,
        "primary_yaw_tolerance_deg": math.degrees(
            PRIMARY_TNPO_COST.yaw_tolerance_rad
        ),
        "sensitivity_yaw_tolerance_deg": math.degrees(
            SENSITIVITY_TNPO_COST.yaw_tolerance_rad
        ),
        "near_optimal_epsilon": NEAR_OPTIMAL_EPSILON,
    }


def named_id(model, object_type, name: str) -> int:
    """按名称读取 MuJoCo object id。"""

    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return int(object_id)


def format_mjcf_values(values: Iterable[float]) -> str:
    """把浮点数组格式化为 MJCF 属性。"""

    return " ".join(f"{float(value):.12g}" for value in values)


def generate_com_xml(
    base_xml_path: Path,
    output_xml_path: Path,
    com_offset_x_m: float,
    com_offset_y_m: float,
) -> None:
    """基于 HCR V2 plane XML 生成只改变 object inertial position 的环境。"""

    base_model, _ = load_model(base_xml_path)
    body_id = named_id(base_model, mujoco.mjtObj.mjOBJ_BODY, "object")
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    object_body = root.find(".//body[@name='object']")
    if object_body is None:
        raise RuntimeError("HCR V2 XML 中未找到 object body")
    object_geom = object_body.find("geom[@name='object_geom']")
    if object_geom is None:
        raise RuntimeError("HCR V2 XML 中未找到 object_geom")

    for inertial in list(object_body.findall("inertial")):
        object_body.remove(inertial)
    inertial = ET.Element(
        "inertial",
        {
            "pos": format_mjcf_values([com_offset_x_m, com_offset_y_m, 0.0]),
            "quat": format_mjcf_values(base_model.body_iquat[body_id]),
            "mass": f"{float(base_model.body_mass[body_id]):.12g}",
            "diaginertia": format_mjcf_values(base_model.body_inertia[body_id]),
        },
    )
    children = list(object_body)
    object_body.insert(children.index(object_geom), inertial)
    object_geom.attrib.pop("mass", None)
    object_geom.attrib.pop("density", None)
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)

    model, _ = load_model(output_xml_path)
    table_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    object_body_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if int(model.geom_type[table_id]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
        raise RuntimeError("生成环境的 table 不再是 plane")
    if not np.allclose(
        model.body_ipos[object_body_id, :2],
        [com_offset_x_m, com_offset_y_m],
        atol=1e-12,
    ):
        raise RuntimeError("生成环境的 COM offset 与请求值不一致")


def com_xml_name(com_x_m: float, com_y_m: float) -> str:
    """构造稳定的 COM XML 文件名。"""

    def token(value: float) -> str:
        sign = "p" if value >= 0.0 else "m"
        return sign + f"{abs(value) * 1000.0:.4f}".replace(".", "p")

    return f"com_x_{token(com_x_m)}mm_y_{token(com_y_m)}mm.xml"


def prepare_environment_xmls(
    conditions: list[dict[str, str]],
    generated_xml_dir: Path,
) -> dict[tuple[float, float], Path]:
    """为当前 collection 的 unique COM pairs 准备 XML。"""

    paths: dict[tuple[float, float], Path] = {}
    for row in conditions:
        com_x = float(row["com_offset_x_m"])
        com_y = float(row["com_offset_y_m"])
        key = (round(com_x, 9), round(com_y, 9))
        if key in paths:
            continue
        if abs(com_x) <= 1e-15 and abs(com_y) <= 1e-15:
            paths[key] = BASE_XML_PATH
            continue
        xml_path = generated_xml_dir / com_xml_name(com_x, com_y)
        generate_com_xml(BASE_XML_PATH, xml_path, com_x, com_y)
        paths[key] = xml_path
    return paths


def build_rollout_input(
    action_row: dict[str, str],
    condition_row: dict[str, str],
    episode_id: int,
) -> dict[str, Any]:
    """把 V2 action 和 condition 转换为 physical-pusher rollout row。"""

    ramp_up = float(action_row["ramp_up_s"])
    hold = float(action_row["hold_s"])
    row: dict[str, Any] = dict(schema.LEVEL1_FIXED_VALUES)
    row.update(
        {
            "dataset_role": (
                f"hcr_v2_e1_{condition_row['scenario']}_"
                f"{condition_row['condition_role']}"
            ),
            "episode_id": episode_id,
            "step_id": 0,
            "group_id": 0,
            "candidate_id": int(action_row["candidate_id"]),
            "contact_region_id": int(action_row["contact_region_id"]),
            "goal_delta_local_x": 0.0,
            "goal_delta_local_y": 0.0,
            "goal_yaw": 0.0,
            "surface_id": int(action_row["surface_id"]),
            "contact_region_row": int(action_row["contact_region_row"]),
            "contact_region_col": int(action_row["contact_region_col"]),
            "contact_point_local_x": float(action_row["contact_point_local_x"]),
            "contact_point_local_y": float(action_row["contact_point_local_y"]),
            "contact_point_local_z": 0.0,
            "contact_normal_local_x": float(action_row["contact_normal_local_x"]),
            "contact_normal_local_y": float(action_row["contact_normal_local_y"]),
            "contact_normal_local_z": 0.0,
            "force_angle_relative_to_normal_deg": float(
                action_row["force_angle_relative_to_normal_deg"]
            ),
            "force_direction_local_x": float(action_row["force_direction_local_x"]),
            "force_direction_local_y": float(action_row["force_direction_local_y"]),
            "force_direction_local_z": 0.0,
            "commanded_force_N": float(action_row["commanded_force_N"]),
            "ramp_up_s": ramp_up,
            "hold_s": hold,
            "ramp_down_s": float(action_row["ramp_down_s"]),
            "command_duration_s": ramp_up + hold,
            "hidden_com_offset_x": float(condition_row["com_offset_x_m"]),
            "hidden_com_offset_y": float(condition_row["com_offset_y_m"]),
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


def set_sliding_friction(model, friction_mu: float) -> None:
    """同步设置 table 与 object geom 的 sliding friction。"""

    table_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    object_id = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    model.geom_friction[table_id, 0] = float(friction_mu)
    model.geom_friction[object_id, 0] = float(friction_mu)


def initialise_collection_worker(action_manifest_path: str, max_actions: int) -> None:
    """在 collection worker 中加载共享 actions。"""

    global _WORKER_ACTION_ROWS
    rows = load_actions(Path(action_manifest_path))
    _WORKER_ACTION_ROWS = rows if max_actions <= 0 else rows[:max_actions]


def inspect_complete_shard(path: Path, expected_action_ids: set[str]) -> bool:
    """检查已存在 shard 是否包含完整且唯一的 action IDs。"""

    if not path.exists():
        return False
    rows = read_csv_rows(path)
    observed = {row["v2_action_id"] for row in rows}
    return len(rows) == len(expected_action_ids) and observed == expected_action_ids


def process_condition_task(task: dict[str, Any]) -> dict[str, Any]:
    """在一个 hidden condition 下采集完整 action outcomes。"""

    condition = task["condition"]
    output_path = Path(task["output_path"])
    expected_action_ids = {row["v2_action_id"] for row in _WORKER_ACTION_ROWS}
    if bool(task["resume"]) and inspect_complete_shard(output_path, expected_action_ids):
        existing_rows = read_csv_rows(output_path)
        return {
            "scenario": condition["scenario"],
            "condition_id": condition["condition_id"],
            "path": str(output_path.resolve()),
            "rollouts": len(expected_action_ids),
            "quality_pass_count": sum(
                int(row["quality_pass"]) for row in existing_rows
            ),
            "simulation_unstable_count": sum(
                int(row["simulation_unstable"]) for row in existing_rows
            ),
            "resumed": 1,
        }

    model, data = load_model(Path(task["xml_path"]))
    set_sliding_friction(model, float(condition["friction_sliding_mu"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality_count = 0
    unstable_count = 0
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROLLOUT_FIELDS)
        writer.writeheader()
        for action_index, action_row in enumerate(_WORKER_ACTION_ROWS):
            episode_id = (
                int(condition["condition_index_within_role"]) * 10_000
                + action_index
            )
            rollout_input = build_rollout_input(action_row, condition, episode_id)
            result = run_physical_pusher_rollout(
                model,
                data,
                rollout_input,
                validate_schema=False,
            )
            output_row = {
                "experiment_id": "E1",
                "protocol_version": PROTOCOL_VERSION,
                "friction_cone": FRICTION_CONE,
                "scenario": condition["scenario"],
                "hidden_parameter_dimension": condition["hidden_parameter_dimension"],
                "condition_role": condition["condition_role"],
                "validation_partition": condition["validation_partition"],
                "condition_id": condition["condition_id"],
                "condition_index_within_role": condition[
                    "condition_index_within_role"
                ],
                "v2_action_id": action_row["v2_action_id"],
                "candidate_id": action_row["candidate_id"],
                "action_param_index": action_row["action_param_index"],
                "contact_region_id": action_row["contact_region_id"],
                "surface_id": action_row["surface_id"],
                "contact_region_row": action_row["contact_region_row"],
                "contact_region_col": action_row["contact_region_col"],
                "force_angle_relative_to_normal_deg": action_row[
                    "force_angle_relative_to_normal_deg"
                ],
                **{field: action_row[field] for field in ACTION_FEATURE_FIELDS},
                "execution_duration_s": action_row["execution_duration_s"],
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
        "rollouts": len(_WORKER_ACTION_ROWS),
        "quality_pass_count": quality_count,
        "simulation_unstable_count": unstable_count,
        "resumed": 0,
    }


def collect_outcomes(
    scenario_value: str,
    role: str,
    validation_partition: str,
    num_workers: int,
    resume: bool,
    max_actions: int,
    smoke: bool,
) -> dict[str, Any]:
    """采集 E1 target-independent condition-action outcomes。"""

    actions = load_actions()
    if max_actions > 0:
        actions = actions[:max_actions]
    all_conditions = load_conditions()
    scenarios = select_scenarios(scenario_value)
    root = E1_DATA_ROOT / "smoke" if smoke else E1_DATA_ROOT
    tasks: list[dict[str, Any]] = []
    for scenario in scenarios:
        selected = select_conditions(
            all_conditions,
            scenario,
            role,
            validation_partition,
        )
        if smoke:
            selected = selected[:1]
        generated_dir = root / "generated_xml" / scenario / role
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
                        root
                        / "outcomes"
                        / scenario
                        / role
                        / f"{condition['condition_id']}.csv"
                    ),
                    "resume": resume,
                }
            )

    worker_count = min(max(1, int(num_workers)), max(1, len(tasks)))
    results: list[dict[str, Any]] = []
    if worker_count == 1:
        initialise_collection_worker(str(ACTION_MANIFEST_PATH), len(actions))
        for index, task in enumerate(tasks, start=1):
            results.append(process_condition_task(task))
            print(f"finished condition {index}/{len(tasks)}")
    else:
        with mp.Pool(
            processes=worker_count,
            initializer=initialise_collection_worker,
            initargs=(str(ACTION_MANIFEST_PATH), len(actions)),
            maxtasksperchild=4,
        ) as pool:
            for index, result in enumerate(
                pool.imap_unordered(process_condition_task, tasks), start=1
            ):
                results.append(result)
                print(f"finished condition {index}/{len(tasks)}")
    results.sort(key=lambda row: (row["scenario"], row["condition_id"]))
    total_rollouts = sum(int(row["rollouts"]) for row in results)
    summary_validation_partition = (
        validation_partition if role == "validation" else "not_applicable"
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        **ENVIRONMENT_METADATA,
        "smoke": smoke,
        "scenario": scenario_value,
        "role": role,
        "validation_partition": summary_validation_partition,
        "conditions": len(tasks),
        "actions_per_condition": len(actions),
        "rollouts": total_rollouts,
        "quality_pass_count": sum(int(row["quality_pass_count"]) for row in results),
        "simulation_unstable_count": sum(
            int(row["simulation_unstable_count"]) for row in results
        ),
        "resumed_conditions": sum(int(row["resumed"]) for row in results),
        "num_workers": worker_count,
        "condition_results": results,
    }
    summary["passed"] = bool(
        total_rollouts == len(tasks) * len(actions)
        and summary["simulation_unstable_count"] == 0
    )
    summary_path = root / f"e1_{scenario_value}_{role}_collection_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


class ArrayOutcomeDataset(torch.utils.data.Dataset):
    """由 NumPy arrays 构造 P2 训练数据集。"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])


def outcome_files(data_root: Path, scenario: str, role: str) -> list[Path]:
    """返回指定场景和角色的 outcome shards。"""

    paths = sorted((data_root / "outcomes" / scenario / role).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"未找到 {scenario}/{role} outcome shards")
    return paths


def load_p2_arrays(
    paths: list[Path],
    scenario: str,
    validation_partition: str | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """加载 P2 features 与 labels，并跳过无效 rollouts。"""

    input_fields = ACTION_FEATURE_FIELDS + SCENARIO_ACTIVE_COORDINATES[scenario]
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    excluded_unstable_count = 0
    for path in paths:
        file_x: list[list[float]] = []
        file_y: list[list[float]] = []
        for row in iter_csv_rows([path]):
            if (
                validation_partition is not None
                and row["validation_partition"] != validation_partition
            ):
                continue
            if int(row["simulation_unstable"]) != 0:
                excluded_unstable_count += 1
                continue
            file_x.append([float(row[field]) for field in input_fields])
            file_y.append([float(row[field]) for field in OUTCOME_FIELDS])
        if file_x:
            x_parts.append(np.asarray(file_x, dtype=np.float32))
            y_parts.append(np.asarray(file_y, dtype=np.float32))
    if not x_parts:
        raise RuntimeError(f"{scenario} 没有可用于 P2 的稳定 outcome rows")
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        excluded_unstable_count,
    )


def set_seed(seed: int) -> None:
    """设置 P2 训练使用的随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_p2(
    scenario: str,
    data_root: Path,
    results_root: Path,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, Any]:
    """训练一个场景的 P2 condition-conditioned MLP。"""

    set_seed(seed)
    train_paths = outcome_files(data_root, scenario, "training")
    validation_paths = outcome_files(data_root, scenario, "validation")
    train_x_raw, train_y, train_excluded_unstable = load_p2_arrays(
        train_paths, scenario
    )
    val_x_raw, val_y, val_excluded_unstable = load_p2_arrays(
        validation_paths,
        scenario,
        validation_partition="model_selection",
    )
    normaliser = StandardNormaliser.fit(train_x_raw)
    train_x = normaliser.transform(train_x_raw).astype(np.float32)
    val_x = normaliser.transform(val_x_raw).astype(np.float32)
    train_dataset = ArrayOutcomeDataset(train_x, train_y)
    val_dataset = ArrayOutcomeDataset(val_x, val_y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    matmul_precision = torch.get_float32_matmul_precision()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_batch_size = max(batch_size, 2_048)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    target_scale = make_target_scale(train_y)
    model = OutcomeMLP(input_dim=train_x.shape[1], hidden_dim=P2_HIDDEN_DIM)
    output_dir = results_root / "p2" / scenario
    checkpoint_path = output_dir / "conditioned_outcome_mlp.pt"
    history = train_model_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs,
        lr=1e-3,
        weight_decay=1e-5,
        device=device,
        ckpt_path=checkpoint_path,
        target_scale=target_scale,
        patience=10,
        min_delta=1e-6,
        model_name=f"hcr_v2_e1_p2_{scenario}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    normaliser_payload = normaliser.to_json_dict()
    normaliser_payload.update(
        {
            "scenario": scenario,
            "input_fields": list(
                ACTION_FEATURE_FIELDS + SCENARIO_ACTIVE_COORDINATES[scenario]
            ),
            "target_fields": list(OUTCOME_FIELDS),
            "target_scale": target_scale.tolist(),
            "hidden_dim": P2_HIDDEN_DIM,
            "hidden_layers": 2,
            "float32_matmul_precision": matmul_precision,
            "training_batch_size": batch_size,
            "validation_batch_size": validation_batch_size,
            "protocol_version": PROTOCOL_VERSION,
            **ENVIRONMENT_METADATA,
        }
    )
    write_json(output_dir / "normaliser.json", normaliser_payload)
    history.update(
        {
            "training_batch_size": batch_size,
            "validation_batch_size": validation_batch_size,
        }
    )
    write_json(output_dir / "training_history.json", history)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        **ENVIRONMENT_METADATA,
        "scenario": scenario,
        "device": str(device),
        "num_workers": num_workers,
        "train_rows": int(train_y.shape[0]),
        "validation_rows": int(val_y.shape[0]),
        "train_excluded_unstable_rows": train_excluded_unstable,
        "validation_excluded_unstable_rows": val_excluded_unstable,
        "input_dimension": int(train_x.shape[1]),
        "hidden_dim": P2_HIDDEN_DIM,
        "hidden_layers": 2,
        "float32_matmul_precision": matmul_precision,
        "training_batch_size": batch_size,
        "validation_batch_size": validation_batch_size,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "normaliser_path": str((output_dir / "normaliser.json").resolve()),
        "best_epoch": history["best_epoch"],
        "best_val_mse_total_scaled": history["best_val_mse_total_scaled"],
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def prepare_p1(
    scenario: str,
    data_root: Path,
    results_root: Path,
) -> Path:
    """由 training-anchor outcomes 建立 P1 artifact。"""

    actions = load_actions()
    paths = outcome_files(data_root, scenario, "training")

    def valid_rows() -> Iterable[dict[str, str]]:
        for row in iter_csv_rows(paths):
            if int(row["simulation_unstable"]) == 0:
                yield row

    interpolator = TensorOutcomeInterpolator.fit(scenario, actions, valid_rows())
    output_path = results_root / "p1" / scenario / "tensor_outcome_interpolator.npz"
    interpolator.save(output_path)
    write_json(
        output_path.parent / "metadata.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            **ENVIRONMENT_METADATA,
            "scenario": scenario,
            "predictor": "p1",
            "artifact_path": str(output_path.resolve()),
        },
    )
    print(f"P1 artifact: {output_path.resolve()}")
    return output_path


def load_condition_outcomes(
    path: Path,
    action_rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """按 action manifest 顺序读取一个 condition 的 MuJoCo outcomes。"""

    rows = read_csv_rows(path)
    row_by_action = {row["v2_action_id"]: row for row in rows}
    expected_ids = [row["v2_action_id"] for row in action_rows]
    if set(row_by_action) != set(expected_ids):
        raise RuntimeError(f"outcome shard 的 action IDs 不完整: {path}")
    outcomes = np.asarray(
        [
            [float(row_by_action[action_id][field]) for field in OUTCOME_FIELDS]
            for action_id in expected_ids
        ],
        dtype=np.float32,
    )
    valid = np.asarray(
        [
            int(row_by_action[action_id]["quality_pass"]) == 1
            and int(row_by_action[action_id]["simulation_unstable"]) == 0
            for action_id in expected_ids
        ],
        dtype=bool,
    )
    return outcomes, valid


def p0_predictions(
    predictor: Level1OutcomePredictor,
    action_rows: list[dict[str, str]],
) -> np.ndarray:
    """把当前 action-only predictor 输出转换为三维数组。"""

    rows = predictor.predict_rows(action_rows)
    return np.asarray(
        [
            [row["pred_delta_x"], row["pred_delta_y"], row["pred_delta_yaw"]]
            for row in rows
        ],
        dtype=np.float32,
    )


def make_predictors(
    scenario: str,
    predictor_names: tuple[str, ...],
    action_rows: list[dict[str, str]],
    results_root: Path,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """加载当前 evaluation 请求的 P0/P1/P2 artifacts。"""

    predictors: dict[str, Any] = {}
    cached_p0: np.ndarray | None = None
    if "p0" in predictor_names:
        model = Level1OutcomePredictor()
        cached_p0 = p0_predictions(model, action_rows)
        predictors["p0"] = model
    if "p1" in predictor_names:
        artifact_path = (
            results_root / "p1" / scenario / "tensor_outcome_interpolator.npz"
        )
        model = TensorOutcomeInterpolator.load(artifact_path)
        expected_ids = np.asarray([row["v2_action_id"] for row in action_rows])
        if not np.array_equal(model.action_ids, expected_ids):
            raise RuntimeError("P1 artifact 的 action 顺序与当前 manifest 不一致")
        predictors["p1"] = model
    if "p2" in predictor_names:
        output_dir = results_root / "p2" / scenario
        predictors["p2"] = ConditionedOutcomePredictor(
            output_dir / "conditioned_outcome_mlp.pt",
            output_dir / "normaliser.json",
        )
    return predictors, cached_p0


def predict_condition(
    name: str,
    predictor: Any,
    cached_p0: np.ndarray | None,
    action_rows: list[dict[str, str]],
    condition_row: dict[str, str],
) -> np.ndarray:
    """对一个 condition 调用指定 outcome predictor。"""

    if name == "p0":
        if cached_p0 is None:
            raise RuntimeError("P0 prediction cache 尚未建立")
        return cached_p0
    if name == "p1":
        return predictor.predict(condition_row)
    if name == "p2":
        return predictor.predict(action_rows, condition_row)
    raise ValueError(f"未知 predictor: {name}")


def objective_specs(role: str) -> list[tuple[str, Any]]:
    """返回正式 objectives，并只在 validation 加入 yaw sensitivity。"""

    specs = [
        ("v1", PRIMARY_TNPO_COST),
        ("position_only", PRIMARY_TNPO_COST),
        ("v2", PRIMARY_TNPO_COST),
    ]
    if role == "validation":
        specs.append(("v2_yaw10_sensitivity", SENSITIVITY_TNPO_COST))
    return specs


def target_quartiles(target_positions: np.ndarray) -> np.ndarray:
    """按 radial distance 的排序把 targets 等量分成四组。"""

    radii = np.linalg.norm(target_positions, axis=1)
    order = np.argsort(radii, kind="stable")
    labels = np.empty(len(radii), dtype=np.int64)
    for quartile, indices in enumerate(np.array_split(order, 4), start=1):
        labels[indices] = quartile
    return labels


def action_boundary_flags(
    action_row: dict[str, str],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, int]:
    """分别判断 oracle action 是否位于各离散维度边界。"""

    output: dict[str, int] = {}
    for field, (lower, upper) in bounds.items():
        value = float(action_row[field])
        output[f"boundary_{field}"] = int(
            math.isclose(value, lower, abs_tol=1e-12)
            or math.isclose(value, upper, abs_tol=1e-12)
        )
    return output


def action_bounds(action_rows: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    """读取 E1 报告需要的 action-grid bounds。"""

    fields = (
        "force_angle_relative_to_normal_deg",
        "commanded_force_N",
        "ramp_up_s",
        "hold_s",
        "ramp_down_s",
    )
    return {
        field: (
            min(float(row[field]) for row in action_rows),
            max(float(row[field]) for row in action_rows),
        )
        for field in fields
    }


def condition_cluster_mean_ci(
    values: np.ndarray,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """对 condition-level estimates 做 cluster bootstrap。"""

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        selected = rng.integers(0, len(array), size=len(array))
        estimates[index] = np.nanmean(array[selected])
    return (
        float(np.nanquantile(estimates, 0.025)),
        float(np.nanquantile(estimates, 0.975)),
    )


DECISION_FIELDS = [
    "scenario",
    "condition_role",
    "condition_id",
    "target_id",
    "predictor",
    "objective",
    "oracle_action_id",
    "selected_action_id",
    "oracle_cost",
    "selected_mujoco_cost",
    "selected_predicted_cost",
    "selection_gap",
    "selected_position_error_m",
    "selected_yaw_error_rad",
    "selected_yaw_error_deg",
    "near_optimal",
    "one_step_success",
    "selected_rollout_valid",
    "selected_action_optimism",
]

PREDICTION_METRIC_FIELDS = [
    "scenario",
    "condition_role",
    "condition_id",
    "predictor",
    "n_actions",
    "planar_error_mean_mm",
    "planar_error_median_mm",
    "planar_error_p90_mm",
    "yaw_error_mean_deg",
    "yaw_error_median_deg",
    "yaw_error_p90_deg",
    "high_yaw_threshold_deg",
    "high_yaw_planar_error_mean_mm",
    "high_yaw_yaw_error_mean_deg",
    "v2_cost_prediction_mae",
]

FEASIBILITY_FIELDS = [
    "scenario",
    "condition_role",
    "condition_id",
    "target_id",
    "target_radius_m",
    "target_radius_quartile",
    "oracle_action_id",
    "oracle_v2_cost",
    "oracle_position_error_m",
    "oracle_yaw_error_deg",
    "oracle_one_step_success",
    "condition_invalid_rollout_rate",
    "boundary_force_angle_relative_to_normal_deg",
    "boundary_commanded_force_N",
    "boundary_ramp_up_s",
    "boundary_hold_s",
    "boundary_ramp_down_s",
]


def metric_matrix_store(
    predictor_names: tuple[str, ...],
    objectives: list[tuple[str, Any]],
    target_count: int,
    condition_count: int,
) -> dict[tuple[str, str, str], np.ndarray]:
    """为 decision metrics 预分配 target-condition matrices。"""

    metrics = (
        "selected_mujoco_cost",
        "selection_gap",
        "selected_position_error_m",
        "selected_yaw_error_deg",
        "near_optimal",
        "one_step_success",
        "selected_rollout_valid",
        "selected_action_optimism",
    )
    return {
        (predictor, objective, metric): np.full(
            (target_count, condition_count), np.nan, dtype=np.float64
        )
        for predictor in predictor_names
        for objective, _ in objectives
        for metric in metrics
    }


def evaluate_scenario(
    scenario: str,
    role: str,
    predictor_names: tuple[str, ...],
    data_root: Path,
    results_root: Path,
    n_bootstrap: int,
) -> dict[str, Any]:
    """执行一个场景的完整 E1 evaluation。"""

    actions = load_actions()
    all_conditions = load_conditions()
    partition = "model_selection" if role == "validation" else "all"
    conditions = select_conditions(all_conditions, scenario, role, partition)
    targets = load_targets(role)
    target_positions = np.asarray(
        [
            [float(row["target_delta_x_m"]), float(row["target_delta_y_m"])]
            for row in targets
        ],
        dtype=np.float64,
    )
    quartiles = target_quartiles(target_positions)
    bounds = action_bounds(actions)
    predictors, cached_p0 = make_predictors(
        scenario,
        predictor_names,
        actions,
        results_root,
    )
    objectives = objective_specs(role)
    matrices = metric_matrix_store(
        predictor_names,
        objectives,
        len(targets),
        len(conditions),
    )
    decision_rows: list[dict[str, Any]] = []
    prediction_metric_rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []

    for condition_index, condition in enumerate(conditions):
        shard_path = (
            data_root
            / "outcomes"
            / scenario
            / role
            / f"{condition['condition_id']}.csv"
        )
        actual, valid = load_condition_outcomes(shard_path, actions)
        predictions = {
            name: predict_condition(
                name,
                predictor,
                cached_p0,
                actions,
                condition,
            )
            for name, predictor in predictors.items()
        }
        actual_position, actual_yaw = pose_error_matrices(actual, target_positions)
        actual_v2_cost = cost_matrix(
            "v2",
            actual_position,
            actual_yaw,
            PRIMARY_TNPO_COST,
        )
        for name, predicted in predictions.items():
            metrics = outcome_error_summary(actual, predicted, valid)
            predicted_position, predicted_yaw = pose_error_matrices(
                predicted, target_positions
            )
            predicted_v2_cost = cost_matrix(
                "v2",
                predicted_position,
                predicted_yaw,
                PRIMARY_TNPO_COST,
            )
            metrics["v2_cost_prediction_mae"] = float(
                np.mean(
                    np.abs(
                        predicted_v2_cost[:, valid] - actual_v2_cost[:, valid]
                    )
                )
            )
            prediction_metric_rows.append(
                {
                    "scenario": scenario,
                    "condition_role": role,
                    "condition_id": condition["condition_id"],
                    "predictor": name,
                    **metrics,
                }
            )

        for objective, tnpo_config in objectives:
            oracle_reference = evaluate_selector(
                actual,
                actual,
                target_positions,
                valid,
                objective=objective,
                tnpo_config=tnpo_config,
            )
            oracle_indices = oracle_reference["oracle_indices"]
            target_indices = np.arange(len(targets))
            oracle_position = actual_position[target_indices, oracle_indices]
            oracle_yaw = actual_yaw[target_indices, oracle_indices]
            oracle_success = (
                (oracle_position <= PRIMARY_TNPO_COST.position_tolerance_m)
                & (oracle_yaw <= PRIMARY_TNPO_COST.yaw_tolerance_rad)
            )

            for target_index, target in enumerate(targets):
                oracle_action = actions[int(oracle_indices[target_index])]
                decision_rows.append(
                    {
                        "scenario": scenario,
                        "condition_role": role,
                        "condition_id": condition["condition_id"],
                        "target_id": target["v2_target_id"],
                        "predictor": "oracle",
                        "objective": objective,
                        "oracle_action_id": oracle_action["v2_action_id"],
                        "selected_action_id": oracle_action["v2_action_id"],
                        "oracle_cost": oracle_reference["oracle_cost"][target_index],
                        "selected_mujoco_cost": oracle_reference["oracle_cost"][
                            target_index
                        ],
                        "selected_predicted_cost": oracle_reference["oracle_cost"][
                            target_index
                        ],
                        "selection_gap": 0.0,
                        "selected_position_error_m": oracle_position[target_index],
                        "selected_yaw_error_rad": oracle_yaw[target_index],
                        "selected_yaw_error_deg": math.degrees(
                            oracle_yaw[target_index]
                        ),
                        "near_optimal": 1 if objective == "v2" else "",
                        "one_step_success": int(oracle_success[target_index]),
                        "selected_rollout_valid": 1,
                        "selected_action_optimism": 0.0,
                    }
                )

            if objective == "v2":
                invalid_rate = 1.0 - float(np.mean(valid))
                for target_index, target in enumerate(targets):
                    action_index = int(oracle_indices[target_index])
                    oracle_action = actions[action_index]
                    feasibility_rows.append(
                        {
                            "scenario": scenario,
                            "condition_role": role,
                            "condition_id": condition["condition_id"],
                            "target_id": target["v2_target_id"],
                            "target_radius_m": float(
                                np.linalg.norm(target_positions[target_index])
                            ),
                            "target_radius_quartile": int(quartiles[target_index]),
                            "oracle_action_id": oracle_action["v2_action_id"],
                            "oracle_v2_cost": oracle_reference["oracle_cost"][
                                target_index
                            ],
                            "oracle_position_error_m": oracle_position[target_index],
                            "oracle_yaw_error_deg": math.degrees(
                                oracle_yaw[target_index]
                            ),
                            "oracle_one_step_success": int(
                                oracle_success[target_index]
                            ),
                            "condition_invalid_rollout_rate": invalid_rate,
                            **action_boundary_flags(oracle_action, bounds),
                        }
                    )

            for name, predicted in predictions.items():
                result = evaluate_selector(
                    actual,
                    predicted,
                    target_positions,
                    valid,
                    objective=objective,
                    tnpo_config=tnpo_config,
                )
                matrices[(name, objective, "selected_mujoco_cost")][
                    :, condition_index
                ] = result["selected_actual_cost"]
                matrices[(name, objective, "selection_gap")][
                    :, condition_index
                ] = result["selection_gap"]
                matrices[(name, objective, "selected_position_error_m")][
                    :, condition_index
                ] = result["selected_position_error"]
                matrices[(name, objective, "selected_yaw_error_deg")][
                    :, condition_index
                ] = np.degrees(result["selected_yaw_error"])
                if objective == "v2":
                    matrices[(name, objective, "near_optimal")][
                        :, condition_index
                    ] = result["near_optimal"].astype(float)
                matrices[(name, objective, "one_step_success")][
                    :, condition_index
                ] = result["selected_success"].astype(float)
                matrices[(name, objective, "selected_rollout_valid")][
                    :, condition_index
                ] = result["selected_valid"].astype(float)
                matrices[(name, objective, "selected_action_optimism")][
                    :, condition_index
                ] = result["optimism"]

                for target_index, target in enumerate(targets):
                    selected_index = int(result["selected_indices"][target_index])
                    decision_rows.append(
                        {
                            "scenario": scenario,
                            "condition_role": role,
                            "condition_id": condition["condition_id"],
                            "target_id": target["v2_target_id"],
                            "predictor": name,
                            "objective": objective,
                            "oracle_action_id": actions[
                                int(result["oracle_indices"][target_index])
                            ]["v2_action_id"],
                            "selected_action_id": actions[selected_index][
                                "v2_action_id"
                            ],
                            "oracle_cost": result["oracle_cost"][target_index],
                            "selected_mujoco_cost": result["selected_actual_cost"][
                                target_index
                            ],
                            "selected_predicted_cost": result[
                                "selected_predicted_cost"
                            ][target_index],
                            "selection_gap": result["selection_gap"][target_index],
                            "selected_position_error_m": result[
                                "selected_position_error"
                            ][target_index],
                            "selected_yaw_error_rad": result[
                                "selected_yaw_error"
                            ][target_index],
                            "selected_yaw_error_deg": math.degrees(
                                result["selected_yaw_error"][target_index]
                            ),
                            "near_optimal": (
                                int(result["near_optimal"][target_index])
                                if objective == "v2"
                                else ""
                            ),
                            "one_step_success": int(
                                result["selected_success"][target_index]
                            ),
                            "selected_rollout_valid": int(
                                result["selected_valid"][target_index]
                            ),
                            "selected_action_optimism": result["optimism"][
                                target_index
                            ],
                        }
                    )

        print(
            f"evaluated {scenario} condition "
            f"{condition_index + 1}/{len(conditions)}"
        )

    output_dir = results_root / "evaluation" / role / scenario
    write_csv(output_dir / "decision_cases.csv", decision_rows, DECISION_FIELDS)
    write_csv(
        output_dir / "outcome_prediction_metrics.csv",
        prediction_metric_rows,
        PREDICTION_METRIC_FIELDS,
    )
    write_csv(
        output_dir / "oracle_feasibility_cases.csv",
        feasibility_rows,
        FEASIBILITY_FIELDS,
    )

    decision_summary: list[dict[str, Any]] = []
    for (predictor, objective, metric), matrix in matrices.items():
        if not np.isfinite(matrix).any():
            continue
        stats = summary_statistics(matrix)
        if objective == "v2":
            ci_low, ci_high = paired_two_way_bootstrap_mean_ci(
                matrix,
                n_resamples=n_bootstrap,
                seed=20260810,
            )
        else:
            ci_low, ci_high = math.nan, math.nan
        decision_summary.append(
            {
                "predictor": predictor,
                "objective": objective,
                "metric": metric,
                **stats,
                "mean_ci95_low": ci_low,
                "mean_ci95_high": ci_high,
            }
        )

    prediction_summary: list[dict[str, Any]] = []
    metric_names = [
        field
        for field in PREDICTION_METRIC_FIELDS
        if field
        not in {"scenario", "condition_role", "condition_id", "predictor", "n_actions"}
    ]
    for predictor in predictor_names:
        predictor_rows = [
            row for row in prediction_metric_rows if row["predictor"] == predictor
        ]
        for metric in metric_names:
            values = np.asarray(
                [float(row[metric]) for row in predictor_rows], dtype=np.float64
            )
            ci_low, ci_high = condition_cluster_mean_ci(
                values,
                n_resamples=n_bootstrap,
                seed=20260810,
            )
            prediction_summary.append(
                {
                    "predictor": predictor,
                    "metric": metric,
                    "condition_mean": float(np.mean(values)),
                    "mean_ci95_low": ci_low,
                    "mean_ci95_high": ci_high,
                }
            )

    feasibility_summary: list[dict[str, Any]] = []
    boundary_fields = [field for field in FEASIBILITY_FIELDS if field.startswith("boundary_")]
    for quartile in range(1, 5):
        rows = [
            row
            for row in feasibility_rows
            if int(row["target_radius_quartile"]) == quartile
        ]
        feasibility_summary.append(
            {
                "target_radius_quartile": quartile,
                "n_cases": len(rows),
                "oracle_v2_cost_mean": float(
                    np.mean([float(row["oracle_v2_cost"]) for row in rows])
                ),
                "oracle_one_step_success_rate": float(
                    np.mean([float(row["oracle_one_step_success"]) for row in rows])
                ),
                "oracle_position_error_mean_mm": float(
                    np.mean([float(row["oracle_position_error_m"]) for row in rows])
                    * 1000.0
                ),
                "oracle_yaw_error_mean_deg": float(
                    np.mean([float(row["oracle_yaw_error_deg"]) for row in rows])
                ),
                **{
                    f"{field}_rate": float(
                        np.mean([float(row[field]) for row in rows])
                    )
                    for field in boundary_fields
                },
            }
        )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        **ENVIRONMENT_METADATA,
        "scenario": scenario,
        "role": role,
        "conditions": len(conditions),
        "targets": len(targets),
        "actions": len(actions),
        "predictors": list(predictor_names),
        "objectives": [name for name, _ in objectives],
        "bootstrap_resamples": n_bootstrap,
        "decision_metrics": decision_summary,
        "outcome_prediction_metrics": prediction_summary,
        "oracle_feasibility_by_target_quartile": feasibility_summary,
        "artifacts": {
            "decision_cases": str((output_dir / "decision_cases.csv").resolve()),
            "outcome_prediction_metrics": str(
                (output_dir / "outcome_prediction_metrics.csv").resolve()
            ),
            "oracle_feasibility_cases": str(
                (output_dir / "oracle_feasibility_cases.csv").resolve()
            ),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def evaluate_all(
    scenario_value: str,
    role: str,
    predictor_names: tuple[str, ...],
    data_root: Path,
    results_root: Path,
    n_bootstrap: int,
) -> dict[str, Any]:
    """执行所选场景并生成 equal-weight macro average。"""

    scenario_summaries = [
        evaluate_scenario(
            scenario,
            role,
            predictor_names,
            data_root,
            results_root,
            n_bootstrap,
        )
        for scenario in select_scenarios(scenario_value)
    ]
    macro_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for summary in scenario_summaries:
        for row in summary["decision_metrics"]:
            macro_groups[(row["predictor"], row["objective"], row["metric"])].append(
                float(row["mean"])
            )
    macro = [
        {
            "predictor": key[0],
            "objective": key[1],
            "metric": key[2],
            "equal_weight_scenario_mean": float(np.mean(values)),
            "scenario_count": len(values),
        }
        for key, values in sorted(macro_groups.items())
    ]
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        **ENVIRONMENT_METADATA,
        "role": role,
        "scenarios": [summary["scenario"] for summary in scenario_summaries],
        "scenario_summaries": scenario_summaries,
        "equal_weight_macro_average": macro,
    }
    output_path = results_root / "evaluation" / role / "combined_summary.json"
    write_json(output_path, combined)
    print(f"Evaluation summary: {output_path.resolve()}")
    return combined


def parse_predictors(value: str) -> tuple[str, ...]:
    """解析逗号分隔的 P0/P1/P2 predictor names。"""

    names = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    unknown = set(names) - {"p0", "p1", "p2"}
    if unknown:
        raise ValueError(f"未知 predictors: {sorted(unknown)}")
    if not names:
        raise ValueError("至少需要一个 predictor")
    return names


def build_parser() -> argparse.ArgumentParser:
    """构造 E1 统一命令行入口。"""

    parser = argparse.ArgumentParser(description="Run HCR V2 E1 workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan", help="只打印 manifests 与计划规模。")

    smoke_parser = subparsers.add_parser("smoke", help="执行三个场景的最小 MuJoCo smoke。")
    smoke_parser.add_argument("--max-actions", type=int, default=2)
    smoke_parser.add_argument("--num-workers", type=int, default=1)

    collect_parser = subparsers.add_parser("collect", help="采集正式 condition-action outcomes。")
    collect_parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), required=True)
    collect_parser.add_argument(
        "--role", choices=("training", "validation", "test"), required=True
    )
    collect_parser.add_argument("--num-workers", type=int, default=8)
    collect_parser.add_argument("--resume", action="store_true")
    collect_parser.add_argument(
        "--max-actions",
        type=int,
        default=0,
        help="仅用于调试；0 表示完整 action_core。",
    )

    p1_parser = subparsers.add_parser("prepare-p1", help="建立 P1 tensor interpolation artifact。")
    p1_parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), required=True)
    p1_parser.add_argument("--data-root", type=Path, default=E1_DATA_ROOT)
    p1_parser.add_argument("--results-root", type=Path, default=E1_RESULTS_ROOT)

    p2_parser = subparsers.add_parser("train-p2", help="训练 P2 condition-conditioned MLP。")
    p2_parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), required=True)
    p2_parser.add_argument("--data-root", type=Path, default=E1_DATA_ROOT)
    p2_parser.add_argument("--results-root", type=Path, default=E1_RESULTS_ROOT)
    p2_parser.add_argument("--max-epochs", type=int, default=300)
    p2_parser.add_argument("--batch-size", type=int, default=2_048)
    p2_parser.add_argument("--num-workers", type=int, default=0)
    p2_parser.add_argument("--seed", type=int, default=20260810)

    evaluate_parser = subparsers.add_parser("evaluate", help="执行 E1 validation 或 test evaluation。")
    evaluate_parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), required=True)
    evaluate_parser.add_argument("--role", choices=("validation", "test"), required=True)
    evaluate_parser.add_argument("--predictors", default="p0,p1,p2")
    evaluate_parser.add_argument("--data-root", type=Path, default=E1_DATA_ROOT)
    evaluate_parser.add_argument("--results-root", type=Path, default=E1_RESULTS_ROOT)
    evaluate_parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser


def main() -> None:
    """执行用户选择的 E1 子命令。"""

    args = build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(plan_summary(), ensure_ascii=False, indent=2))
        return
    if args.command == "smoke":
        collect_outcomes(
            scenario_value="all",
            role="training",
            validation_partition="model_selection",
            num_workers=args.num_workers,
            resume=False,
            max_actions=args.max_actions,
            smoke=True,
        )
        return
    if args.command == "collect":
        collect_outcomes(
            scenario_value=args.scenario,
            role=args.role,
            validation_partition="model_selection",
            num_workers=args.num_workers,
            resume=args.resume,
            max_actions=args.max_actions,
            smoke=False,
        )
        return
    if args.command == "prepare-p1":
        for scenario in select_scenarios(args.scenario):
            prepare_p1(scenario, args.data_root, args.results_root)
        return
    if args.command == "train-p2":
        for scenario in select_scenarios(args.scenario):
            train_p2(
                scenario,
                args.data_root,
                args.results_root,
                args.max_epochs,
                args.batch_size,
                args.num_workers,
                args.seed,
            )
        return
    if args.command == "evaluate":
        evaluate_all(
            scenario_value=args.scenario,
            role=args.role,
            predictor_names=parse_predictors(args.predictors),
            data_root=args.data_root,
            results_root=args.results_root,
            n_bootstrap=args.bootstrap_resamples,
        )
        return
    raise RuntimeError(f"未处理 command: {args.command}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
