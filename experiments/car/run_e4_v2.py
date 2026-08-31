"""运行 Continuous Action Refinement Experiment 4 Version 2。

CAR 是 Continuous Action Refinement 的代码目录缩写。本脚本实现：

- prepare：生成 Version 2 的 Training/Validation action split；
- collect-model-data：采集 condition-conditioned outcome model 数据；
- train-backends：训练 Candidate A 与 Candidate B；
- evaluate-backends：执行 outcome 与 selection-level Validation；
- fit-likelihood：为胜出 backend 拟合 continuous-action likelihood；
- evaluate-likelihood：校准 Student-t covariance inflation；
- calibrate-on-policy-likelihood：校准 closed-loop 序贯似然温度；
- evaluate-exact-selector：确认全部 2,560 个候选的精确后验边缘化；
- benchmark-workers：选择 closed-loop worker count；
- collect：采集 Version 2 closed-loop trajectories；
- evaluate：评价 Version 2 closed-loop Validation/Test。
"""

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# 先初始化 Conda MKL，避免 NumPy 与 PyTorch 以相反顺序加载 OpenMP runtime。
np.linalg.inv(np.eye(1, dtype=np.float64))

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_e2 as car_e2
import run_e4 as car_e4
from experiments.hcr_v2 import run_e1 as hcr_e1
from experiments.hcr_v2 import run_e3 as hcr_e3
from push_core.hcr_v2.e1 import (
    OUTCOME_FIELDS,
    PRIMARY_TNPO_COST,
    SCENARIO_ACTIVE_COORDINATES,
    TensorOutcomeInterpolator,
    read_csv_rows,
)
from push_core.hcr_v2.e2 import (
    FixedNodePosterior,
    OBSERVATION_SCALE,
    ResidualStatistics,
    condition_row_from_normalised,
    make_quadrature_rule,
)
from push_core.hcr_v2.e3 import interpolate_outcome_grid, tnpo_costs
from push_core.hcr_v2.e4 import posterior_mean_hidden_parameters


PROTOCOL_VERSION = "continuous_action_refinement_e4_v2"
CAR_FULL_NAME = "Continuous Action Refinement"
EXPERIMENT_NAME = "Belief-Space Continuous Action Refinement Version 2"
FRICTION_CONE = "elliptic"
DEFAULT_SEED = 0
SCENARIOS = ("friction", "com", "joint")

ACTION_COUNT = 4_536
CONTINUOUS_CANDIDATES_PER_ANCHOR = 128
TRAINING_ACTIONS_PER_CELL = 12
NEW_TRAINING_ACTIONS_PER_CELL = 9
VALIDATION_ACTIONS_PER_CELL = 4
ACTION_CELL_COUNT = 48
TRAINING_ACTION_COUNT = 576
NEW_TRAINING_ACTION_COUNT = 432
VALIDATION_ACTION_COUNT = 192
RANKING_TARGET_COUNT = 64

MAXIMUM_EPOCHS = 1_200
TRAINING_BATCH_SIZE = 131_072
EVALUATION_BATCH_SIZE = 131_072
INITIAL_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.3
MINIMUM_LEARNING_RATE = 1e-6
EARLY_STOPPING_PATIENCE = 30
MINIMUM_IMPROVEMENT = 1e-6
BACKEND_EQUIVALENCE_TOLERANCE = 0.02

STUDENT_T_DEGREES_OF_FREEDOM = 3.0
POINTS_PER_DIMENSION = 17
BELIEF_UPDATE_HORIZON = 4
NUMERICAL_JITTER = 1e-6
COVARIANCE_INFLATION_CANDIDATES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
SEQUENTIAL_LIKELIHOOD_TEMPERATURE_CANDIDATES = (
    1.0,
    1.25,
    1.4,
    1.45,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    12.0,
    16.0,
)
CALIBRATION_COVERAGE_LOWER = 0.93
CALIBRATION_COVERAGE_UPPER = 0.97
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000

DISCRETE_ANCHOR_BUDGET = 20
DISCRETE_PROPOSAL_BUDGET = 100
EXACT_CONTINUOUS_CANDIDATE_BUDGET = (
    DISCRETE_ANCHOR_BUDGET * CONTINUOUS_CANDIDATES_PER_ANCHOR
)
CONTINUOUS_PROMOTION_MARGIN = 0.05
MAXIMUM_PUSHES = 20
NODE_QUERY_CHUNK_SIZE = 65_536

MANIFEST_ROOT = REPOSITORY_ROOT / "manifests" / "car"
HCR_MANIFEST_ROOT = (
    REPOSITORY_ROOT / "manifests" / "hcr_v2"
)
ACTION_SPLIT_PATH = (
    MANIFEST_ROOT / "continuous_action_refinement_e4_v2_model_action_split.csv"
)
V1_ACTION_SPLIT_PATH = (
    MANIFEST_ROOT / "continuous_action_refinement_e4_likelihood_action_split.csv"
)
SEQUENTIAL_TARGET_PATHS = {
    "validation": HCR_MANIFEST_ROOT
    / "hcr_v2_e5_sequential_extension_validation_target_manifest_v1.csv",
    "test": HCR_MANIFEST_ROOT
    / "hcr_v2_e5_sequential_extension_test_target_manifest_v1.csv",
}

DATA_ROOT = PROJECT_ROOT / "data" / "car" / "experiment_4_v2"
RESULTS_ROOT = PROJECT_ROOT / "results" / "car" / "experiment_4_v2"
V1_DATA_ROOT = PROJECT_ROOT / "data" / "car" / "experiment_4"
V1_RESULTS_ROOT = PROJECT_ROOT / "results" / "car" / "experiment_4"
V1_LIBRARY_PATH = V1_DATA_ROOT / "continuous_candidate_residual_library.npz"
E2_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "car"
    / "experiment_2"
    / "continuous_action_refinement_prediction_dataset.npz"
)
E2_DIRECT_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "car"
    / "experiment_2"
    / "models"
    / "direct_continuous_outcome_model.pt"
)
MODEL_ROOT = RESULTS_ROOT / "models"
CANDIDATE_A_PATH = (
    MODEL_ROOT / "candidate_a" / "conditioned_complete_outcome_model.pt"
)
CANDIDATE_B_PATH = (
    MODEL_ROOT / "candidate_b" / "nominal_anchored_interaction_model.pt"
)
SELECTED_BACKEND_PATH = MODEL_ROOT / "selected_backend.json"
LIKELIHOOD_CONFIGURATION_PATH = (
    RESULTS_ROOT / "continuous_likelihood" / "selected_configuration.json"
)
ON_POLICY_TEMPERING_RESULT_PATH = (
    RESULTS_ROOT
    / "continuous_likelihood"
    / "on_policy_sequential_tempering_validation.json"
)

HCR_E5_DATA_ROOT = PROJECT_ROOT / "data" / "hcr_v2" / "e5" / "closed_loop"
HCR_E5_RESULTS_ROOT = PROJECT_ROOT / "results" / "hcr_v2" / "e5"

ACTION_SPLIT_FIELDS = [
    "physical_action_id",
    "physical_action_key",
    "source_v2_action_id",
    "source_action_index",
    "surface_id",
    "contact_region_col",
    "physical_stratum_index",
    "normalized_offset_norm",
    "offset_quartile_index",
    "model_role",
    "role_action_index",
    "cell_selection_rank",
    "permutation_rank",
    "data_source",
    "collection_required",
    "selection_seed",
]

MODEL_OUTCOME_FIELDS = [
    "experiment_id",
    "protocol_version",
    "car_full_name",
    "experiment_name",
    "friction_cone",
    "scenario",
    "model_role",
    "condition_id",
    "condition_role",
    "condition_index_within_role",
    "hidden_parameter_dimension",
    "friction_sliding_mu",
    "com_offset_x_m",
    "com_offset_y_m",
    "hidden_u_friction",
    "hidden_u_com_x",
    "hidden_u_com_y",
    *[field for field in ACTION_SPLIT_FIELDS if field != "model_role"],
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

MODEL_WORKER_ACTIONS: list[dict[str, Any]] = []
V2_STEP_FIELDS = [
    *car_e4.E4_STEP_FIELDS,
    "sequential_likelihood_temperature",
]


def make_json_compatible(value: Any) -> Any:
    """把 NumPy values 与非有限浮点数转换为严格 JSON values。"""

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
    """写入 UTF-8 严格 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            make_json_compatible(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: list[str],
) -> None:
    """写入 UTF-8 BOM CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_likelihood_configuration() -> dict[str, Any]:
    """读取Version 2 likelihood配置。"""

    if not LIKELIHOOD_CONFIGURATION_PATH.exists():
        raise FileNotFoundError("请先完成evaluate-likelihood")
    with LIKELIHOOD_CONFIGURATION_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configured_sequential_temperature(
    scenario: str,
    require_calibrated: bool = True,
) -> float:
    """读取场景对应的序贯似然温度。"""

    configuration = load_likelihood_configuration()
    values = configuration.get("sequential_likelihood_temperature", {})
    if scenario in values:
        return float(values[scenario])
    if require_calibrated:
        raise RuntimeError("请先运行calibrate-on-policy-likelihood --scenario all")
    return 1.0


def select_scenarios(value: str) -> tuple[str, ...]:
    """解析单场景或全部场景。"""

    return SCENARIOS if value == "all" else (value,)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """设置 NumPy 与 PyTorch seed。"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context():
    """仅在 CUDA 上启用 BF16 autocast。"""

    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def require_cuda() -> torch.device:
    """返回正式训练与推理使用的 CUDA device。"""

    if not torch.cuda.is_available():
        raise RuntimeError("E4 V2 正式模型训练与推理需要 CUDA")
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def valid_outcome(row: dict[str, Any]) -> bool:
    """判断一个 atomic push 是否能进入正式统计。"""

    return (
        int(row["quality_pass"]) == 1
        and int(row["simulation_unstable"]) == 0
        and int(row["contact_success"]) == 1
        and int(row["stopped_by_threshold"]) == 1
    )


def hidden_full_from_row(row: dict[str, Any]) -> np.ndarray:
    """读取统一三维 normalised hidden-condition coordinate。"""

    return np.asarray(
        [
            float(row["hidden_u_friction"]),
            float(row["hidden_u_com_x"]),
            float(row["hidden_u_com_y"]),
        ],
        dtype=np.float32,
    )


def expand_active_hidden(scenario: str, values: np.ndarray) -> np.ndarray:
    """把场景 active coordinates 扩展为统一三维 hidden coordinate。"""

    values = np.asarray(values, dtype=np.float32)
    if scenario == "friction":
        return np.concatenate(
            [values[..., :1], np.zeros((*values.shape[:-1], 2), dtype=np.float32)],
            axis=-1,
        )
    if scenario == "com":
        return np.concatenate(
            [np.zeros((*values.shape[:-1], 1), dtype=np.float32), values[..., :2]],
            axis=-1,
        )
    return values[..., :3]


def action_features_np(
    physical_action_ids: np.ndarray,
    library: car_e4.CandidateLibrary,
) -> np.ndarray:
    """构造 18-dimensional continuous physical action representation。"""

    ids = np.asarray(physical_action_ids, dtype=np.int64)
    one_hot = np.eye(12, dtype=np.float32)[library.region_index[ids]]
    q_scaled = (
        library.continuous_q[ids] - car_e2.ACTION_CENTRES
    ) / car_e2.ACTION_SCALES
    return np.concatenate([one_hot, q_scaled], axis=1).astype(np.float32)


def build_action_split() -> list[dict[str, Any]]:
    """按既定48 cells与seed 0生成576/192 action split。"""

    library = car_e4.CandidateLibrary.load(V1_LIBRARY_PATH)
    _, ordered_actions = car_e4.load_ordered_actions()
    anchor_q = np.asarray(
        [car_e2.action_coordinate(row) for row in ordered_actions], dtype=np.float32
    )
    offset = (
        library.continuous_q - anchor_q[library.source_action_index]
    ) / car_e2.OFFSET_SCALES
    offset_norm = np.linalg.vector_norm(offset, axis=1)
    surface = np.asarray(
        [int(ordered_actions[index]["surface_id"]) for index in library.source_action_index],
        dtype=np.int8,
    )
    contact_col = np.asarray(
        [
            int(ordered_actions[index]["contact_region_col"])
            for index in library.source_action_index
        ],
        dtype=np.int8,
    )
    physical_stratum = surface * 3 + contact_col
    quartile = np.empty(len(offset_norm), dtype=np.int8)
    for stratum in range(12):
        mask = physical_stratum == stratum
        boundaries = np.quantile(offset_norm[mask], [0.25, 0.50, 0.75])
        quartile[mask] = np.searchsorted(boundaries, offset_norm[mask], side="right")

    v1_rows = read_csv_rows(V1_ACTION_SPLIT_PATH)
    v1_ids = {
        (int(row["physical_stratum_index"]), int(row["offset_quartile_index"])): []
        for row in v1_rows
    }
    for row in v1_rows:
        v1_ids[
            (int(row["physical_stratum_index"]), int(row["offset_quartile_index"]))
        ].append(int(row["physical_action_id"]))

    rows: list[dict[str, Any]] = []
    role_counts = {"training": 0, "validation": 0}
    for stratum in range(12):
        for quartile_index in range(4):
            indices = np.flatnonzero(
                (physical_stratum == stratum) & (quartile == quartile_index)
            )
            keyed: list[tuple[str, int]] = []
            for physical_action_id in indices:
                action = library.action_row(int(physical_action_id), ordered_actions)
                keyed.append((action["physical_action_key"], int(physical_action_id)))
            keyed.sort(key=lambda item: item[0])
            stable_indices = np.asarray([item[1] for item in keyed], dtype=np.int64)
            seed = int(
                np.random.SeedSequence(
                    [DEFAULT_SEED, stratum, quartile_index]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            permutation = np.random.default_rng(seed).permutation(stable_indices)
            if set(permutation[:6].tolist()) != set(v1_ids[(stratum, quartile_index)]):
                raise RuntimeError(
                    f"cell {stratum}/{quartile_index} 与 Version 1 split 不一致"
                )
            selections = [
                *(('training', rank, 'version_1_training', False) for rank in range(3)),
                *(('training', rank, 'version_2_new', True) for rank in range(6, 15)),
                *(('validation', rank, 'version_2_fresh', True) for rank in range(15, 19)),
            ]
            cell_role_counts = {"training": 0, "validation": 0}
            for role, permutation_rank, data_source, collection_required in selections:
                physical_action_id = int(permutation[permutation_rank])
                action = library.action_row(physical_action_id, ordered_actions)
                rows.append(
                    {
                        "physical_action_id": physical_action_id,
                        "physical_action_key": action["physical_action_key"],
                        "source_v2_action_id": action["source_v2_action_id"],
                        "source_action_index": action["source_action_index"],
                        "surface_id": action["surface_id"],
                        "contact_region_col": action["contact_region_col"],
                        "physical_stratum_index": stratum,
                        "normalized_offset_norm": float(offset_norm[physical_action_id]),
                        "offset_quartile_index": quartile_index,
                        "model_role": role,
                        "role_action_index": role_counts[role],
                        "cell_selection_rank": cell_role_counts[role],
                        "permutation_rank": permutation_rank,
                        "data_source": data_source,
                        "collection_required": int(collection_required),
                        "selection_seed": seed,
                    }
                )
                role_counts[role] += 1
                cell_role_counts[role] += 1
    rows.sort(key=lambda row: (row["model_role"], int(row["role_action_index"])))
    if role_counts != {"training": TRAINING_ACTION_COUNT, "validation": VALIDATION_ACTION_COUNT}:
        raise RuntimeError(f"E4 V2 action split 数量错误: {role_counts}")
    return rows


def prepare_experiment() -> dict[str, Any]:
    """生成 Version 2 model action split，不复制 Version 1 candidate library。"""

    if not V1_LIBRARY_PATH.exists():
        raise FileNotFoundError("请先保留并完成 Experiment 4 Version 1 prepare artifact")
    rows = build_action_split()
    write_csv(ACTION_SPLIT_PATH, rows, ACTION_SPLIT_FIELDS)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "car_full_name": CAR_FULL_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "friction_cone": FRICTION_CONE,
        "default_seed": DEFAULT_SEED,
        "reused_candidate_library": str(V1_LIBRARY_PATH.resolve()),
        "training_actions": TRAINING_ACTION_COUNT,
        "reused_version_1_training_actions": 144,
        "new_training_actions": NEW_TRAINING_ACTION_COUNT,
        "fresh_validation_actions": VALIDATION_ACTION_COUNT,
        "expected_reused_training_outcomes": 22_320,
        "expected_new_training_outcomes": 66_960,
        "expected_fresh_validation_outcomes": 9_216,
        "action_split_path": str(ACTION_SPLIT_PATH.resolve()),
    }
    write_json(RESULTS_ROOT / "preparation_summary.json", summary)
    print(json.dumps(make_json_compatible(summary), ensure_ascii=False, indent=2))
    return summary


def load_model_actions(role: str, collection_only: bool) -> list[dict[str, Any]]:
    """读取并重建一个 model role 的 continuous actions。"""

    rows = [
        row
        for row in read_csv_rows(ACTION_SPLIT_PATH)
        if row["model_role"] == role
        and (not collection_only or int(row["collection_required"]) == 1)
    ]
    rows.sort(key=lambda row: int(row["role_action_index"]))
    library = car_e4.CandidateLibrary.load(V1_LIBRARY_PATH)
    _, ordered_actions = car_e4.load_ordered_actions()
    return [
        {
            **library.action_row(int(row["physical_action_id"]), ordered_actions),
            **row,
        }
        for row in rows
    ]


def initialise_model_worker(actions: list[dict[str, Any]]) -> None:
    """为 condition-level collection worker 保存共享 actions。"""

    global MODEL_WORKER_ACTIONS
    MODEL_WORKER_ACTIONS = actions


def process_model_condition(task: dict[str, Any]) -> list[dict[str, Any]]:
    """在一个 hidden condition 下执行当前 role 的全部 continuous actions。"""

    condition = task["condition"]
    model, data = car_e4.load_model(Path(task["xml_path"]))
    hcr_e1.set_sliding_friction(model, float(condition["friction_sliding_mu"]))
    rows: list[dict[str, Any]] = []
    for action_index, action in enumerate(MODEL_WORKER_ACTIONS):
        episode_id = int(condition["condition_index_within_role"]) * 10_000 + action_index
        rollout_input = car_e4.car_e1.build_rollout_input(action, episode_id)
        rollout_input["dataset_role"] = (
            f"car_e4_v2_model_{action['model_role']}_{condition['scenario']}"
        )
        rollout_input["hidden_com_offset_x"] = float(condition["com_offset_x_m"])
        rollout_input["hidden_com_offset_y"] = float(condition["com_offset_y_m"])
        result = car_e4.car_e1.run_physical_pusher_rollout(
            model, data, rollout_input, validate_schema=False
        )
        rows.append(
            {
                "experiment_id": "E4_V2",
                "protocol_version": PROTOCOL_VERSION,
                "car_full_name": CAR_FULL_NAME,
                "experiment_name": EXPERIMENT_NAME,
                "friction_cone": FRICTION_CONE,
                "scenario": condition["scenario"],
                "model_role": action["model_role"],
                "condition_id": condition["condition_id"],
                "condition_role": condition["condition_role"],
                "condition_index_within_role": condition["condition_index_within_role"],
                "hidden_parameter_dimension": condition["hidden_parameter_dimension"],
                "friction_sliding_mu": condition["friction_sliding_mu"],
                "com_offset_x_m": condition["com_offset_x_m"],
                "com_offset_y_m": condition["com_offset_y_m"],
                "hidden_u_friction": condition["hidden_u_friction"],
                "hidden_u_com_x": condition["hidden_u_com_x"],
                "hidden_u_com_y": condition["hidden_u_com_y"],
                **{field: action.get(field, "") for field in ACTION_SPLIT_FIELDS},
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


def model_outcome_path(scenario: str, role: str) -> Path:
    """返回 Version 2 新采集 outcome path。"""

    return DATA_ROOT / f"condition_model_{role}" / f"{scenario}_outcomes.csv"


def collect_model_data(args: argparse.Namespace) -> dict[str, Any]:
    """按 hidden condition 并行采集新增 Training 或 fresh Validation outcomes。"""

    actions = load_model_actions(args.role, collection_only=True)
    expected_actions = (
        NEW_TRAINING_ACTION_COUNT if args.role == "training" else VALIDATION_ACTION_COUNT
    )
    if len(actions) != expected_actions:
        raise RuntimeError(f"{args.role} collection action 数量错误: {len(actions)}")
    summaries: dict[str, Any] = {}
    for scenario in select_scenarios(args.scenario):
        conditions = hcr_e3.load_conditions(scenario, args.role)
        if args.max_conditions > 0:
            conditions = conditions[: args.max_conditions]
        generated_dir = DATA_ROOT / "generated_xml" / args.role / scenario
        xml_by_com = hcr_e1.prepare_environment_xmls(conditions, generated_dir)
        output_path = model_outcome_path(scenario, args.role)
        existing = read_csv_rows(output_path) if args.resume and output_path.exists() else []
        counts = Counter(row["condition_id"] for row in existing)
        complete_ids = {
            condition_id for condition_id, count in counts.items() if count == len(actions)
        }
        retained = [row for row in existing if row["condition_id"] in complete_ids]
        tasks: list[dict[str, Any]] = []
        for condition in conditions:
            if condition["condition_id"] in complete_ids:
                continue
            com_key = (
                round(float(condition["com_offset_x_m"]), 9),
                round(float(condition["com_offset_y_m"]), 9),
            )
            tasks.append({"condition": condition, "xml_path": str(xml_by_com[com_key])})

        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        processed = 0
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=MODEL_OUTCOME_FIELDS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(retained)
            worker_count = min(max(1, args.num_workers), max(1, len(tasks)))
            if not tasks:
                iterator: Iterable[list[dict[str, Any]]] = []
                pool = None
            elif worker_count == 1:
                initialise_model_worker(actions)
                iterator = map(process_model_condition, tasks)
                pool = None
            else:
                context = mp.get_context("spawn")
                pool = context.Pool(
                    worker_count,
                    initializer=initialise_model_worker,
                    initargs=(actions,),
                    maxtasksperchild=8,
                )
                iterator = pool.imap(process_model_condition, tasks)
            try:
                for result_rows in iterator:
                    writer.writerows(result_rows)
                    handle.flush()
                    processed += 1
                    print(
                        f"finished {scenario} {args.role} condition "
                        f"{processed}/{len(tasks)}"
                    )
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
        rows = read_csv_rows(output_path)
        valid_count = sum(valid_outcome(row) for row in rows)
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "friction_cone": FRICTION_CONE,
            "scenario": scenario,
            "model_role": args.role,
            "conditions": len(conditions),
            "actions_per_condition": len(actions),
            "rollouts": len(rows),
            "valid_rollouts": valid_count,
            "validity_rate": valid_count / len(rows) if rows else 0.0,
            "resumed_conditions": len(complete_ids),
            "new_conditions": processed,
            "num_workers": min(max(1, args.num_workers), max(1, len(tasks))),
            "elapsed_seconds": time.perf_counter() - started,
            "outcome_path": str(output_path.resolve()),
        }
        write_json(RESULTS_ROOT / "collection" / args.role / f"{scenario}.json", summary)
        summaries[scenario] = summary
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "model_role": args.role,
        "scenario_summaries": summaries,
    }
    print(json.dumps(make_json_compatible(combined), ensure_ascii=False, indent=2))
    return combined


def transform_v1_training_rows(scenario: str) -> list[dict[str, Any]]:
    """把 Version 1 Training outcomes映射到 Version 2统一schema。"""

    split_by_id = {
        int(row["physical_action_id"]): row
        for row in read_csv_rows(ACTION_SPLIT_PATH)
        if row["model_role"] == "training" and row["data_source"] == "version_1_training"
    }
    rows = read_csv_rows(V1_DATA_ROOT / "likelihood_training" / f"{scenario}_outcomes.csv")
    outputs: list[dict[str, Any]] = []
    for row in rows:
        physical_id = int(row["physical_action_id"])
        split = split_by_id[physical_id]
        outputs.append(
            {
                **row,
                **split,
                "experiment_id": "E4_V2_REUSED_V1",
                "protocol_version": PROTOCOL_VERSION,
                "experiment_name": EXPERIMENT_NAME,
                "model_role": "training",
            }
        )
    return outputs


def load_cross_rows(role: str) -> dict[str, list[dict[str, Any]]]:
    """读取三场景的完整 Training 或 fresh Validation rows。"""

    outputs: dict[str, list[dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        rows = read_csv_rows(model_outcome_path(scenario, role))
        if role == "training":
            rows = [*transform_v1_training_rows(scenario), *rows]
        rows.sort(
            key=lambda row: (
                int(row["condition_index_within_role"]),
                int(row["role_action_index"]),
            )
        )
        expected_conditions = len(hcr_e3.load_conditions(scenario, role))
        expected_actions = TRAINING_ACTION_COUNT if role == "training" else VALIDATION_ACTION_COUNT
        expected = expected_conditions * expected_actions
        if len(rows) != expected:
            raise RuntimeError(
                f"{scenario}/{role} rows 数量错误: {len(rows)} != {expected}"
            )
        if np.mean([valid_outcome(row) for row in rows]) < 0.99:
            raise RuntimeError(f"{scenario}/{role} rollout validity 低于99%")
        outputs[scenario] = rows
    return outputs


class OutcomeMLP(nn.Module):
    """E4 V2 固定的128×128 SiLU MLP。"""

    def __init__(self, input_dimension: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """预测标准化outcome或condition-induced change。"""

        return self.network(inputs)


def load_e2_nominal_model(
    device: torch.device,
) -> tuple[car_e2.OutcomeMLP, np.ndarray, np.ndarray]:
    """读取冻结的Experiment 2 nominal-condition outcome model。"""

    checkpoint = torch.load(E2_DIRECT_MODEL_PATH, map_location="cpu", weights_only=False)
    model = car_e2.OutcomeMLP(int(checkpoint["input_dimension"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    normalisers = checkpoint["normalisers"]
    label_mean = normalisers["label_mean"].detach().cpu().numpy().astype(np.float32)
    label_scale = normalisers["label_scale"].detach().cpu().numpy().astype(np.float32)
    return model, label_mean, label_scale


def initialise_candidate_a_from_e2(model: OutcomeMLP) -> None:
    """把Experiment 2 nominal mapping转换到HCR标准化outcome坐标。"""

    checkpoint = torch.load(E2_DIRECT_MODEL_PATH, map_location="cpu", weights_only=False)
    old_state = checkpoint["state_dict"]
    label_mean = checkpoint["normalisers"]["label_mean"].float()
    label_scale = checkpoint["normalisers"]["label_scale"].float()
    observation_scale = torch.from_numpy(OBSERVATION_SCALE.astype(np.float32))
    with torch.no_grad():
        model.network[0].weight.zero_()
        model.network[0].weight[:, :18].copy_(old_state["network.0.weight"])
        model.network[0].bias.copy_(old_state["network.0.bias"])
        model.network[2].weight.copy_(old_state["network.2.weight"])
        model.network[2].bias.copy_(old_state["network.2.bias"])
        output_scale = label_scale / observation_scale
        model.network[4].weight.copy_(
            old_state["network.4.weight"] * output_scale[:, None]
        )
        model.network[4].bias.copy_(
            (old_state["network.4.bias"] * label_scale + label_mean)
            / observation_scale
        )


def predict_e2_physical(
    model: car_e2.OutcomeMLP,
    action_features: torch.Tensor,
    label_mean: torch.Tensor,
    label_scale: torch.Tensor,
) -> torch.Tensor:
    """使用冻结Experiment 2 model预测physical outcome。"""

    with autocast_context():
        scaled = model(action_features)
    return scaled.float() * label_scale + label_mean


def candidate_b_correction(
    model: OutcomeMLP,
    action_features: torch.Tensor,
    hidden: torch.Tensor,
    p1_shift_standardized: torch.Tensor,
) -> torch.Tensor:
    """计算严格zero-centred condition-interaction correction。"""

    current = torch.cat([action_features, hidden, p1_shift_standardized], dim=1)
    nominal = torch.cat(
        [
            action_features,
            torch.zeros_like(hidden),
            torch.zeros_like(p1_shift_standardized),
        ],
        dim=1,
    )
    with autocast_context():
        return model(current).float() - model(nominal).float()


@dataclass
class CrossArrays:
    """一个场景的condition-action监督数组。"""

    action_features: np.ndarray
    hidden: np.ndarray
    p1_shift_standardized: np.ndarray
    truth: np.ndarray
    nominal_truth: np.ndarray
    physical_action_id: np.ndarray
    condition_index: np.ndarray
    action_cell: np.ndarray


def scenario_p1_shift(
    scenario: str,
    rows: list[dict[str, Any]],
    physical_ids: np.ndarray,
    library: car_e4.CandidateLibrary,
) -> np.ndarray:
    """计算每一行source anchor相对nominal condition的P1 outcome shift。"""

    p1_path = (
        PROJECT_ROOT
        / "results"
        / "hcr_v2"
        / "e1"
        / "p1"
        / scenario
        / "tensor_outcome_interpolator.npz"
    )
    p1 = TensorOutcomeInterpolator.load(p1_path)
    nominal = p1.predict(condition_row_from_normalised(scenario, np.zeros(len(SCENARIO_ACTIVE_COORDINATES[scenario]))))
    source = library.source_action_index[physical_ids]
    outputs = np.empty((len(rows), 3), dtype=np.float32)
    cache: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        condition_id = str(row["condition_id"])
        if condition_id not in cache:
            cache[condition_id] = p1.predict(row).astype(np.float32)
        outputs[index] = cache[condition_id][source[index]] - nominal[source[index]]
    return outputs


def build_cross_arrays(
    scenario: str,
    rows: list[dict[str, Any]],
    library: car_e4.CandidateLibrary,
) -> CrossArrays:
    """把condition-action rows转换为A/B共用监督数组。"""

    physical_ids = np.asarray([int(row["physical_action_id"]) for row in rows], dtype=np.int64)
    features = action_features_np(physical_ids, library)
    hidden = np.asarray([hidden_full_from_row(row) for row in rows], dtype=np.float32)
    truth = np.asarray(
        [[float(row[field]) for field in OUTCOME_FIELDS] for row in rows],
        dtype=np.float32,
    )
    p1_shift = scenario_p1_shift(scenario, rows, physical_ids, library)
    nominal_by_action: dict[int, np.ndarray] = {}
    if scenario == "joint":
        for row, physical_id, outcome, coordinate in zip(rows, physical_ids, truth, hidden):
            if np.allclose(coordinate, 0.0, atol=1e-8):
                nominal_by_action[int(physical_id)] = outcome
    nominal_truth = np.zeros_like(truth)
    if nominal_by_action:
        nominal_truth = np.asarray(
            [nominal_by_action[int(physical_id)] for physical_id in physical_ids],
            dtype=np.float32,
        )
    action_cells = np.asarray(
        [
            int(row["physical_stratum_index"]) * 4
            + int(row["offset_quartile_index"])
            for row in rows
        ],
        dtype=np.int16,
    )
    return CrossArrays(
        action_features=features,
        hidden=hidden,
        p1_shift_standardized=(p1_shift / OBSERVATION_SCALE).astype(np.float32),
        truth=truth,
        nominal_truth=nominal_truth,
        physical_action_id=physical_ids,
        condition_index=np.asarray(
            [int(row["condition_index_within_role"]) for row in rows], dtype=np.int16
        ),
        action_cell=action_cells,
    )


def attach_joint_nominal_truth(arrays: dict[str, CrossArrays]) -> None:
    """为三场景Training rows附加相同physical action的真实nominal outcome。"""

    joint = arrays["joint"]
    nominal_by_action: dict[int, np.ndarray] = {}
    for physical_id, hidden, outcome in zip(
        joint.physical_action_id, joint.hidden, joint.truth
    ):
        if np.allclose(hidden, 0.0, atol=1e-8):
            nominal_by_action[int(physical_id)] = outcome
    if len(nominal_by_action) != TRAINING_ACTION_COUNT:
        raise RuntimeError(
            f"Joint nominal outcomes 数量错误: {len(nominal_by_action)}"
        )
    for scenario_arrays in arrays.values():
        scenario_arrays.nominal_truth[:] = np.asarray(
            [nominal_by_action[int(value)] for value in scenario_arrays.physical_action_id],
            dtype=np.float32,
        )


def load_e2_nominal_arrays(role: str) -> tuple[np.ndarray, np.ndarray]:
    """读取Candidate A nominal replay的Experiment 2 split。"""

    with np.load(E2_DATASET_PATH, allow_pickle=False) as payload:
        role_code = car_e2.ROLE_CODES[role]
        mask = (payload["role"] == role_code) & payload["valid"]
        one_hot = np.eye(12, dtype=np.float32)[payload["region"][mask]]
        q_scaled = (
            payload["continuous_q"][mask] - car_e2.ACTION_CENTRES
        ) / car_e2.ACTION_SCALES
        features = np.concatenate([one_hot, q_scaled], axis=1).astype(np.float32)
        truth = payload["true_outcome"][mask].astype(np.float32)
    return features, truth


def target_normalized_error(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """计算逐行Target-Normalized Outcome Prediction Error。"""

    position = np.linalg.vector_norm(prediction[:, :2] - truth[:, :2], axis=1)
    yaw = np.abs(
        (prediction[:, 2] - truth[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    )
    return (
        0.5 * position / PRIMARY_TNPO_COST.position_tolerance_m
        + 0.5 * yaw / PRIMARY_TNPO_COST.yaw_tolerance_rad
    )


def predict_in_batches(
    model: OutcomeMLP,
    inputs: torch.Tensor,
    batch_size: int = EVALUATION_BATCH_SIZE,
) -> torch.Tensor:
    """批量运行一个outcome MLP并返回FP32 tensor。"""

    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            with autocast_context():
                outputs.append(model(inputs[start : start + batch_size]).float())
    return torch.cat(outputs, dim=0)


def candidate_a_validation_score(
    model: OutcomeMLP,
    nominal_validation: tuple[torch.Tensor, torch.Tensor],
    cross_validation: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """计算Candidate A四组equal-weight macro Validation error。"""

    scores: list[float] = []
    groups = {"nominal": nominal_validation, **cross_validation}
    for features, truth in groups.values():
        hidden = torch.zeros((len(features), 3), device=features.device)
        if features.shape[1] == 21:
            inputs = features
        else:
            inputs = torch.cat([features, hidden], dim=1)
        prediction = predict_in_batches(model, inputs).cpu().numpy() * OBSERVATION_SCALE
        scores.append(float(np.mean(target_normalized_error(prediction, truth.cpu().numpy()))))
    return float(np.mean(scores))


def candidate_b_prediction_tensor(
    correction_model: OutcomeMLP,
    base_model: car_e2.OutcomeMLP,
    action_features: torch.Tensor,
    hidden: torch.Tensor,
    p1_shift: torch.Tensor,
    label_mean: torch.Tensor,
    label_scale: torch.Tensor,
) -> torch.Tensor:
    """计算Candidate B完整physical outcome。"""

    base = predict_e2_physical(base_model, action_features, label_mean, label_scale)
    correction = candidate_b_correction(
        correction_model, action_features, hidden, p1_shift
    )
    return base + correction * torch.as_tensor(
        OBSERVATION_SCALE, dtype=torch.float32, device=action_features.device
    )


def candidate_b_validation_score(
    correction_model: OutcomeMLP,
    base_model: car_e2.OutcomeMLP,
    validation: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    label_mean: torch.Tensor,
    label_scale: torch.Tensor,
) -> float:
    """计算Candidate B三场景equal-weight macro Validation error。"""

    scores: list[float] = []
    correction_model.eval()
    with torch.inference_mode():
        for action_features, hidden, p1_shift, truth in validation.values():
            parts: list[torch.Tensor] = []
            for start in range(0, len(action_features), EVALUATION_BATCH_SIZE):
                parts.append(
                    candidate_b_prediction_tensor(
                        correction_model,
                        base_model,
                        action_features[start : start + EVALUATION_BATCH_SIZE],
                        hidden[start : start + EVALUATION_BATCH_SIZE],
                        p1_shift[start : start + EVALUATION_BATCH_SIZE],
                        label_mean,
                        label_scale,
                    )
                )
            prediction = torch.cat(parts).cpu().numpy()
            scores.append(float(np.mean(target_normalized_error(prediction, truth.cpu().numpy()))))
    return float(np.mean(scores))


def sampled_group_indices(
    row_count: int,
    sample_count: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """从一个场景group均匀有放回抽取训练索引。"""

    return torch.randint(row_count, (sample_count,), generator=generator, device=device)


def train_candidate_a(
    device: torch.device,
    training: dict[str, CrossArrays],
    validation: dict[str, CrossArrays],
) -> dict[str, Any]:
    """训练Candidate A完整condition-conditioned outcome model。"""

    set_seed(DEFAULT_SEED)
    model = OutcomeMLP(21).to(device)
    initialise_candidate_a_from_e2(model)
    nominal_train_np, nominal_truth_np = load_e2_nominal_arrays("training")
    nominal_val_np, nominal_val_truth_np = load_e2_nominal_arrays("validation")
    train_groups: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "nominal": (
            torch.from_numpy(nominal_train_np).to(device),
            torch.from_numpy(nominal_truth_np / OBSERVATION_SCALE).to(device),
        )
    }
    for scenario, arrays in training.items():
        features = np.concatenate([arrays.action_features, arrays.hidden], axis=1)
        train_groups[scenario] = (
            torch.from_numpy(features).to(device),
            torch.from_numpy(arrays.truth / OBSERVATION_SCALE).to(device),
        )
    nominal_validation = (
        torch.from_numpy(nominal_val_np).to(device),
        torch.from_numpy(nominal_val_truth_np).to(device),
    )
    cross_validation = {
        scenario: (
            torch.from_numpy(np.concatenate([arrays.action_features, arrays.hidden], axis=1)).to(device),
            torch.from_numpy(arrays.truth).to(device),
        )
        for scenario, arrays in validation.items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MINIMUM_LEARNING_RATE,
    )
    generator = torch.Generator(device=device).manual_seed(DEFAULT_SEED)
    group_sample_count = TRAINING_BATCH_SIZE // 4
    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_features: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        for name in ("nominal", *SCENARIOS):
            features, labels = train_groups[name]
            indices = sampled_group_indices(
                len(features), group_sample_count, generator, device
            )
            selected_features = features[indices]
            if name == "nominal":
                selected_features = torch.cat(
                    [selected_features, torch.zeros((len(indices), 3), device=device)],
                    dim=1,
                )
            batch_features.append(selected_features)
            batch_labels.append(labels[indices])
        inputs = torch.cat(batch_features)
        labels = torch.cat(batch_labels)
        order = torch.randperm(len(inputs), generator=generator, device=device)
        with autocast_context():
            prediction = model(inputs[order])
        loss = torch.mean((prediction.float() - labels[order].float()).square())
        loss.backward()
        optimizer.step()
        validation_score = candidate_a_validation_score(
            model, nominal_validation, cross_validation
        )
        scheduler.step(validation_score)
        if validation_score < best_score - MINIMUM_IMPROVEMENT:
            best_score = validation_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "training_standardized_mse": float(loss.detach()),
            "validation_macro_target_normalized_error": validation_score,
            "best_validation_macro_target_normalized_error": best_score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "stale_epochs": stale,
        }
        history.append(row)
        print(
            f"candidate_a epoch {epoch:04d}/{MAXIMUM_EPOCHS}  "
            f"train={row['training_standardized_mse']:.6f}  "
            f"validation={validation_score:.6f}  best={best_score:.6f}  "
            f"lr={row['learning_rate']:.1e}  stale={stale}/{EARLY_STOPPING_PATIENCE}"
        )
        if stale >= EARLY_STOPPING_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("Candidate A 未生成有效checkpoint")
    CANDIDATE_A_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "backend_id": "candidate_a",
            "backend_name": "Hidden-Condition-Conditioned Continuous-Action Outcome Model",
            "input_dimension": 21,
            "hidden_dimensions": [128, 128],
            "activation": "SiLU",
            "output_scale": torch.from_numpy(OBSERVATION_SCALE.astype(np.float32)),
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_macro_target_normalized_error": best_score,
            "training_seed": DEFAULT_SEED,
        },
        CANDIDATE_A_PATH,
    )
    write_csv(
        CANDIDATE_A_PATH.parent / "training_history.csv", history, list(history[0])
    )
    return {
        "backend_id": "candidate_a",
        "checkpoint_path": str(CANDIDATE_A_PATH.resolve()),
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_target_normalized_error": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def train_candidate_b(
    device: torch.device,
    training: dict[str, CrossArrays],
    validation: dict[str, CrossArrays],
) -> dict[str, Any]:
    """训练Candidate B zero-centred condition-interaction correction。"""

    set_seed(DEFAULT_SEED)
    model = OutcomeMLP(24).to(device)
    with torch.no_grad():
        model.network[4].weight.zero_()
        model.network[4].bias.zero_()
    base_model, label_mean_np, label_scale_np = load_e2_nominal_model(device)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    label_mean = torch.from_numpy(label_mean_np).to(device)
    label_scale = torch.from_numpy(label_scale_np).to(device)
    train_groups = {
        scenario: (
            torch.from_numpy(arrays.action_features).to(device),
            torch.from_numpy(arrays.hidden).to(device),
            torch.from_numpy(arrays.p1_shift_standardized).to(device),
            torch.from_numpy((arrays.truth - arrays.nominal_truth) / OBSERVATION_SCALE).to(device),
        )
        for scenario, arrays in training.items()
    }
    validation_groups = {
        scenario: (
            torch.from_numpy(arrays.action_features).to(device),
            torch.from_numpy(arrays.hidden).to(device),
            torch.from_numpy(arrays.p1_shift_standardized).to(device),
            torch.from_numpy(arrays.truth).to(device),
        )
        for scenario, arrays in validation.items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MINIMUM_LEARNING_RATE,
    )
    generator = torch.Generator(device=device).manual_seed(DEFAULT_SEED)
    group_sample_count = TRAINING_BATCH_SIZE // 3
    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        action_parts: list[torch.Tensor] = []
        hidden_parts: list[torch.Tensor] = []
        shift_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        for scenario in SCENARIOS:
            action, hidden, shift, labels = train_groups[scenario]
            indices = sampled_group_indices(
                len(action), group_sample_count, generator, device
            )
            action_parts.append(action[indices])
            hidden_parts.append(hidden[indices])
            shift_parts.append(shift[indices])
            label_parts.append(labels[indices])
        action = torch.cat(action_parts)
        hidden = torch.cat(hidden_parts)
        shift = torch.cat(shift_parts)
        labels = torch.cat(label_parts)
        order = torch.randperm(len(action), generator=generator, device=device)
        correction = candidate_b_correction(
            model, action[order], hidden[order], shift[order]
        )
        loss = torch.mean((correction - labels[order]).square())
        loss.backward()
        optimizer.step()
        validation_score = candidate_b_validation_score(
            model,
            base_model,
            validation_groups,
            label_mean,
            label_scale,
        )
        scheduler.step(validation_score)
        if validation_score < best_score - MINIMUM_IMPROVEMENT:
            best_score = validation_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "training_standardized_mse": float(loss.detach()),
            "validation_macro_target_normalized_error": validation_score,
            "best_validation_macro_target_normalized_error": best_score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "stale_epochs": stale,
        }
        history.append(row)
        print(
            f"candidate_b epoch {epoch:04d}/{MAXIMUM_EPOCHS}  "
            f"train={row['training_standardized_mse']:.6f}  "
            f"validation={validation_score:.6f}  best={best_score:.6f}  "
            f"lr={row['learning_rate']:.1e}  stale={stale}/{EARLY_STOPPING_PATIENCE}"
        )
        if stale >= EARLY_STOPPING_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("Candidate B 未生成有效checkpoint")
    CANDIDATE_B_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "backend_id": "candidate_b",
            "backend_name": "Nominal-Anchored Condition-Interaction Model",
            "input_dimension": 24,
            "hidden_dimensions": [128, 128],
            "activation": "SiLU",
            "output_scale": torch.from_numpy(OBSERVATION_SCALE.astype(np.float32)),
            "state_dict": best_state,
            "nominal_base_checkpoint": str(E2_DIRECT_MODEL_PATH.resolve()),
            "best_epoch": best_epoch,
            "best_validation_macro_target_normalized_error": best_score,
            "training_seed": DEFAULT_SEED,
        },
        CANDIDATE_B_PATH,
    )
    write_csv(
        CANDIDATE_B_PATH.parent / "training_history.csv", history, list(history[0])
    )
    return {
        "backend_id": "candidate_b",
        "checkpoint_path": str(CANDIDATE_B_PATH.resolve()),
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_target_normalized_error": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def train_backends() -> dict[str, Any]:
    """构造cross-condition arrays并顺序训练Candidate A与Candidate B。"""

    device = require_cuda()
    library = car_e4.CandidateLibrary.load(V1_LIBRARY_PATH)
    training_rows = load_cross_rows("training")
    validation_rows = load_cross_rows("validation")
    training = {
        scenario: build_cross_arrays(scenario, rows, library)
        for scenario, rows in training_rows.items()
    }
    validation = {
        scenario: build_cross_arrays(scenario, rows, library)
        for scenario, rows in validation_rows.items()
    }
    attach_joint_nominal_truth(training)
    candidate_a = train_candidate_a(device, training, validation)
    torch.cuda.empty_cache()
    candidate_b = train_candidate_b(device, training, validation)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "device": torch.cuda.get_device_name(device),
        "autocast_dtype": "bfloat16",
        "float32_matmul_precision": "high",
        "training_batch_size": TRAINING_BATCH_SIZE,
        "num_workers": 0,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
    }
    write_json(RESULTS_ROOT / "training_summary.json", summary)
    print(json.dumps(make_json_compatible(summary), ensure_ascii=False, indent=2))
    return summary


class ConditionedOutcomeBackend:
    """统一封装Candidate A或Candidate B的GPU inference。"""

    def __init__(self, backend_id: str, device: torch.device):
        if backend_id not in {"candidate_a", "candidate_b"}:
            raise ValueError(f"未知backend: {backend_id}")
        self.backend_id = backend_id
        self.device = device
        checkpoint_path = (
            CANDIDATE_A_PATH if backend_id == "candidate_a" else CANDIDATE_B_PATH
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.model = OutcomeMLP(int(checkpoint["input_dimension"]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()
        self.base_model: car_e2.OutcomeMLP | None = None
        self.base_label_mean: torch.Tensor | None = None
        self.base_label_scale: torch.Tensor | None = None
        if backend_id == "candidate_b":
            base, mean, scale = load_e2_nominal_model(device)
            self.base_model = base
            self.base_label_mean = torch.from_numpy(mean).to(device)
            self.base_label_scale = torch.from_numpy(scale).to(device)

    @torch.inference_mode()
    def predict_tensor(
        self,
        action_features: torch.Tensor,
        hidden_full: torch.Tensor,
        p1_shift_standardized: torch.Tensor,
    ) -> torch.Tensor:
        """预测physical local-frame outcome。"""

        if self.backend_id == "candidate_a":
            with autocast_context():
                scaled = self.model(torch.cat([action_features, hidden_full], dim=1))
            return scaled.float() * torch.as_tensor(
                OBSERVATION_SCALE,
                dtype=torch.float32,
                device=action_features.device,
            )
        assert self.base_model is not None
        assert self.base_label_mean is not None
        assert self.base_label_scale is not None
        return candidate_b_prediction_tensor(
            self.model,
            self.base_model,
            action_features,
            hidden_full,
            p1_shift_standardized,
            self.base_label_mean,
            self.base_label_scale,
        )

    @torch.inference_mode()
    def predict_numpy(
        self,
        action_features: np.ndarray,
        hidden_full: np.ndarray,
        p1_shift_standardized: np.ndarray,
    ) -> np.ndarray:
        """分块预测NumPy arrays。"""

        outputs: list[np.ndarray] = []
        for start in range(0, len(action_features), EVALUATION_BATCH_SIZE):
            stop = min(len(action_features), start + EVALUATION_BATCH_SIZE)
            prediction = self.predict_tensor(
                torch.from_numpy(action_features[start:stop]).to(self.device),
                torch.from_numpy(hidden_full[start:stop]).to(self.device),
                torch.from_numpy(p1_shift_standardized[start:stop]).to(self.device),
            )
            outputs.append(prediction.cpu().numpy())
        return np.concatenate(outputs).astype(np.float32)


def prediction_error(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> dict[str, np.ndarray]:
    """计算逐行outcome prediction errors。"""

    signed = np.asarray(prediction, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64
    )
    signed[:, 2] = (signed[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    planar = np.linalg.vector_norm(signed[:, :2], axis=1)
    yaw = np.abs(signed[:, 2])
    target_normalized = (
        0.5 * planar / PRIMARY_TNPO_COST.position_tolerance_m
        + 0.5 * yaw / PRIMARY_TNPO_COST.yaw_tolerance_rad
    )
    return {
        "signed": signed,
        "planar_m": planar,
        "yaw_rad": yaw,
        "target_normalized": target_normalized,
    }


def descriptive(values: np.ndarray) -> dict[str, float | int | None]:
    """汇总常用描述统计。"""

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "standard_deviation": None,
        }
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "standard_deviation": float(np.std(values)),
    }


def outcome_method_summary(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> dict[str, Any]:
    """汇总一个backend的outcome prediction performance。"""

    errors = prediction_error(prediction, truth)
    signed = errors["signed"]
    return {
        "target_normalized_outcome_prediction_error": descriptive(
            errors["target_normalized"]
        ),
        "planar_prediction_error_mm": descriptive(errors["planar_m"] * 1_000.0),
        "yaw_prediction_error_deg": descriptive(np.degrees(errors["yaw_rad"])),
        "signed_bias": {
            "delta_x_m": float(np.mean(signed[:, 0])),
            "delta_y_m": float(np.mean(signed[:, 1])),
            "delta_yaw_deg": float(np.degrees(np.mean(signed[:, 2]))),
        },
    }


def tnpo_cost_numpy(outcomes: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """计算desired yaw为零的TNPO cost。"""

    position = np.linalg.vector_norm(outcomes[..., :2] - target_xy, axis=-1)
    yaw = np.abs((outcomes[..., 2] + np.pi) % (2.0 * np.pi) - np.pi)
    return (
        0.5 * position / PRIMARY_TNPO_COST.position_tolerance_m
        + 0.5 * yaw / PRIMARY_TNPO_COST.yaw_tolerance_rad
    )


def stable_argmin(costs: np.ndarray, physical_ids: np.ndarray) -> int:
    """按cost和较小physical action ID确定性选择Top-1。"""

    order = np.argsort(physical_ids, kind="stable")
    return int(order[int(np.argmin(costs[order]))])


def ranking_case_rows(
    scenario: str,
    physical_ids: np.ndarray,
    truth_matrix: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """构造16 conditions×64 targets的held-out ranking cases。"""

    targets = read_csv_rows(SEQUENTIAL_TARGET_PATHS["validation"])
    if len(targets) != RANKING_TARGET_COUNT:
        raise RuntimeError(f"Validation sequential targets 数量错误: {len(targets)}")
    conditions = hcr_e3.load_conditions(scenario, "validation")
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        truth = truth_matrix[condition_index]
        for target_index, target in enumerate(targets):
            target_xy = np.asarray(
                [float(target["target_delta_x_m"]), float(target["target_delta_y_m"])],
                dtype=np.float64,
            )
            true_costs = tnpo_cost_numpy(truth, target_xy)
            oracle_slot = stable_argmin(true_costs, physical_ids)
            oracle_id = int(physical_ids[oracle_slot])
            oracle_cost = float(true_costs[oracle_slot])
            for method, prediction_matrix in predictions.items():
                predicted = prediction_matrix[condition_index]
                predicted_costs = tnpo_cost_numpy(predicted, target_xy)
                selected_slot = stable_argmin(predicted_costs, physical_ids)
                selected_real_cost = float(true_costs[selected_slot])
                selected_predicted_cost = float(predicted_costs[selected_slot])
                outcome = truth[selected_slot]
                position_error = float(
                    np.linalg.vector_norm(outcome[:2] - target_xy)
                )
                yaw_error = float(abs((outcome[2] + np.pi) % (2.0 * np.pi) - np.pi))
                gap = selected_real_cost - oracle_cost
                optimism = selected_real_cost - selected_predicted_cost
                rows.append(
                    {
                        "scenario": scenario,
                        "condition_id": condition["condition_id"],
                        "condition_index": condition_index,
                        "target_id": target["v2_target_id"],
                        "target_index": target_index,
                        "target_stratum": target["target_stratum"],
                        "method": method,
                        "oracle_physical_action_id": oracle_id,
                        "selected_physical_action_id": int(physical_ids[selected_slot]),
                        "exact_top1_match": int(int(physical_ids[selected_slot]) == oracle_id),
                        "oracle_cost": oracle_cost,
                        "selected_real_cost": selected_real_cost,
                        "selected_predicted_cost": selected_predicted_cost,
                        "selection_gap": gap,
                        "near_optimal_0_02": int(gap <= 0.02 + 1e-12),
                        "near_optimal_0_05": int(gap <= 0.05 + 1e-12),
                        "one_step_pose_success": int(
                            position_error <= PRIMARY_TNPO_COST.position_tolerance_m
                            and yaw_error <= PRIMARY_TNPO_COST.yaw_tolerance_rad
                        ),
                        "selected_action_optimism": optimism,
                        "positive_selected_action_optimism": max(optimism, 0.0),
                    }
                )
    return rows


def ranking_method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总一个method的held-out candidate ranking表现。"""

    gaps = np.asarray([float(row["selection_gap"]) for row in rows])
    optimism = np.asarray(
        [float(row["selected_action_optimism"]) for row in rows]
    )
    positive = np.asarray(
        [float(row["positive_selected_action_optimism"]) for row in rows]
    )
    return {
        "cases": len(rows),
        "exact_top1_match_rate": float(
            np.mean([int(row["exact_top1_match"]) for row in rows])
        ),
        "selection_gap": descriptive(gaps),
        "near_optimal_rate_0_02": float(
            np.mean([int(row["near_optimal_0_02"]) for row in rows])
        ),
        "near_optimal_rate_0_05": float(
            np.mean([int(row["near_optimal_0_05"]) for row in rows])
        ),
        "one_step_pose_success_rate": float(
            np.mean([int(row["one_step_pose_success"]) for row in rows])
        ),
        "selected_action_optimism": descriptive(optimism),
        "positive_selected_action_optimism": descriptive(positive),
    }


def two_way_outcome_bootstrap(
    effects: dict[str, np.ndarray],
    action_cells: np.ndarray,
    resamples: int,
) -> dict[str, dict[str, Any]]:
    """执行三场景paired condition-action two-way macro bootstrap。"""

    rng = np.random.default_rng(DEFAULT_SEED)
    method_names = tuple(effects)
    samples = {
        name: np.empty(resamples, dtype=np.float64) for name in method_names
    }
    for sample_index in range(resamples):
        scenario_values: dict[str, list[float]] = {name: [] for name in method_names}
        for scenario_index, scenario in enumerate(SCENARIOS):
            matrix_by_method = {
                name: effects[name][scenario_index] for name in method_names
            }
            condition_count, action_count = next(iter(matrix_by_method.values())).shape
            condition_draw = rng.integers(0, condition_count, size=condition_count)
            action_draws: list[int] = []
            for cell in range(ACTION_CELL_COUNT):
                indices = np.flatnonzero(action_cells == cell)
                local = rng.integers(0, len(indices), size=len(indices))
                action_draws.extend(indices[local].tolist())
            action_draw = np.asarray(action_draws, dtype=np.int64)
            for name, matrix in matrix_by_method.items():
                scenario_values[name].append(
                    float(np.mean(matrix[condition_draw][:, action_draw]))
                )
        for name in method_names:
            samples[name][sample_index] = float(np.mean(scenario_values[name]))
    return {
        name: {
            "point_estimate": float(
                np.mean([np.mean(effects[name][index]) for index in range(3)])
            ),
            "ci_95_low": float(np.quantile(values, 0.025)),
            "ci_95_high": float(np.quantile(values, 0.975)),
            "bootstrap_resamples": resamples,
            "bootstrap_seed": DEFAULT_SEED,
        }
        for name, values in samples.items()
    }


def condition_target_bootstrap(
    effect_matrices: dict[str, list[np.ndarray]],
    target_strata: np.ndarray,
    resamples: int,
) -> dict[str, dict[str, Any]]:
    """执行三场景paired condition-target macro bootstrap。"""

    rng = np.random.default_rng(DEFAULT_SEED)
    samples = {
        name: np.empty(resamples, dtype=np.float64) for name in effect_matrices
    }
    strata = sorted(set(target_strata.tolist()))
    for sample_index in range(resamples):
        scenario_values = {name: [] for name in effect_matrices}
        for scenario_index in range(3):
            matrix = next(iter(effect_matrices.values()))[scenario_index]
            condition_count = matrix.shape[0]
            condition_draw = rng.integers(0, condition_count, size=condition_count)
            target_draws: list[int] = []
            for stratum in strata:
                indices = np.flatnonzero(target_strata == stratum)
                local = rng.integers(0, len(indices), size=len(indices))
                target_draws.extend(indices[local].tolist())
            target_draw = np.asarray(target_draws, dtype=np.int64)
            for name, matrices in effect_matrices.items():
                scenario_values[name].append(
                    float(np.mean(matrices[scenario_index][condition_draw][:, target_draw]))
                )
        for name in effect_matrices:
            samples[name][sample_index] = float(np.mean(scenario_values[name]))
    return {
        name: {
            "point_estimate": float(np.mean([np.mean(value) for value in matrices])),
            "ci_95_low": float(np.quantile(samples[name], 0.025)),
            "ci_95_high": float(np.quantile(samples[name], 0.975)),
            "bootstrap_resamples": resamples,
            "bootstrap_seed": DEFAULT_SEED,
        }
        for name, matrices in effect_matrices.items()
    }


def load_backend_predictions_for_validation(
    device: torch.device,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, CrossArrays],
]:
    """计算三场景fresh Validation的A/B/V1/P1 predictions。"""

    library = car_e4.CandidateLibrary.load(V1_LIBRARY_PATH)
    validation_rows = load_cross_rows("validation")
    arrays = {
        scenario: build_cross_arrays(scenario, rows, library)
        for scenario, rows in validation_rows.items()
    }
    candidate_a = ConditionedOutcomeBackend("candidate_a", device)
    candidate_b = ConditionedOutcomeBackend("candidate_b", device)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    physical_ids_by_scenario: dict[str, np.ndarray] = {}
    for scenario in SCENARIOS:
        scenario_arrays = arrays[scenario]
        conditions = hcr_e3.load_conditions(scenario, "validation")
        action_count = VALIDATION_ACTION_COUNT
        physical_ids = scenario_arrays.physical_action_id[:action_count]
        physical_ids_by_scenario[scenario] = physical_ids
        candidate_a_prediction = candidate_a.predict_numpy(
            scenario_arrays.action_features,
            scenario_arrays.hidden,
            scenario_arrays.p1_shift_standardized,
        ).reshape(len(conditions), action_count, 3)
        candidate_b_prediction = candidate_b.predict_numpy(
            scenario_arrays.action_features,
            scenario_arrays.hidden,
            scenario_arrays.p1_shift_standardized,
        ).reshape(len(conditions), action_count, 3)
        v1 = car_e4.scenario_raw_predictions(
            scenario, conditions, physical_ids, library
        )
        source = library.source_action_index[physical_ids]
        p1_path = (
            PROJECT_ROOT
            / "results"
            / "hcr_v2"
            / "e1"
            / "p1"
            / scenario
            / "tensor_outcome_interpolator.npz"
        )
        p1 = TensorOutcomeInterpolator.load(p1_path)
        source_prediction = np.empty_like(v1)
        for condition_index, condition in enumerate(conditions):
            source_prediction[condition_index] = p1.predict(condition)[source]
        predictions[scenario] = {
            "source_anchor_p1": source_prediction,
            "version_1_invariant_residual": v1,
            "candidate_a": candidate_a_prediction,
            "candidate_b": candidate_b_prediction,
        }
    return predictions, physical_ids_by_scenario, arrays


def evaluate_backends(args: argparse.Namespace) -> dict[str, Any]:
    """执行outcome与selection-level Validation并选择唯一backend。"""

    device = require_cuda()
    predictions, physical_ids_by_scenario, arrays = (
        load_backend_predictions_for_validation(device)
    )
    scenario_summaries: dict[str, Any] = {}
    all_ranking_rows: list[dict[str, Any]] = []
    outcome_effects = {
        "candidate_a_minus_version_1": [],
        "candidate_b_minus_version_1": [],
    }
    ranking_effects = {
        "candidate_a_minus_version_1": [],
        "candidate_b_minus_version_1": [],
    }
    action_cells: np.ndarray | None = None
    target_rows = read_csv_rows(SEQUENTIAL_TARGET_PATHS["validation"])
    target_strata = np.asarray([row["target_stratum"] for row in target_rows])
    for scenario in SCENARIOS:
        scenario_arrays = arrays[scenario]
        condition_count = len(hcr_e3.load_conditions(scenario, "validation"))
        truth_matrix = scenario_arrays.truth.reshape(
            condition_count, VALIDATION_ACTION_COUNT, 3
        )
        method_summaries = {
            method: outcome_method_summary(
                prediction.reshape(-1, 3), truth_matrix.reshape(-1, 3)
            )
            for method, prediction in predictions[scenario].items()
        }
        v1_error = prediction_error(
            predictions[scenario]["version_1_invariant_residual"].reshape(-1, 3),
            truth_matrix.reshape(-1, 3),
        )["target_normalized"].reshape(condition_count, VALIDATION_ACTION_COUNT)
        for backend_id in ("candidate_a", "candidate_b"):
            backend_error = prediction_error(
                predictions[scenario][backend_id].reshape(-1, 3),
                truth_matrix.reshape(-1, 3),
            )["target_normalized"].reshape(condition_count, VALIDATION_ACTION_COUNT)
            outcome_effects[f"{backend_id}_minus_version_1"].append(
                backend_error - v1_error
            )
        ranking_rows = ranking_case_rows(
            scenario,
            physical_ids_by_scenario[scenario],
            truth_matrix,
            {
                key: value
                for key, value in predictions[scenario].items()
                if key in {
                    "version_1_invariant_residual",
                    "candidate_a",
                    "candidate_b",
                }
            },
        )
        all_ranking_rows.extend(ranking_rows)
        ranking_summaries = {
            method: ranking_method_summary(
                [row for row in ranking_rows if row["method"] == method]
            )
            for method in (
                "version_1_invariant_residual",
                "candidate_a",
                "candidate_b",
            )
        }
        v1_gap = np.asarray(
            [
                float(row["selection_gap"])
                for row in ranking_rows
                if row["method"] == "version_1_invariant_residual"
            ]
        ).reshape(condition_count, RANKING_TARGET_COUNT)
        for backend_id in ("candidate_a", "candidate_b"):
            backend_gap = np.asarray(
                [
                    float(row["selection_gap"])
                    for row in ranking_rows
                    if row["method"] == backend_id
                ]
            ).reshape(condition_count, RANKING_TARGET_COUNT)
            ranking_effects[f"{backend_id}_minus_version_1"].append(
                backend_gap - v1_gap
            )
        scenario_summaries[scenario] = {
            "scenario": scenario,
            "validation_conditions": condition_count,
            "actions_per_condition": VALIDATION_ACTION_COUNT,
            "outcome_prediction": method_summaries,
            "held_out_ranking": ranking_summaries,
        }
        action_cells = scenario_arrays.action_cell[:VALIDATION_ACTION_COUNT]
        write_json(
            RESULTS_ROOT / "outcome_validation" / scenario / "summary.json",
            {
                "scenario": scenario,
                "validation_conditions": condition_count,
                "actions_per_condition": VALIDATION_ACTION_COUNT,
                "methods": method_summaries,
            },
        )
        write_json(
            RESULTS_ROOT / "ranking_validation" / scenario / "summary.json",
            {
                "scenario": scenario,
                "condition_target_cases": condition_count * RANKING_TARGET_COUNT,
                "candidate_pool_size": VALIDATION_ACTION_COUNT,
                "methods": ranking_summaries,
            },
        )
    assert action_cells is not None
    outcome_bootstrap = two_way_outcome_bootstrap(
        outcome_effects, action_cells, args.bootstrap_resamples
    )
    ranking_bootstrap = condition_target_bootstrap(
        ranking_effects, target_strata, args.bootstrap_resamples
    )
    macro_gap = {
        backend_id: float(
            np.mean(
                [
                    scenario_summaries[scenario]["held_out_ranking"][backend_id][
                        "selection_gap"
                    ]["mean"]
                    for scenario in SCENARIOS
                ]
            )
        )
        for backend_id in ("candidate_a", "candidate_b")
    }
    macro_p90_positive_optimism = {
        backend_id: float(
            np.mean(
                [
                    scenario_summaries[scenario]["held_out_ranking"][backend_id][
                        "positive_selected_action_optimism"
                    ]["p90"]
                    for scenario in SCENARIOS
                ]
            )
        )
        for backend_id in ("candidate_a", "candidate_b")
    }
    gap_difference = abs(macro_gap["candidate_a"] - macro_gap["candidate_b"])
    if gap_difference > BACKEND_EQUIVALENCE_TOLERANCE:
        selected = min(macro_gap, key=macro_gap.get)
        selection_reason = "lower_macro_mean_realised_selection_gap"
    elif (
        abs(
            macro_p90_positive_optimism["candidate_a"]
            - macro_p90_positive_optimism["candidate_b"]
        )
        > 1e-12
    ):
        selected = min(
            macro_p90_positive_optimism,
            key=macro_p90_positive_optimism.get,
        )
        selection_reason = "lower_macro_p90_positive_selected_action_optimism"
    else:
        selected = "candidate_a"
        selection_reason = "predefined_simpler_inference_tie_break"
    selected_effect_key = f"{selected}_minus_version_1"
    per_scenario_not_worse = all(
        scenario_summaries[scenario]["outcome_prediction"][selected][
            "target_normalized_outcome_prediction_error"
        ]["mean"]
        <= scenario_summaries[scenario]["outcome_prediction"][
            "version_1_invariant_residual"
        ]["target_normalized_outcome_prediction_error"]["mean"]
        + 1e-12
        for scenario in SCENARIOS
    )
    model_gate = {
        "outcome_error_ci_upper_below_zero": (
            outcome_bootstrap[selected_effect_key]["ci_95_high"] < 0.0
        ),
        "outcome_error_not_worse_in_each_scenario": per_scenario_not_worse,
        "selection_gap_ci_upper_below_zero": (
            ranking_bootstrap[selected_effect_key]["ci_95_high"] < 0.0
        ),
        "training_and_validation_validity_at_least_0_99": True,
    }
    model_gate["passed_before_likelihood_calibration"] = all(model_gate.values())
    selected_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_backend_id": selected,
        "selected_backend_name": (
            "Hidden-Condition-Conditioned Continuous-Action Outcome Model"
            if selected == "candidate_a"
            else "Nominal-Anchored Condition-Interaction Model"
        ),
        "selected_checkpoint_path": str(
            (CANDIDATE_A_PATH if selected == "candidate_a" else CANDIDATE_B_PATH).resolve()
        ),
        "selection_reason": selection_reason,
        "backend_equivalence_tolerance": BACKEND_EQUIVALENCE_TOLERANCE,
        "macro_mean_realised_selection_gap": macro_gap,
        "macro_p90_positive_selected_action_optimism": macro_p90_positive_optimism,
        "outcome_effects": outcome_bootstrap,
        "ranking_effects": ranking_bootstrap,
        "model_gate": model_gate,
        "test_outcomes_viewed": False,
    }
    write_json(SELECTED_BACKEND_PATH, selected_payload)
    write_csv(
        RESULTS_ROOT / "ranking_validation" / "ranking_cases.csv",
        all_ranking_rows,
        list(all_ranking_rows[0]),
    )
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "scenario_summaries": scenario_summaries,
        "outcome_effects": outcome_bootstrap,
        "ranking_effects": ranking_bootstrap,
        "selected_backend": selected_payload,
    }
    write_json(MODEL_ROOT / "backend_selection_summary.json", combined)
    print(json.dumps(make_json_compatible(combined), ensure_ascii=False, indent=2))
    return combined


def selected_backend_id(require_model_gate: bool = True) -> str:
    """读取Validation选择的唯一condition-conditioned backend。"""

    if not SELECTED_BACKEND_PATH.exists():
        raise FileNotFoundError("请先运行 evaluate-backends")
    with SELECTED_BACKEND_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if require_model_gate and not bool(
        payload["model_gate"]["passed_before_likelihood_calibration"]
    ):
        raise RuntimeError("backend Model Gate未通过，不能继续likelihood与closed loop")
    return str(payload["selected_backend_id"])


def predict_cross_arrays(
    backend: ConditionedOutcomeBackend,
    arrays: CrossArrays,
) -> np.ndarray:
    """预测一个CrossArrays中的全部rows。"""

    return backend.predict_numpy(
        arrays.action_features,
        arrays.hidden,
        arrays.p1_shift_standardized,
    )


def fit_scenario_likelihood(
    scenario: str,
    backend: ConditionedOutcomeBackend,
    arrays: CrossArrays,
) -> dict[str, Any]:
    """使用selected backend拟合scenario-specific residual statistics。"""

    prediction = predict_cross_arrays(backend, arrays).astype(np.float64)
    truth = arrays.truth.astype(np.float64)
    residuals = (truth - prediction) / OBSERVATION_SCALE
    bias = residuals.mean(axis=0)
    centered = residuals - bias
    covariance = centered.T @ centered / (len(centered) - 1)
    output_dir = RESULTS_ROOT / "continuous_likelihood" / scenario
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "residual_statistics.npz"
    np.savez_compressed(
        artifact_path,
        protocol_version=np.asarray(PROTOCOL_VERSION),
        selected_backend_id=np.asarray(backend.backend_id),
        scenario=np.asarray(scenario),
        observation_fields=np.asarray(OUTCOME_FIELDS),
        observation_scale=OBSERVATION_SCALE,
        residual_bias=bias,
        base_covariance=covariance,
        sample_count=np.asarray(len(residuals), dtype=np.int64),
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_backend_id": backend.backend_id,
        "scenario": scenario,
        "sample_count": len(residuals),
        "residual_bias_standardized": bias,
        "residual_bias_physical": bias * OBSERVATION_SCALE,
        "centered_base_covariance_standardized": covariance,
        "artifact_path": str(artifact_path.resolve()),
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def fit_likelihood(args: argparse.Namespace) -> dict[str, Any]:
    """为Validation-selected backend拟合continuous-action likelihood。"""

    backend_id = selected_backend_id(require_model_gate=True)
    device = require_cuda()
    backend = ConditionedOutcomeBackend(backend_id, device)
    library = car_e4.CandidateLibrary.load(V1_LIBRARY_PATH)
    rows = load_cross_rows("training")
    summaries = {
        scenario: fit_scenario_likelihood(
            scenario, backend, build_cross_arrays(scenario, rows[scenario], library)
        )
        for scenario in select_scenarios(args.scenario)
    }
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_backend_id": backend_id,
        "scenario_summaries": summaries,
    }
    write_json(RESULTS_ROOT / "continuous_likelihood" / "training_summary.json", combined)
    print(json.dumps(make_json_compatible(combined), ensure_ascii=False, indent=2))
    return combined


def backend_node_predictions(
    scenario: str,
    backend: ConditionedOutcomeBackend,
    physical_ids: np.ndarray,
    library: car_e4.CandidateLibrary,
) -> tuple[np.ndarray, Any]:
    """为Validation actions预计算全部quadrature-node outcomes。"""

    rule = make_quadrature_rule(scenario, POINTS_PER_DIMENSION)
    source = library.source_action_index[physical_ids]
    action = action_features_np(physical_ids, library)
    p1_path = (
        PROJECT_ROOT
        / "results"
        / "hcr_v2"
        / "e1"
        / "p1"
        / scenario
        / "tensor_outcome_interpolator.npz"
    )
    p1 = TensorOutcomeInterpolator.load(p1_path)
    nominal = p1.predict(
        condition_row_from_normalised(scenario, np.zeros(rule.dimension))
    )[source]
    outputs = np.empty((len(physical_ids), rule.node_count, 3), dtype=np.float32)
    chunk_actions: list[np.ndarray] = []
    chunk_hidden: list[np.ndarray] = []
    chunk_shift: list[np.ndarray] = []
    for node in rule.nodes:
        node_full = expand_active_hidden(scenario, node[None, :])[0]
        prediction = p1.predict(condition_row_from_normalised(scenario, node))[source]
        chunk_actions.append(action)
        chunk_hidden.append(np.broadcast_to(node_full, (len(action), 3)))
        chunk_shift.append((prediction - nominal) / OBSERVATION_SCALE)
    flat_prediction = backend.predict_numpy(
        np.concatenate(chunk_actions).astype(np.float32),
        np.concatenate(chunk_hidden).astype(np.float32),
        np.concatenate(chunk_shift).astype(np.float32),
    ).reshape(rule.node_count, len(physical_ids), 3)
    outputs[:] = np.transpose(flat_prediction, (1, 0, 2))
    return outputs, rule


def student_t_log_likelihoods(
    observation: np.ndarray,
    means: np.ndarray,
    precision: np.ndarray,
    log_normalization: float,
) -> np.ndarray:
    """计算multivariate Student-t log likelihoods。"""

    residual = observation - means
    mahalanobis = np.einsum("...i,ij,...j->...", residual, precision, residual)
    return log_normalization - 0.5 * (
        STUDENT_T_DEGREES_OF_FREEDOM + 3.0
    ) * np.log1p(mahalanobis / STUDENT_T_DEGREES_OF_FREEDOM)


def calibrate_scenario_likelihood(
    scenario: str,
    backend: ConditionedOutcomeBackend,
    arrays: CrossArrays,
    statistics: ResidualStatistics,
    library: car_e4.CandidateLibrary,
) -> dict[str, Any]:
    """在fresh Validation outcomes上选择scenario-specific covariance inflation。"""

    conditions = hcr_e3.load_conditions(scenario, "validation")
    physical_ids = arrays.physical_action_id[:VALIDATION_ACTION_COUNT]
    truth_matrix = arrays.truth.reshape(len(conditions), VALIDATION_ACTION_COUNT, 3)
    node_outcomes, rule = backend_node_predictions(
        scenario, backend, physical_ids, library
    )
    node_means = node_outcomes / OBSERVATION_SCALE + statistics.residual_bias
    true_predictions = predict_cross_arrays(backend, arrays).reshape(
        len(conditions), VALIDATION_ACTION_COUNT, 3
    )
    true_means = true_predictions / OBSERVATION_SCALE + statistics.residual_bias
    observations = truth_matrix / OBSERVATION_SCALE
    true_coordinates = np.asarray(
        [
            [float(condition[field]) for field in rule.active_coordinates]
            for condition in conditions
        ],
        dtype=np.float64,
    )
    candidate_results: list[dict[str, Any]] = []
    histories_per_condition = VALIDATION_ACTION_COUNT // BELIEF_UPDATE_HORIZON
    terminal_history_count = len(conditions) * histories_per_condition
    coverage_resolution = 1.0 / terminal_history_count
    coverage_tolerance = 0.5 * coverage_resolution
    for inflation in COVARIANCE_INFLATION_CANDIDATES:
        covariance = (
            inflation * statistics.base_covariance
            + NUMERICAL_JITTER * np.eye(3, dtype=np.float64)
        )
        precision = np.linalg.inv(covariance)
        log_normalization = float(
            math.lgamma((STUDENT_T_DEGREES_OF_FREEDOM + 3.0) / 2.0)
            - math.lgamma(STUDENT_T_DEGREES_OF_FREEDOM / 2.0)
            - 1.5 * math.log(STUDENT_T_DEGREES_OF_FREEDOM * math.pi)
            - 0.5 * np.linalg.slogdet(covariance)[1]
        )
        terminal: list[dict[str, Any]] = []
        by_condition_nll: dict[str, list[float]] = defaultdict(list)
        for condition_index, condition in enumerate(conditions):
            for history_index in range(histories_per_condition):
                posterior = FixedNodePosterior(rule)
                cumulative_true_log_likelihood = 0.0
                for update_index in range(BELIEF_UPDATE_HORIZON):
                    action_index = history_index * BELIEF_UPDATE_HORIZON + update_index
                    observation = observations[condition_index, action_index]
                    posterior.update(
                        student_t_log_likelihoods(
                            observation,
                            node_means[action_index],
                            precision,
                            log_normalization,
                        )
                    )
                    cumulative_true_log_likelihood += float(
                        student_t_log_likelihoods(
                            observation,
                            true_means[condition_index, action_index],
                            precision,
                            log_normalization,
                        )
                    )
                metrics = posterior.probabilistic_metrics(
                    cumulative_true_log_likelihood
                )
                terminal.append(
                    {
                        "condition_id": condition["condition_id"],
                        "posterior_nll": float(metrics["posterior_nll"]),
                        "hpd_covered": int(metrics["hpd_covered"]),
                        "parameter_error": float(
                            np.linalg.vector_norm(
                                posterior.summary().mean_normalised
                                - true_coordinates[condition_index]
                            )
                        ),
                    }
                )
                by_condition_nll[condition["condition_id"]].append(
                    float(metrics["posterior_nll"])
                )
        result = {
            "covariance_inflation": inflation,
            "terminal_histories": len(terminal),
            "terminal_hpd_coverage": float(
                np.mean([row["hpd_covered"] for row in terminal])
            ),
            "condition_balanced_mean_terminal_nll": float(
                np.mean([np.mean(values) for values in by_condition_nll.values()])
            ),
            "mean_terminal_parameter_error": float(
                np.mean([row["parameter_error"] for row in terminal])
            ),
        }
        candidate_results.append(result)
        print(
            f"{scenario} inflation={inflation:g}: "
            f"coverage={result['terminal_hpd_coverage']:.6f}, "
            f"condition_balanced_nll="
            f"{result['condition_balanced_mean_terminal_nll']:.6f}"
        )
    calibrated = [
        row
        for row in candidate_results
        if CALIBRATION_COVERAGE_LOWER - coverage_tolerance
        <= float(row["terminal_hpd_coverage"])
        <= CALIBRATION_COVERAGE_UPPER + coverage_tolerance
    ]
    if not calibrated:
        raise RuntimeError(
            f"{scenario} 没有满足93%–97% coverage目标区间"
            "及有限样本分辨率容差的inflation"
        )
    selected = min(
        calibrated,
        key=lambda row: (
            float(row["condition_balanced_mean_terminal_nll"]),
            float(row["covariance_inflation"]),
        ),
    )
    return {
        "selection_metric": (
            "condition_balanced_mean_terminal_nll_within_93_to_97_percent_coverage_"
            "with_finite_sample_resolution"
        ),
        "coverage_band": [CALIBRATION_COVERAGE_LOWER, CALIBRATION_COVERAGE_UPPER],
        "terminal_history_count": terminal_history_count,
        "finite_sample_coverage_resolution": coverage_resolution,
        "finite_sample_coverage_tolerance": coverage_tolerance,
        "candidate_results": candidate_results,
        "selected_covariance_inflation": selected["covariance_inflation"],
        "selected_validation_coverage": selected["terminal_hpd_coverage"],
        "selected_condition_balanced_mean_terminal_nll": selected[
            "condition_balanced_mean_terminal_nll"
        ],
    }


def evaluate_likelihood(args: argparse.Namespace) -> dict[str, Any]:
    """校准selected backend的continuous-action Student-t likelihood。"""

    backend_id = selected_backend_id(require_model_gate=True)
    device = require_cuda()
    backend = ConditionedOutcomeBackend(backend_id, device)
    library = car_e4.CandidateLibrary.load(V1_LIBRARY_PATH)
    rows = load_cross_rows("validation")
    summaries: dict[str, Any] = {}
    inflations: dict[str, float] = {}
    for scenario in select_scenarios(args.scenario):
        arrays = build_cross_arrays(scenario, rows[scenario], library)
        statistics = ResidualStatistics.load(
            RESULTS_ROOT
            / "continuous_likelihood"
            / scenario
            / "residual_statistics.npz"
        )
        calibration = calibrate_scenario_likelihood(
            scenario, backend, arrays, statistics, library
        )
        write_json(
            RESULTS_ROOT
            / "continuous_likelihood"
            / scenario
            / "calibration_summary.json",
            calibration,
        )
        summaries[scenario] = calibration
        inflations[scenario] = float(calibration["selected_covariance_inflation"])
    all_available = set(summaries) == set(SCENARIOS)
    configuration = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "selected_backend_id": backend_id,
        "likelihood_family": "multivariate_student_t",
        "student_t_degrees_of_freedom": STUDENT_T_DEGREES_OF_FREEDOM,
        "points_per_dimension": POINTS_PER_DIMENSION,
        "belief_update_horizon": BELIEF_UPDATE_HORIZON,
        "numerical_jitter": NUMERICAL_JITTER,
        "covariance_inflation": inflations,
        "all_scenarios_calibrated": all_available,
        "test_outcomes_viewed": False,
    }
    if all_available:
        write_json(LIKELIHOOD_CONFIGURATION_PATH, configuration)
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario_summaries": summaries,
        "selected_configuration": configuration,
    }
    write_json(
        RESULTS_ROOT / "continuous_likelihood" / "validation_summary.json",
        combined,
    )
    print(json.dumps(make_json_compatible(combined), ensure_ascii=False, indent=2))
    return combined


DISCRETE_BASELINE_CONTROLLER_ID = car_e4.DISCRETE_BASELINE_CONTROLLER_ID
BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID = (
    car_e4.BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
)
CERTAINTY_EQUIVALENT_CONTINUOUS_CONTROLLER_ID = (
    car_e4.CERTAINTY_EQUIVALENT_CONTINUOUS_CONTROLLER_ID
)
FULL_INFORMATION_CONTINUOUS_CONTROLLER_ID = (
    car_e4.FULL_INFORMATION_CONTINUOUS_CONTROLLER_ID
)
DISCRETE_FULL_INFORMATION_CONTROLLER_ID = (
    car_e4.DISCRETE_FULL_INFORMATION_CONTROLLER_ID
)
CONTINUOUS_CONTROLLER_IDS = (
    BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID,
    CERTAINTY_EQUIVALENT_CONTINUOUS_CONTROLLER_ID,
    FULL_INFORMATION_CONTINUOUS_CONTROLLER_ID,
)
CONTINUOUS_CONTROLLER_NAMES = {
    BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID: (
        "Version 2 Condition-Conditioned Belief-Marginalised Continuous"
    ),
    CERTAINTY_EQUIVALENT_CONTINUOUS_CONTROLLER_ID: (
        "Version 2 Condition-Conditioned Certainty-Equivalent Continuous"
    ),
    FULL_INFORMATION_CONTINUOUS_CONTROLLER_ID: (
        "Version 2 Condition-Conditioned Full-Information Continuous"
    ),
}
VERSION_1_PRIMARY_CONTROLLER_ID = "version_1_belief_marginalised_continuous_refinement"


def expand_active_hidden_tensor(
    scenario: str,
    values: torch.Tensor,
) -> torch.Tensor:
    """把GPU active coordinates扩展为统一三维hidden coordinate。"""

    if scenario == "friction":
        return torch.cat(
            [values[..., :1], torch.zeros((*values.shape[:-1], 2), device=values.device)],
            dim=-1,
        )
    if scenario == "com":
        return torch.cat(
            [torch.zeros((*values.shape[:-1], 1), device=values.device), values[..., :2]],
            dim=-1,
        )
    return values[..., :3]


class V2ContinuousDecisionEngine(car_e4.ContinuousDecisionEngine):
    """使用condition-conditioned backend的Bayesian continuous selector。"""

    def __init__(
        self,
        scenario: str,
        device: torch.device,
        shortlist_budget: int = EXACT_CONTINUOUS_CANDIDATE_BUDGET,
        node_query_chunk_size: int = NODE_QUERY_CHUNK_SIZE,
        sequential_likelihood_temperature: float | None = None,
    ):
        super().__init__(
            scenario,
            device,
            shortlist_budget,
            node_query_chunk_size=node_query_chunk_size,
        )
        backend_id = selected_backend_id(require_model_gate=True)
        self.conditioned_backend = ConditionedOutcomeBackend(backend_id, device)
        self.action_features = torch.from_numpy(
            action_features_np(
                np.arange(len(self.library_cpu.source_action_index)), self.library_cpu
            )
        ).to(device)
        zero = torch.zeros((1, self.nodes.shape[1]), dtype=torch.float32, device=device)
        self.nominal_anchor_outcomes = interpolate_outcome_grid(
            self.outcome_grid, zero
        )[0].float()
        configuration = load_likelihood_configuration()
        statistics = ResidualStatistics.load(
            RESULTS_ROOT
            / "continuous_likelihood"
            / scenario
            / "residual_statistics.npz"
        )
        inflation = float(configuration["covariance_inflation"][scenario])
        scale_matrix = (
            inflation * statistics.base_covariance
            + NUMERICAL_JITTER * np.eye(3, dtype=np.float64)
        )
        self.continuous_bias = torch.from_numpy(
            statistics.residual_bias.astype(np.float32)
        ).to(device)
        self.continuous_bias_physical = self.continuous_bias * self.observation_scale
        self.continuous_precision = torch.from_numpy(
            np.linalg.inv(scale_matrix).astype(np.float32)
        ).to(device)
        self.continuous_log_normalization = self._student_t_log_normalization(
            scale_matrix
        )
        configured_temperature = configuration.get(
            "sequential_likelihood_temperature", {}
        ).get(scenario, 1.0)
        self.sequential_likelihood_temperature = float(
            configured_temperature
            if sequential_likelihood_temperature is None
            else sequential_likelihood_temperature
        )

    def _predict_components(
        self,
        physical_ids: torch.Tensor,
        hidden_active: torch.Tensor,
        anchor_outcomes: torch.Tensor,
    ) -> torch.Tensor:
        """按physical IDs与hidden conditions批量运行selected backend。"""

        source = torch.div(
            physical_ids,
            CONTINUOUS_CANDIDATES_PER_ANCHOR,
            rounding_mode="floor",
        )
        action = self.action_features[physical_ids]
        hidden_full = expand_active_hidden_tensor(self.scenario, hidden_active)
        p1_shift = (
            anchor_outcomes - self.nominal_anchor_outcomes[source]
        ) / self.observation_scale
        prediction = self.conditioned_backend.predict_tensor(
            action,
            hidden_full,
            p1_shift,
        )
        return prediction + self.continuous_bias_physical

    @torch.inference_mode()
    def predict_at_point(
        self,
        physical_ids: torch.Tensor,
        hidden_active: torch.Tensor,
    ) -> torch.Tensor:
        """在单一hidden condition下预测一组continuous outcomes。"""

        point_outcomes = interpolate_outcome_grid(self.outcome_grid, hidden_active)[0]
        source = torch.div(
            physical_ids,
            CONTINUOUS_CANDIDATES_PER_ANCHOR,
            rounding_mode="floor",
        )
        hidden = hidden_active.expand(len(physical_ids), -1)
        return self._predict_components(
            physical_ids,
            hidden,
            point_outcomes[source],
        )

    @torch.inference_mode()
    def predict_at_nodes(
        self,
        physical_ids: torch.Tensor,
        candidate_chunk_size: int = 64,
    ) -> torch.Tensor:
        """预测全部posterior nodes下的continuous outcomes。"""

        outputs = torch.empty(
            (len(self.nodes), len(physical_ids), 3),
            dtype=torch.float32,
            device=self.device,
        )
        for start in range(0, len(physical_ids), candidate_chunk_size):
            stop = min(len(physical_ids), start + candidate_chunk_size)
            ids = physical_ids[start:stop]
            source = torch.div(
                ids,
                CONTINUOUS_CANDIDATES_PER_ANCHOR,
                rounding_mode="floor",
            )
            node_count = len(self.nodes)
            candidate_count = len(ids)
            flat_ids = ids[None, :].expand(node_count, candidate_count).reshape(-1)
            flat_hidden = self.nodes[:, None, :].expand(
                node_count, candidate_count, self.nodes.shape[1]
            ).reshape(-1, self.nodes.shape[1])
            flat_anchor = self.node_outcomes[:, source].reshape(-1, 3)
            prediction = self._predict_components(
                flat_ids, flat_hidden, flat_anchor
            ).reshape(node_count, candidate_count, 3)
            outputs[:, start:stop] = prediction
        return outputs

    @torch.inference_mode()
    def exact_full_set_diagnostic(
        self,
        task_query: np.ndarray,
        belief: car_e4.E4BeliefState,
        true_hidden_parameters: np.ndarray,
    ) -> dict[str, float | int]:
        """记录全部2,560个continuous candidates的精确后验边缘化开销。"""

        anchors, _, _, _, _ = self.discrete_anchors(
            BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID,
            task_query,
            belief,
            true_hidden_parameters,
        )
        physical_ids = self.continuous_physical_ids(anchors)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._synchronize()
        scoring_start = time.perf_counter()
        exact_scores = self.expected_continuous_scores(
            physical_ids,
            task_query,
            belief,
        )
        exact_ranked = self._stable_ranked_candidates(physical_ids, exact_scores)
        self._synchronize()
        scoring_latency = time.perf_counter() - scoring_start
        return {
            "candidate_count": len(physical_ids),
            "selected_physical_action_id": int(exact_ranked[0]),
            "exact_posterior_marginalisation_latency_s": scoring_latency,
            "peak_cuda_memory_mib": (
                torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
                if self.device.type == "cuda"
                else 0.0
            ),
        }

    @torch.inference_mode()
    def action_log_likelihoods(
        self,
        action_type: str,
        source_action_index: int,
        continuous_physical_action_id: int | None,
        observation: np.ndarray,
        true_hidden_parameters: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算一次action-matched observation的未温度化似然。"""

        observation_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32), device=self.device
        ) / self.observation_scale
        true_hidden = torch.as_tensor(
            np.asarray(true_hidden_parameters, dtype=np.float32)[None, :],
            device=self.device,
        )
        if action_type == "continuous":
            physical_id = int(continuous_physical_action_id)
            ids = torch.tensor([physical_id], dtype=torch.long, device=self.device)
            node_means = self.predict_at_nodes(ids)[:, 0] / self.observation_scale
            true_mean = self.predict_at_point(ids, true_hidden)[0] / self.observation_scale
            precision = self.continuous_precision
            log_normalization = self.continuous_log_normalization
        else:
            source_index = int(source_action_index)
            true_outcomes = interpolate_outcome_grid(
                self.outcome_grid, true_hidden
            )[0]
            node_means = (
                self.node_outcomes[:, source_index]
                / self.observation_scale[None, :]
                + self.discrete_bias[None, :]
            )
            true_mean = (
                true_outcomes[source_index] / self.observation_scale
                + self.discrete_bias
            )
            precision = self.discrete_precision
            log_normalization = self.discrete_log_normalization
        node_log_likelihoods = self._student_t_log_likelihood(
            observation_tensor[None, :] - node_means,
            precision,
            log_normalization,
        )
        true_log_likelihood = self._student_t_log_likelihood(
            observation_tensor - true_mean,
            precision,
            log_normalization,
        )
        return node_log_likelihoods, true_log_likelihood

    @torch.inference_mode()
    def update_action_observation(
        self,
        belief: car_e4.E4BeliefState,
        action_type: str,
        source_action_index: int,
        continuous_physical_action_id: int | None,
        observation: np.ndarray,
        true_hidden_parameters: np.ndarray,
    ) -> bool:
        """使用温度化action-matched likelihood更新belief。"""

        if belief.update_count >= BELIEF_UPDATE_HORIZON:
            return False
        node_log_likelihoods, true_log_likelihood = self.action_log_likelihoods(
            action_type,
            source_action_index,
            continuous_physical_action_id,
            observation,
            true_hidden_parameters,
        )
        temperature = self.sequential_likelihood_temperature
        node_log_likelihoods = node_log_likelihoods / temperature
        true_log_likelihood = true_log_likelihood / temperature
        belief.cumulative_node_log_likelihoods += node_log_likelihoods
        belief.cumulative_true_log_likelihood += float(true_log_likelihood)
        unnormalized = self.log_prior_weights + belief.cumulative_node_log_likelihoods
        belief.log_weights = unnormalized - torch.logsumexp(unnormalized, dim=0)
        belief.update_count += 1
        return True

    @torch.inference_mode()
    def screening_scores(
        self,
        physical_ids: torch.Tensor,
        task_query: np.ndarray,
        belief: car_e4.E4BeliefState,
        true_hidden_parameters: np.ndarray,
        controller_id: str,
    ) -> torch.Tensor:
        """使用posterior mean或true condition筛选2,560个continuous candidates。"""

        if controller_id == FULL_INFORMATION_CONTINUOUS_CONTROLLER_ID:
            hidden = torch.as_tensor(
                np.asarray(true_hidden_parameters, dtype=np.float32)[None, :],
                device=self.device,
            )
        else:
            hidden = posterior_mean_hidden_parameters(
                self.nodes, belief.weights[None, :]
            )
        outcomes = self.predict_at_point(physical_ids, hidden)
        task = torch.as_tensor(
            np.asarray(task_query, dtype=np.float32)[None, :], device=self.device
        )
        return tnpo_costs(outcomes[None, :, :], task)[0]

    @torch.inference_mode()
    def expected_continuous_scores(
        self,
        physical_ids: torch.Tensor,
        task_query: np.ndarray,
        belief: car_e4.E4BeliefState,
        candidate_chunk_size: int = 64,
    ) -> torch.Tensor:
        """先计算node-wise TNPO cost，再执行posterior marginalisation。"""

        task = torch.as_tensor(
            np.asarray(task_query, dtype=np.float32)[None, :], device=self.device
        )
        desired_yaw = torch.atan2(task[0, 2], task[0, 3])
        outputs = torch.empty(
            len(physical_ids), dtype=torch.float32, device=self.device
        )
        for start in range(0, len(physical_ids), candidate_chunk_size):
            stop = min(len(physical_ids), start + candidate_chunk_size)
            outcomes = self.predict_at_nodes(
                physical_ids[start:stop], candidate_chunk_size
            )
            position = torch.linalg.vector_norm(
                outcomes[..., :2] - task[0, None, :2], dim=-1
            )
            yaw = torch.abs(
                car_e4.torch_wrap_to_pi(outcomes[..., 2] - desired_yaw)
            )
            costs = (
                0.5 * position / PRIMARY_TNPO_COST.position_tolerance_m
                + 0.5 * yaw / PRIMARY_TNPO_COST.yaw_tolerance_rad
            )
            outputs[start:stop] = torch.einsum(
                "n,nk->k", belief.weights.float(), costs.float()
            )
        return outputs

    @torch.inference_mode()
    def true_condition_predictions(
        self,
        decision: car_e4.ContinuousActionDecision,
        true_hidden_parameters: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """返回执行动作与source anchor的simulator-only predicted outcomes。"""

        hidden = torch.as_tensor(
            np.asarray(true_hidden_parameters, dtype=np.float32)[None, :],
            device=self.device,
        )
        source_outcomes = interpolate_outcome_grid(self.outcome_grid, hidden)[0]
        source_prediction = source_outcomes[decision.source_action_index]
        if decision.selected_action_type == "continuous":
            physical_id = int(decision.continuous_physical_action_id)
            selected_prediction = self.predict_at_point(
                torch.tensor([physical_id], dtype=torch.long, device=self.device),
                hidden,
            )[0]
            normalized_offset = float(
                torch.linalg.vector_norm(
                    (
                        self.continuous_q[physical_id]
                        - self.anchor_q[decision.source_action_index]
                    )
                    / torch.as_tensor(
                        car_e2.OFFSET_SCALES,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
            )
        else:
            selected_prediction = source_prediction
            normalized_offset = 0.0
        return (
            selected_prediction.detach().cpu().numpy(),
            source_prediction.detach().cpu().numpy(),
            normalized_offset,
        )


def tempered_terminal_metrics(
    log_prior_weights: np.ndarray,
    cumulative_node_log_likelihoods: np.ndarray,
    cumulative_true_log_likelihood: float,
    dimension: int,
    temperature: float,
) -> dict[str, float | int]:
    """计算一个温度候选对应的terminal posterior metrics。"""

    scaled_node = cumulative_node_log_likelihoods / temperature
    unnormalised = log_prior_weights + scaled_node
    maximum = float(np.max(unnormalised))
    log_evidence = maximum + math.log(
        float(np.sum(np.exp(unnormalised - maximum)))
    )
    log_weights = unnormalised - log_evidence
    weights = np.exp(log_weights)
    constant = -dimension * math.log(2.0)
    node_log_density = constant + log_weights - log_prior_weights
    order = np.argsort(-node_log_density, kind="stable")
    cumulative_mass = np.cumsum(weights[order])
    threshold_index = int(np.searchsorted(cumulative_mass, 0.95, side="left"))
    threshold_index = min(threshold_index, len(order) - 1)
    threshold = float(node_log_density[order[threshold_index]])
    true_log_density = (
        constant + cumulative_true_log_likelihood / temperature - log_evidence
    )
    return {
        "posterior_nll": -true_log_density,
        "hpd_covered": int(true_log_density >= threshold - 1e-12),
    }


def replay_on_policy_evidence(
    scenario: str,
    engine: V2ContinuousDecisionEngine,
) -> list[dict[str, Any]]:
    """从现有Validation primary trajectories提取未温度化累计证据。"""

    rows = car_e4.load_formal_step_rows(
        DATA_ROOT / "closed_loop", scenario, "validation"
    )
    rows = [
        row
        for row in rows
        if row["controller_id"]
        == BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
    ]
    episodes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        episodes[row["episode_key"]].append(row)

    outputs: list[dict[str, Any]] = []
    for episode_index, episode_rows in enumerate(episodes.values(), start=1):
        episode_rows.sort(key=lambda row: int(row["push_index"]))
        first = episode_rows[0]
        true_hidden = np.asarray(
            [
                float(first[field])
                for field in SCENARIO_ACTIVE_COORDINATES[scenario]
            ],
            dtype=np.float32,
        )
        cumulative_node = np.zeros(len(engine.nodes), dtype=np.float64)
        cumulative_true = 0.0
        update_count = 0
        for row in episode_rows:
            if int(row["belief_updated"]) != 1:
                continue
            observation = np.asarray(
                [
                    float(row["observation_local_delta_x_m"]),
                    float(row["observation_local_delta_y_m"]),
                    float(row["observation_delta_yaw_rad"]),
                ],
                dtype=np.float64,
            )
            continuous_id = row.get("continuous_physical_action_id", "")
            node_log_likelihoods, true_log_likelihood = (
                engine.action_log_likelihoods(
                    row["selected_action_type"],
                    int(row["source_action_index"]),
                    int(continuous_id) if continuous_id != "" else None,
                    observation,
                    true_hidden,
                )
            )
            cumulative_node += node_log_likelihoods.detach().cpu().numpy().astype(
                np.float64
            )
            cumulative_true += float(true_log_likelihood)
            update_count += 1
        if update_count == 0:
            raise RuntimeError(f"{first['episode_key']}没有有效belief update")
        outputs.append(
            {
                "condition_id": first["condition_id"],
                "episode_key": first["episode_key"],
                "update_count": update_count,
                "cumulative_node_log_likelihoods": cumulative_node,
                "cumulative_true_log_likelihood": cumulative_true,
            }
        )
        if episode_index % 128 == 0 or episode_index == len(episodes):
            print(
                f"replayed {scenario} episode "
                f"{episode_index}/{len(episodes)}"
            )
    return outputs


def calibrate_on_policy_likelihood(args: argparse.Namespace) -> dict[str, Any]:
    """用闭环Validation trajectories校准序贯似然温度。"""

    scenarios = select_scenarios(args.scenario)
    if scenarios != SCENARIOS:
        raise ValueError("正式on-policy likelihood calibration必须使用--scenario all")
    device = require_cuda()
    scenario_summaries: dict[str, Any] = {}
    selected_temperatures: dict[str, float] = {}
    for scenario in scenarios:
        engine = V2ContinuousDecisionEngine(
            scenario,
            device,
            sequential_likelihood_temperature=1.0,
        )
        evidence = replay_on_policy_evidence(scenario, engine)
        log_prior = engine.log_prior_weights.detach().cpu().numpy().astype(np.float64)
        dimension = int(engine.nodes.shape[1])
        candidate_results: list[dict[str, Any]] = []
        coverage_tolerance = 0.5 / len(evidence)
        for temperature in SEQUENTIAL_LIKELIHOOD_TEMPERATURE_CANDIDATES:
            metrics = [
                tempered_terminal_metrics(
                    log_prior,
                    row["cumulative_node_log_likelihoods"],
                    row["cumulative_true_log_likelihood"],
                    dimension,
                    temperature,
                )
                for row in evidence
            ]
            by_condition: dict[str, list[float]] = defaultdict(list)
            for row, metric in zip(evidence, metrics):
                by_condition[row["condition_id"]].append(
                    float(metric["posterior_nll"])
                )
            result = {
                "sequential_likelihood_temperature": temperature,
                "terminal_hpd_coverage": float(
                    np.mean([metric["hpd_covered"] for metric in metrics])
                ),
                "condition_balanced_mean_terminal_nll": float(
                    np.mean([np.mean(values) for values in by_condition.values()])
                ),
            }
            candidate_results.append(result)
            print(
                f"{scenario} temperature={temperature:g}: "
                f"coverage={result['terminal_hpd_coverage']:.6f}, "
                "condition_balanced_nll="
                f"{result['condition_balanced_mean_terminal_nll']:.6f}"
            )
        calibrated = [
            row
            for row in candidate_results
            if CALIBRATION_COVERAGE_LOWER - coverage_tolerance
            <= float(row["terminal_hpd_coverage"])
            <= CALIBRATION_COVERAGE_UPPER + coverage_tolerance
        ]
        if not calibrated:
            raise RuntimeError(
                f"{scenario}没有满足93%–97% on-policy coverage目标区间的温度"
            )
        selected = min(
            calibrated,
            key=lambda row: (
                float(row["condition_balanced_mean_terminal_nll"]),
                float(row["sequential_likelihood_temperature"]),
            ),
        )
        selected_temperature = float(
            selected["sequential_likelihood_temperature"]
        )
        selected_temperatures[scenario] = selected_temperature
        scenario_summaries[scenario] = {
            "scenario": scenario,
            "terminal_episodes": len(evidence),
            "coverage_target": [
                CALIBRATION_COVERAGE_LOWER,
                CALIBRATION_COVERAGE_UPPER,
            ],
            "candidate_results": candidate_results,
            "selected_sequential_likelihood_temperature": selected_temperature,
            "selected_replayed_terminal_hpd_coverage": selected[
                "terminal_hpd_coverage"
            ],
            "selected_condition_balanced_mean_terminal_nll": selected[
                "condition_balanced_mean_terminal_nll"
            ],
        }
        del engine
        torch.cuda.empty_cache()

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "calibration_role": "validation",
        "calibration_method": "scenario_specific_sequential_likelihood_tempering",
        "likelihood_family": "multivariate_student_t",
        "candidate_temperatures": list(
            SEQUENTIAL_LIKELIHOOD_TEMPERATURE_CANDIDATES
        ),
        "scenario_summaries": scenario_summaries,
        "all_scenarios_calibrated": set(selected_temperatures) == set(SCENARIOS),
        "test_trajectories_viewed": False,
    }
    write_json(ON_POLICY_TEMPERING_RESULT_PATH, summary)
    configuration = load_likelihood_configuration()
    configuration["sequential_likelihood_tempering_method"] = (
        "power_likelihood_log_likelihood_divided_by_temperature"
    )
    configuration["sequential_likelihood_temperature"] = selected_temperatures
    configuration["on_policy_tempering_calibrated"] = True
    configuration["on_policy_tempering_source"] = (
        "closed_loop_validation_trajectories_before_tempering"
    )
    configuration["test_trajectories_viewed"] = False
    write_json(LIKELIHOOD_CONFIGURATION_PATH, configuration)
    print(json.dumps(make_json_compatible(summary), ensure_ascii=False, indent=2))
    return summary


def load_sequential_targets(role: str) -> list[dict[str, str]]:
    """读取64个Sequential-Extension targets。"""

    rows = read_csv_rows(SEQUENTIAL_TARGET_PATHS[role])
    if len(rows) != 64:
        raise RuntimeError(f"{role} Sequential-Extension target 数量错误: {len(rows)}")
    return rows


def parse_controllers(value: str) -> tuple[str, ...]:
    """解析Version 2 continuous controller IDs。"""

    if value == "all":
        return CONTINUOUS_CONTROLLER_IDS
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(values) - set(CONTINUOUS_CONTROLLER_IDS)
    if unknown:
        raise ValueError(f"未知continuous controllers: {sorted(unknown)}")
    return values


def inspect_complete_shard(
    path: Path,
    target_ids: set[str],
    controller_ids: tuple[str, ...],
    role: str,
    maximum_pushes: int,
) -> bool:
    """判断一个Version 2 condition shard是否完整。"""

    if not path.exists():
        return False
    rows = read_csv_rows(path)
    terminal = [row for row in rows if int(row["is_terminal_push"]) == 1]
    expected = {
        (target_id, controller_id)
        for target_id in target_ids
        for controller_id in controller_ids
    }
    observed = {(row["target_id"], row["controller_id"]) for row in terminal}
    scenarios = {row["scenario"] for row in rows}
    temperature_matches = False
    if len(scenarios) == 1:
        expected_temperature = configured_sequential_temperature(
            next(iter(scenarios))
        )
        temperature_matches = all(
            row.get("sequential_likelihood_temperature", "") != ""
            and math.isclose(
                float(row["sequential_likelihood_temperature"]),
                expected_temperature,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in rows
        )
    return (
        expected == observed
        and {row["protocol_version"] for row in rows} == {PROTOCOL_VERSION}
        and {row["role"] for row in rows} == {role}
        and {int(row["maximum_push_budget"]) for row in rows} == {maximum_pushes}
        and {
            int(row["shortlist_budget"])
            for row in rows
            if row["controller_id"] == BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
        }
        == {EXACT_CONTINUOUS_CANDIDATE_BUDGET}
        and temperature_matches
    )


CLOSED_LOOP_WORKER_CONTEXT: dict[str, Any] = {}


def initialize_closed_loop_worker(
    scenario: str,
    targets: list[dict[str, str]],
    controller_ids: tuple[str, ...],
    role: str,
    maximum_pushes: int,
    node_query_chunk_size: int,
) -> None:
    """为Windows closed-loop worker加载一次GPU artifacts。"""

    torch.set_num_threads(1)
    device = require_cuda()
    engine = V2ContinuousDecisionEngine(
        scenario,
        device,
        EXACT_CONTINUOUS_CANDIDATE_BUDGET,
        node_query_chunk_size=node_query_chunk_size,
    )
    torch.cuda.reset_peak_memory_stats(device)
    CLOSED_LOOP_WORKER_CONTEXT.clear()
    CLOSED_LOOP_WORKER_CONTEXT.update(
        {
            "scenario": scenario,
            "targets": targets,
            "controller_ids": controller_ids,
            "role": role,
            "maximum_pushes": maximum_pushes,
            "engine": engine,
        }
    )


def collect_condition_task(
    task: tuple[int, int, dict[str, str], str, str]
) -> dict[str, Any]:
    """在一个worker中采集完整condition shard。"""

    condition_index, condition_count, condition, xml_text, output_text = task
    context = CLOSED_LOOP_WORKER_CONTEXT
    engine: V2ContinuousDecisionEngine = context["engine"]
    environment_xml = Path(xml_text)
    model, data = car_e4.load_model(environment_xml)
    hcr_e1.set_sliding_friction(model, float(condition["friction_sliding_mu"]))
    object_body_id = car_e4.get_body_id(model)
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(context["targets"]):
        for controller_index, controller_id in enumerate(context["controller_ids"]):
            episode_id = (
                int(condition["condition_index_within_role"]) * 100_000
                + target_index * 10
                + controller_index
            )
            episode_rows = car_e4.run_closed_loop_episode(
                model,
                data,
                object_body_id,
                engine,
                controller_id,
                condition,
                target,
                episode_id,
                environment_xml,
                context["role"],
                context["maximum_pushes"],
            )
            for row in episode_rows:
                row["experiment_id"] = "E4_V2"
                row["protocol_version"] = PROTOCOL_VERSION
                row["experiment_name"] = EXPERIMENT_NAME
                row["controller_name"] = CONTINUOUS_CONTROLLER_NAMES[controller_id]
                row["sequential_likelihood_temperature"] = (
                    engine.sequential_likelihood_temperature
                )
            rows.extend(episode_rows)
    output_path = Path(output_text)
    write_csv(output_path, rows, V2_STEP_FIELDS)
    summary = car_e4.summarize_condition_rows(
        rows,
        context["scenario"],
        condition["condition_id"],
        output_path,
        resumed=0,
    )
    summary.update(
        {
            "condition_index": condition_index,
            "condition_count": condition_count,
            "worker_process_id": os.getpid(),
            "worker_peak_cuda_memory_mib": (
                torch.cuda.max_memory_allocated(engine.device) / (1024.0**2)
            ),
            "worker_peak_cuda_reserved_mib": (
                torch.cuda.max_memory_reserved(engine.device) / (1024.0**2)
            ),
        }
    )
    return summary


def selected_worker_count() -> int:
    """读取benchmark选择的正式worker count。"""

    path = RESULTS_ROOT / "benchmark" / "selected_configuration.json"
    if not path.exists():
        raise FileNotFoundError("请先运行 benchmark-workers")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return int(payload["selected_num_workers"])


def validation_gate_passed() -> bool:
    """读取Version 2 closed-loop Validation gate。"""

    path = RESULTS_ROOT / "evaluation" / "validation" / "gate_decision.json"
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        return bool(json.load(handle).get("passed", False))


def collect_closed_loop(args: argparse.Namespace) -> dict[str, Any]:
    """采集Version 2 Validation或Independent Test trajectories。"""

    if args.role == "test" and not validation_gate_passed():
        raise RuntimeError("Version 2 Validation gate未通过，不能访问Test")
    likelihood_configuration = load_likelihood_configuration()
    if not likelihood_configuration.get("on_policy_tempering_calibrated", False):
        raise RuntimeError("请先运行calibrate-on-policy-likelihood --scenario all")
    controller_ids = parse_controllers(args.controllers)
    targets = load_sequential_targets(args.role)
    if args.max_targets > 0:
        targets = targets[: args.max_targets]
    formal_run = (
        controller_ids == CONTINUOUS_CONTROLLER_IDS
        and args.max_conditions <= 0
        and args.max_targets <= 0
        and args.maximum_pushes == MAXIMUM_PUSHES
    )
    collection_root = DATA_ROOT / ("closed_loop" if formal_run else "smoke")
    if formal_run:
        benchmark_worker_count = selected_worker_count()
        if args.num_workers not in {0, benchmark_worker_count}:
            raise ValueError(
                f"正式采集必须使用benchmark选择的num-workers={benchmark_worker_count}"
            )
        requested_workers = benchmark_worker_count
    else:
        requested_workers = max(1, args.num_workers)
    all_results: list[dict[str, Any]] = []
    for scenario in select_scenarios(args.scenario):
        conditions = hcr_e3.load_conditions(scenario, args.role)
        if args.max_conditions > 0:
            conditions = conditions[: args.max_conditions]
        generated_dir = DATA_ROOT / "generated_xml" / args.role / scenario
        xml_by_com = hcr_e1.prepare_environment_xmls(conditions, generated_dir)
        pending: list[tuple[int, int, dict[str, str], str, str]] = []
        for condition_index, condition in enumerate(conditions, start=1):
            output_path = (
                collection_root
                / args.role
                / scenario
                / f"{condition['condition_id']}.csv"
            )
            if args.resume and inspect_complete_shard(
                output_path,
                {target["v2_target_id"] for target in targets},
                controller_ids,
                args.role,
                args.maximum_pushes,
            ):
                result = car_e4.summarize_condition_rows(
                    read_csv_rows(output_path),
                    scenario,
                    condition["condition_id"],
                    output_path,
                    resumed=1,
                )
                all_results.append(result)
                print(f"resumed {scenario} condition {condition_index}/{len(conditions)}")
                continue
            com_key = (
                round(float(condition["com_offset_x_m"]), 9),
                round(float(condition["com_offset_y_m"]), 9),
            )
            pending.append(
                (
                    condition_index,
                    len(conditions),
                    condition,
                    str(xml_by_com[com_key]),
                    str(output_path),
                )
            )
        worker_count = min(max(1, requested_workers), max(1, len(pending)))
        if pending:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
                initializer=initialize_closed_loop_worker,
                initargs=(
                    scenario,
                    targets,
                    controller_ids,
                    args.role,
                    args.maximum_pushes,
                    args.node_query_chunk_size,
                ),
            ) as executor:
                futures = {
                    executor.submit(collect_condition_task, task): task[0]
                    for task in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    all_results.append(result)
                    print(
                        f"finished {scenario} condition "
                        f"{result['condition_index']}/{result['condition_count']}"
                    )
    all_results.sort(key=lambda row: (row["scenario"], row["condition_id"]))
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "role": args.role,
        "scenario": args.scenario,
        "formal_run": formal_run,
        "controllers": list(controller_ids),
        "candidate_scoring_mode": "exact_full_set_posterior_marginalisation",
        "exact_continuous_candidate_budget": EXACT_CONTINUOUS_CANDIDATE_BUDGET,
        "shortlist_budget": EXACT_CONTINUOUS_CANDIDATE_BUDGET,
        "sequential_likelihood_temperature": {
            scenario: configured_sequential_temperature(scenario)
            for scenario in select_scenarios(args.scenario)
        },
        "conditions": len(all_results),
        "targets_per_condition": len(targets),
        "maximum_attempted_pushes": args.maximum_pushes,
        "num_workers": requested_workers,
        "episodes": sum(row["episodes"] for row in all_results),
        "step_rows": sum(row["step_rows"] for row in all_results),
        "successful_episodes": sum(row["successful_episodes"] for row in all_results),
        "invalid_episodes": sum(row["invalid_episodes"] for row in all_results),
        "maximum_budget_episodes": sum(
            row["maximum_budget_episodes"] for row in all_results
        ),
        "resumed_conditions": sum(row["resumed"] for row in all_results),
        "collection_root": str(collection_root.resolve()),
        "condition_results": all_results,
    }
    write_json(
        collection_root / args.role / f"collection_summary_{args.scenario}.json",
        summary,
    )
    print(json.dumps(make_json_compatible(summary), ensure_ascii=False, indent=2))
    return summary


def evaluate_exact_selector(args: argparse.Namespace) -> dict[str, Any]:
    """在每个场景64个replayed Validation states确认全集精确计算。"""

    device = require_cuda()
    scenario_summaries: dict[str, Any] = {}
    for scenario in select_scenarios(args.scenario):
        engine = V2ContinuousDecisionEngine(
            scenario,
            device,
            EXACT_CONTINUOUS_CANDIDATE_BUDGET,
            node_query_chunk_size=args.node_query_chunk_size,
        )
        all_rows = car_e4.load_formal_step_rows(
            HCR_E5_DATA_ROOT, scenario, "validation"
        )
        eligible = [
            row
            for row in all_rows
            if row["target_group"] == "sequential_extension"
            and row["controller_id"] == DISCRETE_BASELINE_CONTROLLER_ID
            and int(row["episode_invalid"]) == 0
        ]
        histories: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in eligible:
            histories[row["episode_key"]].append(row)
        for episode_rows in histories.values():
            episode_rows.sort(key=lambda row: int(row["push_index"]))
        selected_rows: list[dict[str, str]] = []
        for condition in hcr_e3.load_conditions(scenario, "validation"):
            condition_rows = sorted(
                [row for row in eligible if row["condition_id"] == condition["condition_id"]],
                key=lambda row: (int(row["push_index"]), row["target_id"]),
            )
            selected_rows.extend(car_e4.evenly_spaced_rows(condition_rows, 4))
        if len(selected_rows) != 64:
            raise RuntimeError(
                f"{scenario} exact-selector states 数量错误: {len(selected_rows)}"
            )
        cases: list[dict[str, Any]] = []
        for case_index, row in enumerate(selected_rows):
            belief = car_e4.replay_belief_before_row(
                engine, row, histories[row["episode_key"]]
            )
            true_hidden = np.asarray(
                [float(row[field]) for field in SCENARIO_ACTIVE_COORDINATES[scenario]],
                dtype=np.float32,
            )
            query = np.asarray(
                [
                    float(row["task_local_x_m"]),
                    float(row["task_local_y_m"]),
                    float(row["task_yaw_sin"]),
                    float(row["task_yaw_cos"]),
                ],
                dtype=np.float32,
            )
            values = engine.exact_full_set_diagnostic(query, belief, true_hidden)
            cases.append(
                {
                    "scenario": scenario,
                    "case_index": case_index,
                    "condition_id": row["condition_id"],
                    "target_id": row["target_id"],
                    "target_stratum": row["target_stratum"],
                    "push_index": int(row["push_index"]),
                    "valid_update_count": belief.update_count,
                    "candidate_scoring_mode": (
                        "exact_full_set_posterior_marginalisation"
                    ),
                    **values,
                }
            )
        latencies = np.asarray(
            [float(row["exact_posterior_marginalisation_latency_s"]) for row in cases]
        )
        peak_memory = np.asarray(
            [float(row["peak_cuda_memory_mib"]) for row in cases]
        )
        summary = {
            "scenario": scenario,
            "decision_states": len(cases),
            "candidate_scoring_mode": "exact_full_set_posterior_marginalisation",
            "exact_continuous_candidate_budget": (
                EXACT_CONTINUOUS_CANDIDATE_BUDGET
            ),
            "exact_posterior_marginalisation_latency_s": descriptive(latencies),
            "peak_cuda_memory_mib": descriptive(peak_memory),
            "all_cases_scored_full_candidate_set": all(
                int(row["candidate_count"])
                == EXACT_CONTINUOUS_CANDIDATE_BUDGET
                for row in cases
            ),
        }
        summary["passed"] = bool(summary["all_cases_scored_full_candidate_set"])
        scenario_summaries[scenario] = summary
        output_dir = RESULTS_ROOT / "exact_full_set_validation" / scenario
        write_json(output_dir / "summary.json", summary)
        write_csv(output_dir / "cases.csv", cases, list(cases[0]))
        del engine
        torch.cuda.empty_cache()
        print(f"evaluated {scenario} exact full-set selector")
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_scoring_mode": "exact_full_set_posterior_marginalisation",
        "exact_continuous_candidate_budget": EXACT_CONTINUOUS_CANDIDATE_BUDGET,
        "scenario_summaries": scenario_summaries,
        "passed": set(scenario_summaries) == set(SCENARIOS)
        and all(summary["passed"] for summary in scenario_summaries.values()),
    }
    write_json(
        RESULTS_ROOT / "exact_full_set_validation" / "selected_configuration.json",
        combined,
    )
    print(json.dumps(make_json_compatible(combined), ensure_ascii=False, indent=2))
    return combined


def benchmark_targets() -> list[dict[str, str]]:
    """从四个target strata各取两个benchmark targets。"""

    targets = load_sequential_targets("validation")
    selected: list[dict[str, str]] = []
    for stratum in sorted({row["target_stratum"] for row in targets}):
        rows = sorted(
            [row for row in targets if row["target_stratum"] == stratum],
            key=lambda row: row["v2_target_id"],
        )
        selected.extend(car_e4.evenly_spaced_rows(rows, 2))
    return selected


def run_worker_benchmark(
    num_workers: int,
    node_query_chunk_size: int,
) -> dict[str, Any]:
    """运行一个Joint Validation worker benchmark configuration。"""

    scenario = "joint"
    conditions = hcr_e3.load_conditions(scenario, "validation")[:6]
    targets = benchmark_targets()
    generated_dir = DATA_ROOT / "generated_xml" / "benchmark" / scenario
    xml_by_com = hcr_e1.prepare_environment_xmls(conditions, generated_dir)
    output_root = RESULTS_ROOT / "benchmark" / f"workers_{num_workers}" / "shards"
    tasks: list[tuple[int, int, dict[str, str], str, str]] = []
    for condition_index, condition in enumerate(conditions, start=1):
        com_key = (
            round(float(condition["com_offset_x_m"]), 9),
            round(float(condition["com_offset_y_m"]), 9),
        )
        tasks.append(
            (
                condition_index,
                len(conditions),
                condition,
                str(xml_by_com[com_key]),
                str(output_root / f"{condition['condition_id']}.csv"),
            )
        )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_closed_loop_worker,
            initargs=(
                scenario,
                targets,
                CONTINUOUS_CONTROLLER_IDS,
                "validation",
                5,
                node_query_chunk_size,
            ),
        ) as executor:
            futures = [executor.submit(collect_condition_task, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    elapsed = time.perf_counter() - started
    episodes = sum(int(row["episodes"]) for row in results)
    invalid = sum(int(row["invalid_episodes"]) for row in results)
    worker_reserved: dict[str, float] = {}
    for row in results:
        process_id = str(row["worker_process_id"])
        worker_reserved[process_id] = max(
            worker_reserved.get(process_id, 0.0),
            float(row.get("worker_peak_cuda_reserved_mib", 0.0)),
        )
    reserved = sum(worker_reserved.values())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "num_workers": num_workers,
        "conditions": len(conditions),
        "targets": len(targets),
        "controllers": len(CONTINUOUS_CONTROLLER_IDS),
        "episodes": episodes,
        "elapsed_seconds": elapsed,
        "episodes_per_second": episodes / elapsed if elapsed > 0 else 0.0,
        "invalid_episodes": invalid,
        "invalid_episode_rate": invalid / episodes if episodes else 1.0,
        "runtime_errors": errors,
        "worker_peak_cuda_reserved_mib": worker_reserved,
        "estimated_concurrent_cuda_memory_mib": reserved,
        "eligible_for_selection": (
            not errors
            and episodes == len(conditions) * len(targets) * len(CONTINUOUS_CONTROLLER_IDS)
            and (invalid / episodes if episodes else 1.0) <= 0.01
            and reserved <= 7_680.0
        ),
    }


def benchmark_workers(args: argparse.Namespace) -> dict[str, Any]:
    """比较候选worker counts并选择实际episodes-per-second最高者。"""

    worker_counts = [
        int(value.strip()) for value in args.worker_counts.split(",") if value.strip()
    ]
    configurations: list[dict[str, Any]] = []
    for worker_count in worker_counts:
        result = run_worker_benchmark(worker_count, args.node_query_chunk_size)
        configurations.append(result)
        print(
            f"workers={worker_count}: "
            f"episodes_per_second={result['episodes_per_second']:.4f}"
        )
    eligible = [row for row in configurations if row["eligible_for_selection"]]
    if not eligible:
        raise RuntimeError("没有可用于正式closed-loop collection的worker配置")
    selected = max(eligible, key=lambda row: float(row["episodes_per_second"]))
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_num_workers": int(selected["num_workers"]),
        "selection_rule": (
            "无runtime error、invalid rate不超过1%、估计并发CUDA memory不超过7680 MiB"
            "的配置中选择episodes per second最高者"
        ),
        "candidate_scoring_mode": "exact_full_set_posterior_marginalisation",
        "exact_continuous_candidate_budget": EXACT_CONTINUOUS_CANDIDATE_BUDGET,
        "configurations": configurations,
    }
    write_json(RESULTS_ROOT / "benchmark" / "selected_configuration.json", summary)
    print(json.dumps(make_json_compatible(summary), ensure_ascii=False, indent=2))
    return summary


EVALUATION_CONTROLLER_NAMES = {
    DISCRETE_BASELINE_CONTROLLER_ID: "Discrete Belief-Marginalised Closed Loop",
    VERSION_1_PRIMARY_CONTROLLER_ID: "Version 1 Belief-Marginalised Continuous",
    **CONTINUOUS_CONTROLLER_NAMES,
    DISCRETE_FULL_INFORMATION_CONTROLLER_ID: "Discrete Full-Information",
}
EVALUATION_CONTROLLER_IDS = tuple(EVALUATION_CONTROLLER_NAMES)


def relabel_version_1_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """只为合并评价复制并重命名Version 1 Primary rows。"""

    outputs: list[dict[str, str]] = []
    for row in rows:
        if row["controller_id"] != BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID:
            continue
        copied = dict(row)
        copied["controller_id"] = VERSION_1_PRIMARY_CONTROLLER_ID
        copied["controller_name"] = EVALUATION_CONTROLLER_NAMES[
            VERSION_1_PRIMARY_CONTROLLER_ID
        ]
        copied["episode_key"] = f"{row['episode_key']}|version_1"
        outputs.append(copied)
    return outputs


def load_evaluation_rows(
    scenario: str,
    role: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """读取V2、V1 historical comparator与HCR discrete baseline trajectories。"""

    v2 = car_e4.load_formal_step_rows(DATA_ROOT / "closed_loop", scenario, role)
    v2 = [
        row
        for row in v2
        if row["target_group"] == "sequential_extension"
        and row["controller_id"] in CONTINUOUS_CONTROLLER_IDS
    ]
    if role == "validation":
        v1 = car_e4.load_formal_step_rows(
            V1_DATA_ROOT / "closed_loop", scenario, role
        )
        v1 = relabel_version_1_rows(
            [row for row in v1 if row["target_group"] == "sequential_extension"]
        )
        controller_count = len(EVALUATION_CONTROLLER_IDS)
    else:
        v1 = []
        controller_count = len(EVALUATION_CONTROLLER_IDS) - 1
    discrete = car_e4.load_formal_step_rows(HCR_E5_DATA_ROOT, scenario, role)
    discrete = [
        row
        for row in discrete
        if row["target_group"] == "sequential_extension"
        and row["controller_id"]
        in {DISCRETE_BASELINE_CONTROLLER_ID, DISCRETE_FULL_INFORMATION_CONTROLLER_ID}
    ]
    steps = [*v2, *v1, *discrete]
    terminal = [row for row in steps if int(row["is_terminal_push"]) == 1]
    expected = (
        len(hcr_e3.load_conditions(scenario, role))
        * 64
        * controller_count
    )
    if len(terminal) != expected:
        raise RuntimeError(
            f"{scenario}/{role} terminal rows 数量错误: {len(terminal)} != {expected}"
        )
    return steps, terminal


def paired_controller_matrices(
    terminal_rows: list[dict[str, str]],
) -> tuple[list[str], list[str], dict[str, np.ndarray], dict[str, str]]:
    """把六条routes对齐为condition×target paired matrices。"""

    condition_ids = sorted({row["condition_id"] for row in terminal_rows})
    target_ids = sorted({row["target_id"] for row in terminal_rows})
    c_index = {value: index for index, value in enumerate(condition_ids)}
    t_index = {value: index for index, value in enumerate(target_ids)}
    shape = (len(condition_ids), len(target_ids))
    active_controller_ids = tuple(
        controller_id
        for controller_id in EVALUATION_CONTROLLER_IDS
        if any(row["controller_id"] == controller_id for row in terminal_rows)
    )
    fields = ("success", "auc", "pushes", "final_cost", "maximum_yaw")
    matrices = {
        f"{controller}|{field}": np.full(shape, np.nan, dtype=np.float64)
        for controller in active_controller_ids
        for field in fields
    }
    target_strata: dict[str, str] = {}
    for row in terminal_rows:
        c = c_index[row["condition_id"]]
        t = t_index[row["target_id"]]
        controller = row["controller_id"]
        success = float(row["episode_success"])
        pushes = int(row["terminal_push_count"])
        matrices[f"{controller}|success"][c, t] = success
        matrices[f"{controller}|auc"][c, t] = (
            (MAXIMUM_PUSHES + 1 - pushes) / MAXIMUM_PUSHES if success else 0.0
        )
        matrices[f"{controller}|pushes"][c, t] = pushes
        matrices[f"{controller}|final_cost"][c, t] = float(row["actual_tnpo_cost"])
        matrices[f"{controller}|maximum_yaw"][c, t] = float(
            row["maximum_yaw_deviation_rad"]
        )
        target_strata[row["target_id"]] = row["target_stratum"]
    if any(np.isnan(values).any() for values in matrices.values()):
        raise RuntimeError("Version 2 paired matrices存在缺失episodes")
    return condition_ids, target_ids, matrices, target_strata


def closed_loop_effect_matrices(matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """构造Version 2正式paired effects。"""

    v2 = BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
    ce = CERTAINTY_EQUIVALENT_CONTINUOUS_CONTROLLER_ID
    fi = FULL_INFORMATION_CONTINUOUS_CONTROLLER_ID
    discrete = DISCRETE_BASELINE_CONTROLLER_ID
    discrete_fi = DISCRETE_FULL_INFORMATION_CONTROLLER_ID
    v1 = VERSION_1_PRIMARY_CONTROLLER_ID
    effects = {
        "v2_minus_discrete_success_by_push_auc": (
            matrices[f"{v2}|auc"] - matrices[f"{discrete}|auc"]
        ),
        "v2_minus_certainty_equivalent_success_by_push_auc": (
            matrices[f"{v2}|auc"] - matrices[f"{ce}|auc"]
        ),
        "v2_full_information_minus_discrete_full_information_auc": (
            matrices[f"{fi}|auc"] - matrices[f"{discrete_fi}|auc"]
        ),
        "v2_minus_discrete_episode_success_rate": (
            matrices[f"{v2}|success"] - matrices[f"{discrete}|success"]
        ),
        "v2_full_information_minus_v2_primary_auc": (
            matrices[f"{fi}|auc"] - matrices[f"{v2}|auc"]
        ),
    }
    if f"{v1}|auc" in matrices:
        effects["v2_minus_version_1_success_by_push_auc"] = (
            matrices[f"{v2}|auc"] - matrices[f"{v1}|auc"]
        )
    return effects


def closed_loop_bootstrap(
    terminal_rows: list[dict[str, str]],
    resamples: int,
    scenario_index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """执行paired condition-target-stratified two-way bootstrap。"""

    condition_ids, target_ids, matrices, target_strata = paired_controller_matrices(
        terminal_rows
    )
    effects = closed_loop_effect_matrices(matrices)
    strata = {
        value: np.asarray(
            [index for index, target_id in enumerate(target_ids) if target_strata[target_id] == value]
        )
        for value in sorted(set(target_strata.values()))
    }
    rng = np.random.default_rng(np.random.SeedSequence([DEFAULT_SEED, scenario_index]))
    samples = {name: np.empty(resamples) for name in effects}
    for sample_index in range(resamples):
        c_counts = np.bincount(
            rng.integers(0, len(condition_ids), size=len(condition_ids)),
            minlength=len(condition_ids),
        ).astype(np.float64)
        t_counts = np.zeros(len(target_ids), dtype=np.float64)
        for indices in strata.values():
            local = rng.integers(0, len(indices), size=len(indices))
            t_counts[indices] = np.bincount(local, minlength=len(indices))
        weights = c_counts[:, None] * t_counts[None, :]
        denominator = float(weights.sum())
        for name, values in effects.items():
            samples[name][sample_index] = float(np.sum(weights * values) / denominator)
    summary = {
        name: {
            "point_estimate": float(np.mean(values)),
            "ci_95_low": float(np.quantile(samples[name], 0.025)),
            "ci_95_high": float(np.quantile(samples[name], 0.975)),
            "positive_effect_probability": float(np.mean(samples[name] > 0.0)),
            "bootstrap_resamples": resamples,
            "bootstrap_seed": DEFAULT_SEED,
        }
        for name, values in effects.items()
    }
    return summary, samples


def evaluate_closed_loop(args: argparse.Namespace) -> dict[str, Any]:
    """评价Version 2 closed-loop Validation或Independent Test。"""

    if select_scenarios(args.scenario) != SCENARIOS:
        raise ValueError("正式Version 2评价必须使用--scenario all")
    scenario_summaries: dict[str, Any] = {}
    scenario_samples: dict[str, dict[str, np.ndarray]] = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        step_rows, terminal_rows = load_evaluation_rows(scenario, args.role)
        active_controller_ids = tuple(
            controller_id
            for controller_id in EVALUATION_CONTROLLER_IDS
            if any(row["controller_id"] == controller_id for row in terminal_rows)
        )
        controllers = {
            controller_id: car_e4.e4_controller_summary(
                [row for row in terminal_rows if row["controller_id"] == controller_id],
                [row for row in step_rows if row["controller_id"] == controller_id],
            )
            for controller_id in active_controller_ids
        }
        effects, samples = closed_loop_bootstrap(
            terminal_rows, args.bootstrap_resamples, scenario_index
        )
        v2_terminal = [
            row
            for row in terminal_rows
            if row["controller_id"] in CONTINUOUS_CONTROLLER_IDS
        ]
        v2_steps = [
            row for row in step_rows if row["controller_id"] in CONTINUOUS_CONTROLLER_IDS
        ]
        summary = {
            "scenario": scenario,
            "role": args.role,
            "controllers": controllers,
            "paired_effects": effects,
            "executed_continuous_prediction_diagnostic": car_e4.executed_transfer_summary(
                v2_steps
            ),
            "on_policy_terminal_calibration": car_e4.on_policy_terminal_calibration(
                v2_terminal
            ),
        }
        scenario_summaries[scenario] = summary
        scenario_samples[scenario] = samples
        write_json(
            RESULTS_ROOT / "evaluation" / args.role / scenario / "summary.json",
            summary,
        )
        print(f"evaluated {scenario} {args.role}")
    effect_names = tuple(next(iter(scenario_samples.values())))
    macro_effects: dict[str, Any] = {}
    for name in effect_names:
        samples = np.mean(
            np.stack([scenario_samples[scenario][name] for scenario in SCENARIOS]),
            axis=0,
        )
        point = float(
            np.mean(
                [
                    scenario_summaries[scenario]["paired_effects"][name]["point_estimate"]
                    for scenario in SCENARIOS
                ]
            )
        )
        macro_effects[name] = {
            "point_estimate": point,
            "ci_95_low": float(np.quantile(samples, 0.025)),
            "ci_95_high": float(np.quantile(samples, 0.975)),
            "positive_effect_probability": float(np.mean(samples > 0.0)),
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": DEFAULT_SEED,
        }
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "role": args.role,
        "controller_names": {
            controller_id: EVALUATION_CONTROLLER_NAMES[controller_id]
            for controller_id in scenario_summaries[SCENARIOS[0]]["controllers"]
        },
        "scenario_summaries": scenario_summaries,
        "equal_weight_macro_effects": macro_effects,
    }
    if args.role == "validation":
        with SELECTED_BACKEND_PATH.open("r", encoding="utf-8") as handle:
            backend_selection = json.load(handle)
        with LIKELIHOOD_CONFIGURATION_PATH.open("r", encoding="utf-8") as handle:
            likelihood = json.load(handle)
        exact_selector_path = (
            RESULTS_ROOT
            / "exact_full_set_validation"
            / "selected_configuration.json"
        )
        exact_selector = json.loads(
            exact_selector_path.read_text(encoding="utf-8")
        )
        primary = BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
        criteria = {
            "model_gate_passed": bool(
                backend_selection["model_gate"]["passed_before_likelihood_calibration"]
            ),
            "likelihood_calibrated_in_all_scenarios": bool(
                likelihood["all_scenarios_calibrated"]
            ),
            "exact_full_set_selector_confirmed": bool(exact_selector["passed"]),
            "v2_vs_discrete_auc_ci_lower_above_zero": (
                macro_effects["v2_minus_discrete_success_by_push_auc"]["ci_95_low"]
                > 0.0
            ),
            "v2_vs_version_1_auc_ci_lower_above_zero": (
                macro_effects["v2_minus_version_1_success_by_push_auc"]["ci_95_low"]
                > 0.0
            ),
            "v2_vs_discrete_success_ci_lower_at_least_minus_0_02": (
                macro_effects["v2_minus_discrete_episode_success_rate"]["ci_95_low"]
                >= -0.02
            ),
            "continuous_selection_nonzero_in_all_scenarios": all(
                scenario_summaries[scenario]["controllers"][primary][
                    "continuous_promotion"
                ]["continuous_action_selection_rate_per_decision"]
                > 0.0
                for scenario in SCENARIOS
            ),
            "invalid_episode_rate_at_most_0_01": all(
                scenario_summaries[scenario]["controllers"][controller][
                    "invalid_episode_rate"
                ]
                <= 0.01
                for scenario in SCENARIOS
                for controller in CONTINUOUS_CONTROLLER_IDS
            ),
            "on_policy_terminal_hpd_coverage_at_least_0_90": all(
                scenario_summaries[scenario]["on_policy_terminal_calibration"][
                    "passed"
                ]
                for scenario in SCENARIOS
            ),
        }
        gate = {
            "protocol_version": PROTOCOL_VERSION,
            "criteria": criteria,
            "passed": all(criteria.values()),
        }
        combined["validation_gate"] = gate
        write_json(RESULTS_ROOT / "evaluation" / "validation" / "gate_decision.json", gate)
    write_json(
        RESULTS_ROOT / "evaluation" / args.role / "combined_summary.json",
        combined,
    )
    print(json.dumps(make_json_compatible(combined), ensure_ascii=False, indent=2))
    return combined


def build_parser() -> argparse.ArgumentParser:
    """构建CAR Experiment 4 Version 2统一命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="生成Version 2 model action split")

    collect_model = subparsers.add_parser(
        "collect-model-data", help="采集condition-conditioned model outcomes"
    )
    collect_model.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    collect_model.add_argument("--role", choices=("training", "validation"), required=True)
    collect_model.add_argument("--num-workers", type=int, default=8)
    collect_model.add_argument("--max-conditions", type=int, default=0)
    collect_model.add_argument("--resume", action="store_true")

    subparsers.add_parser("train-backends", help="顺序训练Candidate A与Candidate B")
    backend_eval = subparsers.add_parser(
        "evaluate-backends", help="执行outcome与selection-level Validation"
    )
    backend_eval.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )

    fit = subparsers.add_parser("fit-likelihood", help="拟合selected backend likelihood")
    fit.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    likelihood = subparsers.add_parser(
        "evaluate-likelihood", help="校准Student-t covariance inflation"
    )
    likelihood.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")

    on_policy_likelihood = subparsers.add_parser(
        "calibrate-on-policy-likelihood",
        help="用closed-loop Validation轨迹校准序贯似然温度",
    )
    on_policy_likelihood.add_argument(
        "--scenario", choices=(*SCENARIOS, "all"), default="all"
    )

    exact_selector = subparsers.add_parser(
        "evaluate-exact-selector",
        help="确认全部2,560个candidates的精确后验边缘化",
    )
    exact_selector.add_argument(
        "--scenario", choices=(*SCENARIOS, "all"), default="all"
    )
    exact_selector.add_argument(
        "--node-query-chunk-size", type=int, default=NODE_QUERY_CHUNK_SIZE
    )

    benchmark = subparsers.add_parser(
        "benchmark-workers", help="比较Version 2 closed-loop worker count"
    )
    benchmark.add_argument("--worker-counts", default="1,2,4")
    benchmark.add_argument(
        "--node-query-chunk-size", type=int, default=NODE_QUERY_CHUNK_SIZE
    )

    collect = subparsers.add_parser(
        "collect", help="采集Version 2 Validation或Independent Test trajectories"
    )
    collect.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    collect.add_argument("--role", choices=("validation", "test"), required=True)
    collect.add_argument("--controllers", default="all")
    collect.add_argument("--maximum-pushes", type=int, default=MAXIMUM_PUSHES)
    collect.add_argument("--num-workers", type=int, default=0)
    collect.add_argument("--max-conditions", type=int, default=0)
    collect.add_argument("--max-targets", type=int, default=0)
    collect.add_argument(
        "--node-query-chunk-size", type=int, default=NODE_QUERY_CHUNK_SIZE
    )
    collect.add_argument("--resume", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate", help="评价Version 2 closed-loop Validation或Test"
    )
    evaluate.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    evaluate.add_argument("--role", choices=("validation", "test"), required=True)
    evaluate.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """检查直接影响实验规模的命令行参数。"""

    for name in ("num_workers", "max_conditions", "max_targets"):
        if hasattr(args, name) and int(getattr(args, name)) < 0:
            raise ValueError(f"{name.replace('_', '-')}不能小于0")
    for name in (
        "maximum_pushes",
        "node_query_chunk_size",
        "bootstrap_resamples",
    ):
        if hasattr(args, name) and int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')}必须大于0")
    if hasattr(args, "worker_counts"):
        values = [
            int(value.strip()) for value in args.worker_counts.split(",") if value.strip()
        ]
        if not values or any(value <= 0 for value in values):
            raise ValueError("worker-counts必须是正整数列表")


def main() -> None:
    """执行CAR Experiment 4 Version 2子命令。"""

    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(args)
    print("CAR = Continuous Action Refinement")
    print("Experiment 4 Version 2 = Belief-Space Continuous Action Refinement")
    if args.command == "prepare":
        prepare_experiment()
    elif args.command == "collect-model-data":
        collect_model_data(args)
    elif args.command == "train-backends":
        train_backends()
    elif args.command == "evaluate-backends":
        evaluate_backends(args)
    elif args.command == "fit-likelihood":
        fit_likelihood(args)
    elif args.command == "evaluate-likelihood":
        evaluate_likelihood(args)
    elif args.command == "calibrate-on-policy-likelihood":
        calibrate_on_policy_likelihood(args)
    elif args.command == "evaluate-exact-selector":
        evaluate_exact_selector(args)
    elif args.command == "benchmark-workers":
        benchmark_workers(args)
    elif args.command == "collect":
        collect_closed_loop(args)
    elif args.command == "evaluate":
        evaluate_closed_loop(args)


if __name__ == "__main__":
    main()
