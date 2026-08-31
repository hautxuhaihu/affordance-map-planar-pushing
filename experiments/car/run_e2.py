"""CAR（Continuous Action Refinement）Experiment 2 统一入口。

CAR 是 Continuous Action Refinement 的代码目录缩写。本脚本实现：

- prepare：生成 source-anchor split、ranking targets 与补采 candidates；
- collect：并行采集缺失 source anchors 的 MuJoCo outcomes；
- train：构造紧凑数据集并顺序训练 Direct 与 P1-Anchored models；
- evaluate-validation / evaluate-test：评价 outcome prediction 与离线 ranking。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_e1 as car_e1  # noqa: E402
from push_core.hcr_v2.e1 import (  # noqa: E402
    PRIMARY_TNPO_COST,
    TensorOutcomeInterpolator,
    read_csv_rows,
    wrap_to_pi,
)


PROTOCOL_VERSION = "continuous_action_refinement_e2_v1"
CAR_FULL_NAME = "Continuous Action Refinement"
EXPERIMENT_NAME = "Tensor-Interpolation-Anchored Continuous Outcome Prediction"
FRICTION_CONE = "elliptic"
DEFAULT_SEED = 0
NOMINAL_CONDITION_ID = "J_TRAIN_062"

MANIFEST_ROOT = REPOSITORY_ROOT / "manifests" / "car"
HCR_MANIFEST_ROOT = (
    REPOSITORY_ROOT / "manifests" / "hcr_v2"
)
ACTION_MANIFEST_PATH = HCR_MANIFEST_ROOT / "hcr_v2_action_core_manifest_v1.csv"
TARGET_SOURCE_PATH = HCR_MANIFEST_ROOT / "hcr_v2_core_target_manifest_v1.csv"
E1_TARGET_PATH = MANIFEST_ROOT / "continuous_action_refinement_validation_targets.csv"
E1_ANCHOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "car"
    / "experiment_1"
    / "continuous_action_refinement_anchors.csv"
)
E1_OUTCOME_PATH = (
    PROJECT_ROOT
    / "data"
    / "car"
    / "experiment_1"
    / "continuous_action_refinement_unique_outcomes.csv"
)
P1_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "results"
    / "hcr_v2"
    / "e1"
    / "p1"
    / "joint"
    / "tensor_outcome_interpolator.npz"
)
NOMINAL_DISCRETE_OUTCOME_PATH = (
    PROJECT_ROOT
    / "data"
    / "hcr_v2"
    / "e1"
    / "outcomes"
    / "joint"
    / "training"
    / "J_TRAIN_062.csv"
)

DATA_ROOT = PROJECT_ROOT / "data" / "car" / "experiment_2"
RESULTS_ROOT = PROJECT_ROOT / "results" / "car" / "experiment_2"
MODEL_ROOT = RESULTS_ROOT / "models"
ANCHOR_SPLIT_PATH = MANIFEST_ROOT / "continuous_action_refinement_e2_anchor_split.csv"
VALIDATION_TARGET_PATH = (
    MANIFEST_ROOT / "continuous_action_refinement_e2_validation_target_queries.csv"
)
TEST_TARGET_PATH = MANIFEST_ROOT / "continuous_action_refinement_e2_test_target_queries.csv"
ADDITIONAL_CANDIDATE_PATH = (
    DATA_ROOT / "continuous_action_refinement_additional_candidates.csv"
)
ADDITIONAL_OUTCOME_PATH = (
    DATA_ROOT / "continuous_action_refinement_additional_outcomes.csv"
)
DATASET_PATH = DATA_ROOT / "continuous_action_refinement_prediction_dataset.npz"
DIRECT_MODEL_PATH = MODEL_ROOT / "direct_continuous_outcome_model.pt"
ANCHORED_MODEL_PATH = MODEL_ROOT / "p1_anchored_continuous_residual_model.pt"

ACTION_COUNT = 4_536
CANDIDATES_PER_ANCHOR = 512
TRAINING_ANCHORS_PER_STRATUM = 264
VALIDATION_ANCHORS_PER_STRATUM = 57
TEST_ANCHORS_PER_STRATUM = 57
ROLE_CODES = {"training": 0, "validation": 1, "test": 2}
ROLE_NAMES = {value: key for key, value in ROLE_CODES.items()}
SAMPLE_BUDGETS = (128, 256, 512)
PRIMARY_SAMPLE_BUDGET = 512
RANKING_ANCHOR_BUDGET = 20
NEAR_OPTIMAL_PRIMARY = 0.05
NEAR_OPTIMAL_SENSITIVITY = 0.02

OFFSET_SCALES = np.asarray(
    [0.015, 7.5, 0.15, 0.010, 0.025, 0.010], dtype=np.float32
)
ACTION_CENTRES = np.asarray(
    [0.0, 0.0, 0.80, 0.040, 0.100, 0.030], dtype=np.float32
)
ACTION_SCALES = np.asarray(
    [0.045, 45.0, 0.30, 0.020, 0.050, 0.010], dtype=np.float32
)

MAXIMUM_EPOCHS = 1_200
TRAINING_BATCH_SIZE = 131_072
EVALUATION_BATCH_SIZE = 131_072
INITIAL_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
MINIMUM_LEARNING_RATE = 1e-6
EARLY_STOPPING_PATIENCE = 30
MINIMUM_IMPROVEMENT = 1e-6

ANCHOR_SPLIT_FIELDS = [
    "v2_action_id",
    "source_action_index",
    "surface_id",
    "contact_region_col",
    "physical_stratum_index",
    "permuted_rank_within_stratum",
    "split_role",
    "split_seed",
]
TARGET_FIELDS = [
    "e2_target_index",
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
ADDITIONAL_CANDIDATE_FIELDS = [
    *car_e1.UNIQUE_ACTION_FIELDS,
    "minimum_candidate_budget",
    "split_role",
]
ADDITIONAL_OUTCOME_FIELDS = car_e1.OUTCOME_FIELDS


def make_json_compatible(value: Any) -> Any:
    """把 NumPy 类型和非有限浮点数转换为严格 JSON 值。"""

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


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """固定 Python、NumPy 与 PyTorch 随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_cuda() -> torch.device:
    """返回正式训练和评价使用的 CUDA device。"""

    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 2 GPU-first implementation 需要可用的 CUDA GPU")
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def autocast_context():
    """为 MLP 矩阵计算启用 BF16 autocast。"""

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def action_coordinate(row: dict[str, Any]) -> np.ndarray:
    """把 action row 转换为正式六维连续坐标。"""

    surface_id = int(row["surface_id"])
    tangent = float(
        row["contact_point_local_y"]
        if surface_id in {0, 1}
        else row["contact_point_local_x"]
    )
    return np.asarray(
        [
            tangent,
            float(row["force_angle_relative_to_normal_deg"]),
            float(row["commanded_force_N"]),
            float(row["ramp_up_s"]),
            float(row["hold_s"]),
            float(row["ramp_down_s"]),
        ],
        dtype=np.float32,
    )


def build_anchor_split(actions: list[dict[str, str]]) -> list[dict[str, Any]]:
    """按 12 个 physical strata 生成 source-anchor-disjoint split。"""

    strata: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for action in actions:
        strata[(int(action["surface_id"]), int(action["contact_region_col"]))].append(
            action
        )
    rows: list[dict[str, Any]] = []
    for surface_id in range(4):
        for contact_col in range(3):
            ordered = sorted(
                strata[(surface_id, contact_col)], key=lambda row: row["v2_action_id"]
            )
            if len(ordered) != 378:
                raise RuntimeError(
                    f"physical stratum 大小错误: surface={surface_id}, "
                    f"col={contact_col}, observed={len(ordered)}"
                )
            seed_sequence = np.random.SeedSequence(
                [DEFAULT_SEED, surface_id, contact_col]
            )
            permutation = np.random.default_rng(seed_sequence).permutation(len(ordered))
            for permuted_rank, source_index in enumerate(permutation):
                if permuted_rank < TRAINING_ANCHORS_PER_STRATUM:
                    role = "training"
                elif permuted_rank < (
                    TRAINING_ANCHORS_PER_STRATUM + VALIDATION_ANCHORS_PER_STRATUM
                ):
                    role = "validation"
                else:
                    role = "test"
                action = ordered[int(source_index)]
                rows.append(
                    {
                        "v2_action_id": action["v2_action_id"],
                        "source_action_index": int(action["v2_action_id"][1:]),
                        "surface_id": surface_id,
                        "contact_region_col": contact_col,
                        "physical_stratum_index": surface_id * 3 + contact_col,
                        "permuted_rank_within_stratum": permuted_rank,
                        "split_role": role,
                        "split_seed": DEFAULT_SEED,
                    }
                )
    return sorted(rows, key=lambda row: row["source_action_index"])


def angular_quadrant(delta_x: float, delta_y: float) -> int:
    """按固定坐标轴边界返回 angular quadrant。"""

    if delta_x >= 0.0 and delta_y >= 0.0:
        return 0
    if delta_x < 0.0 and delta_y >= 0.0:
        return 1
    if delta_x < 0.0 and delta_y < 0.0:
        return 2
    return 3


def build_ranking_targets() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """生成互斥且不与 Experiment 1 重叠的 Validation/Test targets。"""

    excluded_ids = {row["v2_target_id"] for row in read_csv_rows(E1_TARGET_PATH)}
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
    if len(candidates) % 4 != 0:
        raise RuntimeError("排除 Experiment 1 targets 后无法等分 radial quartiles")
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
    for radial_index in range(4):
        for angular_index in range(4):
            ordered = sorted(
                strata[(radial_index, angular_index)],
                key=lambda row: row["v2_target_id"],
            )
            seed_sequence = np.random.SeedSequence(
                [DEFAULT_SEED, radial_index, angular_index]
            )
            permutation = np.random.default_rng(seed_sequence).permutation(len(ordered))
            for role, selected_indices in (
                ("validation", permutation[:8]),
                ("test", permutation[8:16]),
            ):
                for selected_index in selected_indices:
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
            {**row, "e2_target_index": index} for index, row in enumerate(rows)
        ]
    return outputs["validation"], outputs["test"]


def prepare_additional_candidates(
    actions: list[dict[str, str]],
    split_by_action_id: dict[str, str],
) -> tuple[int, int]:
    """为 Experiment 1 未覆盖的 source anchors 生成补充 candidates。"""

    e1_anchor_ids = {row["v2_action_id"] for row in read_csv_rows(E1_ANCHOR_PATH)}
    missing_actions = [row for row in actions if row["v2_action_id"] not in e1_anchor_ids]
    if len(e1_anchor_ids) != 2_178 or len(missing_actions) != 2_358:
        raise RuntimeError(
            f"E1/E2 anchor 数量错误: existing={len(e1_anchor_ids)}, "
            f"missing={len(missing_actions)}"
        )
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    first_physical_id = len(e1_anchor_ids) * CANDIDATES_PER_ANCHOR
    physical_id = first_physical_id
    with ADDITIONAL_CANDIDATE_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ADDITIONAL_CANDIDATE_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for anchor_number, action in enumerate(missing_actions, start=1):
            role = split_by_action_id[action["v2_action_id"]]
            for candidate in car_e1.generate_anchor_candidates(action):
                row = {
                    **candidate,
                    "physical_action_id": physical_id,
                    "physical_action_key": car_e1.physical_action_key(candidate),
                    "split_role": role,
                }
                writer.writerow(row)
                physical_id += 1
            if anchor_number % 100 == 0 or anchor_number == len(missing_actions):
                print(
                    f"prepared additional anchor {anchor_number}/{len(missing_actions)}; "
                    f"candidates={physical_id - first_physical_id}"
                )
    return len(missing_actions), physical_id - first_physical_id


def verify_physical_action_identity() -> int:
    """确认完整数据中不存在跨 source anchors 的重复 physical action key。"""

    keys: set[str] = set()
    for path in (E1_OUTCOME_PATH, ADDITIONAL_CANDIDATE_PATH):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = row["physical_action_key"]
                if key in keys:
                    raise RuntimeError(f"重复 physical action key: {key}")
                keys.add(key)
    expected = ACTION_COUNT * CANDIDATES_PER_ANCHOR
    if len(keys) != expected:
        raise RuntimeError(
            f"physical action key 数量错误: expected={expected}, observed={len(keys)}"
        )
    return len(keys)


def prepare_experiment() -> dict[str, Any]:
    """生成 Experiment 2 正式 manifests 与 additional candidates。"""

    actions = sorted(
        read_csv_rows(ACTION_MANIFEST_PATH), key=lambda row: row["v2_action_id"]
    )
    if len(actions) != ACTION_COUNT:
        raise RuntimeError(f"action_core 数量错误: {len(actions)}")
    split_rows = build_anchor_split(actions)
    split_by_action_id = {
        row["v2_action_id"]: row["split_role"] for row in split_rows
    }
    validation_targets, test_targets = build_ranking_targets()
    write_csv(ANCHOR_SPLIT_PATH, split_rows, ANCHOR_SPLIT_FIELDS)
    write_csv(VALIDATION_TARGET_PATH, validation_targets, TARGET_FIELDS)
    write_csv(TEST_TARGET_PATH, test_targets, TARGET_FIELDS)
    additional_anchor_count, additional_candidate_count = prepare_additional_candidates(
        actions, split_by_action_id
    )
    unique_physical_action_count = verify_physical_action_identity()

    role_anchor_counts = {
        role: sum(row["split_role"] == role for row in split_rows)
        for role in ROLE_CODES
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "car_full_name": CAR_FULL_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "friction_cone": FRICTION_CONE,
        "nominal_condition_id": NOMINAL_CONDITION_ID,
        "default_seed": DEFAULT_SEED,
        "source_anchors": len(actions),
        "role_anchor_counts": role_anchor_counts,
        "e1_reused_anchors": len(actions) - additional_anchor_count,
        "additional_anchors": additional_anchor_count,
        "additional_candidates": additional_candidate_count,
        "unique_physical_actions": unique_physical_action_count,
        "complete_continuous_outcomes": len(actions) * CANDIDATES_PER_ANCHOR,
        "validation_ranking_targets": len(validation_targets),
        "test_ranking_targets": len(test_targets),
        "anchor_split_path": str(ANCHOR_SPLIT_PATH.resolve()),
        "validation_target_path": str(VALIDATION_TARGET_PATH.resolve()),
        "test_target_path": str(TEST_TARGET_PATH.resolve()),
        "additional_candidate_path": str(ADDITIONAL_CANDIDATE_PATH.resolve()),
    }
    write_json(RESULTS_ROOT / "preparation_summary.json", summary)
    return summary


def initialise_rollout_worker() -> None:
    """复用 Experiment 1 的 Elliptic nominal-condition worker 初始化。"""

    car_e1.initialise_rollout_worker()


def process_action_batch(actions: list[dict[str, str]]) -> list[dict[str, Any]]:
    """执行一批 MuJoCo rollouts，并改写为 Experiment 2 元数据。"""

    rows = car_e1.process_action_batch(actions)
    for row in rows:
        row["experiment_id"] = "E2"
        row["protocol_version"] = PROTOCOL_VERSION
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
    start_physical_id: int,
    batch_size: int,
) -> Iterator[list[dict[str, str]]]:
    """按 physical action id 流式生成补充采集 batches。"""

    batch: list[dict[str, str]] = []
    with ADDITIONAL_CANDIDATE_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if int(row["physical_action_id"]) < start_physical_id:
                continue
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def collect_outcomes(num_workers: int, batch_size: int, resume: bool) -> dict[str, Any]:
    """使用 14-worker nominal MuJoCo protocol 采集 additional outcomes。"""

    if not ADDITIONAL_CANDIDATE_PATH.exists():
        raise FileNotFoundError("请先运行 prepare 生成 additional candidates")
    started_at = time.perf_counter()
    ADDITIONAL_OUTCOME_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_count, last_id = (
        count_existing_outcomes(ADDITIONAL_OUTCOME_PATH) if resume else (0, -1)
    )
    if not resume and ADDITIONAL_OUTCOME_PATH.exists():
        ADDITIONAL_OUTCOME_PATH.unlink()
    first_physical_id = 2_178 * CANDIDATES_PER_ANCHOR
    start_physical_id = last_id + 1 if existing_count else first_physical_id
    batches = iter_action_batches(start_physical_id, max(1, int(batch_size)))
    mode = "a" if existing_count else "w"
    processed = 0
    valid = 0
    with ADDITIONAL_OUTCOME_PATH.open(
        mode, encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=ADDITIONAL_OUTCOME_FIELDS, extrasaction="ignore"
        )
        if not existing_count:
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
                    print(f"collected {existing_count + processed} additional actions")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    elapsed = time.perf_counter() - started_at
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "num_workers": max(1, int(num_workers)),
        "batch_size": max(1, int(batch_size)),
        "resumed_rows": existing_count,
        "new_rollouts": processed,
        "new_valid_rollouts": valid,
        "elapsed_seconds": elapsed,
        "new_rollouts_per_second": processed / elapsed if processed else 0.0,
        "outcome_path": str(ADDITIONAL_OUTCOME_PATH.resolve()),
    }
    write_json(RESULTS_ROOT / "collection_summary.json", summary)
    return summary


def nominal_p1_outcomes(actions: list[dict[str, str]]) -> np.ndarray:
    """通过冻结的 Joint P1 artifact 读取 nominal source-anchor outcomes。"""

    interpolator = TensorOutcomeInterpolator.load(P1_ARTIFACT_PATH)
    prediction = interpolator.predict(
        {
            "hidden_u_friction": 0.0,
            "hidden_u_com_x": 0.0,
            "hidden_u_com_y": 0.0,
        }
    )
    p1_by_id = {
        action_id: prediction[index]
        for index, action_id in enumerate(interpolator.action_ids)
    }
    return np.asarray([p1_by_id[row["v2_action_id"]] for row in actions], dtype=np.float32)


def build_prediction_dataset() -> dict[str, Any]:
    """把 E1 与 E2 raw outcomes 合并为训练使用的紧凑 NPZ。"""

    if not ADDITIONAL_OUTCOME_PATH.exists():
        raise FileNotFoundError("请先完成 Experiment 2 additional outcome collection")
    actions = sorted(
        read_csv_rows(ACTION_MANIFEST_PATH), key=lambda row: row["v2_action_id"]
    )
    split_rows = read_csv_rows(ANCHOR_SPLIT_PATH)
    split_codes = np.empty(ACTION_COUNT, dtype=np.uint8)
    stratum = np.empty(ACTION_COUNT, dtype=np.uint8)
    for row in split_rows:
        index = int(row["source_action_index"])
        split_codes[index] = ROLE_CODES[row["split_role"]]
        stratum[index] = int(row["physical_stratum_index"])
    anchor_q = np.asarray([action_coordinate(row) for row in actions], dtype=np.float32)
    p1_outcomes = nominal_p1_outcomes(actions)

    expected_rows = ACTION_COUNT * CANDIDATES_PER_ANCHOR
    source_action_index = np.empty(expected_rows, dtype=np.int32)
    sobol_sample_index = np.empty(expected_rows, dtype=np.uint16)
    role = np.empty(expected_rows, dtype=np.uint8)
    region = np.empty(expected_rows, dtype=np.uint8)
    continuous_q = np.empty((expected_rows, 6), dtype=np.float32)
    true_outcome = np.empty((expected_rows, 3), dtype=np.float32)
    valid = np.empty(expected_rows, dtype=np.bool_)
    seen = np.zeros((ACTION_COUNT, CANDIDATES_PER_ANCHOR), dtype=np.bool_)

    row_index = 0
    for path in (E1_OUTCOME_PATH, ADDITIONAL_OUTCOME_PATH):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                action_index = int(row["source_action_index"])
                sample_index = int(row["sobol_sample_index"])
                if seen[action_index, sample_index]:
                    raise RuntimeError(
                        f"重复 continuous action: action={action_index}, sample={sample_index}"
                    )
                seen[action_index, sample_index] = True
                source_action_index[row_index] = action_index
                sobol_sample_index[row_index] = sample_index
                role[row_index] = split_codes[action_index]
                region[row_index] = stratum[action_index]
                continuous_q[row_index] = action_coordinate(row)
                true_outcome[row_index] = [
                    float(row["real_delta_x"]),
                    float(row["real_delta_y"]),
                    float(row["real_delta_yaw"]),
                ]
                valid[row_index] = (
                    int(row["quality_pass"]) == 1
                    and int(row["simulation_unstable"]) == 0
                    and int(row["contact_success"]) == 1
                    and int(row["stopped_by_threshold"]) == 1
                )
                row_index += 1
                if row_index % 100_000 == 0:
                    print(
                        f"prepared compact dataset rows {row_index}/{expected_rows}"
                    )
    if row_index != expected_rows or not np.all(seen):
        raise RuntimeError(
            f"完整 continuous dataset 不完整: expected={expected_rows}, observed={row_index}"
        )

    print(f"saving compact dataset: {DATASET_PATH}")
    np.savez(
        DATASET_PATH,
        protocol_version=np.asarray(PROTOCOL_VERSION),
        source_action_index=source_action_index,
        sobol_sample_index=sobol_sample_index,
        role=role,
        region=region,
        continuous_q=continuous_q,
        anchor_q=anchor_q,
        p1_anchor_outcome=p1_outcomes,
        true_outcome=true_outcome,
        valid=valid,
    )
    role_summary: dict[str, Any] = {}
    for role_name, role_code in ROLE_CODES.items():
        mask = role == role_code
        role_summary[role_name] = {
            "rows": int(mask.sum()),
            "valid_rows": int(np.sum(mask & valid)),
            "validity_rate": float(np.mean(valid[mask])),
        }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "rows": expected_rows,
        "source_anchors": ACTION_COUNT,
        "role_summary": role_summary,
        "dataset_path": str(DATASET_PATH.resolve()),
    }
    write_json(RESULTS_ROOT / "dataset_summary.json", summary)
    return summary


class OutcomeMLP(nn.Module):
    """Experiment 2 固定的 128×128 SiLU outcome MLP。"""

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
        """预测标准化 outcome 或 residual。"""

        return self.network(inputs)


def safe_scale(values: np.ndarray) -> np.ndarray:
    """计算 Training standard deviation，并避免真实零尺度。"""

    scale = values.std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return scale


def build_model_arrays(
    dataset: dict[str, np.ndarray],
    model_type: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """构造一个模型的完整 features、labels 与 Training normalisers。"""

    source = dataset["source_action_index"]
    region = dataset["region"]
    continuous_q = dataset["continuous_q"]
    anchor_q = dataset["anchor_q"][source]
    p1 = dataset["p1_anchor_outcome"][source]
    true = dataset["true_outcome"]
    training_mask = (dataset["role"] == ROLE_CODES["training"]) & dataset["valid"]
    one_hot = np.eye(12, dtype=np.float32)[region]
    q_scaled = (continuous_q - ACTION_CENTRES) / ACTION_SCALES
    if model_type == "direct":
        features = np.concatenate([one_hot, q_scaled], axis=1).astype(np.float32)
        label_mean = true[training_mask].mean(axis=0).astype(np.float32)
        label_scale = safe_scale(true[training_mask])
        labels = ((true - label_mean) / label_scale).astype(np.float32)
        normalisers = {
            "label_mean": label_mean,
            "label_scale": label_scale,
        }
        return features, labels, normalisers

    p1_mean = p1[training_mask].mean(axis=0).astype(np.float32)
    p1_scale = safe_scale(p1[training_mask])
    residual = true - p1
    residual_mean = residual[training_mask].mean(axis=0).astype(np.float32)
    residual_scale = safe_scale(residual[training_mask])
    features = np.concatenate(
        [
            one_hot,
            (anchor_q - ACTION_CENTRES) / ACTION_SCALES,
            (continuous_q - anchor_q) / OFFSET_SCALES,
            (p1 - p1_mean) / p1_scale,
        ],
        axis=1,
    ).astype(np.float32)
    labels = ((residual - residual_mean) / residual_scale).astype(np.float32)
    normalisers = {
        "p1_mean": p1_mean,
        "p1_scale": p1_scale,
        "label_mean": residual_mean,
        "label_scale": residual_scale,
    }
    return features, labels, normalisers


def grouped_scaled_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算 position 与 yaw 同权的 grouped scaled MSE。"""

    squared = (prediction.float() - target.float()).square()
    position = squared[:, :2].mean()
    yaw = squared[:, 2].mean()
    return 0.5 * position + 0.5 * yaw


def target_normalized_error(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """计算 Target-Normalized Outcome Prediction Error。"""

    position = np.linalg.norm(prediction[:, :2] - target[:, :2], axis=1)
    yaw = np.abs(wrap_to_pi(prediction[:, 2] - target[:, 2]))
    return (
        0.5 * position / PRIMARY_TNPO_COST.position_tolerance_m
        + 0.5 * yaw / PRIMARY_TNPO_COST.yaw_tolerance_rad
    )


def validation_score(
    model: OutcomeMLP,
    features: torch.Tensor,
    row_indices: torch.Tensor,
    true_outcome: np.ndarray,
    p1_outcome: np.ndarray,
    normalisers: dict[str, np.ndarray],
    model_type: str,
) -> float:
    """用 Validation mean Target-Normalized error 选择 checkpoint。"""

    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, row_indices.numel(), EVALUATION_BATCH_SIZE):
            indices = row_indices[start : start + EVALUATION_BATCH_SIZE]
            with autocast_context():
                scaled = model(features[indices])
            predictions.append(scaled.float().cpu().numpy())
    scaled_prediction = np.concatenate(predictions, axis=0)
    selected_rows = row_indices.cpu().numpy()
    physical = (
        scaled_prediction * normalisers["label_scale"]
        + normalisers["label_mean"]
    )
    if model_type == "anchored":
        physical = physical + p1_outcome[selected_rows]
    return float(np.mean(target_normalized_error(physical, true_outcome[selected_rows])))


def train_one_model(
    dataset: dict[str, np.ndarray],
    model_type: str,
    output_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """在单 CUDA 进程中训练一个正式 outcome model。"""

    set_seed(DEFAULT_SEED)
    features_np, labels_np, normalisers = build_model_arrays(dataset, model_type)
    valid = dataset["valid"]
    training_indices_np = np.flatnonzero(
        (dataset["role"] == ROLE_CODES["training"]) & valid
    ).astype(np.int64)
    validation_indices_np = np.flatnonzero(
        (dataset["role"] == ROLE_CODES["validation"]) & valid
    ).astype(np.int64)
    features = torch.from_numpy(features_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    training_indices = torch.from_numpy(training_indices_np).to(device)
    validation_indices = torch.from_numpy(validation_indices_np).to(device)
    p1_rows = dataset["p1_anchor_outcome"][dataset["source_action_index"]]

    model = OutcomeMLP(features.shape[1]).to(device)
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
    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = training_indices[
            torch.randperm(training_indices.numel(), generator=generator, device=device)
        ]
        loss_sum = 0.0
        row_count = 0
        for start in range(0, order.numel(), TRAINING_BATCH_SIZE):
            indices = order[start : start + TRAINING_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                prediction = model(features[indices])
            loss = grouped_scaled_mse(prediction, labels[indices])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * indices.numel()
            row_count += indices.numel()
        validation = validation_score(
            model,
            features,
            validation_indices,
            dataset["true_outcome"],
            p1_rows,
            normalisers,
            model_type,
        )
        scheduler.step(validation)
        if validation < best_score - MINIMUM_IMPROVEMENT:
            best_score = validation
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "training_grouped_scaled_mse": loss_sum / row_count,
            "validation_mean_target_normalized_error": validation,
            "best_validation_mean_target_normalized_error": best_score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "stale_epochs": stale_epochs,
            "seconds": seconds,
        }
        history.append(row)
        print(
            f"{model_type} epoch {epoch:03d}/{MAXIMUM_EPOCHS}  "
            f"train={row['training_grouped_scaled_mse']:.6f}  "
            f"validation={validation:.6f}  best={best_score:.6f}  "
            f"lr={row['learning_rate']:.1e}  seconds={seconds:.2f}  "
            f"stale={stale_epochs}/{EARLY_STOPPING_PATIENCE}"
        )
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"{model_type} training 没有生成有效 checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "model_type": model_type,
            "input_dimension": int(features.shape[1]),
            "hidden_dimensions": [128, 128],
            "activation": "SiLU",
            "state_dict": best_state,
            "normalisers": {
                key: torch.from_numpy(value.copy()) for key, value in normalisers.items()
            },
            "best_epoch": best_epoch,
            "best_validation_mean_target_normalized_error": best_score,
            "training_seed": DEFAULT_SEED,
        },
        output_path,
    )
    history_path = MODEL_ROOT / f"{model_type}_training_history.csv"
    write_csv(history_path, history, list(history[0]))
    summary = {
        "model_type": model_type,
        "checkpoint_path": str(output_path.resolve()),
        "training_rows": len(training_indices_np),
        "validation_rows": len(validation_indices_np),
        "input_dimension": int(features.shape[1]),
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_mean_target_normalized_error": best_score,
        "elapsed_seconds": time.perf_counter() - started_at,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
    }
    del features, labels, training_indices, validation_indices, model, optimizer
    torch.cuda.empty_cache()
    return summary


def load_dataset() -> dict[str, np.ndarray]:
    """读取正式 compact prediction dataset。"""

    with np.load(DATASET_PATH, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files if key != "protocol_version"}


def train_models() -> dict[str, Any]:
    """构造 compact dataset，并顺序训练 Direct 与 P1-Anchored models。"""

    if DATASET_PATH.exists():
        existing_dataset = load_dataset()
        role_summary: dict[str, Any] = {}
        for role_name, role_code in ROLE_CODES.items():
            mask = existing_dataset["role"] == role_code
            role_summary[role_name] = {
                "rows": int(mask.sum()),
                "valid_rows": int(np.sum(mask & existing_dataset["valid"])),
                "validity_rate": float(np.mean(existing_dataset["valid"][mask])),
            }
        dataset_summary = {
            "protocol_version": PROTOCOL_VERSION,
            "rows": len(existing_dataset["role"]),
            "source_anchors": ACTION_COUNT,
            "role_summary": role_summary,
            "dataset_path": str(DATASET_PATH.resolve()),
        }
        del existing_dataset
    else:
        dataset_summary = build_prediction_dataset()
    if dataset_summary["role_summary"]["training"]["validity_rate"] < 0.99:
        raise RuntimeError("Training continuous rollout validity rate 低于 99%")
    device = require_cuda()
    dataset = load_dataset()
    direct_summary = train_one_model(dataset, "direct", DIRECT_MODEL_PATH, device)
    anchored_summary = train_one_model(dataset, "anchored", ANCHORED_MODEL_PATH, device)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "device": torch.cuda.get_device_name(device),
        "autocast_dtype": "bfloat16",
        "float32_matmul_precision": "high",
        "training_batch_size": TRAINING_BATCH_SIZE,
        "num_workers": 0,
        "direct": direct_summary,
        "anchored": anchored_summary,
    }
    write_json(RESULTS_ROOT / "training_summary.json", summary)
    return summary


METHOD_NAMES = (
    "source_anchor_p1_baseline",
    "direct_continuous_outcome_model",
    "p1_anchored_continuous_residual_model",
)


def load_checkpoint_model(
    path: Path,
    device: torch.device,
) -> tuple[OutcomeMLP, dict[str, np.ndarray], str]:
    """读取一个正式 checkpoint、normalisers 与 model type。"""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_type = str(checkpoint["model_type"])
    model = OutcomeMLP(int(checkpoint["input_dimension"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    normalisers = {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in checkpoint["normalisers"].items()
    }
    return model, normalisers, model_type


def evaluation_features(
    dataset: dict[str, np.ndarray],
    row_indices: np.ndarray,
    model_type: str,
    normalisers: dict[str, np.ndarray],
) -> np.ndarray:
    """只为一个 evaluation role 构造模型输入。"""

    source = dataset["source_action_index"][row_indices]
    region = dataset["region"][row_indices]
    continuous_q = dataset["continuous_q"][row_indices]
    anchor_q = dataset["anchor_q"][source]
    one_hot = np.eye(12, dtype=np.float32)[region]
    if model_type == "direct":
        return np.concatenate(
            [one_hot, (continuous_q - ACTION_CENTRES) / ACTION_SCALES], axis=1
        ).astype(np.float32)
    p1 = dataset["p1_anchor_outcome"][source]
    return np.concatenate(
        [
            one_hot,
            (anchor_q - ACTION_CENTRES) / ACTION_SCALES,
            (continuous_q - anchor_q) / OFFSET_SCALES,
            (p1 - normalisers["p1_mean"]) / normalisers["p1_scale"],
        ],
        axis=1,
    ).astype(np.float32)


def predict_role(
    dataset: dict[str, np.ndarray],
    row_indices: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
) -> np.ndarray:
    """批量预测一个 role 的 continuous outcomes。"""

    model, normalisers, model_type = load_checkpoint_model(checkpoint_path, device)
    features_np = evaluation_features(dataset, row_indices, model_type, normalisers)
    features = torch.from_numpy(features_np).to(device)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(row_indices), EVALUATION_BATCH_SIZE):
            stop = min(len(row_indices), start + EVALUATION_BATCH_SIZE)
            with autocast_context():
                scaled = model(features[start:stop])
            batches.append(scaled.float().cpu().numpy())
    scaled_prediction = np.concatenate(batches, axis=0)
    prediction = (
        scaled_prediction * normalisers["label_scale"]
        + normalisers["label_mean"]
    )
    if model_type == "anchored":
        source = dataset["source_action_index"][row_indices]
        prediction += dataset["p1_anchor_outcome"][source]
    del model, features
    torch.cuda.empty_cache()
    return prediction.astype(np.float32)


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    """计算 mean、median 与 P90。"""

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def prediction_metric_summary(
    prediction: np.ndarray,
    truth: np.ndarray,
    common_output_scale: np.ndarray,
) -> dict[str, Any]:
    """计算一个 predictor 的正式 outcome-prediction metrics。"""

    planar_m = np.linalg.norm(prediction[:, :2] - truth[:, :2], axis=1)
    yaw_rad = np.abs(wrap_to_pi(prediction[:, 2] - truth[:, 2]))
    signed = prediction - truth
    signed[:, 2] = wrap_to_pi(signed[:, 2])
    scaled_squared = ((prediction - truth) / common_output_scale) ** 2
    grouped_scaled_mse = 0.5 * scaled_squared[:, :2].mean() + 0.5 * scaled_squared[
        :, 2
    ].mean()
    return {
        "target_normalized_outcome_prediction_error": percentile_summary(
            target_normalized_error(prediction, truth)
        ),
        "planar_outcome_error_mm": percentile_summary(planar_m * 1000.0),
        "yaw_outcome_error_deg": percentile_summary(np.degrees(yaw_rad)),
        "signed_bias": {
            "delta_x_m": float(np.mean(signed[:, 0])),
            "delta_y_m": float(np.mean(signed[:, 1])),
            "delta_yaw_deg": float(np.degrees(np.mean(signed[:, 2]))),
        },
        "grouped_scaled_mse": float(grouped_scaled_mse),
    }


def stratified_bootstrap_means(
    unit_values: np.ndarray,
    unit_strata: np.ndarray,
    resamples: int,
) -> np.ndarray:
    """在固定 strata 内成对重采样统计单位。"""

    values = np.asarray(unit_values, dtype=np.float64)
    strata = np.asarray(unit_strata, dtype=np.int64)
    rng = np.random.default_rng(DEFAULT_SEED)
    accumulated = np.zeros(resamples, dtype=np.float64)
    total_units = 0
    for stratum_id in sorted(np.unique(strata)):
        stratum_values = values[strata == stratum_id]
        draws = rng.integers(
            0, len(stratum_values), size=(resamples, len(stratum_values))
        )
        accumulated += stratum_values[draws].sum(axis=1)
        total_units += len(stratum_values)
    return accumulated / total_units


def bootstrap_ci(values: np.ndarray) -> list[float]:
    """返回 percentile 95% confidence interval。"""

    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def build_candidate_matrix(dataset: dict[str, np.ndarray]) -> np.ndarray:
    """建立 source anchor × Sobol index 到 compact row 的映射。"""

    matrix = np.full((ACTION_COUNT, CANDIDATES_PER_ANCHOR), -1, dtype=np.int32)
    matrix[
        dataset["source_action_index"], dataset["sobol_sample_index"].astype(np.int64)
    ] = np.arange(len(dataset["source_action_index"]), dtype=np.int32)
    if np.any(matrix < 0):
        raise RuntimeError("compact dataset 缺少 source-anchor candidate mapping")
    return matrix


def prediction_anchor_rows(
    role: str,
    dataset: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    role_indices: np.ndarray,
    candidate_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    """先在每个 source anchor 内聚合 prediction errors。"""

    local_position = np.full(len(dataset["source_action_index"]), -1, dtype=np.int32)
    local_position[role_indices] = np.arange(len(role_indices), dtype=np.int32)
    rows: list[dict[str, Any]] = []
    role_code = ROLE_CODES[role]
    anchor_roles = np.empty(ACTION_COUNT, dtype=np.uint8)
    anchor_roles[dataset["source_action_index"]] = dataset["role"]
    for action_index in np.flatnonzero(anchor_roles == role_code):
        global_rows = candidate_matrix[action_index]
        valid_global = global_rows[dataset["valid"][global_rows]]
        local_rows = local_position[valid_global]
        row: dict[str, Any] = {
            "source_action_index": int(action_index),
            "v2_action_id": f"A{action_index:04d}",
            "physical_stratum_index": int(dataset["region"][global_rows[0]]),
            "valid_candidates": len(valid_global),
        }
        truth = dataset["true_outcome"][valid_global]
        for method, prediction in predictions.items():
            errors = target_normalized_error(prediction[local_rows], truth)
            row[f"{method}_mean_target_normalized_error"] = float(np.mean(errors))
        rows.append(row)
    return rows


def prediction_strata_summary(
    dataset: dict[str, np.ndarray],
    role_indices: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """按 offset、contact region 与 high-yaw outcome 描述 prediction failure modes。"""

    source = dataset["source_action_index"][role_indices]
    truth = dataset["true_outcome"][role_indices]
    offsets = (
        dataset["continuous_q"][role_indices] - dataset["anchor_q"][source]
    ) / OFFSET_SCALES
    valid = dataset["valid"][role_indices]
    valid_indices = np.flatnonzero(valid)
    offset_norm = np.linalg.norm(offsets, axis=1)
    result: dict[str, Any] = {"offset_norm_quartiles": [], "offset_dimensions": {}}

    norm_boundaries = np.quantile(offset_norm[valid], [0.25, 0.50, 0.75])
    norm_groups = np.digitize(offset_norm, norm_boundaries, right=True)
    for quartile in range(4):
        selected = valid & (norm_groups == quartile)
        row: dict[str, Any] = {"quartile": quartile + 1, "rows": int(selected.sum())}
        for method, prediction in predictions.items():
            row[method] = float(
                np.mean(target_normalized_error(prediction[selected], truth[selected]))
            )
        result["offset_norm_quartiles"].append(row)

    dimension_names = (
        "tangential_contact_coordinate",
        "force_angle",
        "force_magnitude",
        "ramp_up_duration",
        "hold_duration",
        "ramp_down_duration",
    )
    for dimension, name in enumerate(dimension_names):
        magnitude = np.abs(offsets[:, dimension])
        boundaries = np.quantile(magnitude[valid], [0.25, 0.50, 0.75])
        groups = np.digitize(magnitude, boundaries, right=True)
        entries: list[dict[str, Any]] = []
        for quartile in range(4):
            selected = valid & (groups == quartile)
            row = {"quartile": quartile + 1, "rows": int(selected.sum())}
            for method, prediction in predictions.items():
                row[method] = float(
                    np.mean(target_normalized_error(prediction[selected], truth[selected]))
                )
            entries.append(row)
        result["offset_dimensions"][name] = entries

    result["contact_regions"] = []
    for region_index in range(12):
        selected = valid & (dataset["region"][role_indices] == region_index)
        row = {"physical_stratum_index": region_index, "rows": int(selected.sum())}
        for method, prediction in predictions.items():
            row[method] = float(
                np.mean(target_normalized_error(prediction[selected], truth[selected]))
            )
        result["contact_regions"].append(row)

    yaw_threshold = float(np.quantile(np.abs(truth[valid, 2]), 0.90))
    high_yaw = valid & (np.abs(truth[:, 2]) >= yaw_threshold)
    result["highest_10_percent_absolute_yaw"] = {
        "threshold_deg": float(np.degrees(yaw_threshold)),
        "rows": int(high_yaw.sum()),
        **{
            method: float(
                np.mean(target_normalized_error(prediction[high_yaw], truth[high_yaw]))
            )
            for method, prediction in predictions.items()
        },
    }
    return result


def tnpo_cost(outcomes: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """计算一组 outcomes 相对单个 target 的 TNPO cost。"""

    position = np.linalg.norm(outcomes[:, :2] - target_xy[None, :], axis=1)
    yaw = np.abs(wrap_to_pi(outcomes[:, 2]))
    return (
        0.5 * position / PRIMARY_TNPO_COST.position_tolerance_m
        + 0.5 * yaw / PRIMARY_TNPO_COST.yaw_tolerance_rad
    )


def pose_success(outcome: np.ndarray, target_xy: np.ndarray) -> bool:
    """判断单步 outcome 是否满足 10 mm / 5° pose criterion。"""

    return bool(
        np.linalg.norm(outcome[:2] - target_xy)
        <= PRIMARY_TNPO_COST.position_tolerance_m
        and abs(float(wrap_to_pi(outcome[2])))
        <= PRIMARY_TNPO_COST.yaw_tolerance_rad
    )


def candidate_keys(
    global_rows: np.ndarray,
    dataset: dict[str, np.ndarray],
    actions: list[dict[str, str]],
) -> np.ndarray:
    """按正式八位小数格式重建 evaluation candidates 的 physical action keys。"""

    keys: list[str] = []
    for global_row in global_rows:
        action_index = int(dataset["source_action_index"][global_row])
        source = actions[action_index]
        q = dataset["continuous_q"][global_row]
        surface_id = int(source["surface_id"])
        tangent_is_y = surface_id in {0, 1}
        row = {
            "surface_id": surface_id,
            "contact_point_local_x": (
                float(source["contact_point_local_x"]) if tangent_is_y else float(q[0])
            ),
            "contact_point_local_y": (
                float(q[0]) if tangent_is_y else float(source["contact_point_local_y"])
            ),
            "force_angle_relative_to_normal_deg": float(q[1]),
            "commanded_force_N": float(q[2]),
            "ramp_up_s": float(q[3]),
            "hold_s": float(q[4]),
            "ramp_down_s": float(q[5]),
        }
        keys.append(car_e1.physical_action_key(row))
    return np.asarray(keys, dtype=str)


def first_by_cost(costs: np.ndarray, keys: np.ndarray) -> int:
    """按 cost 后 physical action key 的正式顺序返回首个 index。"""

    return int(np.lexsort((keys, costs))[0])


def ranking_evaluation(
    role: str,
    dataset: dict[str, np.ndarray],
    role_indices: np.ndarray,
    predictions: dict[str, np.ndarray],
    candidate_matrix: np.ndarray,
    actions: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """在 role-specific Top-20 anchors 上执行 protected-union ranking。"""

    target_path = VALIDATION_TARGET_PATH if role == "validation" else TEST_TARGET_PATH
    targets = read_csv_rows(target_path)
    nominal_rows = sorted(
        read_csv_rows(NOMINAL_DISCRETE_OUTCOME_PATH), key=lambda row: row["v2_action_id"]
    )
    nominal_outcomes = np.asarray(
        [
            [
                float(row["real_delta_x"]),
                float(row["real_delta_y"]),
                float(row["real_delta_yaw"]),
            ]
            for row in nominal_rows
        ],
        dtype=np.float32,
    )
    nominal_valid = np.asarray(
        [
            int(row["quality_pass"]) == 1
            and int(row["simulation_unstable"]) == 0
            and int(row["contact_success"]) == 1
            and int(row["stopped_by_threshold"]) == 1
            for row in nominal_rows
        ],
        dtype=bool,
    )
    anchor_role = np.empty(ACTION_COUNT, dtype=np.uint8)
    anchor_role[dataset["source_action_index"]] = dataset["role"]
    eligible_anchors = np.flatnonzero(anchor_role == ROLE_CODES[role])
    local_position = np.full(len(dataset["source_action_index"]), -1, dtype=np.int32)
    local_position[role_indices] = np.arange(len(role_indices), dtype=np.int32)
    rows: list[dict[str, Any]] = []
    low_cost_errors: dict[str, list[np.ndarray]] = {
        method: [] for method in METHOD_NAMES
    }

    for target in targets:
        target_xy = np.asarray(
            [float(target["target_delta_x_m"]), float(target["target_delta_y_m"])],
            dtype=np.float32,
        )
        discrete_costs = tnpo_cost(nominal_outcomes, target_xy)
        discrete_costs[~nominal_valid] = np.inf
        full_discrete_oracle = int(np.argsort(discrete_costs, kind="stable")[0])
        role_anchor_order = eligible_anchors[
            np.argsort(discrete_costs[eligible_anchors], kind="stable")
        ]
        top_anchors = role_anchor_order[:RANKING_ANCHOR_BUDGET]
        primary_global_rows = candidate_matrix[top_anchors, :].reshape(-1)
        primary_keys = candidate_keys(primary_global_rows, dataset, actions)

        for sample_budget in SAMPLE_BUDGETS:
            reshaped_rows = primary_global_rows.reshape(
                RANKING_ANCHOR_BUDGET, CANDIDATES_PER_ANCHOR
            )
            global_rows = reshaped_rows[:, :sample_budget].reshape(-1)
            reshaped_keys = primary_keys.reshape(
                RANKING_ANCHOR_BUDGET, CANDIDATES_PER_ANCHOR
            )
            keys = reshaped_keys[:, :sample_budget].reshape(-1)
            local_rows = local_position[global_rows]
            truth = dataset["true_outcome"][global_rows]
            valid = dataset["valid"][global_rows]
            true_continuous_cost = tnpo_cost(truth, target_xy)
            true_continuous_cost[~valid] = np.inf
            true_continuous_index = first_by_cost(true_continuous_cost, keys)
            discrete_real_cost = float(discrete_costs[full_discrete_oracle])
            if true_continuous_cost[true_continuous_index] < discrete_real_cost:
                union_oracle_real_cost = float(
                    true_continuous_cost[true_continuous_index]
                )
            else:
                union_oracle_real_cost = discrete_real_cost

            if sample_budget == PRIMARY_SAMPLE_BUDGET:
                tail_count = max(1, len(global_rows) // 10)
                tail_order = np.argsort(true_continuous_cost, kind="stable")[:tail_count]
                for method, prediction in predictions.items():
                    low_cost_errors[method].append(
                        target_normalized_error(
                            prediction[local_rows[tail_order]], truth[tail_order]
                        )
                    )

            for method, prediction in predictions.items():
                predicted_continuous_cost = tnpo_cost(
                    prediction[local_rows], target_xy
                )
                predicted_continuous_cost[~valid] = np.inf
                predicted_continuous_index = first_by_cost(
                    predicted_continuous_cost, keys
                )
                discrete_predicted_cost = discrete_real_cost
                continuous_selected = bool(
                    predicted_continuous_cost[predicted_continuous_index]
                    < discrete_predicted_cost
                )
                if continuous_selected:
                    selected_global_row = int(global_rows[predicted_continuous_index])
                    selected_real_outcome = truth[predicted_continuous_index]
                    selected_predicted_outcome = prediction[
                        local_rows[predicted_continuous_index]
                    ]
                    selected_real_cost = float(
                        true_continuous_cost[predicted_continuous_index]
                    )
                    selected_predicted_cost = float(
                        predicted_continuous_cost[predicted_continuous_index]
                    )
                    selected_key = str(keys[predicted_continuous_index])
                    selected_action_id = (
                        f"continuous:{int(dataset['source_action_index'][selected_global_row])}:"
                        f"{int(dataset['sobol_sample_index'][selected_global_row])}"
                    )
                else:
                    selected_global_row = -1
                    selected_real_outcome = nominal_outcomes[full_discrete_oracle]
                    selected_predicted_outcome = selected_real_outcome
                    selected_real_cost = discrete_real_cost
                    selected_predicted_cost = discrete_predicted_cost
                    selected_key = ""
                    selected_action_id = f"A{full_discrete_oracle:04d}"

                predicted_order = np.lexsort((keys, predicted_continuous_cost))
                true_order = np.lexsort((keys, true_continuous_cost))
                true_rank = int(
                    np.flatnonzero(true_order == predicted_continuous_index)[0] + 1
                )
                rows.append(
                    {
                        "e2_target_index": int(target["e2_target_index"]),
                        "v2_target_id": target["v2_target_id"],
                        "radial_quartile_index": int(target["radial_quartile_index"]),
                        "angular_quadrant_index": int(target["angular_quadrant_index"]),
                        "method": method,
                        "candidates_per_anchor": sample_budget,
                        "continuous_candidates": len(global_rows),
                        "full_discrete_oracle_action_id": f"A{full_discrete_oracle:04d}",
                        "full_discrete_oracle_cost": discrete_real_cost,
                        "true_union_oracle_cost": union_oracle_real_cost,
                        "selected_action_id": selected_action_id,
                        "selected_physical_action_key": selected_key,
                        "selected_continuous": int(continuous_selected),
                        "selected_real_tnpo_cost": selected_real_cost,
                        "selected_predicted_tnpo_cost": selected_predicted_cost,
                        "selection_gap": selected_real_cost - union_oracle_real_cost,
                        "near_optimal_0p05": int(
                            selected_real_cost - union_oracle_real_cost
                            <= NEAR_OPTIMAL_PRIMARY
                        ),
                        "near_optimal_0p02": int(
                            selected_real_cost - union_oracle_real_cost
                            <= NEAR_OPTIMAL_SENSITIVITY
                        ),
                        "optimism": selected_real_cost - selected_predicted_cost,
                        "one_step_pose_success": int(
                            pose_success(selected_real_outcome, target_xy)
                        ),
                        "continuous_selection_precision": (
                            int(selected_real_cost <= discrete_real_cost - 0.05)
                            if continuous_selected
                            else ""
                        ),
                        "selected_position_error_mm": float(
                            1000.0
                            * np.linalg.norm(selected_real_outcome[:2] - target_xy)
                        ),
                        "selected_yaw_error_deg": float(
                            np.degrees(abs(float(wrap_to_pi(selected_real_outcome[2]))))
                        ),
                        "selected_outcome_prediction_error": float(
                            target_normalized_error(
                                selected_predicted_outcome[None, :],
                                selected_real_outcome[None, :],
                            )[0]
                        ),
                        "predicted_top_20_oracle_coverage": int(
                            true_continuous_index in predicted_order[:20]
                        ),
                        "predicted_top_100_oracle_coverage": int(
                            true_continuous_index in predicted_order[:100]
                        ),
                        "predicted_top_1_true_candidate_rank": true_rank,
                        "exact_continuous_top_1_match": int(
                            predicted_continuous_index == true_continuous_index
                        ),
                    }
                )

    low_cost_summary = {
        method: {
            "rows": int(sum(len(values) for values in collections)),
            "mean_target_normalized_error": float(
                np.mean(np.concatenate(collections))
            ),
        }
        for method, collections in low_cost_errors.items()
    }
    return rows, low_cost_summary


def ranking_summary(
    rows: list[dict[str, Any]],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """汇总 target-level ranking metrics 与 paired bootstrap comparisons。"""

    summary: dict[str, Any] = {"by_candidate_budget": {}}
    for budget in SAMPLE_BUDGETS:
        budget_rows = [row for row in rows if row["candidates_per_anchor"] == budget]
        budget_summary: dict[str, Any] = {}
        for method in METHOD_NAMES:
            method_rows = [row for row in budget_rows if row["method"] == method]
            gaps = np.asarray([float(row["selection_gap"]) for row in method_rows])
            strata = np.asarray(
                [
                    int(row["radial_quartile_index"]) * 4
                    + int(row["angular_quadrant_index"])
                    for row in method_rows
                ]
            )
            gap_bootstrap = stratified_bootstrap_means(
                gaps, strata, bootstrap_resamples
            )
            continuous_rows = [row for row in method_rows if row["selected_continuous"]]
            precision_values = [
                int(row["continuous_selection_precision"])
                for row in continuous_rows
            ]
            budget_summary[method] = {
                "selected_real_tnpo_cost": percentile_summary(
                    np.asarray(
                        [float(row["selected_real_tnpo_cost"]) for row in method_rows]
                    )
                ),
                "selection_gap": {
                    **percentile_summary(gaps),
                    "mean_95_ci": bootstrap_ci(gap_bootstrap),
                },
                "near_optimal_rate_0p05": float(
                    np.mean([int(row["near_optimal_0p05"]) for row in method_rows])
                ),
                "near_optimal_rate_0p02": float(
                    np.mean([int(row["near_optimal_0p02"]) for row in method_rows])
                ),
                "optimism": percentile_summary(
                    np.asarray([float(row["optimism"]) for row in method_rows])
                ),
                "one_step_pose_success_rate": float(
                    np.mean(
                        [int(row["one_step_pose_success"]) for row in method_rows]
                    )
                ),
                "continuous_selection_rate": float(
                    np.mean([int(row["selected_continuous"]) for row in method_rows])
                ),
                "continuous_selection_precision": (
                    float(np.mean(precision_values)) if precision_values else None
                ),
                "selected_position_error_mm": percentile_summary(
                    np.asarray(
                        [float(row["selected_position_error_mm"]) for row in method_rows]
                    )
                ),
                "selected_yaw_error_deg": percentile_summary(
                    np.asarray(
                        [float(row["selected_yaw_error_deg"]) for row in method_rows]
                    )
                ),
                "predicted_top_20_oracle_coverage": float(
                    np.mean(
                        [
                            int(row["predicted_top_20_oracle_coverage"])
                            for row in method_rows
                        ]
                    )
                ),
                "predicted_top_100_oracle_coverage": float(
                    np.mean(
                        [
                            int(row["predicted_top_100_oracle_coverage"])
                            for row in method_rows
                        ]
                    )
                ),
                "predicted_top_1_true_candidate_rank": percentile_summary(
                    np.asarray(
                        [
                            int(row["predicted_top_1_true_candidate_rank"])
                            for row in method_rows
                        ]
                    )
                ),
                "exact_continuous_top_1_match_rate": float(
                    np.mean(
                        [int(row["exact_continuous_top_1_match"]) for row in method_rows]
                    )
                ),
            }

        paired_rows = {
            method: sorted(
                [row for row in budget_rows if row["method"] == method],
                key=lambda row: int(row["e2_target_index"]),
            )
            for method in METHOD_NAMES
        }
        primary_gaps = np.asarray(
            [
                float(row["selection_gap"])
                for row in paired_rows["p1_anchored_continuous_residual_model"]
            ]
        )
        baseline_gaps = np.asarray(
            [
                float(row["selection_gap"])
                for row in paired_rows["source_anchor_p1_baseline"]
            ]
        )
        paired_strata = np.asarray(
            [
                int(row["radial_quartile_index"]) * 4
                + int(row["angular_quadrant_index"])
                for row in paired_rows["source_anchor_p1_baseline"]
            ]
        )
        difference = primary_gaps - baseline_gaps
        difference_bootstrap = stratified_bootstrap_means(
            difference, paired_strata, bootstrap_resamples
        )
        budget_summary["primary_minus_source_anchor_selection_gap"] = {
            "mean_difference": float(np.mean(difference)),
            "mean_difference_95_ci": bootstrap_ci(difference_bootstrap),
        }
        summary["by_candidate_budget"][str(budget)] = budget_summary
    return summary


def evaluate_experiment(role: str, bootstrap_resamples: int) -> dict[str, Any]:
    """执行一个 role 的 prediction 与 ranking 正式评价。"""

    if not DATASET_PATH.exists() or not DIRECT_MODEL_PATH.exists() or not ANCHORED_MODEL_PATH.exists():
        raise FileNotFoundError("请先完成 collect 与 train")
    device = require_cuda()
    dataset = load_dataset()
    role_mask = dataset["role"] == ROLE_CODES[role]
    role_indices = np.flatnonzero(role_mask)
    valid_local = dataset["valid"][role_indices]
    source = dataset["source_action_index"][role_indices]
    truth = dataset["true_outcome"][role_indices]
    baseline_prediction = dataset["p1_anchor_outcome"][source]
    direct_prediction = predict_role(dataset, role_indices, DIRECT_MODEL_PATH, device)
    anchored_prediction = predict_role(dataset, role_indices, ANCHORED_MODEL_PATH, device)
    predictions = {
        "source_anchor_p1_baseline": baseline_prediction,
        "direct_continuous_outcome_model": direct_prediction,
        "p1_anchored_continuous_residual_model": anchored_prediction,
    }

    direct_checkpoint = torch.load(
        DIRECT_MODEL_PATH, map_location="cpu", weights_only=False
    )
    common_output_scale = (
        direct_checkpoint["normalisers"]["label_scale"].cpu().numpy().astype(np.float32)
    )
    prediction_summary = {
        method: prediction_metric_summary(
            prediction[valid_local], truth[valid_local], common_output_scale
        )
        for method, prediction in predictions.items()
    }
    candidate_matrix = build_candidate_matrix(dataset)
    anchor_rows = prediction_anchor_rows(
        role, dataset, predictions, role_indices, candidate_matrix
    )
    anchor_strata = np.asarray(
        [int(row["physical_stratum_index"]) for row in anchor_rows], dtype=np.int64
    )
    prediction_comparisons: dict[str, Any] = {}
    for method in METHOD_NAMES:
        values = np.asarray(
            [float(row[f"{method}_mean_target_normalized_error"]) for row in anchor_rows]
        )
        bootstrap = stratified_bootstrap_means(
            values, anchor_strata, bootstrap_resamples
        )
        prediction_comparisons[method] = {
            "anchor_mean": float(np.mean(values)),
            "anchor_mean_95_ci": bootstrap_ci(bootstrap),
        }
    primary_values = np.asarray(
        [
            float(
                row[
                    "p1_anchored_continuous_residual_model_mean_target_normalized_error"
                ]
            )
            for row in anchor_rows
        ]
    )
    baseline_values = np.asarray(
        [
            float(row["source_anchor_p1_baseline_mean_target_normalized_error"])
            for row in anchor_rows
        ]
    )
    direct_values = np.asarray(
        [
            float(row["direct_continuous_outcome_model_mean_target_normalized_error"])
            for row in anchor_rows
        ]
    )
    h1_difference_bootstrap = stratified_bootstrap_means(
        primary_values - baseline_values, anchor_strata, bootstrap_resamples
    )
    h2_difference_bootstrap = stratified_bootstrap_means(
        primary_values - direct_values, anchor_strata, bootstrap_resamples
    )
    prediction_comparisons["primary_minus_source_anchor"] = {
        "mean_difference": float(np.mean(primary_values - baseline_values)),
        "mean_difference_95_ci": bootstrap_ci(h1_difference_bootstrap),
    }
    prediction_comparisons["primary_minus_direct"] = {
        "mean_difference": float(np.mean(primary_values - direct_values)),
        "mean_difference_95_ci": bootstrap_ci(h2_difference_bootstrap),
    }

    actions = sorted(
        read_csv_rows(ACTION_MANIFEST_PATH), key=lambda row: row["v2_action_id"]
    )
    ranking_rows, low_cost_summary = ranking_evaluation(
        role,
        dataset,
        role_indices,
        predictions,
        candidate_matrix,
        actions,
    )
    ranking = ranking_summary(ranking_rows, bootstrap_resamples)
    ranking["true_low_cost_tail_prediction_error"] = low_cost_summary
    strata_summary = prediction_strata_summary(
        dataset, role_indices, predictions
    )

    output_root = RESULTS_ROOT / "evaluation" / role
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_root / "prediction_anchor_metrics.csv",
        anchor_rows,
        list(anchor_rows[0]),
    )
    write_csv(
        output_root / "ranking_cases.csv", ranking_rows, list(ranking_rows[0])
    )
    dataset_validity_rate = float(np.mean(dataset["valid"][role_indices]))
    h1_ci = prediction_comparisons["primary_minus_source_anchor"][
        "mean_difference_95_ci"
    ]
    h2_ci = prediction_comparisons["primary_minus_direct"]["mean_difference_95_ci"]
    primary_ranking_difference = ranking["by_candidate_budget"]["512"][
        "primary_minus_source_anchor_selection_gap"
    ]
    h3_ci = primary_ranking_difference["mean_difference_95_ci"]
    hypothesis = {
        "h1_supported": bool(h1_ci[1] < 0.0) if role == "test" else None,
        "h2_supported": bool(h2_ci[1] < 0.0) if role == "test" else None,
        "h3_selection_gap_component_supported": (
            bool(h3_ci[1] < 0.0) if role == "test" else None
        ),
        "h3_optimism_review_required": role == "test",
        "automatic_go_decision": None,
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "car_full_name": CAR_FULL_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "friction_cone": FRICTION_CONE,
        "role": role,
        "device": torch.cuda.get_device_name(device),
        "continuous_rows": len(role_indices),
        "valid_continuous_rows": int(np.sum(valid_local)),
        "continuous_rollout_validity_rate": dataset_validity_rate,
        "held_out_source_anchors": len(anchor_rows),
        "ranking_targets": len(read_csv_rows(VALIDATION_TARGET_PATH if role == "validation" else TEST_TARGET_PATH)),
        "bootstrap_resamples": bootstrap_resamples,
        "prediction_metrics": prediction_summary,
        "prediction_anchor_bootstrap": prediction_comparisons,
        "prediction_failure_mode_strata": strata_summary,
        "ranking_metrics": ranking,
        "hypothesis_decisions": hypothesis,
        "interpretation_boundary": (
            "H3 还需要结合 128/256/512 budgets 的 optimism 结果进行人工审核；"
            "代码不为未预先量化的‘明显失控’自行发明阈值。"
        ),
        "prediction_anchor_metrics_path": str(
            (output_root / "prediction_anchor_metrics.csv").resolve()
        ),
        "ranking_cases_path": str((output_root / "ranking_cases.csv").resolve()),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """解析 Experiment 2 命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "CAR (Continuous Action Refinement) Experiment 2: "
            "Tensor-Interpolation-Anchored Continuous Outcome Prediction"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="生成 anchor split、targets 与 additional candidates")
    collect_parser = subparsers.add_parser(
        "collect", help="采集 additional continuous MuJoCo outcomes"
    )
    collect_parser.add_argument("--num-workers", type=int, default=14)
    collect_parser.add_argument("--batch-size", type=int, default=1_024)
    collect_parser.add_argument("--resume", action="store_true")
    subparsers.add_parser("train", help="构造 compact dataset 并训练两套 models")
    validation_parser = subparsers.add_parser(
        "evaluate-validation", help="执行 Validation prediction 与 ranking evaluation"
    )
    validation_parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    test_parser = subparsers.add_parser(
        "evaluate-test", help="执行 Independent Test prediction 与 ranking evaluation"
    )
    test_parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    """执行 CAR Experiment 2 子命令。"""

    args = parse_args()
    print("CAR = Continuous Action Refinement")
    print(f"Experiment 2 = {EXPERIMENT_NAME}")
    if args.command == "prepare":
        payload = prepare_experiment()
    elif args.command == "collect":
        payload = collect_outcomes(
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            resume=args.resume,
        )
    elif args.command == "train":
        payload = train_models()
    elif args.command == "evaluate-validation":
        payload = evaluate_experiment("validation", args.bootstrap_resamples)
    elif args.command == "evaluate-test":
        payload = evaluate_experiment("test", args.bootstrap_resamples)
    else:
        raise ValueError(f"未知 command: {args.command}")
    print(json.dumps(make_json_compatible(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
