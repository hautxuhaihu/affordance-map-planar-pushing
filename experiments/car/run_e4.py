"""运行 CAR（Continuous Action Refinement）Experiment 4。

CAR 是 Continuous Action Refinement 的代码目录缩写。本脚本实现：

- prepare：生成完整 continuous candidate/residual library 与 likelihood action split；
- collect-likelihood：采集 continuous-likelihood Training/Validation outcomes；
- fit-likelihood：拟合 continuous-action residual bias 与 covariance；
- evaluate-transfer-calibration：评价跨 condition transfer 并校准 Student-t likelihood；
- evaluate-shortlist：评价 off-policy 或 on-policy continuous shortlist；
- benchmark-workers：比较 2/4/6 个 closed-loop workers；
- collect：采集 Validation/Test continuous-controller trajectories；
- evaluate：生成 Validation gate 或 predefined shared Test 结果。
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
from typing import Any, Iterable, Iterator

import numpy as np

# 在导入 PyTorch 前初始化 Conda MKL，避免两份 OpenMP runtime 以相反顺序加载。
np.linalg.inv(np.eye(1, dtype=np.float64))

import torch
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_e1 as car_e1
import run_e2 as car_e2
import run_e3 as car_e3
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
from push_core.hcr_v2.e3 import (
    belief_marginalised_probabilities,
    interpolate_outcome_grid,
    tnpo_costs,
    topk_probabilities,
    wrap_to_pi as torch_wrap_to_pi,
)
from push_core.hcr_v2.e4 import (
    point_condition_candidate_scores,
    posterior_expected_candidate_scores,
    posterior_mean_hidden_parameters,
)
from push_core.hcr_v2.e5 import make_task_query, wrap_to_pi as wrap_scalar
from push_core.simulation.physical_pusher_rollout import (
    get_body_id,
    get_object_yaw_qpos,
    load_model,
    reset_state,
    run_physical_pusher_atomic_push,
)


PROTOCOL_VERSION = "continuous_action_refinement_e4_v1"
CAR_FULL_NAME = "Continuous Action Refinement"
EXPERIMENT_NAME = "Belief-Space Continuous Action Refinement"
FRICTION_CONE = "elliptic"
DEFAULT_SEED = 0
SCENARIOS = ("friction", "com", "joint")

ACTION_COUNT = 4_536
CONTINUOUS_CANDIDATES_PER_ANCHOR = 128
SOBOL_INDEX_START = 2_048
SOBOL_INDEX_STOP = 2_176
SOBOL_GENERATED_POINTS = 4_096
LIKELIHOOD_ACTIONS_PER_CELL_PER_ROLE = 3
LIKELIHOOD_ACTIONS_PER_ROLE = 144
STUDENT_T_DEGREES_OF_FREEDOM = 3.0
POINTS_PER_DIMENSION = 17
BELIEF_UPDATE_HORIZON = 4
NUMERICAL_JITTER = 1e-6
COVARIANCE_INFLATION_CANDIDATES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
CALIBRATION_COVERAGE_LOWER = 0.93
CALIBRATION_COVERAGE_UPPER = 0.97
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000

MANIFEST_ROOT = REPOSITORY_ROOT / "manifests" / "car"
LIKELIHOOD_ACTION_SPLIT_PATH = (
    MANIFEST_ROOT / "continuous_action_refinement_e4_likelihood_action_split.csv"
)
HCR_MANIFEST_ROOT = (
    REPOSITORY_ROOT / "manifests" / "hcr_v2"
)
ACTION_MANIFEST_PATH = HCR_MANIFEST_ROOT / "hcr_v2_action_core_manifest_v1.csv"
CONDITION_MANIFEST_PATH = HCR_MANIFEST_ROOT / "hcr_v2_hidden_condition_manifest_v1.csv"
SEQUENTIAL_TARGET_PATHS = {
    "validation": HCR_MANIFEST_ROOT
    / "hcr_v2_e5_sequential_extension_validation_target_manifest_v1.csv",
    "test": HCR_MANIFEST_ROOT
    / "hcr_v2_e5_sequential_extension_test_target_manifest_v1.csv",
}
HCR_E5_DATA_ROOT = PROJECT_ROOT / "data" / "hcr_v2" / "e5" / "closed_loop"
HCR_E5_RESULTS_ROOT = PROJECT_ROOT / "results" / "hcr_v2" / "e5"

DATA_ROOT = PROJECT_ROOT / "data" / "car" / "experiment_4"
RESULTS_ROOT = PROJECT_ROOT / "results" / "car" / "experiment_4"
CANDIDATE_PATH = DATA_ROOT / "continuous_candidates.csv"
LIBRARY_PATH = DATA_ROOT / "continuous_candidate_residual_library.npz"
PREPARATION_SUMMARY_PATH = RESULTS_ROOT / "preparation_summary.json"
ANCHORED_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "car"
    / "experiment_2"
    / "models"
    / "p1_anchored_continuous_residual_model.pt"
)
DIRECT_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "car"
    / "experiment_2"
    / "models"
    / "direct_continuous_outcome_model.pt"
)

CANDIDATE_FIELDS = [*car_e1.UNIQUE_ACTION_FIELDS, "e4_candidate_rank"]
LIKELIHOOD_SPLIT_FIELDS = [
    "physical_action_id",
    "physical_action_key",
    "source_v2_action_id",
    "source_action_index",
    "source_anchor_rank",
    "surface_id",
    "contact_region_col",
    "physical_stratum_index",
    "normalized_offset_norm",
    "offset_quartile_index",
    "likelihood_role",
    "role_action_index",
    "cell_selection_rank",
    "selection_seed",
]
LIKELIHOOD_OUTCOME_FIELDS = [
    "experiment_id",
    "protocol_version",
    "car_full_name",
    "experiment_name",
    "friction_cone",
    "scenario",
    "likelihood_role",
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
    *LIKELIHOOD_SPLIT_FIELDS,
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

LIKELIHOOD_WORKER_ACTIONS: list[dict[str, Any]] = []


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


def select_scenarios(value: str) -> tuple[str, ...]:
    """解析单场景或 all。"""

    return SCENARIOS if value == "all" else (value,)


def load_ordered_actions() -> tuple[list[str], list[dict[str, str]]]:
    """按 HCR V2 12×378 layout 读取 action_core。"""

    action_ids, _ = hcr_e3.load_action_layout()
    rows_by_id = {
        row["v2_action_id"]: row for row in read_csv_rows(ACTION_MANIFEST_PATH)
    }
    rows = [rows_by_id[action_id] for action_id in action_ids]
    if len(rows) != ACTION_COUNT:
        raise RuntimeError(f"action_core 数量错误: {len(rows)}")
    return action_ids, rows


def generate_e4_candidates(anchor: dict[str, str]) -> Iterator[dict[str, Any]]:
    """为一个离散 anchor 生成 E4 fresh Sobol segment。"""

    action_index = int(anchor["v2_action_id"][1:])
    surface_id = int(anchor["surface_id"])
    contact_col = int(anchor["contact_region_col"])
    tangent_is_y = surface_id in {0, 1}
    tangent_center = float(
        anchor[
            "contact_point_local_y" if tangent_is_y else "contact_point_local_x"
        ]
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
    unit_points = sobol.random_base2(m=12)[SOBOL_INDEX_START:SOBOL_INDEX_STOP]
    probabilities = alpha[None, :] + unit_points * (beta - alpha)[None, :]
    samples = centers[None, :] + car_e1._GRID_SIGMA[None, :] * ndtri(probabilities)
    samples = np.round(samples, decimals=8)

    normal_x = float(anchor["contact_normal_local_x"])
    normal_y = float(anchor["contact_normal_local_y"])
    fixed_x = round(float(anchor["contact_point_local_x"]), 8)
    fixed_y = round(float(anchor["contact_point_local_y"]), 8)
    for candidate_rank, values in enumerate(samples):
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
            "sobol_sample_index": SOBOL_INDEX_START + candidate_rank,
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
            "e4_candidate_rank": candidate_rank,
        }


def predict_library_models(
    continuous_q: np.ndarray,
    source_indices: np.ndarray,
    region_indices: np.ndarray,
    ordered_actions: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, str]:
    """在 CUDA 上计算 frozen anchored residual 与 Direct outcomes。"""

    device = car_e2.require_cuda()
    anchor_q = np.asarray(
        [car_e2.action_coordinate(row) for row in ordered_actions], dtype=np.float32
    )
    nominal_p1 = car_e2.nominal_p1_outcomes(ordered_actions)
    one_hot = np.eye(12, dtype=np.float32)[region_indices]
    outputs: dict[str, np.ndarray] = {}
    for name, checkpoint_path in (
        ("anchored", ANCHORED_MODEL_PATH),
        ("direct", DIRECT_MODEL_PATH),
    ):
        model, normalisers, model_type = car_e2.load_checkpoint_model(
            checkpoint_path, device
        )
        if model_type == "anchored":
            features = np.concatenate(
                [
                    one_hot,
                    (anchor_q[source_indices] - car_e2.ACTION_CENTRES)
                    / car_e2.ACTION_SCALES,
                    (continuous_q - anchor_q[source_indices])
                    / car_e2.OFFSET_SCALES,
                    (nominal_p1[source_indices] - normalisers["p1_mean"])
                    / normalisers["p1_scale"],
                ],
                axis=1,
            ).astype(np.float32)
        else:
            features = np.concatenate(
                [
                    one_hot,
                    (continuous_q - car_e2.ACTION_CENTRES)
                    / car_e2.ACTION_SCALES,
                ],
                axis=1,
            ).astype(np.float32)
        tensor = torch.from_numpy(features).to(device)
        prediction_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(features), car_e2.EVALUATION_BATCH_SIZE):
                stop = min(len(features), start + car_e2.EVALUATION_BATCH_SIZE)
                with car_e2.autocast_context():
                    scaled = model(tensor[start:stop])
                prediction_parts.append(scaled.float().cpu().numpy())
        physical = (
            np.concatenate(prediction_parts, axis=0) * normalisers["label_scale"]
            + normalisers["label_mean"]
        )
        outputs[name] = physical.astype(np.float32)
        del model, tensor, features
        torch.cuda.empty_cache()
    return outputs["anchored"], outputs["direct"], torch.cuda.get_device_name(device)


@dataclass(frozen=True)
class CandidateLibrary:
    """E4 continuous candidate 的紧凑运行时表示。"""

    source_action_index: np.ndarray
    candidate_rank: np.ndarray
    region_index: np.ndarray
    continuous_q: np.ndarray
    continuous_residual: np.ndarray
    direct_outcome: np.ndarray

    @classmethod
    def load(cls, path: Path = LIBRARY_PATH) -> "CandidateLibrary":
        """读取 compact candidate/residual library。"""

        with np.load(path, allow_pickle=False) as payload:
            return cls(
                source_action_index=np.asarray(
                    payload["source_action_index"], dtype=np.int32
                ),
                candidate_rank=np.asarray(payload["candidate_rank"], dtype=np.int16),
                region_index=np.asarray(payload["region_index"], dtype=np.uint8),
                continuous_q=np.asarray(payload["continuous_q"], dtype=np.float32),
                continuous_residual=np.asarray(
                    payload["continuous_residual"], dtype=np.float32
                ),
                direct_outcome=np.asarray(payload["direct_outcome"], dtype=np.float32),
            )

    def action_row(
        self,
        physical_action_id: int,
        ordered_actions: list[dict[str, str]],
    ) -> dict[str, Any]:
        """由 compact coordinate 重建一个可执行 continuous action row。"""

        index = int(physical_action_id)
        source_index = int(self.source_action_index[index])
        source = ordered_actions[source_index]
        tangent, angle, force, ramp_up, hold, ramp_down = [
            float(value) for value in self.continuous_q[index]
        ]
        surface_id = int(source["surface_id"])
        tangent_is_y = surface_id in {0, 1}
        contact_x = (
            float(source["contact_point_local_x"]) if tangent_is_y else tangent
        )
        contact_y = (
            tangent if tangent_is_y else float(source["contact_point_local_y"])
        )
        normal_x = float(source["contact_normal_local_x"])
        normal_y = float(source["contact_normal_local_y"])
        direction_x, direction_y = car_e1.rotate_inward_direction(
            normal_x, normal_y, angle
        )
        row = {
            "physical_action_id": index,
            "source_v2_action_id": source["v2_action_id"],
            "source_action_index": source_index,
            "source_candidate_id": source["candidate_id"],
            "source_action_param_index": source["action_param_index"],
            "source_contact_region_id": source["contact_region_id"],
            "sobol_sample_index": SOBOL_INDEX_START + int(self.candidate_rank[index]),
            "surface_id": surface_id,
            "contact_region_row": source["contact_region_row"],
            "contact_region_col": source["contact_region_col"],
            "contact_point_local_x": round(contact_x, 8),
            "contact_point_local_y": round(contact_y, 8),
            "contact_normal_local_x": normal_x,
            "contact_normal_local_y": normal_y,
            "force_angle_relative_to_normal_deg": round(angle, 8),
            "force_direction_local_x": round(direction_x, 12),
            "force_direction_local_y": round(direction_y, 12),
            "commanded_force_N": round(force, 8),
            "ramp_up_s": round(ramp_up, 8),
            "hold_s": round(hold, 8),
            "ramp_down_s": round(ramp_down, 8),
            "execution_duration_s": round(ramp_up + hold + ramp_down, 8),
            "sampling_seed": car_e1.stable_anchor_seed(
                int(source["v2_action_id"][1:])
            ),
            "e4_candidate_rank": int(self.candidate_rank[index]),
        }
        row["physical_action_key"] = car_e1.physical_action_key(row)
        return row


def build_likelihood_action_split(
    library: CandidateLibrary,
    ordered_actions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """按 physical stratum 与 offset quartile 选择 likelihood actions。"""

    anchor_q = np.asarray(
        [car_e2.action_coordinate(row) for row in ordered_actions], dtype=np.float32
    )
    offset = (
        library.continuous_q - anchor_q[library.source_action_index]
    ) / car_e2.OFFSET_SCALES
    offset_norm = np.linalg.vector_norm(offset, axis=1)
    source_surface = np.asarray(
        [int(ordered_actions[index]["surface_id"]) for index in library.source_action_index],
        dtype=np.int8,
    )
    source_col = np.asarray(
        [
            int(ordered_actions[index]["contact_region_col"])
            for index in library.source_action_index
        ],
        dtype=np.int8,
    )
    physical_stratum = source_surface * 3 + source_col
    quartile = np.empty(len(offset_norm), dtype=np.int8)
    for stratum in range(12):
        mask = physical_stratum == stratum
        boundaries = np.quantile(offset_norm[mask], [0.25, 0.50, 0.75])
        quartile[mask] = np.searchsorted(boundaries, offset_norm[mask], side="right")

    selected_rows: list[dict[str, Any]] = []
    role_counts = {"training": 0, "validation": 0}
    for stratum in range(12):
        for quartile_index in range(4):
            indices = np.flatnonzero(
                (physical_stratum == stratum) & (quartile == quartile_index)
            )
            keyed = []
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
            chosen = permutation[: 2 * LIKELIHOOD_ACTIONS_PER_CELL_PER_ROLE]
            for cell_rank, physical_action_id in enumerate(chosen):
                role = (
                    "training"
                    if cell_rank < LIKELIHOOD_ACTIONS_PER_CELL_PER_ROLE
                    else "validation"
                )
                action = library.action_row(int(physical_action_id), ordered_actions)
                selected_rows.append(
                    {
                        "physical_action_id": int(physical_action_id),
                        "physical_action_key": action["physical_action_key"],
                        "source_v2_action_id": action["source_v2_action_id"],
                        "source_action_index": action["source_action_index"],
                        "surface_id": action["surface_id"],
                        "contact_region_col": action["contact_region_col"],
                        "physical_stratum_index": stratum,
                        "normalized_offset_norm": float(offset_norm[physical_action_id]),
                        "offset_quartile_index": quartile_index,
                        "likelihood_role": role,
                        "role_action_index": role_counts[role],
                        "cell_selection_rank": (
                            cell_rank
                            if role == "training"
                            else cell_rank - LIKELIHOOD_ACTIONS_PER_CELL_PER_ROLE
                        ),
                        "selection_seed": seed,
                    }
                )
                role_counts[role] += 1
    selected_rows.sort(
        key=lambda row: (row["likelihood_role"], int(row["role_action_index"]))
    )
    if role_counts != {"training": 144, "validation": 144}:
        raise RuntimeError(f"likelihood action split 数量错误: {role_counts}")
    return selected_rows


def prepare_experiment() -> dict[str, Any]:
    """生成完整 E4 candidate/residual library。"""

    action_ids, ordered_actions = load_ordered_actions()
    row_count = ACTION_COUNT * CONTINUOUS_CANDIDATES_PER_ANCHOR
    source = np.empty(row_count, dtype=np.int32)
    rank = np.empty(row_count, dtype=np.int16)
    region = np.empty(row_count, dtype=np.uint8)
    continuous_q = np.empty((row_count, 6), dtype=np.float32)
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CANDIDATE_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CANDIDATE_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        physical_action_id = 0
        for source_index, action in enumerate(ordered_actions):
            for candidate in generate_e4_candidates(action):
                candidate["source_action_index"] = source_index
                candidate["physical_action_id"] = physical_action_id
                candidate["physical_action_key"] = car_e1.physical_action_key(candidate)
                writer.writerow(candidate)
                source[physical_action_id] = source_index
                rank[physical_action_id] = int(candidate["e4_candidate_rank"])
                region[physical_action_id] = (
                    int(candidate["surface_id"]) * 3
                    + int(candidate["contact_region_col"])
                )
                continuous_q[physical_action_id] = car_e2.action_coordinate(candidate)
                physical_action_id += 1
            if (source_index + 1) % 100 == 0 or source_index + 1 == ACTION_COUNT:
                print(
                    f"prepared anchor {source_index + 1}/{ACTION_COUNT}; "
                    f"candidates={physical_action_id}"
                )
    if physical_action_id != row_count:
        raise RuntimeError(f"E4 candidate 数量错误: {physical_action_id}")

    residual, direct, device_name = predict_library_models(
        continuous_q, source, region, ordered_actions
    )
    np.savez_compressed(
        LIBRARY_PATH,
        protocol_version=np.asarray(PROTOCOL_VERSION),
        action_ids=np.asarray(action_ids),
        source_action_index=source,
        candidate_rank=rank,
        region_index=region,
        continuous_q=continuous_q,
        continuous_residual=residual,
        direct_outcome=direct,
    )
    library = CandidateLibrary.load()
    split_rows = build_likelihood_action_split(library, ordered_actions)
    write_csv(LIKELIHOOD_ACTION_SPLIT_PATH, split_rows, LIKELIHOOD_SPLIT_FIELDS)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "car_full_name": CAR_FULL_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "friction_cone": FRICTION_CONE,
        "default_seed": DEFAULT_SEED,
        "source_anchors": ACTION_COUNT,
        "continuous_candidates_per_anchor": CONTINUOUS_CANDIDATES_PER_ANCHOR,
        "continuous_candidates": row_count,
        "sobol_index_start": SOBOL_INDEX_START,
        "sobol_index_stop_exclusive": SOBOL_INDEX_STOP,
        "anchor_specific_seed": "SeedSequence([0, action_index])",
        "cuda_device": device_name,
        "likelihood_training_actions": 144,
        "likelihood_validation_actions": 144,
        "candidate_path": str(CANDIDATE_PATH.resolve()),
        "library_path": str(LIBRARY_PATH.resolve()),
        "likelihood_action_split_path": str(
            LIKELIHOOD_ACTION_SPLIT_PATH.resolve()
        ),
    }
    write_json(PREPARATION_SUMMARY_PATH, summary)
    print(json.dumps(make_json_compatible(summary), ensure_ascii=False, indent=2))
    return summary


def load_likelihood_actions(role: str) -> list[dict[str, Any]]:
    """读取并重建一个 likelihood role 的 144 个 continuous actions。"""

    split = [
        row
        for row in read_csv_rows(LIKELIHOOD_ACTION_SPLIT_PATH)
        if row["likelihood_role"] == role
    ]
    split.sort(key=lambda row: int(row["role_action_index"]))
    if len(split) != LIKELIHOOD_ACTIONS_PER_ROLE:
        raise RuntimeError(f"{role} likelihood actions 数量错误: {len(split)}")
    _, actions = load_ordered_actions()
    library = CandidateLibrary.load()
    outputs: list[dict[str, Any]] = []
    for split_row in split:
        action = library.action_row(int(split_row["physical_action_id"]), actions)
        outputs.append({**action, **split_row})
    return outputs


def initialise_likelihood_worker(actions: list[dict[str, Any]]) -> None:
    """为 likelihood collection worker 保存共享 actions。"""

    global LIKELIHOOD_WORKER_ACTIONS
    LIKELIHOOD_WORKER_ACTIONS = actions


def continuous_rollout_input(
    action: dict[str, Any], condition: dict[str, str], episode_id: int
) -> dict[str, Any]:
    """构造包含真实 hidden condition 的 continuous rollout input。"""

    row = car_e1.build_rollout_input(action, episode_id)
    row["dataset_role"] = (
        f"car_e4_likelihood_{action['likelihood_role']}_{condition['scenario']}"
    )
    row["hidden_com_offset_x"] = float(condition["com_offset_x_m"])
    row["hidden_com_offset_y"] = float(condition["com_offset_y_m"])
    return row


def process_likelihood_condition(task: dict[str, Any]) -> list[dict[str, Any]]:
    """在一个 hidden condition 下执行 144 个 canonical-reset actions。"""

    condition = task["condition"]
    model, data = load_model(Path(task["xml_path"]))
    hcr_e1.set_sliding_friction(model, float(condition["friction_sliding_mu"]))
    rows: list[dict[str, Any]] = []
    for action_index, action in enumerate(LIKELIHOOD_WORKER_ACTIONS):
        episode_id = (
            int(condition["condition_index_within_role"]) * 1_000 + action_index
        )
        rollout_input = continuous_rollout_input(action, condition, episode_id)
        result = car_e1.run_physical_pusher_rollout(
            model, data, rollout_input, validate_schema=False
        )
        rows.append(
            {
                "experiment_id": "E4",
                "protocol_version": PROTOCOL_VERSION,
                "car_full_name": CAR_FULL_NAME,
                "experiment_name": EXPERIMENT_NAME,
                "friction_cone": FRICTION_CONE,
                "scenario": condition["scenario"],
                "likelihood_role": action["likelihood_role"],
                "condition_id": condition["condition_id"],
                "condition_role": condition["condition_role"],
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
                **{field: action[field] for field in LIKELIHOOD_SPLIT_FIELDS},
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


def likelihood_outcome_path(scenario: str, role: str) -> Path:
    """返回一个场景和 likelihood role 的 outcome path。"""

    return DATA_ROOT / f"likelihood_{role}" / f"{scenario}_outcomes.csv"


def collect_likelihood(args: argparse.Namespace) -> dict[str, Any]:
    """按 hidden condition 并行采集 continuous-likelihood outcomes。"""

    actions = load_likelihood_actions(args.role)
    summaries: dict[str, Any] = {}
    for scenario in select_scenarios(args.scenario):
        conditions = hcr_e3.load_conditions(scenario, args.role)
        if args.max_conditions > 0:
            conditions = conditions[: args.max_conditions]
        generated_dir = DATA_ROOT / "generated_xml" / args.role / scenario
        xml_by_com = hcr_e1.prepare_environment_xmls(conditions, generated_dir)
        output_path = likelihood_outcome_path(scenario, args.role)
        existing_rows = read_csv_rows(output_path) if args.resume and output_path.exists() else []
        by_condition = Counter(row["condition_id"] for row in existing_rows)
        complete_ids = {
            condition_id
            for condition_id, count in by_condition.items()
            if count == LIKELIHOOD_ACTIONS_PER_ROLE
        }
        retained_rows = [
            row for row in existing_rows if row["condition_id"] in complete_ids
        ]
        tasks = []
        for condition in conditions:
            if condition["condition_id"] in complete_ids:
                continue
            com_key = (
                round(float(condition["com_offset_x_m"]), 9),
                round(float(condition["com_offset_y_m"]), 9),
            )
            tasks.append(
                {"condition": condition, "xml_path": str(xml_by_com[com_key])}
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        processed_conditions = 0
        mode = "w"
        with output_path.open(mode, encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=LIKELIHOOD_OUTCOME_FIELDS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(retained_rows)
            worker_count = min(max(1, args.num_workers), max(1, len(tasks)))
            if len(tasks) == 0:
                result_iterator: Iterable[list[dict[str, Any]]] = []
                pool = None
            elif worker_count == 1:
                initialise_likelihood_worker(actions)
                result_iterator = map(process_likelihood_condition, tasks)
                pool = None
            else:
                context = mp.get_context("spawn")
                pool = context.Pool(
                    worker_count,
                    initializer=initialise_likelihood_worker,
                    initargs=(actions,),
                    maxtasksperchild=8,
                )
                result_iterator = pool.imap(process_likelihood_condition, tasks)
            try:
                for result_rows in result_iterator:
                    writer.writerows(result_rows)
                    handle.flush()
                    processed_conditions += 1
                    print(
                        f"finished {scenario} {args.role} condition "
                        f"{processed_conditions}/{len(tasks)}"
                    )
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
        final_rows = read_csv_rows(output_path)
        valid_count = sum(
            int(row["quality_pass"]) == 1
            and int(row["simulation_unstable"]) == 0
            and int(row["contact_success"]) == 1
            and int(row["stopped_by_threshold"]) == 1
            for row in final_rows
        )
        elapsed = time.perf_counter() - started
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "friction_cone": FRICTION_CONE,
            "scenario": scenario,
            "likelihood_role": args.role,
            "conditions": len(conditions),
            "actions_per_condition": LIKELIHOOD_ACTIONS_PER_ROLE,
            "rollouts": len(final_rows),
            "valid_rollouts": valid_count,
            "validity_rate": valid_count / len(final_rows) if final_rows else 0.0,
            "resumed_conditions": len(complete_ids),
            "new_conditions": processed_conditions,
            "num_workers": min(max(1, args.num_workers), max(1, len(tasks))),
            "elapsed_seconds": elapsed,
            "outcome_path": str(output_path.resolve()),
        }
        write_json(
            RESULTS_ROOT
            / "continuous_likelihood"
            / scenario
            / f"{args.role}_collection_summary.json",
            summary,
        )
        summaries[scenario] = summary
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "likelihood_role": args.role,
        "scenario_summaries": summaries,
    }
    print(json.dumps(make_json_compatible(combined), ensure_ascii=False, indent=2))
    return combined


def scenario_raw_predictions(
    scenario: str,
    conditions: list[dict[str, str]],
    physical_action_ids: np.ndarray,
    library: CandidateLibrary,
) -> np.ndarray:
    """计算 condition × continuous-action raw transferred predictions。"""

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
    source = library.source_action_index[physical_action_ids]
    residual = library.continuous_residual[physical_action_ids]
    predictions = np.empty((len(conditions), len(physical_action_ids), 3), dtype=np.float64)
    for condition_index, condition in enumerate(conditions):
        anchor = p1.predict(condition).astype(np.float64)[source]
        predictions[condition_index] = anchor + residual
    return predictions


def fit_scenario_likelihood(scenario: str) -> dict[str, Any]:
    """拟合一个场景的 continuous-action residual statistics。"""

    rows = read_csv_rows(likelihood_outcome_path(scenario, "training"))
    conditions = hcr_e3.load_conditions(scenario, "training")
    expected = len(conditions) * LIKELIHOOD_ACTIONS_PER_ROLE
    if len(rows) != expected:
        raise RuntimeError(f"{scenario} likelihood Training rows 数量错误: {len(rows)}")
    rows.sort(
        key=lambda row: (
            int(row["condition_index_within_role"]),
            int(row["role_action_index"]),
        )
    )
    physical_ids = np.asarray(
        [int(row["physical_action_id"]) for row in rows[:144]], dtype=np.int64
    )
    library = CandidateLibrary.load()
    prediction = scenario_raw_predictions(
        scenario, conditions, physical_ids, library
    ).reshape(-1, 3)
    truth = np.asarray(
        [[float(row[field]) for field in OUTCOME_FIELDS] for row in rows],
        dtype=np.float64,
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
    residuals = (truth[valid] - prediction[valid]) / OBSERVATION_SCALE
    bias = residuals.mean(axis=0)
    centered = residuals - bias
    covariance = centered.T @ centered / (len(centered) - 1)
    output_dir = RESULTS_ROOT / "continuous_likelihood" / scenario
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "residual_statistics.npz"
    np.savez_compressed(
        artifact_path,
        protocol_version=np.asarray(PROTOCOL_VERSION),
        scenario=np.asarray(scenario),
        observation_fields=np.asarray(OUTCOME_FIELDS),
        observation_scale=OBSERVATION_SCALE,
        residual_bias=bias,
        base_covariance=covariance,
        sample_count=np.asarray(len(residuals), dtype=np.int64),
    )
    raw_bias = bias * OBSERVATION_SCALE
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "scenario": scenario,
        "training_conditions": len(conditions),
        "actions_per_condition": LIKELIHOOD_ACTIONS_PER_ROLE,
        "sample_count": len(residuals),
        "excluded_invalid_rows": int((~valid).sum()),
        "validity_rate": float(np.mean(valid)),
        "residual_bias_standardized": bias,
        "residual_bias_physical": {
            "delta_x_mm": raw_bias[0] * 1_000.0,
            "delta_y_mm": raw_bias[1] * 1_000.0,
            "delta_yaw_deg": math.degrees(raw_bias[2]),
        },
        "centered_base_covariance_standardized": covariance,
        "artifact_path": str(artifact_path.resolve()),
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def fit_likelihood(args: argparse.Namespace) -> dict[str, Any]:
    """拟合所选场景的 continuous-action likelihood statistics。"""

    summaries = {
        scenario: fit_scenario_likelihood(scenario)
        for scenario in select_scenarios(args.scenario)
    }
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario_summaries": summaries,
    }
    write_json(
        RESULTS_ROOT / "continuous_likelihood" / "training_summary.json", combined
    )
    print(json.dumps(make_json_compatible(combined), ensure_ascii=False, indent=2))
    return combined


def prediction_error(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """计算 transfer Validation 使用的逐 rollout prediction errors。"""

    delta = prediction - truth
    delta[:, 2] = (delta[:, 2] + math.pi) % (2.0 * math.pi) - math.pi
    planar = np.linalg.vector_norm(delta[:, :2], axis=1)
    yaw = np.abs(delta[:, 2])
    target_normalized = (
        0.5 * planar / PRIMARY_TNPO_COST.position_tolerance_m
        + 0.5 * yaw / PRIMARY_TNPO_COST.yaw_tolerance_rad
    )
    return {
        "target_normalized": target_normalized,
        "planar_m": planar,
        "yaw_rad": yaw,
        "signed": delta,
    }


def descriptive(values: np.ndarray) -> dict[str, float | int | None]:
    """返回 mean、median 与 P90。"""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def transfer_method_summary(
    prediction: np.ndarray, truth: np.ndarray
) -> dict[str, Any]:
    """汇总一个 transfer method 的 outcome-prediction metrics。"""

    errors = prediction_error(prediction, truth)
    signed = errors["signed"]
    return {
        "target_normalized_outcome_prediction_error": descriptive(
            errors["target_normalized"]
        ),
        "planar_outcome_error_mm": descriptive(errors["planar_m"] * 1_000.0),
        "yaw_outcome_error_deg": descriptive(np.degrees(errors["yaw_rad"])),
        "signed_bias": {
            "delta_x_m": float(np.mean(signed[:, 0])),
            "delta_y_m": float(np.mean(signed[:, 1])),
            "delta_yaw_deg": float(math.degrees(np.mean(signed[:, 2]))),
        },
    }


def condition_action_bootstrap(
    effects: np.ndarray,
    action_cells: np.ndarray,
    resamples: int,
) -> list[float]:
    """执行 paired condition-action two-way bootstrap。"""

    condition_count, action_count = effects.shape
    cell_indices = [np.flatnonzero(action_cells == cell) for cell in range(48)]
    generator = np.random.default_rng(DEFAULT_SEED)
    values = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        sampled_conditions = generator.integers(
            0, condition_count, size=condition_count
        )
        sampled_actions = np.concatenate(
            [
                indices[
                    generator.integers(0, len(indices), size=len(indices))
                ]
                for indices in cell_indices
            ]
        )
        values[sample_index] = float(
            np.nanmean(effects[np.ix_(sampled_conditions, sampled_actions)])
        )
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def student_t_log_likelihoods(
    observations_standardized: np.ndarray,
    means_standardized: np.ndarray,
    precision: np.ndarray,
    log_normalization: float,
) -> np.ndarray:
    """计算 multivariate Student-t log likelihoods。"""

    residual = observations_standardized - means_standardized
    mahalanobis = np.einsum("...i,ij,...j->...", residual, precision, residual)
    return log_normalization - 0.5 * (STUDENT_T_DEGREES_OF_FREEDOM + 3.0) * np.log1p(
        mahalanobis / STUDENT_T_DEGREES_OF_FREEDOM
    )


def calibration_node_means(
    scenario: str,
    physical_ids: np.ndarray,
    library: CandidateLibrary,
    bias: np.ndarray,
) -> tuple[np.ndarray, Any]:
    """为 144 个 Validation actions 预计算 quadrature node means。"""

    rule = make_quadrature_rule(scenario, POINTS_PER_DIMENSION)
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
    source = library.source_action_index[physical_ids]
    residual = library.continuous_residual[physical_ids].astype(np.float64)
    node_means = np.empty(
        (len(physical_ids), rule.node_count, 3), dtype=np.float32
    )
    for node_index, node in enumerate(rule.nodes):
        anchor = p1.predict(
            condition_row_from_normalised(scenario, node)
        ).astype(np.float64)[source]
        node_means[:, node_index] = (
            (anchor + residual) / OBSERVATION_SCALE + bias
        ).astype(np.float32)
    return node_means, rule


def calibrate_scenario(
    scenario: str,
    rows: list[dict[str, str]],
    conditions: list[dict[str, str]],
    physical_ids: np.ndarray,
    truth_matrix: np.ndarray,
    valid_matrix: np.ndarray,
    library: CandidateLibrary,
    statistics: ResidualStatistics,
) -> dict[str, Any]:
    """评价 continuous Student-t covariance-inflation grid。"""

    node_means, rule = calibration_node_means(
        scenario, physical_ids, library, statistics.residual_bias
    )
    true_coordinates = np.asarray(
        [
            [float(condition[field]) for field in rule.active_coordinates]
            for condition in conditions
        ],
        dtype=np.float64,
    )
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
    source = library.source_action_index[physical_ids]
    residual = library.continuous_residual[physical_ids].astype(np.float64)
    true_means = np.empty_like(truth_matrix, dtype=np.float64)
    for condition_index, condition in enumerate(conditions):
        anchor = p1.predict(condition).astype(np.float64)[source]
        true_means[condition_index] = (
            anchor + residual
        ) / OBSERVATION_SCALE + statistics.residual_bias

    observations = truth_matrix / OBSERVATION_SCALE
    candidate_results: list[dict[str, Any]] = []
    for inflation in COVARIANCE_INFLATION_CANDIDATES:
        covariance = (
            inflation * statistics.base_covariance
            + NUMERICAL_JITTER * np.eye(3, dtype=np.float64)
        )
        precision = np.linalg.inv(covariance)
        logdet = float(np.linalg.slogdet(covariance)[1])
        degrees = STUDENT_T_DEGREES_OF_FREEDOM
        log_normalization = (
            math.lgamma((degrees + 3.0) / 2.0)
            - math.lgamma(degrees / 2.0)
            - 1.5 * math.log(degrees * math.pi)
            - 0.5 * logdet
        )
        terminal_rows: list[dict[str, Any]] = []
        update_nll: dict[int, list[float]] = defaultdict(list)
        update_mean_error: dict[int, list[float]] = defaultdict(list)
        for condition_index, condition in enumerate(conditions):
            for history_index in range(36):
                history_slice = slice(history_index * 4, history_index * 4 + 4)
                if not bool(np.all(valid_matrix[condition_index, history_slice])):
                    continue
                posterior = FixedNodePosterior(rule)
                cumulative_true_log_likelihood = 0.0
                for update_index in range(BELIEF_UPDATE_HORIZON):
                    action_slot = history_index * 4 + update_index
                    observation = observations[condition_index, action_slot]
                    node_log_likelihoods = student_t_log_likelihoods(
                        observation,
                        node_means[action_slot],
                        precision,
                        log_normalization,
                    )
                    posterior.update(node_log_likelihoods)
                    cumulative_true_log_likelihood += float(
                        student_t_log_likelihoods(
                            observation,
                            true_means[condition_index, action_slot],
                            precision,
                            log_normalization,
                        )
                    )
                    posterior_metrics = posterior.probabilistic_metrics(
                        cumulative_true_log_likelihood
                    )
                    mean_error = float(
                        np.linalg.vector_norm(
                            posterior.summary().mean_normalised
                            - true_coordinates[condition_index]
                        )
                    )
                    update_nll[update_index + 1].append(
                        float(posterior_metrics["posterior_nll"])
                    )
                    update_mean_error[update_index + 1].append(mean_error)
                weights = posterior.weights
                terminal_metrics = posterior.probabilistic_metrics(
                    cumulative_true_log_likelihood
                )
                terminal_rows.append(
                    {
                        "condition_id": condition["condition_id"],
                        "history_index": history_index,
                        "posterior_nll": terminal_metrics["posterior_nll"],
                        "hpd_covered": terminal_metrics["hpd_covered"],
                        "effective_sample_size": float(1.0 / np.sum(weights**2)),
                        "uncertainty_contraction": terminal_metrics[
                            "uncertainty_contraction"
                        ],
                    }
                )
        if not terminal_rows:
            raise RuntimeError(f"{scenario} 没有完整有效的 four-update histories")
        by_condition_nll: dict[str, list[float]] = defaultdict(list)
        for row in terminal_rows:
            by_condition_nll[row["condition_id"]].append(float(row["posterior_nll"]))
        result = {
            "covariance_inflation": inflation,
            "degrees_of_freedom": STUDENT_T_DEGREES_OF_FREEDOM,
            "points_per_dimension": POINTS_PER_DIMENSION,
            "node_count": rule.node_count,
            "update_horizon": BELIEF_UPDATE_HORIZON,
            "terminal_histories": len(terminal_rows),
            "terminal_hpd_coverage": float(
                np.mean([row["hpd_covered"] for row in terminal_rows])
            ),
            "condition_balanced_mean_terminal_nll": float(
                np.mean([np.mean(values) for values in by_condition_nll.values()])
            ),
            "mean_terminal_effective_sample_size": float(
                np.mean([row["effective_sample_size"] for row in terminal_rows])
            ),
            "mean_terminal_uncertainty_contraction": float(
                np.mean([row["uncertainty_contraction"] for row in terminal_rows])
            ),
            "update_curve": {
                str(update): {
                    "mean_posterior_nll": float(np.mean(update_nll[update])),
                    "mean_parameter_error": float(
                        np.mean(update_mean_error[update])
                    ),
                }
                for update in range(1, BELIEF_UPDATE_HORIZON + 1)
            },
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
        if CALIBRATION_COVERAGE_LOWER
        <= float(row["terminal_hpd_coverage"])
        <= CALIBRATION_COVERAGE_UPPER
    ]
    if not calibrated:
        raise RuntimeError(f"{scenario} 没有满足 93%–97% coverage band 的 inflation")
    selected = min(
        calibrated,
        key=lambda row: (
            float(row["condition_balanced_mean_terminal_nll"]),
            float(row["covariance_inflation"]),
        ),
    )
    return {
        "selection_metric": (
            "condition_balanced_mean_terminal_nll_within_93_to_97_percent_coverage"
        ),
        "coverage_band": [CALIBRATION_COVERAGE_LOWER, CALIBRATION_COVERAGE_UPPER],
        "candidate_results": candidate_results,
        "selected_covariance_inflation": selected["covariance_inflation"],
        "selected_validation_coverage": selected["terminal_hpd_coverage"],
        "selected_condition_balanced_mean_terminal_nll": selected[
            "condition_balanced_mean_terminal_nll"
        ],
    }


def evaluate_transfer_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """评价 transfer 并选择 continuous-action likelihood inflation。"""

    library = CandidateLibrary.load()
    split_rows = [
        row
        for row in read_csv_rows(LIKELIHOOD_ACTION_SPLIT_PATH)
        if row["likelihood_role"] == "validation"
    ]
    split_rows.sort(key=lambda row: int(row["role_action_index"]))
    physical_ids = np.asarray(
        [int(row["physical_action_id"]) for row in split_rows], dtype=np.int64
    )
    action_cells = np.asarray(
        [
            int(row["physical_stratum_index"]) * 4
            + int(row["offset_quartile_index"])
            for row in split_rows
        ],
        dtype=np.int16,
    )
    scenario_summaries: dict[str, Any] = {}
    selected_inflations: dict[str, float] = {}
    for scenario in select_scenarios(args.scenario):
        rows = read_csv_rows(likelihood_outcome_path(scenario, "validation"))
        conditions = hcr_e3.load_conditions(scenario, "validation")
        expected = len(conditions) * LIKELIHOOD_ACTIONS_PER_ROLE
        if len(rows) != expected:
            raise RuntimeError(f"{scenario} likelihood Validation rows 数量错误: {len(rows)}")
        rows.sort(
            key=lambda row: (
                int(row["condition_index_within_role"]),
                int(row["role_action_index"]),
            )
        )
        truth_matrix = np.asarray(
            [[float(row[field]) for field in OUTCOME_FIELDS] for row in rows],
            dtype=np.float64,
        ).reshape(len(conditions), 144, 3)
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
        valid_matrix = valid.reshape(len(conditions), 144)
        statistics_path = (
            RESULTS_ROOT
            / "continuous_likelihood"
            / scenario
            / "residual_statistics.npz"
        )
        statistics = ResidualStatistics.load(statistics_path)
        raw = scenario_raw_predictions(scenario, conditions, physical_ids, library)
        bias_physical = statistics.residual_bias * OBSERVATION_SCALE
        primary = raw + bias_physical
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
        baseline = np.empty_like(raw)
        for condition_index, condition in enumerate(conditions):
            baseline[condition_index] = p1.predict(condition)[source]
        direct = np.broadcast_to(
            library.direct_outcome[physical_ids][None, :, :], raw.shape
        ).copy()
        truth = truth_matrix.reshape(-1, 3)
        methods = {
            "transfer_baseline_0": baseline.reshape(-1, 3),
            "transfer_diagnostic_1": direct.reshape(-1, 3),
            "transfer_primary_2": primary.reshape(-1, 3),
        }
        method_summaries = {
            name: transfer_method_summary(prediction[valid], truth[valid])
            for name, prediction in methods.items()
        }
        baseline_error = prediction_error(methods["transfer_baseline_0"], truth)[
            "target_normalized"
        ].reshape(len(conditions), 144)
        primary_error = prediction_error(methods["transfer_primary_2"], truth)[
            "target_normalized"
        ].reshape(len(conditions), 144)
        effect = primary_error - baseline_error
        effect[~valid_matrix] = np.nan
        effect_ci = condition_action_bootstrap(
            effect, action_cells, args.bootstrap_resamples
        )
        calibration = calibrate_scenario(
            scenario,
            rows,
            conditions,
            physical_ids,
            truth_matrix,
            valid_matrix,
            library,
            statistics,
        )
        selected_inflations[scenario] = float(
            calibration["selected_covariance_inflation"]
        )
        training_rows = read_csv_rows(
            likelihood_outcome_path(scenario, "training")
        )
        expected_training_rows = (
            len(hcr_e3.load_conditions(scenario, "training"))
            * LIKELIHOOD_ACTIONS_PER_ROLE
        )
        if len(training_rows) != expected_training_rows:
            raise RuntimeError(
                f"{scenario} likelihood Training rows 数量错误: "
                f"{len(training_rows)} != {expected_training_rows}"
            )
        training_validity = float(
            np.mean(
                [
                    int(row["quality_pass"]) == 1
                    and int(row["simulation_unstable"]) == 0
                    and int(row["contact_success"]) == 1
                    and int(row["stopped_by_threshold"]) == 1
                    for row in training_rows
                ]
            )
        )
        transfer_pass = bool(
            float(np.nanmean(effect)) < 0.0
            and effect_ci[1] < 0.0
            and float(np.mean(valid)) >= 0.99
            and training_validity >= 0.99
        )
        scenario_summary = {
            "protocol_version": PROTOCOL_VERSION,
            "friction_cone": FRICTION_CONE,
            "scenario": scenario,
            "validation_conditions": len(conditions),
            "actions_per_condition": 144,
            "rollouts": len(rows),
            "valid_rollouts": int(valid.sum()),
            "validity_rate": float(np.mean(valid)),
            "training_validity_rate": training_validity,
            "method_summaries": method_summaries,
            "primary_minus_baseline_mean_target_normalized_error": float(
                np.nanmean(effect)
            ),
            "primary_minus_baseline_ci95": effect_ci,
            "paired_condition_action_bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": DEFAULT_SEED,
            "transfer_pass": transfer_pass,
            "calibration": calibration,
        }
        scenario_dir = RESULTS_ROOT / "transfer_validation" / scenario
        write_json(scenario_dir / "summary.json", scenario_summary)
        write_json(
            RESULTS_ROOT
            / "continuous_likelihood"
            / scenario
            / "calibration_summary.json",
            calibration,
        )
        scenario_summaries[scenario] = scenario_summary
    all_scenarios_available = set(scenario_summaries) == set(SCENARIOS)
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "scenario_summaries": scenario_summaries,
        "selected_covariance_inflation": selected_inflations,
        "all_transfer_requirements_passed": bool(
            all_scenarios_available
            and all(summary["transfer_pass"] for summary in scenario_summaries.values())
        ),
    }
    write_json(RESULTS_ROOT / "transfer_validation" / "combined_summary.json", combined)
    if all_scenarios_available:
        configuration = {
            "protocol_version": PROTOCOL_VERSION,
            "friction_cone": FRICTION_CONE,
            "likelihood_family": "multivariate_student_t",
            "student_t_degrees_of_freedom": STUDENT_T_DEGREES_OF_FREEDOM,
            "points_per_dimension": POINTS_PER_DIMENSION,
            "belief_update_horizon": BELIEF_UPDATE_HORIZON,
            "numerical_jitter": NUMERICAL_JITTER,
            "covariance_inflation": selected_inflations,
            "all_transfer_requirements_passed": combined[
                "all_transfer_requirements_passed"
            ],
            "test_outcomes_viewed": False,
        }
        write_json(
            RESULTS_ROOT
            / "continuous_likelihood"
            / "selected_configuration.json",
            configuration,
        )
    print(json.dumps(make_json_compatible(combined), ensure_ascii=False, indent=2))
    return combined


DISCRETE_BASELINE_CONTROLLER_ID = "belief_marginalised_closed_loop"
BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID = (
    "belief_marginalised_continuous_refinement"
)
CERTAINTY_EQUIVALENT_CONTINUOUS_CONTROLLER_ID = (
    "certainty_equivalent_continuous_refinement"
)
FULL_INFORMATION_CONTINUOUS_CONTROLLER_ID = (
    "continuous_full_information_state_feedback"
)
CONTINUOUS_CONTROLLER_NAMES = {
    BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID: (
        "Belief-Marginalised Continuous-Refinement Closed-Loop Controller"
    ),
    CERTAINTY_EQUIVALENT_CONTINUOUS_CONTROLLER_ID: (
        "Certainty-Equivalent Continuous-Refinement Closed-Loop Controller"
    ),
    FULL_INFORMATION_CONTINUOUS_CONTROLLER_ID: (
        "Continuous Full-Information State-Feedback Diagnostic"
    ),
}
CONTINUOUS_CONTROLLER_IDS = tuple(CONTINUOUS_CONTROLLER_NAMES)
DISCRETE_FULL_INFORMATION_CONTROLLER_ID = (
    "full_information_tensor_interpolation_state_feedback"
)
DISCRETE_ANCHOR_BUDGET = 20
DISCRETE_PROPOSAL_BUDGET = 100
SHORTLIST_BUDGETS = (64, 128, 256)
CONTINUOUS_PROMOTION_MARGIN = 0.05
MAXIMUM_PUSHES = 20
NODE_QUERY_CHUNK_SIZE = 65_536

E4_STEP_FIELDS = [
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
    "posterior_nll_post",
    "posterior_hpd_covered_post",
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
    "selected_action_type",
    "v2_action_id",
    "source_v2_action_id",
    "source_action_index",
    "continuous_physical_action_id",
    "continuous_physical_action_key",
    "continuous_candidate_rank",
    "normalized_offset_norm",
    "candidate_source",
    "candidate_count",
    "shortlist_budget",
    "predicted_discrete_tnpo_cost",
    "predicted_continuous_tnpo_cost",
    "predicted_promotion_margin",
    "predicted_selected_tnpo_cost",
    "true_condition_predicted_delta_x_m",
    "true_condition_predicted_delta_y_m",
    "true_condition_predicted_delta_yaw_rad",
    "source_anchor_predicted_delta_x_m",
    "source_anchor_predicted_delta_y_m",
    "source_anchor_predicted_delta_yaw_rad",
    "executed_prediction_tnpo_error",
    "source_anchor_prediction_tnpo_error",
    "actual_position_error_m",
    "actual_yaw_error_rad",
    "actual_tnpo_cost",
    "maximum_yaw_deviation_rad",
    "cumulative_absolute_yaw_excursion_rad",
    "success_after_push",
    "valid_observation",
    "quality_pass",
    "simulation_unstable",
    "contact_success",
    "stopped_by_threshold",
    "num_contacts",
    "settle_time_s",
    "proposal_latency_s",
    "discrete_selection_latency_s",
    "continuous_screening_latency_s",
    "continuous_scoring_latency_s",
    "belief_update_latency_s",
    "simulation_latency_s",
    "episode_success",
    "episode_invalid",
    "terminal_reason",
    "terminal_push_count",
    "is_terminal_push",
]


@dataclass
class E4BeliefState:
    """保存 fixed-node posterior 与 true-condition calibration state。"""

    cumulative_node_log_likelihoods: torch.Tensor
    log_weights: torch.Tensor
    cumulative_true_log_likelihood: float = 0.0
    update_count: int = 0

    @property
    def weights(self) -> torch.Tensor:
        """返回归一化 posterior weights。"""

        return torch.softmax(self.log_weights, dim=0)


@dataclass(frozen=True)
class ContinuousActionDecision:
    """记录一次 discrete-and-continuous protected action decision。"""

    selected_action_type: str
    source_action_index: int
    source_anchor_rank: int
    continuous_physical_action_id: int | None
    selected_score: float
    discrete_score: float
    continuous_score: float
    promotion_margin: float
    candidate_source: str
    candidate_count: int
    shortlist_budget: int
    proposal_latency_s: float
    discrete_selection_latency_s: float
    continuous_screening_latency_s: float
    continuous_scoring_latency_s: float


def selected_continuous_configuration() -> dict[str, Any]:
    """读取并核对 Validation 已选择的 continuous likelihood 配置。"""

    path = (
        RESULTS_ROOT / "continuous_likelihood" / "selected_configuration.json"
    )
    if not path.exists():
        raise FileNotFoundError("请先完成 evaluate-transfer-calibration")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not bool(payload["all_transfer_requirements_passed"]):
        raise RuntimeError("continuous transfer/calibration 未通过 Validation gate")
    return payload


def selected_shortlist_budget(allow_provisional: bool = True) -> int:
    """读取最终或 provisional shortlist budget。"""

    path = RESULTS_ROOT / "shortlist_validation" / "selected_configuration.json"
    if not path.exists():
        raise FileNotFoundError("请先完成 off-policy shortlist Validation")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("final_shortlist_budget") is not None:
        return int(payload["final_shortlist_budget"])
    if allow_provisional and payload.get("provisional_shortlist_budget") is not None:
        return int(payload["provisional_shortlist_budget"])
    raise RuntimeError("shortlist Validation 尚未产生可运行的 budget")


class ContinuousDecisionEngine:
    """组合 HCR V2 belief route 与 bounded continuous refinement。"""

    def __init__(
        self,
        scenario: str,
        device: torch.device,
        shortlist_budget: int,
        node_query_chunk_size: int = NODE_QUERY_CHUNK_SIZE,
    ):
        self.scenario = scenario
        self.device = device
        self.shortlist_budget = int(shortlist_budget)
        self.node_query_chunk_size = int(node_query_chunk_size)
        self.action_ids, self.ordered_actions = load_ordered_actions()
        self.action_id_to_index = {
            action_id: index for index, action_id in enumerate(self.action_ids)
        }
        nodes_np, prior_np = hcr_e3.make_fixed_quadrature(scenario)
        self.nodes = torch.from_numpy(nodes_np.astype(np.float32)).to(device)
        self.prior_weights = torch.from_numpy(prior_np.astype(np.float32)).to(device)
        self.log_prior_weights = torch.log(self.prior_weights)
        self.outcome_grid = hcr_e3.load_p1_grid(
            scenario, self.action_ids, device
        )
        self.node_outcomes = interpolate_outcome_grid(
            self.outcome_grid, self.nodes
        ).float()
        self.proposal_model, self.task_normaliser = hcr_e3.load_trained_proposal(
            scenario, device
        )
        self.library_cpu = CandidateLibrary.load()
        self.continuous_residual = torch.from_numpy(
            self.library_cpu.continuous_residual
        ).to(device)
        self.continuous_q = torch.from_numpy(self.library_cpu.continuous_q).to(device)
        self.anchor_q = torch.from_numpy(
            np.asarray(
                [car_e2.action_coordinate(row) for row in self.ordered_actions],
                dtype=np.float32,
            )
        ).to(device)
        self.observation_scale = torch.tensor(
            OBSERVATION_SCALE, dtype=torch.float32, device=device
        )
        self.discrete_bias, self.discrete_precision = hcr_e3.load_residual_parameters(
            scenario, device
        )
        discrete_covariance_path = (
            PROJECT_ROOT
            / "results"
            / "hcr_v2"
            / "e2"
            / "residual_statistics"
            / scenario
            / "residual_statistics.npz"
        )
        with np.load(discrete_covariance_path, allow_pickle=False) as payload:
            discrete_covariance = np.asarray(
                payload["base_covariance"], dtype=np.float64
            )
        discrete_scale = (
            hcr_e3.COVARIANCE_INFLATION[scenario] * discrete_covariance
            + NUMERICAL_JITTER * np.eye(3, dtype=np.float64)
        )
        self.discrete_log_normalization = self._student_t_log_normalization(
            discrete_scale
        )

        continuous_configuration = selected_continuous_configuration()
        statistics_path = (
            RESULTS_ROOT
            / "continuous_likelihood"
            / scenario
            / "residual_statistics.npz"
        )
        statistics = ResidualStatistics.load(statistics_path)
        inflation = float(
            continuous_configuration["covariance_inflation"][scenario]
        )
        continuous_scale = (
            inflation * statistics.base_covariance
            + NUMERICAL_JITTER * np.eye(3, dtype=np.float64)
        )
        self.continuous_bias = torch.from_numpy(
            statistics.residual_bias.astype(np.float32)
        ).to(device)
        self.continuous_bias_physical = (
            self.continuous_bias * self.observation_scale
        )
        self.continuous_precision = torch.from_numpy(
            np.linalg.inv(continuous_scale).astype(np.float32)
        ).to(device)
        self.continuous_log_normalization = self._student_t_log_normalization(
            continuous_scale
        )

    @staticmethod
    def _student_t_log_normalization(scale_matrix: np.ndarray) -> float:
        """计算三维 Student-t 的 log normalization。"""

        degrees = STUDENT_T_DEGREES_OF_FREEDOM
        return float(
            math.lgamma((degrees + 3.0) / 2.0)
            - math.lgamma(degrees / 2.0)
            - 1.5 * math.log(degrees * math.pi)
            - 0.5 * np.linalg.slogdet(scale_matrix)[1]
        )

    def _autocast(self):
        """只为 proposal MLP 启用 BF16。"""

        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _synchronize(self) -> None:
        """在计时边界同步 CUDA stream。"""

        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def new_belief(self) -> E4BeliefState:
        """建立 bounded-uniform fixed-node prior。"""

        zeros = torch.zeros_like(self.prior_weights)
        return E4BeliefState(zeros, self.log_prior_weights.clone())

    def belief_summary(self, belief: E4BeliefState) -> dict[str, Any]:
        """汇总 posterior moments 与 true-condition calibration。"""

        weights = belief.weights
        mean = torch.einsum("n,nd->d", weights, self.nodes)
        centered = self.nodes - mean[None, :]
        covariance = torch.einsum("n,ni,nj->ij", weights, centered, centered)
        summary: dict[str, Any] = {
            "mean_normalised": mean.detach().cpu().numpy(),
            "covariance_trace_normalised": float(torch.trace(covariance)),
            "update_count": belief.update_count,
            "posterior_nll": None,
            "hpd_covered": None,
        }
        if belief.update_count > 0:
            log_evidence = torch.logsumexp(
                self.log_prior_weights + belief.cumulative_node_log_likelihoods,
                dim=0,
            )
            dimension = self.nodes.shape[1]
            true_log_density = (
                -dimension * math.log(2.0)
                + belief.cumulative_true_log_likelihood
                - float(log_evidence)
            )
            node_log_density = (
                -dimension * math.log(2.0)
                + belief.log_weights
                - self.log_prior_weights
            )
            order = torch.argsort(node_log_density, descending=True, stable=True)
            cumulative = torch.cumsum(weights[order], dim=0)
            threshold_index = int(
                torch.searchsorted(
                    cumulative,
                    torch.tensor(0.95, dtype=cumulative.dtype, device=self.device),
                )
            )
            threshold_index = min(threshold_index, len(order) - 1)
            threshold = float(node_log_density[order[threshold_index]])
            summary["posterior_nll"] = -true_log_density
            summary["hpd_covered"] = int(true_log_density >= threshold - 1e-12)
        return summary

    def _student_t_log_likelihood(
        self,
        residual: torch.Tensor,
        precision: torch.Tensor,
        log_normalization: float,
    ) -> torch.Tensor:
        """计算 GPU Student-t log likelihood。"""

        mahalanobis = torch.einsum(
            "...i,ij,...j->...", residual, precision, residual
        )
        return log_normalization - 0.5 * (
            STUDENT_T_DEGREES_OF_FREEDOM + 3.0
        ) * torch.log1p(mahalanobis / STUDENT_T_DEGREES_OF_FREEDOM)

    @torch.inference_mode()
    def update_belief(
        self,
        belief: E4BeliefState,
        decision: ContinuousActionDecision,
        observation: np.ndarray,
        true_hidden_parameters: np.ndarray,
    ) -> bool:
        """根据实际执行的 discrete 或 continuous action 更新 belief。"""

        return self.update_action_observation(
            belief=belief,
            action_type=decision.selected_action_type,
            source_action_index=decision.source_action_index,
            continuous_physical_action_id=decision.continuous_physical_action_id,
            observation=observation,
            true_hidden_parameters=true_hidden_parameters,
        )

    @torch.inference_mode()
    def update_action_observation(
        self,
        belief: E4BeliefState,
        action_type: str,
        source_action_index: int,
        continuous_physical_action_id: int | None,
        observation: np.ndarray,
        true_hidden_parameters: np.ndarray,
    ) -> bool:
        """用显式 action token 更新 belief，供 episode 与 replay 共用。"""

        if belief.update_count >= BELIEF_UPDATE_HORIZON:
            return False
        observation_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32), device=self.device
        ) / self.observation_scale
        true_hidden = torch.as_tensor(
            np.asarray(true_hidden_parameters, dtype=np.float32)[None, :],
            device=self.device,
        )
        true_outcomes = interpolate_outcome_grid(self.outcome_grid, true_hidden)[0]
        if action_type == "continuous":
            physical_id = int(continuous_physical_action_id)
            source_index = int(source_action_index)
            residual_physical = self.continuous_residual[physical_id]
            node_means = (
                self.node_outcomes[:, source_index]
                + residual_physical[None, :]
            ) / self.observation_scale[None, :] + self.continuous_bias[None, :]
            true_mean = (
                true_outcomes[source_index] + residual_physical
            ) / self.observation_scale + self.continuous_bias
            precision = self.continuous_precision
            log_normalization = self.continuous_log_normalization
        else:
            source_index = int(source_action_index)
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
        belief.cumulative_node_log_likelihoods += node_log_likelihoods
        belief.cumulative_true_log_likelihood += float(true_log_likelihood)
        unnormalized = (
            self.log_prior_weights + belief.cumulative_node_log_likelihoods
        )
        belief.log_weights = unnormalized - torch.logsumexp(unnormalized, dim=0)
        belief.update_count += 1
        return True

    @staticmethod
    def _stable_ranked_candidates(
        candidates: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """按 score 和较小 manifest index 给候选确定性排序。"""

        action_order = torch.argsort(candidates, stable=True)
        sorted_candidates = candidates[action_order]
        sorted_scores = scores[action_order]
        score_order = torch.argsort(sorted_scores, stable=True)
        return sorted_candidates[score_order]

    @torch.inference_mode()
    def discrete_anchors(
        self,
        controller_id: str,
        task_query: np.ndarray,
        belief: E4BeliefState,
        true_hidden_parameters: np.ndarray,
    ) -> tuple[torch.Tensor, float, str, float, float]:
        """生成并按正式 discrete score 排序 Top-20 anchors。"""

        raw_task = torch.as_tensor(
            np.asarray(task_query, dtype=np.float32)[None, :], device=self.device
        )
        normalized_task = self.task_normaliser.transform(raw_task)
        posterior_weights = belief.weights[None, :]
        true_hidden = torch.as_tensor(
            np.asarray(true_hidden_parameters, dtype=np.float32)[None, :],
            device=self.device,
        )
        self._synchronize()
        proposal_start = time.perf_counter()
        if controller_id == BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID:
            with self._autocast():
                probabilities = belief_marginalised_probabilities(
                    self.proposal_model,
                    normalized_task,
                    self.nodes,
                    posterior_weights,
                    node_query_chunk_size=self.node_query_chunk_size,
                )
            candidates, _ = topk_probabilities(
                probabilities, DISCRETE_PROPOSAL_BUDGET
            )
            candidate_source = "belief_marginalised_top100"
            hidden = None
        elif controller_id == CERTAINTY_EQUIVALENT_CONTINUOUS_CONTROLLER_ID:
            hidden = posterior_mean_hidden_parameters(
                self.nodes, posterior_weights
            )
            with self._autocast():
                probabilities = self.proposal_model.point_action_probabilities(
                    normalized_task, hidden
                )
            candidates, _ = topk_probabilities(
                probabilities, DISCRETE_PROPOSAL_BUDGET
            )
            candidate_source = "posterior_mean_top100"
        else:
            hidden = true_hidden
            candidates = torch.arange(
                ACTION_COUNT, dtype=torch.long, device=self.device
            )[None, :]
            candidate_source = "full_action_library"
        self._synchronize()
        proposal_latency = time.perf_counter() - proposal_start

        selection_start = time.perf_counter()
        if controller_id == BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID:
            scores = posterior_expected_candidate_scores(
                self.node_outcomes,
                posterior_weights,
                candidates,
                raw_task,
                case_chunk_size=1,
            )[0]
        else:
            scores = point_condition_candidate_scores(
                self.outcome_grid,
                hidden,
                candidates,
                raw_task,
                case_chunk_size=1,
            )[0]
        ranked = self._stable_ranked_candidates(candidates[0], scores)
        anchors = ranked[:DISCRETE_ANCHOR_BUDGET]
        selected_slot = torch.nonzero(candidates[0] == anchors[0], as_tuple=False)[0, 0]
        discrete_score = float(scores[selected_slot])
        self._synchronize()
        selection_latency = time.perf_counter() - selection_start
        return anchors, discrete_score, candidate_source, proposal_latency, selection_latency

    def continuous_physical_ids(self, anchors: torch.Tensor) -> torch.Tensor:
        """把 Top-20 source indices 展开为 2,560 个 physical action IDs。"""

        ranks = torch.arange(
            CONTINUOUS_CANDIDATES_PER_ANCHOR,
            dtype=torch.long,
            device=self.device,
        )
        return (
            anchors[:, None] * CONTINUOUS_CANDIDATES_PER_ANCHOR
            + ranks[None, :]
        ).reshape(-1)

    @torch.inference_mode()
    def screening_scores(
        self,
        physical_ids: torch.Tensor,
        task_query: np.ndarray,
        belief: E4BeliefState,
        true_hidden_parameters: np.ndarray,
        controller_id: str,
    ) -> torch.Tensor:
        """在 posterior mean 或 true condition 下计算 2,560 个 screening costs。"""

        if controller_id == FULL_INFORMATION_CONTINUOUS_CONTROLLER_ID:
            hidden = torch.as_tensor(
                np.asarray(true_hidden_parameters, dtype=np.float32)[None, :],
                device=self.device,
            )
        else:
            hidden = posterior_mean_hidden_parameters(
                self.nodes, belief.weights[None, :]
            )
        point_outcomes = interpolate_outcome_grid(self.outcome_grid, hidden)[0]
        source = torch.div(
            physical_ids,
            CONTINUOUS_CANDIDATES_PER_ANCHOR,
            rounding_mode="floor",
        )
        continuous_outcomes = (
            point_outcomes[source]
            + self.continuous_residual[physical_ids]
            + self.continuous_bias_physical[None, :]
        )
        task = torch.as_tensor(
            np.asarray(task_query, dtype=np.float32)[None, :], device=self.device
        )
        return tnpo_costs(continuous_outcomes[None, :, :], task)[0]

    @torch.inference_mode()
    def expected_continuous_scores(
        self,
        physical_ids: torch.Tensor,
        task_query: np.ndarray,
        belief: E4BeliefState,
        candidate_chunk_size: int = 256,
    ) -> torch.Tensor:
        """对 continuous candidates执行 exact posterior expected-cost scoring。"""

        task = torch.as_tensor(
            np.asarray(task_query, dtype=np.float32)[None, :], device=self.device
        )
        desired_yaw = torch.atan2(task[0, 2], task[0, 3])
        outputs = torch.empty(
            len(physical_ids), dtype=torch.float32, device=self.device
        )
        for start in range(0, len(physical_ids), candidate_chunk_size):
            stop = min(len(physical_ids), start + candidate_chunk_size)
            ids = physical_ids[start:stop]
            source = torch.div(
                ids,
                CONTINUOUS_CANDIDATES_PER_ANCHOR,
                rounding_mode="floor",
            )
            outcomes = (
                self.node_outcomes[:, source]
                + self.continuous_residual[ids][None, :, :]
                + self.continuous_bias_physical[None, None, :]
            )
            position_error = torch.linalg.vector_norm(
                outcomes[..., :2] - task[0, None, :2], dim=-1
            )
            yaw_error = torch.abs(
                torch_wrap_to_pi(outcomes[..., 2] - desired_yaw)
            )
            costs = (
                0.5 * position_error / PRIMARY_TNPO_COST.position_tolerance_m
                + 0.5 * yaw_error / PRIMARY_TNPO_COST.yaw_tolerance_rad
            )
            outputs[start:stop] = torch.einsum(
                "n,nk->k", belief.weights.float(), costs.float()
            )
        return outputs

    @torch.inference_mode()
    def select_action(
        self,
        controller_id: str,
        task_query: np.ndarray,
        belief: E4BeliefState,
        true_hidden_parameters: np.ndarray,
    ) -> ContinuousActionDecision:
        """执行 Top-20 refinement、screening、exact scoring 与 protected promotion。"""

        anchors, discrete_score, source_name, proposal_latency, discrete_latency = (
            self.discrete_anchors(
                controller_id, task_query, belief, true_hidden_parameters
            )
        )
        physical_ids = self.continuous_physical_ids(anchors)
        self._synchronize()
        screening_start = time.perf_counter()
        screen_scores = self.screening_scores(
            physical_ids,
            task_query,
            belief,
            true_hidden_parameters,
            controller_id,
        )
        physical_order = torch.argsort(physical_ids, stable=True)
        score_order = torch.argsort(
            screen_scores[physical_order], stable=True
        )
        ranked_physical = physical_ids[physical_order][score_order]
        self._synchronize()
        screening_latency = time.perf_counter() - screening_start

        self._synchronize()
        scoring_start = time.perf_counter()
        if controller_id == BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID:
            compared_ids = ranked_physical[: self.shortlist_budget]
            compared_scores = self.expected_continuous_scores(
                compared_ids, task_query, belief
            )
            stable = self._stable_ranked_candidates(compared_ids, compared_scores)
            best_continuous_id = int(stable[0])
            score_slot = torch.nonzero(
                compared_ids == best_continuous_id, as_tuple=False
            )[0, 0]
            continuous_score = float(compared_scores[score_slot])
        else:
            best_continuous_id = int(ranked_physical[0])
            score_slot = torch.nonzero(
                physical_ids == best_continuous_id, as_tuple=False
            )[0, 0]
            continuous_score = float(screen_scores[score_slot])
        self._synchronize()
        scoring_latency = time.perf_counter() - scoring_start
        promotion_margin = discrete_score - continuous_score
        select_continuous = promotion_margin >= CONTINUOUS_PROMOTION_MARGIN
        selected_source_index = (
            best_continuous_id // CONTINUOUS_CANDIDATES_PER_ANCHOR
            if select_continuous
            else int(anchors[0])
        )
        source_anchor_rank = int(
            torch.nonzero(
                anchors == selected_source_index, as_tuple=False
            )[0, 0]
        ) + 1
        return ContinuousActionDecision(
            selected_action_type="continuous" if select_continuous else "discrete",
            source_action_index=selected_source_index,
            source_anchor_rank=source_anchor_rank,
            continuous_physical_action_id=(
                best_continuous_id if select_continuous else None
            ),
            selected_score=continuous_score if select_continuous else discrete_score,
            discrete_score=discrete_score,
            continuous_score=continuous_score,
            promotion_margin=promotion_margin,
            candidate_source=source_name,
            candidate_count=len(physical_ids),
            shortlist_budget=(
                self.shortlist_budget
                if controller_id == BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
                else len(physical_ids)
            ),
            proposal_latency_s=proposal_latency,
            discrete_selection_latency_s=discrete_latency,
            continuous_screening_latency_s=screening_latency,
            continuous_scoring_latency_s=scoring_latency,
        )

    @torch.inference_mode()
    def shortlist_diagnostic(
        self,
        task_query: np.ndarray,
        belief: E4BeliefState,
        true_hidden_parameters: np.ndarray,
    ) -> dict[int, dict[str, float | int]]:
        """比较 64/128/256 shortlist 与完整 2,560-candidate reference。"""

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
        screening_start = time.perf_counter()
        screen = self.screening_scores(
            physical_ids,
            task_query,
            belief,
            true_hidden_parameters,
            BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID,
        )
        physical_order = torch.argsort(physical_ids, stable=True)
        score_order = torch.argsort(screen[physical_order], stable=True)
        ranked = physical_ids[physical_order][score_order]
        self._synchronize()
        screening_latency = time.perf_counter() - screening_start
        self._synchronize()
        exact_start = time.perf_counter()
        exact = self.expected_continuous_scores(physical_ids, task_query, belief)
        self._synchronize()
        exact_latency = time.perf_counter() - exact_start
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
            if self.device.type == "cuda"
            else 0.0
        )
        exact_ranked = self._stable_ranked_candidates(physical_ids, exact)
        full_best_id = int(exact_ranked[0])
        full_slot = torch.nonzero(physical_ids == full_best_id, as_tuple=False)[0, 0]
        full_best_cost = float(exact[full_slot])
        outputs: dict[int, dict[str, float | int]] = {}
        for budget in SHORTLIST_BUDGETS:
            ids = ranked[:budget]
            slots = torch.searchsorted(
                torch.sort(physical_ids).values,
                ids,
            )
            sorted_physical, original_slots = torch.sort(physical_ids)
            exact_slots = original_slots[slots]
            costs = exact[exact_slots]
            best = self._stable_ranked_candidates(ids, costs)[0]
            best_slot = torch.nonzero(ids == best, as_tuple=False)[0, 0]
            shortlist_cost = float(costs[best_slot])
            outputs[budget] = {
                "full_best_physical_action_id": full_best_id,
                "shortlist_best_physical_action_id": int(best),
                "exact_top1_contained": int(torch.any(ids == full_best_id)),
                "shortlist_expected_cost_gap": shortlist_cost - full_best_cost,
                "screening_latency_s": screening_latency,
                "full_set_exact_marginalisation_latency_s": exact_latency,
                "peak_cuda_memory_mib": peak_memory,
            }
        return outputs

    @torch.inference_mode()
    def true_condition_predictions(
        self,
        decision: ContinuousActionDecision,
        true_hidden_parameters: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """返回执行动作与 source anchor 的 simulator-only predicted outcomes。"""

        hidden = torch.as_tensor(
            np.asarray(true_hidden_parameters, dtype=np.float32)[None, :],
            device=self.device,
        )
        outcomes = interpolate_outcome_grid(self.outcome_grid, hidden)[0]
        source_prediction = outcomes[decision.source_action_index]
        if decision.selected_action_type == "continuous":
            physical_id = int(decision.continuous_physical_action_id)
            selected_prediction = (
                source_prediction
                + self.continuous_residual[physical_id]
                + self.continuous_bias_physical
            )
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


def active_hidden_parameters(
    scenario: str, condition: dict[str, str]
) -> np.ndarray:
    """读取一个场景的 active normalized hidden coordinates。"""

    return np.asarray(
        [float(condition[field]) for field in SCENARIO_ACTIVE_COORDINATES[scenario]],
        dtype=np.float32,
    )


def read_pose(model: Any, data: Any, object_body_id: int) -> tuple[float, float, float]:
    """读取物体平面位置与 yaw。"""

    return (
        float(data.xpos[object_body_id][0]),
        float(data.xpos[object_body_id][1]),
        float(get_object_yaw_qpos(model, data)),
    )


def actual_tnpo_cost(position_error_m: float, yaw_error_rad: float) -> float:
    """计算真实 terminal pose 的 TNPO cost。"""

    return (
        0.5 * position_error_m / PRIMARY_TNPO_COST.position_tolerance_m
        + 0.5 * yaw_error_rad / PRIMARY_TNPO_COST.yaw_tolerance_rad
    )


def outcome_prediction_tnpo_error(
    prediction: np.ndarray, observation: np.ndarray
) -> float:
    """计算 predicted local outcome 与真实 observation 的 normalized error。"""

    position = float(
        np.linalg.vector_norm(
            np.asarray(prediction[:2], dtype=np.float64)
            - np.asarray(observation[:2], dtype=np.float64)
        )
    )
    yaw = abs(wrap_scalar(float(prediction[2]) - float(observation[2])))
    return actual_tnpo_cost(position, yaw)


def belief_log_fields(
    scenario: str, summary: dict[str, Any], suffix: str
) -> dict[str, Any]:
    """把 active posterior mean 写入统一三维日志字段。"""

    values = {
        "hidden_u_friction": 0.0,
        "hidden_u_com_x": 0.0,
        "hidden_u_com_y": 0.0,
    }
    mean = np.asarray(summary["mean_normalised"], dtype=np.float64)
    for field, value in zip(SCENARIO_ACTIVE_COORDINATES[scenario], mean):
        values[field] = float(value)
    return {
        f"belief_mean_hidden_u_friction_{suffix}": values["hidden_u_friction"],
        f"belief_mean_hidden_u_com_x_{suffix}": values["hidden_u_com_x"],
        f"belief_mean_hidden_u_com_y_{suffix}": values["hidden_u_com_y"],
        f"belief_covariance_trace_{suffix}": summary[
            "covariance_trace_normalised"
        ],
    }


def continuous_execution_action(
    action: dict[str, Any], physical_action_id: int
) -> dict[str, Any]:
    """补齐 HCR rollout input 需要的 continuous action fields。"""

    return {
        **action,
        "candidate_id": physical_action_id,
        "contact_region_id": action["source_contact_region_id"],
        "action_param_index": action["source_action_param_index"],
        "v2_action_id": action["source_v2_action_id"],
    }


def run_closed_loop_episode(
    model: Any,
    data: Any,
    object_body_id: int,
    engine: ContinuousDecisionEngine,
    controller_id: str,
    condition: dict[str, str],
    target: dict[str, str],
    episode_id: int,
    environment_xml: Path,
    role: str,
    maximum_pushes: int,
) -> list[dict[str, Any]]:
    """执行一个 continuous-refinement condition-target-controller episode。"""

    reset_input = hcr_e1.build_rollout_input(
        engine.ordered_actions[0], condition, episode_id
    )
    reset_state(model, data, reset_input)
    initial_x, initial_y, initial_yaw = read_pose(model, data, object_body_id)
    target_world_x = initial_x + float(target["target_delta_x_m"])
    target_world_y = initial_y + float(target["target_delta_y_m"])
    target_world_yaw = initial_yaw
    belief = engine.new_belief()
    true_hidden = active_hidden_parameters(engine.scenario, condition)
    episode_key = (
        f"{engine.scenario}|{condition['condition_id']}|"
        f"{target['v2_target_id']}|{controller_id}"
    )
    rows: list[dict[str, Any]] = []
    maximum_yaw_deviation = 0.0
    cumulative_yaw_excursion = 0.0
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
        selected_prediction, source_prediction, offset_norm = (
            engine.true_condition_predictions(decision, true_hidden)
        )
        source_action = engine.ordered_actions[decision.source_action_index]
        continuous_action: dict[str, Any] | None = None
        if decision.selected_action_type == "continuous":
            physical_id = int(decision.continuous_physical_action_id)
            continuous_action = engine.library_cpu.action_row(
                physical_id, engine.ordered_actions
            )
            action = continuous_execution_action(continuous_action, physical_id)
        else:
            physical_id = -1
            action = source_action
        rollout_input = hcr_e1.build_rollout_input(action, condition, episode_id)
        rollout_input["dataset_role"] = (
            f"car_e4_{role}_{engine.scenario}_{controller_id}"
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
        observation = np.asarray(
            [
                math.cos(pre_yaw) * (metric_x - pre_x)
                + math.sin(pre_yaw) * (metric_y - pre_y),
                -math.sin(pre_yaw) * (metric_x - pre_x)
                + math.cos(pre_yaw) * (metric_y - pre_y),
                wrap_scalar(metric_yaw - pre_yaw),
            ],
            dtype=np.float64,
        )
        belief_updated = False
        belief_update_latency = 0.0
        belief_controller = controller_id in {
            BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID,
            CERTAINTY_EQUIVALENT_CONTINUOUS_CONTROLLER_ID,
        }
        if valid_observation and belief_controller:
            engine._synchronize()
            update_start = time.perf_counter()
            belief_updated = engine.update_belief(
                belief, decision, observation, true_hidden
            )
            engine._synchronize()
            belief_update_latency = time.perf_counter() - update_start
        post_belief = engine.belief_summary(belief)

        position_error = math.hypot(
            metric_x - target_world_x, metric_y - target_world_y
        )
        yaw_error = abs(wrap_scalar(metric_yaw - target_world_yaw))
        maximum_yaw_deviation = max(
            maximum_yaw_deviation,
            abs(wrap_scalar(metric_yaw - initial_yaw)),
        )
        if reliable_post_pose:
            cumulative_yaw_excursion += abs(wrap_scalar(metric_yaw - pre_yaw))
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

        continuous_key = (
            continuous_action["physical_action_key"]
            if continuous_action is not None
            else ""
        )
        row: dict[str, Any] = {
            "experiment_id": "E4",
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
            "controller_name": CONTINUOUS_CONTROLLER_NAMES[controller_id],
            "controller_id": controller_id,
            "episode_key": episode_key,
            "push_index": push_index,
            "attempted_push_count": push_index,
            "maximum_push_budget": maximum_pushes,
            "valid_update_count_pre": pre_belief["update_count"],
            "valid_update_count_post": post_belief["update_count"],
            "belief_updated": int(belief_updated),
            "posterior_nll_post": post_belief["posterior_nll"],
            "posterior_hpd_covered_post": post_belief["hpd_covered"],
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
            "selected_action_type": decision.selected_action_type,
            "v2_action_id": source_action["v2_action_id"],
            "source_v2_action_id": source_action["v2_action_id"],
            "source_action_index": decision.source_action_index,
            "source_anchor_rank": decision.source_anchor_rank,
            "continuous_physical_action_id": (
                physical_id if physical_id >= 0 else ""
            ),
            "continuous_physical_action_key": continuous_key,
            "continuous_candidate_rank": (
                continuous_action["e4_candidate_rank"]
                if continuous_action is not None
                else ""
            ),
            "normalized_offset_norm": offset_norm,
            "candidate_source": decision.candidate_source,
            "candidate_count": decision.candidate_count,
            "shortlist_budget": decision.shortlist_budget,
            "predicted_discrete_tnpo_cost": decision.discrete_score,
            "predicted_continuous_tnpo_cost": decision.continuous_score,
            "predicted_promotion_margin": decision.promotion_margin,
            "predicted_selected_tnpo_cost": decision.selected_score,
            "true_condition_predicted_delta_x_m": selected_prediction[0],
            "true_condition_predicted_delta_y_m": selected_prediction[1],
            "true_condition_predicted_delta_yaw_rad": selected_prediction[2],
            "source_anchor_predicted_delta_x_m": source_prediction[0],
            "source_anchor_predicted_delta_y_m": source_prediction[1],
            "source_anchor_predicted_delta_yaw_rad": source_prediction[2],
            "executed_prediction_tnpo_error": (
                outcome_prediction_tnpo_error(selected_prediction, observation)
                if valid_observation
                else ""
            ),
            "source_anchor_prediction_tnpo_error": (
                outcome_prediction_tnpo_error(source_prediction, observation)
                if valid_observation
                else ""
            ),
            "actual_position_error_m": position_error,
            "actual_yaw_error_rad": yaw_error,
            "actual_tnpo_cost": actual_tnpo_cost(position_error, yaw_error),
            "maximum_yaw_deviation_rad": maximum_yaw_deviation,
            "cumulative_absolute_yaw_excursion_rad": cumulative_yaw_excursion,
            "success_after_push": int(success),
            "valid_observation": int(valid_observation),
            "quality_pass": result["quality_pass"],
            "simulation_unstable": result["simulation_unstable"],
            "contact_success": result["contact_success"],
            "stopped_by_threshold": result["stopped_by_threshold"],
            "num_contacts": result["num_contacts"],
            "settle_time_s": result["settle_time_s"],
            "proposal_latency_s": decision.proposal_latency_s,
            "discrete_selection_latency_s": decision.discrete_selection_latency_s,
            "continuous_screening_latency_s": decision.continuous_screening_latency_s,
            "continuous_scoring_latency_s": decision.continuous_scoring_latency_s,
            "belief_update_latency_s": belief_update_latency,
            "simulation_latency_s": simulation_latency,
        }
        row.update(belief_log_fields(engine.scenario, pre_belief, "pre"))
        row.update(belief_log_fields(engine.scenario, post_belief, "post"))
        rows.append(row)
        if terminal_reason in {"invalid_push", "success"}:
            break

    for row_index, row in enumerate(rows):
        row["episode_success"] = int(terminal_reason == "success")
        row["episode_invalid"] = int(terminal_reason == "invalid_push")
        row["terminal_reason"] = terminal_reason
        row["terminal_push_count"] = len(rows)
        row["is_terminal_push"] = int(row_index == len(rows) - 1)
    return rows


def load_sequential_targets(role: str) -> list[dict[str, str]]:
    """读取 64 个 E5 Sequential-Extension targets。"""

    rows = read_csv_rows(SEQUENTIAL_TARGET_PATHS[role])
    if len(rows) != 64:
        raise RuntimeError(f"{role} Sequential-Extension target 数量错误: {len(rows)}")
    return rows


def parse_controllers(value: str) -> tuple[str, ...]:
    """解析 continuous controller IDs。"""

    if value == "all":
        return CONTINUOUS_CONTROLLER_IDS
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(values) - set(CONTINUOUS_CONTROLLER_IDS)
    if unknown:
        raise ValueError(f"未知 continuous controllers: {sorted(unknown)}")
    return values


def inspect_complete_shard(
    path: Path,
    target_ids: set[str],
    controller_ids: tuple[str, ...],
    role: str,
    maximum_pushes: int,
    shortlist_budget: int,
) -> bool:
    """检查 closed-loop condition shard 是否完整。"""

    if not path.exists():
        return False
    rows = read_csv_rows(path)
    terminal = [row for row in rows if int(row["is_terminal_push"]) == 1]
    observed = {(row["target_id"], row["controller_id"]) for row in terminal}
    expected = {
        (target_id, controller_id)
        for target_id in target_ids
        for controller_id in controller_ids
    }
    primary_rows = [
        row
        for row in rows
        if row["controller_id"]
        == BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
    ]
    return (
        observed == expected
        and {row["protocol_version"] for row in rows} == {PROTOCOL_VERSION}
        and {row["role"] for row in rows} == {role}
        and {int(row["maximum_push_budget"]) for row in rows} == {maximum_pushes}
        and (
            BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID not in controller_ids
            or {int(row["shortlist_budget"]) for row in primary_rows}
            == {shortlist_budget}
        )
    )


def summarize_condition_rows(
    rows: list[dict[str, Any]] | list[dict[str, str]],
    scenario: str,
    condition_id: str,
    path: Path,
    resumed: int,
) -> dict[str, Any]:
    """汇总一个 condition shard。"""

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
        "resumed": resumed,
    }


CLOSED_LOOP_WORKER_CONTEXT: dict[str, Any] = {}


def initialize_closed_loop_worker(
    scenario: str,
    targets: list[dict[str, str]],
    controller_ids: tuple[str, ...],
    role: str,
    maximum_pushes: int,
    shortlist_budget: int,
    node_query_chunk_size: int,
) -> None:
    """为 Windows process worker 加载一次 GPU artifacts。"""

    torch.set_num_threads(1)
    device = car_e2.require_cuda()
    engine = ContinuousDecisionEngine(
        scenario,
        device,
        shortlist_budget,
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
    """在一个 process worker 中采集完整 condition shard。"""

    condition_index, condition_count, condition, xml_text, output_text = task
    context = CLOSED_LOOP_WORKER_CONTEXT
    engine: ContinuousDecisionEngine = context["engine"]
    environment_xml = Path(xml_text)
    model, data = load_model(environment_xml)
    hcr_e1.set_sliding_friction(model, float(condition["friction_sliding_mu"]))
    object_body_id = get_body_id(model)
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(context["targets"]):
        for controller_index, controller_id in enumerate(context["controller_ids"]):
            episode_id = (
                int(condition["condition_index_within_role"]) * 100_000
                + target_index * 10
                + controller_index
            )
            rows.extend(
                run_closed_loop_episode(
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
            )
    output_path = Path(output_text)
    write_csv(output_path, rows, E4_STEP_FIELDS)
    summary = summarize_condition_rows(
        rows,
        context["scenario"],
        condition["condition_id"],
        output_path,
        resumed=0,
    )
    summary["condition_index"] = condition_index
    summary["condition_count"] = condition_count
    summary["worker_process_id"] = os.getpid()
    summary["worker_peak_cuda_memory_mib"] = (
        torch.cuda.max_memory_allocated(engine.device) / (1024.0**2)
    )
    summary["worker_peak_cuda_reserved_mib"] = (
        torch.cuda.max_memory_reserved(engine.device) / (1024.0**2)
    )
    return summary


def validation_gate_passed() -> bool:
    """读取 closed-loop Validation gate。"""

    path = RESULTS_ROOT / "evaluation" / "validation" / "gate_decision.json"
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        return bool(json.load(handle).get("passed", False))


def selected_worker_count(shortlist_budget: int) -> int:
    """读取 concurrent benchmark 选择的正式 worker count。"""

    path = RESULTS_ROOT / "benchmark" / "selected_configuration.json"
    if not path.exists():
        raise FileNotFoundError("请先完成 benchmark-workers")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload["shortlist_budget"]) != int(shortlist_budget):
        raise RuntimeError("shortlist budget 已变化，请重新运行 benchmark-workers")
    return int(payload["selected_num_workers"])


def collect_closed_loop(args: argparse.Namespace) -> dict[str, Any]:
    """采集 E4 Validation 或 predefined shared Test trajectories。"""

    if args.role == "test" and not validation_gate_passed():
        raise RuntimeError("Validation selection-level gate 未通过，不能访问 Test")
    controller_ids = parse_controllers(args.controllers)
    targets = load_sequential_targets(args.role)
    if args.max_targets > 0:
        targets = targets[: args.max_targets]
    shortlist_budget = selected_shortlist_budget(allow_provisional=args.role == "validation")
    formal_run = (
        controller_ids == CONTINUOUS_CONTROLLER_IDS
        and args.max_conditions <= 0
        and args.max_targets <= 0
        and args.maximum_pushes == MAXIMUM_PUSHES
    )
    collection_root = (
        DATA_ROOT / "closed_loop" if formal_run else DATA_ROOT / "smoke"
    )
    if formal_run:
        benchmark_workers_count = selected_worker_count(shortlist_budget)
        if args.num_workers not in {0, benchmark_workers_count}:
            raise ValueError(
                "正式采集必须使用 benchmark 选择的 num-workers="
                f"{benchmark_workers_count}"
            )
        requested_workers = benchmark_workers_count
    else:
        requested_workers = args.num_workers if args.num_workers > 0 else 1
    all_results: list[dict[str, Any]] = []
    for scenario in select_scenarios(args.scenario):
        conditions = hcr_e3.load_conditions(scenario, args.role)
        if args.max_conditions > 0:
            conditions = conditions[: args.max_conditions]
        generated_dir = DATA_ROOT / "generated_xml" / args.role / scenario
        xml_by_com = hcr_e1.prepare_environment_xmls(conditions, generated_dir)
        pending = []
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
                shortlist_budget,
            ):
                result = summarize_condition_rows(
                    read_csv_rows(output_path),
                    scenario,
                    condition["condition_id"],
                    output_path,
                    resumed=1,
                )
                all_results.append(result)
                print(
                    f"resumed {scenario} condition {condition_index}/{len(conditions)}"
                )
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
                    shortlist_budget,
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
        "shortlist_budget": shortlist_budget,
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
        collection_root
        / args.role
        / f"collection_summary_{args.scenario}.json",
        summary,
    )
    print(json.dumps(make_json_compatible(summary), ensure_ascii=False, indent=2))
    return summary


def benchmark_targets() -> list[dict[str, str]]:
    """从四个 Sequential-Extension strata 各取两个稳定 targets。"""

    targets = load_sequential_targets("validation")
    selected: list[dict[str, str]] = []
    for stratum in sorted({row["target_stratum"] for row in targets}):
        rows = sorted(
            [row for row in targets if row["target_stratum"] == stratum],
            key=lambda row: row["v2_target_id"],
        )
        selected.extend(evenly_spaced_rows(rows, 2))
    if len(selected) != 8:
        raise RuntimeError(f"benchmark targets 数量错误: {len(selected)}")
    return selected


def run_worker_benchmark_configuration(
    num_workers: int,
    shortlist_budget: int,
    node_query_chunk_size: int,
) -> dict[str, Any]:
    """运行一个 Joint concurrent worker-count benchmark。"""

    scenario = "joint"
    role = "validation"
    targets = benchmark_targets()
    conditions = hcr_e3.load_conditions(scenario, role)[:6]
    generated_dir = DATA_ROOT / "generated_xml" / "benchmark" / scenario
    xml_by_com = hcr_e1.prepare_environment_xmls(conditions, generated_dir)
    output_root = DATA_ROOT / "benchmark" / f"workers_{num_workers}"
    tasks = []
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
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    completion_times: list[float] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize_closed_loop_worker,
        initargs=(
            scenario,
            targets,
            CONTINUOUS_CONTROLLER_IDS,
            role,
            MAXIMUM_PUSHES,
            shortlist_budget,
            node_query_chunk_size,
        ),
    ) as executor:
        futures = [executor.submit(collect_condition_task, task) for task in tasks]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(repr(error))
            completion_times.append(time.perf_counter() - start)
    elapsed = time.perf_counter() - start
    worker_memory: dict[int, float] = {}
    worker_reserved_memory: dict[int, float] = {}
    for result in results:
        process_id = int(result["worker_process_id"])
        worker_memory[process_id] = max(
            worker_memory.get(process_id, 0.0),
            float(result["worker_peak_cuda_memory_mib"]),
        )
        worker_reserved_memory[process_id] = max(
            worker_reserved_memory.get(process_id, 0.0),
            float(result["worker_peak_cuda_reserved_mib"]),
        )
    episodes = sum(int(result["episodes"]) for result in results)
    decisions = sum(int(result["step_rows"]) for result in results)
    invalid = sum(int(result["invalid_episodes"]) for result in results)
    progress_intervals = np.diff(np.asarray([0.0, *completion_times]))
    estimated_concurrent_cuda_memory = float(sum(worker_reserved_memory.values()))
    resource_stable = estimated_concurrent_cuda_memory <= 7_680.0
    return {
        "protocol_version": PROTOCOL_VERSION,
        "num_workers": num_workers,
        "conditions": len(conditions),
        "targets": len(targets),
        "controllers": len(CONTINUOUS_CONTROLLER_IDS),
        "episodes": episodes,
        "decisions": decisions,
        "elapsed_seconds": elapsed,
        "episodes_per_second": episodes / elapsed if elapsed else 0.0,
        "decisions_per_second": decisions / elapsed if elapsed else 0.0,
        "invalid_episodes": invalid,
        "invalid_episode_rate": invalid / episodes if episodes else 1.0,
        "runtime_errors": errors,
        "longest_no_progress_interval_s": (
            float(np.max(progress_intervals)) if len(progress_intervals) else elapsed
        ),
        "worker_peak_cuda_memory_mib": worker_memory,
        "worker_peak_cuda_reserved_mib": worker_reserved_memory,
        "estimated_concurrent_cuda_memory_mib": estimated_concurrent_cuda_memory,
        "resource_stable": resource_stable,
        "eligible_for_selection": (
            not errors
            and len(results) == len(conditions)
            and invalid / episodes <= 0.01
            and resource_stable
        ),
    }


def benchmark_workers(args: argparse.Namespace) -> dict[str, Any]:
    """比较 2/4/6 workers 并选择稳定配置中吞吐量最高者。"""

    shortlist_budget = selected_shortlist_budget(allow_provisional=True)
    worker_counts = tuple(
        int(value.strip())
        for value in args.worker_counts.split(",")
        if value.strip()
    )
    summaries = []
    for num_workers in worker_counts:
        summary = run_worker_benchmark_configuration(
            num_workers,
            shortlist_budget,
            args.node_query_chunk_size,
        )
        summaries.append(summary)
        write_json(
            RESULTS_ROOT / "benchmark" / f"workers_{num_workers}_summary.json",
            summary,
        )
        print(
            f"workers={num_workers}: "
            f"episodes_per_second={summary['episodes_per_second']:.4f}"
        )
    eligible = [row for row in summaries if row["eligible_for_selection"]]
    if not eligible:
        raise RuntimeError("2/4/6 workers 均未形成稳定 benchmark 配置")
    selected = max(eligible, key=lambda row: row["episodes_per_second"])
    configuration = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_num_workers": selected["num_workers"],
        "selection_rule": (
            "无 runtime error、invalid rate <= 1%、估计并发 CUDA memory <= 7680 MiB "
            "的配置中选择 episodes per second 最高者"
        ),
        "shortlist_budget": shortlist_budget,
        "configurations": summaries,
    }
    write_json(
        RESULTS_ROOT / "benchmark" / "selected_configuration.json",
        configuration,
    )
    print(
        json.dumps(make_json_compatible(configuration), ensure_ascii=False, indent=2)
    )
    return configuration


def load_formal_step_rows(
    root: Path,
    scenario: str,
    role: str,
) -> list[dict[str, str]]:
    """按 hidden-condition manifest 顺序读取正式 condition shards。"""

    rows: list[dict[str, str]] = []
    for condition in hcr_e3.load_conditions(scenario, role):
        path = root / role / scenario / f"{condition['condition_id']}.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少 condition shard: {path}")
        rows.extend(read_csv_rows(path))
    return rows


def evenly_spaced_rows(
    rows: list[dict[str, str]], count: int
) -> list[dict[str, str]]:
    """从稳定排序后的 rows 中选择互异的等间隔 ranks。"""

    if len(rows) < count:
        raise RuntimeError(f"有效 decision states 不足: {len(rows)} < {count}")
    indices = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    return [rows[int(index)] for index in indices]


def select_shortlist_state_rows(
    scenario: str,
    state_source: str,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    """按正式规则从每个 Validation condition 选择 8 个 decision states。"""

    if state_source == "off_policy":
        all_rows = load_formal_step_rows(
            HCR_E5_DATA_ROOT, scenario, "validation"
        )
        eligible = [
            row
            for row in all_rows
            if row["target_group"] == "sequential_extension"
            and row["controller_id"] == DISCRETE_BASELINE_CONTROLLER_ID
            and int(row["episode_invalid"]) == 0
        ]
        sort_key = lambda row: (int(row["push_index"]), row["target_id"])
    else:
        all_rows = load_formal_step_rows(
            DATA_ROOT / "closed_loop", scenario, "validation"
        )
        eligible = [
            row
            for row in all_rows
            if row["target_group"] == "sequential_extension"
            and row["controller_id"]
            == BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
            and int(row["episode_invalid"]) == 0
        ]
        sort_key = lambda row: (row["target_id"], int(row["push_index"]))
    histories: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        histories[row["episode_key"]].append(row)
    for episode_rows in histories.values():
        episode_rows.sort(key=lambda row: int(row["push_index"]))

    selected: list[dict[str, str]] = []
    condition_ids = [
        row["condition_id"]
        for row in hcr_e3.load_conditions(scenario, "validation")
    ]
    for condition_id in condition_ids:
        condition_rows = sorted(
            [row for row in eligible if row["condition_id"] == condition_id],
            key=sort_key,
        )
        selected.extend(evenly_spaced_rows(condition_rows, 8))
    if len(selected) != 128:
        raise RuntimeError(f"shortlist cases 数量错误: {len(selected)} != 128")
    return selected, histories


def replay_belief_before_row(
    engine: ContinuousDecisionEngine,
    row: dict[str, str],
    episode_rows: list[dict[str, str]],
) -> E4BeliefState:
    """按原始 action-observation 顺序重放当前 decision 之前的 posterior。"""

    belief = engine.new_belief()
    true_hidden = np.asarray(
        [float(row[field]) for field in SCENARIO_ACTIVE_COORDINATES[engine.scenario]],
        dtype=np.float32,
    )
    current_push = int(row["push_index"])
    for history_row in episode_rows:
        if int(history_row["push_index"]) >= current_push:
            break
        if int(history_row["valid_observation"]) != 1:
            continue
        source_id = history_row.get("source_v2_action_id", "")
        if not source_id:
            source_id = history_row["v2_action_id"]
        source_index = engine.action_id_to_index[source_id]
        action_type = history_row.get("selected_action_type", "discrete")
        physical_text = history_row.get("continuous_physical_action_id", "")
        physical_id = int(physical_text) if physical_text else None
        observation = np.asarray(
            [
                float(history_row["observation_local_delta_x_m"]),
                float(history_row["observation_local_delta_y_m"]),
                float(history_row["observation_delta_yaw_rad"]),
            ],
            dtype=np.float64,
        )
        engine.update_action_observation(
            belief,
            action_type,
            source_index,
            physical_id,
            observation,
            true_hidden,
        )
    return belief


def shortlist_budget_summary(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总一个 shortlist budget 的 containment 与 expected-cost gap。"""

    gaps = np.asarray(
        [float(row["shortlist_expected_cost_gap"]) for row in case_rows],
        dtype=np.float64,
    )
    containment = np.asarray(
        [int(row["exact_top1_contained"]) for row in case_rows],
        dtype=np.float64,
    )
    mean_gap = float(np.mean(gaps))
    p90_gap = float(np.quantile(gaps, 0.90))
    return {
        "cases": len(case_rows),
        "exact_top1_containment_rate": float(np.mean(containment)),
        "shortlist_expected_cost_gap": descriptive(gaps),
        "screening_latency_s": descriptive(
            np.asarray(
                [float(row["screening_latency_s"]) for row in case_rows]
            )
        ),
        "full_set_exact_marginalisation_latency_s": descriptive(
            np.asarray(
                [
                    float(row["full_set_exact_marginalisation_latency_s"])
                    for row in case_rows
                ]
            )
        ),
        "peak_cuda_memory_mib": descriptive(
            np.asarray(
                [float(row["peak_cuda_memory_mib"]) for row in case_rows]
            )
        ),
        "mean_gap_requirement_passed": mean_gap <= 0.02 + 1e-12,
        "p90_gap_requirement_passed": p90_gap <= 0.05 + 1e-12,
        "passed": mean_gap <= 0.02 + 1e-12 and p90_gap <= 0.05 + 1e-12,
    }


def select_common_shortlist_budget(
    source_summaries: dict[str, dict[str, Any]],
) -> int | None:
    """选择对全部 state sources 和 scenarios 都通过的最小预算。"""

    for budget in SHORTLIST_BUDGETS:
        key = str(budget)
        if all(
            bool(scenario_summary["budgets"][key]["passed"])
            for source_summary in source_summaries.values()
            for scenario_summary in source_summary.values()
        ):
            return budget
    return None


def evaluate_shortlist(args: argparse.Namespace) -> dict[str, Any]:
    """评价 off-policy 或 on-policy posterior-mean screening shortlist。"""

    selected_scenarios = select_scenarios(args.scenario)
    if selected_scenarios != SCENARIOS:
        raise ValueError("shortlist budget 必须使用三场景共同选择，请使用 --scenario all")
    if args.state_source == "on_policy":
        selected_shortlist_budget(allow_provisional=True)
    device = car_e2.require_cuda()
    scenario_summaries: dict[str, Any] = {}
    for scenario in selected_scenarios:
        engine = ContinuousDecisionEngine(
            scenario,
            device,
            shortlist_budget=max(SHORTLIST_BUDGETS),
            node_query_chunk_size=args.node_query_chunk_size,
        )
        selected_rows, histories = select_shortlist_state_rows(
            scenario, args.state_source
        )
        case_rows: list[dict[str, Any]] = []
        for case_index, row in enumerate(selected_rows):
            belief = replay_belief_before_row(
                engine, row, histories[row["episode_key"]]
            )
            true_hidden = np.asarray(
                [
                    float(row[field])
                    for field in SCENARIO_ACTIVE_COORDINATES[scenario]
                ],
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
            diagnostics = engine.shortlist_diagnostic(query, belief, true_hidden)
            for budget, values in diagnostics.items():
                case_rows.append(
                    {
                        "scenario": scenario,
                        "state_source": args.state_source,
                        "case_index": case_index,
                        "condition_id": row["condition_id"],
                        "target_id": row["target_id"],
                        "target_stratum": row["target_stratum"],
                        "push_index": int(row["push_index"]),
                        "valid_update_count": belief.update_count,
                        "shortlist_budget": budget,
                        **values,
                    }
                )
        budgets = {
            str(budget): shortlist_budget_summary(
                [row for row in case_rows if row["shortlist_budget"] == budget]
            )
            for budget in SHORTLIST_BUDGETS
        }
        scenario_summary = {
            "scenario": scenario,
            "state_source": args.state_source,
            "decision_states": len(selected_rows),
            "budgets": budgets,
        }
        scenario_summaries[scenario] = scenario_summary
        output_dir = (
            RESULTS_ROOT
            / "shortlist_validation"
            / args.state_source
            / scenario
        )
        write_json(output_dir / "summary.json", scenario_summary)
        write_csv(
            output_dir / "cases.csv",
            case_rows,
            [
                "scenario",
                "state_source",
                "case_index",
                "condition_id",
                "target_id",
                "target_stratum",
                "push_index",
                "valid_update_count",
                "shortlist_budget",
                "full_best_physical_action_id",
                "shortlist_best_physical_action_id",
                "exact_top1_contained",
                "shortlist_expected_cost_gap",
                "screening_latency_s",
                "full_set_exact_marginalisation_latency_s",
                "peak_cuda_memory_mib",
            ],
        )
        print(f"evaluated {args.state_source} {scenario} shortlist")

    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "state_source": args.state_source,
        "scenarios": scenario_summaries,
    }
    combined_path = (
        RESULTS_ROOT
        / "shortlist_validation"
        / args.state_source
        / "combined_summary.json"
    )
    write_json(combined_path, combined)

    selected_path = (
        RESULTS_ROOT / "shortlist_validation" / "selected_configuration.json"
    )
    previous: dict[str, Any] = {}
    if selected_path.exists():
        with selected_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
    if args.state_source == "off_policy":
        provisional = select_common_shortlist_budget(
            {"off_policy": scenario_summaries}
        )
        if provisional is None:
            raise RuntimeError("256 shortlist 未满足 off-policy gap criteria")
        configuration = {
            "protocol_version": PROTOCOL_VERSION,
            "provisional_shortlist_budget": provisional,
            "final_shortlist_budget": None,
            "validation_rerun_required": False,
            "state_sources_evaluated": ["off_policy"],
        }
    else:
        off_policy_path = (
            RESULTS_ROOT
            / "shortlist_validation"
            / "off_policy"
            / "combined_summary.json"
        )
        if not off_policy_path.exists():
            raise FileNotFoundError("请先完成 off-policy shortlist Validation")
        with off_policy_path.open("r", encoding="utf-8") as handle:
            off_policy = json.load(handle)["scenarios"]
        final_budget = select_common_shortlist_budget(
            {
                "off_policy": off_policy,
                "on_policy": scenario_summaries,
            }
        )
        if final_budget is None:
            raise RuntimeError("256 shortlist 未同时满足 off-policy/on-policy criteria")
        provisional = int(previous["provisional_shortlist_budget"])
        observed_budgets = {
            int(row["shortlist_budget"])
            for scenario in SCENARIOS
            for row in load_formal_step_rows(
                DATA_ROOT / "closed_loop", scenario, "validation"
            )
            if row["controller_id"]
            == BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
        }
        configuration = {
            "protocol_version": PROTOCOL_VERSION,
            "provisional_shortlist_budget": provisional,
            "final_shortlist_budget": final_budget,
            "validation_rerun_required": observed_budgets != {final_budget},
            "validation_observed_shortlist_budgets": sorted(observed_budgets),
            "state_sources_evaluated": ["off_policy", "on_policy"],
        }
    write_json(selected_path, configuration)
    combined["selected_configuration"] = configuration
    print(json.dumps(make_json_compatible(combined), ensure_ascii=False, indent=2))
    return combined


EVALUATION_CONTROLLER_NAMES = {
    DISCRETE_BASELINE_CONTROLLER_ID: (
        "Discrete Belief-Marginalised Closed-Loop Baseline"
    ),
    **CONTINUOUS_CONTROLLER_NAMES,
    DISCRETE_FULL_INFORMATION_CONTROLLER_ID: (
        "Discrete Full-Information Tensor-Interpolation State-Feedback Diagnostic"
    ),
}
EVALUATION_CONTROLLER_IDS = tuple(EVALUATION_CONTROLLER_NAMES)


def load_evaluation_rows(
    scenario: str, role: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """读取 E4 continuous trajectories 与 E5 reused discrete baselines。"""

    e4_steps = load_formal_step_rows(
        DATA_ROOT / "closed_loop", scenario, role
    )
    e4_steps = [
        row
        for row in e4_steps
        if row["target_group"] == "sequential_extension"
        and row["controller_id"] in CONTINUOUS_CONTROLLER_IDS
    ]
    e5_steps = load_formal_step_rows(HCR_E5_DATA_ROOT, scenario, role)
    e5_steps = [
        row
        for row in e5_steps
        if row["target_group"] == "sequential_extension"
        and row["controller_id"]
        in {
            DISCRETE_BASELINE_CONTROLLER_ID,
            DISCRETE_FULL_INFORMATION_CONTROLLER_ID,
        }
    ]
    steps = [*e4_steps, *e5_steps]
    terminal = [row for row in steps if int(row["is_terminal_push"]) == 1]
    conditions = hcr_e3.load_conditions(scenario, role)
    expected = len(conditions) * 64 * len(EVALUATION_CONTROLLER_IDS)
    if len(terminal) != expected:
        raise RuntimeError(
            f"{scenario}/{role} terminal rows 数量错误: {len(terminal)} != {expected}"
        )
    episode_keys = [
        (row["condition_id"], row["target_id"], row["controller_id"])
        for row in terminal
    ]
    if len(set(episode_keys)) != expected:
        raise RuntimeError(f"{scenario}/{role} paired episodes 不唯一")
    return steps, terminal


def numeric_values(
    rows: list[dict[str, str]], field: str
) -> np.ndarray:
    """读取非空且有限的浮点字段。"""

    values = []
    for row in rows:
        text = row.get(field, "")
        if text in {"", None}:
            continue
        value = float(text)
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def e4_controller_summary(
    terminal_rows: list[dict[str, str]],
    step_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """汇总一条 controller route 的 episode、trajectory 与 runtime 指标。"""

    success = np.asarray(
        [int(row["episode_success"]) for row in terminal_rows],
        dtype=np.float64,
    )
    terminal_pushes = np.asarray(
        [int(row["terminal_push_count"]) for row in terminal_rows],
        dtype=np.int64,
    )
    success_curve = [
        float(np.mean(success * (terminal_pushes <= push_index)))
        for push_index in range(1, MAXIMUM_PUSHES + 1)
    ]
    successful_pushes = terminal_pushes[success.astype(bool)]
    invalid = np.asarray(
        [int(row["episode_invalid"]) for row in terminal_rows],
        dtype=np.float64,
    )
    continuous_steps = [
        row for row in step_rows if row.get("selected_action_type") == "continuous"
    ]
    episodes_with_continuous = {
        row["episode_key"] for row in continuous_steps
    }
    continuous_per_episode = Counter(
        row["episode_key"] for row in continuous_steps
    )
    continuous_counts = np.asarray(
        [continuous_per_episode.get(row["episode_key"], 0) for row in terminal_rows],
        dtype=np.float64,
    )
    invalid_reasons = Counter()
    for row in terminal_rows:
        if int(row["episode_invalid"]) != 1:
            continue
        for field, name in (
            ("simulation_unstable", "simulation_unstable"),
            ("quality_pass", "quality_failure"),
            ("contact_success", "contact_failure"),
            ("stopped_by_threshold", "settling_failure"),
        ):
            value = int(row[field])
            if (field == "simulation_unstable" and value == 1) or (
                field != "simulation_unstable" and value == 0
            ):
                invalid_reasons[name] += 1
    runtime_fields = (
        "proposal_latency_s",
        "discrete_selection_latency_s",
        "continuous_screening_latency_s",
        "continuous_scoring_latency_s",
        "belief_update_latency_s",
        "simulation_latency_s",
    )
    runtime = {
        field: descriptive(numeric_values(step_rows, field))
        for field in runtime_fields
        if any(field in row for row in step_rows)
    }
    decision_latency = []
    for row in step_rows:
        fields = (
            "proposal_latency_s",
            "discrete_selection_latency_s",
            "continuous_screening_latency_s",
            "continuous_scoring_latency_s",
        )
        if not any(field in row for field in fields):
            continue
        decision_latency.append(
            sum(float(row.get(field, 0.0) or 0.0) for field in fields)
        )
    runtime["total_decision_latency_s"] = descriptive(
        np.asarray(decision_latency, dtype=np.float64)
    )
    return {
        "episodes": len(terminal_rows),
        "episode_success_rate": float(np.mean(success)),
        "one_push_success_rate": float(
            np.mean(success * (terminal_pushes == 1))
        ),
        "success_by_push_curve": success_curve,
        "success_by_push_auc": float(np.mean(success_curve)),
        "attempted_pushes_to_termination": descriptive(terminal_pushes),
        "pushes_to_success": descriptive(successful_pushes),
        "final_position_error_m": descriptive(
            numeric_values(terminal_rows, "actual_position_error_m")
        ),
        "final_yaw_error_rad": descriptive(
            numeric_values(terminal_rows, "actual_yaw_error_rad")
        ),
        "final_tnpo_cost": descriptive(
            numeric_values(terminal_rows, "actual_tnpo_cost")
        ),
        "maximum_yaw_deviation_rad": descriptive(
            numeric_values(terminal_rows, "maximum_yaw_deviation_rad")
        ),
        "cumulative_absolute_yaw_excursion_rad": descriptive(
            numeric_values(
                terminal_rows, "cumulative_absolute_yaw_excursion_rad"
            )
        ),
        "invalid_episode_rate": float(np.mean(invalid)),
        "invalid_reason_counts": dict(invalid_reasons),
        "continuous_promotion": {
            "continuous_action_selection_rate_per_decision": (
                len(continuous_steps) / len(step_rows) if step_rows else 0.0
            ),
            "episodes_with_at_least_one_continuous_action_rate": (
                len(episodes_with_continuous) / len(terminal_rows)
                if terminal_rows
                else 0.0
            ),
            "continuous_actions_per_episode": descriptive(continuous_counts),
            "predicted_promotion_margin": descriptive(
                numeric_values(continuous_steps, "predicted_promotion_margin")
            ),
            "source_anchor_rank": descriptive(
                numeric_values(continuous_steps, "source_anchor_rank")
            ),
        },
        "runtime_s": runtime,
    }


def paired_evaluation_matrices(
    terminal_rows: list[dict[str, str]],
) -> tuple[list[str], list[str], dict[str, np.ndarray], dict[str, str]]:
    """将五条 route 对齐为 condition×target paired matrices。"""

    condition_ids = sorted({row["condition_id"] for row in terminal_rows})
    target_ids = sorted({row["target_id"] for row in terminal_rows})
    condition_index = {value: index for index, value in enumerate(condition_ids)}
    target_index = {value: index for index, value in enumerate(target_ids)}
    shape = (len(condition_ids), len(target_ids))
    fields = (
        "success",
        "success_auc",
        "attempted_pushes",
        "final_cost",
        "maximum_yaw",
        "cumulative_yaw",
    )
    matrices = {
        f"{controller_id}|{field}": np.full(shape, np.nan, dtype=np.float64)
        for controller_id in EVALUATION_CONTROLLER_IDS
        for field in fields
    }
    target_strata: dict[str, str] = {}
    for row in terminal_rows:
        c = condition_index[row["condition_id"]]
        t = target_index[row["target_id"]]
        controller = row["controller_id"]
        success = float(row["episode_success"])
        pushes = int(row["terminal_push_count"])
        matrices[f"{controller}|success"][c, t] = success
        matrices[f"{controller}|success_auc"][c, t] = (
            (MAXIMUM_PUSHES + 1 - pushes) / MAXIMUM_PUSHES
            if success == 1.0
            else 0.0
        )
        matrices[f"{controller}|attempted_pushes"][c, t] = pushes
        matrices[f"{controller}|final_cost"][c, t] = float(
            row["actual_tnpo_cost"]
        )
        matrices[f"{controller}|maximum_yaw"][c, t] = float(
            row["maximum_yaw_deviation_rad"]
        )
        cumulative_text = row.get("cumulative_absolute_yaw_excursion_rad", "")
        matrices[f"{controller}|cumulative_yaw"][c, t] = (
            float(cumulative_text) if cumulative_text else np.nan
        )
        target_strata[row["target_id"]] = row["target_stratum"]
    required = [
        values
        for key, values in matrices.items()
        if not key.endswith("|cumulative_yaw")
    ]
    if any(np.isnan(values).any() for values in required):
        raise RuntimeError("E4 paired matrices 存在缺失 episodes")
    return condition_ids, target_ids, matrices, target_strata


def e4_effect_arrays(matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """构造 H1、H2、H3 与 full-information reference gap effects。"""

    bm = BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
    ce = CERTAINTY_EQUIVALENT_CONTINUOUS_CONTROLLER_ID
    cfi = FULL_INFORMATION_CONTINUOUS_CONTROLLER_ID
    dbm = DISCRETE_BASELINE_CONTROLLER_ID
    dfi = DISCRETE_FULL_INFORMATION_CONTROLLER_ID
    return {
        "h1_success_by_push_auc": (
            matrices[f"{bm}|success_auc"] - matrices[f"{dbm}|success_auc"]
        ),
        "h1_episode_success_rate": (
            matrices[f"{bm}|success"] - matrices[f"{dbm}|success"]
        ),
        "h2_success_by_push_auc": (
            matrices[f"{bm}|success_auc"] - matrices[f"{ce}|success_auc"]
        ),
        "h3_success_by_push_auc": (
            matrices[f"{cfi}|success_auc"] - matrices[f"{dfi}|success_auc"]
        ),
        "rq4_full_information_auc_gap": (
            matrices[f"{cfi}|success_auc"] - matrices[f"{bm}|success_auc"]
        ),
        "rq4_full_information_push_gap": (
            matrices[f"{bm}|attempted_pushes"]
            - matrices[f"{cfi}|attempted_pushes"]
        ),
        "rq4_full_information_final_cost_gap": (
            matrices[f"{bm}|final_cost"] - matrices[f"{cfi}|final_cost"]
        ),
        "rq4_full_information_maximum_yaw_gap": (
            matrices[f"{bm}|maximum_yaw"] - matrices[f"{cfi}|maximum_yaw"]
        ),
    }


def e4_paired_two_way_bootstrap(
    terminal_rows: list[dict[str, str]],
    resamples: int,
    scenario_index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """执行 seed 0 的 condition-target-stratified paired two-way bootstrap。"""

    condition_ids, target_ids, matrices, target_strata = (
        paired_evaluation_matrices(terminal_rows)
    )
    effects = e4_effect_arrays(matrices)
    stratum_indices = {
        stratum: np.asarray(
            [
                index
                for index, target_id in enumerate(target_ids)
                if target_strata[target_id] == stratum
            ],
            dtype=np.int64,
        )
        for stratum in sorted(set(target_strata.values()))
    }
    rng = np.random.default_rng(np.random.SeedSequence([DEFAULT_SEED, scenario_index]))
    samples = {
        name: np.empty(resamples, dtype=np.float64) for name in effects
    }
    curve_samples = {
        controller_id: np.empty(
            (resamples, MAXIMUM_PUSHES), dtype=np.float64
        )
        for controller_id in EVALUATION_CONTROLLER_IDS
    }
    condition_count = len(condition_ids)
    for sample_index in range(resamples):
        condition_counts = np.bincount(
            rng.integers(0, condition_count, size=condition_count),
            minlength=condition_count,
        ).astype(np.float64)
        target_counts = np.zeros(len(target_ids), dtype=np.float64)
        for indices in stratum_indices.values():
            local_draws = rng.integers(0, len(indices), size=len(indices))
            target_counts[indices] = np.bincount(
                local_draws, minlength=len(indices)
            )
        weights = condition_counts[:, None] * target_counts[None, :]
        denominator = float(weights.sum())
        for name, values in effects.items():
            samples[name][sample_index] = float(
                np.sum(weights * values) / denominator
            )
        for controller_id in EVALUATION_CONTROLLER_IDS:
            success = matrices[f"{controller_id}|success"]
            pushes = matrices[f"{controller_id}|attempted_pushes"]
            for push_index in range(1, MAXIMUM_PUSHES + 1):
                achieved = success * (pushes <= push_index)
                curve_samples[controller_id][sample_index, push_index - 1] = (
                    float(np.sum(weights * achieved) / denominator)
                )
    summary = {
        name: {
            "point_estimate": float(np.mean(values)),
            "ci_95_low": float(np.quantile(samples[name], 0.025)),
            "ci_95_high": float(np.quantile(samples[name], 0.975)),
            "positive_effect_probability": float(np.mean(samples[name] > 0.0)),
            "bootstrap_resamples": resamples,
            "bootstrap_master_seed": DEFAULT_SEED,
        }
        for name, values in effects.items()
    }
    curve_bands: dict[str, Any] = {}
    for controller_id in EVALUATION_CONTROLLER_IDS:
        success = matrices[f"{controller_id}|success"]
        pushes = matrices[f"{controller_id}|attempted_pushes"]
        point = [
            float(np.mean(success * (pushes <= push_index)))
            for push_index in range(1, MAXIMUM_PUSHES + 1)
        ]
        curve_bands[controller_id] = {
            "push_indices": list(range(1, MAXIMUM_PUSHES + 1)),
            "point_estimate": point,
            "ci_95_low": np.quantile(
                curve_samples[controller_id], 0.025, axis=0
            ).tolist(),
            "ci_95_high": np.quantile(
                curve_samples[controller_id], 0.975, axis=0
            ).tolist(),
            "bootstrap_resamples": resamples,
            "bootstrap_master_seed": DEFAULT_SEED,
        }
    return summary, samples, curve_bands


def executed_transfer_summary(
    step_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """汇总 Primary controller 实际执行 continuous pushes 的 transfer error。"""

    def method_error_summary(
        selected_rows: list[dict[str, str]], prefix: str
    ) -> dict[str, Any]:
        """计算一个 prediction backend 相对真实 local observation 的误差。"""

        if not selected_rows:
            return {
                "valid_pushes": 0,
                "target_normalized_prediction_error": descriptive(
                    np.asarray([], dtype=np.float64)
                ),
                "planar_prediction_error_mm": descriptive(
                    np.asarray([], dtype=np.float64)
                ),
                "yaw_prediction_error_deg": descriptive(
                    np.asarray([], dtype=np.float64)
                ),
                "signed_bias": None,
            }
        prediction = np.asarray(
            [
                [
                    float(row[f"{prefix}_predicted_delta_x_m"]),
                    float(row[f"{prefix}_predicted_delta_y_m"]),
                    float(row[f"{prefix}_predicted_delta_yaw_rad"]),
                ]
                for row in selected_rows
            ],
            dtype=np.float64,
        )
        observation = np.asarray(
            [
                [
                    float(row["observation_local_delta_x_m"]),
                    float(row["observation_local_delta_y_m"]),
                    float(row["observation_delta_yaw_rad"]),
                ]
                for row in selected_rows
            ],
            dtype=np.float64,
        )
        errors = prediction_error(prediction, observation)
        signed = errors["signed"]
        return {
            "valid_pushes": len(selected_rows),
            "target_normalized_prediction_error": descriptive(
                errors["target_normalized"]
            ),
            "planar_prediction_error_mm": descriptive(
                errors["planar_m"] * 1_000.0
            ),
            "yaw_prediction_error_deg": descriptive(
                np.degrees(errors["yaw_rad"])
            ),
            "signed_bias": {
                "delta_x_m": float(np.mean(signed[:, 0])),
                "delta_y_m": float(np.mean(signed[:, 1])),
                "delta_yaw_deg": float(math.degrees(np.mean(signed[:, 2]))),
            },
        }

    rows = [
        row
        for row in step_rows
        if row["controller_id"]
        == BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
        and row.get("selected_action_type") == "continuous"
        and int(row["valid_observation"]) == 1
    ]
    offset_values = numeric_values(rows, "normalized_offset_norm")
    largest_offset_threshold = (
        float(np.quantile(offset_values, 0.75)) if len(offset_values) else None
    )
    subsets = {
        "overall": rows,
        "high_yaw": [
            row
            for row in rows
            if abs(float(row["observation_delta_yaw_rad"]))
            >= math.radians(5.0)
        ],
        "largest_offset_quartile": [
            row
            for row in rows
            if largest_offset_threshold is not None
            and float(row["normalized_offset_norm"])
            >= largest_offset_threshold
        ],
    }
    summary: dict[str, Any] = {
        "largest_offset_quartile_threshold": largest_offset_threshold,
        "subsets": {},
    }
    for name, subset_rows in subsets.items():
        continuous_error = numeric_values(
            subset_rows, "executed_prediction_tnpo_error"
        )
        source_error = numeric_values(
            subset_rows, "source_anchor_prediction_tnpo_error"
        )
        passed = (
            len(continuous_error) > 0
            and len(continuous_error) == len(source_error)
            and float(np.mean(continuous_error))
            <= float(np.mean(source_error)) + 1e-12
        )
        summary["subsets"][name] = {
            "valid_continuous_pushes": len(subset_rows),
            "transfer_primary_2": method_error_summary(
                subset_rows, "true_condition"
            ),
            "source_anchor_baseline": method_error_summary(
                subset_rows, "source_anchor"
            ),
            "continuous_prediction_tnpo_error": descriptive(continuous_error),
            "source_anchor_prediction_tnpo_error": descriptive(source_error),
            "continuous_not_worse_than_source_anchor": passed,
        }
    summary["all_requirements_passed"] = all(
        values["continuous_not_worse_than_source_anchor"]
        for values in summary["subsets"].values()
    )
    summary["by_push_index"] = {
        str(push_index): {
            "transfer_primary_2": method_error_summary(
                [row for row in rows if int(row["push_index"]) == push_index],
                "true_condition",
            ),
            "source_anchor_baseline": method_error_summary(
                [row for row in rows if int(row["push_index"]) == push_index],
                "source_anchor",
            ),
        }
        for push_index in sorted({int(row["push_index"]) for row in rows})
    }
    phases = {
        "finite_online_update_phase": [
            row for row in rows if int(row["valid_update_count_pre"]) < 4
        ],
        "fixed_posterior_control_phase": [
            row for row in rows if int(row["valid_update_count_pre"]) >= 4
        ],
    }
    summary["by_posterior_update_phase"] = {
        phase: {
            "transfer_primary_2": method_error_summary(
                phase_rows, "true_condition"
            ),
            "source_anchor_baseline": method_error_summary(
                phase_rows, "source_anchor"
            ),
        }
        for phase, phase_rows in phases.items()
    }
    return summary


def on_policy_terminal_calibration(
    terminal_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """计算 Primary controller terminal 95% HPD empirical coverage。"""

    rows = [
        row
        for row in terminal_rows
        if row["controller_id"]
        == BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
        and row.get("posterior_hpd_covered_post", "") != ""
    ]
    values = np.asarray(
        [int(row["posterior_hpd_covered_post"]) for row in rows],
        dtype=np.float64,
    )
    coverage = float(np.mean(values)) if len(values) else None
    return {
        "episodes_with_defined_terminal_hpd": len(rows),
        "terminal_hpd_coverage": coverage,
        "requirement": "coverage >= 0.90",
        "passed": coverage is not None and coverage >= 0.90,
    }


def closed_loop_validation_gate(
    scenario_summaries: dict[str, Any],
    macro_effects: dict[str, Any],
) -> dict[str, Any]:
    """按定稿的六项 selection-level criteria 生成 Validation gate。"""

    shortlist_path = (
        RESULTS_ROOT / "shortlist_validation" / "selected_configuration.json"
    )
    shortlist_configuration: dict[str, Any] = {}
    if shortlist_path.exists():
        with shortlist_path.open("r", encoding="utf-8") as handle:
            shortlist_configuration = json.load(handle)
    h1_auc = macro_effects["h1_success_by_push_auc"]
    h1_success = macro_effects["h1_episode_success_rate"]
    criterion_1 = h1_auc["ci_95_low"] > 0.0
    criterion_2 = h1_success["ci_95_low"] >= -0.02
    criterion_3 = all(
        summary["controllers"][
            BELIEF_MARGINALISED_CONTINUOUS_CONTROLLER_ID
        ]["continuous_promotion"]["continuous_action_selection_rate_per_decision"]
        > 0.0
        for summary in scenario_summaries.values()
    )
    criterion_4 = all(
        controller_summary["invalid_episode_rate"] <= 0.01
        for summary in scenario_summaries.values()
        for controller_id, controller_summary in summary["controllers"].items()
        if controller_id in CONTINUOUS_CONTROLLER_IDS
    )
    criterion_5 = all(
        summary["executed_continuous_transfer_diagnostic"][
            "all_requirements_passed"
        ]
        for summary in scenario_summaries.values()
    )
    criterion_6 = all(
        summary["on_policy_terminal_calibration"]["passed"]
        for summary in scenario_summaries.values()
    )
    shortlist_passed = (
        shortlist_configuration.get("final_shortlist_budget") is not None
        and not shortlist_configuration.get("validation_rerun_required", True)
        and set(shortlist_configuration.get("state_sources_evaluated", []))
        == {"off_policy", "on_policy"}
    )
    criteria = {
        "macro_auc_ci_lower_greater_than_zero": criterion_1,
        "macro_success_difference_ci_lower_at_least_minus_0_02": criterion_2,
        "continuous_selection_nonzero_in_all_scenarios": criterion_3,
        "invalid_episode_rate_at_most_0_01": criterion_4,
        "executed_continuous_transfer_not_worse_than_source_anchor": criterion_5,
        "on_policy_terminal_hpd_coverage_at_least_0_90": criterion_6,
        "off_policy_and_on_policy_shortlist_finalised": shortlist_passed,
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "shortlist_configuration": shortlist_configuration,
    }


def evaluate_closed_loop(args: argparse.Namespace) -> dict[str, Any]:
    """评价 E4 Validation 或 predefined shared Test closed-loop results。"""

    selected_scenarios = select_scenarios(args.scenario)
    if selected_scenarios != SCENARIOS:
        raise ValueError("正式 E4 评价必须使用 --scenario all")
    scenario_summaries: dict[str, Any] = {}
    scenario_samples: dict[str, dict[str, np.ndarray]] = {}
    for scenario_index, scenario in enumerate(selected_scenarios):
        step_rows, terminal_rows = load_evaluation_rows(scenario, args.role)
        controllers = {
            controller_id: e4_controller_summary(
                [
                    row
                    for row in terminal_rows
                    if row["controller_id"] == controller_id
                ],
                [
                    row
                    for row in step_rows
                    if row["controller_id"] == controller_id
                ],
            )
            for controller_id in EVALUATION_CONTROLLER_IDS
        }
        effects, samples, curve_bands = e4_paired_two_way_bootstrap(
            terminal_rows,
            args.bootstrap_resamples,
            scenario_index,
        )
        summary = {
            "scenario": scenario,
            "role": args.role,
            "controllers": controllers,
            "paired_effects": effects,
            "success_by_push_confidence_bands": curve_bands,
            "executed_continuous_transfer_diagnostic": executed_transfer_summary(
                step_rows
            ),
            "on_policy_terminal_calibration": on_policy_terminal_calibration(
                terminal_rows
            ),
        }
        scenario_summaries[scenario] = summary
        scenario_samples[scenario] = samples
        output_dir = RESULTS_ROOT / "evaluation" / args.role / scenario
        write_json(output_dir / "summary.json", summary)
        write_csv(
            output_dir / "paired_effects.csv",
            [
                {"effect": name, **values}
                for name, values in effects.items()
            ],
            [
                "effect",
                "point_estimate",
                "ci_95_low",
                "ci_95_high",
                "positive_effect_probability",
                "bootstrap_resamples",
                "bootstrap_master_seed",
            ],
        )
        curve_rows = []
        for controller_id, values in curve_bands.items():
            for index, push_index in enumerate(values["push_indices"]):
                curve_rows.append(
                    {
                        "controller_id": controller_id,
                        "push_index": push_index,
                        "point_estimate": values["point_estimate"][index],
                        "ci_95_low": values["ci_95_low"][index],
                        "ci_95_high": values["ci_95_high"][index],
                        "bootstrap_resamples": values["bootstrap_resamples"],
                        "bootstrap_master_seed": values[
                            "bootstrap_master_seed"
                        ],
                    }
                )
        write_csv(
            output_dir / "success_by_push_confidence_bands.csv",
            curve_rows,
            [
                "controller_id",
                "push_index",
                "point_estimate",
                "ci_95_low",
                "ci_95_high",
                "bootstrap_resamples",
                "bootstrap_master_seed",
            ],
        )
        print(f"evaluated {scenario} {args.role}")

    effect_names = next(iter(scenario_samples.values())).keys()
    macro_effects: dict[str, Any] = {}
    for name in effect_names:
        macro_samples = np.mean(
            np.stack(
                [scenario_samples[scenario][name] for scenario in SCENARIOS],
                axis=0,
            ),
            axis=0,
        )
        macro_effects[name] = {
            "point_estimate": float(
                np.mean(
                    [
                        scenario_summaries[scenario]["paired_effects"][name][
                            "point_estimate"
                        ]
                        for scenario in SCENARIOS
                    ]
                )
            ),
            "ci_95_low": float(np.quantile(macro_samples, 0.025)),
            "ci_95_high": float(np.quantile(macro_samples, 0.975)),
            "positive_effect_probability": float(np.mean(macro_samples > 0.0)),
            "equal_weight_scenarios": list(SCENARIOS),
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_master_seed": DEFAULT_SEED,
        }
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "role": args.role,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_master_seed": DEFAULT_SEED,
        "scenarios": scenario_summaries,
        "equal_weight_macro_paired_effects": macro_effects,
        "hypothesis_support": {
            "h1_supported": (
                macro_effects["h1_success_by_push_auc"]["ci_95_low"] > 0.0
                and macro_effects["h1_episode_success_rate"]["ci_95_low"]
                >= -0.02
            ),
            "h2_supported": macro_effects["h2_success_by_push_auc"][
                "ci_95_low"
            ]
            > 0.0,
            "h3_supported": macro_effects["h3_success_by_push_auc"][
                "ci_95_low"
            ]
            > 0.0,
        },
    }
    output_dir = RESULTS_ROOT / "evaluation" / args.role
    if args.role == "validation":
        gate = closed_loop_validation_gate(scenario_summaries, macro_effects)
        combined["validation_gate"] = gate
        write_json(output_dir / "gate_decision.json", gate)
    output_path = output_dir / "combined_summary.json"
    write_json(output_path, combined)
    print(f"Combined summary: {output_path.resolve()}")
    return combined


def build_parser() -> argparse.ArgumentParser:
    """构建 CAR Experiment 4 的统一命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "prepare",
        help="生成 continuous candidate/residual library 和 likelihood split",
    )

    likelihood_parser = subparsers.add_parser(
        "collect-likelihood",
        help="采集 continuous-action likelihood outcomes",
    )
    likelihood_parser.add_argument(
        "--scenario", choices=(*SCENARIOS, "all"), default="all"
    )
    likelihood_parser.add_argument(
        "--role", choices=("training", "validation"), required=True
    )
    likelihood_parser.add_argument("--num-workers", type=int, default=8)
    likelihood_parser.add_argument("--max-conditions", type=int, default=0)
    likelihood_parser.add_argument("--resume", action="store_true")

    fit_parser = subparsers.add_parser(
        "fit-likelihood",
        help="拟合 continuous-action residual statistics",
    )
    fit_parser.add_argument(
        "--scenario", choices=(*SCENARIOS, "all"), default="all"
    )

    transfer_parser = subparsers.add_parser(
        "evaluate-transfer-calibration",
        help="评价 cross-condition transfer 并校准 Student-t likelihood",
    )
    transfer_parser.add_argument(
        "--scenario", choices=(*SCENARIOS, "all"), default="all"
    )
    transfer_parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )

    shortlist_parser = subparsers.add_parser(
        "evaluate-shortlist",
        help="评价 off-policy 或 on-policy continuous shortlist",
    )
    shortlist_parser.add_argument(
        "--scenario", choices=(*SCENARIOS, "all"), default="all"
    )
    shortlist_parser.add_argument(
        "--state-source",
        choices=("off-policy", "on-policy"),
        required=True,
    )
    shortlist_parser.add_argument(
        "--node-query-chunk-size", type=int, default=NODE_QUERY_CHUNK_SIZE
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark-workers",
        help="比较 Joint Validation 的 2/4/6 concurrent workers",
    )
    benchmark_parser.add_argument("--worker-counts", default="2,4,6")
    benchmark_parser.add_argument(
        "--node-query-chunk-size", type=int, default=NODE_QUERY_CHUNK_SIZE
    )

    collect_parser = subparsers.add_parser(
        "collect",
        help="采集 Validation 或 predefined shared Test closed-loop trajectories",
    )
    collect_parser.add_argument(
        "--scenario", choices=(*SCENARIOS, "all"), default="all"
    )
    collect_parser.add_argument(
        "--role", choices=("validation", "test"), required=True
    )
    collect_parser.add_argument("--controllers", default="all")
    collect_parser.add_argument("--maximum-pushes", type=int, default=MAXIMUM_PUSHES)
    collect_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="0 表示使用 benchmark 选择的 worker count",
    )
    collect_parser.add_argument("--max-conditions", type=int, default=0)
    collect_parser.add_argument("--max-targets", type=int, default=0)
    collect_parser.add_argument(
        "--node-query-chunk-size", type=int, default=NODE_QUERY_CHUNK_SIZE
    )
    collect_parser.add_argument("--resume", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="评价 closed-loop Validation gate 或 Independent Test",
    )
    evaluate_parser.add_argument(
        "--scenario", choices=(*SCENARIOS, "all"), default="all"
    )
    evaluate_parser.add_argument(
        "--role", choices=("validation", "test"), required=True
    )
    evaluate_parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """检查直接影响正式计算规模的命令行参数。"""

    for name in (
        "num_workers",
        "max_conditions",
        "max_targets",
    ):
        if hasattr(args, name) and int(getattr(args, name)) < 0:
            raise ValueError(f"{name.replace('_', '-')} 不能小于 0")
    for name in (
        "maximum_pushes",
        "node_query_chunk_size",
        "bootstrap_resamples",
    ):
        if hasattr(args, name) and int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} 必须大于 0")
    if hasattr(args, "worker_counts"):
        values = [
            int(value.strip())
            for value in args.worker_counts.split(",")
            if value.strip()
        ]
        if not values or any(value <= 0 for value in values):
            raise ValueError("worker-counts 必须是正整数列表")


def main() -> None:
    """执行 CAR Experiment 4 子命令。"""

    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(args)
    if hasattr(args, "state_source"):
        args.state_source = args.state_source.replace("-", "_")
    print("CAR = Continuous Action Refinement")
    print("Experiment 4 = Belief-Space Continuous Action Refinement")
    if args.command == "prepare":
        prepare_experiment()
    elif args.command == "collect-likelihood":
        collect_likelihood(args)
    elif args.command == "fit-likelihood":
        fit_likelihood(args)
    elif args.command == "evaluate-transfer-calibration":
        evaluate_transfer_calibration(args)
    elif args.command == "evaluate-shortlist":
        evaluate_shortlist(args)
    elif args.command == "benchmark-workers":
        benchmark_workers(args)
    elif args.command == "collect":
        collect_closed_loop(args)
    elif args.command == "evaluate":
        evaluate_closed_loop(args)


if __name__ == "__main__":
    main()
