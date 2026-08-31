"""HCR V2 E3 的 GPU-first 训练、评估与 benchmark 入口。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from push_core.hcr_v2.e1 import OUTCOME_FIELDS, SCENARIO_ACTIVE_COORDINATES, read_csv_rows
from push_core.hcr_v2.e3 import (
    ACTION_COUNT,
    ACTIONS_PER_REGION,
    LABEL_TEMPERATURE,
    REGION_COUNT,
    TASK_DIMENSION,
    TaskNormaliser,
    TwoStageProposal,
    action_soft_labels,
    belief_marginalised_probabilities,
    interpolate_outcome_grid,
    make_task_queries,
    model_parameter_count,
    posterior_weights_from_histories,
    region_soft_labels_from_costs,
    soft_cross_entropy,
    tnpo_costs,
    topk_probabilities,
    wrap_to_pi,
)
from push_core.project_paths import HCR_V2_DATA_DIR, HCR_V2_RESULTS_DIR


PROTOCOL_VERSION = "hcr_v2_e3_v1"
FRICTION_CONE = "elliptic"
SCENARIOS = ("friction", "com", "joint")
TRAINING_SEED = 0
MAXIMUM_TRAINING_EPOCHS = 600
LEARNING_RATE_REDUCTION_FACTOR = 0.3
LEARNING_RATE_SCHEDULER_PATIENCE = 10
MINIMUM_LEARNING_RATE = 9e-5
YAW_GRID_DEG = (-90.0, -45.0, -20.0, -5.0, 0.0, 5.0, 20.0, 45.0, 90.0)
PRIMARY_K = 100
K_VALUES = (20, 50, 100, 200)
TOPK_SAVE_COUNT = 200
POINTS_PER_DIMENSION = 17
STUDENT_T_DEGREES_OF_FREEDOM = 3.0
BELIEF_UPDATE_HORIZON = 4
NUMERICAL_JITTER = 1e-6
OBSERVATION_SCALE = np.asarray(
    [0.010, 0.010, math.radians(5.0)], dtype=np.float64
)
COVARIANCE_INFLATION = {"friction": 64.0, "com": 1.0, "joint": 4.0}
GAUSS_LEGENDRE_17_POINTS = np.asarray(
    [
        -0.9905754753144174,
        -0.9506755217687678,
        -0.8802391537269859,
        -0.7815140038968014,
        -0.6576711592166908,
        -0.5126905370864769,
        -0.3512317634538763,
        -0.17848418149584785,
        0.0,
        0.17848418149584785,
        0.3512317634538763,
        0.5126905370864769,
        0.6576711592166908,
        0.7815140038968014,
        0.8802391537269859,
        0.9506755217687678,
        0.9905754753144174,
    ],
    dtype=np.float64,
)
GAUSS_LEGENDRE_17_PRIOR_WEIGHTS = np.asarray(
    [
        0.01207415143427476,
        0.0277297646869933,
        0.04251807415858954,
        0.055941923596701824,
        0.06756818423426261,
        0.07702288053840506,
        0.08400205107822498,
        0.08828135268349627,
        0.08972323517810327,
        0.08828135268349627,
        0.08400205107822498,
        0.07702288053840506,
        0.06756818423426261,
        0.055941923596701824,
        0.04251807415858954,
        0.0277297646869933,
        0.01207415143427476,
    ],
    dtype=np.float64,
)

MANIFEST_DIR = REPOSITORY_ROOT / "manifests" / "hcr_v2"
ACTION_MANIFEST_PATH = MANIFEST_DIR / "hcr_v2_action_core_manifest_v1.csv"
CONDITION_MANIFEST_PATH = MANIFEST_DIR / "hcr_v2_hidden_condition_manifest_v1.csv"
TARGET_MANIFEST_PATH = MANIFEST_DIR / "hcr_v2_core_target_manifest_v1.csv"
E1_OUTCOME_ROOT = HCR_V2_DATA_DIR / "e1" / "outcomes"
E1_P1_ROOT = HCR_V2_RESULTS_DIR / "e1" / "p1"
E2_HISTORY_ROOT = HCR_V2_DATA_DIR / "e2" / "histories"
E2_RESIDUAL_ROOT = HCR_V2_RESULTS_DIR / "e2" / "residual_statistics"
E3_DATA_ROOT = HCR_V2_DATA_DIR / "e3"
E3_RESULTS_ROOT = HCR_V2_RESULTS_DIR / "e3"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写出严格 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    """按固定字段顺序写出 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_scenarios(value: str) -> tuple[str, ...]:
    """解析单场景或 all。"""

    return SCENARIOS if value == "all" else (value,)


def require_cuda() -> torch.device:
    """E3 正式实现只接受 CUDA device。"""

    if not torch.cuda.is_available():
        raise RuntimeError("E3 GPU-first implementation 需要可用的 CUDA GPU")
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    """为 CUDA MLP 计算启用 BF16 autocast。"""

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_action_layout() -> tuple[list[str], list[int]]:
    """读取并确认 12×378 action layout。"""

    rows = read_csv_rows(ACTION_MANIFEST_PATH)
    region_ids = sorted({int(row["contact_region_id"]) for row in rows})
    if len(rows) != ACTION_COUNT or len(region_ids) != REGION_COUNT:
        raise RuntimeError("E3 action_core 不是 12×378 layout")
    region_to_index = {region_id: index for index, region_id in enumerate(region_ids)}
    ordered: list[str | None] = [None] * ACTION_COUNT
    for row in rows:
        region_index = region_to_index[int(row["contact_region_id"])]
        action_index = int(row["action_param_index"])
        ordered[region_index * ACTIONS_PER_REGION + action_index] = row["v2_action_id"]
    if any(action_id is None for action_id in ordered):
        raise RuntimeError("action_core region/action_param layout 不完整")
    return [str(action_id) for action_id in ordered], region_ids


def load_conditions(scenario: str, role: str) -> list[dict[str, str]]:
    """读取场景对应的 Training、Validation 或 Test conditions。"""

    rows = [
        row
        for row in read_csv_rows(CONDITION_MANIFEST_PATH)
        if row["scenario"] == scenario and row["condition_role"] == role
    ]
    if role == "validation":
        rows = [row for row in rows if row["validation_partition"] == "model_selection"]
    rows.sort(key=lambda row: int(row["condition_index_within_role"]))
    expected = {
        "training": {"friction": 5, "com": 25, "joint": 125},
        "validation": {"friction": 16, "com": 16, "joint": 16},
        "test": {"friction": 32, "com": 32, "joint": 32},
    }[role][scenario]
    if len(rows) != expected:
        raise RuntimeError(f"{scenario} {role} condition 数量错误: {len(rows)}")
    return rows


def load_target_positions(role: str) -> tuple[list[str], np.ndarray]:
    """读取 E3 Training 或 checkpoint Validation target positions。"""

    rows = [
        row
        for row in read_csv_rows(TARGET_MANIFEST_PATH)
        if row["split_role"] == role
    ]
    expected = 3268 if role == "training" else 408
    if len(rows) != expected:
        raise RuntimeError(f"{role} core target 数量错误: {len(rows)}")
    return (
        [row["v2_target_id"] for row in rows],
        np.asarray(
            [
                [float(row["target_delta_x_m"]), float(row["target_delta_y_m"])]
                for row in rows
            ],
            dtype=np.float32,
        ),
    )


def load_outcome_array(
    scenario: str,
    role: str,
    conditions: list[dict[str, str]],
    action_ids: list[str],
) -> np.ndarray:
    """读取 E1 condition-action outcomes 并重排为 12×378 layout。"""

    action_index = {action_id: index for index, action_id in enumerate(action_ids)}
    outcomes = np.empty((len(conditions), ACTION_COUNT, 3), dtype=np.float32)
    for condition_index, condition in enumerate(conditions):
        path = E1_OUTCOME_ROOT / scenario / role / f"{condition['condition_id']}.csv"
        rows = read_csv_rows(path)
        if len(rows) != ACTION_COUNT:
            raise RuntimeError(f"E1 outcome shard action 数量错误: {path}")
        for row in rows:
            index = action_index[row["v2_action_id"]]
            outcomes[condition_index, index] = [
                float(row[field]) for field in OUTCOME_FIELDS
            ]
    return outcomes


def active_hidden_array(
    scenario: str,
    conditions: list[dict[str, str]],
) -> np.ndarray:
    """提取 active normalised hidden coordinates。"""

    fields = SCENARIO_ACTIVE_COORDINATES[scenario]
    return np.asarray(
        [[float(row[field]) for field in fields] for row in conditions],
        dtype=np.float32,
    )


def make_fixed_quadrature(scenario: str) -> tuple[np.ndarray, np.ndarray]:
    """使用 E2 已固定的 17 点 Gauss–Legendre tensor-product rule。"""

    dimension = len(SCENARIO_ACTIVE_COORDINATES[scenario])
    node_meshes = np.meshgrid(
        *([GAUSS_LEGENDRE_17_POINTS] * dimension), indexing="ij"
    )
    nodes = np.stack([mesh.reshape(-1) for mesh in node_meshes], axis=1)
    weight_meshes = np.meshgrid(
        *([GAUSS_LEGENDRE_17_PRIOR_WEIGHTS] * dimension), indexing="ij"
    )
    weights = np.ones_like(weight_meshes[0])
    for mesh in weight_meshes:
        weights *= mesh
    prior_weights = weights.reshape(-1)
    prior_weights /= prior_weights.sum()
    return nodes, prior_weights


def label_artifact_path(scenario: str, role: str) -> Path:
    """返回 compact label tensor artifact 路径。"""

    return E3_DATA_ROOT / "labels" / scenario / f"{role}.pt"


def label_metadata_path(scenario: str, role: str) -> Path:
    """返回 compact label metadata 路径。"""

    return E3_DATA_ROOT / "labels" / scenario / f"{role}_metadata.json"


@torch.inference_mode()
def prepare_label_role(
    scenario: str,
    role: str,
    query_chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """在 GPU 上生成一个场景和角色的 compact Region labels。"""

    action_ids, region_ids = load_action_layout()
    conditions = load_conditions(scenario, role)
    target_ids, positions = load_target_positions(role)
    outcomes_cpu = load_outcome_array(scenario, role, conditions, action_ids)
    hidden_cpu = active_hidden_array(scenario, conditions)
    yaw_grid = torch.deg2rad(
        torch.tensor(YAW_GRID_DEG, dtype=torch.float32, device=device)
    )
    tasks = make_task_queries(
        torch.from_numpy(positions).to(device), yaw_grid
    ).float()
    outcomes = torch.from_numpy(outcomes_cpu).to(device)
    labels = torch.empty(
        (len(conditions), tasks.shape[0], REGION_COUNT),
        dtype=torch.float32,
        device=device,
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for condition_index in range(len(conditions)):
        condition_outcomes = outcomes[condition_index]
        for start in range(0, tasks.shape[0], query_chunk_size):
            stop = min(tasks.shape[0], start + query_chunk_size)
            costs = tnpo_costs(condition_outcomes, tasks[start:stop])
            labels[condition_index, start:stop] = region_soft_labels_from_costs(costs)
        print(
            f"{scenario} {role}: prepared condition "
            f"{condition_index + 1}/{len(conditions)}"
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    artifact = {
        "task_queries": tasks.cpu(),
        "hidden_parameters": torch.from_numpy(hidden_cpu),
        "outcomes": torch.from_numpy(outcomes_cpu),
        "region_labels": labels.cpu(),
    }
    output_path = label_artifact_path(scenario, role)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "scenario": scenario,
        "role": role,
        "training_seed": TRAINING_SEED,
        "yaw_grid_deg": list(YAW_GRID_DEG),
        "target_count": len(target_ids),
        "task_query_count": int(tasks.shape[0]),
        "condition_count": len(conditions),
        "query_condition_case_count": int(len(conditions) * tasks.shape[0]),
        "region_count": REGION_COUNT,
        "actions_per_region": ACTIONS_PER_REGION,
        "label_temperature": LABEL_TEMPERATURE,
        "condition_ids": [row["condition_id"] for row in conditions],
        "region_ids": region_ids,
        "artifact_path": str(output_path.resolve()),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
    }
    write_json(label_metadata_path(scenario, role), metadata)
    return metadata


def prepare_training(args: argparse.Namespace) -> dict[str, Any]:
    """为 Training 和 checkpoint Validation 生成 compact label artifacts。"""

    device = require_cuda()
    summaries: dict[str, Any] = {}
    for scenario in select_scenarios(args.scenario):
        summaries[scenario] = {
            role: prepare_label_role(
                scenario, role, args.query_chunk_size, device
            )
            for role in ("training", "validation")
        }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "device": torch.cuda.get_device_name(device),
        "scenarios": summaries,
    }
    write_json(E3_DATA_ROOT / "labels" / "combined_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def load_label_artifact(scenario: str, role: str, device: torch.device) -> dict[str, torch.Tensor]:
    """读取一个 compact label artifact 并放入 GPU。"""

    path = label_artifact_path(scenario, role)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    return {key: value.to(device) for key, value in artifact.items()}


def make_normaliser(training_tasks: torch.Tensor) -> TaskNormaliser:
    """使用 Training query positions 建立固定 normaliser。"""

    mean = training_tasks[:, :2].mean(dim=0)
    std = training_tasks[:, :2].std(dim=0, unbiased=False).clamp_min(1e-6)
    return TaskNormaliser(mean, std)


def case_indices(
    flat_indices: torch.Tensor,
    query_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将 flat case indices 转换为 condition/query indices。"""

    return torch.div(flat_indices, query_count, rounding_mode="floor"), torch.remainder(
        flat_indices, query_count
    )


def train_region_pass(
    model: TwoStageProposal,
    optimizer: torch.optim.Optimizer,
    artifact: dict[str, torch.Tensor],
    normaliser: TaskNormaliser,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    """执行一个完整 Region Training pass。"""

    query_count = artifact["task_queries"].shape[0]
    case_count = artifact["hidden_parameters"].shape[0] * query_count
    order = torch.randperm(case_count, generator=generator, device="cuda")
    total_loss = torch.zeros((), dtype=torch.float32, device="cuda")
    total_cases = 0
    model.region_model.train()
    for start in range(0, case_count, batch_size):
        indices = order[start : start + batch_size]
        condition_index, query_index = case_indices(indices, query_count)
        tasks = normaliser.transform(artifact["task_queries"][query_index])
        hidden = artifact["hidden_parameters"][condition_index]
        labels = artifact["region_labels"][condition_index, query_index]
        inputs = model.proposal_inputs(tasks, hidden)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(inputs.device):
            logits = model.region_model(inputs)
            loss = soft_cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().float() * len(indices)
        total_cases += len(indices)
    return float(total_loss / total_cases)


def train_action_pass(
    model: TwoStageProposal,
    optimizer: torch.optim.Optimizer,
    artifact: dict[str, torch.Tensor],
    normaliser: TaskNormaliser,
    batch_size: int,
    epoch_index: int,
    generator: torch.Generator,
) -> float:
    """执行一个 importance-weighted balanced Action Training pass。"""

    query_count = artifact["task_queries"].shape[0]
    case_count = artifact["hidden_parameters"].shape[0] * query_count
    order = torch.randperm(case_count, generator=generator, device="cuda")
    numerator = torch.zeros((), dtype=torch.float32, device="cuda")
    denominator = torch.zeros((), dtype=torch.float32, device="cuda")
    model.action_model.train()
    for start in range(0, case_count, batch_size):
        indices = order[start : start + batch_size]
        condition_index, query_index = case_indices(indices, query_count)
        region_index = torch.remainder(indices + epoch_index, REGION_COUNT)
        raw_tasks = artifact["task_queries"][query_index]
        tasks = normaliser.transform(raw_tasks)
        hidden = artifact["hidden_parameters"][condition_index]
        outcomes = artifact["outcomes"].reshape(
            artifact["outcomes"].shape[0], REGION_COUNT, ACTIONS_PER_REGION, 3
        )[condition_index, region_index]
        labels = action_soft_labels(tnpo_costs(outcomes, raw_tasks))
        region_mass = artifact["region_labels"][
            condition_index, query_index, region_index
        ]
        importance_weights = REGION_COUNT * region_mass
        one_hot = torch.nn.functional.one_hot(
            region_index, num_classes=REGION_COUNT
        ).to(tasks.dtype)
        inputs = model.proposal_inputs(tasks, hidden)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(inputs.device):
            logits = model.action_model(inputs, one_hot)
            loss = soft_cross_entropy(logits, labels, importance_weights)
        loss.backward()
        optimizer.step()
        batch_weight = importance_weights.sum().detach()
        numerator += loss.detach().float() * batch_weight
        denominator += batch_weight
    return float(numerator / denominator.clamp_min(1e-12))


@torch.inference_mode()
def validation_losses(
    model: TwoStageProposal,
    artifact: dict[str, torch.Tensor],
    normaliser: TaskNormaliser,
    batch_size: int,
) -> tuple[float, float]:
    """使用固定 balanced-region sample 计算 checkpoint losses。"""

    model.eval()
    query_count = artifact["task_queries"].shape[0]
    case_count = artifact["hidden_parameters"].shape[0] * query_count
    region_numerator = 0.0
    action_numerator = 0.0
    action_denominator = 0.0
    outcomes_by_region = artifact["outcomes"].reshape(
        artifact["outcomes"].shape[0], REGION_COUNT, ACTIONS_PER_REGION, 3
    )
    for start in range(0, case_count, batch_size):
        indices = torch.arange(start, min(case_count, start + batch_size), device="cuda")
        condition_index, query_index = case_indices(indices, query_count)
        region_index = torch.remainder(indices, REGION_COUNT)
        raw_tasks = artifact["task_queries"][query_index]
        tasks = normaliser.transform(raw_tasks)
        hidden = artifact["hidden_parameters"][condition_index]
        inputs = model.proposal_inputs(tasks, hidden)
        region_labels = artifact["region_labels"][condition_index, query_index]
        with autocast_context(inputs.device):
            region_logits = model.region_model(inputs)
        region_losses = -(
            region_labels.float() * torch.log_softmax(region_logits.float(), dim=-1)
        ).sum(dim=-1)
        region_numerator += float(region_losses.sum())

        outcomes = outcomes_by_region[condition_index, region_index]
        action_labels = action_soft_labels(tnpo_costs(outcomes, raw_tasks))
        region_mass = region_labels.gather(1, region_index[:, None]).squeeze(1)
        importance_weights = REGION_COUNT * region_mass
        one_hot = torch.nn.functional.one_hot(
            region_index, num_classes=REGION_COUNT
        ).to(tasks.dtype)
        with autocast_context(inputs.device):
            action_logits = model.action_model(inputs, one_hot)
        action_losses = -(
            action_labels.float() * torch.log_softmax(action_logits.float(), dim=-1)
        ).sum(dim=-1)
        action_numerator += float((action_losses * importance_weights).sum())
        action_denominator += float(importance_weights.sum())
    return (
        region_numerator / case_count,
        action_numerator / max(action_denominator, 1e-12),
    )


def checkpoint_paths(scenario: str) -> tuple[Path, Path]:
    """返回 proposal checkpoint 与 normaliser 路径。"""

    root = E3_RESULTS_ROOT / "models" / scenario
    return root / "two_stage_proposal.pt", root / "normaliser.json"


def train_scenario(
    scenario: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """训练一个场景的 two-stage proposal。"""

    set_seed(args.seed)
    training = load_label_artifact(scenario, "training", device)
    validation = load_label_artifact(scenario, "validation", device)
    normaliser = make_normaliser(training["task_queries"])
    model = TwoStageProposal(training["hidden_parameters"].shape[1]).to(device)
    region_optimizer = torch.optim.AdamW(
        model.region_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    action_optimizer = torch.optim.AdamW(
        model.action_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    region_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        region_optimizer,
        mode="min",
        factor=LEARNING_RATE_REDUCTION_FACTOR,
        patience=LEARNING_RATE_SCHEDULER_PATIENCE,
        threshold=args.minimum_improvement,
        threshold_mode="abs",
        min_lr=MINIMUM_LEARNING_RATE,
    )
    action_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        action_optimizer,
        mode="min",
        factor=LEARNING_RATE_REDUCTION_FACTOR,
        patience=LEARNING_RATE_SCHEDULER_PATIENCE,
        threshold=args.minimum_improvement,
        threshold_mode="abs",
        min_lr=MINIMUM_LEARNING_RATE,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    best_region = math.inf
    best_action = math.inf
    best_region_state: dict[str, torch.Tensor] | None = None
    best_action_state: dict[str, torch.Tensor] | None = None
    best_region_epoch = 0
    best_action_epoch = 0
    stale_region = 0
    stale_action = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.max_epochs + 1):
        epoch_started = time.perf_counter()
        region_learning_rate = float(region_optimizer.param_groups[0]["lr"])
        action_learning_rate = float(action_optimizer.param_groups[0]["lr"])
        train_region_loss = train_region_pass(
            model, region_optimizer, training, normaliser, args.region_batch_size, generator
        )
        train_action_loss = train_action_pass(
            model,
            action_optimizer,
            training,
            normaliser,
            args.action_batch_size,
            epoch - 1,
            generator,
        )
        val_region_loss, val_action_loss = validation_losses(
            model, validation, normaliser, args.validation_batch_size
        )
        region_improved = val_region_loss < best_region - args.minimum_improvement
        action_improved = val_action_loss < best_action - args.minimum_improvement
        if region_improved:
            best_region = val_region_loss
            best_region_epoch = epoch
            best_region_state = {
                key: value.detach().cpu().clone()
                for key, value in model.region_model.state_dict().items()
            }
            stale_region = 0
        else:
            stale_region += 1
        if action_improved:
            best_action = val_action_loss
            best_action_epoch = epoch
            best_action_state = {
                key: value.detach().cpu().clone()
                for key, value in model.action_model.state_dict().items()
            }
            stale_action = 0
        else:
            stale_action += 1
        region_scheduler.step(val_region_loss)
        action_scheduler.step(val_action_loss)
        next_region_learning_rate = float(region_optimizer.param_groups[0]["lr"])
        next_action_learning_rate = float(action_optimizer.param_groups[0]["lr"])
        if next_region_learning_rate < region_learning_rate:
            stale_region = 0
        if next_action_learning_rate < action_learning_rate:
            stale_action = 0
        torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "train_region_cross_entropy": train_region_loss,
            "train_action_cross_entropy": train_action_loss,
            "validation_region_cross_entropy": val_region_loss,
            "validation_action_cross_entropy": val_action_loss,
            "best_validation_region_cross_entropy": best_region,
            "best_validation_action_cross_entropy": best_action,
            "stale_region_epochs": stale_region,
            "stale_action_epochs": stale_action,
            "region_learning_rate": region_learning_rate,
            "action_learning_rate": action_learning_rate,
            "next_region_learning_rate": next_region_learning_rate,
            "next_action_learning_rate": next_action_learning_rate,
            "seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"{scenario} epoch {epoch:03d}/{args.max_epochs}  "
            f"region={val_region_loss:.6f}  action={val_action_loss:.6f}  "
            f"lr={next_region_learning_rate:.1e}/{next_action_learning_rate:.1e}  "
            f"seconds={epoch_seconds:.2f}  "
            f"stale={stale_region}/{stale_action}"
        )
        if stale_region >= args.patience and stale_action >= args.patience:
            break

    if best_region_state is None or best_action_state is None:
        raise RuntimeError("Training 没有生成有效 checkpoint")
    model.region_model.load_state_dict(best_region_state)
    model.action_model.load_state_dict(best_action_state)
    checkpoint_path, normaliser_path = checkpoint_paths(scenario)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "scenario": scenario,
            "hidden_parameter_dimension": model.hidden_parameter_dimension,
            "training_seed": args.seed,
            "region_model_state_dict": best_region_state,
            "action_model_state_dict": best_action_state,
        },
        checkpoint_path,
    )
    write_json(
        normaliser_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "scenario": scenario,
            "position_mean": normaliser.position_mean.detach().cpu().tolist(),
            "position_std": normaliser.position_std.detach().cpu().tolist(),
        },
    )
    elapsed = time.perf_counter() - started
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "scenario": scenario,
        "device": torch.cuda.get_device_name(device),
        "autocast_dtype": "bfloat16",
        "float32_matmul_precision": "high",
        "training_seed": args.seed,
        "parameter_count": model_parameter_count([model]),
        "region_batch_size": args.region_batch_size,
        "action_batch_size": args.action_batch_size,
        "validation_batch_size": args.validation_batch_size,
        "initial_learning_rate": args.learning_rate,
        "learning_rate_scheduler": "ReduceLROnPlateau",
        "learning_rate_reduction_factor": LEARNING_RATE_REDUCTION_FACTOR,
        "learning_rate_scheduler_patience": LEARNING_RATE_SCHEDULER_PATIENCE,
        "minimum_learning_rate": MINIMUM_LEARNING_RATE,
        "weight_decay": args.weight_decay,
        "maximum_epochs": args.max_epochs,
        "completed_epochs": len(history),
        "patience": args.patience,
        "minimum_improvement": args.minimum_improvement,
        "best_region_epoch": best_region_epoch,
        "best_action_epoch": best_action_epoch,
        "best_validation_region_cross_entropy": best_region,
        "best_validation_action_cross_entropy": best_action,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "normaliser_path": str(normaliser_path.resolve()),
    }
    output_root = checkpoint_path.parent
    write_json(output_root / "training_summary.json", summary)
    write_json(output_root / "training_history.json", {"epochs": history})
    return summary


def train_proposal(args: argparse.Namespace) -> dict[str, Any]:
    """按单 GPU 任务顺序训练指定场景。"""

    device = require_cuda()
    summaries = {
        scenario: train_scenario(scenario, args, device)
        for scenario in select_scenarios(args.scenario)
    }
    combined = {"protocol_version": PROTOCOL_VERSION, "scenarios": summaries}
    write_json(E3_RESULTS_ROOT / "models" / "combined_training_summary.json", combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return combined


def load_history_episodes(scenario: str, role: str) -> list[list[dict[str, str]]]:
    """读取 E2 histories 并按 episode 分组。"""

    paths = sorted((E2_HISTORY_ROOT / role / scenario).glob("*.csv"))
    expected_conditions = 16 if role == "validation" else 32
    if len(paths) != expected_conditions:
        raise RuntimeError(f"{scenario} {role} history condition 数量错误")
    episodes: list[list[dict[str, str]]] = []
    for path in paths:
        current: list[dict[str, str]] = []
        current_key = ""
        for row in read_csv_rows(path):
            if current and row["episode_key"] != current_key:
                episodes.append(current)
                current = []
            current_key = row["episode_key"]
            current.append(row)
        if current:
            episodes.append(current)
    expected_episodes = 4096 if role == "validation" else 8192
    if len(episodes) != expected_episodes:
        raise RuntimeError(f"{scenario} {role} episode 数量错误: {len(episodes)}")
    return episodes


def build_decision_cases(
    scenario: str,
    role: str,
    action_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """按 min(5, L) 规则构造每个 eligible episode 的因果 decision case。"""

    episodes = load_history_episodes(scenario, role)
    action_index = {action_id: index for index, action_id in enumerate(action_ids)}
    active_fields = SCENARIO_ACTIVE_COORDINATES[scenario]
    metadata: list[dict[str, Any]] = []
    task_queries: list[list[float]] = []
    hidden_parameters: list[list[float]] = []
    action_histories: list[list[int]] = []
    observations: list[list[list[float]]] = []
    masks: list[list[float]] = []
    excluded_one_push = 0

    for episode in episodes:
        length = len(episode)
        if length < 2:
            excluded_one_push += 1
            continue
        decision_push = min(5, length)
        decision_row = episode[decision_push - 1]
        history_rows = episode[: decision_push - 1]
        pre_x = float(decision_row["pre_push_x_m"])
        pre_y = float(decision_row["pre_push_y_m"])
        pre_yaw = float(decision_row["pre_push_yaw_rad"])
        target_x = float(decision_row["target_world_x_m"])
        target_y = float(decision_row["target_world_y_m"])
        target_yaw = float(decision_row["target_world_yaw_rad"])
        delta_x = target_x - pre_x
        delta_y = target_y - pre_y
        cosine = math.cos(pre_yaw)
        sine = math.sin(pre_yaw)
        local_x = cosine * delta_x + sine * delta_y
        local_y = -sine * delta_x + cosine * delta_y
        yaw_correction = (target_yaw - pre_yaw + math.pi) % (2.0 * math.pi) - math.pi

        case_actions = [0] * BELIEF_UPDATE_HORIZON
        case_observations = [[0.0, 0.0, 0.0] for _ in range(BELIEF_UPDATE_HORIZON)]
        case_mask = [0.0] * BELIEF_UPDATE_HORIZON
        for update_index, row in enumerate(history_rows):
            case_actions[update_index] = action_index[row["v2_action_id"]]
            case_observations[update_index] = [
                float(row["observation_local_delta_x_m"]) / OBSERVATION_SCALE[0],
                float(row["observation_local_delta_y_m"]) / OBSERVATION_SCALE[1],
                float(row["observation_delta_yaw_rad"]) / OBSERVATION_SCALE[2],
            ]
            case_mask[update_index] = 1.0

        first = episode[0]
        metadata.append(
            {
                "case_index": len(metadata),
                "scenario": scenario,
                "role": role,
                "condition_id": first["condition_id"],
                "target_id": first["target_id"],
                "target_group": first["target_group"],
                "target_stratum": first["target_stratum"],
                "episode_key": first["episode_key"],
                "episode_push_count": length,
                "decision_push": decision_push,
                "belief_update_count": len(history_rows),
                "current_x_m": pre_x,
                "current_y_m": pre_y,
                "current_yaw_rad": pre_yaw,
                "remaining_local_x_m": local_x,
                "remaining_local_y_m": local_y,
                "desired_yaw_correction_rad": yaw_correction,
            }
        )
        task_queries.append(
            [local_x, local_y, math.sin(yaw_correction), math.cos(yaw_correction)]
        )
        hidden_parameters.append([float(first[field]) for field in active_fields])
        action_histories.append(case_actions)
        observations.append(case_observations)
        masks.append(case_mask)

    arrays = {
        "task_queries": np.asarray(task_queries, dtype=np.float32),
        "true_hidden_parameters": np.asarray(hidden_parameters, dtype=np.float32),
        "action_indices": np.asarray(action_histories, dtype=np.int64),
        "observations_standardised": np.asarray(observations, dtype=np.float32),
        "observation_mask": np.asarray(masks, dtype=np.float32),
        "excluded_one_push_count": np.asarray(excluded_one_push, dtype=np.int64),
    }
    return metadata, arrays


def load_p1_grid(
    scenario: str,
    action_ids: list[str],
    device: torch.device,
) -> torch.Tensor:
    """读取 P1 grid 并重排 action dimension。"""

    path = E1_P1_ROOT / scenario / "tensor_outcome_interpolator.npz"
    with np.load(path, allow_pickle=False) as payload:
        p1_action_ids = [str(value) for value in payload["action_ids"]]
        outcome_grid = np.asarray(payload["outcome_grid"], dtype=np.float32)
    index = {action_id: position for position, action_id in enumerate(p1_action_ids)}
    order = np.asarray([index[action_id] for action_id in action_ids], dtype=np.int64)
    return torch.from_numpy(outcome_grid[..., order, :]).to(device)


def load_residual_parameters(
    scenario: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 E2 residual bias 与最终 Student-t precision。"""

    path = E2_RESIDUAL_ROOT / scenario / "residual_statistics.npz"
    with np.load(path, allow_pickle=False) as payload:
        bias = np.asarray(payload["residual_bias"], dtype=np.float32)
        covariance = np.asarray(payload["base_covariance"], dtype=np.float32)
    covariance_tensor = torch.from_numpy(covariance).to(device)
    scale_matrix = (
        COVARIANCE_INFLATION[scenario] * covariance_tensor
        + NUMERICAL_JITTER * torch.eye(3, dtype=torch.float32, device=device)
    )
    precision = torch.linalg.inv(scale_matrix)
    return torch.from_numpy(bias).to(device), precision


def replay_case_posteriors(
    scenario: str,
    arrays: dict[str, np.ndarray],
    action_ids: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在 GPU 上重放 E2 four-update Student-t posterior。"""

    quadrature_nodes, quadrature_weights = make_fixed_quadrature(scenario)
    nodes = torch.from_numpy(quadrature_nodes.astype(np.float32)).to(device)
    prior_weights = torch.from_numpy(quadrature_weights.astype(np.float32)).to(device)
    outcome_grid = load_p1_grid(scenario, action_ids, device)
    node_means = interpolate_outcome_grid(outcome_grid, nodes)
    node_means = node_means.permute(1, 0, 2).contiguous()
    observation_scale = torch.tensor(OBSERVATION_SCALE, dtype=torch.float32, device=device)
    node_means_standardised = node_means / observation_scale[None, None, :]
    bias, precision = load_residual_parameters(scenario, device)
    posterior_weights = posterior_weights_from_histories(
        node_means_standardised=node_means_standardised,
        prior_weights=prior_weights,
        action_indices=torch.from_numpy(arrays["action_indices"]).to(device),
        observations_standardised=torch.from_numpy(
            arrays["observations_standardised"]
        ).to(device),
        observation_mask=torch.from_numpy(arrays["observation_mask"]).to(device),
        residual_bias_standardised=bias,
        precision=precision,
        degrees_of_freedom=STUDENT_T_DEGREES_OF_FREEDOM,
    )
    return nodes, prior_weights, posterior_weights


def load_trained_proposal(
    scenario: str,
    device: torch.device,
) -> tuple[TwoStageProposal, TaskNormaliser]:
    """读取一个场景的 fixed proposal checkpoint。"""

    checkpoint_path, normaliser_path = checkpoint_paths(scenario)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TwoStageProposal(int(checkpoint["hidden_parameter_dimension"]))
    model.region_model.load_state_dict(checkpoint["region_model_state_dict"])
    model.action_model.load_state_dict(checkpoint["action_model_state_dict"])
    model.to(device).eval()
    with normaliser_path.open("r", encoding="utf-8") as handle:
        normaliser_payload = json.load(handle)
    normaliser = TaskNormaliser(
        torch.tensor(normaliser_payload["position_mean"], dtype=torch.float32, device=device),
        torch.tensor(normaliser_payload["position_std"], dtype=torch.float32, device=device),
    )
    return model, normaliser


@torch.inference_mode()
def point_method_topk(
    model: TwoStageProposal,
    tasks: torch.Tensor,
    hidden_parameters: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """分批计算 point-conditioned Top-200。"""

    all_indices: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    for start in range(0, tasks.shape[0], batch_size):
        stop = min(tasks.shape[0], start + batch_size)
        with autocast_context(tasks.device):
            probabilities = model.point_action_probabilities(
                tasks[start:stop], hidden_parameters[start:stop]
            )
        indices, values = topk_probabilities(probabilities, TOPK_SAVE_COUNT)
        all_indices.append(indices)
        all_probabilities.append(values)
    return torch.cat(all_indices), torch.cat(all_probabilities)


@torch.inference_mode()
def infer_candidate_sets(
    scenario: str,
    model: TwoStageProposal,
    normaliser: TaskNormaliser,
    arrays: dict[str, np.ndarray],
    nodes: torch.Tensor,
    posterior_weights: torch.Tensor,
    point_batch_size: int,
    node_query_chunk_size: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """生成四种 E3 proposal methods 的 Top-200 candidates。"""

    device = nodes.device
    raw_tasks = torch.from_numpy(arrays["task_queries"]).to(device)
    tasks = normaliser.transform(raw_tasks)
    true_hidden = torch.from_numpy(arrays["true_hidden_parameters"]).to(device)
    nominal_hidden = torch.zeros_like(true_hidden)
    posterior_mean = torch.einsum("bn,nd->bd", posterior_weights, nodes)

    results = {
        "nominal_condition": point_method_topk(
            model, tasks, nominal_hidden, point_batch_size
        ),
        "posterior_mean": point_method_topk(
            model, tasks, posterior_mean, point_batch_size
        ),
        "ground_truth_condition": point_method_topk(
            model, tasks, true_hidden, point_batch_size
        ),
    }
    with autocast_context(device):
        belief_probabilities = belief_marginalised_probabilities(
            model,
            tasks,
            nodes,
            posterior_weights,
            node_query_chunk_size=node_query_chunk_size,
        )
    results["belief_marginalised"] = topk_probabilities(
        belief_probabilities, TOPK_SAVE_COUNT
    )
    return results


def reference_metrics(
    scenario: str,
    role: str,
    metadata: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    candidate_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    action_ids: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, np.ndarray]]]]:
    """使用 E1 MuJoCo outcome tables 计算 candidate-oracle metrics。"""

    conditions = load_conditions(scenario, role)
    condition_ids = [row["condition_id"] for row in conditions]
    condition_index = {condition_id: index for index, condition_id in enumerate(condition_ids)}
    outcomes = torch.from_numpy(
        load_outcome_array(scenario, role, conditions, action_ids)
    ).to(device)
    tasks = torch.from_numpy(arrays["task_queries"]).to(device)
    case_condition = torch.tensor(
        [condition_index[row["condition_id"]] for row in metadata],
        dtype=torch.long,
        device=device,
    )
    methods = tuple(candidate_sets)
    stored: dict[str, dict[int, dict[str, list[np.ndarray]]]] = {
        method: {
            k: defaultdict(list) for k in K_VALUES
        }
        for method in methods
    }

    for start in range(0, len(metadata), batch_size):
        stop = min(len(metadata), start + batch_size)
        batch_outcomes = outcomes[case_condition[start:stop]]
        batch_tasks = tasks[start:stop]
        costs = tnpo_costs(batch_outcomes, batch_tasks)
        library_cost = costs.amin(dim=1)
        position_error = torch.linalg.vector_norm(
            batch_outcomes[:, :, :2] - batch_tasks[:, None, :2], dim=-1
        )
        desired_yaw = torch.atan2(batch_tasks[:, 2], batch_tasks[:, 3])
        yaw_error = torch.abs(
            wrap_to_pi(batch_outcomes[:, :, 2] - desired_yaw[:, None])
        )
        success = (position_error <= 0.010) & (yaw_error <= math.radians(5.0))
        for method in methods:
            top_indices = candidate_sets[method][0][start:stop]
            for k in K_VALUES:
                indices = top_indices[:, :k]
                candidate_costs = torch.gather(costs, 1, indices)
                candidate_success = torch.gather(success, 1, indices)
                candidate_cost = candidate_costs.amin(dim=1)
                gap = candidate_cost - library_cost
                values = {
                    "library_cost": library_cost,
                    "candidate_cost": candidate_cost,
                    "proposal_gap": gap,
                    "near_optimal_0p10": (gap <= 0.10).float(),
                    "near_optimal_0p05": (gap <= 0.05).float(),
                    "exact_oracle_coverage": (candidate_cost <= library_cost + 1e-9).float(),
                    "candidate_one_step_success": candidate_success.any(dim=1).float(),
                }
                for name, tensor in values.items():
                    stored[method][k][name].append(tensor.cpu().numpy())

    combined: dict[str, dict[int, dict[str, np.ndarray]]] = {
        method: {
            k: {
                name: np.concatenate(parts)
                for name, parts in values.items()
            }
            for k, values in method_rows.items()
        }
        for method, method_rows in stored.items()
    }
    metric_rows: list[dict[str, Any]] = []
    for method in methods:
        for k in K_VALUES:
            values = combined[method][k]
            for case_index, case in enumerate(metadata):
                metric_rows.append(
                    {
                        **case,
                        "method": method,
                        "k": k,
                        **{
                            name: float(metric[case_index])
                            for name, metric in values.items()
                        },
                    }
                )
    return metric_rows, combined


def descriptive_summary(values: np.ndarray) -> dict[str, float | int]:
    """返回论文结果表使用的紧凑描述统计。"""

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def paired_two_way_bootstrap(
    metadata: list[dict[str, Any]],
    primary_values: dict[str, dict[str, np.ndarray]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """对 conditions 与 stratum-specific targets 执行 paired two-way bootstrap。"""

    condition_ids = sorted({row["condition_id"] for row in metadata})
    target_ids = sorted({row["target_id"] for row in metadata})
    condition_lookup = {value: index for index, value in enumerate(condition_ids)}
    target_lookup = {value: index for index, value in enumerate(target_ids)}
    case_condition = np.asarray(
        [condition_lookup[row["condition_id"]] for row in metadata], dtype=np.int64
    )
    case_target = np.asarray(
        [target_lookup[row["target_id"]] for row in metadata], dtype=np.int64
    )
    targets_by_stratum: dict[str, list[int]] = defaultdict(list)
    for row in metadata:
        key = f"{row['target_group']}::{row['target_stratum']}"
        index = target_lookup[row["target_id"]]
        if index not in targets_by_stratum[key]:
            targets_by_stratum[key].append(index)

    methods = tuple(primary_values)
    metrics = ("candidate_cost", "proposal_gap", "near_optimal_0p10")
    value_matrix = np.stack(
        [primary_values[method][metric] for metric in metrics for method in methods],
        axis=1,
    ).astype(np.float64)
    estimates = np.empty((resamples, value_matrix.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    chunk_size = 100
    for start in range(0, resamples, chunk_size):
        stop = min(resamples, start + chunk_size)
        size = stop - start
        condition_counts = rng.multinomial(
            len(condition_ids),
            np.full(len(condition_ids), 1.0 / len(condition_ids)),
            size=size,
        )
        target_counts = np.zeros((size, len(target_ids)), dtype=np.int32)
        for indices in targets_by_stratum.values():
            draws = rng.multinomial(
                len(indices), np.full(len(indices), 1.0 / len(indices)), size=size
            )
            target_counts[:, indices] = draws
        weights = (
            condition_counts[:, case_condition] * target_counts[:, case_target]
        ).astype(np.float64)
        denominator = weights.sum(axis=1, keepdims=True)
        weighted_values = weights[:, :, None] * value_matrix[None, :, :]
        estimates[start:stop] = weighted_values.sum(axis=1) / denominator

    method_summaries: dict[str, Any] = {}
    column = 0
    for metric in metrics:
        for method in methods:
            samples = estimates[:, column]
            method_summaries.setdefault(method, {})[metric] = {
                "point_estimate": float(np.mean(primary_values[method][metric])),
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
            }
            column += 1

    index = {
        (metric, method): metric_index * len(methods) + method_index
        for metric_index, metric in enumerate(metrics)
        for method_index, method in enumerate(methods)
    }
    comparisons: dict[str, Any] = {}
    for baseline in ("nominal_condition", "posterior_mean"):
        key = f"belief_marginalised_vs_{baseline}"
        comparisons[key] = {}
        for metric in metrics:
            if metric == "near_optimal_0p10":
                samples = (
                    estimates[:, index[(metric, "belief_marginalised")]]
                    - estimates[:, index[(metric, baseline)]]
                )
                point = float(
                    np.mean(primary_values["belief_marginalised"][metric])
                    - np.mean(primary_values[baseline][metric])
                )
                direction = "belief_minus_baseline"
            else:
                samples = (
                    estimates[:, index[(metric, baseline)]]
                    - estimates[:, index[(metric, "belief_marginalised")]]
                )
                point = float(
                    np.mean(primary_values[baseline][metric])
                    - np.mean(primary_values["belief_marginalised"][metric])
                )
                direction = "baseline_minus_belief"
            comparisons[key][metric] = {
                "effect_direction": direction,
                "point_estimate": point,
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
            }
    return {
        "resamples": resamples,
        "seed": seed,
        "method_mean_confidence_intervals": method_summaries,
        "paired_comparisons": comparisons,
    }


def save_candidate_artifact(
    path: Path,
    metadata: list[dict[str, Any]],
    candidate_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    action_ids: list[str],
) -> None:
    """保存 Top-200 action indices、IDs 与 probabilities。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "episode_keys": np.asarray([row["episode_key"] for row in metadata]),
        "action_ids": np.asarray(action_ids),
    }
    for method, (indices, probabilities) in candidate_sets.items():
        payload[f"{method}_indices"] = indices.cpu().numpy().astype(np.int32)
        payload[f"{method}_probabilities"] = probabilities.cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **payload)


def evaluate_scenario(
    scenario: str,
    role: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """运行一个场景的正式 E3 Validation 或 Test。"""

    action_ids, _ = load_action_layout()
    metadata, arrays = build_decision_cases(scenario, role, action_ids)
    model, normaliser = load_trained_proposal(scenario, device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    nodes, _, posterior_weights = replay_case_posteriors(
        scenario, arrays, action_ids, device
    )
    candidate_sets = infer_candidate_sets(
        scenario,
        model,
        normaliser,
        arrays,
        nodes,
        posterior_weights,
        args.point_batch_size,
        args.node_query_chunk_size,
    )
    metric_rows, metric_arrays = reference_metrics(
        scenario,
        role,
        metadata,
        arrays,
        candidate_sets,
        action_ids,
        device,
        args.reference_batch_size,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    primary_values = {
        method: metric_arrays[method][PRIMARY_K]
        for method in metric_arrays
    }
    bootstrap = paired_two_way_bootstrap(
        metadata, primary_values, args.bootstrap_resamples, args.bootstrap_seed
    )
    methods_summary = {
        method: {
            str(k): {
                name: descriptive_summary(values)
                for name, values in metric_arrays[method][k].items()
            }
            for k in K_VALUES
        }
        for method in metric_arrays
    }
    output_root = E3_RESULTS_ROOT / "evaluation" / role / scenario
    candidate_path = output_root / "top200_candidates.npz"
    save_candidate_artifact(candidate_path, metadata, candidate_sets, action_ids)
    fields = list(metric_rows[0])
    write_csv(output_root / "decision_case_metrics.csv", metric_rows, fields)
    write_json(output_root / "bootstrap_summary.json", bootstrap)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "scenario": scenario,
        "role": role,
        "eligible_case_count": len(metadata),
        "excluded_one_push_count": int(arrays["excluded_one_push_count"]),
        "points_per_dimension": POINTS_PER_DIMENSION,
        "node_count": int(nodes.shape[0]),
        "belief_update_horizon": BELIEF_UPDATE_HORIZON,
        "student_t_degrees_of_freedom": STUDENT_T_DEGREES_OF_FREEDOM,
        "covariance_inflation": COVARIANCE_INFLATION[scenario],
        "primary_k": PRIMARY_K,
        "k_values": list(K_VALUES),
        "methods": methods_summary,
        "runtime": {
            "elapsed_seconds": elapsed,
            "node_query_chunk_size": args.node_query_chunk_size,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        },
        "candidate_artifact": str(candidate_path.resolve()),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def evaluate_role(args: argparse.Namespace) -> dict[str, Any]:
    """按单 CUDA 任务顺序运行 Validation 或 Test。"""

    device = require_cuda()
    summaries: dict[str, Any] = {}
    for scenario_index, scenario in enumerate(select_scenarios(args.scenario)):
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.bootstrap_seed = args.bootstrap_seed + scenario_index
        summaries[scenario] = evaluate_scenario(
            scenario, args.role, scenario_args, device
        )
        print(
            f"{scenario} {args.role}: summary="
            f"{E3_RESULTS_ROOT / 'evaluation' / args.role / scenario / 'summary.json'}"
        )
        torch.cuda.empty_cache()
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "role": args.role,
        "scenarios": summaries,
    }
    path = E3_RESULTS_ROOT / "evaluation" / args.role / "combined_summary.json"
    write_json(path, combined)
    print(f"Combined summary: {path.resolve()}")
    return combined


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """测量一个 Training pass 与 64-case exact marginalisation。"""

    scenario = args.scenario
    device = require_cuda()
    set_seed(args.seed)
    training = load_label_artifact(scenario, "training", device)
    normaliser = make_normaliser(training["task_queries"])
    model = TwoStageProposal(training["hidden_parameters"].shape[1]).to(device)
    region_optimizer = torch.optim.AdamW(model.region_model.parameters(), lr=1e-3)
    action_optimizer = torch.optim.AdamW(model.action_model.parameters(), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    region_loss = train_region_pass(
        model,
        region_optimizer,
        training,
        normaliser,
        args.region_batch_size,
        generator,
    )
    action_loss = train_action_pass(
        model,
        action_optimizer,
        training,
        normaliser,
        args.action_batch_size,
        0,
        generator,
    )
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    training_peak = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    model.eval()

    action_ids, _ = load_action_layout()
    _, arrays = build_decision_cases(scenario, "validation", action_ids)
    arrays = {
        key: value[: args.decision_cases]
        if isinstance(value, np.ndarray) and value.ndim > 0
        else value
        for key, value in arrays.items()
    }
    nodes, _, posterior_weights = replay_case_posteriors(
        scenario, arrays, action_ids, device
    )
    tasks = normaliser.transform(
        torch.from_numpy(arrays["task_queries"]).to(device)
    )
    torch.cuda.reset_peak_memory_stats(device)
    marginal_started = time.perf_counter()
    with autocast_context(device):
        probabilities = belief_marginalised_probabilities(
            model,
            tasks,
            nodes,
            posterior_weights,
            args.node_query_chunk_size,
        )
    topk_probabilities(probabilities, TOPK_SAVE_COUNT)
    torch.cuda.synchronize(device)
    marginal_seconds = time.perf_counter() - marginal_started
    marginal_peak = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": scenario,
        "device": torch.cuda.get_device_name(device),
        "training_pass": {
            "query_condition_cases": int(
                training["task_queries"].shape[0]
                * training["hidden_parameters"].shape[0]
            ),
            "region_loss": region_loss,
            "action_loss": action_loss,
            "seconds": training_seconds,
            "peak_cuda_memory_mib": training_peak,
        },
        "exact_marginalisation": {
            "cases": args.decision_cases,
            "nodes_per_case": int(nodes.shape[0]),
            "node_queries": int(args.decision_cases * nodes.shape[0]),
            "node_query_chunk_size": args.node_query_chunk_size,
            "seconds": marginal_seconds,
            "peak_cuda_memory_mib": marginal_peak,
        },
    }
    path = E3_RESULTS_ROOT / "benchmark" / f"{scenario}_summary.json"
    write_json(path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def add_evaluation_arguments(parser: argparse.ArgumentParser, role: str) -> None:
    """添加 Validation/Test 共享参数。"""

    parser.set_defaults(handler=evaluate_role, role=role)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), required=True)
    parser.add_argument("--point-batch-size", type=int, default=16_384)
    parser.add_argument("--node-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--reference-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    default_seed = 2026081401 if role == "validation" else 2026081404
    parser.add_argument("--bootstrap-seed", type=int, default=default_seed)


def build_parser() -> argparse.ArgumentParser:
    """建立 E3 命令行接口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-training", help="在 GPU 上生成 Training/Validation compact labels。"
    )
    prepare.add_argument("--scenario", choices=(*SCENARIOS, "all"), required=True)
    prepare.add_argument("--query-chunk-size", type=int, default=1024)
    prepare.set_defaults(handler=prepare_training)

    train = subparsers.add_parser(
        "train-proposal", help="训练统一 two-stage inverse proposal。"
    )
    train.add_argument("--scenario", choices=(*SCENARIOS, "all"), required=True)
    train.add_argument("--region-batch-size", type=int, default=32_768)
    train.add_argument("--action-batch-size", type=int, default=32_768)
    train.add_argument("--validation-batch-size", type=int, default=32_768)
    train.add_argument("--max-epochs", type=int, default=MAXIMUM_TRAINING_EPOCHS)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--minimum-improvement", type=float, default=1e-5)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--seed", type=int, default=TRAINING_SEED)
    train.set_defaults(handler=train_proposal)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="测量完整 Training pass 与 64-case marginalisation。"
    )
    benchmark_parser.add_argument("--scenario", choices=SCENARIOS, default="joint")
    benchmark_parser.add_argument("--region-batch-size", type=int, default=32_768)
    benchmark_parser.add_argument("--action-batch-size", type=int, default=32_768)
    benchmark_parser.add_argument("--node-query-chunk-size", type=int, default=65_536)
    benchmark_parser.add_argument("--decision-cases", type=int, default=64)
    benchmark_parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    benchmark_parser.set_defaults(handler=benchmark)

    validation = subparsers.add_parser(
        "evaluate-validation", help="运行正式 E3 Validation evaluation。"
    )
    add_evaluation_arguments(validation, "validation")
    test = subparsers.add_parser(
        "evaluate-test", help="运行固定配置 Independent Test evaluation。"
    )
    add_evaluation_arguments(test, "test")
    return parser


def main() -> None:
    """执行所选 E3 子命令。"""

    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
