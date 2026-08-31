"""复用 HCR V2 E1–E5 产物计算单一 cost-derived affordance map。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# 提前初始化 NumPy polynomial 后端，避免在 PyTorch 后延迟加载 OpenMP。
np.polynomial.legendre.leggauss(2)

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_e3 as e3_runner
import run_e5 as e5_runner
from push_core.hcr_v2.e3 import (
    ACTIONS_PER_REGION,
    ACTION_COUNT,
    LABEL_TEMPERATURE,
    REGION_COUNT,
    action_soft_labels,
    interpolate_outcome_grid,
    tnpo_costs,
)
from push_core.hcr_v2.e4 import (
    point_condition_candidate_scores,
    posterior_expected_candidate_scores,
)
from push_core.project_paths import HCR_V2_RESULTS_DIR


PROTOCOL_VERSION = "cost_derived_affordance_map_v1"
RESULT_ROOT = HCR_V2_RESULTS_DIR / "affordance_map_supplementary"
K_VALUES = (1, 20, 50, 100)
POSTERIOR_REPLAY_TOLERANCE = 1e-4
ANALYSIS_MODES = ("known_condition", "belief_conditioned", "closed_loop")

DECISION_FIELDS = [
    "protocol_version",
    "analysis_mode",
    "role",
    "scenario",
    "target_group",
    "condition_id",
    "target_id",
    "episode_key",
    "push_index",
    "decision_state_source",
    "valid_update_count_pre",
    "affordance_method",
    "top1_action_id",
    "top1_region_index",
    "highest_probability_region_index",
    "region_top_matches_action_top",
    "probability_sum",
    "minimum_probability",
    "hierarchy_reconstruction_max_abs_error",
    "within_region_sum_max_abs_error",
    "top1_matches_minimum_predicted_cost",
    "top1_direct_mujoco_regret",
    "posterior_mean_max_abs_error_pre",
    "posterior_covariance_trace_abs_error_pre",
    "posterior_mean_max_abs_error_post",
    "posterior_covariance_trace_abs_error_post",
    "expected_cost_time_s",
    "probability_time_s",
    "region_aggregation_time_s",
    "total_map_time_s",
    "peak_cuda_memory_mib",
]
for k in K_VALUES:
    DECISION_FIELDS.extend(
        [
            f"top{k}_retained_probability_mass",
            f"top{k}_direct_mujoco_gap",
            f"top{k}_exact_mujoco_oracle_coverage",
            f"top{k}_direct_mujoco_near_0p05",
            f"top{k}_direct_mujoco_near_0p10",
        ]
    )

MAP_FIELDS = [
    "protocol_version",
    "analysis_mode",
    "role",
    "scenario",
    "target_group",
    "condition_id",
    "target_id",
    "episode_key",
    "push_index",
    "decision_state_source",
    "valid_update_count_pre",
    "task_local_x_m",
    "task_local_y_m",
    "action_index",
    "v2_action_id",
    "region_index",
    "contact_region_id",
    "action_param_index",
    "complete_action_probability",
    "contact_region_probability",
    "within_region_action_probability",
    "predicted_tnpo_cost",
    "direct_mujoco_tnpo_cost",
    "probability_rank",
]


def require_device(value: str) -> torch.device:
    """解析计算设备。"""

    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用 CUDA")
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"未知 device: {value}")


def synchronise(device: torch.device) -> None:
    """在计时边界同步 CUDA。"""

    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """以固定字段顺序写出 UTF-8 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写出严格 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def ordered_indices(values: np.ndarray, descending: bool) -> np.ndarray:
    """按数值和 action index 形成确定性排序。"""

    action_indices = np.arange(len(values), dtype=np.int64)
    primary = -values if descending else values
    return np.lexsort((action_indices, primary))


def derive_hierarchy(
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """由完整动作概率推导 region 和 region 内概率。"""

    matrix = np.asarray(probabilities, dtype=np.float64).reshape(
        REGION_COUNT, ACTIONS_PER_REGION
    )
    region = matrix.sum(axis=1)
    conditional = np.divide(
        matrix,
        region[:, None],
        out=np.zeros_like(matrix),
        where=region[:, None] > 0.0,
    )
    reconstructed = region[:, None] * conditional
    reconstruction_error = float(np.max(np.abs(reconstructed - matrix)))
    active_regions = region > 0.0
    conditional_error = (
        float(
            np.max(
                np.abs(conditional[active_regions].sum(axis=1) - 1.0)
            )
        )
        if np.any(active_regions)
        else 0.0
    )
    return region, conditional, reconstruction_error, conditional_error


def read_episode_groups(
    role: str,
    scenario: str,
    target_group: str,
    maximum_episodes: int,
    require_update_four: bool,
    episode_keys: set[str] | None,
) -> list[list[dict[str, str]]]:
    """稳定读取 belief-marginalised E5 episodes。"""

    root = e5_runner.E5_DATA_ROOT / "closed_loop" / role / scenario
    if not root.exists():
        raise FileNotFoundError(root)
    selected: list[list[dict[str, str]]] = []
    for path in sorted(root.glob("*.csv")):
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["controller_id"] != e5_runner.BELIEF_MARGINALISED_CONTROLLER_ID:
                    continue
                if target_group != "all" and row["target_group"] != target_group:
                    continue
                grouped[row["episode_key"]].append(row)
        for episode_key in sorted(grouped):
            if episode_keys is not None and episode_key not in episode_keys:
                continue
            rows = sorted(grouped[episode_key], key=lambda row: int(row["push_index"]))
            rows = append_update_four_terminal_state(rows)
            if require_update_four and max(
                int(row["valid_update_count_pre"]) for row in rows
            ) < 4:
                continue
            selected.append(rows)
            if maximum_episodes > 0 and len(selected) >= maximum_episodes:
                return selected
    if not selected:
        raise RuntimeError(
            f"没有找到符合条件的 E5 episode: {role}/{scenario}/{target_group}"
        )
    return selected


def append_update_four_terminal_state(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """在只有 post=4 时补一个不执行动作的 terminal decision state。"""

    if not rows:
        return rows
    if any(int(row["valid_update_count_pre"]) == 4 for row in rows):
        return rows
    last = rows[-1]
    if int(last["valid_update_count_post"]) != 4:
        return rows
    query = e5_runner.make_task_query(
        np.asarray(
            [float(last["post_push_x_m"]), float(last["post_push_y_m"])],
            dtype=np.float64,
        ),
        float(last["post_push_yaw_rad"]),
        np.asarray(
            [float(last["target_world_x_m"]), float(last["target_world_y_m"])],
            dtype=np.float64,
        ),
        float(last["target_world_yaw_rad"]),
    )
    terminal = dict(last)
    terminal["push_index"] = str(int(last["push_index"]) + 1)
    terminal["decision_state_source"] = "post_update_terminal"
    terminal["valid_update_count_pre"] = "4"
    terminal["valid_update_count_post"] = "4"
    terminal["belief_updated"] = "0"
    terminal["task_local_x_m"] = str(float(query[0]))
    terminal["task_local_y_m"] = str(float(query[1]))
    terminal["task_yaw_sin"] = str(float(query[2]))
    terminal["task_yaw_cos"] = str(float(query[3]))
    for coordinate in ("friction", "com_x", "com_y"):
        terminal[f"belief_mean_hidden_u_{coordinate}_pre"] = last[
            f"belief_mean_hidden_u_{coordinate}_post"
        ]
    terminal["belief_covariance_trace_pre"] = last[
        "belief_covariance_trace_post"
    ]
    return [*rows, terminal]


def posterior_log_errors(
    engine: e5_runner.ClosedLoopDecisionEngine,
    belief: e5_runner.BeliefState,
    row: dict[str, str],
    suffix: str,
) -> tuple[float, float]:
    """比较重放 posterior 与 E5 日志摘要。"""

    summary = engine.belief_summary(belief)
    expected = e5_runner.belief_log_fields(engine.scenario, summary, suffix)
    mean_errors = []
    for field in (
        "belief_mean_hidden_u_friction",
        "belief_mean_hidden_u_com_x",
        "belief_mean_hidden_u_com_y",
    ):
        key = f"{field}_{suffix}"
        mean_errors.append(abs(float(row[key]) - float(expected[key])))
    covariance_key = f"belief_covariance_trace_{suffix}"
    covariance_error = abs(
        float(row[covariance_key]) - float(expected[covariance_key])
    )
    return max(mean_errors), covariance_error


def task_query_from_row(row: dict[str, str]) -> np.ndarray:
    """读取 E5 当前状态下的四维 task query。"""

    return np.asarray(
        [
            float(row["task_local_x_m"]),
            float(row["task_local_y_m"]),
            float(row["task_yaw_sin"]),
            float(row["task_yaw_cos"]),
        ],
        dtype=np.float32,
    )


@torch.inference_mode()
def map_from_belief(
    engine: e5_runner.ClosedLoopDecisionEngine,
    task_query: np.ndarray,
    belief: e5_runner.BeliefState,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """由完整 posterior 计算 cost-derived affordance map。"""

    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    raw_task = torch.as_tensor(
        task_query[None, :], dtype=torch.float32, device=engine.device
    )
    synchronise(engine.device)
    total_start = time.perf_counter()
    cost_start = time.perf_counter()
    node_costs = tnpo_costs(engine.node_outcomes[None, ...], raw_task)[0]
    expected_costs = torch.einsum("n,na->a", belief.weights.float(), node_costs)
    synchronise(engine.device)
    cost_time = time.perf_counter() - cost_start

    probability_start = time.perf_counter()
    probabilities = action_soft_labels(
        expected_costs[None, :], LABEL_TEMPERATURE
    )[0]
    synchronise(engine.device)
    probability_time = time.perf_counter() - probability_start

    aggregation_start = time.perf_counter()
    probabilities_cpu = probabilities.detach().cpu().numpy()
    expected_costs_cpu = expected_costs.detach().cpu().numpy()
    derive_hierarchy(probabilities_cpu)
    aggregation_time = time.perf_counter() - aggregation_start
    total_time = time.perf_counter() - total_start
    peak_memory = (
        torch.cuda.max_memory_allocated(engine.device) / (1024.0**2)
        if engine.device.type == "cuda"
        else 0.0
    )
    return probabilities_cpu, expected_costs_cpu, {
        "expected_cost_time_s": cost_time,
        "probability_time_s": probability_time,
        "region_aggregation_time_s": aggregation_time,
        "total_map_time_s": total_time,
        "peak_cuda_memory_mib": peak_memory,
    }


@torch.inference_mode()
def map_from_known_condition(
    engine: e5_runner.ClosedLoopDecisionEngine,
    task_query: np.ndarray,
    hidden_parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """由一个已知 friction/COM point 计算 cost-derived affordance map。"""

    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    raw_task = torch.as_tensor(
        task_query[None, :], dtype=torch.float32, device=engine.device
    )
    hidden = torch.as_tensor(
        hidden_parameters[None, :], dtype=torch.float32, device=engine.device
    )
    synchronise(engine.device)
    total_start = time.perf_counter()
    cost_start = time.perf_counter()
    predicted_outcomes = interpolate_outcome_grid(engine.outcome_grid, hidden)
    predicted_costs = tnpo_costs(predicted_outcomes, raw_task)[0]
    synchronise(engine.device)
    cost_time = time.perf_counter() - cost_start

    probability_start = time.perf_counter()
    probabilities = action_soft_labels(
        predicted_costs[None, :], LABEL_TEMPERATURE
    )[0]
    synchronise(engine.device)
    probability_time = time.perf_counter() - probability_start

    aggregation_start = time.perf_counter()
    probabilities_cpu = probabilities.detach().cpu().numpy()
    predicted_costs_cpu = predicted_costs.detach().cpu().numpy()
    derive_hierarchy(probabilities_cpu)
    aggregation_time = time.perf_counter() - aggregation_start
    total_time = time.perf_counter() - total_start
    peak_memory = (
        torch.cuda.max_memory_allocated(engine.device) / (1024.0**2)
        if engine.device.type == "cuda"
        else 0.0
    )
    return probabilities_cpu, predicted_costs_cpu, {
        "expected_cost_time_s": cost_time,
        "probability_time_s": probability_time,
        "region_aggregation_time_s": aggregation_time,
        "total_map_time_s": total_time,
        "peak_cuda_memory_mib": peak_memory,
    }


def direct_costs_from_outcomes(
    outcomes: np.ndarray,
    task_query: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """由已有 direct MuJoCo outcome library 计算目标相关 cost。"""

    with torch.inference_mode():
        outcome_tensor = torch.as_tensor(
            outcomes[None, ...], dtype=torch.float32, device=device
        )
        task_tensor = torch.as_tensor(
            task_query[None, :], dtype=torch.float32, device=device
        )
        return tnpo_costs(outcome_tensor, task_tensor)[0].detach().cpu().numpy()


def direct_costs_from_outcomes_numpy(
    outcomes: np.ndarray,
    task_query: np.ndarray,
) -> np.ndarray:
    """在批量 decision 汇总中用 NumPy 计算 direct MuJoCo cost。"""

    position_error = np.linalg.norm(
        outcomes[:, :2] - task_query[None, :2], axis=1
    )
    desired_yaw = math.atan2(float(task_query[2]), float(task_query[3]))
    yaw_difference = outcomes[:, 2] - desired_yaw
    yaw_error = np.abs((yaw_difference + math.pi) % (2.0 * math.pi) - math.pi)
    return (
        0.5 * position_error / 0.010
        + 0.5 * yaw_error / math.radians(5.0)
    )


def probabilities_from_costs(costs: np.ndarray) -> np.ndarray:
    """稳定地把一组 cost 转换为正式 affordance probabilities。"""

    shifted = np.asarray(costs, dtype=np.float64) - float(np.min(costs))
    unnormalised = np.exp(-shifted / LABEL_TEMPERATURE)
    return unnormalised / float(np.sum(unnormalised))


def decision_statistics(
    probabilities: np.ndarray,
    predicted_costs: np.ndarray,
    direct_costs: np.ndarray | None,
) -> dict[str, Any]:
    """计算结构、候选质量和区域/动作关系。"""

    order = ordered_indices(probabilities, descending=True)
    cost_order = ordered_indices(predicted_costs, descending=False)
    region, conditional, reconstruction_error, conditional_error = derive_hierarchy(
        probabilities
    )
    top1_index = int(order[0])
    payload: dict[str, Any] = {
        "top1_action_index": top1_index,
        "top1_region_index": top1_index // ACTIONS_PER_REGION,
        "highest_probability_region_index": int(np.argmax(region)),
        "region_top_matches_action_top": int(
            np.argmax(region) == top1_index // ACTIONS_PER_REGION
        ),
        "probability_sum": float(np.sum(probabilities)),
        "minimum_probability": float(np.min(probabilities)),
        "hierarchy_reconstruction_max_abs_error": reconstruction_error,
        "within_region_sum_max_abs_error": conditional_error,
        "top1_matches_minimum_predicted_cost": int(top1_index == int(cost_order[0])),
        "top1_direct_mujoco_regret": "",
    }
    direct_minimum = None if direct_costs is None else float(np.min(direct_costs))
    if direct_costs is not None:
        payload["top1_direct_mujoco_regret"] = float(
            direct_costs[top1_index] - direct_minimum
        )
    for k in K_VALUES:
        indices = order[:k]
        payload[f"top{k}_retained_probability_mass"] = float(
            np.sum(probabilities[indices])
        )
        if direct_costs is None:
            payload[f"top{k}_direct_mujoco_gap"] = ""
            payload[f"top{k}_exact_mujoco_oracle_coverage"] = ""
            payload[f"top{k}_direct_mujoco_near_0p05"] = ""
            payload[f"top{k}_direct_mujoco_near_0p10"] = ""
            continue
        gap = float(np.min(direct_costs[indices]) - direct_minimum)
        direct_top1 = int(np.argmin(direct_costs))
        payload[f"top{k}_direct_mujoco_gap"] = gap
        payload[f"top{k}_exact_mujoco_oracle_coverage"] = int(
            direct_top1 in indices
        )
        payload[f"top{k}_direct_mujoco_near_0p05"] = int(gap <= 0.05)
        payload[f"top{k}_direct_mujoco_near_0p10"] = int(gap <= 0.10)
    return payload


def validate_structure(statistics: dict[str, Any]) -> None:
    """执行 affordance-map 硬结构检查。"""

    if abs(float(statistics["probability_sum"]) - 1.0) > 1e-5:
        raise RuntimeError("affordance probability sum 不是 1")
    if float(statistics["minimum_probability"]) < -1e-8:
        raise RuntimeError("affordance map 出现负概率")
    if float(statistics["hierarchy_reconstruction_max_abs_error"]) > 1e-7:
        raise RuntimeError("region/action hierarchy reconstruction 失败")
    if float(statistics["within_region_sum_max_abs_error"]) > 1e-5:
        raise RuntimeError("within-region probabilities 未归一化")
    if int(statistics["top1_matches_minimum_predicted_cost"]) != 1:
        raise RuntimeError("最高 affordance probability 不对应最低 predicted cost")


def append_map_rows(
    output: list[dict[str, Any]],
    metadata: dict[str, Any],
    action_ids: list[str],
    region_ids: list[int],
    probabilities: np.ndarray,
    predicted_costs: np.ndarray,
    direct_costs: np.ndarray | None,
) -> None:
    """把一个 decision 的 4,536-action map 写入 row buffer。"""

    order = ordered_indices(probabilities, descending=True)
    ranks = np.empty(ACTION_COUNT, dtype=np.int64)
    ranks[order] = np.arange(1, ACTION_COUNT + 1)
    region, conditional, _, _ = derive_hierarchy(probabilities)
    for action_index, probability in enumerate(probabilities):
        region_index = action_index // ACTIONS_PER_REGION
        action_parameter_index = action_index % ACTIONS_PER_REGION
        row = dict(metadata)
        row.update(
            {
                "action_index": action_index,
                "v2_action_id": action_ids[action_index],
                "region_index": region_index,
                "contact_region_id": region_ids[region_index],
                "action_param_index": action_parameter_index,
                "complete_action_probability": float(probability),
                "contact_region_probability": float(region[region_index]),
                "within_region_action_probability": float(
                    conditional[region_index, action_parameter_index]
                ),
                "predicted_tnpo_cost": float(predicted_costs[action_index]),
                "direct_mujoco_tnpo_cost": (
                    "" if direct_costs is None else float(direct_costs[action_index])
                ),
                "probability_rank": int(ranks[action_index]),
            }
        )
        output.append(row)


def base_metadata(
    row: dict[str, str], analysis_mode: str
) -> dict[str, Any]:
    """提取 decision 与 map rows 共用字段。"""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_mode": analysis_mode,
        "role": row["role"],
        "scenario": row["scenario"],
        "target_group": row["target_group"],
        "condition_id": row["condition_id"],
        "target_id": row["target_id"],
        "episode_key": row["episode_key"],
        "push_index": int(row["push_index"]),
        "decision_state_source": row.get("decision_state_source", "pre_push"),
        "valid_update_count_pre": int(row["valid_update_count_pre"]),
        "affordance_method": row.get("affordance_method", "cost_derived"),
        "task_local_x_m": float(row["task_local_x_m"]),
        "task_local_y_m": float(row["task_local_y_m"]),
    }


def decision_case_metadata(
    row: dict[str, Any],
    task_query: np.ndarray,
    affordance_method: str,
) -> dict[str, Any]:
    """把 E3/E4 decision case 转换为补充实验字段。"""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_mode": "belief_conditioned",
        "role": row["role"],
        "scenario": row["scenario"],
        "target_group": row["target_group"],
        "condition_id": row["condition_id"],
        "target_id": row["target_id"],
        "episode_key": row["episode_key"],
        "push_index": int(row["decision_push"]),
        "decision_state_source": "e3_e4_decision_case",
        "valid_update_count_pre": int(row["belief_update_count"]),
        "affordance_method": affordance_method,
        "task_local_x_m": float(task_query[0]),
        "task_local_y_m": float(task_query[1]),
    }


@torch.inference_mode()
def analyse_belief_conditioned_cases(
    scenario: str,
    role: str,
    action_ids: list[str],
    device: torch.device,
    case_chunk_size: int,
    maximum_cases: int,
) -> tuple[list[dict[str, Any]], int]:
    """在 E3/E4 cases 上比较 nominal 与 posterior cost-derived maps。"""

    metadata, arrays = e3_runner.build_decision_cases(scenario, role, action_ids)
    if maximum_cases > 0:
        metadata = metadata[:maximum_cases]
        arrays = {
            key: value[:maximum_cases]
            if isinstance(value, np.ndarray) and value.ndim > 0
            else value
            for key, value in arrays.items()
        }
    nodes, _, posterior_weights = e3_runner.replay_case_posteriors(
        scenario, arrays, action_ids, device
    )
    task_queries = torch.from_numpy(arrays["task_queries"]).to(device)
    p1_grid = e3_runner.load_p1_grid(scenario, action_ids, device)
    node_outcomes = interpolate_outcome_grid(p1_grid, nodes)
    case_count = len(metadata)
    candidates = torch.arange(
        ACTION_COUNT, dtype=torch.long, device=device
    )[None, :].expand(case_count, -1)
    nominal_hidden = torch.zeros(
        (case_count, nodes.shape[1]), dtype=torch.float32, device=device
    )
    effective_chunk_size = 1 if scenario == "joint" else case_chunk_size

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronise(device)
    posterior_started = time.perf_counter()
    posterior_costs = posterior_expected_candidate_scores(
        node_outcomes,
        posterior_weights,
        candidates,
        task_queries,
        case_chunk_size=effective_chunk_size,
    )
    synchronise(device)
    posterior_time_per_case = (time.perf_counter() - posterior_started) / case_count

    nominal_started = time.perf_counter()
    nominal_costs = point_condition_candidate_scores(
        p1_grid,
        nominal_hidden,
        candidates,
        task_queries,
        case_chunk_size=max(1, case_chunk_size),
    )
    synchronise(device)
    nominal_time_per_case = (time.perf_counter() - nominal_started) / case_count
    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )

    posterior_costs_cpu = posterior_costs.cpu().numpy()
    nominal_costs_cpu = nominal_costs.cpu().numpy()
    task_queries_cpu = np.asarray(arrays["task_queries"], dtype=np.float32)
    del posterior_costs, nominal_costs, candidates, node_outcomes, p1_grid
    if device.type == "cuda":
        torch.cuda.empty_cache()

    condition_index, direct_outcome_array = load_direct_outcomes(
        scenario, role, action_ids
    )
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(metadata):
        task_query = task_queries_cpu[case_index]
        direct_costs = direct_costs_from_outcomes_numpy(
            direct_outcome_array[condition_index[case["condition_id"]]],
            task_query,
        )
        for method, predicted_costs, expected_cost_time in (
            (
                "posterior_expected_cost",
                posterior_costs_cpu[case_index],
                posterior_time_per_case,
            ),
            (
                "nominal_condition_cost",
                nominal_costs_cpu[case_index],
                nominal_time_per_case,
            ),
        ):
            probability_started = time.perf_counter()
            probabilities = probabilities_from_costs(predicted_costs)
            probability_time = time.perf_counter() - probability_started
            aggregation_started = time.perf_counter()
            derive_hierarchy(probabilities)
            aggregation_time = time.perf_counter() - aggregation_started
            statistics = decision_statistics(
                probabilities, predicted_costs, direct_costs
            )
            validate_structure(statistics)
            row = decision_case_metadata(case, task_query, method)
            row.update(statistics)
            row.update(
                {
                    "top1_action_id": action_ids[
                        statistics["top1_action_index"]
                    ],
                    "posterior_mean_max_abs_error_pre": "",
                    "posterior_covariance_trace_abs_error_pre": "",
                    "posterior_mean_max_abs_error_post": "",
                    "posterior_covariance_trace_abs_error_post": "",
                    "expected_cost_time_s": expected_cost_time,
                    "probability_time_s": probability_time,
                    "region_aggregation_time_s": aggregation_time,
                    "total_map_time_s": (
                        expected_cost_time + probability_time + aggregation_time
                    ),
                    "peak_cuda_memory_mib": peak_memory_mib,
                }
            )
            rows.append(row)
    return rows, case_count


def analyse_closed_loop_episode(
    engine: e5_runner.ClosedLoopDecisionEngine,
    episode_rows: list[dict[str, str]],
    action_ids: list[str],
    region_ids: list[int],
    maximum_decisions: int,
    save_full_maps: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """重放一条 E5 trajectory 并生成闭环 cost-derived maps。"""

    action_index_by_id = {action_id: index for index, action_id in enumerate(action_ids)}
    belief = engine.new_belief()
    summaries: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    selected_rows = episode_rows[:maximum_decisions]
    for row in selected_rows:
        pre_mean_error, pre_covariance_error = posterior_log_errors(
            engine, belief, row, "pre"
        )
        if belief.update_count != int(row["valid_update_count_pre"]):
            raise RuntimeError("重放 posterior update count 与 E5 pre 日志不一致")
        task_query = task_query_from_row(row)
        probabilities, predicted_costs, timings = map_from_belief(
            engine, task_query, belief
        )
        statistics = decision_statistics(probabilities, predicted_costs, None)
        validate_structure(statistics)
        metadata = base_metadata(row, "closed_loop")
        summary = dict(metadata)
        summary.update(statistics)
        summary.update(timings)
        summary["top1_action_id"] = action_ids[statistics["top1_action_index"]]
        summary["posterior_mean_max_abs_error_pre"] = pre_mean_error
        summary["posterior_covariance_trace_abs_error_pre"] = pre_covariance_error
        if save_full_maps:
            append_map_rows(
                map_rows,
                metadata,
                action_ids,
                region_ids,
                probabilities,
                predicted_costs,
                None,
            )

        action_index = action_index_by_id[row["v2_action_id"]]
        if int(row["belief_updated"]) == 1:
            observation = np.asarray(
                [
                    float(row["observation_local_delta_x_m"]),
                    float(row["observation_local_delta_y_m"]),
                    float(row["observation_delta_yaw_rad"]),
                ],
                dtype=np.float32,
            )
            if not engine.update_belief(belief, action_index, observation):
                raise RuntimeError("E5 日志要求 posterior update，但重放未执行")
        post_mean_error, post_covariance_error = posterior_log_errors(
            engine, belief, row, "post"
        )
        if belief.update_count != int(row["valid_update_count_post"]):
            raise RuntimeError("重放 posterior update count 与 E5 post 日志不一致")
        if max(pre_mean_error, post_mean_error) > POSTERIOR_REPLAY_TOLERANCE:
            raise RuntimeError("重放 posterior mean 与 E5 日志不一致")
        if (
            max(pre_covariance_error, post_covariance_error)
            > POSTERIOR_REPLAY_TOLERANCE
        ):
            raise RuntimeError("重放 posterior covariance 与 E5 日志不一致")
        summary["posterior_mean_max_abs_error_post"] = post_mean_error
        summary[
            "posterior_covariance_trace_abs_error_post"
        ] = post_covariance_error
        summaries.append(summary)
    return summaries, map_rows


def analyse_known_condition_episode(
    engine: e5_runner.ClosedLoopDecisionEngine,
    row: dict[str, str],
    action_ids: list[str],
    region_ids: list[int],
    direct_outcomes: np.ndarray,
    save_full_maps: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """分析一个 initial state 的 known-condition map 与 MuJoCo reference。"""

    task_query = task_query_from_row(row)
    hidden = e5_runner.active_hidden_parameters(engine.scenario, row)
    probabilities, predicted_costs, timings = map_from_known_condition(
        engine, task_query, hidden
    )
    direct_costs = direct_costs_from_outcomes(
        direct_outcomes, task_query, engine.device
    )
    statistics = decision_statistics(probabilities, predicted_costs, direct_costs)
    validate_structure(statistics)
    metadata = base_metadata(row, "known_condition")
    summary = dict(metadata)
    summary.update(statistics)
    summary.update(timings)
    summary["top1_action_id"] = action_ids[statistics["top1_action_index"]]
    summary["posterior_mean_max_abs_error_pre"] = ""
    summary["posterior_covariance_trace_abs_error_pre"] = ""
    summary["posterior_mean_max_abs_error_post"] = ""
    summary["posterior_covariance_trace_abs_error_post"] = ""
    map_rows: list[dict[str, Any]] = []
    if save_full_maps:
        append_map_rows(
            map_rows,
            metadata,
            action_ids,
            region_ids,
            probabilities,
            predicted_costs,
            direct_costs,
        )
    return summary, map_rows


def load_direct_outcomes(
    scenario: str,
    role: str,
    action_ids: list[str],
) -> tuple[dict[str, int], np.ndarray]:
    """读取与 condition IDs 对齐的 direct MuJoCo outcome library。"""

    conditions = e3_runner.load_conditions(scenario, role)
    outcomes = e3_runner.load_outcome_array(scenario, role, conditions, action_ids)
    return {
        row["condition_id"]: index for index, row in enumerate(conditions)
    }, outcomes


def finite_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    """提取一个字段中的有限数值。"""

    values = []
    for row in rows:
        value = row.get(field, "")
        if value in {"", None}:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=np.float64)


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    """汇总连续数值。"""

    if len(values) == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def aggregate_decisions(
    analysis_mode: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """按场景和 update count 生成正式汇总。"""

    output: dict[str, Any] = {"overall": {}, "by_scenario": {}}
    output["overall"]["total_map_time_s"] = numeric_summary(
        finite_values(rows, "total_map_time_s")
    )
    output["overall"]["peak_cuda_memory_mib"] = numeric_summary(
        finite_values(rows, "peak_cuda_memory_mib")
    )
    output["overall"]["region_action_mismatch_rate"] = (
        float(
            np.mean(
                1.0
                - finite_values(rows, "region_top_matches_action_top")
            )
        )
        if rows
        else None
    )
    for scenario in e5_runner.SCENARIOS:
        subset = [row for row in rows if row["scenario"] == scenario]
        if not subset:
            continue
        scenario_payload: dict[str, Any] = {
            "decisions": len(subset),
            "total_map_time_s": numeric_summary(
                finite_values(subset, "total_map_time_s")
            ),
            "region_action_mismatch_rate": float(
                np.mean(
                    1.0
                    - finite_values(subset, "region_top_matches_action_top")
                )
            ),
        }
        if analysis_mode in {"known_condition", "belief_conditioned"}:
            method_groups = (
                {
                    method: [
                        row for row in subset if row["affordance_method"] == method
                    ]
                    for method in sorted(
                        {str(row["affordance_method"]) for row in subset}
                    )
                }
                if analysis_mode == "belief_conditioned"
                else {"cost_derived": subset}
            )
            method_payloads: dict[str, Any] = {}
            for method, method_rows in method_groups.items():
                method_payload: dict[str, Any] = {
                    "decisions": len(method_rows),
                    "region_action_mismatch_rate": float(
                        np.mean(
                            1.0
                            - finite_values(
                                method_rows, "region_top_matches_action_top"
                            )
                        )
                    ),
                    "top1_direct_mujoco_regret": numeric_summary(
                        finite_values(method_rows, "top1_direct_mujoco_regret")
                    ),
                }
                for k in K_VALUES:
                    method_payload[f"top{k}"] = {
                    "direct_mujoco_gap": numeric_summary(
                            finite_values(
                                method_rows, f"top{k}_direct_mujoco_gap"
                            )
                    ),
                    "exact_oracle_coverage": float(
                        np.mean(
                            finite_values(
                                    method_rows,
                                    f"top{k}_exact_mujoco_oracle_coverage",
                            )
                        )
                    ),
                    "near_0p05_rate": float(
                        np.mean(
                            finite_values(
                                    method_rows,
                                    f"top{k}_direct_mujoco_near_0p05",
                            )
                        )
                    ),
                    "near_0p10_rate": float(
                        np.mean(
                            finite_values(
                                    method_rows,
                                    f"top{k}_direct_mujoco_near_0p10",
                            )
                        )
                    ),
                    "retained_probability_mass": numeric_summary(
                        finite_values(
                                method_rows, f"top{k}_retained_probability_mass"
                        )
                    ),
                }
                method_payloads[method] = method_payload
            if analysis_mode == "known_condition":
                scenario_payload.update(method_payloads["cost_derived"])
            else:
                scenario_payload["by_method"] = method_payloads
        else:
            by_update: dict[str, Any] = {}
            for update_count in (0, 1, 2, 3, 4):
                update_rows = [
                    row
                    for row in subset
                    if int(row["valid_update_count_pre"]) == update_count
                ]
                if not update_rows:
                    continue
                by_update[str(update_count)] = {
                    "decisions": len(update_rows),
                    "region_action_mismatch_rate": float(
                        np.mean(
                            1.0
                            - finite_values(
                                update_rows, "region_top_matches_action_top"
                            )
                        )
                    ),
                    "retained_probability_mass_top100": numeric_summary(
                        finite_values(
                            update_rows, "top100_retained_probability_mass"
                        )
                    ),
                }
            scenario_payload["by_update_count"] = by_update
        output["by_scenario"][scenario] = scenario_payload
    return output


def write_report(
    path: Path,
    analysis_mode: str,
    role: str,
    aggregates: dict[str, Any],
) -> None:
    """写出简洁、可审计的 Markdown 结果报告。"""

    lines = [
        f"# Cost-Derived Affordance Map: {analysis_mode.replace('_', ' ').title()}",
        "",
        f"- Role: `{role}`",
        f"- Protocol: `{PROTOCOL_VERSION}`",
        f"- Temperature: `{LABEL_TEMPERATURE}`",
        "",
        "## Scenario Summary",
        "",
    ]
    if analysis_mode == "known_condition":
        lines.extend(
            [
                "| Scenario | Decisions | Top-1 mean regret | Top-20 exact coverage | Top-100 exact coverage | Top-100 near-0.05 | Region/action mismatch |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for scenario, payload in aggregates["by_scenario"].items():
            lines.append(
                f"| {scenario} | {payload['decisions']} | "
                f"{payload['top1_direct_mujoco_regret']['mean']:.6f} | "
                f"{payload['top20']['exact_oracle_coverage']:.2%} | "
                f"{payload['top100']['exact_oracle_coverage']:.2%} | "
                f"{payload['top100']['near_0p05_rate']:.2%} | "
                f"{payload['region_action_mismatch_rate']:.2%} |"
            )
    elif analysis_mode == "belief_conditioned":
        lines.extend(
            [
                "| Scenario | Affordance method | Decisions | Top-1 mean regret | Top-20 exact coverage | Top-100 exact coverage | Top-100 near-0.05 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for scenario, payload in aggregates["by_scenario"].items():
            for method, values in payload["by_method"].items():
                lines.append(
                    f"| {scenario} | {method} | {values['decisions']} | "
                    f"{values['top1_direct_mujoco_regret']['mean']:.6f} | "
                    f"{values['top20']['exact_oracle_coverage']:.2%} | "
                    f"{values['top100']['exact_oracle_coverage']:.2%} | "
                    f"{values['top100']['near_0p05_rate']:.2%} |"
                )
    else:
        lines.extend(
            [
                "| Scenario | Decisions | Region/action mismatch | Mean map time (s) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for scenario, payload in aggregates["by_scenario"].items():
            lines.append(
                f"| {scenario} | {payload['decisions']} | "
                f"{payload['region_action_mismatch_rate']:.2%} | "
                f"{payload['total_map_time_s']['mean']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The probabilities are a monotonic representation of predicted posterior-expected TNPO cost. They are not calibrated physical success probabilities and do not establish a global optimum outside the 4,536-action library.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行三种 cost-derived affordance-map 分析。"""

    device = require_device(args.device)
    scenarios = e5_runner.select_scenarios(args.scenario)
    action_ids, region_ids = e3_runner.load_action_layout()
    decision_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    episode_counts: dict[str, int] = {}
    for scenario in scenarios:
        if args.analysis_mode == "belief_conditioned":
            summaries, case_count = analyse_belief_conditioned_cases(
                scenario,
                args.role,
                action_ids,
                device,
                args.case_chunk_size,
                args.max_episodes,
            )
            decision_rows.extend(summaries)
            episode_counts[scenario] = case_count
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        engine = e5_runner.load_decision_engine(
            scenario, action_ids, device, args.node_query_chunk_size
        )
        episodes = read_episode_groups(
            args.role,
            scenario,
            args.target_group,
            args.max_episodes,
            args.require_update_four,
            args.episode_key_set,
        )
        episode_counts[scenario] = len(episodes)
        if args.analysis_mode == "known_condition":
            condition_index, direct_outcome_array = load_direct_outcomes(
                scenario, args.role, action_ids
            )
            for episode in episodes:
                row = episode[0]
                if int(row["push_index"]) != 1:
                    raise RuntimeError("known-condition evaluation 必须使用 initial decision")
                index = condition_index[row["condition_id"]]
                summary, full_rows = analyse_known_condition_episode(
                    engine,
                    row,
                    action_ids,
                    region_ids,
                    direct_outcome_array[index],
                    args.save_full_maps,
                )
                decision_rows.append(summary)
                map_rows.extend(full_rows)
        else:
            for episode in episodes:
                summaries, full_rows = analyse_closed_loop_episode(
                    engine,
                    episode,
                    action_ids,
                    region_ids,
                    args.max_decisions_per_episode,
                    args.save_full_maps,
                )
                decision_rows.extend(summaries)
                map_rows.extend(full_rows)
        del engine
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir = RESULT_ROOT / "evaluation" / args.role / args.analysis_mode
    if args.output_tag:
        output_dir = output_dir / args.output_tag
    decision_path = output_dir / "affordance_map_by_decision.csv"
    write_csv(decision_path, decision_rows, DECISION_FIELDS)
    if args.save_full_maps:
        write_csv(
            output_dir / "affordance_map_complete_actions.csv",
            map_rows,
            MAP_FIELDS,
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_mode": args.analysis_mode,
        "role": args.role,
        "scenarios": list(scenarios),
        "target_group": args.target_group,
        "output_tag": args.output_tag,
        "selected_episode_keys": sorted(args.episode_key_set or []),
        "label_temperature": LABEL_TEMPERATURE,
        "device": str(device),
        "episode_counts": episode_counts,
        "decision_count": len(decision_rows),
        "full_map_row_count": len(map_rows),
        "existing_e1_to_e5_artifacts_modified": False,
        "decision_summary": str(decision_path.resolve()),
    }
    aggregates = aggregate_decisions(args.analysis_mode, decision_rows)
    payload["aggregates"] = aggregates
    if decision_rows:
        payload["maximum_total_map_time_s"] = max(
            float(row["total_map_time_s"]) for row in decision_rows
        )
        payload["maximum_peak_cuda_memory_mib"] = max(
            float(row["peak_cuda_memory_mib"]) for row in decision_rows
        )
        if args.analysis_mode == "closed_loop":
            payload["maximum_posterior_replay_mean_error"] = max(
                max(
                    float(row["posterior_mean_max_abs_error_pre"]),
                    float(row["posterior_mean_max_abs_error_post"]),
                )
                for row in decision_rows
            )
            payload["maximum_posterior_replay_covariance_error"] = max(
                max(
                    float(row["posterior_covariance_trace_abs_error_pre"]),
                    float(row["posterior_covariance_trace_abs_error_post"]),
                )
                for row in decision_rows
            )
    write_json(output_dir / "summary.json", payload)
    write_report(
        output_dir / "report.md",
        args.analysis_mode,
        args.role,
        aggregates,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    """构造补充实验入口。"""

    parser = argparse.ArgumentParser(
        description="计算 HCR V2 单一 cost-derived affordance map。"
    )
    parser.add_argument(
        "--analysis-mode", choices=ANALYSIS_MODES, default="closed_loop"
    )
    parser.add_argument("--role", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--scenario", choices=(*e5_runner.SCENARIOS, "all"), default="friction"
    )
    parser.add_argument(
        "--target-group",
        choices=("core", "sequential_extension", "all"),
        default="sequential_extension",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1,
        help="0 表示读取全部符合条件的 episodes。",
    )
    parser.add_argument("--max-decisions-per-episode", type=int, default=5)
    parser.add_argument("--require-update-four", action="store_true")
    parser.add_argument("--save-full-maps", action="store_true")
    parser.add_argument(
        "--episode-keys",
        default="",
        help="逗号分隔的精确 episode keys；空值表示按其他条件读取。",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="在 analysis-mode 下创建独立输出子目录。",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--node-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--case-chunk-size", type=int, default=4)
    return parser


def main() -> None:
    """解析参数并执行补充实验。"""

    args = build_parser().parse_args()
    if args.max_episodes < 0:
        raise ValueError("max_episodes 不能小于 0")
    if args.max_decisions_per_episode <= 0:
        raise ValueError("max_decisions_per_episode 必须大于 0")
    if args.case_chunk_size <= 0:
        raise ValueError("case_chunk_size 必须大于 0")
    if args.analysis_mode == "known_condition":
        args.require_update_four = False
        args.max_decisions_per_episode = 1
    args.episode_key_set = {
        value.strip() for value in args.episode_keys.split(",") if value.strip()
    } or None
    run(args)


if __name__ == "__main__":
    main()
