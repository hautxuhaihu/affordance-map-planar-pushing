"""HCR V2 E4 的 GPU-first Top-K action-selection 实验入口。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_e3 as e3_runner
from push_core.hcr_v2.e3 import interpolate_outcome_grid, tnpo_costs, wrap_to_pi
from push_core.hcr_v2.e4 import (
    METHOD_IDS,
    PRIMARY_K,
    point_condition_candidate_scores,
    posterior_expected_candidate_scores,
    posterior_mean_hidden_parameters,
    ranking_diagnostics,
    select_minimum_actions,
)
from push_core.project_paths import HCR_V2_RESULTS_DIR


PROTOCOL_VERSION = "hcr_v2_e4_v1"
FRICTION_CONE = "elliptic"
SCENARIOS = ("friction", "com", "joint")
E4_RESULTS_ROOT = HCR_V2_RESULTS_DIR / "e4"
METHOD_NAMES = {
    "Method1": "Belief-Marginalised Proposal Top-1",
    "Method2": "Belief-Marginalised Candidates with Posterior-Mean Cost Selection",
    "Method3": "Belief-Marginalised Candidates with Posterior-Marginalised Expected-Cost Selection",
    "Method4": "Posterior-Mean Proposal and Posterior-Mean Cost Selection",
    "Method5": "True-Condition P1 Action Selection",
    "Method6": "Top-K MuJoCo Oracle Action Selection",
}
METHOD_CANDIDATE_SOURCE = {
    "Method1": "belief_marginalised",
    "Method2": "belief_marginalised",
    "Method3": "belief_marginalised",
    "Method4": "posterior_mean",
    "Method5": "belief_marginalised",
    "Method6": "belief_marginalised",
}
SUMMARY_METRICS = (
    "selected_tnpo_cost",
    "selection_gap",
    "total_gap",
    "near_optimal_0p10",
    "one_step_pose_success",
)


def select_scenarios(value: str) -> tuple[str, ...]:
    """解析单场景或 all。"""

    return SCENARIOS if value == "all" else (value,)


def load_candidate_artifact(
    scenario: str,
    role: str,
    metadata: list[dict[str, Any]],
    action_ids: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """读取并对齐 E3 的 Belief-Marginalised 与 Posterior-Mean Top-100。"""

    path = (
        e3_runner.E3_RESULTS_ROOT
        / "evaluation"
        / role
        / scenario
        / "top200_candidates.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        episode_keys = [str(value) for value in payload["episode_keys"]]
        artifact_action_ids = [str(value) for value in payload["action_ids"]]
        expected_episode_keys = [str(row["episode_key"]) for row in metadata]
        if episode_keys[: len(expected_episode_keys)] != expected_episode_keys:
            raise RuntimeError(f"E3 candidate artifact episode 顺序不一致: {path}")
        if artifact_action_ids != action_ids:
            raise RuntimeError(f"E3 candidate artifact action 顺序不一致: {path}")
        arrays = {
            "belief_marginalised_indices": torch.from_numpy(
                np.asarray(
                    payload["belief_marginalised_indices"][: len(metadata), :PRIMARY_K]
                )
            ).to(device=device, dtype=torch.long),
            "belief_marginalised_probabilities": torch.from_numpy(
                np.asarray(
                    payload["belief_marginalised_probabilities"][
                        : len(metadata), :PRIMARY_K
                    ]
                )
            ).to(device=device, dtype=torch.float32),
            "posterior_mean_indices": torch.from_numpy(
                np.asarray(
                    payload["posterior_mean_indices"][: len(metadata), :PRIMARY_K]
                )
            ).to(device=device, dtype=torch.long),
            "posterior_mean_probabilities": torch.from_numpy(
                np.asarray(
                    payload["posterior_mean_probabilities"][
                        : len(metadata), :PRIMARY_K
                    ]
                )
            ).to(device=device, dtype=torch.float32),
        }
    arrays["artifact_path"] = path
    return arrays


def slice_case_arrays(
    arrays: dict[str, np.ndarray],
    case_count: int,
) -> dict[str, np.ndarray]:
    """截取 benchmark 使用的前若干 decision cases。"""

    return {
        key: value[:case_count]
        if isinstance(value, np.ndarray) and value.ndim > 0
        else value
        for key, value in arrays.items()
    }


@torch.inference_mode()
def compute_selector_decisions(
    scenario: str,
    arrays: dict[str, np.ndarray],
    candidates: dict[str, Any],
    action_ids: list[str],
    device: torch.device,
    case_chunk_size: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """计算 Method1–Method5，Method6 在 true-reference 阶段计算。"""

    nodes, _, posterior_weights = e3_runner.replay_case_posteriors(
        scenario,
        arrays,
        action_ids,
        device,
    )
    task_queries = torch.from_numpy(arrays["task_queries"]).to(device)
    true_hidden = torch.from_numpy(arrays["true_hidden_parameters"]).to(device)
    p1_grid = e3_runner.load_p1_grid(scenario, action_ids, device)
    node_outcomes = interpolate_outcome_grid(p1_grid, nodes)
    posterior_mean = posterior_mean_hidden_parameters(nodes, posterior_weights)
    belief_candidates = candidates["belief_marginalised_indices"]
    posterior_mean_candidates = candidates["posterior_mean_indices"]

    combined_candidates = torch.cat(
        [belief_candidates, posterior_mean_candidates], dim=1
    )
    combined_mean_scores = point_condition_candidate_scores(
        p1_grid,
        posterior_mean,
        combined_candidates,
        task_queries,
        case_chunk_size,
    )
    method2_scores = combined_mean_scores[:, :PRIMARY_K]
    method4_scores = combined_mean_scores[:, PRIMARY_K:]
    method3_scores = posterior_expected_candidate_scores(
        node_outcomes,
        posterior_weights,
        belief_candidates,
        task_queries,
        case_chunk_size,
    )
    method5_scores = point_condition_candidate_scores(
        p1_grid,
        true_hidden,
        belief_candidates,
        task_queries,
        case_chunk_size,
    )

    score_tables = {
        "Method1": -candidates["belief_marginalised_probabilities"],
        "Method2": method2_scores,
        "Method3": method3_scores,
        "Method4": method4_scores,
        "Method5": method5_scores,
    }
    candidate_tables = {
        "Method1": belief_candidates,
        "Method2": belief_candidates,
        "Method3": belief_candidates,
        "Method4": posterior_mean_candidates,
        "Method5": belief_candidates,
    }
    selected_actions: dict[str, torch.Tensor] = {
        "Method1": belief_candidates[:, 0]
    }
    selected_scores: dict[str, torch.Tensor] = {
        "Method1": score_tables["Method1"][:, 0]
    }
    for method in ("Method2", "Method3", "Method4", "Method5"):
        selected_actions[method], selected_scores[method] = select_minimum_actions(
            candidate_tables[method], score_tables[method]
        )
    return (
        selected_actions,
        selected_scores,
        score_tables,
        posterior_weights,
        p1_grid,
    )


def empty_metric_store(case_count: int) -> dict[str, dict[str, np.ndarray]]:
    """建立每种方法的 case-level metric arrays。"""

    metrics = (
        "selected_action_index",
        "candidate_oracle_action_index",
        "library_oracle_action_index",
        "decision_score",
        "selected_tnpo_cost",
        "candidate_oracle_cost",
        "library_oracle_cost",
        "proposal_gap",
        "selection_gap",
        "total_gap",
        "near_optimal_0p10",
        "near_optimal_0p05",
        "near_optimal_0p02",
        "one_step_pose_success",
        "exact_action_id_top1_match",
        "cost_equivalent_top1_match",
        "candidate_oracle_cost_equivalent_match",
        "actual_position_error_m",
        "actual_yaw_error_rad",
        "p1_position_prediction_error_m",
        "p1_yaw_prediction_error_rad",
        "spearman_rank_correlation",
        "pairwise_ordering_accuracy",
    )
    return {
        method: {
            metric: np.full(case_count, np.nan, dtype=np.float64)
            for metric in metrics
        }
        for method in METHOD_IDS
    }


@torch.inference_mode()
def evaluate_true_reference(
    scenario: str,
    role: str,
    metadata: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    candidates: dict[str, Any],
    selected_actions: dict[str, torch.Tensor],
    selected_scores: dict[str, torch.Tensor],
    score_tables: dict[str, torch.Tensor],
    p1_grid: torch.Tensor,
    action_ids: list[str],
    device: torch.device,
    reference_batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    """使用 E1 MuJoCo outcomes 评价六种方法的真实 selected-action performance。"""

    case_count = len(metadata)
    store = empty_metric_store(case_count)
    conditions = e3_runner.load_conditions(scenario, role)
    condition_ids = [row["condition_id"] for row in conditions]
    condition_lookup = {
        condition_id: index for index, condition_id in enumerate(condition_ids)
    }
    condition_outcomes = torch.from_numpy(
        e3_runner.load_outcome_array(scenario, role, conditions, action_ids)
    ).to(device)
    case_conditions = torch.tensor(
        [condition_lookup[row["condition_id"]] for row in metadata],
        dtype=torch.long,
        device=device,
    )
    task_queries = torch.from_numpy(arrays["task_queries"]).to(device)
    true_hidden = torch.from_numpy(arrays["true_hidden_parameters"]).to(device)
    belief_candidates = candidates["belief_marginalised_indices"]
    posterior_mean_candidates = candidates["posterior_mean_indices"]
    selected_actions["Method6"] = torch.empty(
        case_count, dtype=torch.long, device=device
    )
    selected_scores["Method6"] = torch.empty(
        case_count, dtype=torch.float32, device=device
    )

    for start in range(0, case_count, reference_batch_size):
        stop = min(case_count, start + reference_batch_size)
        tasks = task_queries[start:stop]
        true_outcomes = condition_outcomes[case_conditions[start:stop]]
        true_costs = tnpo_costs(true_outcomes, tasks)
        library_cost, library_action = true_costs.min(dim=1)
        belief_indices = belief_candidates[start:stop]
        mean_indices = posterior_mean_candidates[start:stop]
        belief_true_costs = torch.gather(true_costs, 1, belief_indices)
        mean_true_costs = torch.gather(true_costs, 1, mean_indices)
        belief_oracle_action, belief_oracle_cost = select_minimum_actions(
            belief_indices, belief_true_costs
        )
        mean_oracle_action, mean_oracle_cost = select_minimum_actions(
            mean_indices, mean_true_costs
        )
        selected_actions["Method6"][start:stop] = belief_oracle_action
        selected_scores["Method6"][start:stop] = belief_oracle_cost

        true_condition_p1 = interpolate_outcome_grid(
            p1_grid,
            true_hidden[start:stop],
        )
        batch_score_tables = {
            method: scores[start:stop]
            for method, scores in score_tables.items()
        }
        batch_candidate_tables = {
            "Method1": belief_indices,
            "Method2": belief_indices,
            "Method3": belief_indices,
            "Method4": mean_indices,
            "Method5": belief_indices,
        }
        batch_true_candidate_costs = {
            "Method1": belief_true_costs,
            "Method2": belief_true_costs,
            "Method3": belief_true_costs,
            "Method4": mean_true_costs,
            "Method5": belief_true_costs,
        }

        for method in METHOD_IDS:
            actions = selected_actions[method][start:stop]
            selected_cost = torch.gather(true_costs, 1, actions[:, None])[:, 0]
            selected_outcome = torch.gather(
                true_outcomes,
                1,
                actions[:, None, None].expand(-1, 1, 3),
            )[:, 0]
            selected_p1_outcome = torch.gather(
                true_condition_p1,
                1,
                actions[:, None, None].expand(-1, 1, 3),
            )[:, 0]
            position_error = torch.linalg.vector_norm(
                selected_outcome[:, :2] - tasks[:, :2], dim=1
            )
            desired_yaw = torch.atan2(tasks[:, 2], tasks[:, 3])
            yaw_error = torch.abs(wrap_to_pi(selected_outcome[:, 2] - desired_yaw))
            p1_position_error = torch.linalg.vector_norm(
                selected_p1_outcome[:, :2] - selected_outcome[:, :2], dim=1
            )
            p1_yaw_error = torch.abs(
                wrap_to_pi(selected_p1_outcome[:, 2] - selected_outcome[:, 2])
            )

            if METHOD_CANDIDATE_SOURCE[method] == "posterior_mean":
                candidate_oracle_action = mean_oracle_action
                candidate_oracle_cost = mean_oracle_cost
            else:
                candidate_oracle_action = belief_oracle_action
                candidate_oracle_cost = belief_oracle_cost
            proposal_gap = candidate_oracle_cost - library_cost
            selection_gap = selected_cost - candidate_oracle_cost
            total_gap = selected_cost - library_cost
            values = {
                "selected_action_index": actions,
                "candidate_oracle_action_index": candidate_oracle_action,
                "library_oracle_action_index": library_action,
                "decision_score": selected_scores[method][start:stop],
                "selected_tnpo_cost": selected_cost,
                "candidate_oracle_cost": candidate_oracle_cost,
                "library_oracle_cost": library_cost,
                "proposal_gap": proposal_gap,
                "selection_gap": selection_gap,
                "total_gap": total_gap,
                "near_optimal_0p10": (total_gap <= 0.10).float(),
                "near_optimal_0p05": (total_gap <= 0.05).float(),
                "near_optimal_0p02": (total_gap <= 0.02).float(),
                "one_step_pose_success": (
                    (position_error <= 0.010)
                    & (yaw_error <= math.radians(5.0))
                ).float(),
                "exact_action_id_top1_match": (actions == library_action).float(),
                "cost_equivalent_top1_match": (
                    selected_cost <= library_cost + 1e-9
                ).float(),
                "candidate_oracle_cost_equivalent_match": (
                    selected_cost <= candidate_oracle_cost + 1e-9
                ).float(),
                "actual_position_error_m": position_error,
                "actual_yaw_error_rad": yaw_error,
                "p1_position_prediction_error_m": p1_position_error,
                "p1_yaw_prediction_error_rad": p1_yaw_error,
            }
            if method != "Method6":
                spearman, pairwise = ranking_diagnostics(
                    batch_score_tables[method],
                    batch_true_candidate_costs[method],
                    batch_candidate_tables[method],
                )
                values["spearman_rank_correlation"] = spearman
                values["pairwise_ordering_accuracy"] = pairwise
            for metric, tensor in values.items():
                store[method][metric][start:stop] = tensor.detach().cpu().numpy()
    return store


def descriptive_summary(values: np.ndarray) -> dict[str, float | int | None]:
    """返回论文表格使用的有限值描述统计。"""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
    }


def paired_two_way_bootstrap(
    metadata: list[dict[str, Any]],
    metrics: dict[str, dict[str, np.ndarray]],
    resamples: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """在 GPU 上对 condition 与 stratum-specific targets 执行 paired bootstrap。"""

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
        target_index = target_lookup[row["target_id"]]
        if target_index not in targets_by_stratum[key]:
            targets_by_stratum[key].append(target_index)

    columns = [
        (method, metric)
        for metric in SUMMARY_METRICS
        for method in METHOD_IDS
    ]
    value_matrix = torch.from_numpy(
        np.stack(
            [metrics[method][metric] for method, metric in columns], axis=1
        ).astype(np.float32)
    ).to(device)
    estimates = torch.empty(
        (resamples, len(columns)), dtype=torch.float32, device=device
    )
    case_condition_tensor = torch.from_numpy(case_condition).to(device)
    case_target_tensor = torch.from_numpy(case_target).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    chunk_size = 100
    for start in range(0, resamples, chunk_size):
        stop = min(resamples, start + chunk_size)
        size = stop - start
        condition_draws = torch.randint(
            len(condition_ids),
            (size, len(condition_ids)),
            generator=generator,
            device=device,
        )
        condition_counts = torch.zeros(
            (size, len(condition_ids)), dtype=torch.int32, device=device
        )
        condition_counts.scatter_add_(
            1,
            condition_draws,
            torch.ones_like(condition_draws, dtype=torch.int32),
        )
        target_counts = torch.zeros(
            (size, len(target_ids)), dtype=torch.int32, device=device
        )
        for target_indices in targets_by_stratum.values():
            stratum_targets = torch.tensor(
                target_indices, dtype=torch.long, device=device
            )
            draws = torch.randint(
                len(target_indices),
                (size, len(target_indices)),
                generator=generator,
                device=device,
            )
            sampled_targets = stratum_targets[draws]
            target_counts.scatter_add_(
                1,
                sampled_targets,
                torch.ones_like(sampled_targets, dtype=torch.int32),
            )
        weights = (
            condition_counts[:, case_condition_tensor]
            * target_counts[:, case_target_tensor]
        ).float()
        denominator = weights.sum(dim=1, keepdim=True)
        estimates[start:stop] = (weights @ value_matrix) / denominator

    column_lookup = {column: index for index, column in enumerate(columns)}
    method_intervals: dict[str, Any] = {}
    for method, metric in columns:
        samples = estimates[:, column_lookup[(method, metric)]]
        point = float(value_matrix[:, column_lookup[(method, metric)]].mean())
        method_intervals.setdefault(method, {})[metric] = {
            "point_estimate": point,
            "ci95_low": float(torch.quantile(samples, 0.025)),
            "ci95_high": float(torch.quantile(samples, 0.975)),
        }

    comparisons: dict[str, Any] = {}
    definitions = {
        "H1_Method3_vs_Method1": ("Method3", "Method1", "selection_gap"),
        "H2_Method3_vs_Method2": ("Method3", "Method2", "selection_gap"),
        "H3_Method3_vs_Method4": ("Method3", "Method4", "total_gap"),
    }
    for name, (primary, baseline, primary_metric) in definitions.items():
        comparison: dict[str, Any] = {}
        for metric in (
            primary_metric,
            "selected_tnpo_cost",
            "near_optimal_0p10",
            "one_step_pose_success",
        ):
            primary_samples = estimates[:, column_lookup[(primary, metric)]]
            baseline_samples = estimates[:, column_lookup[(baseline, metric)]]
            if metric in ("near_optimal_0p10", "one_step_pose_success"):
                samples = primary_samples - baseline_samples
                point = float(
                    value_matrix[:, column_lookup[(primary, metric)]].mean()
                    - value_matrix[:, column_lookup[(baseline, metric)]].mean()
                )
                direction = "primary_minus_baseline"
            else:
                samples = baseline_samples - primary_samples
                point = float(
                    value_matrix[:, column_lookup[(baseline, metric)]].mean()
                    - value_matrix[:, column_lookup[(primary, metric)]].mean()
                )
                direction = "baseline_minus_primary"
            comparison[metric] = {
                "effect_direction": direction,
                "point_estimate": point,
                "ci95_low": float(torch.quantile(samples, 0.025)),
                "ci95_high": float(torch.quantile(samples, 0.975)),
            }
        comparisons[name] = comparison
    return {
        "resamples": resamples,
        "seed": seed,
        "method_mean_confidence_intervals": method_intervals,
        "paired_comparisons": comparisons,
    }


def build_result_rows(
    metadata: list[dict[str, Any]],
    metrics: dict[str, dict[str, np.ndarray]],
    action_ids: list[str],
) -> list[dict[str, Any]]:
    """把 method arrays 转换为 case-level CSV rows。"""

    rows: list[dict[str, Any]] = []
    for method in METHOD_IDS:
        method_metrics = metrics[method]
        for case_index, case in enumerate(metadata):
            selected_index = int(method_metrics["selected_action_index"][case_index])
            candidate_oracle_index = int(
                method_metrics["candidate_oracle_action_index"][case_index]
            )
            library_oracle_index = int(
                method_metrics["library_oracle_action_index"][case_index]
            )
            row = {
                **case,
                "method_id": method,
                "method_name": METHOD_NAMES[method],
                "candidate_source": METHOD_CANDIDATE_SOURCE[method],
                "deployable": int(method in ("Method1", "Method2", "Method3", "Method4")),
                "selected_action_id": action_ids[selected_index],
                "candidate_oracle_action_id": action_ids[candidate_oracle_index],
                "library_oracle_action_id": action_ids[library_oracle_index],
            }
            for metric, values in method_metrics.items():
                value = values[case_index]
                row[metric] = None if not np.isfinite(value) else float(value)
            row["selected_action_index"] = selected_index
            row["candidate_oracle_action_index"] = candidate_oracle_index
            row["library_oracle_action_index"] = library_oracle_index
            rows.append(row)
    return rows


def summarise_methods(
    metrics: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, float]]:
    """汇总全部方法的正式指标，并单独返回误差分解。"""

    summaries = {
        method: {
            metric: descriptive_summary(values)
            for metric, values in method_metrics.items()
            if metric
            not in (
                "selected_action_index",
                "candidate_oracle_action_index",
                "library_oracle_action_index",
                "decision_score",
            )
        }
        for method, method_metrics in metrics.items()
    }
    error_decomposition = {
        "Method3_minus_Method5_mean_selection_gap": float(
            np.mean(metrics["Method3"]["selection_gap"])
            - np.mean(metrics["Method5"]["selection_gap"])
        ),
        "Method5_minus_Method6_mean_selection_gap": float(
            np.mean(metrics["Method5"]["selection_gap"])
            - np.mean(metrics["Method6"]["selection_gap"])
        ),
        "Method6_mean_proposal_gap": float(
            np.mean(metrics["Method6"]["proposal_gap"])
        ),
    }
    return summaries, error_decomposition


def paired_win_tie_loss(
    metrics: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float | int]]:
    """按真实 selected TNPO cost 汇总 Method3 的 paired win/tie/loss。"""

    comparisons = {
        "Method3_vs_Method1": "Method1",
        "Method3_vs_Method2": "Method2",
        "Method3_vs_Method4": "Method4",
    }
    primary_cost = metrics["Method3"]["selected_tnpo_cost"]
    summaries: dict[str, dict[str, float | int]] = {}
    for name, baseline in comparisons.items():
        difference = primary_cost - metrics[baseline]["selected_tnpo_cost"]
        wins = int(np.sum(difference < -1e-9))
        ties = int(np.sum(np.abs(difference) <= 1e-9))
        losses = int(np.sum(difference > 1e-9))
        count = int(difference.size)
        summaries[name] = {
            "n": count,
            "primary_win_count": wins,
            "tie_count": ties,
            "primary_loss_count": losses,
            "primary_win_rate": wins / count,
            "tie_rate": ties / count,
            "primary_loss_rate": losses / count,
            "tie_tolerance": 1e-9,
        }
    return summaries


def evaluate_scenario(
    scenario: str,
    role: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """运行一个场景的正式 E4 Validation 或 shared Test-split evaluation。"""

    action_ids, _ = e3_runner.load_action_layout()
    metadata, arrays = e3_runner.build_decision_cases(scenario, role, action_ids)
    candidates = load_candidate_artifact(
        scenario, role, metadata, action_ids, device
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    (
        selected_actions,
        selected_scores,
        score_tables,
        posterior_weights,
        p1_grid,
    ) = compute_selector_decisions(
        scenario,
        arrays,
        candidates,
        action_ids,
        device,
        args.case_chunk_size,
    )
    metrics = evaluate_true_reference(
        scenario,
        role,
        metadata,
        arrays,
        candidates,
        selected_actions,
        selected_scores,
        score_tables,
        p1_grid,
        action_ids,
        device,
        args.reference_batch_size,
    )
    torch.cuda.synchronize(device)
    selector_seconds = time.perf_counter() - started
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024.0**2)

    bootstrap_started = time.perf_counter()
    bootstrap = paired_two_way_bootstrap(
        metadata,
        metrics,
        args.bootstrap_resamples,
        args.bootstrap_seed,
        device,
    )
    bootstrap_seconds = time.perf_counter() - bootstrap_started
    output_root = E4_RESULTS_ROOT / "evaluation" / role / scenario
    rows = build_result_rows(metadata, metrics, action_ids)
    e3_runner.write_csv(output_root / "selected_actions.csv", rows, list(rows[0]))
    e3_runner.write_json(output_root / "bootstrap_summary.json", bootstrap)
    method_summaries, error_decomposition = summarise_methods(metrics)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "scenario": scenario,
        "role": role,
        "evaluation_wording": (
            "predefined shared E3/E4 Test-split evaluation"
            if role == "test"
            else "E4 Validation evaluation"
        ),
        "eligible_case_count": len(metadata),
        "excluded_one_push_count": int(arrays["excluded_one_push_count"]),
        "primary_k": PRIMARY_K,
        "methods": METHOD_NAMES,
        "method_summaries": method_summaries,
        "error_decomposition": error_decomposition,
        "paired_win_tie_loss": paired_win_tie_loss(metrics),
        "primary_hypothesis_endpoints": {
            "H1": "Method3 vs Method1 paired mean selection-gap improvement",
            "H2": "Method3 vs Method2 paired mean selection-gap improvement",
            "H3": "Method3 vs Method4 paired mean total-gap improvement",
        },
        "runtime": {
            "selector_and_reference_seconds": selector_seconds,
            "bootstrap_seconds": bootstrap_seconds,
            "total_seconds": selector_seconds + bootstrap_seconds,
            "case_chunk_size": args.case_chunk_size,
            "reference_batch_size": args.reference_batch_size,
            "peak_cuda_memory_mib": peak_memory_mib,
        },
        "posterior_shape": list(posterior_weights.shape),
        "candidate_artifact": str(candidates["artifact_path"]),
        "selected_actions_path": str((output_root / "selected_actions.csv").resolve()),
        "bootstrap_path": str((output_root / "bootstrap_summary.json").resolve()),
    }
    e3_runner.write_json(output_root / "summary.json", summary)
    print(f"{scenario} {role}: summary={output_root / 'summary.json'}")
    return summary


def evaluate_role(args: argparse.Namespace) -> dict[str, Any]:
    """顺序评价指定的一个或全部场景。"""

    device = e3_runner.require_cuda()
    summaries: dict[str, Any] = {}
    for scenario_index, scenario in enumerate(select_scenarios(args.scenario)):
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.bootstrap_seed = args.bootstrap_seed + scenario_index
        summaries[scenario] = evaluate_scenario(
            scenario, args.role, scenario_args, device
        )
    macro_average: dict[str, Any] = {}
    for method in METHOD_IDS:
        macro_average[method] = {}
        metric_names = summaries[next(iter(summaries))]["method_summaries"][method]
        for metric in metric_names:
            means = [
                summaries[scenario]["method_summaries"][method][metric]["mean"]
                for scenario in summaries
            ]
            macro_average[method][metric] = (
                float(np.mean(means)) if all(value is not None for value in means) else None
            )
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "friction_cone": FRICTION_CONE,
        "role": args.role,
        "evaluation_wording": (
            "predefined shared E3/E4 Test-split evaluation"
            if args.role == "test"
            else "E4 Validation evaluation"
        ),
        "scenarios": summaries,
        "equal_weight_macro_average": macro_average,
    }
    path = E4_RESULTS_ROOT / "evaluation" / args.role / "combined_summary.json"
    e3_runner.write_json(path, combined)
    print(f"Combined summary: {path.resolve()}")
    return combined


@torch.inference_mode()
def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """测量少量 Joint cases 的完整 E4 selector/reference GPU 路径。"""

    device = e3_runner.require_cuda()
    action_ids, _ = e3_runner.load_action_layout()
    metadata, arrays = e3_runner.build_decision_cases(
        args.scenario, "validation", action_ids
    )
    case_count = min(args.decision_cases, len(metadata))
    metadata = metadata[:case_count]
    arrays = slice_case_arrays(arrays, case_count)
    candidates = load_candidate_artifact(
        args.scenario, "validation", metadata, action_ids, device
    )
    for key, value in tuple(candidates.items()):
        if isinstance(value, torch.Tensor):
            candidates[key] = value[:case_count]

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    (
        selected_actions,
        selected_scores,
        score_tables,
        posterior_weights,
        p1_grid,
    ) = compute_selector_decisions(
        args.scenario,
        arrays,
        candidates,
        action_ids,
        device,
        args.case_chunk_size,
    )
    evaluate_true_reference(
        args.scenario,
        "validation",
        metadata,
        arrays,
        candidates,
        selected_actions,
        selected_scores,
        score_tables,
        p1_grid,
        action_ids,
        device,
        args.reference_batch_size,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "scenario": args.scenario,
        "device": torch.cuda.get_device_name(device),
        "cases": case_count,
        "nodes_per_case": int(posterior_weights.shape[1]),
        "candidate_count": PRIMARY_K,
        "candidate_node_scores": int(
            case_count * posterior_weights.shape[1] * PRIMARY_K
        ),
        "seconds": elapsed,
        "cases_per_second": case_count / elapsed,
        "case_chunk_size": args.case_chunk_size,
        "reference_batch_size": args.reference_batch_size,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
    }
    path = E4_RESULTS_ROOT / "benchmark" / f"{args.scenario}_summary.json"
    e3_runner.write_json(path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def add_evaluation_arguments(parser: argparse.ArgumentParser, role: str) -> None:
    """添加 Validation/Test-split 共享参数。"""

    parser.set_defaults(handler=evaluate_role, role=role)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), required=True)
    parser.add_argument("--case-chunk-size", type=int, default=64)
    parser.add_argument("--reference-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    default_seed = 2026081501 if role == "validation" else 2026081504
    parser.add_argument("--bootstrap-seed", type=int, default=default_seed)


def build_parser() -> argparse.ArgumentParser:
    """建立 E4 命令行接口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="测量少量 cases 的完整 GPU selector/reference 路径。"
    )
    benchmark_parser.add_argument("--scenario", choices=SCENARIOS, default="joint")
    benchmark_parser.add_argument("--decision-cases", type=int, default=64)
    benchmark_parser.add_argument("--case-chunk-size", type=int, default=64)
    benchmark_parser.add_argument("--reference-batch-size", type=int, default=64)
    benchmark_parser.set_defaults(handler=benchmark)

    validation = subparsers.add_parser(
        "evaluate-validation", help="运行正式 E4 Validation evaluation。"
    )
    add_evaluation_arguments(validation, "validation")
    test = subparsers.add_parser(
        "evaluate-test", help="运行 predefined shared E3/E4 Test-split evaluation。"
    )
    add_evaluation_arguments(test, "test")
    return parser


def main() -> None:
    """执行所选 E4 子命令。"""

    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
