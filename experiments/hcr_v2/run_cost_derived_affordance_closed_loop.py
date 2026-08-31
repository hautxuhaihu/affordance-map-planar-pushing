"""运行并评价 cost-derived full-library affordance closed-loop controller。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
from push_core.hcr_v2.e3 import ACTION_COUNT, LABEL_TEMPERATURE, action_soft_labels
from push_core.hcr_v2.e4 import (
    posterior_expected_candidate_scores,
    select_minimum_actions,
)
from push_core.hcr_v2.e5 import ActionDecision, ClosedLoopDecisionEngine
from push_core.project_paths import HCR_V2_DATA_DIR, HCR_V2_RESULTS_DIR


PROTOCOL_VERSION = "cost_derived_affordance_closed_loop_v1"
CONTROLLER_ID = "cost_derived_affordance_closed_loop"
CONTROLLER_NAME = "Cost-Derived Affordance Closed-Loop Controller"
DATA_ROOT = HCR_V2_DATA_DIR / "affordance_map_supplementary" / "closed_loop"
RESULT_ROOT = HCR_V2_RESULTS_DIR / "affordance_map_supplementary" / "closed_loop"
BOOTSTRAP_RESAMPLES = 10_000
PROCESS_WORKER_CONTEXT: dict[str, Any] = {}


class CostDerivedDecisionEngine(ClosedLoopDecisionEngine):
    """在完整动作库上直接最小化 posterior-expected TNPO cost。"""

    @torch.inference_mode()
    def select_action(
        self,
        controller_id: str,
        task_query: np.ndarray,
        belief,
        true_hidden_parameters: np.ndarray,
    ) -> ActionDecision:
        """复用 E5 接口，但不调用 learned proposal。"""

        if controller_id != e5_runner.BELIEF_MARGINALISED_CONTROLLER_ID:
            raise ValueError(f"内部 controller id 错误: {controller_id}")
        raw_task = torch.as_tensor(
            np.asarray(task_query, dtype=np.float32)[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        posterior_weights = belief.weights[None, :]
        candidates = torch.arange(
            ACTION_COUNT, dtype=torch.long, device=self.device
        )[None, :]

        self._synchronise()
        proposal_start = time.perf_counter()
        scores = posterior_expected_candidate_scores(
            self.node_outcomes,
            posterior_weights,
            candidates,
            raw_task,
            case_chunk_size=1,
        )
        probabilities = action_soft_labels(scores, LABEL_TEMPERATURE)
        self._synchronise()
        proposal_latency = time.perf_counter() - proposal_start

        selection_start = time.perf_counter()
        selected, selected_scores = select_minimum_actions(candidates, scores)
        selected_index = int(selected.item())
        selected_probability = float(probabilities[0, selected_index])
        self._synchronise()
        selection_latency = time.perf_counter() - selection_start
        return ActionDecision(
            action_index=selected_index,
            decision_score=float(selected_scores.item()),
            proposal_probability=selected_probability,
            candidate_count=ACTION_COUNT,
            candidate_source="cost_derived_full_library",
            proposal_latency_s=proposal_latency,
            selection_latency_s=selection_latency,
        )


def load_engine(
    scenario: str,
    action_ids: list[str],
    device: torch.device,
    node_query_chunk_size: int,
) -> CostDerivedDecisionEngine:
    """加载 E1–E4 产物并构造隔离 controller engine。"""

    base = e5_runner.load_decision_engine(
        scenario, action_ids, device, node_query_chunk_size
    )
    return CostDerivedDecisionEngine(
        scenario=base.scenario,
        action_ids=list(base.action_ids),
        outcome_grid=base.outcome_grid,
        nodes=base.nodes,
        prior_weights=base.prior_weights,
        node_outcomes=base.node_outcomes,
        proposal_model=base.proposal_model,
        task_normaliser=base.task_normaliser,
        residual_bias_standardised=base.residual_bias_standardised,
        precision=base.precision,
        device=base.device,
        node_query_chunk_size=base.node_query_chunk_size,
    )


def rewrite_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把内部复用的 E5 controller 标识替换为补充实验标识。"""

    old_id = e5_runner.BELIEF_MARGINALISED_CONTROLLER_ID
    for row in rows:
        row["experiment_id"] = "AFFORDANCE_MAP_PHASE_B"
        row["protocol_version"] = PROTOCOL_VERSION
        row["controller_name"] = CONTROLLER_NAME
        row["controller_id"] = CONTROLLER_ID
        row["episode_key"] = row["episode_key"].replace(old_id, CONTROLLER_ID)
    return rows


def inspect_complete_shard(
    path: Path,
    target_ids: set[str],
    maximum_pushes: int,
    role: str,
) -> bool:
    """检查隔离 condition shard 是否完整。"""

    if not path.exists():
        return False
    rows = e5_runner.read_csv_rows(path)
    terminal = [row for row in rows if int(row["is_terminal_push"]) == 1]
    return (
        {row["target_id"] for row in terminal} == target_ids
        and len(terminal) == len(target_ids)
        and {row["controller_id"] for row in rows} == {CONTROLLER_ID}
        and {row["protocol_version"] for row in rows} == {PROTOCOL_VERSION}
        and {row["role"] for row in rows} == {role}
        and {int(row["maximum_push_budget"]) for row in rows}
        == {maximum_pushes}
    )


def summarise_condition(
    rows: list[dict[str, Any]] | list[dict[str, str]],
    scenario: str,
    condition_id: str,
    path: Path,
    resumed: int,
) -> dict[str, Any]:
    """汇总一个补充 condition shard。"""

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


def collect_condition(
    scenario: str,
    condition: dict[str, str],
    condition_index: int,
    condition_count: int,
    engine: CostDerivedDecisionEngine,
    environment_xml: Path,
    output_path: Path,
    targets: list[dict[str, str]],
    action_rows_by_id: dict[str, dict[str, str]],
    role: str,
    maximum_pushes: int,
) -> dict[str, Any]:
    """在一个独立 MuJoCo state 中采集 cost-derived controller。"""

    cuda_stream = torch.cuda.Stream(device=engine.device)
    with torch.cuda.stream(cuda_stream):
        model, data = e5_runner.load_model(environment_xml)
        e5_runner.set_sliding_friction(
            model, float(condition["friction_sliding_mu"])
        )
        object_body_id = e5_runner.get_body_id(model)
        rows: list[dict[str, Any]] = []
        for target_index, target in enumerate(targets):
            episode_id = (
                int(condition["condition_index_within_role"]) * 100_000
                + target_index * 10
            )
            episode_rows = e5_runner.run_closed_loop_episode(
                model=model,
                data=data,
                object_body_id=object_body_id,
                engine=engine,
                controller_id=e5_runner.BELIEF_MARGINALISED_CONTROLLER_ID,
                condition=condition,
                target=target,
                action_rows_by_id=action_rows_by_id,
                episode_id=episode_id,
                environment_xml=environment_xml,
                role=role,
                maximum_pushes=maximum_pushes,
            )
            rows.extend(rewrite_episode_rows(episode_rows))
        cuda_stream.synchronize()
    e5_runner.write_csv(output_path, rows, e5_runner.STEP_FIELDS)
    result = summarise_condition(
        rows, scenario, condition["condition_id"], output_path, resumed=0
    )
    result["condition_index"] = condition_index
    result["condition_count"] = condition_count
    return result


def initialise_process_worker(
    scenario: str,
    targets: list[dict[str, str]],
    role: str,
    maximum_pushes: int,
    node_query_chunk_size: int,
) -> None:
    """为 Windows spawn worker 加载一次场景级 GPU artifacts。"""

    torch.set_num_threads(1)
    device = e3_runner.require_cuda()
    action_rows = e5_runner.load_actions(e5_runner.ACTION_MANIFEST_PATH)
    action_ids, _ = e3_runner.load_action_layout()
    engine = load_engine(scenario, action_ids, device, node_query_chunk_size)
    PROCESS_WORKER_CONTEXT.clear()
    PROCESS_WORKER_CONTEXT.update(
        {
            "scenario": scenario,
            "targets": targets,
            "role": role,
            "maximum_pushes": maximum_pushes,
            "engine": engine,
            "action_rows_by_id": {
                row["v2_action_id"]: row for row in action_rows
            },
        }
    )


def collect_process_task(
    task: tuple[int, int, dict[str, str], str, str],
) -> dict[str, Any]:
    """在已初始化的独立进程中采集一个 condition shard。"""

    condition_index, condition_count, condition, xml_text, output_text = task
    context = PROCESS_WORKER_CONTEXT
    return collect_condition(
        scenario=context["scenario"],
        condition=condition,
        condition_index=condition_index,
        condition_count=condition_count,
        engine=context["engine"],
        environment_xml=Path(xml_text),
        output_path=Path(output_text),
        targets=context["targets"],
        action_rows_by_id=context["action_rows_by_id"],
        role=context["role"],
        maximum_pushes=context["maximum_pushes"],
    )


def select_targets(role: str, target_group: str) -> list[dict[str, str]]:
    """读取并筛选 E5 frozen targets。"""

    targets = e5_runner.load_e5_targets(role)
    if target_group == "all":
        return targets
    return [row for row in targets if row["target_group"] == target_group]


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """采集独立 Phase B Validation 或 Test trajectories。"""

    if not torch.cuda.is_available():
        raise RuntimeError("Phase B collection 需要 CUDA")
    device = (
        e3_runner.require_cuda()
        if args.worker_mode == "thread"
        else torch.device("cuda")
    )
    scenarios = e5_runner.select_scenarios(args.scenario)
    targets = select_targets(args.role, args.target_group)
    target_ids = {row["v2_target_id"] for row in targets}
    action_rows = e5_runner.load_actions(e5_runner.ACTION_MANIFEST_PATH)
    action_rows_by_id = {row["v2_action_id"]: row for row in action_rows}
    action_ids, _ = e3_runner.load_action_layout()
    condition_results: list[dict[str, Any]] = []
    configured_workers_by_scenario: dict[str, int] = {}

    for scenario in scenarios:
        engine = None
        if args.worker_mode == "thread":
            engine = load_engine(
                scenario, action_ids, device, args.node_query_chunk_size
            )
        conditions = e3_runner.load_conditions(scenario, args.role)
        if args.max_conditions > 0:
            conditions = conditions[: args.max_conditions]
        configured_workers_by_scenario[scenario] = min(
            1 if scenario == "joint" else args.num_workers,
            len(conditions),
        )
        generated_dir = DATA_ROOT / "generated_xml" / args.role / scenario
        xml_by_com = e5_runner.prepare_environment_xmls(conditions, generated_dir)
        pending: list[tuple[int, dict[str, str], Path, Path]] = []
        for condition_index, condition in enumerate(conditions, start=1):
            output_path = (
                DATA_ROOT
                / args.role
                / args.target_group
                / scenario
                / f"{condition['condition_id']}.csv"
            )
            if args.resume and inspect_complete_shard(
                output_path, target_ids, args.maximum_pushes, args.role
            ):
                condition_results.append(
                    summarise_condition(
                        e5_runner.read_csv_rows(output_path),
                        scenario,
                        condition["condition_id"],
                        output_path,
                        resumed=1,
                    )
                )
                continue
            com_key = (
                round(float(condition["com_offset_x_m"]), 9),
                round(float(condition["com_offset_y_m"]), 9),
            )
            pending.append(
                (condition_index, condition, xml_by_com[com_key], output_path)
            )
        requested_workers = 1 if scenario == "joint" else args.num_workers
        worker_count = min(requested_workers, len(pending))
        if worker_count > 0:
            if args.worker_mode == "process":
                executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=mp.get_context("spawn"),
                    initializer=initialise_process_worker,
                    initargs=(
                        scenario,
                        targets,
                        args.role,
                        args.maximum_pushes,
                        args.node_query_chunk_size,
                    ),
                )
                futures = {
                    executor.submit(
                        collect_process_task,
                        (
                            condition_index,
                            len(conditions),
                            condition,
                            str(environment_xml),
                            str(output_path),
                        ),
                    ): condition_index
                    for condition_index, condition, environment_xml, output_path in pending
                }
            else:
                executor = ThreadPoolExecutor(max_workers=worker_count)
                futures = {
                    executor.submit(
                        collect_condition,
                        scenario,
                        condition,
                        condition_index,
                        len(conditions),
                        engine,
                        environment_xml,
                        output_path,
                        targets,
                        action_rows_by_id,
                        args.role,
                        args.maximum_pushes,
                    ): condition_index
                    for condition_index, condition, environment_xml, output_path in pending
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

    condition_results.sort(key=lambda row: (row["scenario"], row["condition_id"]))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": args.role,
        "scenarios": list(scenarios),
        "condition_worker_type": args.worker_mode,
        "requested_num_workers": args.num_workers,
        "configured_workers_by_scenario": configured_workers_by_scenario,
        "device": torch.cuda.get_device_name(device),
        "target_group": args.target_group,
        "targets_per_condition": len(targets),
        "conditions": len(condition_results),
        "episodes": sum(int(row["episodes"]) for row in condition_results),
        "step_rows": sum(int(row["step_rows"]) for row in condition_results),
        "successful_episodes": sum(
            int(row["successful_episodes"]) for row in condition_results
        ),
        "invalid_episodes": sum(
            int(row["invalid_episodes"]) for row in condition_results
        ),
        "resumed_conditions": sum(int(row["resumed"]) for row in condition_results),
        "existing_e1_to_e5_artifacts_modified": False,
        "condition_results": condition_results,
    }
    write_path = RESULT_ROOT / "collection" / args.role / args.target_group / "summary.json"
    e5_runner.write_json(write_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def load_supplementary_rows(
    role: str, target_group: str, scenario: str
) -> list[dict[str, str]]:
    """读取一个场景的 Phase B trajectories。"""

    root = DATA_ROOT / role / target_group / scenario
    if not root.exists() and target_group != "all":
        root = DATA_ROOT / role / "all" / scenario
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("*.csv")):
        for row in e5_runner.read_csv_rows(path):
            if target_group != "all" and row["target_group"] != target_group:
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"缺少 Phase B rows: {root}")
    return rows


def load_comparator_rows(
    role: str, target_group: str, scenario: str
) -> list[dict[str, str]]:
    """读取匹配的 existing E5 belief-marginalised comparator rows。"""

    root = e5_runner.E5_DATA_ROOT / "closed_loop" / role / scenario
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("*.csv")):
        for row in e5_runner.read_csv_rows(path):
            if row["controller_id"] != e5_runner.BELIEF_MARGINALISED_CONTROLLER_ID:
                continue
            if target_group != "all" and row["target_group"] != target_group:
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"缺少 E5 comparator rows: {root}")
    return rows


def terminal_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """提取每个 episode 的 terminal row。"""

    return [row for row in rows if int(row["is_terminal_push"]) == 1]


def pair_cases(
    cost_rows: list[dict[str, str]],
    comparator_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """按 condition-target 对齐两个 controller。"""

    cost_by_case = {
        (row["condition_id"], row["target_id"]): row
        for row in terminal_rows(cost_rows)
    }
    comparator_by_case = {
        (row["condition_id"], row["target_id"]): row
        for row in terminal_rows(comparator_rows)
    }
    if set(cost_by_case) != set(comparator_by_case):
        raise RuntimeError("Phase B 与 E5 comparator cases 不一致")
    paired = []
    for condition_id, target_id in sorted(cost_by_case):
        cost = cost_by_case[(condition_id, target_id)]
        comparator = comparator_by_case[(condition_id, target_id)]
        cost_success = int(cost["episode_success"])
        comparator_success = int(comparator["episode_success"])
        cost_pushes = int(cost["terminal_push_count"])
        comparator_pushes = int(comparator["terminal_push_count"])
        cost_auc = (
            (e5_runner.MAXIMUM_PUSHES + 1 - cost_pushes)
            / e5_runner.MAXIMUM_PUSHES
            if cost_success
            else 0.0
        )
        comparator_auc = (
            (e5_runner.MAXIMUM_PUSHES + 1 - comparator_pushes)
            / e5_runner.MAXIMUM_PUSHES
            if comparator_success
            else 0.0
        )
        paired.append(
            {
                "condition_id": condition_id,
                "target_id": target_id,
                "target_stratum": cost["target_stratum"],
                "cost_derived_success": cost_success,
                "comparator_success": comparator_success,
                "success_difference": cost_success - comparator_success,
                "cost_derived_final_cost": float(cost["actual_tnpo_cost"]),
                "comparator_final_cost": float(comparator["actual_tnpo_cost"]),
                "final_cost_improvement": float(comparator["actual_tnpo_cost"])
                - float(cost["actual_tnpo_cost"]),
                "cost_derived_pushes": cost_pushes,
                "comparator_pushes": comparator_pushes,
                "cost_derived_auc": cost_auc,
                "comparator_auc": comparator_auc,
                "auc_difference": cost_auc - comparator_auc,
            }
        )
    return paired


def paired_bootstrap(
    rows: list[dict[str, Any]], resamples: int, seed: int
) -> dict[str, Any]:
    """执行 condition 与 target-stratum 两向配对 bootstrap。"""

    conditions = sorted({row["condition_id"] for row in rows})
    targets = sorted({row["target_id"] for row in rows})
    condition_slot = {value: index for index, value in enumerate(conditions)}
    target_slot = {value: index for index, value in enumerate(targets)}
    shape = (len(conditions), len(targets))
    metrics = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in ("success_difference", "final_cost_improvement", "auc_difference")
    }
    strata_by_target: dict[str, str] = {}
    for row in rows:
        i = condition_slot[row["condition_id"]]
        j = target_slot[row["target_id"]]
        for name in metrics:
            metrics[name][i, j] = float(row[name])
        strata_by_target[row["target_id"]] = row["target_stratum"]
    if any(np.isnan(values).any() for values in metrics.values()):
        raise RuntimeError("paired bootstrap matrix 存在缺失 cases")
    strata = sorted(set(strata_by_target.values()))
    stratum_indices = {
        stratum: np.asarray(
            [
                index
                for index, target_id in enumerate(targets)
                if strata_by_target[target_id] == stratum
            ],
            dtype=np.int64,
        )
        for stratum in strata
    }
    rng = np.random.default_rng(seed)
    samples = {
        name: np.empty(resamples, dtype=np.float64) for name in metrics
    }
    for sample_index in range(resamples):
        condition_counts = np.bincount(
            rng.integers(0, len(conditions), size=len(conditions)),
            minlength=len(conditions),
        ).astype(np.float64)
        target_counts = np.zeros(len(targets), dtype=np.float64)
        for indices in stratum_indices.values():
            draws = rng.integers(0, len(indices), size=len(indices))
            target_counts[indices] = np.bincount(draws, minlength=len(indices))
        weights = condition_counts[:, None] * target_counts[None, :]
        denominator = float(weights.sum())
        for name, values in metrics.items():
            samples[name][sample_index] = float(
                np.sum(weights * values) / denominator
            )
    return {
        name: {
            "point_estimate": float(np.mean(values)),
            "ci_95_low": float(np.quantile(samples[name], 0.025)),
            "ci_95_high": float(np.quantile(samples[name], 0.975)),
            "positive_effect_probability": float(np.mean(samples[name] > 0.0)),
            "resamples": resamples,
            "seed": seed,
        }
        for name, values in metrics.items()
    }


def cost_win_tie_loss(rows: list[dict[str, Any]]) -> dict[str, int]:
    """按 final TNPO cost 统计 cost-derived 相对 comparator 的胜负。"""

    wins = ties = losses = 0
    for row in rows:
        difference = row["cost_derived_final_cost"] - row["comparator_final_cost"]
        if difference < -1e-12:
            wins += 1
        elif difference > 1e-12:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "ties": ties, "losses": losses}


def write_paired_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 paired case-level result。"""

    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_report(path: Path, combined: dict[str, Any]) -> None:
    """写出便于论文核对的 Phase B Markdown 结果表。"""

    lines = [
        "**Cost-Derived Affordance Closed-Loop Evaluation**",
        "",
        f"Dataset role: `{combined['role']}`. Each scenario uses paired episodes "
        "with the existing `belief_marginalised_closed_loop` controller.",
        "",
        "| Scenario | Episodes | Success | Mean pushes | Mean final cost | Comparator cost | Cost improvement (95% CI) | AUC difference (95% CI) | Win / tie / loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario, summary in combined["scenarios"].items():
        cost = summary["cost_derived"]
        comparator = summary["existing_belief_marginalised"]
        cost_effect = summary["paired_effects"]["final_cost_improvement"]
        auc_effect = summary["paired_effects"]["auc_difference"]
        counts = summary["final_cost_win_tie_loss"]
        lines.append(
            f"| {scenario.capitalize()} | {cost['episodes']} | "
            f"{100.0 * cost['episode_success_rate']:.2f}% | "
            f"{cost['pushes_to_success']['mean']:.4f} | "
            f"{cost['final_tnpo_cost']['mean']:.6f} | "
            f"{comparator['final_tnpo_cost']['mean']:.6f} | "
            f"{cost_effect['point_estimate']:.6f} "
            f"[{cost_effect['ci_95_low']:.6f}, {cost_effect['ci_95_high']:.6f}] | "
            f"{auc_effect['point_estimate']:.6f} "
            f"[{auc_effect['ci_95_low']:.6f}, {auc_effect['ci_95_high']:.6f}] | "
            f"{counts['wins']} / {counts['ties']} / {counts['losses']} |"
        )
    lines.extend(
        [
            "",
            "Cost improvement is defined as comparator final cost minus cost-derived "
            "final cost. The confidence intervals use the fixed condition-by-target "
            "two-way paired bootstrap specified by the evaluation protocol.",
            "",
            "The cost-derived affordance controller preserves closed-loop task "
            "completion while making the probability map and final action ordering "
            "the same cost-derived calculation. Confidence intervals containing zero "
            "are not interpreted as evidence of improved task performance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """评价 Phase B 并与 existing E5 controller 配对比较。"""

    scenarios = e5_runner.select_scenarios(args.scenario)
    summaries: dict[str, Any] = {}
    for scenario_index, scenario in enumerate(scenarios):
        cost_rows = load_supplementary_rows(args.role, args.target_group, scenario)
        comparator_rows = load_comparator_rows(args.role, args.target_group, scenario)
        cost_terminal = terminal_rows(cost_rows)
        comparator_terminal = terminal_rows(comparator_rows)
        paired = pair_cases(cost_rows, comparator_rows)
        summary = {
            "scenario": scenario,
            "role": args.role,
            "target_group": args.target_group,
            "cost_derived": e5_runner.controller_summary(cost_terminal, cost_rows),
            "existing_belief_marginalised": e5_runner.controller_summary(
                comparator_terminal, comparator_rows
            ),
            "paired_effects": paired_bootstrap(
                paired,
                args.bootstrap_resamples,
                args.bootstrap_seed + scenario_index,
            ),
            "final_cost_win_tie_loss": cost_win_tie_loss(paired),
        }
        output_dir = RESULT_ROOT / "evaluation" / args.role / args.target_group / scenario
        e5_runner.write_json(output_dir / "summary.json", summary)
        write_paired_csv(output_dir / "paired_cases.csv", paired)
        summaries[scenario] = summary
    combined = {
        "protocol_version": PROTOCOL_VERSION,
        "role": args.role,
        "target_group": args.target_group,
        "bootstrap_resamples": args.bootstrap_resamples,
        "scenarios": summaries,
    }
    output_path = RESULT_ROOT / "evaluation" / args.role / args.target_group / "summary.json"
    e5_runner.write_json(output_path, combined)
    write_evaluation_report(output_path.with_name("report.md"), combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return combined


def build_parser() -> argparse.ArgumentParser:
    """构造 Phase B 命令行入口。"""

    parser = argparse.ArgumentParser(
        description="运行 cost-derived affordance full-library closed-loop controller。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collection = subparsers.add_parser("collect")
    collection.add_argument("--role", choices=("validation", "test"), required=True)
    collection.add_argument(
        "--scenario", choices=(*e5_runner.SCENARIOS, "all"), default="all"
    )
    collection.add_argument(
        "--target-group",
        choices=("core", "sequential_extension", "all"),
        default="all",
    )
    collection.add_argument("--maximum-pushes", type=int, default=20)
    collection.add_argument("--num-workers", type=int, default=2)
    collection.add_argument(
        "--worker-mode", choices=("process", "thread"), default="process"
    )
    collection.add_argument("--max-conditions", type=int, default=0)
    collection.add_argument("--node-query-chunk-size", type=int, default=65_536)
    collection.add_argument("--resume", action="store_true")
    collection.set_defaults(handler=collect)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--role", choices=("validation", "test"), required=True)
    evaluation.add_argument(
        "--scenario", choices=(*e5_runner.SCENARIOS, "all"), default="all"
    )
    evaluation.add_argument(
        "--target-group",
        choices=("core", "sequential_extension", "all"),
        default="all",
    )
    evaluation.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    evaluation.add_argument("--bootstrap-seed", type=int, default=2026082900)
    evaluation.set_defaults(handler=evaluate)
    return parser


def main() -> None:
    """解析参数并执行 Phase B。"""

    args = build_parser().parse_args()
    if hasattr(args, "num_workers") and args.num_workers <= 0:
        raise ValueError("num_workers 必须大于 0")
    if hasattr(args, "maximum_pushes") and args.maximum_pushes <= 0:
        raise ValueError("maximum_pushes 必须大于 0")
    args.handler(args)


if __name__ == "__main__":
    main()
