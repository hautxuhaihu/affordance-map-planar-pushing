"""从 Phase A CSV 绘制单一 cost-derived affordance map。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT
RESULT_ROOT = PROJECT_ROOT / "results" / "hcr_v2" / "affordance_map_supplementary"
ACTION_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "manifests"
    / "hcr_v2"
    / "hcr_v2_action_core_manifest_v1.csv"
)
REGION_COUNT = 12
ACTIONS_PER_REGION = 378
ACTION_COUNT = REGION_COUNT * ACTIONS_PER_REGION


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 UTF-8 CSV。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_action_rows() -> list[dict[str, str]]:
    """按 HCR V2 12×378 布局排列 action manifest。"""

    rows = read_csv(ACTION_MANIFEST_PATH)
    region_ids = sorted({int(row["contact_region_id"]) for row in rows})
    region_to_index = {region_id: index for index, region_id in enumerate(region_ids)}
    ordered: list[dict[str, str] | None] = [None] * ACTION_COUNT
    for row in rows:
        flat_index = (
            region_to_index[int(row["contact_region_id"])] * ACTIONS_PER_REGION
            + int(row["action_param_index"])
        )
        ordered[flat_index] = row
    if any(row is None for row in ordered):
        raise RuntimeError("action manifest 不是完整的 12×378 layout")
    return [row for row in ordered if row is not None]


def group_decisions(
    rows: list[dict[str, str]],
) -> list[tuple[tuple[str, int], list[dict[str, str]]]]:
    """按 episode 和 push index 分组。"""

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["episode_key"], int(row["push_index"]))].append(row)
    return sorted(grouped.items(), key=lambda item: item[0])


def filename_fragment(value: str) -> str:
    """生成安全文件名片段。"""

    return "".join(character if character.isalnum() else "_" for character in value)


def decision_arrays(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """从一个 decision 的 4,536 rows 恢复完整与 region probabilities。"""

    if len(rows) != ACTION_COUNT:
        raise RuntimeError(f"decision rows 数量错误: {len(rows)}")
    probabilities = np.zeros(ACTION_COUNT, dtype=np.float64)
    for row in rows:
        probabilities[int(row["action_index"])] = float(
            row["complete_action_probability"]
        )
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
        raise RuntimeError("affordance probability sum 不是 1")
    region_probabilities = probabilities.reshape(
        REGION_COUNT, ACTIONS_PER_REGION
    ).sum(axis=1)
    return probabilities, region_probabilities, rows[0]


def region_positions(action_rows: list[dict[str, str]]) -> np.ndarray:
    """读取 12 个 region marker 的 object-local positions。"""

    return np.asarray(
        [
            [
                float(action_rows[index * ACTIONS_PER_REGION]["contact_point_local_x"]),
                float(action_rows[index * ACTIONS_PER_REGION]["contact_point_local_y"]),
            ]
            for index in range(REGION_COUNT)
        ],
        dtype=np.float64,
    )


def draw_map(
    axis,
    probabilities: np.ndarray,
    region_probability: np.ndarray,
    metadata: dict[str, str],
    action_rows: list[dict[str, str]],
    positions: np.ndarray,
    colour_maximum: float,
    title: str,
):
    """在一个 axis 上绘制 cost-derived affordance map。"""

    scatter = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=region_probability,
        s=110.0 + 1100.0 * region_probability,
        cmap="viridis",
        vmin=0.0,
        vmax=colour_maximum,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )
    for region_index, (x_value, y_value) in enumerate(positions):
        axis.text(
            x_value,
            y_value,
            str(region_index),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if region_probability[region_index] > 0.08 else "black",
            zorder=4,
        )
    extent_x = float(np.max(np.abs(positions[:, 0])))
    extent_y = float(np.max(np.abs(positions[:, 1])))
    axis.add_patch(
        plt.Rectangle(
            (-extent_x, -extent_y),
            2.0 * extent_x,
            2.0 * extent_y,
            fill=False,
            color="0.25",
            linewidth=1.5,
            zorder=1,
        )
    )
    top_action = action_rows[int(np.argmax(probabilities))]
    axis.arrow(
        float(top_action["contact_point_local_x"]),
        float(top_action["contact_point_local_y"]),
        0.028 * float(top_action["force_direction_local_x"]),
        0.028 * float(top_action["force_direction_local_y"]),
        width=0.0015,
        head_width=0.006,
        color="#d62728",
        length_includes_head=True,
        zorder=5,
    )
    target = np.asarray(
        [float(metadata["task_local_x_m"]), float(metadata["task_local_y_m"])],
        dtype=np.float64,
    )
    target_norm = float(np.linalg.norm(target))
    if target_norm > 1e-12:
        target = target / target_norm * min(target_norm, 0.065)
        axis.arrow(
            0.0,
            0.0,
            target[0],
            target[1],
            width=0.0012,
            head_width=0.005,
            color="#17becf",
            length_includes_head=True,
            zorder=5,
        )
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_xlim(-extent_x - 0.075, extent_x + 0.075)
    axis.set_ylim(-extent_y - 0.075, extent_y + 0.075)
    axis.set_xlabel("object-local x (m)")
    axis.set_ylabel("object-local y (m)")
    axis.grid(alpha=0.18)
    return scatter


def plot_static(
    rows: list[dict[str, str]],
    action_rows: list[dict[str, str]],
    output_stem: Path,
) -> None:
    """生成一个 decision state 的正式单-map 图。"""

    probabilities, region_probability, metadata = decision_arrays(rows)
    positions = region_positions(action_rows)
    figure, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    scatter = draw_map(
        axis,
        probabilities,
        region_probability,
        metadata,
        action_rows,
        positions,
        float(np.max(region_probability)),
        "Cost-derived affordance map",
    )
    figure.colorbar(scatter, ax=axis, label="contact-region probability")
    if metadata["analysis_mode"] == "known_condition":
        context = "known friction and COM"
    else:
        context = f"after {metadata['valid_update_count_pre']} belief updates"
    figure.suptitle(f"{metadata['scenario'].capitalize()} scenario | {context}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=220)
    figure.savefig(output_stem.with_suffix(".svg"))
    plt.close(figure)


def plot_sequence(
    decisions: list[tuple[tuple[str, int], list[dict[str, str]]]],
    action_rows: list[dict[str, str]],
    output_stem: Path,
) -> bool:
    """生成同一 episode 在 update 0/1/2/4 的四面板序列。"""

    by_episode: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for (episode_key, _), rows in decisions:
        by_episode[episode_key].append(rows)
    for episode_key in sorted(by_episode):
        checkpoints: dict[int, list[dict[str, str]]] = {}
        for rows in sorted(
            by_episode[episode_key], key=lambda item: int(item[0]["push_index"])
        ):
            update_count = int(rows[0]["valid_update_count_pre"])
            if update_count in {0, 1, 2, 4} and update_count not in checkpoints:
                checkpoints[update_count] = rows
        if set(checkpoints) != {0, 1, 2, 4}:
            continue
        arrays = [decision_arrays(checkpoints[index]) for index in (0, 1, 2, 4)]
        colour_maximum = max(float(np.max(item[1])) for item in arrays)
        positions = region_positions(action_rows)
        figure, axes = plt.subplots(
            1, 4, figsize=(18.0, 4.5), constrained_layout=True
        )
        scatter = None
        for axis, update_count, (probabilities, region, metadata) in zip(
            axes, (0, 1, 2, 4), arrays
        ):
            title = "Initial" if update_count == 0 else f"After update {update_count}"
            scatter = draw_map(
                axis,
                probabilities,
                region,
                metadata,
                action_rows,
                positions,
                colour_maximum,
                title,
            )
        if scatter is not None:
            figure.colorbar(scatter, ax=axes, label="contact-region probability")
        scenario = checkpoints[0][0]["scenario"].capitalize()
        figure.suptitle(
            f"Closed-loop cost-derived affordance sequence ({scenario})"
        )
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_stem.with_suffix(".png"), dpi=220)
        figure.savefig(output_stem.with_suffix(".svg"))
        plt.close(figure)
        return True
    return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    """读取 Phase A 完整 map CSV 并生成正式图。"""

    input_path = (
        RESULT_ROOT
        / "evaluation"
        / args.role
        / args.analysis_mode
    )
    if args.output_tag:
        input_path = input_path / args.output_tag
    input_path = input_path / "affordance_map_complete_actions.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"缺少完整 map CSV，请先使用 --save-full-maps 运行分析: {input_path}"
        )
    rows = read_csv(input_path)
    action_rows = load_action_rows()
    decisions = group_decisions(rows)
    output_dir = RESULT_ROOT / "figures" / args.role / args.analysis_mode
    if args.output_tag:
        output_dir = output_dir / args.output_tag
    static_outputs: list[str] = []
    for (episode_key, push_index), decision_rows in decisions[: args.max_figures]:
        stem = output_dir / f"{filename_fragment(episode_key)}_push_{push_index:02d}"
        plot_static(decision_rows, action_rows, stem)
        static_outputs.extend([str(stem.with_suffix(".png")), str(stem.with_suffix(".svg"))])
    sequence_written = False
    sequence_stem = output_dir / "closed_loop_affordance_sequence"
    if args.analysis_mode == "closed_loop":
        sequence_written = plot_sequence(decisions, action_rows, sequence_stem)
    payload = {
        "role": args.role,
        "analysis_mode": args.analysis_mode,
        "output_tag": args.output_tag,
        "input": str(input_path.resolve()),
        "available_decisions": len(decisions),
        "static_figures_written": len(static_outputs) // 2,
        "closed_loop_sequence_written": sequence_written,
        "static_outputs": static_outputs,
        "sequence_outputs": (
            [str(sequence_stem.with_suffix(".png")), str(sequence_stem.with_suffix(".svg"))]
            if sequence_written
            else []
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    """构造绘图入口。"""

    parser = argparse.ArgumentParser(
        description="绘制 HCR V2 cost-derived affordance maps。"
    )
    parser.add_argument(
        "--analysis-mode",
        choices=("known_condition", "closed_loop"),
        default="closed_loop",
    )
    parser.add_argument("--role", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-figures", type=int, default=1)
    parser.add_argument("--output-tag", default="")
    return parser


def main() -> None:
    """解析参数并执行绘图。"""

    args = build_parser().parse_args()
    if args.max_figures <= 0:
        raise ValueError("max_figures 必须大于 0")
    run(args)


if __name__ == "__main__":
    main()
