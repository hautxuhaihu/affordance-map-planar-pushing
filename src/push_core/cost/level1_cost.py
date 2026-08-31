"""Level 1 SE(2) cost function for planar pushing.

本模块实现第一版 Level 1 cost 计算，核心公式来自
`docs/architecture_design/cost function definition.md`：

    L = ||p - p*||_2 + |theta - theta*| / tau_theta

其中：
- p = (delta_x, delta_y) 是 outcome 平面位移
- p* = (goal_delta_x, goal_delta_y) 是目标平面位移
- theta 是 outcome yaw（unwrapped）
- theta* 是目标 yaw（unwrapped）
- tau_theta 是旋转容忍度（默认约 10 deg = 0.1745 rad）

重要语义说明：
- goal 来自外部任务输入（用户指令），不来自 outcome dataset 或 predictor 输出。
- oracle outcome（真实结果）的字段定义来自 direct_region_force_schema.md，
  但在 action-selection 阶段实际从 predictor 输出 CSV 中读取其保留的真实 outcome 列。
- predicted outcome 来自 predictor 输出 CSV（pred_delta_x/pred_delta_y/pred_delta_yaw）。
- 第一版 cost 不使用 force / surface_id / contact_point / hidden_* 等字段。
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Level1Goal:
    """Level 1 任务目标（相对初始状态的目标变化量）。

    由外部任务输入构建，不来自 outcome dataset 或 predictor 输出。

    delta_yaw 是 unwrapped 目标 yaw 变化，不做 wrap_to_pi。
    """

    delta_x: float
    delta_y: float
    delta_yaw: float


@dataclass
class Level1Outcome:
    """Level 1 一次 push 动作产生的实际/预测 outcome。

    delta_yaw 是 unwrapped yaw 变化，不做 wrap_to_pi。
    """

    delta_x: float
    delta_y: float
    delta_yaw: float


# 默认 tau_theta = 10 deg ≈ 0.1745 rad
_DEFAULT_TAU_THETA: float = 0.17453292519943295


def make_level1_goal(
    delta_x: float,
    delta_y: float,
    delta_yaw: float = 0.0,
) -> Level1Goal:
    """构造第一版 task goal。

    该函数是创建 goal 的唯一推荐方式，明确表达 goal 来自外部任务输入，
    例如用户指令转换后的相对位移和相对 yaw。

    参数：
        delta_x:   目标相对位移 X (m)
        delta_y:   目标相对位移 Y (m)
        delta_yaw: 目标相对 yaw 变化 (rad)；默认为 0（不旋转）

    返回：
        Level1Goal
    """
    return Level1Goal(
        delta_x=float(delta_x), delta_y=float(delta_y), delta_yaw=float(delta_yaw)
    )


def compute_level1_cost(
    outcome: Level1Outcome,
    goal: Level1Goal,
    tau_theta: float = _DEFAULT_TAU_THETA,
) -> float:
    """计算 Level 1 SE(2) cost。

    公式：
        position_error = sqrt((outcome.delta_x - goal.delta_x)^2
                            + (outcome.delta_y - goal.delta_y)^2)

        yaw_error = outcome.delta_yaw - goal.delta_yaw  # unwrapped

        cost = position_error + abs(yaw_error) / tau_theta

    参数：
        outcome:   实际或预测的 outcome
        goal:      目标（相对初始状态的目标变化量），来自外部任务输入
        tau_theta: 旋转容忍度，控制 yaw 误差在 cost 中的权重

    返回：
        cost 标量（米 + 归一化弧度）
    """
    dx = outcome.delta_x - goal.delta_x
    dy = outcome.delta_y - goal.delta_y
    position_error = (dx * dx + dy * dy) ** 0.5

    yaw_error = outcome.delta_yaw - goal.delta_yaw
    yaw_error_abs = yaw_error if yaw_error >= 0.0 else -yaw_error

    return position_error + yaw_error_abs / tau_theta


# ─── schema row helper ──────────────────────────────────────────────
#
# 以下 helper 从 predictor 输出 CSV row 中读取 outcome。
# 字段定义来自 direct_region_force_schema.md，但实际 row 通常来自
# predict_level1_outcome.py 产生的预测 CSV。
#
# goal 不在此处读取，应使用 make_level1_goal() 由外部任务输入构造。


def true_outcome_from_row(row: dict[str, Any]) -> Level1Outcome:
    """从包含真实 rollout outcome 的 row 中读取 oracle outcome。

    字段定义来源：direct_region_force_schema.md §3.6
        delta_x
        delta_y
        delta_yaw

    实际 row 来源：（在 action-selection 阶段）
        predictor 输出 CSV 中保留的真实 outcome 列。

    返回：
        Level1Outcome(delta_x, delta_y, delta_yaw)
    """
    return Level1Outcome(
        delta_x=float(row["delta_x"]),
        delta_y=float(row["delta_y"]),
        delta_yaw=float(row["delta_yaw"]),
    )


def predicted_outcome_from_row(row: dict[str, Any]) -> Level1Outcome:
    """从 predictor 输出 CSV row 中读取 predicted outcome。

    字段定义来源：direct_region_force_schema.md §3.6（预测阶段补充）
        pred_delta_x
        pred_delta_y
        pred_delta_yaw

    实际 row 来源：
        predict_level1_outcome.py 输出 CSV 中的预测列。

    返回：
        Level1Outcome(delta_x, delta_y, delta_yaw)
    """
    return Level1Outcome(
        delta_x=float(row["pred_delta_x"]),
        delta_y=float(row["pred_delta_y"]),
        delta_yaw=float(row["pred_delta_yaw"]),
    )
