"""Level 1 benchmark task goals.

本模块基于 level1_task_goals.md 定义第一版 160 个 benchmark goals。

这些 goal 来自外部任务输入设计，不是从 outcome dataset 或 predictor 输出读取。

使用方式：
    from push_core.action_space.tasks import make_benchmark_goals

    goals = make_benchmark_goals()
    for goal_name, goal in goals:
        cost = compute_level1_cost(outcome, goal)
"""

import math

from push_core.cost.level1_cost import Level1Goal

# ─── 方向定义 ────────────────────────────────────────────────────────

# 16 个方向的角度与名称
DIRECTIONS_16: list[tuple[int, float, str]] = [
    (0, 0.0, "pos_x"),
    (1, 22.5, "pos_x_pos_y_22.5"),
    (2, 45.0, "pos_x_pos_y_45"),
    (3, 67.5, "pos_y_pos_x_67.5"),
    (4, 90.0, "pos_y"),
    (5, 112.5, "pos_y_neg_x_112.5"),
    (6, 135.0, "neg_x_pos_y_135"),
    (7, 157.5, "neg_x_pos_y_157.5"),
    (8, 180.0, "neg_x"),
    (9, 202.5, "neg_x_neg_y_202.5"),
    (10, 225.0, "neg_x_neg_y_225"),
    (11, 247.5, "neg_y_neg_x_247.5"),
    (12, 270.0, "neg_y"),
    (13, 292.5, "neg_y_pos_x_292.5"),
    (14, 315.0, "pos_x_neg_y_315"),
    (15, 337.5, "pos_x_neg_y_337.5"),
]

# 10 个距离档
DISTANCES_10: list[float] = [
    0.03,
    0.06,
    0.09,
    0.12,
    0.15,
    0.18,
    0.21,
    0.24,
    0.27,
    0.30,
]

# ─── 构造 benchmark goals ────────────────────────────────────────────


def make_benchmark_goals() -> list[tuple[str, Level1Goal]]:
    """生成 16 方向 × 10 距离 = 160 个 benchmark goals。

    每个 goal 的命名格式：
        {direction_name}__{distance:04d}

    例如：
        pos_x__0030   → +X，目标位移 0.03 m
        pos_y__0150   → +Y，目标位移 0.15 m
        neg_x__0300   → -X，目标位移 0.30 m

    返回：
        [(goal_name, Level1Goal), ...]
    """
    goals: list[tuple[str, Level1Goal]] = []
    for _, angle_deg, dir_name in DIRECTIONS_16:
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        for distance in DISTANCES_10:
            delta_x = distance * cos_t
            delta_y = distance * sin_t
            name = f"{dir_name}__{int(distance * 1000):04d}"

            # 处理 -0.000 这种不美观的值
            if abs(delta_x) < 1e-12:
                delta_x = 0.0
            if abs(delta_y) < 1e-12:
                delta_y = 0.0

            goals.append(
                (name, Level1Goal(delta_x=delta_x, delta_y=delta_y, delta_yaw=0.0))
            )
    return goals


def make_benchmark_goal(
    angle_deg: float,
    distance: float,
    delta_yaw: float = 0.0,
) -> Level1Goal:
    """构造单个 benchmark goal（不加入 160 个标准集，用于自定义任务）。

    参数：
        angle_deg: 目标移动方向角度 (度)，0° = +X，逆时针为正
        distance:  目标移动距离 (m)
        delta_yaw: 目标 yaw (rad)；默认 0

    返回：
        Level1Goal
    """
    theta = math.radians(angle_deg)
    delta_x = distance * math.cos(theta)
    delta_y = distance * math.sin(theta)

    if abs(delta_x) < 1e-12:
        delta_x = 0.0
    if abs(delta_y) < 1e-12:
        delta_y = 0.0

    return Level1Goal(delta_x=delta_x, delta_y=delta_y, delta_yaw=delta_yaw)


# ─── 分组查询 ────────────────────────────────────────────────────────


def goals_by_direction() -> dict[str, list[tuple[str, Level1Goal]]]:
    """按方向名称分组返回 benchmark goals。"""
    groups: dict[str, list[tuple[str, Level1Goal]]] = {}
    for name, goal in make_benchmark_goals():
        dir_name = name.rsplit("__", 1)[0]
        if dir_name not in groups:
            groups[dir_name] = []
        groups[dir_name].append((name, goal))
    return groups


def goals_by_distance() -> dict[float, list[tuple[str, Level1Goal]]]:
    """按距离分组返回 benchmark goals。"""
    groups: dict[float, list[tuple[str, Level1Goal]]] = {}
    for name, goal in make_benchmark_goals():
        if goal.delta_x == 0.0 and goal.delta_y == 0.0:
            continue
        dist = round(math.hypot(goal.delta_x, goal.delta_y), 6)
        if dist not in groups:
            groups[dist] = []
        groups[dist].append((name, goal))
    return groups
