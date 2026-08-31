"""Level 1 candidate generator for direct-region-force dataset.

生成 Level 1 候选动作行，每行包含 schema.FIELD_NAMES 的 61 字段，
可通过 schema.validate_row() 的基础校验。

只生成 candidate，不运行 MuJoCo，不生成真实 outcome label。
"""

import csv
import math
from pathlib import Path
from typing import Any

from push_core.schema import level1_schema as schema

# ──────────────────────────────────────────────
# 1. 几何与物理常数
# ──────────────────────────────────────────────

HALF_SIZE = 0.045  # 正方体半边长 (m)
SIDE_LENGTH = 0.09  # 正方体边长 (m)

# 3x3 网格偏移量（相对面中心）
GRID_VALUES = (-0.03, 0.0, 0.03)


def _inertia_value() -> float:
    """正方体实心方体 I = (1/6) * m * a^2 三轴同值。"""
    return (1.0 / 6.0) * schema.LEVEL1_FIXED_VALUES["hidden_mass"] * SIDE_LENGTH**2


INERTIA_VALUE = _inertia_value()


# ──────────────────────────────────────────────
# 2. Generator 枚举维度
# ──────────────────────────────────────────────

CANDIDATE_ENUM_FIELDS: tuple[str, ...] = (
    "surface_id",
    "contact_region_row",
    "contact_region_col",
    "force_angle_relative_to_normal_deg",
    "commanded_force_N",
    "ramp_up_s",
    "hold_s",
    "ramp_down_s",
)
"""Generator 实际参与笛卡尔积枚举的字段名。dataset_role 不作为枚举维度。"""


# ──────────────────────────────────────────────
# 3. 派生函数
# ──────────────────────────────────────────────


def compute_contact_point(
    surface_id: int, row: int, col: int
) -> tuple[float, float, float]:
    """根据 surface_id、row、col 计算接触点的局部坐标 (x, y, z)。"""
    _ = row  # Level 1 固定高度
    gx = GRID_VALUES[col]
    if surface_id == 0:  # +X
        return (HALF_SIZE, gx, 0.0)
    elif surface_id == 1:  # -X
        return (-HALF_SIZE, gx, 0.0)
    elif surface_id == 2:  # +Y
        return (gx, HALF_SIZE, 0.0)
    elif surface_id == 3:  # -Y
        return (gx, -HALF_SIZE, 0.0)
    else:
        raise ValueError(f"Invalid surface_id: {surface_id}")


def compute_force_direction(
    surface_id: int, angle_deg: float
) -> tuple[float, float, float]:
    """计算力方向的局部坐标 (fx, fy, fz)。

    外法线 -> 内法线 -> XY 旋转 angle_deg -> 归一化。
    """
    normal = schema.SURFACE_NORMAL_LOCAL[surface_id]
    inward_x = -normal[0]
    inward_y = -normal[1]
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    fx = inward_x * cos_t - inward_y * sin_t
    fy = inward_x * sin_t + inward_y * cos_t
    norm = math.sqrt(fx * fx + fy * fy)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0)
    return (fx / norm, fy / norm, 0.0)


# ──────────────────────────────────────────────
# 4. 单行生成
# ──────────────────────────────────────────────


def make_candidate_row(
    candidate_id: int,
    dataset_role: str,
    surface_id: int,
    row: int,
    col: int,
    angle_deg: float,
    commanded_force_n: float,
    ramp_up_s_val: float,
    hold_s_val: float,
    ramp_down_s_val: float,
) -> dict[str, Any]:
    """生成一行完整的 candidate dict。"""
    d: dict[str, Any] = dict(schema.LEVEL1_FIXED_VALUES)

    d["dataset_role"] = dataset_role
    d["candidate_id"] = candidate_id

    d["surface_id"] = surface_id
    d["contact_region_row"] = row
    d["contact_region_col"] = col
    d["contact_region_id"] = surface_id * 9 + row * 3 + col

    cx, cy, cz = compute_contact_point(surface_id, row, col)
    d["contact_point_local_x"] = cx
    d["contact_point_local_y"] = cy
    d["contact_point_local_z"] = cz

    nx, ny, nz = schema.SURFACE_NORMAL_LOCAL[surface_id]
    d["contact_normal_local_x"] = nx
    d["contact_normal_local_y"] = ny
    d["contact_normal_local_z"] = nz

    d["force_angle_relative_to_normal_deg"] = angle_deg
    fx, fy, fz = compute_force_direction(surface_id, angle_deg)
    d["force_direction_local_x"] = fx
    d["force_direction_local_y"] = fy
    d["force_direction_local_z"] = fz

    d["commanded_force_N"] = commanded_force_n
    d["ramp_up_s"] = ramp_up_s_val
    d["hold_s"] = hold_s_val
    d["ramp_down_s"] = ramp_down_s_val
    d["command_duration_s"] = ramp_up_s_val + hold_s_val

    d["goal_delta_local_x"] = 0.0
    d["goal_delta_local_y"] = 0.0

    # label 占位
    d["delta_x"] = 0.0
    d["delta_y"] = 0.0
    d["delta_yaw"] = 0.0
    d["delta_z"] = 0.0

    # 诊断占位
    d["final_qpos_x"] = 0.0
    d["final_qpos_y"] = 0.0
    d["final_qpos_yaw"] = 0.0
    d["settle_time_s"] = 0.0
    d["simulation_unstable"] = 0
    d["quality_pass"] = 0
    d["contact_success"] = 0
    d["num_contacts"] = 0
    d["stopped_by_threshold"] = 0

    d["hidden_inertia_xx"] = INERTIA_VALUE
    d["hidden_inertia_yy"] = INERTIA_VALUE
    d["hidden_inertia_zz"] = INERTIA_VALUE

    errors = schema.validate_row(d)
    if errors:
        raise ValueError(
            f"candidate_id={candidate_id} validation failed: {'; '.join(errors)}"
        )
    return d


# ──────────────────────────────────────────────
# 5. 批量生成
# ──────────────────────────────────────────────


def generate_level1_candidates(
    dataset_role: str = "train",
) -> list[dict[str, Any]]:
    """生成 Level 1 全量候选动作列表。"""
    if dataset_role not in schema.LEVEL1_ENUMS["dataset_role"]:
        raise ValueError(
            f"Invalid dataset_role: {dataset_role!r}. "
            f"Allowed: {schema.LEVEL1_ENUMS['dataset_role']}"
        )

    rows: list[dict[str, Any]] = []
    candidate_id = 0

    for surface_id in schema.LEVEL1_ENUMS["surface_id"]:
        for row in schema.LEVEL1_ENUMS["contact_region_row"]:
            for col in schema.LEVEL1_ENUMS["contact_region_col"]:
                for angle in schema.LEVEL1_ENUMS["force_angle_relative_to_normal_deg"]:
                    for force_n in schema.LEVEL1_ENUMS["commanded_force_N"]:
                        for ramp_up in schema.LEVEL1_ENUMS["ramp_up_s"]:
                            for hold in schema.LEVEL1_ENUMS["hold_s"]:
                                for ramp_down in schema.LEVEL1_ENUMS["ramp_down_s"]:
                                    rows.append(
                                        make_candidate_row(
                                            candidate_id=candidate_id,
                                            dataset_role=dataset_role,
                                            surface_id=surface_id,
                                            row=row,
                                            col=col,
                                            angle_deg=angle,
                                            commanded_force_n=force_n,
                                            ramp_up_s_val=ramp_up,
                                            hold_s_val=hold,
                                            ramp_down_s_val=ramp_down,
                                        )
                                    )
                                    candidate_id += 1
    return rows


# ──────────────────────────────────────────────
# 6. CSV 写出
# ──────────────────────────────────────────────


def write_candidates_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """将 candidate 行列表写入 CSV 文件。使用 utf-8-sig 编码。"""
    path = Path(path)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema.FIELD_NAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
