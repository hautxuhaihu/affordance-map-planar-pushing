"""Level 1 pushing 的 61 字段兼容 schema。

该 schema 最初来自 V2-B direct-region-force 数据契约。当前 `msc_proj_v2`
继续复用相同字段顺序、action 参数和 hidden oracle 分类，使历史数据、physical
pusher 数据、forward model 与 inverse model 保持兼容。

当前主线的真实执行语义由
`push_core.simulation.physical_pusher_rollout` 和
`assets/xml/msc_rod_pusher_box.xml` 决定：力施加到 capsule rod pusher，物体只
通过真实接触受力。下方保留的 direct-region-force 常量只描述 schema 的历史
来源，不代表当前 physical-pusher rollout 仍直接对 object 施力。

下游 generator、collector 和 validator 必须引此模块作为唯一字段来源，不得
重复定义 schema。
"""

import csv
import io
import math
from dataclasses import dataclass
from typing import Any

# ──────────────────────────────────────────────
# 1. 单字段元数据
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class FieldSpec:
    """单个 CSV 字段的定义元数据。

    Attributes:
        name: 字段名，与 CSV 表头一致。
        description: 中文说明。
        data_type: "string" | "int" | "float"。
        level1: Level 1 取值范围原始文本（展示用，不用于解析）。
        is_model_input: 是否为 outcome model 推荐输入。
        is_label: 是否为预测标签。
        is_oracle: 是否为 hidden oracle 字段（禁止作为模型输入）。
        is_diagnostic: 是否为质量诊断字段（不参与训练）。
    """

    name: str
    description: str
    data_type: str  # "string" | "int" | "float"
    level1: str
    is_model_input: bool = False
    is_label: bool = False
    is_oracle: bool = False
    is_diagnostic: bool = False


# ──────────────────────────────────────────────
# 2. Schema 常量（非逐行字段）
# ──────────────────────────────────────────────

# 以下标识是 61 字段 schema 的历史来源信息。它们不在当前 CSV 字段中，也不用于
# 选择 physical-pusher 或 direct-force rollout。执行机制必须由 simulation 模块显式决定。
SCHEMA_VERSION = "direct_region_force_schema_v1"
FORCE_APPLICATION_TYPE = "direct_region_force"
PUSHER_CONTROL_TYPE = "none"
FORCE_CONTROLLER_TYPE = "direct_object_external_force"

OBJECT_SIZE_X = 0.09
OBJECT_SIZE_Y = 0.09
OBJECT_SIZE_Z = 0.09

OBJECT_JOINT_DAMPING = 0.0
OBJECT_JOINT_FRICTIONLOSS = 0.0

# 浮点比较容差
FLOAT_TOL = 1e-9


# ──────────────────────────────────────────────
# 3. 61 个字段定义（按 CSV 顺序）
# ──────────────────────────────────────────────

_FIELDS = (
    # — 标识字段 (8) —
    FieldSpec(
        "dataset_role",
        "数据集角色划分标识",
        "string",
        "train, val_iid, val_ood, test_iid, test_ood",
    ),
    FieldSpec("episode_id", "一组连续 rollout 的编号", "int", "0"),
    FieldSpec("step_id", "episode 内的步序号", "int", "0"),
    FieldSpec("group_id", "Level3A reset-based probe 分组编号", "int", "0"),
    FieldSpec(
        "hidden_condition_id",
        "当前样本的 hidden condition 组合编号",
        "int",
        "0（单种 hidden）",
    ),
    FieldSpec(
        "trial_id",
        "同一 (state,action,hidden_condition) 的重复编号",
        "int",
        "无重复: 0",
    ),
    FieldSpec("candidate_id", "当前 state 下的候选动作枚举编号", "int", "单调递增整数"),
    FieldSpec(
        "contact_region_id",
        "全局候选区域编号 (surface_id*9+row*3+col)",
        "int",
        "{0,1,...,35} (4面*9点); >=36 预留",
    ),
    # — 初始状态字段 (9) —
    FieldSpec(
        "object_qpos_initial_x",
        "物体初始 x 世界坐标 (m)",
        "float",
        "固定: 0",
        is_model_input=False,
    ),
    FieldSpec(
        "object_qpos_initial_y",
        "物体初始 y 世界坐标 (m)",
        "float",
        "固定: 0",
        is_model_input=False,
    ),
    FieldSpec(
        "object_qpos_initial_z",
        "物体初始 z 世界坐标 (m)",
        "float",
        "0.045; |delta_z|>0.01 视为异常",
        is_diagnostic=True,
    ),
    FieldSpec(
        "object_qpos_initial_yaw",
        "物体初始偏航角 (rad)",
        "float",
        "固定: 0",
        is_model_input=False,
    ),
    FieldSpec(
        "object_initial_sin_yaw",
        "初始 yaw 的正弦编码",
        "float",
        "[-1, 1]; sin(object_qpos_initial_yaw)",
        is_model_input=False,
    ),
    FieldSpec(
        "object_initial_cos_yaw",
        "初始 yaw 的余弦编码",
        "float",
        "[-1, 1]; cos(object_qpos_initial_yaw)",
        is_model_input=False,
    ),
    FieldSpec(
        "object_qvel_initial_x",
        "初始 x 线速度 (m/s)",
        "float",
        "固定: 0（每次 push 前静止）",
    ),
    FieldSpec(
        "object_qvel_initial_y",
        "初始 y 线速度 (m/s)",
        "float",
        "固定: 0（每次 push 前静止）",
    ),
    FieldSpec(
        "object_qvel_initial_yaw",
        "初始偏航角速度 (rad/s)",
        "float",
        "固定: 0（每次 push 前静止）",
    ),
    # — 目标字段 (3) —
    FieldSpec(
        "goal_delta_local_x",
        "目标位移在物体局部坐标系下的 X (m)",
        "float",
        "由指令决定; 仅用于 cost 计算, 不参与 outcome model 训练",
    ),
    FieldSpec(
        "goal_delta_local_y",
        "目标位移在物体局部坐标系下的 Y (m)",
        "float",
        "由指令决定; 仅用于 cost 计算, 不参与 outcome model 训练",
    ),
    FieldSpec("goal_yaw", "目标偏航角 (rad)", "float", "暂不使用（固定 0）"),
    # — 候选接触区域字段 (9) —
    FieldSpec(
        "surface_id",
        "被施力的物体表面编号",
        "int",
        "0: +X; 1: -X; 2: +Y; 3: -Y",
        is_model_input=False,
    ),
    FieldSpec(
        "contact_region_row",
        "当前表面 3x3 网格的行编号",
        "int",
        "{0,1,2}; 0=底行, 2=顶行",
        is_model_input=False,
    ),
    FieldSpec(
        "contact_region_col",
        "当前表面 3x3 网格的列编号",
        "int",
        "{0,1,2}; 0=左列, 2=右列",
        is_model_input=False,
    ),
    FieldSpec(
        "contact_point_local_x",
        "施力点相对几何中心的局部 X (m)",
        "float",
        "[-0.045, 0.045]",
        is_model_input=True,
    ),
    FieldSpec(
        "contact_point_local_y",
        "施力点相对几何中心的局部 Y (m)",
        "float",
        "[-0.045, 0.045]",
        is_model_input=True,
    ),
    FieldSpec(
        "contact_point_local_z",
        "施力点相对几何中心的局部 Z (m)",
        "float",
        "0（固定中间高度）",
    ),
    FieldSpec(
        "contact_normal_local_x",
        "表面外法线的局部 X 分量",
        "float",
        "{-1, 0, 1}",
        is_model_input=True,
    ),
    FieldSpec(
        "contact_normal_local_y",
        "表面外法线的局部 Y 分量",
        "float",
        "{-1, 0, 1}",
        is_model_input=True,
    ),
    FieldSpec(
        "contact_normal_local_z",
        "表面外法线的局部 Z 分量",
        "float",
        "0（侧面，水平推）",
    ),
    # — 力命令字段 (9) —
    FieldSpec(
        "force_angle_relative_to_normal_deg",
        "力方向相对于内法线的偏转角 (度)",
        "float",
        "{-45, -30, -15, 0, 15, 30, 45}",
    ),
    FieldSpec(
        "force_direction_local_x",
        "力方向在物体局部坐标系下的 X 分量",
        "float",
        "[-1, 1]; 单位向量, 由 force_angle 计算",
        is_model_input=True,
    ),
    FieldSpec(
        "force_direction_local_y",
        "力方向在物体局部坐标系下的 Y 分量",
        "float",
        "[-1, 1]; 单位向量, 由 force_angle 计算",
        is_model_input=True,
    ),
    FieldSpec(
        "force_direction_local_z",
        "力方向在物体局部坐标系下的 Z 分量",
        "float",
        "0（水平推）",
    ),
    FieldSpec(
        "commanded_force_N",
        "命令力大小 (N)",
        "float",
        "{0.5, 0.8, 1.1}",
        is_model_input=True,
    ),
    FieldSpec(
        "ramp_up_s",
        "力从 0 平滑上升到目标值的时间 (s)",
        "float",
        "{0.02, 0.04, 0.06}",
        is_model_input=True,
    ),
    FieldSpec(
        "hold_s", "目标力保持时间 (s)", "float", "{0.05, 0.10, 0.15}", is_model_input=True
    ),
    FieldSpec(
        "ramp_down_s",
        "力从目标值平滑释放到 0 的时间 (s)",
        "float",
        "0.02 或 0.04",
        is_model_input=True,
    ),
    FieldSpec(
        "command_duration_s",
        "总施力时长 (=ramp_up_s+hold_s+ramp_down_s) (s)",
        "float",
        "由 ramp_up_s+hold_s+ramp_down_s 计算得出",
    ),
    # — Primary labels (4) —
    FieldSpec(
        "delta_x",
        "rollout 结束时几何中心 X 位移 (m)",
        "float",
        "(-inf, +inf); +X 为正方向",
        is_label=True,
    ),
    FieldSpec(
        "delta_y",
        "rollout 结束时几何中心 Y 位移 (m)",
        "float",
        "(-inf, +inf); +Y 为正方向",
        is_label=True,
    ),
    FieldSpec(
        "delta_yaw",
        "rollout 结束时 yaw 变化量 (rad)",
        "float",
        "unwrapped; 由 object_yaw hinge qpos 差值计算; 可超过 ±pi 和多圈",
        is_label=True,
    ),
    FieldSpec(
        "delta_z",
        "rollout 结束时 z 位移 (m)",
        "float",
        "约等于 0 正常; 绝对值 >0.01 视为异常",
        is_diagnostic=True,
    ),
    # — 诊断字段 (9) —
    FieldSpec(
        "final_qpos_x",
        "rollout 结束时的最终 x 位置 (m)",
        "float",
        "用于 debug 和复现实验",
        is_diagnostic=True,
    ),
    FieldSpec(
        "final_qpos_y",
        "rollout 结束时的最终 y 位置 (m)",
        "float",
        "用于 debug 和复现实验",
        is_diagnostic=True,
    ),
    FieldSpec(
        "final_qpos_yaw",
        "rollout 结束时的最终偏航角 (rad)",
        "float",
        "用于 debug 和复现实验",
        is_diagnostic=True,
    ),
    FieldSpec(
        "settle_time_s",
        "力释放后等待物体停止的实际时间 (s)",
        "float",
        "[0, SETTLE_MAX_S]",
        is_diagnostic=True,
    ),
    FieldSpec(
        "simulation_unstable",
        "仿真是否出现数值不稳定",
        "int",
        "0: 稳定; 1: 出现 NaN 或爆炸",
        is_diagnostic=True,
    ),
    FieldSpec(
        "quality_pass",
        "样本是否通过基础质量检查",
        "int",
        "0: 丢弃; 1: 可训练",
        is_diagnostic=True,
    ),
    FieldSpec(
        "contact_success",
        "施力过程中至少 1 个接触对激活",
        "int",
        "0: 无接触（异常）; 1: 有接触",
        is_diagnostic=True,
    ),
    FieldSpec(
        "num_contacts",
        "施力结束时激活的接触对数量",
        "int",
        ">=0; 当前至少 1（物体-桌面）",
        is_diagnostic=True,
    ),
    FieldSpec(
        "stopped_by_threshold",
        "物体在超时前速度降至停止阈值以下",
        "int",
        "0: 超时未停; 1: 阈值停止",
        is_diagnostic=True,
    ),
    # — Hidden oracle 字段 (10) —
    FieldSpec(
        "hidden_com_offset_x",
        "物体质心在局部坐标下的 X 偏移 (m)",
        "float",
        "0.0",
        is_oracle=True,
    ),
    FieldSpec(
        "hidden_com_offset_y",
        "物体质心在局部坐标下的 Y 偏移 (m)",
        "float",
        "0.0",
        is_oracle=True,
    ),
    FieldSpec(
        "hidden_com_offset_z",
        "物体质心在局部坐标下的 Z 偏移 (m)",
        "float",
        "0.0（几何对称）",
        is_oracle=True,
    ),
    FieldSpec("hidden_mass", "物体质量 (kg)", "float", "0.10", is_oracle=True),
    FieldSpec(
        "hidden_inertia_xx",
        "物体绕局部 X 轴的转动惯量 (kg*m^2)",
        "float",
        "由 box 质量/尺寸解析计算",
        is_oracle=True,
    ),
    FieldSpec(
        "hidden_inertia_yy",
        "物体绕局部 Y 轴的转动惯量 (kg*m^2)",
        "float",
        "由 box 质量/尺寸解析计算",
        is_oracle=True,
    ),
    FieldSpec(
        "hidden_inertia_zz",
        "物体绕局部 Z 轴的转动惯量 (kg*m^2)",
        "float",
        "由 box 质量/尺寸解析计算",
        is_oracle=True,
    ),
    FieldSpec(
        "hidden_object_table_sliding_friction",
        "object-table 接触滑动摩擦系数",
        "float",
        "0.4",
        is_oracle=True,
    ),
    FieldSpec(
        "hidden_object_table_torsional_friction",
        "object-table 接触扭转摩擦系数",
        "float",
        "0.005",
        is_oracle=True,
    ),
    FieldSpec(
        "hidden_object_table_rolling_friction",
        "object-table 接触滚动摩擦系数 (预留)",
        "float",
        "0.0001（当前 condim=4 不生效）",
        is_oracle=True,
    ),
)

assert len(_FIELDS) == 61, f"Expected 61 fields, got {len(_FIELDS)}"


# ──────────────────────────────────────────────
# 4. 派生常量
# ──────────────────────────────────────────────

FIELDS: tuple[FieldSpec, ...] = _FIELDS
"""按 CSV 顺序排列的 61 字段完整定义。"""

FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in FIELDS)
"""仅字段名，用于 CSV 表头。"""

FIELD_BY_NAME: dict[str, FieldSpec] = {f.name: f for f in FIELDS}
"""按字段名查询 FieldSpec。"""

# 用 csv.writer 生成标准 CSV 表头行
_CSV_BUFFER = io.StringIO()
_csv_writer = csv.writer(_CSV_BUFFER, lineterminator="\n")
_csv_writer.writerow(FIELD_NAMES)
CSV_HEADER_LINE: str = _CSV_BUFFER.getvalue().rstrip("\n")
"""双引号包裹的标准 CSV 表头字符串。不包含 BOM。"""


# 分组常量

MODEL_INPUT_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.is_model_input)
"""Outcome model 推荐输入字段（10 个）。"""

assert len(MODEL_INPUT_FIELDS) == 10, (
    f"Expected 10 model input fields, got {len(MODEL_INPUT_FIELDS)}"
)

LABEL_FIELDS: tuple[str, ...] = ("delta_x", "delta_y", "delta_yaw")
"""第一版预测目标（3 个）。"""

assert all(f.is_label for f in FIELDS if f.name in LABEL_FIELDS)

ORACLE_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.is_oracle)
"""Hidden oracle 字段（10 个），禁止作为模型输入。"""

assert len(ORACLE_FIELDS) == 10, f"Expected 10 oracle fields, got {len(ORACLE_FIELDS)}"

DIAGNOSTIC_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.is_diagnostic)
"""质量诊断字段。"""


# ──────────────────────────────────────────────
# 5. Level 1 Typed Constants
# ──────────────────────────────────────────────

LEVEL1_ENUMS: dict[str, tuple[Any, ...]] = {
    "dataset_role": ("train", "val_iid", "val_ood", "test_iid", "test_ood"),
    "surface_id": (0, 1, 2, 3),
    "contact_region_row": (0, 1, 2),
    "contact_region_col": (0, 1, 2),
    "force_angle_relative_to_normal_deg": (-45, -30, -15, 0, 15, 30, 45),
    "commanded_force_N": (0.5, 0.8, 1.1),
    "ramp_up_s": (0.02, 0.04, 0.06),
    "hold_s": (0.05, 0.10, 0.15),
    "ramp_down_s": (0.02, 0.04),
}
"""Level 1 有穷枚举字段的取值集合。generator 直接引用。

注意：contact_normal_local_x/y 不由 generator 独立采样，
它们由 surface_id 通过 SURFACE_NORMAL_LOCAL 映射决定。
"""


SURFACE_NORMAL_LOCAL: dict[int, tuple[float, float, float]] = {
    0: (1.0, 0.0, 0.0),  # +X
    1: (-1.0, 0.0, 0.0),  # -X
    2: (0.0, 1.0, 0.0),  # +Y
    3: (0.0, -1.0, 0.0),  # -Y
}
"""surface_id 到局部外法线 (nx, ny, nz) 的映射。generator 直接用此表填充法线字段。"""


LEVEL1_FIXED_VALUES: dict[str, Any] = {
    # 标识
    "episode_id": 0,
    "step_id": 0,
    "group_id": 0,
    "hidden_condition_id": 0,
    "trial_id": 0,
    # 初始状态
    "object_qpos_initial_x": 0.0,
    "object_qpos_initial_y": 0.0,
    "object_qpos_initial_z": 0.045,
    "object_qpos_initial_yaw": 0.0,
    "object_initial_sin_yaw": 0.0,
    "object_initial_cos_yaw": 1.0,
    "object_qvel_initial_x": 0.0,
    "object_qvel_initial_y": 0.0,
    "object_qvel_initial_yaw": 0.0,
    # 目标
    "goal_yaw": 0.0,
    # 接触区域
    "contact_point_local_z": 0.0,
    "contact_normal_local_z": 0.0,
    # 力方向
    "force_direction_local_z": 0.0,
    # hidden oracle 固定值
    "hidden_com_offset_x": 0.0,
    "hidden_com_offset_y": 0.0,
    "hidden_com_offset_z": 0.0,
    "hidden_mass": 0.10,
    "hidden_object_table_sliding_friction": 0.4,
    "hidden_object_table_torsional_friction": 0.005,
    "hidden_object_table_rolling_friction": 0.0001,
}
"""Level 1 固定值的字段到其固定值的映射。validator 和 generator 使用。"""


# ──────────────────────────────────────────────
# 6. 辅助函数
# ──────────────────────────────────────────────


def get_field(name: str) -> FieldSpec:
    """按名称查找字段，找不到抛 KeyError。"""
    if name not in FIELD_BY_NAME:
        raise KeyError(f"Unknown field: {name!r}")
    return FIELD_BY_NAME[name]


def validate_header(field_names: list[str] | tuple[str, ...]) -> list[str]:
    """验证字段顺序是否完全等于 FIELD_NAMES。

    Returns:
        错误列表，空列表表示通过。
    """
    errors: list[str] = []
    if len(field_names) != len(FIELD_NAMES):
        errors.append(
            f"Field count mismatch: got {len(field_names)}, expected {len(FIELD_NAMES)}"
        )
    for i, (got, expected) in enumerate(zip(field_names, FIELD_NAMES)):
        if got != expected:
            errors.append(
                f"Field name mismatch at position {i}: "
                f"got {got!r}, expected {expected!r}"
            )
    return errors


def validate_row(row: dict[str, Any]) -> list[str]:
    """验证一行数据是否符合 Level 1 schema。

    检查内容：
    - 字段集合必须等于 FIELD_NAMES
    - 类型转换（string/int/float）
    - LEVEL1_FIXED_VALUES 中的字段值匹配
    - LEVEL1_ENUMS 中的字段值属于枚举
    - 模型输入不含 oracle 字段
    - 布尔字段值为 0 或 1

    Returns:
        错误列表，空列表表示通过。
    """
    errors: list[str] = []

    # 检查字段集合
    row_keys = set(row.keys())
    expected_keys = set(FIELD_NAMES)
    missing = expected_keys - row_keys
    extra = row_keys - expected_keys
    if missing:
        errors.append(f"Missing fields: {sorted(missing)}")
    if extra:
        errors.append(f"Extra fields: {sorted(extra)}")

    # 类型转换与值校验
    for field in FIELDS:
        name = field.name
        if name not in row:
            continue
        value = row[name]
        dtype = field.data_type

        # 类型转换
        try:
            if dtype == "string":
                _ = str(value)
            elif dtype == "int":
                # 允许 float 转换（如 0.0 → 0）
                v = int(float(str(value)))
                if abs(float(str(value)) - v) > FLOAT_TOL:
                    errors.append(f"{name}: {value!r} is not a valid int")
            elif dtype == "float":
                _ = float(str(value))
        except (ValueError, TypeError):
            errors.append(f"{name}: cannot convert {value!r} to {dtype}")

        # 固定值检查
        if name in LEVEL1_FIXED_VALUES:
            expected = LEVEL1_FIXED_VALUES[name]
            try:
                if isinstance(expected, float):
                    v_float = float(str(value))
                    if not math.isclose(v_float, expected, abs_tol=FLOAT_TOL):
                        errors.append(f"{name}: expected {expected}, got {v_float}")
                else:
                    v_raw = (
                        int(float(str(value)))
                        if field.data_type == "int"
                        else str(value)
                    )
                    if v_raw != expected:
                        errors.append(f"{name}: expected {expected}, got {v_raw}")
            except (ValueError, TypeError):
                pass  # 类型转换失败已在前面报告

        # 枚举检查
        if name in LEVEL1_ENUMS:
            allowed = LEVEL1_ENUMS[name]
            try:
                if field.data_type == "int":
                    v = int(float(str(value)))
                elif field.data_type == "float":
                    v = float(str(value))
                else:
                    v = str(value)
                if v not in allowed:
                    errors.append(f"{name}: value {v!r} not in allowed {allowed}")
            except (ValueError, TypeError):
                pass

    # 模型输入不能包含 oracle 字段
    for fname in MODEL_INPUT_FIELDS:
        if fname in ORACLE_FIELDS:
            errors.append(f"Model input {fname!r} is in ORACLE_FIELDS")

    # 标签字段必须存在（已在字段集合检查中覆盖）
    # 布尔字段值检查
    bool_fields = [
        "simulation_unstable",
        "quality_pass",
        "contact_success",
        "stopped_by_threshold",
    ]
    for bf in bool_fields:
        if bf in row:
            v = row[bf]
            try:
                vi = int(float(str(v)))
                if vi not in (0, 1):
                    errors.append(f"{bf}: value {vi!r} must be 0 or 1")
            except (ValueError, TypeError):
                errors.append(f"{bf}: cannot convert {v!r} to int")

    return errors
