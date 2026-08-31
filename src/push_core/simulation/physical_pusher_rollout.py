"""Level 1 physical-pusher rollout collector.

本模块保持 Level 1 / Level 1.5 的 action space 和 61 字段 schema 不变。
唯一变化是执行方式：

- 旧实现：使用 mj_applyFT 将力直接施加到 object；
- 新实现：将同样的力命令施加到 rod_pusher，object 只通过真实 contact 受力。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from push_core.schema import level1_schema as schema
from push_core.action_space.candidates import generate_level1_candidates
from push_core.project_paths import ASSETS_DIR


DEFAULT_XML_PATH = ASSETS_DIR / "xml" / "msc_rod_pusher_box.xml"

OBJECT_BODY_NAME = "object"
OBJECT_GEOM_NAME = "object_geom"
PUSHER_BODY_NAME = "rod_pusher"
PUSHER_GEOM_NAME = "rod_pusher_geom"

PUSHER_BASE_X = 0.094
PUSHER_TIP_OFFSET = 0.035
APPROACH_DISTANCE = 0.014

CONTACT_WARMUP_STEPS = 30
SETTLE_MAX_S = 2.0
STOP_LINEAR_VEL = 0.005
STOP_YAW_VEL = 0.02
MIN_SETTLE_S = 0.10


@dataclass(frozen=True)
class PusherPlan:
    """由旧 action row 派生出的 physical pusher 初始位姿和施力方向。"""

    start_center_x: float
    start_center_y: float
    yaw: float
    force_direction_world: np.ndarray


@dataclass
class ContactStats:
    """记录 pusher-object 接触诊断。"""

    contact_step_count: int = 0
    max_contact_count: int = 0
    max_contact_force: float = 0.0


def wrap_to_pi(angle: float) -> float:
    """将角度归一化到 [-pi, pi]。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def get_named_id(model, obj_type, name: str) -> int:
    """按名称获取 MuJoCo 对象 id。"""
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return int(obj_id)


def get_joint_qposadr(model, joint_name: str) -> int:
    """获取 joint 在 qpos 中的起始索引。"""
    joint_id = get_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[joint_id])


def get_joint_dofadr(model, joint_name: str) -> int:
    """获取 joint 在 qvel 中的起始索引。"""
    joint_id = get_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_dofadr[joint_id])


def get_body_id(model, body_name: str = OBJECT_BODY_NAME) -> int:
    """获取 body id。"""
    return get_named_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)


def get_object_yaw_qpos(model, data) -> float:
    """读取 object_yaw hinge joint 的累计角度，保留 unwrapped yaw。"""
    return float(data.qpos[get_joint_qposadr(model, "object_yaw")])


def load_model(xml_path: str | Path | None = None):
    """加载 physical-pusher MuJoCo 模型。"""
    path = Path(xml_path) if xml_path else DEFAULT_XML_PATH
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    return model, data


def make_pusher_plan_from_row(row: dict[str, Any]) -> PusherPlan:
    """从旧 Level 1 action row 派生 pusher 初始位姿。

    旧 row 中的 contact point 和 force direction 仍然是唯一 action 语义来源。
    """
    contact_x = float(row["contact_point_local_x"])
    contact_y = float(row["contact_point_local_y"])
    direction = np.array(
        [
            float(row["force_direction_local_x"]),
            float(row["force_direction_local_y"]),
            0.0,
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(direction[:2]))
    if norm <= 1e-12:
        raise ValueError(f"candidate_id={row['candidate_id']} has zero force direction")
    direction = direction / norm

    # fingertip 放在接触点外侧，避免初始穿透；随后对 pusher 施加旧 action 的力。
    start_tip = np.array([contact_x, contact_y, 0.0]) - direction * APPROACH_DISTANCE
    start_center = start_tip - direction * PUSHER_TIP_OFFSET
    yaw = wrap_to_pi(math.atan2(direction[1], direction[0]) - math.pi)

    return PusherPlan(
        start_center_x=float(start_center[0]),
        start_center_y=float(start_center[1]),
        yaw=yaw,
        force_direction_world=direction,
    )


def make_pusher_plan_at_object_pose(
    row: dict[str, Any],
    object_x: float,
    object_y: float,
    object_yaw: float,
) -> PusherPlan:
    """把 current-box local action 转换为当前 world pose 下的 pusher 计划。"""

    contact_local = np.array(
        [
            float(row["contact_point_local_x"]),
            float(row["contact_point_local_y"]),
        ],
        dtype=float,
    )
    direction_local = np.array(
        [
            float(row["force_direction_local_x"]),
            float(row["force_direction_local_y"]),
        ],
        dtype=float,
    )
    direction_local /= float(np.linalg.norm(direction_local))
    cosine = math.cos(object_yaw)
    sine = math.sin(object_yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    contact_world = np.array([object_x, object_y]) + rotation @ contact_local
    direction_world_xy = rotation @ direction_local
    direction_world = np.array(
        [direction_world_xy[0], direction_world_xy[1], 0.0],
        dtype=float,
    )
    start_tip = contact_world - direction_world_xy * APPROACH_DISTANCE
    start_center = start_tip - direction_world_xy * PUSHER_TIP_OFFSET
    yaw = wrap_to_pi(math.atan2(direction_world_xy[1], direction_world_xy[0]) - math.pi)
    return PusherPlan(
        start_center_x=float(start_center[0]),
        start_center_y=float(start_center[1]),
        yaw=yaw,
        force_direction_world=direction_world,
    )


def set_pusher_pose(
    model,
    data,
    center_x: float,
    center_y: float,
    yaw: float,
    vx: float = 0.0,
    vy: float = 0.0,
    yaw_rate: float = 0.0,
) -> None:
    """设置 pusher 平面位姿。

    XML 中 rod_pusher body 初始 yaw 为 180 度，因此 x/y slide qpos 与世界坐标有符号差。
    """
    data.qpos[get_joint_qposadr(model, "pusher_x")] = PUSHER_BASE_X - center_x
    data.qpos[get_joint_qposadr(model, "pusher_y")] = -center_y
    data.qpos[get_joint_qposadr(model, "pusher_yaw")] = yaw
    data.qvel[get_joint_dofadr(model, "pusher_x")] = -vx
    data.qvel[get_joint_dofadr(model, "pusher_y")] = -vy
    data.qvel[get_joint_dofadr(model, "pusher_yaw")] = yaw_rate


def reset_state(model, data, row: dict[str, Any]) -> PusherPlan:
    """根据旧 candidate row 重置 object 和 pusher。"""
    mujoco.mj_resetData(model, data)

    object_body_id = get_body_id(model, OBJECT_BODY_NAME)
    base_z = float(model.body_pos[object_body_id][2])
    target_z = float(row["object_qpos_initial_z"])
    qpos_z = target_z - base_z

    qpos_map = {
        "object_x": row["object_qpos_initial_x"],
        "object_y": row["object_qpos_initial_y"],
        "object_z": qpos_z,
        "object_yaw": row["object_qpos_initial_yaw"],
    }
    qvel_map = {
        "object_x": row["object_qvel_initial_x"],
        "object_y": row["object_qvel_initial_y"],
        "object_z": 0.0,
        "object_yaw": row["object_qvel_initial_yaw"],
    }
    for name, val in qpos_map.items():
        data.qpos[get_joint_qposadr(model, name)] = val
    for name, val in qvel_map.items():
        data.qvel[get_joint_dofadr(model, name)] = val

    plan = make_pusher_plan_from_row(row)
    set_pusher_pose(model, data, plan.start_center_x, plan.start_center_y, plan.yaw)
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    return plan


def force_scale(t: float, ramp_up_s: float, hold_s: float, ramp_down_s: float) -> float:
    """计算旧 direct-force 线性 ramp 力曲线。"""
    if t < ramp_up_s:
        return t / ramp_up_s
    t_after_up = t - ramp_up_s
    if t_after_up < hold_s:
        return 1.0
    t_after_hold = t_after_up - hold_s
    if t_after_hold < ramp_down_s:
        return 1.0 - t_after_hold / ramp_down_s
    return 0.0


def apply_force_to_pusher(model, data, force_world: np.ndarray) -> None:
    """对 rod_pusher 施加力命令，不直接对 object 施力。"""
    pusher_body_id = get_body_id(model, PUSHER_BODY_NAME)
    qfrc = np.zeros(model.nv)
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_applyFT(
        model,
        data,
        force_world,
        np.zeros(3),
        data.xpos[pusher_body_id],
        pusher_body_id,
        qfrc,
    )
    data.qfrc_applied[:] = qfrc


def observe_pusher_object_contact(model, data, stats: ContactStats) -> None:
    """统计当前步的 pusher-object 接触。"""
    object_geom_id = get_named_id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM_NAME)
    pusher_geom_id = get_named_id(model, mujoco.mjtObj.mjOBJ_GEOM, PUSHER_GEOM_NAME)
    force = np.zeros(6)
    had_contact = False
    for i in range(data.ncon):
        contact = data.contact[i]
        geom_pair = {int(contact.geom1), int(contact.geom2)}
        if geom_pair != {object_geom_id, pusher_geom_id}:
            continue
        had_contact = True
        mujoco.mj_contactForce(model, data, i, force)
        stats.max_contact_force = max(stats.max_contact_force, float(np.linalg.norm(force[:3])))
    if had_contact:
        stats.contact_step_count += 1
    stats.max_contact_count = max(stats.max_contact_count, int(data.ncon))


def object_is_stopped(model, data) -> bool:
    """判断物体是否停止，沿用旧 collector 的平面速度阈值。"""
    vx = data.qvel[get_joint_dofadr(model, "object_x")]
    vy = data.qvel[get_joint_dofadr(model, "object_y")]
    yaw_rate = data.qvel[get_joint_dofadr(model, "object_yaw")]
    return (
        abs(vx) < STOP_LINEAR_VEL
        and abs(vy) < STOP_LINEAR_VEL
        and abs(yaw_rate) < STOP_YAW_VEL
    )


def _execute_push_from_current_state(
    model,
    data,
    row: dict[str, Any],
    plan: PusherPlan,
    validate_schema: bool = True,
) -> dict[str, Any]:
    """从当前 object state 执行一次 push；调用前 pusher 已放置到起点。"""

    out: dict[str, Any] = dict(row)
    object_body_id = get_body_id(model, OBJECT_BODY_NAME)
    timestep = float(model.opt.timestep)
    stats = ContactStats()

    for _ in range(CONTACT_WARMUP_STEPS):
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)
        observe_pusher_object_contact(model, data, stats)

    x0 = data.xpos[object_body_id][0].copy()
    y0 = data.xpos[object_body_id][1].copy()
    z0 = data.xpos[object_body_id][2].copy()
    yaw0 = get_object_yaw_qpos(model, data)

    ramp_up = float(row["ramp_up_s"])
    hold = float(row["hold_s"])
    ramp_down = float(row["ramp_down_s"])
    command_total_s = ramp_up + hold + ramp_down
    command_steps = int(command_total_s / timestep)
    force_magnitude = float(row["commanded_force_N"])

    for step in range(command_steps):
        t = step * timestep
        scale = force_scale(t, ramp_up, hold, ramp_down)
        force_world = plan.force_direction_world * force_magnitude * scale
        apply_force_to_pusher(model, data, force_world)
        mujoco.mj_step(model, data)
        observe_pusher_object_contact(model, data, stats)

    # 力结束后把 pusher 移回起点，避免 pusher 惯性在 settle 阶段继续推物体。
    set_pusher_pose(model, data, plan.start_center_x, plan.start_center_y, plan.yaw)
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)

    settle_steps = 0
    stopped_steps = 0
    settle_max_steps = int(SETTLE_MAX_S / timestep)
    min_settle_steps = int(MIN_SETTLE_S / timestep)
    stopped_by_threshold = False
    for step in range(settle_max_steps):
        mujoco.mj_step(model, data)
        observe_pusher_object_contact(model, data, stats)
        settle_steps = step + 1
        if object_is_stopped(model, data):
            stopped_steps += 1
        else:
            stopped_steps = 0
        if settle_steps >= min_settle_steps and stopped_steps >= 50:
            stopped_by_threshold = True
            break

    xf = data.xpos[object_body_id][0].copy()
    yf = data.xpos[object_body_id][1].copy()
    zf = data.xpos[object_body_id][2].copy()
    yawf = get_object_yaw_qpos(model, data)

    has_nan = (
        np.any(np.isnan(data.qpos))
        or np.any(np.isnan(data.qvel))
        or np.any(np.isinf(data.qpos))
        or np.any(np.isinf(data.qvel))
    )

    out["delta_x"] = float(xf - x0)
    out["delta_y"] = float(yf - y0)
    out["delta_yaw"] = float(yawf - yaw0)
    out["delta_z"] = float(zf - z0)
    out["final_qpos_x"] = float(xf)
    out["final_qpos_y"] = float(yf)
    out["final_qpos_yaw"] = float(yawf)
    out["settle_time_s"] = settle_steps * timestep
    out["simulation_unstable"] = 1 if has_nan else 0
    out["stopped_by_threshold"] = 1 if stopped_by_threshold else 0
    out["num_contacts"] = int(stats.max_contact_count)
    out["contact_success"] = 1 if stats.contact_step_count > 0 else 0
    out["quality_pass"] = (
        1
        if (
            not has_nan
            and abs(out["delta_z"]) <= 0.01
            and out["contact_success"] == 1
            and stopped_by_threshold
        )
        else 0
    )

    if validate_schema:
        errors = schema.validate_row(out)
        if errors:
            raise ValueError(
                f"candidate_id={row['candidate_id']} validation failed: {'; '.join(errors)}"
            )
    return out


def run_physical_pusher_rollout(
    model,
    data,
    row: dict[str, Any],
    validate_schema: bool = True,
) -> dict[str, Any]:
    """重置 object 后执行一个 physical-pusher rollout。"""

    plan = reset_state(model, data, row)
    return _execute_push_from_current_state(
        model,
        data,
        row,
        plan,
        validate_schema=validate_schema,
    )


def run_physical_pusher_atomic_push(
    model,
    data,
    row: dict[str, Any],
) -> dict[str, Any]:
    """保留当前 object state，执行一个 current-box local atomic push。"""

    mujoco.mj_forward(model, data)
    object_body_id = get_body_id(model, OBJECT_BODY_NAME)
    object_x = float(data.xpos[object_body_id][0])
    object_y = float(data.xpos[object_body_id][1])
    object_yaw = get_object_yaw_qpos(model, data)
    plan = make_pusher_plan_at_object_pose(
        row,
        object_x,
        object_y,
        object_yaw,
    )
    set_pusher_pose(
        model,
        data,
        plan.start_center_x,
        plan.start_center_y,
        plan.yaw,
    )
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    return _execute_push_from_current_state(
        model,
        data,
        row,
        plan,
        validate_schema=False,
    )


def collect_physical_pusher_level1_rollouts(
    max_candidates: int | None = None,
    dataset_role: str = "train",
) -> list[dict[str, Any]]:
    """生成旧 Level 1 candidates，并用 physical pusher 重新执行 rollout。"""
    model, data = load_model()
    candidates = generate_level1_candidates(dataset_role=dataset_role)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return [run_physical_pusher_rollout(model, data, row) for row in candidates]


def write_rollouts_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """按旧 schema 字段顺序写 CSV。"""
    path = Path(path)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema.FIELD_NAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
