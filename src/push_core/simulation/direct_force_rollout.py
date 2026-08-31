"""Level 1 rollout collector for direct-region-force dataset.

对每个 candidate，在 MuJoCo 中执行 direct-region-force rollout，
填写真实的 delta_x/y/yaw/z、final state 和质量诊断字段。
"""

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from push_core.schema import level1_schema as schema
from push_core.action_space.candidates import generate_level1_candidates
from push_core.project_paths import ASSETS_DIR

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

DEFAULT_XML_PATH = (
    ASSETS_DIR / "xml" / "direct_force_box.xml"
)

CONTACT_WARMUP_STEPS = 30  # 施力前接触预热步数
SETTLE_MAX_S = 2.0  # 等待停止最大时间 (s)
STOP_LINEAR_VEL = 0.005  # 停止判断：线速度阈值 (m/s)
STOP_YAW_VEL = 0.02  # 停止判断：角速度阈值 (rad/s)
MIN_SETTLE_S = 0.10  # 最短等待停止时间 (s)


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────


def wrap_to_pi(angle: float) -> float:
    """将角度归一化到 [-pi, pi]。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def get_joint_qposadr(model, joint_name: str) -> int:
    """获取 joint 在 qpos 中的起始索引。"""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Joint not found: {joint_name}")
    return model.jnt_qposadr[joint_id]


def get_joint_dofadr(model, joint_name: str) -> int:
    """获取 joint 在 qvel 中的起始索引。"""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Joint not found: {joint_name}")
    return model.jnt_dofadr[joint_id]


def get_body_id(model, body_name: str = "object") -> int:
    """获取 object body 的 id。"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Body not found: {body_name}")
    return body_id


def quat_to_yaw(quat: np.ndarray) -> float:
    """从 MuJoCo 四元数 [qw, qx, qy, qz] 提取 yaw。"""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def get_object_yaw_qpos(model, data) -> float:
    """读取 object_yaw hinge joint 的累计角度，保留超过一周的 unwrapped yaw。"""
    return float(data.qpos[get_joint_qposadr(model, "object_yaw")])


# ──────────────────────────────────────────────
# 模型加载与状态重置
# ──────────────────────────────────────────────


def load_model(xml_path: str | Path | None = None):
    """加载 MuJoCo 模型，返回 (model, data)。"""
    path = Path(xml_path) if xml_path else DEFAULT_XML_PATH
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    return model, data


def reset_object_state(model, data, row: dict[str, Any]) -> None:
    """根据 candidate row 重置物体状态。

    XML 中 body 已有 pos="0 0 0.045"，object_z 是 slide joint 偏移，
    因此 qpos_z = 目标世界高度 - body 基准高度。
    """
    mujoco.mj_resetData(model, data)

    body_id = get_body_id(model)
    base_z = float(model.body_pos[body_id][2])
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
        "object_z": 0.0,  # z 初始速度固定为 0
        "object_yaw": row["object_qvel_initial_yaw"],
    }

    for name, val in qpos_map.items():
        data.qpos[get_joint_qposadr(model, name)] = val
    for name, val in qvel_map.items():
        data.qvel[get_joint_dofadr(model, name)] = val

    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


# ──────────────────────────────────────────────
# 坐标转换
# ──────────────────────────────────────────────


def local_vector_to_world(
    model, data, body_name: str, vec_local: tuple[float, float, float]
) -> np.ndarray:
    """将局部方向向量转换到世界坐标。"""
    body_id = get_body_id(model, body_name)
    R = data.xmat[body_id].reshape(3, 3)
    return R @ np.array(vec_local, dtype=float)


def local_point_to_world(
    model, data, body_name: str, point_local: tuple[float, float, float]
) -> np.ndarray:
    """将局部坐标点转换到世界坐标。"""
    body_id = get_body_id(model, body_name)
    R = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + R @ np.array(point_local, dtype=float)


# ──────────────────────────────────────────────
# 力控制
# ──────────────────────────────────────────────


def force_scale(t: float, ramp_up_s: float, hold_s: float, ramp_down_s: float) -> float:
    """计算当前时间的力缩放系数（线性 ramps）。"""
    if t < ramp_up_s:
        return t / ramp_up_s
    t_after_up = t - ramp_up_s
    if t_after_up < hold_s:
        return 1.0
    t_after_hold = t_after_up - hold_s
    if t_after_hold < ramp_down_s:
        return 1.0 - t_after_hold / ramp_down_s
    return 0.0


def object_is_stopped(model, data) -> bool:
    """判断物体是否已停止（只检查平面速度）。"""
    vx = data.qvel[get_joint_dofadr(model, "object_x")]
    vy = data.qvel[get_joint_dofadr(model, "object_y")]
    yaw_rate = data.qvel[get_joint_dofadr(model, "object_yaw")]
    return (
        abs(vx) < STOP_LINEAR_VEL
        and abs(vy) < STOP_LINEAR_VEL
        and abs(yaw_rate) < STOP_YAW_VEL
    )


# ──────────────────────────────────────────────
# 单行 rollout
# ──────────────────────────────────────────────


def run_candidate_rollout(model, data, row: dict[str, Any]) -> dict[str, Any]:
    """在 MuJoCo 中执行一个 candidate 的 rollout，返回完整的 61 字段行。"""
    out: dict[str, Any] = dict(row)
    body_id = get_body_id(model)
    timestep = model.opt.timestep

    # 重置状态
    reset_object_state(model, data, row)

    # 接触预热
    for _ in range(CONTACT_WARMUP_STEPS):
        data.qfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)

    # 记录初始位姿
    x0 = data.xpos[body_id][0].copy()
    y0 = data.xpos[body_id][1].copy()
    z0 = data.xpos[body_id][2].copy()
    yaw0 = get_object_yaw_qpos(model, data)

    # 施力参数
    ramp_up = row["ramp_up_s"]
    hold = row["hold_s"]
    ramp_down = row["ramp_down_s"]
    command_total_s = ramp_up + hold + ramp_down
    command_steps = int(command_total_s / timestep)

    force_dir_local = (
        row["force_direction_local_x"],
        row["force_direction_local_y"],
        row["force_direction_local_z"],
    )
    # 动作开始时将局部方向转为世界方向，后续保持固定（外部操作者沿世界方向推）
    force_dir_world_fixed = local_vector_to_world(
        model, data, "object", force_dir_local
    )
    force_magnitude = row["commanded_force_N"]

    # Phase 1: 施力
    for step in range(command_steps):
        t = step * timestep
        scale = force_scale(t, ramp_up, hold, ramp_down)

        # 施力点跟随物体表面；力方向保持世界方向固定
        point_local = (
            row["contact_point_local_x"],
            row["contact_point_local_y"],
            row["contact_point_local_z"],
        )
        point_world = local_point_to_world(model, data, "object", point_local)
        force_world = force_dir_world_fixed * force_magnitude * scale

        data.qfrc_applied[:] = 0.0
        torque_world = np.zeros(3)
        qfrc = np.zeros(model.nv)
        mujoco.mj_applyFT(
            model, data, force_world, torque_world, point_world, body_id, qfrc
        )
        data.qfrc_applied[:] = qfrc
        mujoco.mj_step(model, data)

    # Phase 2: 等待停止
    data.qfrc_applied[:] = 0.0
    settle_steps = 0
    stopped_steps = 0
    settle_max_steps = int(SETTLE_MAX_S / timestep)
    min_settle_steps = int(MIN_SETTLE_S / timestep)
    stopped_by_threshold = False

    for step in range(settle_max_steps):
        mujoco.mj_step(model, data)
        settle_steps = step + 1

        if object_is_stopped(model, data):
            stopped_steps += 1
        else:
            stopped_steps = 0

        if settle_steps >= min_settle_steps and stopped_steps >= 50:
            stopped_by_threshold = True
            break

    settle_time_s = settle_steps * timestep

    # 读取最终状态
    xf = data.xpos[body_id][0].copy()
    yf = data.xpos[body_id][1].copy()
    zf = data.xpos[body_id][2].copy()
    yawf = get_object_yaw_qpos(model, data)

    # 仿真稳定性检查
    has_nan = (
        np.any(np.isnan(data.qpos))
        or np.any(np.isnan(data.qvel))
        or np.any(np.isinf(data.qpos))
        or np.any(np.isinf(data.qvel))
    )

    # 填入真实值
    out["delta_x"] = float(xf - x0)
    out["delta_y"] = float(yf - y0)
    out["delta_yaw"] = float(yawf - yaw0)
    out["delta_z"] = float(zf - z0)

    out["final_qpos_x"] = float(xf)
    out["final_qpos_y"] = float(yf)
    out["final_qpos_yaw"] = float(yawf)
    out["settle_time_s"] = settle_time_s

    out["simulation_unstable"] = 1 if has_nan else 0
    out["stopped_by_threshold"] = 1 if stopped_by_threshold else 0
    out["num_contacts"] = int(data.ncon)
    out["contact_success"] = 1 if data.ncon > 0 else 0

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

    # 校验
    errors = schema.validate_row(out)
    if errors:
        raise ValueError(
            f"candidate_id={row['candidate_id']} validation failed: {'; '.join(errors)}"
        )
    return out


# ──────────────────────────────────────────────
# 批量采集
# ──────────────────────────────────────────────


def collect_level1_rollouts(
    max_candidates: int | None = None,
    dataset_role: str = "train",
) -> list[dict[str, Any]]:
    """生成 Level 1 candidate 并执行 rollout，返回完整结果行列表。"""
    model, data = load_model()
    candidates = generate_level1_candidates(dataset_role=dataset_role)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    results: list[dict[str, Any]] = []
    for cand in candidates:
        result = run_candidate_rollout(model, data, cand)
        results.append(result)
    return results


# ──────────────────────────────────────────────
# CSV 写出
# ──────────────────────────────────────────────


def write_rollouts_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """将 rollout 结果行写入 CSV。使用 utf-8-sig 编码。"""
    import csv

    path = Path(path)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema.FIELD_NAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
