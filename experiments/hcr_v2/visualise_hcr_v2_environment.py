"""HCR V2 MuJoCo 环境可视化入口。

该入口只用于查看 HCR V2 的 infinite-plane physical-pusher 环境，
不会执行推动动作或修改模型参数。

运行方式：
    python -X utf8 experiments\\hcr_v2\\visualise_hcr_v2_environment.py
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
XML_PATH = PROJECT_ROOT / "assets" / "xml" / "msc_rod_pusher_box_hcr_v2.xml"

TABLE_GEOM_NAME = "table"
OBJECT_GEOM_NAME = "object_geom"
PUSHER_BODY_NAME = "rod_pusher"
PUSHER_GEOM_NAME = "rod_pusher_geom"
PUSHER_TIP_MARKER_NAME = "rod_pusher_tip_marker"

CAMERA_DISTANCE = 1.625
CAMERA_AZIMUTH = 90.0
CAMERA_ELEVATION = -45.0
CAMERA_LOOKAT = np.array([0.0, 0.0, 0.035])
VIEWER_SYNC_EVERY_STEPS = 20
TIME_SCALE = 0.1


def get_id(model, obj_type, name: str) -> int:
    """按名称获取 MuJoCo 对象 id。"""
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return obj_id


def configure_camera(viewer) -> None:
    """沿用 V1 viewer 的相机设置，便于直接比较环境外观。"""
    viewer.cam.distance = CAMERA_DISTANCE
    viewer.cam.azimuth = CAMERA_AZIMUTH
    viewer.cam.elevation = CAMERA_ELEVATION
    viewer.cam.lookat[:] = CAMERA_LOOKAT


def print_model_summary(model, data) -> None:
    """打印 infinite-plane 与初始几何关系的最小检查信息。"""
    table_geom_id = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM_NAME)
    object_geom_id = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM_NAME)
    pusher_body_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, PUSHER_BODY_NAME)
    pusher_geom_id = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, PUSHER_GEOM_NAME)
    tip_geom_id = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, PUSHER_TIP_MARKER_NAME)

    mujoco.mj_forward(model, data)
    object_plus_x_face = float(data.geom_xpos[object_geom_id][0]) + float(
        model.geom_size[object_geom_id][0]
    )
    tip_nearest_x = float(data.geom_xpos[tip_geom_id][0]) - float(
        model.geom_size[tip_geom_id][0]
    )
    initial_gap = tip_nearest_x - object_plus_x_face
    table_is_plane = int(model.geom_type[table_geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
    contact_pairs = [
        {int(data.contact[index].geom1), int(data.contact[index].geom2)}
        for index in range(data.ncon)
    ]
    table_object_contacts = sum(
        pair == {table_geom_id, object_geom_id} for pair in contact_pairs
    )
    pusher_object_contacts = sum(
        pair == {pusher_geom_id, object_geom_id} for pair in contact_pairs
    )

    print("=" * 72)
    cone_name = (
        "elliptic"
        if int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
        else "pyramidal"
    )
    print("HCR V2 Elliptic Infinite-Plane MuJoCo Environment")
    print("=" * 72)
    print(f"XML path: {XML_PATH}")
    print(f"table geom type: {'plane' if table_is_plane else model.geom_type[table_geom_id]}")
    print(f"table geom position: {model.geom_pos[table_geom_id]}")
    print(f"table visual size: {model.geom_size[table_geom_id]}")
    print(f"table friction: {model.geom_friction[table_geom_id]}")
    print(f"friction cone: {cone_name}")
    print(f"model nq/nv: {model.nq}/{model.nv}")
    print(f"pusher body xpos: {data.xpos[pusher_body_id]}")
    print(f"estimated initial x gap: {initial_gap: .6f} m")
    print(f"initial table-object contacts: {table_object_contacts}")
    print(f"initial pusher-object contacts: {pusher_object_contacts}")
    print("MuJoCo plane 在碰撞检测中无限延伸，当前 size 只保留 V1 的可视范围。")
    print("关闭 MuJoCo viewer 窗口即可退出。")


def sync_realtime(model, viewer) -> None:
    """同步 viewer，并沿用 V1 的慢速播放设置。"""
    step_start = time.time()
    viewer.sync()
    target_dt = model.opt.timestep * VIEWER_SYNC_EVERY_STEPS / max(TIME_SCALE, 1e-6)
    sleep_time = target_dt - (time.time() - step_start)
    if sleep_time > 0:
        time.sleep(sleep_time)


def main() -> None:
    """加载 HCR V2 XML 并打开 MuJoCo viewer。"""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print_model_summary(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_camera(viewer)
        viewer.sync()
        step = 0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            if step % VIEWER_SYNC_EVERY_STEPS == 0:
                sync_realtime(model, viewer)
            step += 1


if __name__ == "__main__":
    main()
