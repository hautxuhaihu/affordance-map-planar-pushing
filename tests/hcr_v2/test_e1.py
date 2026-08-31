"""HCR V2 E1 的最小语义测试。"""

from itertools import product
import math
from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from push_core.hcr_v2.e1 import (
    PRIMARY_TNPO_COST,
    SCENARIO_ACTIVE_COORDINATES,
    TensorOutcomeInterpolator,
    action_distance_components,
    cost_matrix,
    evaluate_selector,
    paired_two_way_bootstrap_mean_ci,
    pose_error_matrices,
)


def test_primary_tnpo_cost_equal_tolerance_importance():
    """一个 position tolerance 与一个 yaw tolerance 应产生相同代价。"""

    position_error = np.asarray([[0.010, 0.0]])
    yaw_error = np.asarray([[0.0, math.radians(5.0)]])
    cost = cost_matrix("v2", position_error, yaw_error, PRIMARY_TNPO_COST)

    assert np.allclose(cost, [[0.5, 0.5]])


def test_pose_error_wraps_yaw_to_initial_orientation():
    """V2 yaw error 应使用相对初始方向的 wrapped angle。"""

    outcomes = np.asarray([[0.0, 0.0, 2.0 * math.pi - math.radians(1.0)]])
    position, yaw = pose_error_matrices(outcomes, np.asarray([[0.0, 0.0]]))

    assert np.allclose(position, [[0.0]])
    assert np.allclose(yaw, [[math.radians(1.0)]])


def test_predicted_selector_cannot_use_mujoco_validity_mask():
    """预测最优但 MuJoCo 无效的 action 必须被记录为 selector failure。"""

    actual = np.asarray(
        [
            [0.010, 0.0, 0.0],
            [0.020, 0.0, 0.0],
            [0.010, 0.0, 0.0],
        ]
    )
    predicted = np.asarray(
        [
            [0.000, 0.0, 0.0],
            [0.020, 0.0, 0.0],
            [0.010, 0.0, 0.0],
        ]
    )
    target = np.asarray([[0.010, 0.0]])
    valid = np.asarray([True, True, False])
    result = evaluate_selector(actual, predicted, target, valid)

    assert result["selected_indices"].tolist() == [2]
    assert result["selected_valid"].tolist() == [False]
    assert result["selected_success"].tolist() == [False]
    assert result["near_optimal"].tolist() == [False]
    assert np.isnan(result["selected_actual_cost"][0])


def make_linear_interpolation_rows(
    scenario: str,
    action_count: int = 2,
) -> tuple[list[dict[str, str]], list[dict[str, float | str]]]:
    """构造对 multilinear interpolation 可精确恢复的 synthetic grid。"""

    action_rows = [{"v2_action_id": f"A{index:04d}"} for index in range(action_count)]
    coordinate_fields = SCENARIO_ACTIVE_COORDINATES[scenario]
    anchors = (-1.0, -0.5, 0.0, 0.5, 1.0)
    outcome_rows: list[dict[str, float | str]] = []
    for coordinates in product(anchors, repeat=len(coordinate_fields)):
        for action_index, action in enumerate(action_rows):
            base = sum((axis + 1.0) * value for axis, value in enumerate(coordinates))
            outcome_rows.append(
                {
                    "v2_action_id": action["v2_action_id"],
                    **dict(zip(coordinate_fields, coordinates)),
                    "real_delta_x": base + action_index,
                    "real_delta_y": 2.0 * base - action_index,
                    "real_delta_yaw": -base + 0.5 * action_index,
                }
            )
    return action_rows, outcome_rows


def test_tensor_interpolator_is_exact_for_linear_1d_2d_3d():
    """同一 P1 实现应对 1D、2D、3D linear outcomes 精确插值。"""

    query_values = {
        "hidden_u_friction": 0.25,
        "hidden_u_com_x": -0.25,
        "hidden_u_com_y": 0.75,
    }
    for scenario in ("friction", "com", "joint"):
        action_rows, outcome_rows = make_linear_interpolation_rows(scenario)
        model = TensorOutcomeInterpolator.fit(scenario, action_rows, outcome_rows)
        prediction = model.predict(query_values)
        coordinates = [
            query_values[field] for field in SCENARIO_ACTIVE_COORDINATES[scenario]
        ]
        base = sum((axis + 1.0) * value for axis, value in enumerate(coordinates))
        expected = np.asarray(
            [
                [base, 2.0 * base, -base],
                [base + 1.0, 2.0 * base - 1.0, -base + 0.5],
            ]
        )
        assert np.allclose(prediction, expected, atol=1e-7)
        assert np.allclose(
            model.predict_action(query_values, "A0001"),
            expected[1],
            atol=1e-7,
        )

        boundary_query = {field: 1.0 for field in SCENARIO_ACTIVE_COORDINATES[scenario]}
        boundary_prediction = model.predict(boundary_query)
        boundary_base = sum(
            axis + 1.0 for axis in range(len(SCENARIO_ACTIVE_COORDINATES[scenario]))
        )
        boundary_expected = np.asarray(
            [
                [boundary_base, 2.0 * boundary_base, -boundary_base],
                [
                    boundary_base + 1.0,
                    2.0 * boundary_base - 1.0,
                    -boundary_base + 0.5,
                ],
            ]
        )
        assert np.allclose(boundary_prediction, boundary_expected, atol=1e-7)


def test_two_way_bootstrap_constant_matrix_has_exact_interval():
    """常量 crossed matrix 的 bootstrap interval 应退化到同一个值。"""

    low, high = paired_two_way_bootstrap_mean_ci(
        np.full((8, 4), 2.5),
        n_resamples=100,
        seed=1,
    )

    assert low == 2.5
    assert high == 2.5


def test_action_distance_components_are_zero_for_same_action():
    """同一个 action 的四类归一化距离都应为零。"""

    action = {
        "contact_point_local_x": 0.045,
        "contact_point_local_y": 0.0,
        "force_direction_local_x": -1.0,
        "force_direction_local_y": 0.0,
        "commanded_force_N": 0.8,
        "ramp_up_s": 0.04,
        "hold_s": 0.10,
        "ramp_down_s": 0.02,
    }
    library = [
        {**action, "commanded_force_N": 0.5, "ramp_up_s": 0.02, "hold_s": 0.05},
        action,
        {**action, "commanded_force_N": 1.1, "ramp_up_s": 0.06, "hold_s": 0.15, "ramp_down_s": 0.04},
    ]
    distances = action_distance_components(action, action, library)

    assert all(math.isclose(value, 0.0, abs_tol=1e-12) for value in distances.values())


def test_hcr_v2_environment_uses_plane():
    """E1 默认 XML 的支撑面必须保持为 MuJoCo plane。"""

    xml_path = PROJECT_ROOT / "assets" / "xml" / "msc_rod_pusher_box_hcr_v2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")

    assert table_id >= 0
    assert int(model.geom_type[table_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
