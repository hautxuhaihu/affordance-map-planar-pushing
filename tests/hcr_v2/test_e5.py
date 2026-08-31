"""HCR V2 E5 的最小闭环决策语义测试。"""

import math
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "hcr_v2"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import run_e5
from push_core.hcr_v2.e5 import (
    BELIEF_MARGINALISED_CONTROLLER_ID,
    BELIEF_UPDATE_HORIZON,
    CERTAINTY_EQUIVALENT_CONTROLLER_ID,
    CONTROLLER_IDS,
    FULL_INFORMATION_CONTROLLER_ID,
    NOMINAL_CONTROLLER_ID,
    BeliefState,
    ClosedLoopDecisionEngine,
    make_task_query,
)


def test_task_query_uses_current_box_frame_and_initial_yaw():
    """World target 应转换到 current-box local frame，并指向初始朝向。"""

    query = make_task_query(
        current_position_xy=np.asarray([0.2, -0.1]),
        current_yaw_rad=math.pi / 2.0,
        target_position_xy=np.asarray([0.2, -0.08]),
        initial_yaw_rad=0.0,
    )

    assert np.allclose(query[:2], [0.02, 0.0], atol=1e-7)
    assert np.allclose(query[2:], [-1.0, 0.0], atol=1e-7)


def test_belief_update_stops_after_four_valid_observations():
    """E5 posterior 只允许最初四个有效 observations 进入更新。"""

    engine = object.__new__(ClosedLoopDecisionEngine)
    engine.device = torch.device("cpu")
    engine.node_outcomes = torch.zeros((2, 1, 3))
    engine.observation_scale_tensor = torch.tensor([0.010, 0.010, math.pi / 36.0])
    engine.residual_bias_standardised = torch.zeros(3)
    engine.precision = torch.eye(3)
    belief = BeliefState(log_weights=torch.log(torch.tensor([0.5, 0.5])))

    update_results = [
        engine.update_belief(belief, 0, np.zeros(3))
        for _ in range(BELIEF_UPDATE_HORIZON + 1)
    ]

    assert update_results == [True, True, True, True, False]
    assert belief.update_count == BELIEF_UPDATE_HORIZON


def test_validation_and_test_polar_grids_are_disjoint():
    """交错的 Validation/Test candidate grids 在四位 key 下不应重叠。"""

    validation = run_e5.generate_polar_candidates("validation")
    test = run_e5.generate_polar_candidates("test")
    validation_keys = {row["canonical_position_key"] for row in validation}
    test_keys = {row["canonical_position_key"] for row in test}

    assert len(validation) == 900
    assert len(test) == 864
    assert validation_keys.isdisjoint(test_keys)


def test_primary_effect_directions_follow_confirmed_hypotheses():
    """三项 paired effects 的正方向必须与正式 E5 假设一致。"""

    metrics = {
        f"{BELIEF_MARGINALISED_CONTROLLER_ID}|success": np.asarray([[1.0]]),
        f"{NOMINAL_CONTROLLER_ID}|success": np.asarray([[0.0]]),
        f"{BELIEF_MARGINALISED_CONTROLLER_ID}|final_cost": np.asarray([[0.2]]),
        f"{CERTAINTY_EQUIVALENT_CONTROLLER_ID}|final_cost": np.asarray([[0.5]]),
        f"{BELIEF_MARGINALISED_CONTROLLER_ID}|success_auc": np.asarray([[0.8]]),
        f"{CERTAINTY_EQUIVALENT_CONTROLLER_ID}|success_auc": np.asarray([[0.6]]),
    }

    effects = run_e5.primary_effect_arrays(metrics)

    assert np.allclose(
        effects["primary_hypothesis_1_episode_success_rate"], 1.0
    )
    assert np.allclose(
        effects["primary_hypothesis_2_mean_final_tnpo_cost"], 0.3
    )
    assert np.allclose(
        effects["primary_hypothesis_3_success_by_push_auc"], 0.2
    )


def test_success_by_push_confidence_band_uses_terminal_push_count():
    """Success-by-Push band 应在 controller 的成功 push 位置发生跃迁。"""

    terminal_pushes = {
        NOMINAL_CONTROLLER_ID: 20,
        CERTAINTY_EQUIVALENT_CONTROLLER_ID: 3,
        BELIEF_MARGINALISED_CONTROLLER_ID: 2,
        FULL_INFORMATION_CONTROLLER_ID: 1,
    }
    rows = []
    for controller_id in CONTROLLER_IDS:
        success = int(controller_id != NOMINAL_CONTROLLER_ID)
        rows.append(
            {
                "condition_id": "COND_000",
                "target_id": "TARGET_000",
                "target_stratum": "SE-Q1",
                "controller_id": controller_id,
                "episode_success": str(success),
                "terminal_push_count": str(terminal_pushes[controller_id]),
                "actual_tnpo_cost": "0.1",
            }
        )

    _, _, bands = run_e5.paired_two_way_bootstrap(rows, 20, 123)

    assert bands[NOMINAL_CONTROLLER_ID]["point_estimate"] == [0.0] * 20
    for controller_id in CONTROLLER_IDS:
        if controller_id == NOMINAL_CONTROLLER_ID:
            continue
        push_index = terminal_pushes[controller_id]
        expected = [0.0] * (push_index - 1) + [1.0] * (21 - push_index)
        assert bands[controller_id]["point_estimate"] == expected
        assert bands[controller_id]["ci_95_low"] == expected
        assert bands[controller_id]["ci_95_high"] == expected
