"""HCR V2 E2 的最小数学语义测试。"""

import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from push_core.hcr_v2.e2 import (
    FixedNodePosterior,
    PairedEstimatorSuite,
    local_motion_observation,
    make_quadrature_rule,
    parameter_error_metrics,
)


def test_quadrature_rule_unifies_1d_2d_3d_uniform_priors():
    """同一 tensor-product rule 应产生 9、81 和 729 个 primary nodes。"""

    expected_node_counts = {"friction": 9, "com": 81, "joint": 729}
    for scenario, expected_node_count in expected_node_counts.items():
        rule = make_quadrature_rule(scenario)
        posterior = FixedNodePosterior(rule)
        summary = posterior.summary()

        assert rule.node_count == expected_node_count
        assert math.isclose(float(rule.prior_weights.sum()), 1.0, abs_tol=1e-12)
        assert np.allclose(summary.mean_normalised, 0.0, atol=1e-14)
        assert np.allclose(
            summary.covariance_normalised,
            np.eye(rule.dimension) / 3.0,
            atol=1e-14,
        )
        assert math.isclose(summary.uncertainty_contraction, 0.0, abs_tol=1e-14)


def test_local_motion_observation_uses_pre_push_box_frame():
    """整体旋转后的 world displacement 应恢复为相同 local observation。"""

    observation = local_motion_observation(
        pre_position_xy=np.asarray([0.2, -0.1]),
        pre_yaw_rad=math.pi / 2.0,
        post_position_xy=np.asarray([0.2, -0.08]),
        post_yaw_rad=math.pi / 2.0 + 0.1,
    )

    assert np.allclose(observation, [0.02, 0.0, 0.1], atol=1e-14)


def test_uniform_likelihood_keeps_prior_unchanged():
    """对 hidden condition 不敏感的 observation 不应虚假收缩 posterior。"""

    posterior = FixedNodePosterior(make_quadrature_rule("friction"))
    posterior.update(np.full(9, -2.5))

    assert np.allclose(posterior.weights, posterior.rule.prior_weights)
    assert math.isclose(
        posterior.summary().uncertainty_contraction,
        0.0,
        abs_tol=1e-14,
    )


def test_paired_estimators_separate_current_and_accumulated_evidence():
    """Single-observation 应重置 prior，而 Sequential 应累积两次 evidence。"""

    rule = make_quadrature_rule("friction")
    suite = PairedEstimatorSuite(rule)
    log_likelihood = -0.5 * ((rule.nodes[:, 0] - 0.6) / 0.2) ** 2

    suite.update_from_log_likelihoods(log_likelihood)
    single_after_one = suite.posteriors["single_observation"].weights.copy()
    suite.update_from_log_likelihoods(log_likelihood)

    assert np.allclose(
        suite.posteriors["single_observation"].weights,
        single_after_one,
    )
    assert (
        suite.posteriors["sequential_bayesian"]
        .summary()
        .covariance_normalised[0, 0]
        < suite.posteriors["single_observation"]
        .summary()
        .covariance_normalised[0, 0]
    )
    assert suite.posteriors["prior_only"].update_count == 0
    assert suite.posteriors["single_observation"].update_count == 1
    assert suite.posteriors["sequential_bayesian"].update_count == 2


def test_continuous_prior_density_and_hpd_coverage():
    """无 observation 时 posterior density 应等于 bounded-uniform prior density。"""

    for scenario in ("friction", "com", "joint"):
        posterior = FixedNodePosterior(make_quadrature_rule(scenario))
        metrics = posterior.probabilistic_metrics(0.0)

        assert math.isclose(
            metrics["posterior_nll"],
            posterior.rule.dimension * math.log(2.0),
            abs_tol=1e-12,
        )
        assert metrics["hpd_covered"] == 1


def test_joint_point_metrics_keep_friction_and_com_separate():
    """Joint 不应把不同物理单位合并为一个误差。"""

    metrics = parameter_error_metrics(
        "joint",
        estimate_normalised=np.asarray([0.0, 0.0, 0.0]),
        true_normalised=np.asarray([0.5, 1.0, -1.0]),
    )

    assert math.isclose(metrics["friction_absolute_error"], 0.1)
    assert math.isclose(
        metrics["com_euclidean_error_mm"],
        math.sqrt(30.0**2 + 30.0**2),
    )


def test_student_t_log_likelihood_has_heavier_tails_than_gaussian():
    """同一 scale 下 Student-t 对大 residual 的惩罚应弱于 Gaussian。"""

    covariance = np.eye(3, dtype=np.float64)
    precision = np.eye(3, dtype=np.float64)
    residual = np.asarray([8.0, 0.0, 0.0], dtype=np.float64)
    gaussian_log_likelihood = (
        -0.5 * float(residual @ precision @ residual)
        - 0.5 * 3.0 * math.log(2.0 * math.pi)
    )
    degrees = 3.0
    student_t_log_normalisation = (
        math.lgamma((degrees + 3.0) / 2.0)
        - math.lgamma(degrees / 2.0)
        - 1.5 * math.log(degrees * math.pi)
        - 0.5 * float(np.linalg.slogdet(covariance)[1])
    )
    student_t_log_likelihood = student_t_log_normalisation - 0.5 * (
        degrees + 3.0
    ) * math.log1p(float(residual @ precision @ residual) / degrees)

    assert student_t_log_likelihood > gaussian_log_likelihood
