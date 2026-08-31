import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from network_model import (  # noqa: E402
    cramer_rao_worst_standard_deviation,
    cycle_covariance,
    current_vector_potential,
    design_matrix,
    dipole_vector_potential,
    kinematic_sagnac_delay,
    loop_area_vector,
    loop_reciprocity_response,
    loop_source_bounds,
    projected_design_matrix,
    rigid_rotating_sphere,
    segment_design_row,
    square_loop,
    unit,
    whitened_projected_design_matrix,
)


def test_segment_orientation_reversal() -> None:
    start = np.array([2.0, -1.0, 0.5])
    end = np.array([2.5, 1.0, -0.3])
    forward = segment_design_row(start, end)
    reverse = segment_design_row(end, start)
    assert np.allclose(forward, -reverse, atol=1e-13)


def test_radial_segment_has_removable_zero_limit() -> None:
    assert np.allclose(
        segment_design_row(np.array([2.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0])),
        np.zeros(3),
    )


def test_analytic_loop_matches_quadrature() -> None:
    spin = unit(np.array([0.3, -0.5, 0.8]))
    loop = square_loop(np.zeros(3), [1.0, 0.2, -0.1], 4.0)
    analytic = design_matrix([loop]) @ spin
    numeric = loop_reciprocity_response(
        loop, lambda points: dipole_vector_potential(points, spin), quadrature_order=64
    )
    assert np.allclose(analytic[0], numeric, rtol=1e-12, atol=1e-12)


def test_orthogonal_loops_have_full_rank() -> None:
    loops = [square_loop(np.zeros(3), normal, 5.0) for normal in np.eye(3)]
    matrix = design_matrix(loops)
    assert np.linalg.matrix_rank(matrix, tol=1e-11) == 3


def test_centered_concentric_coplanar_loops_are_rank_one() -> None:
    loops = [
        square_loop(np.zeros(3), [0.0, 0.0, 1.0], size)
        for size in (4.0, 5.0, 6.0)
    ]
    matrix = design_matrix(loops)
    assert np.linalg.matrix_rank(matrix, tol=1e-11) == 1


def test_coplanarity_alone_does_not_force_rank_one() -> None:
    loops = [
        square_loop(center, [0.0, 0.0, 1.0], half_side)
        for center, half_side in (
            (np.array([0.0, 0.0, 1.0]), 4.0),
            (np.array([2.0, 0.0, 1.0]), 5.0),
            (np.array([0.0, 2.0, 1.0]), 6.0),
        )
    ]
    assert np.linalg.matrix_rank(design_matrix(loops), tol=1e-11) == 3


def test_current_normalization_has_unit_dipole_coefficient() -> None:
    source = rigid_rotating_sphere(
        resolution=13, angular_velocity=np.array([0.2, 0.1, -0.15])
    )
    point = np.array([[15.0, 4.5, -3.0]])
    direct = current_vector_potential(point, source)
    dipole = dipole_vector_potential(point, source.angular_momentum)
    assert np.allclose(direct, dipole, rtol=2.0e-4, atol=1.0e-9)


def test_area_sagnac_and_crlb_formulas() -> None:
    loop = square_loop(np.zeros(3), [0.0, 0.0, 1.0], 2.0)
    area = loop_area_vector(loop)
    assert np.allclose(area, np.array([0.0, 0.0, 16.0]))
    delay = kinematic_sagnac_delay(loop, np.array([0.0, 0.0, 2.0]), 4.0)
    assert np.isclose(delay, 8.0)

    matrix = np.diag([4.0, 2.0, 0.5])
    assert np.isclose(cramer_rao_worst_standard_deviation(matrix, 0.1), 0.2)


def test_nuisance_projection_removes_one_visible_direction() -> None:
    matrix = np.eye(3)
    nuisance = np.array([[1.0], [0.0], [0.0]])
    effective = projected_design_matrix(matrix, nuisance)
    assert np.linalg.matrix_rank(effective) == 2
    assert np.allclose(effective[:, 0], 0.0)


def test_edge_noise_propagates_to_correlated_cycle_noise() -> None:
    cycles = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    covariance = cycle_covariance(cycles, 2.0)
    assert np.allclose(covariance, np.array([[8.0, 4.0], [4.0, 8.0]]))


def test_whitened_nuisance_projection() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    nuisance = np.ones((3, 1))
    covariance = np.array(
        [[2.0, 0.5, 0.0], [0.5, 1.5, 0.2], [0.0, 0.2, 1.0]]
    )
    effective = whitened_projected_design_matrix(
        matrix, nuisance, covariance
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = (
        eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    )
    whitened_nuisance = whitening @ nuisance
    # The effective columns lie in the complement of the whitened nuisance.
    assert np.allclose(whitened_nuisance.T @ effective, 0.0, atol=1.0e-12)


def test_source_bound_hierarchy() -> None:
    source = rigid_rotating_sphere(resolution=7)
    loop = square_loop(np.array([3.0, 0.0, 0.0]), [1.0, 0.0, 0.0], 0.8)
    response = abs(
        loop_reciprocity_response(
            loop,
            lambda points: current_vector_potential(points, source),
            quadrature_order=12,
        )
    )
    bounds = loop_source_bounds(loop, source, quadrature_order=10)
    assert response <= bounds["kernel_bound"]
    assert bounds["kernel_bound"] <= bounds["separated_bound"]
    assert bounds["separated_bound"] <= bounds["energy_bound"]
