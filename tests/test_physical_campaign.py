import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from network_model import kinematic_sagnac_delay  # noqa: E402
from run_physical_campaign import (  # noqa: E402
    GALILEO_INCLINATION_DEG,
    GALILEO_ORBIT_RADIUS_M,
    SECOND_ORBIT_RADIUS_M,
    nuisance_dimension_budget,
    normal,
    orbital_angular_speed,
    orbit_loop,
    scalar_orbital_sagnac_nuisance,
    timing_design_matrix,
)
from network_model import projected_design_matrix  # noqa: E402


def test_galileo_orbital_angular_speed_and_period() -> None:
    omega = orbital_angular_speed(GALILEO_ORBIT_RADIUS_M)
    period_hours = 2.0 * np.pi / omega / 3600.0
    assert np.isclose(omega, 1.2397420193713848e-4, rtol=1.0e-12)
    assert np.isclose(period_hours, 14.078164849807255, rtol=1.0e-12)


def test_orbital_sagnac_uses_orbit_plane_normal() -> None:
    node = 120.0
    loop = orbit_loop(GALILEO_ORBIT_RADIUS_M, node)
    omega_vector = orbital_angular_speed(GALILEO_ORBIT_RADIUS_M) * normal(
        GALILEO_INCLINATION_DEG, node
    )
    delay = kinematic_sagnac_delay(loop, omega_vector)
    assert np.isclose(delay, 9.668594014392951e-6, rtol=1.0e-12)


def test_keplerian_sagnac_template_scales_as_square_root_radius() -> None:
    outer = orbit_loop(GALILEO_ORBIT_RADIUS_M, 0.0)
    inner = orbit_loop(SECOND_ORBIT_RADIUS_M, 0.0)
    template = scalar_orbital_sagnac_nuisance([outer, inner])[:, 0]
    assert np.isclose(template[1] / template[0], np.sqrt(0.78), rtol=1.0e-12)


def test_six_cycle_nuisance_dimension_budget() -> None:
    budget = nuisance_dimension_budget(6).set_index("nuisance_dimension_p")
    assert budget.spin_rank_ceiling.to_dict() == {1: 3, 2: 3, 3: 3, 4: 2}
    assert budget.dimension_margin_Nc_minus_3_minus_p.to_dict() == {
        1: 2,
        2: 1,
        3: 0,
        4: -1,
    }
    assert bool(budget.loc[3, "full_spin_rank_dimensionally_possible"])
    assert not bool(budget.loc[4, "full_spin_rank_dimensionally_possible"])


def test_radius_tolerance_conversions_differ_by_factor_four() -> None:
    fractional_control = 3.027e-12
    area_only = 0.5 * fractional_control * GALILEO_ORBIT_RADIUS_M
    kepler_consistent = 2.0 * fractional_control * GALILEO_ORBIT_RADIUS_M
    assert np.isclose(kepler_consistent / area_only, 4.0, rtol=1.0e-15)


def test_symmetric_three_plane_design_is_nearly_but_not_exactly_isotropic() -> None:
    loops = [orbit_loop(GALILEO_ORBIT_RADIUS_M, node) for node in (0.0, 120.0, 240.0)]
    matrix = timing_design_matrix(loops)
    singular = np.linalg.svd(matrix, compute_uv=False)
    spin_axis_delays = matrix @ np.array([0.0, 0.0, 1.0])
    assert np.isclose(singular[0], singular[1], rtol=1.0e-12)
    assert singular[1] > singular[2]
    assert np.isclose(
        singular[2], np.sqrt(3.0) * np.max(np.abs(spin_axis_delays)), rtol=1.0e-12
    )
    assert np.isclose(singular[0] / singular[2], 1.0483289143578542, rtol=1.0e-12)


def test_multiradius_operational_sample_count() -> None:
    nodes = (0.0, 120.0, 240.0)
    loops = [orbit_loop(GALILEO_ORBIT_RADIUS_M, node) for node in nodes]
    loops += [orbit_loop(SECOND_ORBIT_RADIUS_M, node) for node in nodes]
    effective = projected_design_matrix(
        timing_design_matrix(loops), scalar_orbital_sagnac_nuisance(loops)
    )
    sigma_min = np.linalg.svd(effective, compute_uv=False)[-1]
    assert np.linalg.matrix_rank(effective) == 3
    assert np.isclose(sigma_min, 1.515445145300242e-17, rtol=1.0e-12)
    assert int(np.ceil((1.0e-15 / sigma_min) ** 2)) == 4355
