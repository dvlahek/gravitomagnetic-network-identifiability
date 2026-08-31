"""Earth/Galileo-scale timing, nuisance, and E-optimal design campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from network_model import (
    cramer_rao_worst_standard_deviation,
    design_matrix,
    kinematic_sagnac_delay,
    loop_area_vector,
    projected_design_matrix,
    square_loop,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

G_SI = 6.67430e-11
C_SI = 299_792_458.0
EARTH_MASS_KG = 5.9722e24
EARTH_MU_M3_S2 = 3.986004418e14
EARTH_RADIUS_M = 6_378_137.0
EARTH_OMEGA_RAD_S = 7.2921150e-5
EARTH_INERTIA_FACTOR = 0.3307
GALILEO_ORBIT_RADIUS_M = 29_600_000.0
GALILEO_INCLINATION_DEG = 56.0
SECOND_RADIUS_FACTOR = 0.78
SECOND_ORBIT_RADIUS_M = SECOND_RADIUS_FACTOR * GALILEO_ORBIT_RADIUS_M
REPRESENTATIVE_GALILEO_ORBIT_RESIDUAL_M = 0.05
SPIN_PARAMETER_DIMENSION = 3
SIX_CYCLE_COUNT = 6


def latex_scientific(value: float, digits: int = 3) -> str:
    """Format a number as LaTeX scientific notation for generated macros."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def normal(inclination_deg: float, ascending_node_deg: float) -> np.ndarray:
    inclination = np.deg2rad(inclination_deg)
    node = np.deg2rad(ascending_node_deg)
    return np.array(
        [
            np.sin(inclination) * np.cos(node),
            np.sin(inclination) * np.sin(node),
            np.cos(inclination),
        ]
    )


def orbit_loop(radius_m: float, ascending_node_deg: float) -> np.ndarray:
    # Vertices lie at the chosen geocentric radius because their distance from
    # the origin is sqrt(2) times square_loop's half-side.
    return square_loop(
        np.zeros(3),
        normal(GALILEO_INCLINATION_DEG, ascending_node_deg),
        radius_m / np.sqrt(2.0),
    )


def earth_angular_momentum_si() -> float:
    return float(
        EARTH_INERTIA_FACTOR
        * EARTH_MASS_KG
        * EARTH_RADIUS_M**2
        * EARTH_OMEGA_RAD_S
    )


def orbital_angular_speed(radius_m: float) -> float:
    """Circular Keplerian orbital angular speed at geocentric radius."""
    return float(np.sqrt(EARTH_MU_M3_S2 / float(radius_m) ** 3))


def timing_design_matrix(loops: list[np.ndarray]) -> np.ndarray:
    angular_momentum_geometric_m2 = G_SI * earth_angular_momentum_si() / C_SI**3
    return design_matrix(loops) * angular_momentum_geometric_m2 / C_SI


def optimize_three_planes(step_deg: float) -> tuple[list[float], float]:
    """Fix one node at zero and maximize sigma_min over the other two."""
    candidates = np.arange(step_deg, 360.0, step_deg)
    best_nodes: list[float] | None = None
    best_value = -np.inf
    for second_index, second in enumerate(candidates):
        for third in candidates[second_index + 1 :]:
            loops = [
                orbit_loop(GALILEO_ORBIT_RADIUS_M, angle)
                for angle in (0.0, float(second), float(third))
            ]
            sigma_min = float(np.linalg.svd(timing_design_matrix(loops), compute_uv=False)[-1])
            if sigma_min > best_value:
                best_value = sigma_min
                best_nodes = [0.0, float(second), float(third)]
    if best_nodes is None:
        raise RuntimeError("The design scan did not retain a candidate.")
    return best_nodes, best_value


def scalar_orbital_sagnac_nuisance(loops: list[np.ndarray]) -> np.ndarray:
    """Known orbital-Sagnac template with one unknown fractional amplitude."""
    column = []
    for loop in loops:
        area = loop_area_vector(loop)
        radius = float(np.mean(np.linalg.norm(loop, axis=1)))
        plane_normal = area / np.linalg.norm(area)
        omega_vector = orbital_angular_speed(radius) * plane_normal
        column.append(kinematic_sagnac_delay(loop, omega_vector))
    return np.asarray(column, dtype=float)[:, None]


def nuisance_dimension_budget(
    cycle_count: int, max_nuisance_dimension: int = 4
) -> pd.DataFrame:
    """Necessary dimension budget, independent of nuisance-column orientation.

    For rank(H)=p, rank(P_H^perp Q) cannot exceed min(3, N_c-p).  This
    table deliberately reports a ceiling rather than inventing additional
    nuisance templates whose physical definitions have not been specified.
    """
    rows = []
    for nuisance_dimension in range(1, max_nuisance_dimension + 1):
        complement_dimension = max(cycle_count - nuisance_dimension, 0)
        rank_ceiling = min(SPIN_PARAMETER_DIMENSION, complement_dimension)
        rows.append(
            {
                "cycle_count": cycle_count,
                "nuisance_dimension_p": nuisance_dimension,
                "complement_dimension_Nc_minus_p": complement_dimension,
                "spin_rank_ceiling": rank_ceiling,
                "dimension_margin_Nc_minus_3_minus_p": (
                    cycle_count - SPIN_PARAMETER_DIMENSION - nuisance_dimension
                ),
                "full_spin_rank_dimensionally_possible": (
                    rank_ceiling == SPIN_PARAMETER_DIMENSION
                ),
            }
        )
    return pd.DataFrame(rows)


def geometry_row(
    label: str,
    loops: list[np.ndarray],
    nodes_deg: list[float],
    nuisance: bool,
) -> dict[str, float | int | str]:
    matrix = timing_design_matrix(loops)
    effective = (
        projected_design_matrix(matrix, scalar_orbital_sagnac_nuisance(loops))
        if nuisance
        else matrix
    )
    singular = np.linalg.svd(effective, compute_uv=False)
    rank_tolerance = max(effective.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > rank_tolerance))
    sigma_min = float(singular[-1])
    radii = np.asarray(
        [float(np.mean(np.linalg.norm(loop, axis=1))) for loop in loops]
    )
    distinct_radii = np.unique(np.round(radii, decimals=6))
    return {
        "geometry": label,
        "cycle_count": len(loops),
        "nodes_deg": ";".join(f"{value:.1f}" for value in nodes_deg),
        "radii_m": ";".join(f"{value:.1f}" for value in radii),
        "distinct_radius_count": len(distinct_radii),
        "nuisance_projected": nuisance,
        "nuisance_columns": 1 if nuisance else 0,
        "nuisance_structure": (
            "one shared fractional orbital-Sagnac amplitude" if nuisance else "none"
        ),
        "rank": rank,
        "sigma_1_s_per_earth_spin": float(singular[0]),
        "sigma_2_s_per_earth_spin": float(singular[1]),
        "sigma_3_s_per_earth_spin": sigma_min,
        "condition_number": float(singular[0] / sigma_min) if sigma_min > rank_tolerance else np.inf,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    step_deg = 15.0 if args.mode == "smoke" else 2.0

    clustered_nodes = [0.0, 15.0, 30.0]
    optimized_nodes, optimized_sigma = optimize_three_planes(step_deg)
    clustered_loops = [
        orbit_loop(GALILEO_ORBIT_RADIUS_M, angle) for angle in clustered_nodes
    ]
    optimized_loops = [
        orbit_loop(GALILEO_ORBIT_RADIUS_M, angle) for angle in optimized_nodes
    ]

    # A second radius breaks the exact radial scaling degeneracy between the
    # circular-orbit Sagnac template (omega_orb R^2 scales as R^(1/2)) and the
    # dipole response (design rows scale as R^(-1)).
    augmented_nodes = optimized_nodes + optimized_nodes
    augmented_loops = optimized_loops + [
        orbit_loop(SECOND_ORBIT_RADIUS_M, angle)
        for angle in optimized_nodes
    ]

    geometry = pd.DataFrame(
        [
            geometry_row("clustered_three", clustered_loops, clustered_nodes, False),
            geometry_row("optimized_three", optimized_loops, optimized_nodes, False),
            geometry_row(
                "optimized_three_with_orbital_nuisance",
                optimized_loops,
                optimized_nodes,
                True,
            ),
            geometry_row(
                "multiradius_six_with_orbital_nuisance",
                augmented_loops,
                augmented_nodes,
                True,
            ),
        ]
    )
    geometry.to_csv(RESULTS / "physical_geometry_design.csv", index=False)
    six_cycle_sigma_min = float(
        geometry.loc[
            geometry.geometry == "multiradius_six_with_orbital_nuisance",
            "sigma_3_s_per_earth_spin",
        ].iloc[0]
    )
    dimension_budget = nuisance_dimension_budget(SIX_CYCLE_COUNT)
    dimension_budget["computed_projected_rank"] = pd.array(
        [int(geometry.loc[
            geometry.geometry == "multiradius_six_with_orbital_nuisance", "rank"
        ].iloc[0]), None, None, None],
        dtype="Int64",
    )
    dimension_budget["status"] = [
        "computed for the specified scalar orbital-Sagnac template",
        "full rank is dimensionally possible; nuisance columns not specified",
        "boundary case with no dimension margin; nuisance columns not specified",
        "full spin rank is impossible for six cycles",
    ]
    dimension_budget.to_csv(RESULTS / "nuisance_dimension_budget.csv", index=False)
    earth_spin_direction = np.array([0.0, 0.0, 1.0])
    timing_matrix = timing_design_matrix(optimized_loops)
    gm_delays = timing_matrix @ earth_spin_direction
    galileo_orbital_omega = orbital_angular_speed(GALILEO_ORBIT_RADIUS_M)
    galileo_orbital_period_s = 2.0 * np.pi / galileo_orbital_omega
    kinematic_delays = np.array([
        kinematic_sagnac_delay(
            loop,
            galileo_orbital_omega
            * normal(GALILEO_INCLINATION_DEG, node),
        )
        for loop, node in zip(optimized_loops, optimized_nodes)
    ])
    timing_floor_s = 1.0e-15
    operational_crlb_worst = timing_floor_s / six_cycle_sigma_min
    operational_sample_count = int(np.ceil(operational_crlb_worst**2))
    sigma_min = float(np.linalg.svd(timing_matrix, compute_uv=False)[-1])
    optimized_singular_values = np.linalg.svd(timing_matrix, compute_uv=False)
    crlb_worst = cramer_rao_worst_standard_deviation(timing_matrix, timing_floor_s)
    max_gm = float(np.max(np.abs(gm_delays)))
    max_kinematic = float(np.max(np.abs(kinematic_delays)))
    required_fractional_subtraction = max_gm / max_kinematic
    area_only_radius_tolerance_m = (
        0.5 * required_fractional_subtraction * GALILEO_ORBIT_RADIUS_M
    )
    # For a circular Keplerian orbit, omega*S scales as R^(1/2), so
    # delta(omega*S)/(omega*S) = (1/2) delta R/R.
    kepler_consistent_radius_tolerance_m = (
        2.0 * required_fractional_subtraction * GALILEO_ORBIT_RADIUS_M
    )
    design_gain = optimized_sigma / float(
        np.linalg.svd(timing_design_matrix(clustered_loops), compute_uv=False)[-1]
    )
    ideal_sample_count = int(np.ceil(crlb_worst**2))
    earth_energy_geometric_m = G_SI * EARTH_MASS_KG / C_SI**2
    earth_vmax_over_c = EARTH_OMEGA_RAD_S * EARTH_RADIUS_M / C_SI
    square_cycle_length_m = 4.0 * np.sqrt(2.0) * GALILEO_ORBIT_RADIUS_M
    source_cycle_separation_m = (
        GALILEO_ORBIT_RADIUS_M / np.sqrt(2.0) - EARTH_RADIUS_M
    )
    earth_geometry_bound_factor = (
        8.0 * square_cycle_length_m / source_cycle_separation_m
    )
    earth_energy_bound_s = (
        earth_geometry_bound_factor
        * earth_vmax_over_c
        * earth_energy_geometric_m
        / C_SI
    )

    benchmark = pd.DataFrame(
        {
            "cycle": np.arange(1, len(optimized_loops) + 1),
            "ascending_node_deg": optimized_nodes,
            "gravitomagnetic_delay_s": gm_delays,
            "orbital_sagnac_delay_s": kinematic_delays,
            "absolute_ratio_orbital_sagnac_to_gravitomagnetic": np.abs(kinematic_delays)
            / np.maximum(np.abs(gm_delays), np.finfo(float).tiny),
        }
    )
    benchmark.to_csv(RESULTS / "earth_galileo_benchmark.csv", index=False)

    summary = {
        "run_mode": args.mode,
        "publication_results": args.mode == "full",
        "optimization_step_deg": step_deg,
        "kinematic_model": "circular Keplerian orbital Sagnac",
        "galileo_orbital_omega_rad_s": galileo_orbital_omega,
        "galileo_orbital_period_s": galileo_orbital_period_s,
        "earth_angular_momentum_kg_m2_s": earth_angular_momentum_si(),
        "optimized_nodes_deg": optimized_nodes,
        "clustered_sigma_min_s_per_earth_spin": float(
            np.linalg.svd(timing_design_matrix(clustered_loops), compute_uv=False)[-1]
        ),
        "optimized_sigma_min_s_per_earth_spin": optimized_sigma,
        "optimized_singular_values_s_per_earth_spin": optimized_singular_values.tolist(),
        "optimized_condition_number": float(
            optimized_singular_values[0] / optimized_singular_values[-1]
        ),
        "sigma_min_to_sqrt3_max_spin_axis_delay": float(
            sigma_min / (np.sqrt(3.0) * max_gm)
        ),
        "max_gravitomagnetic_cycle_delay_s": max_gm,
        "max_kinematic_sagnac_delay_s": max_kinematic,
        "kinematic_to_gravitomagnetic_ratio": max_kinematic / max_gm,
        "required_fractional_kinematic_subtraction": required_fractional_subtraction,
        "max_orbital_sagnac_delay_s": max_kinematic,
        "orbital_sagnac_to_gravitomagnetic_ratio": max_kinematic / max_gm,
        "required_fractional_orbital_sagnac_subtraction": required_fractional_subtraction,
        "equivalent_area_only_radius_tolerance_m": area_only_radius_tolerance_m,
        "equivalent_kepler_consistent_radius_tolerance_m": (
            kepler_consistent_radius_tolerance_m
        ),
        "representative_galileo_orbit_residual_m": REPRESENTATIVE_GALILEO_ORBIT_RESIDUAL_M,
        "orbit_residual_to_area_only_tolerance_ratio": (
            REPRESENTATIVE_GALILEO_ORBIT_RESIDUAL_M / area_only_radius_tolerance_m
        ),
        "orbit_residual_to_kepler_tolerance_ratio": (
            REPRESENTATIVE_GALILEO_ORBIT_RESIDUAL_M
            / kepler_consistent_radius_tolerance_m
        ),
        "benchmark_timing_floor_s": timing_floor_s,
        "crlb_worst_relative_spin_std_at_floor": crlb_worst,
        "optimized_to_clustered_sigma_min_gain": design_gain,
        "ideal_independent_samples_for_unit_relative_crlb": ideal_sample_count,
        "earth_energy_geometric_m": earth_energy_geometric_m,
        "earth_vmax_over_c": earth_vmax_over_c,
        "earth_cycle_length_m": square_cycle_length_m,
        "earth_source_cycle_separation_m": source_cycle_separation_m,
        "earth_geometry_bound_factor": earth_geometry_bound_factor,
        "earth_energy_bound_s": earth_energy_bound_s,
        "earth_energy_bound_to_signal_ratio": earth_energy_bound_s / max_gm,
        "three_cycle_projected_rank": int(
            geometry.loc[
                geometry.geometry == "optimized_three_with_orbital_nuisance", "rank"
            ].iloc[0]
        ),
        "six_cycle_projected_rank": int(
            geometry.loc[
                geometry.geometry == "multiradius_six_with_orbital_nuisance", "rank"
            ].iloc[0]
        ),
        "six_cycle_projected_sigma_min_s_per_earth_spin": six_cycle_sigma_min,
        "second_radius_factor": SECOND_RADIUS_FACTOR,
        "second_radius_m": SECOND_ORBIT_RADIUS_M,
        "nuisance_column_count": 1,
        "nuisance_amplitude_shared_across_radii": True,
        "full_spin_parameter_dimension": SPIN_PARAMETER_DIMENSION,
        "six_cycle_maximum_nuisance_dimension_for_full_spin_rank": (
            SIX_CYCLE_COUNT - SPIN_PARAMETER_DIMENSION
        ),
        "six_cycle_p4_spin_rank_ceiling": int(
            dimension_budget.loc[
                dimension_budget.nuisance_dimension_p == 4, "spin_rank_ceiling"
            ].iloc[0]
        ),
        "operational_crlb_worst_relative_spin_std_at_floor": operational_crlb_worst,
        "operational_independent_samples_for_unit_relative_crlb": operational_sample_count,
    }
    (RESULTS / "physical_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )

    macro_lines = [
        "% Generated by src/run_physical_campaign.py; do not edit manually.",
        rf"\newcommand{{\PhysicalRunMode}}{{{args.mode}}}",
        rf"\newcommand{{\PhysicalPublicationReady}}{{{str(args.mode == 'full').lower()}}}",
        rf"\newcommand{{\EarthAngularMomentum}}{{{latex_scientific(summary['earth_angular_momentum_kg_m2_s'])}}}",
        rf"\newcommand{{\OptimizedNodes}}{{{', '.join(f'{v:.0f}' for v in optimized_nodes)}}}",
        rf"\newcommand{{\GalileoOrbitalOmega}}{{{latex_scientific(galileo_orbital_omega)}}}",
        rf"\newcommand{{\GalileoOrbitalPeriodHours}}{{{galileo_orbital_period_s / 3600.0:.2f}}}",
        rf"\newcommand{{\EarthGMDelay}}{{{latex_scientific(max_gm)}}}",
        rf"\newcommand{{\EarthKinematicDelay}}{{{latex_scientific(max_kinematic)}}}",
        rf"\newcommand{{\EarthKinematicRatio}}{{{latex_scientific(summary['kinematic_to_gravitomagnetic_ratio'])}}}",
        rf"\newcommand{{\RequiredKinematicSuppression}}{{{latex_scientific(required_fractional_subtraction)}}}",
        rf"\newcommand{{\OrbitalSagnacDelay}}{{{latex_scientific(max_kinematic)}}}",
        rf"\newcommand{{\OrbitalToGMRatio}}{{{latex_scientific(summary['kinematic_to_gravitomagnetic_ratio'])}}}",
        rf"\newcommand{{\RequiredOrbitalSuppression}}{{{latex_scientific(required_fractional_subtraction)}}}",
        rf"\newcommand{{\EarthTimingSigmaMin}}{{{latex_scientific(sigma_min)}}}",
        rf"\newcommand{{\OptimizedConditionNumber}}{{{summary['optimized_condition_number']:.3f}}}",
        rf"\newcommand{{\ClockBenchmarkFloor}}{{{latex_scientific(timing_floor_s, digits=1)}}}",
        rf"\newcommand{{\ClockBenchmarkCRLB}}{{{crlb_worst:.3f}}}",
        rf"\newcommand{{\DesignGain}}{{{design_gain:.2f}}}",
        rf"\newcommand{{\IdealSampleCount}}{{{ideal_sample_count}}}",
        rf"\newcommand{{\OperationalClockBenchmarkCRLB}}{{{operational_crlb_worst:.3f}}}",
        rf"\newcommand{{\OperationalSampleCount}}{{{operational_sample_count}}}",
        rf"\newcommand{{\EarthEnergyGeometric}}{{{latex_scientific(earth_energy_geometric_m)}}}",
        rf"\newcommand{{\EarthVmaxOverC}}{{{latex_scientific(earth_vmax_over_c)}}}",
        rf"\newcommand{{\EarthGeometryBound}}{{{earth_geometry_bound_factor:.2f}}}",
        rf"\newcommand{{\EarthEnergyBound}}{{{latex_scientific(earth_energy_bound_s)}}}",
        rf"\newcommand{{\EarthEnergyBoundRatio}}{{{earth_energy_bound_s / max_gm:.2f}}}",
        rf"\newcommand{{\ThreeCycleNuisanceRank}}{{{summary['three_cycle_projected_rank']}}}",
        rf"\newcommand{{\SixCycleNuisanceRank}}{{{summary['six_cycle_projected_rank']}}}",
        rf"\newcommand{{\SixCycleSigmaMin}}{{{latex_scientific(six_cycle_sigma_min)}}}",
        rf"\newcommand{{\SecondRadiusFactor}}{{{SECOND_RADIUS_FACTOR:.2f}}}",
        rf"\newcommand{{\SecondRadius}}{{{latex_scientific(SECOND_ORBIT_RADIUS_M)}}}",
        rf"\newcommand{{\AreaOnlyRadiusTolerance}}{{{latex_scientific(area_only_radius_tolerance_m)}}}",
        rf"\newcommand{{\AreaOnlyRadiusToleranceMicrometers}}{{{area_only_radius_tolerance_m * 1.0e6:.1f}}}",
        rf"\newcommand{{\KeplerRadiusTolerance}}{{{latex_scientific(kepler_consistent_radius_tolerance_m)}}}",
        rf"\newcommand{{\KeplerRadiusToleranceMicrometers}}{{{kepler_consistent_radius_tolerance_m * 1.0e6:.1f}}}",
        rf"\newcommand{{\RepresentativeGalileoOrbitResidualCm}}{{{REPRESENTATIVE_GALILEO_ORBIT_RESIDUAL_M * 100.0:.0f}}}",
        rf"\newcommand{{\OrbitToleranceGap}}{{{summary['orbit_residual_to_area_only_tolerance_ratio']:.0f}}}",
        rf"\newcommand{{\KeplerOrbitToleranceGap}}{{{summary['orbit_residual_to_kepler_tolerance_ratio']:.0f}}}",
        rf"\newcommand{{\SixCycleMaxNuisanceDimension}}{{{summary['six_cycle_maximum_nuisance_dimension_for_full_spin_rank']}}}",
        rf"\newcommand{{\SixCyclePFourRankCeiling}}{{{summary['six_cycle_p4_spin_rank_ceiling']}}}",
    ]
    (RESULTS / "physical_macros.tex").write_text("\n".join(macro_lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    labels = ["clustered", "E-optimal"]
    for label, loops, color in zip(
        labels, [clustered_loops, optimized_loops], ["#c43c39", "#1769aa"]
    ):
        values = np.linalg.svd(timing_design_matrix(loops), compute_uv=False)
        axes[0].plot([1, 2, 3], values, "o-", label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set_xticks([1, 2, 3], [r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"])
    axes[0].set_ylabel("Seconds per Earth-spin unit")
    axes[0].set_title("Galileo-plane E-optimal design")
    axes[0].grid(which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    x = np.arange(len(optimized_loops))
    width = 0.36
    axes[1].bar(x - width / 2, np.abs(gm_delays), width, label="gravitomagnetic")
    axes[1].bar(
        x + width / 2,
        np.abs(kinematic_delays),
        width,
        label="orbital Sagnac",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, [f"cycle {index + 1}" for index in x])
    axes[1].set_ylabel("Absolute cycle delay (s)")
    axes[1].set_title("The surviving orbital nuisance")
    axes[1].grid(which="both", axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "earth_galileo_benchmark.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
