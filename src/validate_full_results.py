"""Strict post-run validation for publication-mode numerical outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    geometry = pd.read_csv(RESULTS / "geometry_singular_values.csv").set_index(
        "geometry"
    )
    bounds = pd.read_csv(RESULTS / "source_bounds.csv")
    multipole = pd.read_csv(RESULTS / "multipole_mismatch.csv")
    physical = json.loads((RESULTS / "physical_summary.json").read_text())
    physical_geometry = pd.read_csv(RESULTS / "physical_geometry_design.csv").set_index(
        "geometry"
    )
    nuisance_budget = pd.read_csv(RESULTS / "nuisance_dimension_budget.csv").set_index(
        "nuisance_dimension_p"
    )

    require(summary["run_mode"] == "full", "Results were not generated in full mode.")
    require(summary["publication_results"] is True, "Publication flag is false.")
    require(
        summary["normalization"]["green_coefficient"] == 2.0,
        "The corrected Green-function normalization is missing.",
    )
    require(summary["max_analytic_numeric_error"] < 1e-10, "Dipole check failed.")
    require(int(geometry.loc["good", "rank"]) == 3, "Good geometry lost rank.")
    require(
        int(geometry.loc["near_degenerate", "rank"]) == 3,
        "Near-degenerate geometry lost rank.",
    )
    require(
        int(geometry.loc["degenerate", "rank"]) == 1,
        "Degenerate geometry has the wrong rank.",
    )
    require(summary["bound_hierarchy_all_points"] is True, "Bound hierarchy failed.")
    require(
        summary["max_noise_bound_ratio"] <= 1.0 + 1e-9,
        "Reconstruction error exceeded the samplewise stability bound.",
    )
    require(
        bool((bounds.absolute_response <= bounds.kernel_bound * (1.0 + 1e-10)).all()),
        "Exact response exceeded the kernel bound.",
    )
    require(
        bool((bounds.kernel_bound <= bounds.separated_bound * (1.0 + 1e-10)).all()),
        "Kernel bound exceeded the separated-source bound.",
    )
    require(
        bool((bounds.separated_bound <= bounds.energy_bound * (1.0 + 1e-10)).all()),
        "Separated-source bound exceeded the energy bound.",
    )
    slope = float(multipole.asymptotic_loglog_slope.iloc[0])
    require(-2.2 < slope < -1.8, f"Unexpected far-zone mismatch slope: {slope:.4f}")
    require(
        abs(slope + 2.0) < 0.1,
        "The fitted multipole slope does not confirm the analytic R^-2 prediction.",
    )
    require(physical["run_mode"] == "full", "Physical campaign was not run in full mode.")
    require(
        physical["publication_results"] is True,
        "Physical publication flag is false.",
    )
    require(
        int(
            physical_geometry.loc[
                "optimized_three_with_orbital_nuisance", "rank"
            ]
        )
        == 2,
        "The three-cycle nuisance example should lose one identifiable direction.",
    )
    require(
        int(
            physical_geometry.loc[
                "multiradius_six_with_orbital_nuisance", "rank"
            ]
        )
        == 3,
        "The multiradius design did not recover nuisance-aware rank three.",
    )
    require(
        physical["kinematic_to_gravitomagnetic_ratio"] > 1.0e6,
        "The physical campaign did not expose the dominant kinematic nuisance.",
    )
    require(
        physical["kinematic_model"] == "circular Keplerian orbital Sagnac",
        "The physical campaign is not using the orbital Sagnac model.",
    )
    require(
        1.2e-4 < physical["galileo_orbital_omega_rad_s"] < 1.3e-4,
        "The Galileo orbital angular speed is outside the expected range.",
    )
    require(
        physical["second_radius_factor"] == 0.78,
        "The second-radius geometry is not the documented 0.78R design.",
    )
    require(
        physical["nuisance_column_count"] == 1
        and physical["nuisance_amplitude_shared_across_radii"] is True,
        "The multiradius nuisance model is not a single shared amplitude.",
    )
    require(
        physical["operational_independent_samples_for_unit_relative_crlb"] == 4355,
        "The nuisance-aware operational sample count changed unexpectedly.",
    )
    require(
        nuisance_budget.spin_rank_ceiling.astype(int).to_dict()
        == {1: 3, 2: 3, 3: 3, 4: 2},
        "The six-cycle nuisance-dimension ceiling is inconsistent.",
    )
    require(
        int(nuisance_budget.loc[1, "computed_projected_rank"]) == 3,
        "The specified one-column nuisance example did not retain rank three.",
    )
    require(
        physical["six_cycle_maximum_nuisance_dimension_for_full_spin_rank"] == 3,
        "The six-cycle nuisance budget should allow at most p=3 dimensionally.",
    )
    require(
        abs(
            physical["equivalent_kepler_consistent_radius_tolerance_m"]
            / physical["equivalent_area_only_radius_tolerance_m"]
            - 4.0
        )
        < 1.0e-12,
        "The two radius-tolerance conversions should differ by a factor four.",
    )
    require(
        1.0 < physical["optimized_condition_number"] < 1.05,
        "The symmetric design should be nearly, but not exactly, isotropic.",
    )

    print("Full-result validation passed.")
    print(f"Analytic/numeric error: {summary['max_analytic_numeric_error']:.3e}")
    print(f"Near-degenerate condition number: {geometry.loc['near_degenerate', 'condition_number']:.3f}")
    print(f"Far-zone mismatch slope: {slope:.4f}")
    print(f"Bound points checked: {len(bounds)}")
    print(
        "Orbital-Sagnac/GM ratio: "
        f"{physical['kinematic_to_gravitomagnetic_ratio']:.3e}"
    )


if __name__ == "__main__":
    main()
