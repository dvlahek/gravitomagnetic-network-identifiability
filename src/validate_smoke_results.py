"""Fast consistency checks for smoke-mode outputs.

This validator confirms that both numerical campaigns completed and that their
qualitative conclusions are intact.  It deliberately does not mark results as
publication ready.
"""

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
    physical = json.loads((RESULTS / "physical_summary.json").read_text())
    geometry = pd.read_csv(RESULTS / "geometry_singular_values.csv").set_index(
        "geometry"
    )
    physical_geometry = pd.read_csv(
        RESULTS / "physical_geometry_design.csv"
    ).set_index("geometry")
    nuisance_budget = pd.read_csv(RESULTS / "nuisance_dimension_budget.csv").set_index(
        "nuisance_dimension_p"
    )

    require(summary["run_mode"] == "smoke", "Core campaign is not a smoke run.")
    require(physical["run_mode"] == "smoke", "Physical campaign is not a smoke run.")
    require(summary["normalization"]["green_coefficient"] == 2.0, "Wrong normalization.")
    require(summary["max_analytic_numeric_error"] < 1.0e-10, "Dipole check failed.")
    require(int(geometry.loc["good", "rank"]) == 3, "Good geometry lost rank.")
    require(int(geometry.loc["degenerate", "rank"]) == 1, "Degenerate rank changed.")
    require(summary["bound_hierarchy_all_points"] is True, "Bound hierarchy failed.")
    require(abs(summary["multipole_far_zone_slope"] + 2.0) < 0.1, "Wrong multipole slope.")
    require(
        int(physical_geometry.loc["optimized_three_with_orbital_nuisance", "rank"]) == 2,
        "Three-cycle nuisance rank changed.",
    )
    require(
        int(physical_geometry.loc["multiradius_six_with_orbital_nuisance", "rank"]) == 3,
        "Multiradius design did not recover rank three.",
    )
    require(
        physical["kinematic_to_gravitomagnetic_ratio"] > 1.0e6,
        "Dominant kinematic nuisance was not reproduced.",
    )
    require(
        nuisance_budget.spin_rank_ceiling.astype(int).to_dict()
        == {1: 3, 2: 3, 3: 3, 4: 2},
        "Six-cycle nuisance dimension budget changed.",
    )
    require(
        abs(
            physical["equivalent_kepler_consistent_radius_tolerance_m"]
            / physical["equivalent_area_only_radius_tolerance_m"]
            - 4.0
        )
        < 1.0e-12,
        "Radius-tolerance conversion changed.",
    )
    print("Smoke validation passed (not publication results).")


if __name__ == "__main__":
    main()
