"""Generate all numerical results and figures for the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from network_model import (
    current_vector_potential,
    design_matrix,
    dipole_vector_potential,
    loop_reciprocity_response,
    loop_source_bounds,
    rigid_rotating_sphere,
    square_loop,
    unit,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def latex_scientific(value: float, digits: int = 3) -> str:
    """Format a number as LaTeX scientific notation for generated macros."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def geometry_experiment() -> tuple[dict[str, list[np.ndarray]], dict[str, np.ndarray]]:
    radius = 5.0
    good = [
        square_loop(np.zeros(3), normal, radius)
        for normal in np.eye(3)
    ]
    near = [
        square_loop(np.zeros(3), [0.00, 0.00, 1.00], radius),
        square_loop(np.zeros(3), [0.08, 0.00, 1.00], radius),
        square_loop(np.zeros(3), [0.00, 0.08, 1.00], radius),
    ]
    # These concentric loops share the plane z=0, and the dipole center is the
    # common loop center.  Coplanarity alone would not imply rank one.
    degenerate = [
        square_loop(np.zeros(3), [0.0, 0.0, 1.0], scale)
        for scale in (4.0, 5.0, 6.0)
    ]
    loops = {"good": good, "near_degenerate": near, "degenerate": degenerate}
    matrices = {name: design_matrix(value) for name, value in loops.items()}

    rows = []
    for name, matrix in matrices.items():
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix, tol=1e-11))
        condition = float(np.inf if singular[-1] < 1e-14 else singular[0] / singular[-1])
        rows.append(
            {
                "geometry": name,
                "rank": rank,
                "sigma_1": singular[0],
                "sigma_2": singular[1],
                "sigma_3": singular[2],
                "condition_number": condition,
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS / "geometry_singular_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(3)
    width = 0.24
    positive = [
        value
        for matrix in matrices.values()
        for value in np.linalg.svd(matrix, compute_uv=False)
        if value > 1e-14
    ]
    display_floor = min(positive) * 0.18
    for offset, (name, matrix) in enumerate(matrices.items()):
        singular = np.linalg.svd(matrix, compute_uv=False)
        display = np.maximum(singular, display_floor)
        bars = ax.bar(
            x + (offset - 1) * width,
            display,
            width,
            label=name.replace("_", " "),
        )
        for index, value in enumerate(singular):
            if value <= 1e-14:
                ax.text(
                    bars[index].get_x() + 0.5 * bars[index].get_width(),
                    display_floor * 1.05,
                    "0",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_yscale("log")
    ax.set_xticks(x, [r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"])
    ax.set_ylabel("Singular value (zeros shown at plotting floor)")
    ax.set_title("Geometry controls angular-momentum identifiability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "singular_values.png")
    return loops, matrices


def nonidentifiability_experiment(matrix: np.ndarray) -> dict[str, float | list[float]]:
    true_spin = unit(np.array([0.35, -0.48, 0.80]))
    _, singular, vt = np.linalg.svd(matrix)
    rank = int(np.sum(singular > 1e-11))
    null_vector = unit(vt[rank])
    alternative_spin = true_spin + 0.7 * null_vector
    data_1 = matrix @ true_spin
    data_2 = matrix @ alternative_spin
    result = {
        "rank": rank,
        "true_spin": true_spin.tolist(),
        "alternative_spin": alternative_spin.tolist(),
        "null_direction": null_vector.tolist(),
        "spin_difference_norm": float(np.linalg.norm(alternative_spin - true_spin)),
        "data_difference_norm": float(np.linalg.norm(data_2 - data_1)),
    }
    (RESULTS / "nonidentifiability.json").write_text(json.dumps(result, indent=2))
    return result


def noise_experiment(
    matrices: dict[str, np.ndarray],
    trials: int,
    levels: int,
    seed: int = 20260830,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    true_spin = unit(np.array([0.35, -0.48, 0.80]))
    relative_noise = np.logspace(-6, -1, levels)
    records = []
    for name in ("good", "near_degenerate"):
        matrix = matrices[name]
        clean = matrix @ true_spin
        signal_rms = np.linalg.norm(clean) / np.sqrt(len(clean))
        sigma_min = np.linalg.svd(matrix, compute_uv=False)[-1]
        inverse = np.linalg.pinv(matrix)
        for relative in relative_noise:
            sigma = relative * signal_rms
            errors = []
            noise_norms = []
            for _ in range(trials):
                noise = rng.normal(scale=sigma, size=len(clean))
                estimate = inverse @ (clean + noise)
                errors.append(np.linalg.norm(estimate - true_spin))
                noise_norms.append(np.linalg.norm(noise) / sigma_min)
            records.append(
                {
                    "geometry": name,
                    "relative_noise": relative,
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "mean_error": float(np.mean(errors)),
                    "rms_deterministic_bound": float(
                        np.sqrt(np.mean(np.square(noise_norms)))
                    ),
                    "sigma_min": sigma_min,
                }
            )
    frame = pd.DataFrame(records)
    frame.to_csv(RESULTS / "noise_stability.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    colors = {"good": "#1769aa", "near_degenerate": "#c43c39"}
    for name in colors:
        subset = frame[frame.geometry == name]
        ax.loglog(
            subset.relative_noise,
            subset.rmse,
            marker="o",
            color=colors[name],
            label=f"{name.replace('_', ' ')}: RMSE",
        )
        ax.loglog(
            subset.relative_noise,
            subset.rms_deterministic_bound,
            linestyle="--",
            color=colors[name],
            alpha=0.8,
            label=f"{name.replace('_', ' ')}: bound",
        )
    ax.set_xlabel("Noise standard deviation / signal RMS")
    ax.set_ylabel("Angular-momentum error")
    ax.set_title("Stability is controlled by the smallest singular value")
    ax.grid(which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "noise_stability.png")
    return frame


def multipole_experiment(radius_count: int, quadrature_order: int) -> pd.DataFrame:
    total_spin = unit(np.array([0.35, -0.48, 0.80]))
    internal = np.array([0.20, 0.10, -0.05])
    spin_1 = 0.65 * total_spin + internal
    spin_2 = 0.35 * total_spin - internal
    center_1 = np.array([0.55, 0.10, -0.15])
    center_2 = np.array([-0.55, -0.10, 0.15])

    records = []
    for radius in np.geomspace(2.5, 25.0, radius_count):
        loops = [square_loop(np.zeros(3), normal, radius) for normal in np.eye(3)]
        central_matrix = design_matrix(loops)

        def field(points: np.ndarray) -> np.ndarray:
            return dipole_vector_potential(points, spin_1, center=center_1) + dipole_vector_potential(
                points, spin_2, center=center_2
            )

        data = np.array(
            [
                loop_reciprocity_response(
                    loop, field, quadrature_order=quadrature_order
                )
                for loop in loops
            ]
        )
        estimate = np.linalg.solve(central_matrix, data)
        records.append(
            {
                "network_radius": radius,
                "relative_model_error": float(np.linalg.norm(estimate - total_spin)),
                "estimated_jx": estimate[0],
                "estimated_jy": estimate[1],
                "estimated_jz": estimate[2],
            }
        )
    frame = pd.DataFrame(records)
    fit_count = min(8, max(3, radius_count // 2))
    fit = np.polyfit(
        np.log(frame.network_radius.to_numpy()[-fit_count:]),
        np.log(frame.relative_model_error.to_numpy()[-fit_count:]),
        1,
    )
    frame["asymptotic_loglog_slope"] = fit[0]
    frame["predicted_asymptotic_slope"] = -2.0
    frame.to_csv(RESULTS / "multipole_mismatch.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.loglog(
        frame.network_radius,
        frame.relative_model_error,
        "o-",
        color="#6a3d9a",
        label="two-center source reconstructed as one dipole",
    )
    reference = frame.relative_model_error.iloc[-1] * (
        frame.network_radius / frame.network_radius.iloc[-1]
    ) ** fit[0]
    ax.loglog(
        frame.network_radius,
        reference,
        "--",
        color="black",
        label=fr"far-zone fit: slope {fit[0]:.2f}",
    )
    ax.set_xlabel("Network radius / source separation")
    ax.set_ylabel(r"$\|\widehat{\mathbf{J}}-\mathbf{J}\|/\|\mathbf{J}\|$")
    ax.set_title("Higher-current-multipole contamination decays in the far zone")
    ax.grid(which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "multipole_mismatch.png")
    return frame


def bound_experiment(
    source_resolution: int,
    distance_count: int,
    response_quadrature: int,
    bound_quadrature: int,
) -> tuple[pd.DataFrame, dict[str, float | list[float]]]:
    source = rigid_rotating_sphere(
        resolution=source_resolution,
        radius=1.0,
        density=1.0,
        angular_velocity=np.array([0.2, 0.0, 0.0]),
    )
    records = []
    for distance in np.geomspace(2.5, 16.0, distance_count):
        loop = square_loop(np.array([distance, 0.0, 0.0]), [1.0, 0.0, 0.0], 0.8)
        field = lambda points: current_vector_potential(points, source)
        response = abs(
            loop_reciprocity_response(
                loop, field, quadrature_order=response_quadrature
            )
        )
        bounds = loop_source_bounds(
            loop, source, quadrature_order=bound_quadrature
        )
        records.append({"distance": distance, "absolute_response": response, **bounds})
    frame = pd.DataFrame(records)
    frame["response_over_kernel"] = frame.absolute_response / frame.kernel_bound
    frame["kernel_over_separated"] = frame.kernel_bound / frame.separated_bound
    frame["separated_over_energy"] = frame.separated_bound / frame.energy_bound
    frame.to_csv(RESULTS / "source_bounds.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.loglog(frame.distance, frame.absolute_response, "o-", label="exact current-generated response")
    ax.loglog(frame.distance, frame.kernel_bound, "s--", label="kernel bound")
    ax.loglog(frame.distance, frame.separated_bound, "^--", label="separated-current bound")
    ax.loglog(frame.distance, frame.energy_bound, "d:", label="energy-cost bound")
    ax.set_xlabel("Source-loop center distance")
    ax.set_ylabel("Magnitude in geometrized units")
    ax.set_title("Source and energy bounds retain the required hierarchy")
    ax.grid(which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "source_bounds.png")

    summary = {
        "discrete_source_points": int(len(source.points)),
        "current_l1": source.current_l1,
        "vmax_energy": source.vmax * source.energy,
        "angular_momentum": source.angular_momentum.tolist(),
        "max_response_over_kernel": float(frame.response_over_kernel.max()),
        "max_kernel_over_separated": float(frame.kernel_over_separated.max()),
        "max_separated_over_energy": float(frame.separated_over_energy.max()),
    }
    return frame, summary


def analytic_numeric_check(loops: list[np.ndarray]) -> float:
    spin = unit(np.array([0.35, -0.48, 0.80]))
    matrix = design_matrix(loops)
    analytic = matrix @ spin
    field = lambda points: dipole_vector_potential(points, spin)
    numeric = np.array([loop_reciprocity_response(loop, field) for loop in loops])
    discrepancy = float(np.max(np.abs(analytic - numeric)))
    pd.DataFrame(
        {
            "analytic": analytic,
            "numeric": numeric,
            "absolute_error": np.abs(analytic - numeric),
        }
    ).to_csv(RESULTS / "analytic_numeric_check.csv", index=False)
    return discrepancy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke verifies the pipeline quickly; full generates publication data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        settings = {
            "trials": 24,
            "noise_levels": 5,
            "radius_count": 6,
            "multipole_quadrature": 18,
            "source_resolution": 7,
            "distance_count": 4,
            "response_quadrature": 12,
            "bound_quadrature": 10,
        }
    else:
        settings = {
            "trials": 2000,
            "noise_levels": 17,
            "radius_count": 24,
            "multipole_quadrature": 64,
            "source_resolution": 25,
            "distance_count": 24,
            "response_quadrature": 56,
            "bound_quadrature": 48,
        }

    loops, matrices = geometry_experiment()
    nonidentifiability = nonidentifiability_experiment(matrices["degenerate"])
    noise = noise_experiment(
        matrices,
        trials=settings["trials"],
        levels=settings["noise_levels"],
    )
    multipole = multipole_experiment(
        radius_count=settings["radius_count"],
        quadrature_order=settings["multipole_quadrature"],
    )
    bounds, source_summary = bound_experiment(
        source_resolution=settings["source_resolution"],
        distance_count=settings["distance_count"],
        response_quadrature=settings["response_quadrature"],
        bound_quadrature=settings["bound_quadrature"],
    )
    max_analytic_error = analytic_numeric_check(loops["good"])

    summary = {
        "run_mode": args.mode,
        "publication_results": args.mode == "full",
        "normalization": {
            "metric_cross_term": "-4 A_i dx^i dt",
            "poisson_equation": "nabla^2 A_i = -8 pi J_i",
            "green_coefficient": 2.0,
            "dipole_kappa": 1.0,
        },
        "settings": settings,
        "max_analytic_numeric_error": max_analytic_error,
        "geometry_ranks": {
            name: int(np.linalg.matrix_rank(matrix, tol=1e-11))
            for name, matrix in matrices.items()
        },
        "geometry_condition_numbers": {
            name: (
                float(np.linalg.cond(matrix))
                if np.isfinite(np.linalg.cond(matrix))
                else None
            )
            for name, matrix in matrices.items()
        },
        "nonidentifiability": nonidentifiability,
        "max_noise_bound_ratio": float((noise.rmse / noise.rms_deterministic_bound).max()),
        "multipole_far_zone_slope": float(multipole.asymptotic_loglog_slope.iloc[0]),
        "multipole_slope_error_from_prediction": float(
            abs(multipole.asymptotic_loglog_slope.iloc[0] + 2.0)
        ),
        "source_bound_summary": source_summary,
        "bound_hierarchy_all_points": bool(
            np.all(bounds.absolute_response <= bounds.kernel_bound)
            and np.all(bounds.kernel_bound <= bounds.separated_bound)
            and np.all(bounds.separated_bound <= bounds.energy_bound)
        ),
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    geometry_rows = pd.read_csv(RESULTS / "geometry_singular_values.csv").set_index(
        "geometry"
    )
    macro_lines = [
        "% Generated by src/run_experiments.py; do not edit manually.",
        rf"\newcommand{{\RunMode}}{{{args.mode}}}",
        rf"\newcommand{{\PublicationReady}}{{{str(args.mode == 'full').lower()}}}",
        rf"\newcommand{{\GoodRank}}{{{summary['geometry_ranks']['good']}}}",
        rf"\newcommand{{\NearRank}}{{{summary['geometry_ranks']['near_degenerate']}}}",
        rf"\newcommand{{\DegenerateRank}}{{{summary['geometry_ranks']['degenerate']}}}",
        rf"\newcommand{{\NearCondition}}{{{geometry_rows.loc['near_degenerate', 'condition_number']:.3f}}}",
        rf"\newcommand{{\NoiseTrials}}{{{settings['trials']}}}",
        rf"\newcommand{{\AnalyticNumericError}}{{{latex_scientific(max_analytic_error)}}}",
        rf"\newcommand{{\NullSpinDifference}}{{{nonidentifiability['spin_difference_norm']:.3f}}}",
        rf"\newcommand{{\NullDataDifference}}{{{latex_scientific(nonidentifiability['data_difference_norm'])}}}",
        rf"\newcommand{{\MultipoleSlope}}{{{summary['multipole_far_zone_slope']:.3f}}}",
        rf"\newcommand{{\MultipoleSlopeError}}{{{latex_scientific(summary['multipole_slope_error_from_prediction'])}}}",
        rf"\newcommand{{\MaxResponseKernelRatio}}{{{source_summary['max_response_over_kernel']:.4f}}}",
        rf"\newcommand{{\MaxKernelSeparatedRatio}}{{{source_summary['max_kernel_over_separated']:.4f}}}",
        rf"\newcommand{{\MaxSeparatedEnergyRatio}}{{{source_summary['max_separated_over_energy']:.4f}}}",
        rf"\newcommand{{\SourceGridPoints}}{{{source_summary['discrete_source_points']}}}",
    ]
    (RESULTS / "results_macros.tex").write_text("\n".join(macro_lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
