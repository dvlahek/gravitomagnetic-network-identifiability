"""Finite-network gravitomagnetic reciprocity model.

All formulas use geometrized units (G = c = 1) and the weak-field convention

    A(x) = 2 int J(x') / |x-x'| d^3x',
    R_gamma = -4 int_gamma A . dx.

For a compact stationary current, the leading far-zone vector potential is
    A_dip(x) = (L x x) / |x|^3,
where L is the source angular momentum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from numpy.polynomial.legendre import leggauss


Array = np.ndarray


def unit(vector: Array) -> Array:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("A zero vector cannot be normalized.")
    return vector / norm


def square_loop(center: Array, normal: Array, half_side: float) -> Array:
    """Return four oriented vertices with right-hand normal ``normal``."""
    center = np.asarray(center, dtype=float)
    normal = unit(normal)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, normal)) > 0.85:
        trial = np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(trial, normal))
    v = np.cross(normal, u)
    h = float(half_side)
    return np.array(
        [
            center - h * u - h * v,
            center + h * u - h * v,
            center + h * u + h * v,
            center - h * u + h * v,
        ]
    )


def segments(vertices: Array) -> Iterable[tuple[Array, Array]]:
    vertices = np.asarray(vertices, dtype=float)
    for index in range(len(vertices)):
        yield vertices[index], vertices[(index + 1) % len(vertices)]


def segment_design_row(start: Array, end: Array, kappa: float = 1.0) -> Array:
    """Exact dipole response row for one straight segment.

    If A = kappa * (L x x)/|x|^3, the segment reciprocity response is
    r = q . L with q returned here.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    length = np.linalg.norm(delta)
    if length == 0:
        raise ValueError("Segment endpoints must be distinct.")
    tangent = delta / length
    parallel0 = float(np.dot(start, tangent))
    perpendicular = start - parallel0 * tangent
    impact2 = float(np.dot(perpendicular, perpendicular))
    if impact2 <= 1e-14:
        # The apparent 0 times infinity in the closed form is removable.
        # A radial segment has (L x x).dx = 0 and therefore a zero row.
        return np.zeros(3)

    parallel1 = parallel0 + length

    def primitive(value: float) -> float:
        return value / (impact2 * np.sqrt(impact2 + value * value))

    radial_integral = primitive(parallel1) - primitive(parallel0)
    return -4.0 * kappa * np.cross(start, tangent) * radial_integral


def loop_design_row(vertices: Array, kappa: float = 1.0) -> Array:
    return sum(
        (segment_design_row(start, end, kappa=kappa) for start, end in segments(vertices)),
        np.zeros(3),
    )


def design_matrix(loops: Iterable[Array], kappa: float = 1.0) -> Array:
    return np.vstack([loop_design_row(loop, kappa=kappa) for loop in loops])


def dipole_vector_potential(
    points: Array,
    angular_momentum: Array,
    center: Array | None = None,
    kappa: float = 1.0,
    softening: float = 0.0,
) -> Array:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    angular_momentum = np.asarray(angular_momentum, dtype=float)
    center = np.zeros(3) if center is None else np.asarray(center, dtype=float)
    relative = points - center
    radius2 = np.sum(relative * relative, axis=1) + softening * softening
    if np.any(radius2 <= 0):
        raise ValueError("Evaluation point coincides with the dipole center.")
    return kappa * np.cross(angular_momentum, relative) / radius2[:, None] ** 1.5


def integrate_loop(
    vertices: Array,
    vector_field: Callable[[Array], Array],
    quadrature_order: int = 48,
) -> float:
    """Compute the oriented line integral int A . dx."""
    nodes, weights = leggauss(quadrature_order)
    total = 0.0
    for start, end in segments(vertices):
        delta = end - start
        parameters = 0.5 * (nodes + 1.0)
        points = start[None, :] + parameters[:, None] * delta[None, :]
        values = vector_field(points)
        total += 0.5 * float(np.sum(weights * (values @ delta)))
    return total


def loop_reciprocity_response(
    vertices: Array,
    vector_field: Callable[[Array], Array],
    quadrature_order: int = 48,
) -> float:
    return -4.0 * integrate_loop(vertices, vector_field, quadrature_order)


@dataclass(frozen=True)
class DiscreteCurrentSource:
    points: Array
    current: Array
    density: Array
    cell_volume: float
    support_radius: float
    vmax: float

    @property
    def current_l1(self) -> float:
        return float(np.sum(np.linalg.norm(self.current, axis=1)) * self.cell_volume)

    @property
    def energy(self) -> float:
        return float(np.sum(self.density) * self.cell_volume)

    @property
    def angular_momentum(self) -> Array:
        return np.sum(np.cross(self.points, self.current), axis=0) * self.cell_volume


def rigid_rotating_sphere(
    resolution: int = 15,
    radius: float = 1.0,
    density: float = 1.0,
    angular_velocity: Array | None = None,
) -> DiscreteCurrentSource:
    """Midpoint discretization of a uniform rigidly rotating sphere."""
    if resolution < 5:
        raise ValueError("resolution must be at least 5")
    angular_velocity = (
        np.array([0.2, 0.0, 0.0])
        if angular_velocity is None
        else np.asarray(angular_velocity, dtype=float)
    )
    step = 2.0 * radius / resolution
    axis = np.linspace(-radius + 0.5 * step, radius - 0.5 * step, resolution)
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack(mesh, axis=-1).reshape(-1, 3)
    points = points[np.linalg.norm(points, axis=1) <= radius]
    rho = np.full(len(points), float(density))
    current = rho[:, None] * np.cross(angular_velocity, points)
    vmax = float(np.linalg.norm(angular_velocity) * radius)
    return DiscreteCurrentSource(
        points=points,
        current=current,
        density=rho,
        cell_volume=step**3,
        support_radius=float(radius),
        vmax=vmax,
    )


def current_vector_potential(points: Array, source: DiscreteCurrentSource) -> Array:
    """Evaluate A(x) = 2 int J(x')/|x-x'| d^3x'."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    output = np.zeros_like(points)
    for index, point in enumerate(points):
        difference = point[None, :] - source.points
        distance = np.linalg.norm(difference, axis=1)
        if np.any(distance == 0):
            raise ValueError("Field evaluation entered the source discretization.")
        output[index] = (
            2.0
            * np.sum(source.current / distance[:, None], axis=0)
            * source.cell_volume
        )
    return output


def loop_length(vertices: Array) -> float:
    return float(sum(np.linalg.norm(end - start) for start, end in segments(vertices)))


def point_segment_distance(point: Array, start: Array, end: Array) -> float:
    delta = end - start
    parameter = np.dot(point - start, delta) / np.dot(delta, delta)
    parameter = float(np.clip(parameter, 0.0, 1.0))
    closest = start + parameter * delta
    return float(np.linalg.norm(point - closest))


def loop_source_bounds(
    vertices: Array,
    source: DiscreteCurrentSource,
    quadrature_order: int = 36,
) -> dict[str, float]:
    """Detailed kernel, separated-current, and energy bounds for one loop."""
    nodes, weights = leggauss(quadrature_order)
    scalar_kernel = np.zeros(len(source.points))
    min_axis_distance = np.inf
    total_length = 0.0
    origin = np.zeros(3)

    for start, end in segments(vertices):
        delta = end - start
        length = float(np.linalg.norm(delta))
        total_length += length
        parameters = 0.5 * (nodes + 1.0)
        samples = start[None, :] + parameters[:, None] * delta[None, :]
        distances = np.linalg.norm(
            samples[:, None, :] - source.points[None, :, :], axis=2
        )
        scalar_kernel += 0.5 * length * np.sum(weights[:, None] / distances, axis=0)
        min_axis_distance = min(
            min_axis_distance, point_segment_distance(origin, start, end)
        )

    dmin = min_axis_distance - source.support_radius
    if dmin <= 0:
        raise ValueError("The loop is not separated from the source support.")

    kernel_bound = (
        8.0
        * np.sum(scalar_kernel * np.linalg.norm(source.current, axis=1))
        * source.cell_volume
    )
    separated_bound = 8.0 * total_length / dmin * source.current_l1
    energy_bound = 8.0 * total_length / dmin * source.vmax * source.energy
    return {
        "kernel_bound": float(kernel_bound),
        "separated_bound": float(separated_bound),
        "energy_bound": float(energy_bound),
        "dmin": float(dmin),
        "length": float(total_length),
    }


def loop_area_vector(vertices: Array) -> Array:
    """Return the oriented polygon area vector."""
    vertices = np.asarray(vertices, dtype=float)
    area = np.zeros(3)
    for start, end in segments(vertices):
        area += np.cross(start, end)
    return 0.5 * area


def kinematic_sagnac_delay(
    vertices_m: Array,
    angular_velocity_rad_s: Array,
    speed_of_light_m_s: float = 299_792_458.0,
) -> float:
    """Leading rigid-rotation Sagnac delay in SI seconds."""
    area = loop_area_vector(vertices_m)
    omega = np.asarray(angular_velocity_rad_s, dtype=float)
    return float(4.0 * np.dot(omega, area) / speed_of_light_m_s**2)


def nuisance_projector(nuisance_matrix: Array, rcond: float = 1e-12) -> Array:
    """Orthogonal projector onto the complement of a nuisance column space."""
    nuisance = np.atleast_2d(np.asarray(nuisance_matrix, dtype=float))
    if nuisance.shape[1] == 0:
        return np.eye(nuisance.shape[0])
    return np.eye(nuisance.shape[0]) - nuisance @ np.linalg.pinv(
        nuisance, rcond=rcond
    )


def projected_design_matrix(
    matrix: Array, nuisance_matrix: Array, rcond: float = 1e-12
) -> Array:
    """Design matrix after removal of linearly parameterized nuisance terms."""
    matrix = np.asarray(matrix, dtype=float)
    return nuisance_projector(nuisance_matrix, rcond=rcond) @ matrix


def cycle_covariance(cycle_matrix: Array, edge_noise_standard_deviation: float) -> Array:
    """Propagate independent equal-variance edge noise into cycle space."""
    cycle_matrix = np.asarray(cycle_matrix, dtype=float)
    sigma_edge = float(edge_noise_standard_deviation)
    if sigma_edge < 0:
        raise ValueError("edge_noise_standard_deviation must be non-negative")
    return sigma_edge**2 * cycle_matrix @ cycle_matrix.T


def whitened_projected_design_matrix(
    matrix: Array,
    nuisance_matrix: Array,
    covariance: Array,
    rcond: float = 1e-12,
) -> Array:
    """Whiten cycle data and project a linear nuisance column space."""
    matrix = np.asarray(matrix, dtype=float)
    nuisance = np.asarray(nuisance_matrix, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (matrix.shape[0], matrix.shape[0]):
        raise ValueError("covariance has the wrong shape")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = rcond * max(float(np.max(eigenvalues)), 1.0)
    if np.any(eigenvalues <= tolerance):
        raise ValueError("covariance must be positive definite on cycle space")
    whitening = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    whitened_matrix = whitening @ matrix
    whitened_nuisance = whitening @ nuisance
    return projected_design_matrix(
        whitened_matrix, whitened_nuisance, rcond=rcond
    )


def cramer_rao_worst_standard_deviation(
    matrix: Array, noise_standard_deviation: float
) -> float:
    """Worst-direction CR standard deviation for isotropic Gaussian noise."""
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    sigma_min = float(singular[-1])
    if sigma_min <= 0:
        return float(np.inf)
    return float(noise_standard_deviation / sigma_min)
