"""Create closed triangular STL meshes for linearly tapered NACA 4-series wings."""

from pathlib import Path

import numpy as np
from stl import mesh


def _series_code(series):
    return series if isinstance(series, str) else "".join(str(d) for d in series)


def get_NACA_4_camber(x, c, series, CL_design=0.0):
    code = _series_code(series)
    u = np.asarray(x, dtype=float) / c
    if code.startswith("UL"):
        y = np.zeros_like(u)
        interior = (u > 0) & (u < 1)
        v = u[interior]
        y[interior] = c * CL_design / (4 * np.pi) * (
            (v - 1) * np.log1p(-v) - v * np.log(v)
        )
        return y
    m, p = 0.01 * int(code[0]), 0.1 * int(code[1])
    if m == 0:
        return np.zeros_like(u)
    return c * m * np.where(
        u < p,
        (2 * p * u - u**2) / p**2,
        (1 - 2 * p + 2 * p * u - u**2) / (1 - p) ** 2,
    )


def get_NACA_4_camber_der(x, c, series, CL_design=0.0):
    code = _series_code(series)
    u = np.asarray(x, dtype=float) / c
    if code.startswith("UL"):
        if CL_design == 0:
            return np.zeros_like(u)
        with np.errstate(divide="ignore"):
            return CL_design / (4 * np.pi) * (np.log1p(-u) - np.log(u))
    m, p = 0.01 * int(code[0]), 0.1 * int(code[1])
    if m == 0:
        return np.zeros_like(u)
    return 2 * m * np.where(
        u < p, (p - u) / p**2, (p - u) / (1 - p) ** 2
    )


def _thickness_coefficients(closed):
    return (
        (2.980, -1.320, -3.286, 2.441, -0.815)
        if closed
        else (2.969, -1.260, -3.516, 2.843, -1.015)
    )


def get_NACA_4_thickness(x, c, series, closed=False):
    t_m = 0.01 * int(_series_code(series)[2:]) * c
    u = np.asarray(x, dtype=float) / c
    a, b, d, e, f = _thickness_coefficients(closed)
    thickness = t_m * (
        a * np.sqrt(u) + b * u + d * u**2 + e * u**3 + f * u**4
    )
    return np.where(u == 1, 0.0, thickness) if closed else thickness


def get_NACA_4_surface(x, c, series, closed=False, CL_design=0.0):
    x = np.asarray(x, dtype=float)
    y_c = get_NACA_4_camber(x, c, series, CL_design)
    slope = get_NACA_4_camber_der(x, c, series, CL_design)
    thickness = get_NACA_4_thickness(x, c, series, closed)
    angle = np.arctan(slope)
    x_u = x - thickness / 2 * np.sin(angle)
    x_l = x + thickness / 2 * np.sin(angle)
    y_u = y_c + thickness / 2 * np.cos(angle)
    y_l = y_c - thickness / 2 * np.cos(angle)
    return x_u, x_l, y_u, y_l


def get_NACA_4_nodes(n, c, series, closed=False, CL_design=0.0):
    if n < 3:
        raise ValueError("n must be at least 3")
    if n % 2 == 0:
        dtheta = 2 * np.pi / (n - 1)
        x = 0.5 * (1.0 - np.cos((np.arange(n // 2) + 0.5) * dtheta))
    else:
        dtheta = 2 * np.pi / (n - 1)
        x = 0.5 * (1.0 - np.cos((np.arange(n // 2) + 1) * dtheta))

    x_u, x_l, y_u, y_l = get_NACA_4_surface(
        c * x, c, series, closed, CL_design
    )
    nodes = np.zeros((n, 2))
    nodes[: n // 2, 0] = x_l[::-1]
    nodes[: n // 2, 1] = y_l[::-1]
    if n % 2 == 0:
        nodes[n // 2 :, 0] = x_u
        nodes[n // 2 :, 1] = y_u
    else:
        nodes[(n // 2) + 1 :, 0] = x_u
        nodes[(n // 2) + 1 :, 1] = y_u
    return nodes


def _validate_wing_inputs(RA, RT, b, series, n_span, n_chord, sweep):
    values = np.asarray([RA, RT, b, sweep], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Wing dimensions and sweep must be finite")
    if RA <= 0 or RT <= 0 or b <= 0:
        raise ValueError("RA (aspect ratio), RT (taper ratio), and b must be positive")
    if not isinstance(n_span, (int, np.integer)) or n_span < 2:
        raise ValueError("n_span must be an integer of at least 2")
    if not isinstance(n_chord, (int, np.integer)) or n_chord < 3:
        raise ValueError("n_chord must be an integer of at least 3")
    code = _series_code(series)
    if len(code) != 4 or not code.isdigit():
        raise ValueError("series must be a four-digit NACA series code")


def _positions(length, count, clustered):
    parameter = np.linspace(0.0, 1.0, count)
    if clustered:
        parameter = 0.5 * (1.0 - np.cos(np.pi * parameter))
    return length * parameter


def create_wing(
    RA,
    RT,
    b,
    series,
    n_span,
    n_chord,
    span_cluster=False,
    chord_cluster=False,
    sweep=0.0,
    close_tips=True,
):
    """Return triangles for a closed, full-span, linearly tapered wing.

    ``RA`` is aspect ratio, ``RT`` is tip/root taper ratio, and ``b`` is total
    span. Root and tip chord are derived from the trapezoidal planform area.
    ``n_span`` is the number of stations from centerline to either tip and
    ``n_chord`` is the number of chord stations on each surface. Sweep is the
    leading-edge sweep angle in degrees. Set ``close_tips=False`` to leave both
    tip cross-sections open. Coordinates are chord/span/thickness on x/y/z,
    respectively, and facet normals point out of the wing.
    """
    _validate_wing_inputs(RA, RT, b, series, n_span, n_chord, sweep)

    half_span = b / 2.0
    root_chord = 2.0 * b / (RA * (1.0 + RT))
    tip_chord = RT * root_chord
    positive_span = _positions(half_span, n_span, span_cluster)
    span_positions = np.concatenate((-positive_span[:0:-1], positive_span))
    loops = []

    for span_position in span_positions:
        fraction = abs(span_position) / half_span
        chord = root_chord + (tip_chord - root_chord) * fraction
        x = _positions(chord, n_chord, chord_cluster)
        x_u, x_l, z_u, z_l = get_NACA_4_surface(
            x, chord, series, closed=True
        )
        sweep_offset = abs(span_position) * np.tan(np.radians(sweep))

        # Clockwise in x-z: LE -> upper TE -> lower LE. Endpoints are shared,
        # making both leading and trailing edges watertight.
        loop_x = np.concatenate((x_u, x_l[-2:0:-1])) + sweep_offset
        loop_z = np.concatenate((z_u, z_l[-2:0:-1]))
        loop_y = np.full(loop_x.shape, span_position)
        loops.append(np.column_stack((loop_x, loop_y, loop_z)))

    loops = np.asarray(loops)
    triangles = []
    loop_size = loops.shape[1]
    for station in range(len(span_positions) - 1):
        current = loops[station]
        following = loops[station + 1]
        outer_is_current = (
            abs(span_positions[station]) > abs(span_positions[station + 1])
        )
        for index in range(loop_size):
            next_index = (index + 1) % loop_size
            # At the tapered trailing edge, point both adjacent panel
            # diagonals toward the outer (shorter-chord) station. This keeps
            # the opposite vertices upstream of the complete edge, as
            # required by Mesh.find_kutta_edges().
            use_opposite_diagonal = (
                index == n_chord - 2 and not outer_is_current
            ) or (
                index == n_chord - 1 and outer_is_current
            )
            if use_opposite_diagonal:
                triangles.append(
                    [current[index], current[next_index], following[index]]
                )
                triangles.append(
                    [current[next_index], following[next_index], following[index]]
                )
            else:
                triangles.append(
                    [current[index], current[next_index], following[next_index]]
                )
                triangles.append(
                    [current[index], following[next_index], following[index]]
                )

    if close_tips:
        # Close each tip with a fan around an interior section centroid.
        for station, reverse in ((0, True), (-1, False)):
            tip = loops[station]
            center = np.mean(tip, axis=0)
            for index in range(loop_size):
                next_index = (index + 1) % loop_size
                if reverse:
                    triangles.append([center, tip[next_index], tip[index]])
                else:
                    triangles.append([center, tip[index], tip[next_index]])

    return np.asarray(triangles, dtype=float)


def export_wing_to_stl(wing, filename="wing.stl"):
    """Write triangles returned by :func:`create_wing` to an STL file."""
    triangles = np.asarray(wing, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("wing must have shape (number_of_triangles, 3, 3)")
    if len(triangles) == 0 or not np.all(np.isfinite(triangles)):
        raise ValueError("wing must contain finite triangles")
    double_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    if np.any(double_areas == 0.0):
        raise ValueError("wing contains zero-area triangles")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    wing_mesh = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    wing_mesh.vectors[:] = triangles
    wing_mesh.update_normals()
    wing_mesh.save(str(filename))
    return filename
