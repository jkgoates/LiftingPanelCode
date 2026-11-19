"""MAE 6500 - Potential Flow: Lifting Vortex Code with Trailing Vortices
Module Author: Mason Anderson (System Setup / Solve, Comparisons, Algorithm)
Project Team:  Mason Anderson, Jeremy Goates, Nguyen Phung

Organization: Utah State University
Course: MAE 6500 - Potential Flow
Project: Lifting Vortex Code with Trailing Vortices (3D Panel Method)

References:
    Hunsaker, D. F., "Panel Code Addition to CACTUS",
        Sandia National Laboratories Report, Albuquerque, NM (Aug. 19, 2010):
        Solution Methodology (pp. 5 - 13):
            Biot-Savart Straight Segment & Vortex Ring Velocity (Eqs. 1 - 3)
            Non-Lifting Boundary Condition & Pinned System (Eqs. 4 - 7)
            Lifting Bodies: Trailing Horseshoe Vortices & Velocity (Eqs. 8 - 10)
            Lifting Boundary Condition, 3D Kutta Condition & Pinned System (Eqs. 11 - 14)
            Pressure Evaluation & Surface Force/Moment Integration (Eq. 15, p. 13)
            Surface Velocity Evaluation with Principal-Value Jump (Eq. 16)
        Appendix A: Area of a Polygon via Stokes's Theorem (Eqs. A.1 - A.14)
        Appendix B: Centroid of a Polygon (Eqs. B.1 - B.8)
        Appendix C: Least Squares Gradient Estimation (Eqs. C.1 - C.5, p. C-2)

    Katz, J., and Plotkin, A., "Low-Speed Aerodynamics", 2nd Edition:
        Chapter 8: Three-Dimensional Small-Disturbance Solutions
            §8.4    Far-Field Calculation of Induced Drag & Lift (Eqs. 8.146 - 8.147)
        Chapter 10: Three-Dimensional Singularity Elements
            §10.4.5 Straight Vortex Line Segment VORTXL (Eq. 10.116)
        Chapter 12: Three-Dimensional Numerical Solutions
            §12.1   Lifting-Line Solution by Horseshoe Elements (Eqs. 12.1 - 12.10a)
            §12.3   Lifting-Surface Solution by Vortex Ring Elements (Eqs. 12.15 - 12.28, p. 347)
            §12.5   First-Order Potential-Based Panel Methods (Eqs. 12.29 - 12.41)

    Phillips, W. F., "Mechanics of Flight", 2nd Edition:
        Chapter 2: Potential Flow Fundamentals
            §2.13 Vortex Sheets and Panels (Biot-Savart Kernel)
        Chapter 4: Panel Methods and Boundary Value Problems

Assignment / Module Cross-Reference Index:
-----------------------------------------------------------------------------------------------------------------------------
Step    Module / Primary Function           Reference Eq.          Description
-----------------------------------------------------------------------------------------------------------------------------
S0a     load_input_json()                   Input Config           Parse group project input.json (mesh, flow, output)
S0b     extract_mesh_data()                 Mesh Adapter           Extract vertices, faces, centroids, unit normals from Mesh
S0c     match_wake_to_panels()              Wake Adapter           Pair Horseshoe wake objects to adjacent TE panels (Kutta)
S1      compute_panel_geometry()            Hunsaker p.5, App. A-B Panel areas, outward normals, and centroids
S2      biot_savart_segment()               Hunsaker Eqs. 1, 3     Velocity induced by finite straight vortex segment
S3      vortex_ring_velocity()              Hunsaker Eqs. 2, 3     Velocity induced by closed polygon vortex ring
S4      trailing_horseshoe_velocity()       Hunsaker Eqs. 8, 10    Velocity induced by semi-infinite horseshoe vortex
S5      find_trailing_edge_segments()       Hunsaker p.7-9, K&P 12 Detect adjacent upper/lower panels at sharp TE
S6      build_influence_matrix()            Hunsaker Eqs. 3-14     Assemble [A] matrix (BCs, Kutta, rank pinning)
S7      build_rhs_vector()                  Hunsaker Eqs. 5, 11-14 Freestream normal velocity RHS {-V_inf . n}
S8      solve_circulation_system()          Hunsaker Eq. 14, App.C Least-squares solve for panel & wake circulations
S9      compute_centroid_velocities()       Hunsaker Eqs. 2, 9     Average induced velocity at panel control points
S10     compute_surface_velocity_jump()     Hunsaker Eq.16, C1-C5  Local least-squares gradient jump (+- 0.5 grad_s Gamma)
S11     compute_pressure_coefficients()     Hunsaker Eq. 15        Surface Cp via Bernoulli equation
S12     integrate_surface_forces()          Hunsaker p.13, K&P 12  Surface pressure force & moment integration (CL, CD, CS)
S13     kutta_joukowski_lift()              K&P Eqs. 8.147, 12.9   Far-field Trefftz plane wake lift verification
S13b    trefftz_plane_induced_drag()        K&P 8.146, 12.10a      Far-field Trefftz-plane wake induced drag (CD_KJ)
S14     evaluate_flow_field_velocity()      Hunsaker Eqs. 2, 9, 10 Velocity query at arbitrary 3D points (for streamlines)
S15     system_setup_and_solve()            Workflow Master        End-to-end driver: mesh & flight op -> full solution dict
S16     solve_from_mesh()                   src/ Integration       Direct driver for src.mesh.Mesh & input.json config
S17     Solver (class)                      src/main.py Interface  Object-oriented wrapper matching main.py Solver(mesh, wake)
-----------------------------------------------------------------------------------------------------------------------------
"""

import json
import math
import numpy as np


# =============================================================================
# S0. INPUT CONFIGURATION & SRC/MESH ADAPTERS
# =============================================================================

def load_input_json(filename="input.json"):
    """Load and parse the group project JSON configuration file (input.json).

    Supports the project schema:
        {
            "mesh":   {"filename": ..., "S_ref": ..., "c_ref": ...},
            "flow":   {"V_inf": [Vx, Vy, Vz]},
            "output": {"report_file": ..., "body_file": ..., "wake_file": ...,
                       "off_body_velocity_points": ...}
        }

    Parameters:
        filename: str, path to JSON input file

    Returns:
        input_data: dict containing parsed 'mesh', 'flow', and 'output' sections
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    mesh_cfg = data.get("mesh", {})
    flow_cfg = data.get("flow", {})
    output_cfg = data.get("output", {})

    # Handle fallback if mesh_file is placed at top level (as referenced in src/main.py)
    if "filename" not in mesh_cfg and "mesh_file" in data:
        mesh_cfg["filename"] = data["mesh_file"]

    return {
        "mesh": mesh_cfg,
        "flow": flow_cfg,
        "output": output_cfg,
    }


def extract_mesh_data(mesh):
    """Extract standardized NumPy arrays from Jeremy's src.mesh.Mesh object.
    (Adapter for src/mesh.py and src/panels.py)

    Notes on src/mesh.py conventions:
        - mesh._panel_vertex_indices stores VTK-style [3, v0_idx, v1_idx, v2_idx]
          per triangle; this strips the leading vertex count to produce (N, 3).
        - mesh.n stores unnormalized cross(v1 - v0, v2 - v0) from Tri.get_normal(),
          whose magnitude is 2 * dA; this normalizes mesh.n to unit outward normals.

    Parameters:
        mesh: Mesh instance from src.mesh (with vertices, _panel_vertex_indices,
              cp, n, and dA attributes)

    Returns:
        mesh_dict: dict with 'vertices', 'faces', 'centroids', 'normals', 'areas'
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    raw_faces = [list(row[1:]) for row in mesh._panel_vertex_indices]
    # Keep a rectangular int array when all panels share the same vertex count (e.g., all tris or all quads);
    # otherwise preserve a list of 1D int arrays for mixed-polygon meshes.
    if len(raw_faces) > 0 and len({len(f) for f in raw_faces}) == 1:
        faces = np.asarray(raw_faces, dtype=int)
    else:
        faces = [np.asarray(f, dtype=int) for f in raw_faces]
    centroids = np.asarray(mesh.cp, dtype=float)
    areas = np.asarray(mesh.dA, dtype=float)

    raw_n = np.asarray(mesh.n, dtype=float)
    n_norms = np.linalg.norm(raw_n, axis=1, keepdims=True)
    valid = n_norms[:, 0] > 1e-14
    normals = np.zeros_like(raw_n)
    normals[valid] = raw_n[valid] / n_norms[valid]
    normals[~valid] = np.array([0.0, 0.0, 1.0])

    return {
        "vertices": vertices,
        "faces": faces,
        "centroids": centroids,
        "normals": normals,
        "areas": areas,
    }


def match_wake_to_panels(vertices, faces, wake):
    """Pair Jeremy's Horseshoe wake filaments with their adjacent surface panels
    so the 3D Kutta condition (Hunsaker Eq. 12: Gamma_A - Gamma_B + Gamma_C = 0)
    is assembled with exact edge-winding signs.

    Parameters:
        vertices: (n_v, 3) ndarray of mesh vertices
        faces:    (m, 3) ndarray of triangular face vertex indices
        wake:     list of Horseshoe objects from mesh.initialize_wake()

    Returns:
        te_segments: list of dicts with 'upper_panel', 'lower_panel', 'p1', 'p2', 'delta_y'
    """
    edge_map = {}
    for f_idx, face in enumerate(faces):
        n_verts = len(face)
        for k in range(n_verts):
            a = int(face[k])
            b = int(face[(k + 1) % n_verts])
            key = (min(a, b), max(a, b))
            edge_map.setdefault(key, []).append((f_idx, a, b))

    te_segments = []
    for hs in wake:
        p1 = np.asarray(hs.v0, dtype=float)
        p2 = np.asarray(hs.v1, dtype=float)
        i0 = int(np.argmin(np.linalg.norm(vertices - p1, axis=1)))
        i1 = int(np.argmin(np.linalg.norm(vertices - p2, axis=1)))
        key = (min(i0, i1), max(i0, i1))

        shared = edge_map.get(key, [])
        if len(shared) != 2:
            raise ValueError(
                f"Wake horseshoe edge {key} is shared by {len(shared)} panels (expected 2)."
            )

        # Panel A traverses edge in same direction as horseshoe (i0 -> i1): +Gamma_A
        # Panel B traverses edge in opposite direction (i1 -> i0): -Gamma_B
        f0, a0, b0 = shared[0]
        f1, a1, b1 = shared[1]
        if a0 == i0 and b0 == i1:
            panel_a, panel_b = f0, f1
        else:
            panel_a, panel_b = f1, f0

        te_segments.append({
            "upper_panel": panel_a,
            "lower_panel": panel_b,
            "p1": p1,
            "p2": p2,
            "delta_y": abs(float(p2[1] - p1[1])),
        })

    return te_segments


# =============================================================================
# S1. PANEL GEOMETRY COMPUTATION
# =============================================================================

def compute_panel_geometry(vertices, faces):
    """Compute centroid, outward unit normal, and surface area for each panel.
    (Hunsaker Sandia Report p. 5, Appendices A & B; Katz & Plotkin §12.3, Eq. 12.20)

    Supports triangular, quadrilateral, and arbitrary planar n-sided polygon
    panels (including mixed meshes):
        - Fast vectorized path when all panels are 3-vertex triangles.
        - General planar polygon path (Hunsaker Eqs. A.14, B.8) via
          sub-triangle fan decomposition from the first vertex:
              a_k   = 0.5 * (p_k - p_0) x (p_{k+1} - p_0)
              A_vec = sum_k a_k,   A = ||A_vec||,   n = A_vec / A
              r_CG  = sum_k (a_k . n) * (p_0 + p_k + p_{k+1}) / (3 * A)

    Parameters:
        vertices: (n_v, 3) ndarray of vertex coordinates [x, y, z]
        faces:    (n_f, 3) or (n_f, 4) ndarray, or sequence of 1D integer vertex
                  index lists/arrays for mixed-polygon meshes

    Returns:
        centroids: (n_f, 3) ndarray of panel centroid coordinates
        normals:   (n_f, 3) ndarray of outward unit normal vectors
        areas:     (n_f,) ndarray of panel planar surface areas
    """
    vertices = np.asarray(vertices, dtype=float)

    # Fast vectorized path for uniform triangular meshes
    if isinstance(faces, np.ndarray) and faces.ndim == 2 and faces.shape[1] == 3:
        p0 = vertices[faces[:, 0]]
        p1 = vertices[faces[:, 1]]
        p2 = vertices[faces[:, 2]]

        # Panel centroids (Hunsaker Eq. B.8; Katz & Plotkin §12.3)
        centroids = (p0 + p1 + p2) / 3.0

        # Cross product of facet edge vectors (Hunsaker p. 5)
        cross = np.cross(p1 - p0, p2 - p0)
        cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)

        # Avoid division by zero on degenerate zero-area facets
        valid = (cross_norm[:, 0] > 1e-14)
        normals = np.zeros_like(cross)
        normals[valid] = cross[valid] / cross_norm[valid]
        normals[~valid] = np.array([0.0, 0.0, 1.0])

        # Facet surface area (Hunsaker Appendix A, Eq. A.14)
        areas = 0.5 * cross_norm[:, 0]
        return centroids, normals, areas

    # General n-sided planar polygon path (quads, n-gons, or mixed meshes)
    n_f = len(faces)
    centroids = np.zeros((n_f, 3), dtype=float)
    normals = np.zeros((n_f, 3), dtype=float)
    areas = np.zeros(n_f, dtype=float)

    for i, face in enumerate(faces):
        pts = vertices[np.asarray(face, dtype=int)]
        n_v = len(pts)
        if n_v < 3:
            raise ValueError(f"Panel {i} has fewer than 3 vertices ({n_v}).")

        p0 = pts[0]
        vec_area = np.zeros(3, dtype=float)
        sub_areas = []
        sub_centroids = []

        for k in range(1, n_v - 1):
            a_k = 0.5 * np.cross(pts[k] - p0, pts[k + 1] - p0)
            c_k = (p0 + pts[k] + pts[k + 1]) / 3.0
            vec_area += a_k
            sub_areas.append(a_k)
            sub_centroids.append(c_k)

        area_i = float(np.linalg.norm(vec_area))
        if area_i > 1e-14:
            n_i = vec_area / area_i
            # Signed sub-triangle area weights handle both convex and non-convex planar polygons
            cg_num = np.zeros(3, dtype=float)
            for a_k, c_k in zip(sub_areas, sub_centroids):
                cg_num += float(np.dot(a_k, n_i)) * c_k
            centroids[i] = cg_num / area_i
            normals[i] = n_i
            areas[i] = area_i
        else:
            centroids[i] = np.mean(pts, axis=0)
            normals[i] = np.array([0.0, 0.0, 1.0])
            areas[i] = 0.0

    return centroids, normals, areas


# =============================================================================
# S2. BIOT-SAVART STRAIGHT VORTEX FILAMENT KERNEL
# =============================================================================

def biot_savart_segment(p1, p2, r_eval, core_radius=1e-8):
    """Compute velocity induced at field point r_eval by a straight vortex segment
    from p1 to p2 with unit circulation (Gamma = 1.0).
    (Hunsaker Sandia Report Eqs. 1 & 3; Phillips & Snyder 2000; Katz & Plotkin §10.4.5, Eq. 10.116)

    Mathematical Formulation:
        Let r1_vec = r_eval - p1,  r2_vec = r_eval - p2
        v = (1 / 4*pi) * [ (r1 + r2) * (r1_vec x r2_vec) ] /
                         [ r1 * r2 * (r1 * r2 + r1_vec . r2_vec) ]

    Parameters:
        p1:          (3,) ndarray, start point of vortex segment
        p2:          (3,) ndarray, end point of vortex segment
        r_eval:      (3,) ndarray, evaluation field point
        core_radius: float, viscous vortex core radius cutoff

    Returns:
        v: (3,) ndarray, induced velocity vector [vx, vy, vz] per unit circulation
    """
    r1_vec = r_eval - p1
    r2_vec = r_eval - p2

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    if core_radius > 0:
        r1 = math.sqrt(r1**2 + core_radius**2)
        r2 = math.sqrt(r2**2 + core_radius**2)

    cross12 = np.cross(r1_vec, r2_vec)
    dot12 = np.dot(r1_vec, r2_vec)
    denom = 4.0 * math.pi * r1 * r2 * (r1 * r2 + dot12)

    if abs(denom) < 1e-12:
        return np.zeros(3)

    return (r1 + r2) * cross12 / denom


# =============================================================================
# S3. CLOSED VORTEX RING VELOCITY INFLUENCE
# =============================================================================

def vortex_ring_velocity(panel_vertices, r_eval, core_radius=1e-8):
    """Compute the velocity induced at r_eval by a closed polygonal vortex ring
    of unit circulation (Gamma = 1.0).
    (Hunsaker Sandia Report Eqs. 2 & 3; Katz & Plotkin §12.3, Eqs. 12.16 - 12.17)

    By Helmholtz's vortex theorems, summing the Biot-Savart velocity of all
    boundary segments forming a closed polygon creates a divergence-free vortex ring:
        v_ij = sum_{k=1}^n v_ijk

    Parameters:
        panel_vertices: (n, 3) ndarray of polygon vertex coordinates in cyclic order
        r_eval:         (3,) ndarray of evaluation point coordinates
        core_radius:    float, vortex core cutoff

    Returns:
        v: (3,) ndarray, induced velocity vector per unit circulation
    """
    n_edges = len(panel_vertices)
    v_total = np.zeros(3)
    for k in range(n_edges):
        p_start = panel_vertices[k]
        p_end = panel_vertices[(k + 1) % n_edges]
        v_total += biot_savart_segment(p_start, p_end, r_eval, core_radius=core_radius)
    return v_total


# =============================================================================
# S4. TRAILING HORSESHOE VORTEX INFLUENCE (SEMI-INFINITE WAKE)
# =============================================================================

def trailing_horseshoe_velocity(p1, p2, u_inf, r_eval, core_radius=1e-8):
    """Compute velocity induced at r_eval by a trailing horseshoe vortex
    shed from trailing-edge segment p1 -> p2 with unit circulation (Gamma = 1.0).
    (Hunsaker Sandia Report Eqs. 8 & 10; Katz & Plotkin §12.1, Eqs. 12.4 - 12.4a)

    A horseshoe vortex consists of three parts:
        1. Semi-infinite trailing leg coming from downstream infinity to p1
        2. Bound trailing-edge filament from p1 to p2
        3. Semi-infinite trailing leg running from p2 back to downstream infinity

    Mathematical Formulation (Hunsaker Eq. 10):
        w_il = (1 / 4*pi) * [ -(u_inf x r1_vec) / (r1 * (r1 - u_inf . r1_vec))
                            + ( (r1 + r2) * (r1_vec x r2_vec) ) / (r1 * r2 * (r1 * r2 + r1_vec . r2_vec))
                            + (u_inf x r2_vec) / (r2 * (r2 - u_inf . r2_vec)) ]

    Parameters:
        p1:          (3,) ndarray, bound segment start point
        p2:          (3,) ndarray, bound segment end point
        u_inf:       (3,) ndarray, freestream unit direction vector
        r_eval:      (3,) ndarray, evaluation field point
        core_radius: float, vortex core cutoff

    Returns:
        w: (3,) ndarray, induced velocity vector per unit wake circulation
    """
    r1_vec = r_eval - p1
    r2_vec = r_eval - p2

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    if core_radius > 0:
        r1 = math.sqrt(r1**2 + core_radius**2)
        r2 = math.sqrt(r2**2 + core_radius**2)

    u_inf = np.asarray(u_inf, dtype=float)
    u_norm = np.linalg.norm(u_inf)
    if u_norm > 1e-14:
        u_inf = u_inf / u_norm

    # 1. Trailing leg 1: from infinity to p1 along u_inf (Hunsaker Eq. 10 term 1)
    cross_u1 = np.cross(u_inf, r1_vec)
    denom_u1 = 4.0 * math.pi * r1 * (r1 - np.dot(u_inf, r1_vec))
    v_trail1 = -cross_u1 / denom_u1 if abs(denom_u1) > 1e-12 else np.zeros(3)

    # 2. Bound segment from p1 to p2 (Hunsaker Eq. 10 middle term)
    cross12 = np.cross(r1_vec, r2_vec)
    dot12 = np.dot(r1_vec, r2_vec)
    denom_b = 4.0 * math.pi * r1 * r2 * (r1 * r2 + dot12)
    v_bound = (r1 + r2) * cross12 / denom_b if abs(denom_b) > 1e-12 else np.zeros(3)

    # 3. Trailing leg 2: from p2 downstream to infinity along u_inf (Hunsaker Eq. 10 term 3)
    cross_u2 = np.cross(u_inf, r2_vec)
    denom_u2 = 4.0 * math.pi * r2 * (r2 - np.dot(u_inf, r2_vec))
    v_trail2 = cross_u2 / denom_u2 if abs(denom_u2) > 1e-12 else np.zeros(3)

    return v_trail1 + v_bound + v_trail2


# =============================================================================
# S5. TRAILING EDGE DETECTION & PAIRING
# =============================================================================

def find_trailing_edge_segments(vertices, faces, centroids, normals,
                                aft_fraction=0.5, max_cos_angle=0.2):
    """Identify sharp trailing-edge segments where upper and lower panels meet.
    (Hunsaker Sandia Report "Lifting Bodies", pp. 7 - 9; Katz & Plotkin §12.3 & §12.5, Eq. 12.33)

    Finds topological surface edges that meet three criteria:
        1. Shared by exactly two facets (internal surface edge)
        2. Aft of aft_fraction * x_max (trailing-edge region)
        3. Sharp angle between surface normals: dot(n_upper, n_lower) < max_cos_angle
        4. Spanwise oriented: abs(y2 - y1) > 1e-4 (avoids chordwise tip-cap edges)

    Parameters:
        vertices:      (n_v, 3) ndarray of vertex coordinates
        faces:         (n_f, 3) ndarray of face vertex indices
        centroids:     (n_f, 3) ndarray of panel centroids
        normals:       (n_f, 3) ndarray of panel outward normals
        aft_fraction:  float, minimum normalized x/x_max location
        max_cos_angle: float, maximum normal dot product for sharp TE classification

    Returns:
        te_segments: list of dicts, each containing:
                     'upper_panel': index of upper panel
                     'lower_panel': index of lower panel
                     'p1': (3,) ndarray start of bound segment
                     'p2': (3,) ndarray end of bound segment
                     'delta_y': float spanwise width
    """
    vertices = np.asarray(vertices, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    normals = np.asarray(normals, dtype=float)

    x_max = np.max(vertices[:, 0])
    x_min = np.min(vertices[:, 0])
    x_threshold = x_min + aft_fraction * (x_max - x_min)

    # Build half-edge dictionary: edge_key -> list of (face_idx, directed_p1, directed_p2)
    edge_map = {}
    for f_idx, face in enumerate(faces):
        n_verts = len(face)
        for k in range(n_verts):
            i1 = int(face[k])
            i2 = int(face[(k + 1) % n_verts])
            key = (min(i1, i2), max(i1, i2))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append((f_idx, i1, i2))

    te_segments = []
    for key, shared in edge_map.items():
        if len(shared) != 2:
            continue

        f_A, i1_A, i2_A = shared[0]
        f_B, i1_B, i2_B = shared[1]

        p1 = vertices[key[0]]
        p2 = vertices[key[1]]
        mid_edge = 0.5 * (p1 + p2)

        # Aft check
        if mid_edge[0] < x_threshold:
            continue

        # Spanwise width check (must have spanwise extent, not pure chordwise)
        delta_y = abs(p2[1] - p1[1])
        if delta_y < 1e-4:
            continue

        # Sharpness check
        cos_norm = np.dot(normals[f_A], normals[f_B])
        if cos_norm > max_cos_angle:
            continue

        # Differentiate upper vs lower by normal z-component
        if normals[f_A, 2] >= normals[f_B, 2]:
            upper_idx = f_A
            lower_idx = f_B
            p_start = vertices[i1_A]
            p_end = vertices[i2_A]
        else:
            upper_idx = f_B
            lower_idx = f_A
            p_start = vertices[i1_B]
            p_end = vertices[i2_B]

        # Ensure p1 -> p2 points along positive y (port to starboard) for bound vortex orientation
        if p_end[1] < p_start[1]:
            p_start, p_end = p_end, p_start

        te_segments.append({
            'upper_panel': upper_idx,
            'lower_panel': lower_idx,
            'p1': p_start,
            'p2': p_end,
            'delta_y': delta_y
        })

    # Sort trailing edges by spanwise coordinate
    te_segments.sort(key=lambda te: 0.5 * (te['p1'][1] + te['p2'][1]))
    return te_segments


# =============================================================================
# S6. INFLUENCE COEFFICIENT MATRIX ASSEMBLY
# =============================================================================

def _extract_ring_edges(vertices, faces):
    """Build flat directed-edge arrays (P1, P2, edge_panel_idx) for all surface panels."""
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        m, k_sides = faces.shape
        p_verts = vertices[faces]  # (m, k, 3)
        P1 = p_verts.reshape(-1, 3)
        P2 = np.roll(p_verts, -1, axis=1).reshape(-1, 3)
        return P1, P2, None, m, k_sides

    p1_list, p2_list, panel_idx_list = [], [], []
    for j, face in enumerate(faces):
        idx = np.asarray(face, dtype=int)
        pts = vertices[idx]
        p1_list.append(pts)
        p2_list.append(np.roll(pts, -1, axis=0))
        panel_idx_list.append(np.full(len(idx), j, dtype=int))
    P1 = np.vstack(p1_list)
    P2 = np.vstack(p2_list)
    edge_panel = np.concatenate(panel_idx_list)
    return P1, P2, edge_panel, len(faces), None


def _horseshoe_influence_batch(eval_pts, te_segments, u_inf, core_radius=1e-8):
    """Vectorized semi-infinite horseshoe influence (Hunsaker Eq. 10) at eval_pts (B, 3).
    Returns W of shape (B, p, 3).
    """
    b_len = len(eval_pts)
    p = len(te_segments)
    if p == 0:
        return np.zeros((b_len, 0, 3), dtype=float)

    P1 = np.array([te['p1'] for te in te_segments], dtype=float)  # (p, 3)
    P2 = np.array([te['p2'] for te in te_segments], dtype=float)  # (p, 3)
    u = np.asarray(u_inf, dtype=float)
    u_norm = np.linalg.norm(u)
    if u_norm > 1e-14:
        u = u / u_norm

    dx1 = eval_pts[:, 0:1] - P1[:, 0]
    dy1 = eval_pts[:, 1:2] - P1[:, 1]
    dz1 = eval_pts[:, 2:3] - P1[:, 2]
    dx2 = eval_pts[:, 0:1] - P2[:, 0]
    dy2 = eval_pts[:, 1:2] - P2[:, 1]
    dz2 = eval_pts[:, 2:3] - P2[:, 2]

    cr2 = core_radius * core_radius
    r1 = np.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + cr2)
    r2 = np.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + cr2)

    # 1. Trailing leg 1: -(u x r1) / (4 pi r1 (r1 - u . r1))
    u_dot_r1 = u[0] * dx1 + u[1] * dy1 + u[2] * dz1
    den1 = 4.0 * math.pi * r1 * (r1 - u_dot_r1)
    ok1 = np.abs(den1) > 1e-12
    f1 = np.zeros_like(den1)
    f1[ok1] = -1.0 / den1[ok1]
    wx = f1 * (u[1] * dz1 - u[2] * dy1)
    wy = f1 * (u[2] * dx1 - u[0] * dz1)
    wz = f1 * (u[0] * dy1 - u[1] * dx1)

    # 2. Bound segment: (r1 + r2)(r1 x r2) / (4 pi r1 r2 (r1 r2 + r1 . r2))
    dot12 = dx1 * dx2 + dy1 * dy2 + dz1 * dz2
    denb = 4.0 * math.pi * r1 * r2 * (r1 * r2 + dot12)
    okb = np.abs(denb) > 1e-12
    fb = np.zeros_like(denb)
    fb[okb] = (r1[okb] + r2[okb]) / denb[okb]
    wx += fb * (dy1 * dz2 - dz1 * dy2)
    wy += fb * (dz1 * dx2 - dx1 * dz2)
    wz += fb * (dx1 * dy2 - dy1 * dx2)

    # 3. Trailing leg 2: +(u x r2) / (4 pi r2 (r2 - u . r2))
    u_dot_r2 = u[0] * dx2 + u[1] * dy2 + u[2] * dz2
    den2 = 4.0 * math.pi * r2 * (r2 - u_dot_r2)
    ok2 = np.abs(den2) > 1e-12
    f2 = np.zeros_like(den2)
    f2[ok2] = 1.0 / den2[ok2]
    wx += f2 * (u[1] * dz2 - u[2] * dy2)
    wy += f2 * (u[2] * dx2 - u[0] * dz2)
    wz += f2 * (u[0] * dy2 - u[1] * dx2)

    return np.stack([wx, wy, wz], axis=-1)


def build_influence_matrix(vertices, faces, centroids, normals, te_segments,
                           u_inf, pin_panel_idx=0, core_radius=1e-8,
                           compute_v_tensor=None, chunk_size=128):
    """Assemble the aerodynamic influence coefficient matrix [A] (vectorized, chunked).
    (Hunsaker Sandia Report Eqs. 3 - 14; Katz & Plotkin Eqs. 12.22, 12.33)

    Structure of System Matrix [A] (size (m + p + 1) x (m + p)):
        - Rows 0 to m-1: Flow tangency boundary conditions (V . n = 0) at all m panel centroids:
            sum_{j=1}^m (v_ij . n_i) * Gamma_j + sum_{l=1}^p (w_il . n_i) * Gamma_{w,l} = -V_inf . n_i
            (Hunsaker Eqs. 4 & 11; Katz & Plotkin Eqs. 12.1, 12.22)

        - Rows m to m+p-1: Kutta condition at each sharp trailing-edge segment:
            Gamma_A - Gamma_B + Gamma_C = 0
            where A is upper panel, B is lower panel, C is wake horseshoe l
            (Hunsaker Eq. 12; Katz & Plotkin §12.5, Eq. 12.33)

        - Row m+p: Singularity / rank-deficiency resolution constraint:
            Gamma_{pin} = 0
            (Hunsaker Eq. 14 for lifting bodies with trailing vortices)

    Parameters:
        vertices:         (n_v, 3) ndarray of mesh vertices
        faces:            (n_f, 3)/(n_f, 4) ndarray or list of vertex index arrays
        centroids:        (n_f, 3) ndarray of panel centroids (control points)
        normals:          (n_f, 3) ndarray of outward unit normals
        te_segments:      list of trailing edge segment dictionaries
        u_inf:            (3,) ndarray, freestream unit direction vector
        pin_panel_idx:    int, index of panel pinned to zero circulation (default: 0)
        core_radius:      float, vortex core cutoff radius
        compute_v_tensor: bool or None (if None, computed only when m <= 1500 to save RAM)
        chunk_size:       int, row batch size for vectorized Biot-Savart evaluation

    Returns:
        A:        ((m + p + 1), (m + p)) ndarray, full influence matrix
        V_tensor: (m, (m + p), 3) ndarray or None (when m > 1500 to avoid GB allocation)
    """
    m = len(faces)
    p = len(te_segments)
    n_vars = m + p
    n_eqs = m + p + 1

    A = np.zeros((n_eqs, n_vars), dtype=float)
    if compute_v_tensor is None:
        compute_v_tensor = (m <= 1500)
    V_tensor = np.zeros((m, n_vars, 3), dtype=float) if compute_v_tensor else None

    P1, P2, edge_panel, _, k_sides = _extract_ring_edges(vertices, faces)
    cr2 = core_radius * core_radius

    # 1. Flow tangency boundary condition: Rows 0 to m-1 (Hunsaker Eqs. 4 & 11)
    for i0 in range(0, m, chunk_size):
        i1 = min(i0 + chunk_size, m)
        b_len = i1 - i0
        ci = centroids[i0:i1]
        ni = normals[i0:i1]

        dx1 = ci[:, 0:1] - P1[:, 0]
        dy1 = ci[:, 1:2] - P1[:, 1]
        dz1 = ci[:, 2:3] - P1[:, 2]
        dx2 = ci[:, 0:1] - P2[:, 0]
        dy2 = ci[:, 1:2] - P2[:, 1]
        dz2 = ci[:, 2:3] - P2[:, 2]

        r1 = np.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + cr2)
        r2 = np.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + cr2)

        cx = dy1 * dz2 - dz1 * dy2
        cy = dz1 * dx2 - dx1 * dz2
        cz = dx1 * dy2 - dy1 * dx2
        dot12 = dx1 * dx2 + dy1 * dy2 + dz1 * dz2
        den = 4.0 * math.pi * r1 * r2 * (r1 * r2 + dot12)

        ok = np.abs(den) > 1e-12
        fac = np.zeros_like(den)
        fac[ok] = (r1[ok] + r2[ok]) / den[ok]

        ex = fac * cx
        ey = fac * cy
        ez = fac * cz

        if k_sides is not None:
            vx = ex.reshape(b_len, m, k_sides).sum(axis=2)
            vy = ey.reshape(b_len, m, k_sides).sum(axis=2)
            vz = ez.reshape(b_len, m, k_sides).sum(axis=2)
        else:
            vx = np.zeros((b_len, m), dtype=float)
            vy = np.zeros((b_len, m), dtype=float)
            vz = np.zeros((b_len, m), dtype=float)
            for row in range(b_len):
                vx[row] = np.bincount(edge_panel, weights=ex[row], minlength=m)
                vy[row] = np.bincount(edge_panel, weights=ey[row], minlength=m)
                vz[row] = np.bincount(edge_panel, weights=ez[row], minlength=m)

        A[i0:i1, :m] = vx * ni[:, 0:1] + vy * ni[:, 1:2] + vz * ni[:, 2:3]
        if V_tensor is not None:
            V_tensor[i0:i1, :m, 0] = vx
            V_tensor[i0:i1, :m, 1] = vy
            V_tensor[i0:i1, :m, 2] = vz

        if p > 0:
            W = _horseshoe_influence_batch(ci, te_segments, u_inf, core_radius=core_radius)
            A[i0:i1, m:] = np.einsum('blj,bj->bl', W, ni)
            if V_tensor is not None:
                V_tensor[i0:i1, m:, :] = W

    # 2. Kutta condition: Rows m to m+p-1 (Hunsaker Eq. 12)
    for l in range(p):
        te = te_segments[l]
        row_idx = m + l
        A[row_idx, te['upper_panel']] = 1.0   # Gamma_A
        A[row_idx, te['lower_panel']] = -1.0  # -Gamma_B
        A[row_idx, m + l]             = 1.0   # +Gamma_C

    # 3. Rank deficiency pinning equation: Row m+p (Hunsaker Eq. 14)
    A[m + p, pin_panel_idx] = 1.0

    return A, V_tensor


# =============================================================================
# S7. RIGHT-HAND-SIDE FREESTREAM NORMAL VELOCITY VECTOR
# =============================================================================

def build_rhs_vector(normals, V_inf_vec, n_te_segments):
    """Assemble right-hand side vector {b} for the aerodynamic linear system.
    (Hunsaker Sandia Report Eqs. 5, 11, 13 - 14; Katz & Plotkin Eqs. 12.8, 12.24)

    Structure of RHS Vector {b} (size m + p + 1):
        - Elements 0 to m-1: -V_inf . n_i  (freestream normal flow at centroids)
        - Elements m to m+p-1: 0.0         (Kutta condition: Gamma_A - Gamma_B + Gamma_C = 0)
        - Element m+p: 0.0                 (Pinned circulation: Gamma_pin = 0)

    Parameters:
        normals:        (m, 3) ndarray of panel outward normals
        V_inf_vec:      (3,) ndarray of freestream velocity vector [V_inf_x, V_inf_y, V_inf_z]
        n_te_segments:  int, number of trailing edge segments p

    Returns:
        b: (m + p + 1,) ndarray, right-hand-side vector
    """
    m = len(normals)
    p = n_te_segments
    b = np.zeros(m + p + 1, dtype=float)

    # Freestream normal flux on each surface panel (Hunsaker Eqs. 5 & 11)
    b[:m] = -np.dot(normals, V_inf_vec)

    # Kutta condition rows and pinning row remain 0.0 (Hunsaker Eqs. 12 & 14)
    return b


# =============================================================================
# S8. CIRCULATION SYSTEM LEAST-SQUARES SOLVE
# =============================================================================

def _gmres_normal(A, b, tol=1e-8, restart=80, maxiter=800):
    """Pure-NumPy Jacobi-preconditioned restarted GMRES on the normal equations
    [A^T A]{x} = [A^T]{b} (matrix-free matvec A^T(A v) avoids forming A^T A).
    """
    ATb = A.T @ b
    n = A.shape[1]
    col_norm_sq = np.sum(A * A, axis=0)
    col_norm_sq[col_norm_sq < 1e-14] = 1.0
    M_inv = 1.0 / col_norm_sq

    x = np.zeros(n, dtype=float)
    b_norm = float(np.linalg.norm(M_inv * ATb))
    if b_norm == 0.0:
        return x

    def matvec(v):
        return M_inv * (A.T @ (A @ v))

    rhs = M_inv * ATb
    n_restarts = max(1, maxiter // restart)
    for _ in range(n_restarts):
        r = rhs - matvec(x)
        beta = float(np.linalg.norm(r))
        if beta / b_norm < tol:
            break

        V = np.zeros((restart + 1, n), dtype=float)
        H = np.zeros((restart + 1, restart), dtype=float)
        V[0] = r / beta
        g = np.zeros(restart + 1, dtype=float)
        g[0] = beta
        cs = np.zeros(restart, dtype=float)
        sn = np.zeros(restart, dtype=float)
        k_end = restart

        for j in range(restart):
            w = matvec(V[j])
            for i in range(j + 1):
                h_ij = float(np.dot(w, V[i]))
                H[i, j] = h_ij
                w -= h_ij * V[i]
            h_next = float(np.linalg.norm(w))
            H[j + 1, j] = h_next
            if h_next > 1e-14:
                V[j + 1] = w / h_next

            for i in range(j):
                tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp

            denom = math.hypot(H[j, j], H[j + 1, j])
            if denom > 1e-16:
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom
            else:
                cs[j], sn[j] = 1.0, 0.0
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            tmp = cs[j] * g[j]
            g[j + 1] = -sn[j] * g[j]
            g[j] = tmp

            if abs(g[j + 1]) / b_norm < tol:
                k_end = j + 1
                break

        y = np.linalg.solve(H[:k_end, :k_end], g[:k_end])
        x += y @ V[:k_end]
        if abs(g[k_end]) / b_norm < tol:
            break

    return x


def solve_circulation_system(A, b, method="auto", tol=1e-8, maxiter=800, restart=80,
                             row_scale=True):
    """Solve the overdetermined linear system [A]{Gamma} = {b}.
    (Hunsaker Sandia Report Eq. 14 & Appendix C; Katz & Plotkin §12.3)

    Methods:
        - 'lstsq':  Full SVD least squares (np.linalg.lstsq).
        - 'normal': Normal equations [Aw^T Aw]{Gamma} = [Aw^T]{bw} via BLAS/LAPACK.
        - 'qr':     Reduced QR least squares without squaring condition number.
        - 'gmres':  Matrix-free Jacobi-preconditioned restarted GMRES on normal equations.
        - 'auto':   Uses 'lstsq' for n_vars <= 1000 and 'normal' for n_vars > 1000
                    (falling back to QR/'lstsq' if ill-conditioned).

    Parameters:
        A:         ((m + p + 1), (m + p)) ndarray, influence matrix
        b:         (m + p + 1,) ndarray, right-hand side vector
        method:    str, 'auto', 'lstsq', 'normal', 'qr', or 'gmres'
        tol:       float, relative residual tolerance for GMRES
        maxiter:   int, max GMRES iterations
        restart:   int, GMRES Krylov subspace restart size
        row_scale: bool, if True equilibrate each row by its L2 norm so tiny sliver
                   panels do not dominate least-squares weighting (default: True)

    Returns:
        gamma_total: (m + p,) ndarray, solved circulation values
        residuals:   (1,) ndarray, physical unweighted sum of squared residuals
    """
    n_vars = A.shape[1]
    if row_scale:
        row_norms = np.linalg.norm(A, axis=1)
        row_norms[row_norms < 1e-12] = 1.0
        Aw = A / row_norms[:, None]
        bw = b / row_norms
    else:
        Aw = A
        bw = b

    gamma_total = None
    if method == "gmres":
        gamma_total = _gmres_normal(Aw, bw, tol=tol, restart=restart, maxiter=maxiter)
    elif method == "normal" or (method == "auto" and n_vars > 1000):
        try:
            ATA = Aw.T @ Aw
            ATb = Aw.T @ bw
            gamma_total = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            gamma_total = None
    elif method == "qr":
        try:
            Q, R = np.linalg.qr(Aw, mode="reduced")
            gamma_total = np.linalg.solve(R, Q.T @ bw)
        except np.linalg.LinAlgError:
            gamma_total = None

    if gamma_total is None:
        if method in ("normal", "qr") or (method == "auto" and n_vars > 1000):
            try:
                Q, R = np.linalg.qr(Aw, mode="reduced")
                gamma_total = np.linalg.solve(R, Q.T @ bw)
            except np.linalg.LinAlgError:
                gamma_total = None
        if gamma_total is None:
            gamma_total, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)

    res = A @ gamma_total - b
    return gamma_total, np.array([float(np.dot(res, res))])


# =============================================================================
# S9. CENTROID INDUCED VELOCITY EVALUATION
# =============================================================================

def compute_centroid_velocities(vertices, faces, centroids, te_segments,
                                gamma_panels, gamma_wake, V_inf_vec,
                                V_tensor=None, core_radius=1e-8, chunk_size=128):
    """Compute the total velocity at each panel centroid (control point).
    (Hunsaker Sandia Report Eqs. 2 & 9; Katz & Plotkin §12.5, Eq. 12.39)

    Total velocity at centroid i:
        V_i = V_inf + sum_{j=1}^m Gamma_j * v_ij + sum_{l=1}^p Gamma_{w,l} * w_il

    If V_tensor was precomputed during influence matrix assembly (S6), this computes
    the velocity field via tensor contraction; otherwise it evaluates the chunked
    vectorized Biot-Savart sum directly without allocating an (m, m, 3) tensor.

    Parameters:
        vertices:     (n_v, 3) ndarray of mesh vertices
        faces:        (m, 3)/(m, 4) ndarray or list of panel vertex indices
        centroids:    (m, 3) ndarray of panel centroids
        te_segments:  list of trailing edge segment dictionaries
        gamma_panels: (m,) ndarray of panel circulations
        gamma_wake:   (p,) ndarray of wake circulations
        V_inf_vec:    (3,) ndarray of freestream velocity
        V_tensor:     (m, (m + p), 3) optional precomputed influence tensor
        core_radius:  float, vortex core cutoff radius
        chunk_size:   int, batch size for vectorized evaluation when V_tensor is None

    Returns:
        V_centroids: (m, 3) ndarray of velocity vectors at panel centroids
    """
    m = len(centroids)
    p = len(te_segments)
    gamma_total = np.concatenate([gamma_panels, gamma_wake])

    if V_tensor is not None:
        # Fast vector product: sum_k V_tensor[i, k, :] * gamma[k] (Hunsaker Eqs. 2 & 9)
        V_ind = np.tensordot(V_tensor, gamma_total, axes=([1], [0]))  # shape (m, 3)
        return V_inf_vec + V_ind

    V_centroids = np.tile(V_inf_vec, (m, 1))
    P1, P2, edge_panel, _, k_sides = _extract_ring_edges(vertices, faces)
    if k_sides is not None:
        edge_gamma = np.repeat(gamma_panels, k_sides)
    else:
        edge_gamma = gamma_panels[edge_panel]

    u_inf = V_inf_vec / np.linalg.norm(V_inf_vec)
    cr2 = core_radius * core_radius

    for i0 in range(0, m, chunk_size):
        i1 = min(i0 + chunk_size, m)
        ci = centroids[i0:i1]

        dx1 = ci[:, 0:1] - P1[:, 0]
        dy1 = ci[:, 1:2] - P1[:, 1]
        dz1 = ci[:, 2:3] - P1[:, 2]
        dx2 = ci[:, 0:1] - P2[:, 0]
        dy2 = ci[:, 1:2] - P2[:, 1]
        dz2 = ci[:, 2:3] - P2[:, 2]

        r1 = np.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + cr2)
        r2 = np.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + cr2)

        cx = dy1 * dz2 - dz1 * dy2
        cy = dz1 * dx2 - dx1 * dz2
        cz = dx1 * dy2 - dy1 * dx2
        dot12 = dx1 * dx2 + dy1 * dy2 + dz1 * dz2
        den = 4.0 * math.pi * r1 * r2 * (r1 * r2 + dot12)

        ok = np.abs(den) > 1e-12
        fac = np.zeros_like(den)
        fac[ok] = (r1[ok] + r2[ok]) / den[ok]

        V_centroids[i0:i1, 0] += (fac * cx) @ edge_gamma
        V_centroids[i0:i1, 1] += (fac * cy) @ edge_gamma
        V_centroids[i0:i1, 2] += (fac * cz) @ edge_gamma

        if p > 0:
            W = _horseshoe_influence_batch(ci, te_segments, u_inf, core_radius=core_radius)
            V_centroids[i0:i1] += np.einsum('blj,l->bj', W, gamma_wake)

    return V_centroids


# =============================================================================
# S10. LOCAL LEAST-SQUARES SURFACE VELOCITY JUMP (APPENDIX C)
# =============================================================================

def compute_surface_velocity_jump(centroids, normals, gamma_panels, V_centroids,
                                  n_neighbors=12, faces=None, te_segments=None,
                                  min_normal_dot=0.80):
    """Compute physical surface velocities incorporating the discontinuous
    tangential velocity jump across the surface vortex sheet.
    (Hunsaker Sandia Report Eq. 16 & Appendix C, Eqs. C.1 - C.5, p. C-2; Katz & Plotkin §12.5, Eqs. 12.37 - 12.39)

    Theoretical Context:
        Biot-Savart evaluation directly on the boundary only computes the mean
        (principal value) velocity across the vortex sheet. For outward unit
        normals, the physical outer-surface velocity (Hunsaker Eq. 16, V1) is:
            V_surf,i = V_t,i - 0.5 * (grad_s(Gamma_i) . v_hat_t,i) * v_hat_t,i

        where grad_s(Gamma_i) is fitted via inverse-distance-weighted regularized
        least squares in panel i's local tangent plane over same-side neighbors
        (excluding sharp-edge neighbors with dot(n_i, n_j) < min_normal_dot and
        Kutta TE pairs per Appendix C p. C-2).

    Parameters:
        centroids:      (m, 3) ndarray of panel centroids
        normals:        (m, 3) ndarray of panel outward normals
        gamma_panels:   (m,) ndarray of panel circulations
        V_centroids:    (m, 3) ndarray of average centroid velocities
        n_neighbors:    int, target neighbor count for local gradient stencil (default: 12)
        faces:          optional face connectivity for topological Appendix C neighbors
        te_segments:    optional trailing-edge segment list to exclude Kutta pairs
        min_normal_dot: float, minimum normal dot product to include neighbor (default: 0.80)

    Returns:
        V_surf: (m, 3) ndarray of true physical surface velocity vectors
    """
    m = len(centroids)
    V_surf = np.zeros_like(V_centroids)

    vert_to_panels = None
    if faces is not None:
        vert_to_panels = {}
        for fi, face in enumerate(faces):
            for vi in face:
                vert_to_panels.setdefault(int(vi), set()).add(fi)

    te_pairs = set()
    if te_segments is not None:
        for te in te_segments:
            u_p = int(te['upper_panel'])
            l_p = int(te['lower_panel'])
            te_pairs.add((u_p, l_p))
            te_pairs.add((l_p, u_p))

    for i in range(m):
        r_i = centroids[i]
        n_i = normals[i]
        g_i = gamma_panels[i]

        # Build local orthonormal tangent basis (t1, t2) on panel i's plane (App. C)
        ref = np.array([1.0, 0.0, 0.0]) if abs(n_i[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n_i, ref)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n_i, t1)

        nearest = np.array([], dtype=int)
        if vert_to_panels is not None:
            nbrs = set()
            for vi in faces[i]:
                nbrs |= vert_to_panels[int(vi)]
            nbrs.discard(i)
            topo_valid = [
                j for j in nbrs
                if np.dot(n_i, normals[j]) >= min_normal_dot and (i, j) not in te_pairs
            ]
            if len(topo_valid) >= 4:
                nearest = np.array(topo_valid, dtype=int)

        # Fallback / expand to same-side nearest neighbors if 1-ring has < 4 smooth panels
        if len(nearest) < 4:
            same_side = np.dot(normals, n_i) >= min_normal_dot
            same_side[i] = False
            for j_te in (p_idx for (p0, p_idx) in te_pairs if p0 == i):
                same_side[j_te] = False
            valid_idx = np.flatnonzero(same_side)
            if len(valid_idx) >= 2:
                dists = np.linalg.norm(centroids[valid_idx] - r_i, axis=1)
                nearest = valid_idx[np.argsort(dists)[:n_neighbors]]

        grad_s = np.zeros(3)
        if len(nearest) >= 2:
            delta_r = centroids[nearest] - r_i
            dist_nbr = np.linalg.norm(delta_r, axis=1)
            weights = 1.0 / np.maximum(dist_nbr, 1e-12)
            A_grad = np.column_stack([delta_r @ t1, delta_r @ t2]) * weights[:, None]
            delta_gamma = (gamma_panels[nearest] - g_i) * weights

            # Regularized 2D least-squares fit in tangent plane (Hunsaker Eqs. C.4 - C.5)
            coeffs, _, rank, _ = np.linalg.lstsq(A_grad, delta_gamma, rcond=0.05)
            if rank == 2:
                grad_s = coeffs[0] * t1 + coeffs[1] * t2

        # Tangential velocity vector at centroid
        v_tang = V_centroids[i] - np.dot(V_centroids[i], n_i) * n_i
        speed_tang = np.linalg.norm(v_tang)

        # Outer surface velocity V1 with half-sheet tangential jump (Hunsaker Eq. 16, Fig. 5)
        if speed_tang > 1e-10:
            v_dir = v_tang / speed_tang
            V_surf[i] = v_tang - 0.5 * np.dot(grad_s, v_dir) * v_dir
        else:
            V_surf[i] = v_tang - 0.5 * grad_s

    return V_surf


# =============================================================================
# S11. SURFACE PRESSURE COEFFICIENT DISTRIBUTION
# =============================================================================

def compute_pressure_coefficients(V_surf, V_inf_mag):
    """Compute surface pressure coefficient Cp at each panel from Bernoulli's equation.
    (Hunsaker Sandia Report Eq. 15; Katz & Plotkin §12.5, Eq. 12.40)

    Mathematical Formulation:
        Cp_i = 1.0 - ( ||V_surf,i|| / V_inf )^2

    Parameters:
        V_surf:    (m, 3) ndarray of surface velocity vectors
        V_inf_mag: float, freestream velocity magnitude

    Returns:
        Cp: (m,) ndarray of surface pressure coefficients
    """
    V_mag_sq = np.sum(V_surf**2, axis=1)
    Cp = 1.0 - V_mag_sq / (V_inf_mag**2)
    return Cp


# =============================================================================
# S12. SURFACE PRESSURE INTEGRATION FOR AERODYNAMIC FORCES & MOMENTS
# =============================================================================

def integrate_surface_forces(Cp, areas, normals, centroids, alpha_rad, beta_rad,
                             S_ref, b_ref, c_ref, r_ref=np.zeros(3),
                             rho_inf=1.225, V_inf_mag=10.0):
    """Integrate surface pressure coefficients to obtain net aerodynamic forces
    and moments in body and wind axes.
    (Hunsaker Sandia Report p. 13; Katz & Plotkin §12.5, Eq. 12.41)

    Mathematical Formulation:
        Body force:
            F_body = -q_inf * sum_{i=1}^m Cp_i * A_i * n_i
        Body moment about r_ref:
            M_body = -q_inf * sum_{i=1}^m Cp_i * A_i * [ (r_i - r_ref) x n_i ]

        Transformation to Wind Axes (Lift, Induced Drag, Side Force):
            L = -F_z * cos(alpha) + F_x * sin(alpha)
            D =  F_x * cos(alpha) * cos(beta) + F_y * sin(beta) + F_z * sin(alpha) * cos(beta)
            Y = -F_x * cos(alpha) * sin(beta) + F_y * cos(beta) - F_z * sin(alpha) * sin(beta)

    Parameters:
        Cp:        (m,) ndarray of panel pressure coefficients
        areas:     (m,) ndarray of panel surface areas
        normals:   (m, 3) ndarray of panel outward unit normals
        centroids: (m, 3) ndarray of panel centroids
        alpha_rad: float, angle of attack in radians
        beta_rad:  float, sideslip angle in radians
        S_ref:     float, reference planform area
        b_ref:     float, reference wingspan
        c_ref:     float, reference chord
        r_ref:     (3,) ndarray, moment reference center (default: [0, 0, 0])
        rho_inf:   float, freestream density (kg/m^3)
        V_inf_mag: float, freestream velocity magnitude (m/s)

    Returns:
        forces_dict: dict containing total forces, moments, and dimensionless coefficients:
                     'CL', 'CD_press', 'CS', 'Cm_x', 'Cm_y', 'Cm_z',
                     'Lift', 'Drag', 'SideForce', 'F_body', 'M_body'
    """
    q_inf = 0.5 * rho_inf * (V_inf_mag**2)

    # Differential force on each panel: dF_i = -Cp_i * q_inf * A_i * n_i (Hunsaker p. 13; K&P Eq. 12.41)
    dF = -Cp[:, np.newaxis] * q_inf * areas[:, np.newaxis] * normals
    F_body = np.sum(dF, axis=0)

    # Differential moment about r_ref: dM_i = (r_i - r_ref) x dF_i (Hunsaker p. 13)
    r_arm = centroids - r_ref
    dM = np.cross(r_arm, dF)
    M_body = np.sum(dM, axis=0)

    # Body-to-wind transformation (Phillips §1.3)
    cos_a = math.cos(alpha_rad)
    sin_a = math.sin(alpha_rad)
    cos_b = math.cos(beta_rad)
    sin_b = math.sin(beta_rad)

    Fx = F_body[0]
    Fy = F_body[1]
    Fz = F_body[2]

    # Lift (perpendicular to V_inf in pitch plane, positive upward along +z)
    Lift = Fz * cos_a - Fx * sin_a

    # Drag (aligned with V_inf)
    Drag = Fx * cos_a * cos_b + Fy * sin_b + Fz * sin_a * cos_b

    # Side force
    SideForce = -Fx * cos_a * sin_b + Fy * cos_b - Fz * sin_a * sin_b

    # Non-dimensional coefficients
    denom_force = q_inf * S_ref
    CL = Lift / denom_force
    CD_press = Drag / denom_force
    CS = SideForce / denom_force

    Cmx = M_body[0] / (denom_force * b_ref)
    Cmy = M_body[1] / (denom_force * c_ref)
    Cmz = M_body[2] / (denom_force * b_ref)

    return {
        'CL': CL,
        'CD_press': CD_press,
        'CS': CS,
        'Cm_x': Cmx,
        'Cm_y': Cmy,
        'Cm_z': Cmz,
        'Lift': Lift,
        'Drag': Drag,
        'SideForce': SideForce,
        'F_body': F_body,
        'M_body': M_body
    }


# =============================================================================
# S13. TREFFTZ-PLANE / KUTTA-JOUKOWSKI FAR-FIELD LIFT VERIFICATION
# =============================================================================

def kutta_joukowski_lift(gamma_wake, te_segments, V_inf_mag, S_ref, rho_inf=1.225):
    """Compute total lift using the far-field Trefftz-plane Kutta-Joukowski theorem.
    (Katz & Plotkin §8.4 Eq. 8.147 & §12.1 Eq. 12.9)

    Theoretical Context:
        Far downstream in the Trefftz plane, the net vertical momentum flux
        carried by the trailing horseshoe vortices matches the bound circulation:
            L_KJ = rho_inf * V_inf * sum_{l=1}^p Gamma_{w,l} * delta_y_l
            CL_KJ = (2 / (V_inf * S_ref)) * sum_{l=1}^p Gamma_{w,l} * delta_y_l

        Comparing CL_KJ to the surface-pressure integrated CL provides an essential
        independent consistency check on mesh discretization and solver convergence.

    Parameters:
        gamma_wake:  (p,) ndarray of solved wake circulations
        te_segments: list of trailing edge segment dictionaries
        V_inf_mag:   float, freestream velocity magnitude
        S_ref:       float, reference wing area
        rho_inf:     float, freestream density

    Returns:
        Lift_KJ: float, total lift force via Kutta-Joukowski (N)
        CL_KJ:   float, non-dimensional lift coefficient via Kutta-Joukowski
    """
    total_circ_span = 0.0
    for l, te in enumerate(te_segments):
        # Signed bound circulation * dy: Gamma_bound * dy = Gamma_wake * (p2.y - p1.y)
        # preserves exact lift sign for positive and negative angles of attack
        total_circ_span += gamma_wake[l] * (te['p2'][1] - te['p1'][1])

    Lift_KJ = rho_inf * V_inf_mag * total_circ_span
    CL_KJ = (2.0 / (V_inf_mag * S_ref)) * total_circ_span

    return Lift_KJ, CL_KJ


# =============================================================================
# S13b. TREFFTZ-PLANE FAR-FIELD INDUCED DRAG
# =============================================================================

def trefftz_plane_induced_drag(gamma_wake, te_segments, V_inf_mag, S_ref, rho_inf=1.225):
    """Compute induced drag via far-field Trefftz-plane wake integration.
    (Katz & Plotkin §8.4 Eq. 8.146; §12.1 Eq. 12.10a; §12.3 p. 347)

    Evaluates the downwash at each wake strip midpoint ym_i induced by all
    semi-infinite trailing vortex pairs and integrates:
        D_KJ = rho_inf * sum_i G_bound,i * w_wing,i * dy_i
        CD_KJ = D_KJ / (q_inf * S_ref)

    Parameters:
        gamma_wake:  (p,) ndarray of solved wake circulations
        te_segments: list of trailing edge segment dictionaries
        V_inf_mag:   float, freestream velocity magnitude
        S_ref:       float, reference wing area
        rho_inf:     float, freestream density

    Returns:
        Drag_KJ: float, Trefftz-plane induced drag force (N)
        CD_KJ:   float, Trefftz-plane induced drag coefficient
    """
    p = len(te_segments)
    if p == 0:
        return 0.0, 0.0

    y1 = np.array([min(te['p1'][1], te['p2'][1]) for te in te_segments], dtype=float)
    y2 = np.array([max(te['p1'][1], te['p2'][1]) for te in te_segments], dtype=float)
    ym = 0.5 * (y1 + y2)
    dy = y2 - y1

    # Signed bound circulation (+ for upward lift): Gamma_bound = gamma_wake * sign(p2.y - p1.y)
    signs = np.array([1.0 if te['p2'][1] >= te['p1'][1] else -1.0 for te in te_segments], dtype=float)
    G_bound = np.asarray(gamma_wake, dtype=float) * signs

    # Downwash at each wake strip midpoint ym_i (positive downward):
    # w_wing,i = (1 / 4*pi) * sum_j G_bound_j * [1/(ym_i - y1_j) - 1/(ym_i - y2_j)]
    d1 = ym[:, None] - y1[None, :]
    d2 = ym[:, None] - y2[None, :]
    inv1 = np.where(np.abs(d1) > 1e-12, 1.0 / np.where(np.abs(d1) > 1e-12, d1, 1.0), 0.0)
    inv2 = np.where(np.abs(d2) > 1e-12, 1.0 / np.where(np.abs(d2) > 1e-12, d2, 1.0), 0.0)
    w_wing = (1.0 / (4.0 * math.pi)) * ((inv1 - inv2) @ G_bound)

    Drag_KJ = float(rho_inf * np.sum(G_bound * w_wing * dy))
    q_inf = 0.5 * rho_inf * (V_inf_mag ** 2)
    CD_KJ = Drag_KJ / (q_inf * S_ref)
    return Drag_KJ, CD_KJ


# =============================================================================
# S14. FLOW-FIELD VELOCITY EVALUATION (STREAMLINE QUERY)
# =============================================================================

def evaluate_flow_field_velocity(r_eval, vertices, faces, te_segments,
                                 gamma_panels, gamma_wake, V_inf_vec,
                                 core_radius=1e-8):
    """Compute potential flow velocity at an arbitrary 3D spatial field point.
    (Hunsaker Sandia Report Eqs. 2, 3, 9 & 10; Katz & Plotkin §12.1 Eq. 12.4a & §12.3 Eq. 12.16)

    Evaluates:
        V(r) = V_inf + sum_{j=1}^m Gamma_j * v_ring,j(r) + sum_{l=1}^p Gamma_{w,l} * w_wake,l(r)

    This function serves as the exact derivative callback d(r)/dt = V(r)
    required by Jeremy's streamline tracer (e.g., RK4 integration).

    Parameters:
        r_eval:       (3,) ndarray of coordinates [x, y, z]
        vertices:     (n_v, 3) ndarray of mesh vertices
        faces:        (m, 3) ndarray of panel vertex indices
        te_segments:  list of trailing edge segment dictionaries
        gamma_panels: (m,) ndarray of panel circulations
        gamma_wake:   (p,) ndarray of wake circulations
        V_inf_vec:    (3,) ndarray of freestream velocity vector
        core_radius:  float, vortex core cutoff radius

    Returns:
        V: (3,) ndarray of local velocity vector [vx, vy, vz]
    """
    r_pt = np.asarray(r_eval, dtype=float).reshape(1, 3)
    u_inf = V_inf_vec / np.linalg.norm(V_inf_vec)

    P1, P2, edge_panel, _, k_sides = _extract_ring_edges(vertices, faces)
    edge_gamma = (
        np.repeat(gamma_panels, k_sides)
        if k_sides is not None
        else gamma_panels[edge_panel]
    )

    dx1 = r_pt[:, 0:1] - P1[:, 0]
    dy1 = r_pt[:, 1:2] - P1[:, 1]
    dz1 = r_pt[:, 2:3] - P1[:, 2]
    dx2 = r_pt[:, 0:1] - P2[:, 0]
    dy2 = r_pt[:, 1:2] - P2[:, 1]
    dz2 = r_pt[:, 2:3] - P2[:, 2]

    cr2 = core_radius * core_radius
    r1 = np.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + cr2)
    r2 = np.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + cr2)

    cx = dy1 * dz2 - dz1 * dy2
    cy = dz1 * dx2 - dx1 * dz2
    cz = dx1 * dy2 - dy1 * dx2
    dot12 = dx1 * dx2 + dy1 * dy2 + dz1 * dz2
    den = 4.0 * math.pi * r1 * r2 * (r1 * r2 + dot12)

    ok = np.abs(den) > 1e-12
    fac = np.zeros_like(den)
    fac[ok] = (r1[ok] + r2[ok]) / den[ok]

    V = np.array([
        V_inf_vec[0] + float(((fac * cx) @ edge_gamma)[0]),
        V_inf_vec[1] + float(((fac * cy) @ edge_gamma)[0]),
        V_inf_vec[2] + float(((fac * cz) @ edge_gamma)[0]),
    ], dtype=float)

    if len(te_segments) > 0:
        W = _horseshoe_influence_batch(r_pt, te_segments, u_inf, core_radius=core_radius)
        V += W[0].T @ gamma_wake

    return V


# =============================================================================
# S15. MASTER SYSTEM SETUP & SOLVE WORKFLOW
# =============================================================================

def system_setup_and_solve(vertices, faces, operating_cond, te_segments=None,
                           centroids=None, normals=None, areas=None,
                           pin_panel_idx=0, core_radius=1e-8, n_neighbors=12,
                           verbose=True):
    """Master high-level driver that executes the complete 3D potential flow solve.
    (Hunsaker Sandia Report Complete Workflow; Katz & Plotkin Chapter 12)

    Executes Steps S1 through S13b:
        1. S1:   Compute panel centroids, outward normals, and areas (unless provided).
        2. S5:   Find trailing edge segments (if not explicitly provided).
        3. S6:   Assemble influence matrix [A] (tangency + Kutta + rank pinning).
        4. S7:   Build RHS vector {b}.
        5. S8:   Solve overdetermined linear system for panel and wake circulations.
        6. S9:   Compute centroid induced velocities.
        7. S10:  Compute surface velocity jump via local least-squares gradient.
        8. S11:  Compute surface pressure coefficient distribution Cp.
        9. S12:  Integrate surface pressures for CL, CD_press, CS, and moments.
        10. S13/b: Compute far-field Kutta-Joukowski lift and Trefftz-plane drag.

    Parameters:
        vertices:       (n_v, 3) ndarray of mesh vertices
        faces:          (m, 3) ndarray of panel vertex indices
        operating_cond: dict containing either:
                        - 'V_inf': (3,) velocity vector [Vx, Vy, Vz] (input.json format), OR
                        - 'freestream_velocity', 'alpha_deg', 'beta_deg'
                        Plus optional 'density', 'ref_area'/'S_ref', 'ref_span'/'b_ref',
                        'ref_chord'/'c_ref', 'ref_center'.
        te_segments:    list of trailing edge segment dicts (optional; auto-detected if None)
        centroids:      (m, 3) optional precomputed panel centroids
        normals:        (m, 3) optional precomputed outward unit normals
        areas:          (m,) optional precomputed panel areas
        pin_panel_idx:  int, index of panel pinned to zero circulation (default 0)
        core_radius:    float, vortex core cutoff radius (default 1e-8)
        n_neighbors:    int, neighbor count for local gradient fit (default 12)
        verbose:        bool, print status banners to terminal

    Returns:
        solution: dict containing all computed aerodynamic variables:
                  'centroids': ndarray (m, 3)
                  'normals': ndarray (m, 3)
                  'areas': ndarray (m,)
                  'te_segments': list of p segments
                  'gamma_panels': ndarray (m,)
                  'gamma_wake': ndarray (p,)
                  'V_centroids': ndarray (m, 3)
                  'V_surf': ndarray (m, 3)
                  'Cp': ndarray (m,)
                  'forces': dict from integrate_surface_forces
                  'Lift_KJ': float
                  'CL_KJ': float
                  'Drag_KJ': float
                  'CD_KJ': float
                  'residuals': residual norm
    """
    if verbose:
        print("=" * 80)
        print("  MAE 6500: 3D LIFTING VORTEX CODE WITH TRAILING VORTICES")
        print("  System Setup and Solve Module")
        print("=" * 80)

    # 1. Panel Geometry (S1)
    if centroids is None or normals is None or areas is None:
        centroids, normals, areas = compute_panel_geometry(vertices, faces)
    m = len(faces)
    if verbose:
        print(f"  [S1] Mesh Geometry: {len(vertices)} vertices, {m} surface panels.")

    # Flight operating parameters (supports both input.json 'V_inf' vector and scalar/angle form)
    rho_inf = float(operating_cond.get('density', 1.225))
    if 'V_inf' in operating_cond:
        V_inf_vec = np.asarray(operating_cond['V_inf'], dtype=float)
        V_inf_mag = float(np.linalg.norm(V_inf_vec))
        if V_inf_mag < 1e-14:
            raise ValueError("Freestream velocity vector V_inf cannot be zero.")
        u_inf = V_inf_vec / V_inf_mag
        alpha_rad = math.atan2(V_inf_vec[2], V_inf_vec[0])
        beta_rad = math.asin(max(-1.0, min(1.0, V_inf_vec[1] / V_inf_mag)))
        alpha_deg = math.degrees(alpha_rad)
        beta_deg = math.degrees(beta_rad)
    else:
        V_inf_mag = float(operating_cond.get('freestream_velocity', 10.0))
        alpha_deg = float(operating_cond.get('alpha_deg', 0.0))
        beta_deg = float(operating_cond.get('beta_deg', 0.0))
        alpha_rad = math.radians(alpha_deg)
        beta_rad = math.radians(beta_deg)
        V_inf_vec = np.array([
            V_inf_mag * math.cos(alpha_rad) * math.cos(beta_rad),
            V_inf_mag * math.sin(beta_rad),
            V_inf_mag * math.sin(alpha_rad) * math.cos(beta_rad)
        ])
        u_inf = V_inf_vec / V_inf_mag

    # Reference geometric quantities (supports both S_ref/c_ref/b_ref and ref_area/ref_chord/ref_span)
    x_span = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    y_span = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    S_ref = float(operating_cond.get('S_ref', operating_cond.get('ref_area', x_span * y_span)))
    b_ref = float(operating_cond.get('b_ref', operating_cond.get('ref_span', y_span)))
    c_ref = float(operating_cond.get('c_ref', operating_cond.get('ref_chord', x_span)))
    r_ref = np.asarray(operating_cond.get('ref_center', [0.0, 0.0, 0.0]), dtype=float)

    # 2. Trailing Edge Detection (S5)
    if te_segments is None:
        te_segments = find_trailing_edge_segments(vertices, faces, centroids, normals)
    p = len(te_segments)
    if verbose:
        print(f"  [S5] Trailing Edge: {p} trailing-edge segments identified for Kutta condition.")

    # 3. Assemble Influence Matrix (S6)
    if verbose:
        print(f"  [S6] Assembling ({m + p + 1} x {m + p}) aerodynamic influence matrix [A]...")
    A, V_tensor = build_influence_matrix(vertices, faces, centroids, normals, te_segments,
                                         u_inf, pin_panel_idx=pin_panel_idx, core_radius=core_radius)

    # 4. Assemble RHS Vector (S7)
    b = build_rhs_vector(normals, V_inf_vec, p)

    # 5. Solve for Circulations (S8)
    if verbose:
        print("  [S8] Solving linear circulation system via least squares...")
    gamma_total, residuals = solve_circulation_system(A, b)
    gamma_panels = gamma_total[:m]
    gamma_wake = gamma_total[m:]

    # 6. Centroid Induced Velocities (S9)
    if verbose:
        print("  [S9] Evaluating induced velocities at panel control points...")
    V_centroids = compute_centroid_velocities(vertices, faces, centroids, te_segments,
                                             gamma_panels, gamma_wake, V_inf_vec,
                                             V_tensor=V_tensor, core_radius=core_radius)

    # 7. Surface Velocity Jump (S10)
    if verbose:
        print("  [S10] Applying Appendix C local least-squares velocity jump (-0.5 grad_s Gamma)...")
    V_surf = compute_surface_velocity_jump(centroids, normals, gamma_panels, V_centroids,
                                           n_neighbors=n_neighbors, faces=faces,
                                           te_segments=te_segments)

    # 8. Pressure Coefficients (S11)
    Cp = compute_pressure_coefficients(V_surf, V_inf_mag)

    # 9. Force & Moment Integration (S12)
    forces = integrate_surface_forces(Cp, areas, normals, centroids, alpha_rad, beta_rad,
                                     S_ref, b_ref, c_ref, r_ref=r_ref,
                                     rho_inf=rho_inf, V_inf_mag=V_inf_mag)

    # 10. Trefftz-Plane / Kutta-Joukowski Lift & Induced Drag (S13, S13b)
    Lift_KJ, CL_KJ = kutta_joukowski_lift(gamma_wake, te_segments, V_inf_mag, S_ref, rho_inf=rho_inf)
    Drag_KJ, CD_KJ = trefftz_plane_induced_drag(gamma_wake, te_segments, V_inf_mag, S_ref, rho_inf=rho_inf)

    if verbose:
        print("-" * 80)
        print("  SOLUTION SUMMARY:")
        print(f"    Alpha:                     {alpha_deg:.2f} deg")
        print(f"    Surface Integrated Lift:   CL = {forces['CL']:.4f}  (Lift = {forces['Lift']:.2f} N)")
        print(f"    Trefftz-Plane Wake Lift:   CL = {CL_KJ:.4f}  (Lift = {Lift_KJ:.2f} N)")
        print(f"    Lift Agreement Delta:      abs(CL_press - CL_KJ) = {abs(forces['CL'] - CL_KJ):.5f}")
        print(f"    Pressure Induced Drag:     CD = {forces['CD_press']:.5f}")
        print(f"    Trefftz-Plane Wake Drag:   CD = {CD_KJ:.5f}  (Drag = {Drag_KJ:.2f} N)")
        print(f"    Pitching Moment (Cm_y):    Cm = {forces['Cm_y']:.4f}")
        print("=" * 80)

    return {
        'centroids': centroids,
        'normals': normals,
        'areas': areas,
        'te_segments': te_segments,
        'gamma_panels': gamma_panels,
        'gamma_wake': gamma_wake,
        'V_centroids': V_centroids,
        'V_surf': V_surf,
        'Cp': Cp,
        'forces': forces,
        'Lift_KJ': Lift_KJ,
        'CL_KJ': CL_KJ,
        'Drag_KJ': Drag_KJ,
        'CD_KJ': CD_KJ,
        'residuals': residuals
    }


# =============================================================================
# S16. SRC/ INTEGRATION WRAPPER FOR MESH OBJECT & INPUT.JSON
# =============================================================================

def solve_from_mesh(mesh, input_data, te_segments=None, pin_panel_idx=0,
                    core_radius=1e-8, n_neighbors=12, verbose=True):
    """High-level functional wrapper accepting Jeremy's src.mesh.Mesh object
    and the parsed input.json configuration dictionary.

    Parameters:
        mesh:        Mesh object from src/mesh.py
        input_data:  dict from load_input_json() containing 'mesh' and 'flow'
        te_segments: optional list of trailing-edge segment dicts (auto-detected if None)
        pin_panel_idx: int, pinned circulation panel index (default 0)
        core_radius: float, vortex core cutoff radius
        n_neighbors: int, neighbor count for Appendix C gradient fit
        verbose:     bool, print summary output

    Returns:
        solution: dict from system_setup_and_solve()
    """
    mesh_data = extract_mesh_data(mesh)

    mesh_cfg = input_data.get("mesh", {})
    flow_cfg = input_data.get("flow", {})

    operating_cond = dict(flow_cfg)
    for key in ("S_ref", "c_ref", "b_ref"):
        if key in mesh_cfg:
            operating_cond[key] = mesh_cfg[key]

    return system_setup_and_solve(
        vertices=mesh_data["vertices"],
        faces=mesh_data["faces"],
        operating_cond=operating_cond,
        te_segments=te_segments,
        centroids=mesh_data["centroids"],
        normals=mesh_data["normals"],
        areas=mesh_data["areas"],
        pin_panel_idx=pin_panel_idx,
        core_radius=core_radius,
        n_neighbors=n_neighbors,
        verbose=verbose,
    )


# =============================================================================
# S17. OBJECT-ORIENTED SOLVER WRAPPER (SRC/MAIN.PY INTERFACE)
# =============================================================================

class Solver:
    """Object-oriented solver interface matching src/main.py:
        mySolver = Solver(mesh, wake)
        mySolver.calculate_influence_matrix()
        mySolver.solve_panel_strengths()

    Delegates all numerical work to the verified functional routines (S1-S14).
    """

    def __init__(self, mesh, wake, V_inf=None, S_ref=None, c_ref=None, b_ref=None,
                 r_ref=None, rho_inf=1.225, pin_panel_idx=0, core_radius=1e-8,
                 n_neighbors=12):
        self.mesh = mesh
        self.wake = wake if wake is not None else []
        self.rho_inf = float(rho_inf)
        self.pin_panel_idx = int(pin_panel_idx)
        self.core_radius = float(core_radius)
        self.n_neighbors = int(n_neighbors)

        mesh_data = extract_mesh_data(mesh)
        self.vertices = mesh_data["vertices"]
        self.faces = mesh_data["faces"]
        self.centroids = mesh_data["centroids"]
        self.normals = mesh_data["normals"]
        self.areas = mesh_data["areas"]

        if len(self.wake) > 0:
            self.te_segments = match_wake_to_panels(self.vertices, self.faces, self.wake)
        else:
            self.te_segments = []

        # Default freestream vector from wake direction if V_inf is not explicitly passed
        if V_inf is not None:
            self._set_freestream(V_inf)
        elif len(self.wake) > 0:
            self._set_freestream(self.wake[0].direction)
        else:
            self._set_freestream([1.0, 0.0, 0.0])

        x_span = float(np.max(self.vertices[:, 0]) - np.min(self.vertices[:, 0]))
        y_span = float(np.max(self.vertices[:, 1]) - np.min(self.vertices[:, 1]))
        self.S_ref = float(S_ref) if S_ref is not None else x_span * y_span
        self.b_ref = float(b_ref) if b_ref is not None else y_span
        self.c_ref = float(c_ref) if c_ref is not None else x_span
        self.r_ref = (
            np.asarray(r_ref, dtype=float)
            if r_ref is not None
            else np.zeros(3, dtype=float)
        )

        self.influence_matrix = None
        self.A = None
        self.V_tensor = None
        self.b = None
        self.gamma_panels = None
        self.gamma_wake = None
        self.panel_strengths = None
        self.V_centroids = None
        self.V_surf = None
        self.Cp = None
        self.forces = None
        self.Lift_KJ = None
        self.CL_KJ = None
        self.Drag_KJ = None
        self.CD_KJ = None
        self.residuals = None

    def _set_freestream(self, V_inf):
        V_inf_vec = np.asarray(V_inf, dtype=float)
        V_inf_mag = float(np.linalg.norm(V_inf_vec))
        if V_inf_mag < 1e-14:
            raise ValueError("Freestream velocity vector V_inf cannot be zero.")
        self.V_inf_vec = V_inf_vec
        self.V_inf_mag = V_inf_mag
        self.u_inf = V_inf_vec / V_inf_mag
        self.alpha_rad = math.atan2(V_inf_vec[2], V_inf_vec[0])
        self.beta_rad = math.asin(max(-1.0, min(1.0, V_inf_vec[1] / V_inf_mag)))

    def calculate_influence_matrix(self):
        """Assemble aerodynamic influence coefficient matrix [A] and velocity tensor (S6)."""
        self.A, self.V_tensor = build_influence_matrix(
            self.vertices,
            self.faces,
            self.centroids,
            self.normals,
            self.te_segments,
            self.u_inf,
            pin_panel_idx=self.pin_panel_idx,
            core_radius=self.core_radius,
        )
        self.influence_matrix = self.A
        return self.influence_matrix

    def solve_panel_strengths(self, V_inf=None, verbose=True):
        """Solve for panel & wake circulations and evaluate surface Cp and forces (S7-S13)."""
        if V_inf is not None:
            self._set_freestream(V_inf)
        if self.influence_matrix is None:
            self.calculate_influence_matrix()

        m = len(self.faces)
        p = len(self.te_segments)
        self.b = build_rhs_vector(self.normals, self.V_inf_vec, p)

        gamma_total, self.residuals = solve_circulation_system(self.influence_matrix, self.b)
        self.gamma_panels = gamma_total[:m]
        self.gamma_wake = gamma_total[m:]
        self.panel_strengths = self.gamma_panels

        self.V_centroids = compute_centroid_velocities(
            self.vertices,
            self.faces,
            self.centroids,
            self.te_segments,
            self.gamma_panels,
            self.gamma_wake,
            self.V_inf_vec,
            V_tensor=self.V_tensor,
            core_radius=self.core_radius,
        )
        self.V_surf = compute_surface_velocity_jump(
            self.centroids,
            self.normals,
            self.gamma_panels,
            self.V_centroids,
            n_neighbors=self.n_neighbors,
            faces=self.faces,
            te_segments=self.te_segments,
        )
        self.Cp = compute_pressure_coefficients(self.V_surf, self.V_inf_mag)
        self.forces = integrate_surface_forces(
            self.Cp,
            self.areas,
            self.normals,
            self.centroids,
            self.alpha_rad,
            self.beta_rad,
            self.S_ref,
            self.b_ref,
            self.c_ref,
            r_ref=self.r_ref,
            rho_inf=self.rho_inf,
            V_inf_mag=self.V_inf_mag,
        )
        self.Lift_KJ, self.CL_KJ = kutta_joukowski_lift(
            self.gamma_wake,
            self.te_segments,
            self.V_inf_mag,
            self.S_ref,
            rho_inf=self.rho_inf,
        )
        self.Drag_KJ, self.CD_KJ = trefftz_plane_induced_drag(
            self.gamma_wake,
            self.te_segments,
            self.V_inf_mag,
            self.S_ref,
            rho_inf=self.rho_inf,
        )

        if verbose:
            print("-" * 80)
            print("  SOLVER SUMMARY:")
            print(f"    Alpha:                     {math.degrees(self.alpha_rad):.2f} deg")
            print(f"    Surface Integrated Lift:   CL = {self.forces['CL']:.4f}")
            print(f"    Trefftz-Plane Wake Lift:   CL = {self.CL_KJ:.4f}")
            print(f"    Pressure Induced Drag:     CD = {self.forces['CD_press']:.5f}")
            print(f"    Trefftz-Plane Wake Drag:   CD = {self.CD_KJ:.5f}")
            print(f"    Pitching Moment (Cm_y):    Cm = {self.forces['Cm_y']:.4f}")
            print("-" * 80)

        return self.panel_strengths

    def evaluate_velocity(self, r_eval):
        """Evaluate total potential flow velocity at an arbitrary 3D point (S14)."""
        return evaluate_flow_field_velocity(
            np.asarray(r_eval, dtype=float),
            self.vertices,
            self.faces,
            self.te_segments,
            self.gamma_panels,
            self.gamma_wake,
            self.V_inf_vec,
            core_radius=self.core_radius,
        )

    def get_solution_dict(self):
        """Return full solution dictionary for post-processing and export."""
        return {
            "centroids": self.centroids,
            "normals": self.normals,
            "areas": self.areas,
            "te_segments": self.te_segments,
            "gamma_panels": self.gamma_panels,
            "gamma_wake": self.gamma_wake,
            "V_centroids": self.V_centroids,
            "V_surf": self.V_surf,
            "Cp": self.Cp,
            "forces": self.forces,
            "Lift_KJ": self.Lift_KJ,
            "CL_KJ": self.CL_KJ,
            "Drag_KJ": self.Drag_KJ,
            "CD_KJ": self.CD_KJ,
            "residuals": self.residuals,
        }


# =============================================================================
# UTILITY / DEMO: ASCII STL PARSER & STANDALONE VERIFICATION
# =============================================================================

def load_stl_ascii(filename):
    """Simple functional parser to load vertices and triangular faces from an ASCII .stl file.
    (Can be used directly or replaced by Jeremy's mesh import module).

    Parameters:
        filename: str, path to ASCII .stl file

    Returns:
        vertices: (n_v, 3) ndarray of unique vertex coordinates
        faces:    (n_f, 3) ndarray of vertex indices
    """
    raw_verts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0] == "vertex":
                raw_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])

    raw_verts = np.asarray(raw_verts, dtype=float)
    if len(raw_verts) % 3 != 0:
        raise ValueError(f"Total vertices in STL ({len(raw_verts)}) is not divisible by 3.")

    num_triangles = len(raw_verts) // 3

    # Deduplicate shared vertices to reconstruct topology
    unique_verts, inverse_indices = np.unique(
        np.round(raw_verts, decimals=8), axis=0, return_inverse=True
    )
    faces = inverse_indices.reshape(num_triangles, 3)

    return unique_verts, faces


if __name__ == "__main__":
    import os
    print("Running standalone verification of system_solver.py...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    stl_candidate = os.path.join(base_dir, "wing_geometry.stl")
    if not os.path.exists(stl_candidate):
        stl_candidate = os.path.join("Projects", "Group Project", "wing_geometry.stl")

    if os.path.exists(stl_candidate):
        print(f"Loading benchmark CAD mesh: {stl_candidate}")
        v_test, f_test = load_stl_ascii(stl_candidate)
    else:
        raise FileNotFoundError(f"Expected benchmark mesh at {stl_candidate}")

    # Test 1: Standard scalar/angle operating condition
    op_test = {
        'freestream_velocity': 10.0,
        'alpha_deg': 2.0,
        'beta_deg': 0.0,
        'ref_area': 10.0,
        'ref_span': 10.0,
        'ref_chord': 1.0,
        'ref_center': [0.25, 0.0, 0.0]
    }
    sol = system_setup_and_solve(v_test, f_test, op_test, verbose=True)

    # Test 2: Verify src/mesh.py + input.json adapter compatibility
    input_path = os.path.join(base_dir, "input.json")
    if os.path.exists(input_path):
        cfg = load_input_json(input_path)
        print(f"\nLoaded input.json config: flow={cfg['flow']}, mesh={cfg['mesh']}")

    print("Verification complete! System setup and solve executed successfully.")

