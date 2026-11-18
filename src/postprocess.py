# Post-processing and export (Nguyen Phung): writes the report, the VTK files and the off-body velocity file
# from the solver's results, and runs the grid convergence study.

import os
import csv
import json
import math
import time

import numpy as np

from mesh import Mesh
from solver import Solver


def make_folder_for(path):
    # create the folder a file goes in, if it is not there yet
    folder = os.path.dirname(path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)


# ---------- report ----------

def write_report(path, mesh_file, solver, solution):
    # the JSON report: inputs, mesh size, coefficients and forces
    forces = solution["forces"]
    report = {"mesh_file": str(mesh_file),
              "n_panels": int(len(solver.faces)),
              "n_vertices": int(len(solver.vertices)),
              "n_wake_horseshoes": int(len(solution["te_segments"])),
              "V_inf": [float(v) for v in solver.V_inf_vec],
              "alpha[deg]": float(np.degrees(solver.alpha_rad)),
              "beta[deg]": float(np.degrees(solver.beta_rad)),
              "S_ref": float(solver.S_ref),
              "b_ref": float(solver.b_ref),
              "c_ref": float(solver.c_ref),
              "coefficients": {"CL": float(forces["CL"]),
                               "CL_KJ": float(solution["CL_KJ"]),
                               "CD_press": float(forces["CD_press"]),
                               "CS": float(forces["CS"]),
                               "Cm_x": float(forces["Cm_x"]),
                               "Cm_y": float(forces["Cm_y"]),
                               "Cm_z": float(forces["Cm_z"])},
              "forces": {"Lift": float(forces["Lift"]),
                         "Lift_KJ": float(solution["Lift_KJ"]),
                         "Drag": float(forces["Drag"]),
                         "SideForce": float(forces["SideForce"]),
                         "F_body": [float(v) for v in forces["F_body"]],
                         "M_body": [float(v) for v in forces["M_body"]]}}
    make_folder_for(path)
    with open(path, "w") as f:
        json.dump(report, f, indent=4)
    print("Wrote", path)


# ---------- VTK files ----------

def write_vtk(path, points, cells, scalars, vectors, title):
    # scalars and vectors hold one value per cell
    size = 0
    for c in cells:
        size += len(c) + 1
    make_folder_for(path)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(title + "\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write("POINTS {0} float\n".format(len(points)))
        for p in points:
            f.write("{0:.8e} {1:.8e} {2:.8e}\n".format(p[0], p[1], p[2]))
        f.write("POLYGONS {0} {1}\n".format(len(cells), size))
        for c in cells:
            f.write(str(len(c)))
            for v in c:
                f.write(" " + str(int(v)))
            f.write("\n")
        f.write("CELL_DATA {0}\n".format(len(cells)))
        for name in scalars:
            f.write("SCALARS {0} float 1\n".format(name))
            f.write("LOOKUP_TABLE default\n")
            for value in scalars[name]:
                f.write("{0:.8e}\n".format(value))
        for name in vectors:
            f.write("VECTORS {0} float\n".format(name))
            for v in vectors[name]:
                f.write("{0:.8e} {1:.8e} {2:.8e}\n".format(v[0], v[1], v[2]))
    print("Wrote", path)


def write_body_vtk(path, solver, solution):
    # the body surface with the circulation, Cp, surface velocity and normal of every panel
    scalars = {"gamma": solution["gamma_panels"], "Cp": solution["Cp"]}
    vectors = {"V_surf": solution["V_surf"], "normal": solution["normals"]}
    write_vtk(path, solver.vertices, solver.faces, scalars, vectors, "body surface")


def write_wake_vtk(path, solver, solution, length):
    # each wake horseshoe is drawn as one 4 corner cell: its trailingedge segment swept downstream by "length".
    #  "length" is only how much of them is shown
    points = []
    cells = []
    segments = solution["te_segments"]
    for s in range(len(segments)):
        p1 = segments[s]["p1"]
        p2 = segments[s]["p2"]
        points.append(p1)
        points.append(p1 + length * solver.u_inf)
        points.append(p2 + length * solver.u_inf)
        points.append(p2)
        cells.append([4 * s, 4 * s + 1, 4 * s + 2, 4 * s + 3])
    write_vtk(path, points, cells, {"gamma_wake": solution["gamma_wake"]}, {}, "wake")


# ---------- off-body velocities ----------

def write_off_body_csv(path, points, velocity):
    # the offbody points with the velocity the solver found at each one
    make_folder_for(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "u", "v", "w"])
        for i in range(len(points)):
            writer.writerow([points[i][0], points[i][1], points[i][2], velocity[i][0], velocity[i][1], velocity[i][2]])
    print("Wrote", path)


def grid_points(grid):
    # a regular box of points from the "off_body_grid" part of input.json.
    # returns the first corner, the step in x, y, z, the point counts, and the points with x changing fastest
    dims = [int(grid["n_x"]), int(grid["n_y"]), int(grid["n_z"])]
    origin = [float(grid["x_min"]), float(grid["y_min"]), float(grid["z_min"])]
    top = [float(grid["x_max"]), float(grid["y_max"]), float(grid["z_max"])]
    spacing = [0.0, 0.0, 0.0]
    for a in range(3):
        if dims[a] > 1:
            spacing[a] = (top[a] - origin[a]) / (dims[a] - 1)
    points = []
    for k in range(dims[2]):
        for j in range(dims[1]):
            for i in range(dims[0]):
                points.append([origin[0] + i * spacing[0], origin[1] + j * spacing[1], origin[2] + k * spacing[2]])
    return origin, spacing, dims, np.array(points)


def write_field_vtk(path, origin, spacing, dims, velocity):
    # the velocity on a regular box of points, as a VTK file ParaView can draw streamlines from.
    # velocity: one row per point, in the same order as grid_points gives them
    make_folder_for(path)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("off-body velocity field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write("DIMENSIONS {0} {1} {2}\n".format(dims[0], dims[1], dims[2]))
        f.write("ORIGIN {0:.8e} {1:.8e} {2:.8e}\n".format(origin[0], origin[1], origin[2]))
        # a direction with only one point still needs a non-zero spacing in the file
        f.write("SPACING {0:.8e} {1:.8e} {2:.8e}\n".format(spacing[0] or 1.0, spacing[1] or 1.0, spacing[2] or 1.0))
        f.write("POINT_DATA {0}\n".format(len(velocity)))
        f.write("VECTORS velocity float\n")
        for v in velocity:
            f.write("{0:.8e} {1:.8e} {2:.8e}\n".format(v[0], v[1], v[2]))
        f.write("SCALARS speed float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in velocity:
            f.write("{0:.8e}\n".format(np.linalg.norm(v)))
    print("Wrote", path)


def export_velocity_field(path, grid, solver):
    # ask the solver for the velocity at every grid point and write the field file.
    # points inside the wing are not meaningful flow; start streamlines outside it
    origin, spacing, dims, points = grid_points(grid)
    print("Evaluating velocity at {0} grid points...".format(len(points)))
    velocity = np.zeros((len(points), 3))
    for i in range(len(points)):
        velocity[i] = solver.evaluate_velocity(points[i])
    write_field_vtk(path, origin, spacing, dims, velocity)


def export_solution(input_data, solver, solution):
    # write every output file for one solved case. file names come from the "output" part of input.json
    output = input_data.get("output", {})
    # the streamline field is the slowest export; "enabled": false in "off_body_grid" skips it
    grid = input_data.get("off_body_grid")
    if grid is not None and grid.get("enabled", True):
        export_velocity_field(output.get("off_body_file", "output/off_body.vtk"), grid, solver)
    elif grid is not None:
        print("Streamline field skipped (off_body_grid enabled = false)")
    write_report(output.get("report_file", "output/report.json"), input_data["mesh"]["filename"], solver, solution)
    write_body_vtk(output.get("body_file", "output/body.vtk"), solver, solution)
    write_wake_vtk(output.get("wake_file", "output/wake.vtk"), solver, solution, output.get("wake_length_shown", 5.0))
    if "off_body_velocities" in solution:
        write_off_body_csv(output.get("off_body_velocity_file", "output/off_body_velocity.csv"),
                           solution["off_body_points"], solution["off_body_velocities"])


# ---------- grid convergence ----------

def solve_for_convergence(mesh_file, input_data):
    # solve one mesh with the same steps as main.py and return its row of the convergence table
    start = time.time()
    V_inf = input_data["flow"]["V_inf"]
    mesh_cfg = input_data["mesh"]
    mesh = Mesh(mesh_file)
    wake = mesh.initialize_wake(mesh.find_kutta_edges(V_inf), V_inf)
    solver = Solver(mesh, wake, V_inf=V_inf, S_ref=mesh_cfg.get("S_ref"), c_ref=mesh_cfg.get("c_ref"),
                    b_ref=mesh_cfg.get("b_ref"))
    solver.calculate_influence_matrix()
    solver.solve_panel_strengths(verbose=False)
    return {"mesh": os.path.basename(str(mesh_file)),
            "n_panels": int(mesh.N),
            "CL": float(solver.forces["CL"]),
            "CL_KJ": float(solver.CL_KJ),
            "CD_press": float(solver.forces["CD_press"]),
            "solve_time[s]": time.time() - start}


def convergence_table(rows):
    # rows: one dictionary per mesh with "mesh", "n_panels", "CL", "CL_KJ", "CD_press".
    # sorts coarsest first and adds the percent change of each from the finest mesh
    rows = sorted(rows, key=lambda r: r["n_panels"])
    finest = rows[-1]
    for r in rows:
        for name in ["CL", "CL_KJ", "CD_press"]:
            if finest[name] != 0.0:
                r[name + "_change[%]"] = 100.0 * (r[name] - finest[name]) / abs(finest[name])
            else:
                r[name + "_change[%]"] = 0.0
    return rows


def order_of_convergence(n_panels, values):
    # order of convergence, Richardson extrapolation and GCI from three meshes (docs Secs. 6.3 - 6.6).
    # n_panels, values: three numbers each, in any order. Mesh 1 is the finest, mesh 3 the coarsest
    pairs = sorted(zip(n_panels, values), key=lambda pair: pair[0], reverse=True)
    N1, f1 = pairs[0]
    N2, f2 = pairs[1]
    N3, f3 = pairs[2]

    # refinement ratios from h ~ 1/sqrt(N)
    r21 = math.sqrt(N1 / N2)
    r32 = math.sqrt(N2 / N3)
    e21 = f2 - f1
    e32 = f3 - f2
    result = {"r21": r21, "r32": r32, "R": None, "behaviour": "", "p": None, "extrapolated": None,
              "fine_value": f1, "e21[%]": None, "GCI_fine[%]": None}
    if f1 != 0.0:
        result["e21[%]"] = 100.0 * abs(e21 / f1)

    # no change between the meshes at all
    if e21 == 0.0 and e32 == 0.0:
        result["behaviour"] = "no change"
        result["extrapolated"] = f1
        result["GCI_fine[%]"] = 0.0
        return result
    if e32 == 0.0:
        result["behaviour"] = "diverging"
        return result

    # convergence ratio R = e21 / e32: 0 < R < 1 monotonic, R < 0 oscillating, R >= 1 diverging
    R = e21 / e32
    result["R"] = R
    if R < 0.0:
        result["behaviour"] = "oscillating"
        return result
    # R within round-off of 1 means equal steps (no convergence); without the 1e-9 it would give p = 0
    if R >= 1.0 - 1e-9:
        result["behaviour"] = "diverging"
        return result
    result["behaviour"] = "monotonic"

    # order p by fixed-point iteration (docs Sec. 6.4); s = 1 for monotonic convergence
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)
    for k in range(200):
        q = math.log((r21**p - 1.0) / (r32**p - 1.0))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-12:
            p = p_new
            break
        p = p_new
    result["p"] = p

    # Richardson extrapolation (Sec. 6.5) and Grid Convergence Index with a safety factor of 1.25 (Sec. 6.6)
    result["extrapolated"] = f1 + (f1 - f2) / (r21**p - 1.0)
    if f1 != 0.0:
        result["GCI_fine[%]"] = 100.0 * 1.25 * abs((f1 - f2) / f1) / (r21**p - 1.0)
    return result


def convergence_summary(rows):
    # order, extrapolation and GCI of CL, CL_KJ and CD_press from the three finest meshes in the table.
    # returns one dictionary per quantity, or an empty list with fewer than three meshes
    if len(rows) < 3:
        return []
    finest = sorted(rows, key=lambda r: r["n_panels"])[-3:]
    summary = []
    for name in ["CL", "CL_KJ", "CD_press"]:
        result = order_of_convergence([r["n_panels"] for r in finest], [r[name] for r in finest])
        result["quantity"] = name
        result["meshes"] = ", ".join(r["mesh"] for r in finest)
        summary.append(result)
    return summary


def print_convergence_summary(summary):
    # the summary as a small table in the terminal
    print("Grid convergence from the three finest meshes:")
    print("  {0:<9} {1:<12} {2:>6} {3:>12} {4:>12} {5:>8}".format("quantity", "behaviour", "p", "fine mesh",
                                                                  "extrapolated", "GCI[%]"))
    for s in summary:
        p = "" if s["p"] is None else "{0:.2f}".format(s["p"])
        ext = "" if s["extrapolated"] is None else "{0:.5f}".format(s["extrapolated"])
        gci = "" if s["GCI_fine[%]"] is None else "{0:.2f}".format(s["GCI_fine[%]"])
        print("  {0:<9} {1:<12} {2:>6} {3:>12.5f} {4:>12} {5:>8}".format(s["quantity"], s["behaviour"], p,
                                                                         s["fine_value"], ext, gci))


def write_convergence(rows, csv_path, figure_path, summary_path=None):
    # the convergence table as a CSV file, the order/GCI summary as a second CSV file (three or more
    # meshes), and a plot against panel count
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["mesh", "n_panels", "CL", "CL_KJ", "CD_press", "CL_change[%]", "CL_KJ_change[%]", "CD_press_change[%]"]
    # solve time is a column only when every mesh has one
    has_time = all("solve_time[s]" in r for r in rows)
    if has_time:
        names.append("solve_time[s]")
    make_folder_for(csv_path)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for r in rows:
            writer.writerow([r[n] for n in names])
    print("Wrote", csv_path)

    summary = convergence_summary(rows)
    if summary_path is not None and len(summary) > 0:
        columns = ["quantity", "meshes", "r21", "r32", "R", "behaviour", "p", "fine_value", "extrapolated",
                   "e21[%]", "GCI_fine[%]"]
        make_folder_for(summary_path)
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for s in summary:
                # blank cells where a value does not apply (e.g. no order for oscillating results)
                writer.writerow(["" if s[c] is None else s[c] for c in columns])
        print("Wrote", summary_path)
    if len(summary) > 0:
        print_convergence_summary(summary)

    # plot: CL, CD and (if known) solve time against panel count
    n = [r["n_panels"] for r in rows]
    n_plots = 3 if has_time else 2
    fig, ax = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 3.6))
    ax[0].plot(n, [r["CL"] for r in rows], "o-", label="surface pressure")
    ax[0].plot(n, [r["CL_KJ"] for r in rows], "s--", label="Kutta-Joukowski")
    ax[0].set_ylabel("CL")
    ax[1].plot(n, [r["CD_press"] for r in rows], "o-", label="surface pressure")
    ax[1].set_ylabel("CD (pressure)")
    # extrapolated values as dotted lines, for results that converge monotonically
    for s in summary:
        if s["extrapolated"] is None:
            continue
        a = ax[1] if s["quantity"] == "CD_press" else ax[0]
        a.axhline(s["extrapolated"], color="gray", linestyle=":", linewidth=1)
        a.annotate("{0} extrapolated {1:.4f}".format(s["quantity"], s["extrapolated"]), (n[0], s["extrapolated"]),
                   fontsize=7, color="gray", va="bottom")
    ax[0].legend(fontsize=8)
    for a in ax[:2]:
        a.set_xlabel("number of panels")
        a.set_xscale("log")
        a.grid(True)
    if has_time:
        ax[2].plot(n, [r["solve_time[s]"] for r in rows], "o-")
        ax[2].set_xlabel("number of panels")
        ax[2].set_ylabel("solve time [s]")
        ax[2].set_xscale("log")
        ax[2].set_yscale("log")
        ax[2].grid(True, which="both")
    fig.suptitle("Grid convergence")
    fig.tight_layout()
    make_folder_for(figure_path)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print("Wrote", figure_path)


def run_convergence(input_data):
    # solve every mesh listed under "convergence" in input.json and write the table and the plot.
    # the solve time grows fast with panel count
    settings = input_data.get("convergence", {})
    mesh_files = settings.get("mesh_files", [])
    if len(mesh_files) == 0:
        return []
    rows = []
    for mesh_file in mesh_files:
        print("Convergence study: solving", mesh_file)
        rows.append(solve_for_convergence(mesh_file, input_data))
    rows = convergence_table(rows)
    output = input_data.get("output", {})
    write_convergence(rows, output.get("convergence_file", "output/convergence.csv"),
                      output.get("convergence_figure", "output/convergence.png"),
                      output.get("convergence_summary_file", "output/convergence_summary.csv"))
    return rows
