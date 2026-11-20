# Runs the wing at one angle of attack on several meshes (Nguyen Phung).
# Each mesh is solved once with the same steps as main.py, its files are written to output/<mesh>/,
# and the convergence table, summary and plot are made from all of them.
#
# Run from the LiftingPanelCode folder:
#     py -3 src/run_cases.py 5 coarse medium fine
# The first number is the angle of attack in degrees; the names pick meshes/wing_RA10_<name>.stl.
# With no arguments it runs 5 degrees on coarse, medium and fine.
# The streamline field (off_body.vtk) is switched on and off with "enabled" under "off_body_grid" in input.json.
#
# Uses the group's Solver from solver.py. The convergence files are rewritten after every mesh,
# so the finished meshes are kept even if a later, larger mesh is stopped.

import os
import sys
import copy
import json
import time

import numpy as np

from mesh import Mesh
from solver import Solver
import postprocess

# every file goes under this folder
OUTPUT_FOLDER = "output"


def run_one_mesh(name, base):
    # solve one mesh and write its files; returns its row of the convergence table
    start = time.time()
    data = copy.deepcopy(base)
    mesh_file = "meshes/wing_RA10_{0}.stl".format(name)
    data["mesh"]["filename"] = mesh_file

    # each mesh writes into its own folder, e.g. output/coarse/body.vtk
    for key in ["report_file", "body_file", "wake_file", "off_body_file", "off_body_velocity_file"]:
        if key in data["output"]:
            data["output"][key] = data["output"][key].replace("output/", "{0}/{1}/".format(OUTPUT_FOLDER, name))

    # same steps as main.py
    mesh = Mesh(mesh_file)
    V_inf = data["flow"]["V_inf"]
    wake = mesh.initialize_wake(mesh.find_kutta_edges(V_inf), V_inf)
    solver = Solver(mesh, wake, V_inf=V_inf, S_ref=data["mesh"].get("S_ref"), c_ref=data["mesh"].get("c_ref"),
                    b_ref=data["mesh"].get("b_ref"))
    solver.calculate_influence_matrix()
    solver.solve_panel_strengths(verbose=False)
    solution = solver.get_solution_dict()
    solve_time = time.time() - start

    # velocity at the points listed in the off-body points file, if there is one
    points_file = data["output"].get("off_body_velocity_points")
    if points_file and os.path.exists(points_file):
        points = np.loadtxt(points_file, delimiter=",", ndmin=2)
        velocity = np.zeros((len(points), 3))
        for i in range(len(points)):
            velocity[i] = solver.evaluate_velocity(points[i])
        solution["off_body_points"] = points
        solution["off_body_velocities"] = velocity

    postprocess.export_solution(data, solver, solution)
    total_time = time.time() - start

    row = {"mesh": os.path.basename(mesh_file), "n_panels": int(mesh.N)}
    row.update(postprocess.coefficient_row(solver))
    row["solve_time[s]"] = solve_time
    print("{0}: {1} panels, CL surface/Trefftz = {2:.5f}/{3:.5f}, CD surface/Trefftz = {4:.5f}/{5:.5f}, "
          "solve {6:.0f} s, exports {7:.0f} s".format(name, row["n_panels"], row["CL_surface"], row["CL_trefftz"],
                                                     row["CD_surface"], row["CD_trefftz"], solve_time,
                                                     total_time - solve_time), flush=True)
    return row


def main():
    # angle of attack and mesh names from the command line
    if len(sys.argv) > 1:
        alpha_deg = float(sys.argv[1])
    else:
        alpha_deg = 5.0
    names = sys.argv[2:]
    if len(names) == 0:
        names = ["coarse", "medium", "fine"]

    with open("input.json", "r") as f:
        base = json.load(f)

    # the freestream comes from the angle of attack, not from input.json
    alpha = np.radians(alpha_deg)
    base["flow"]["V_inf"] = [float(np.cos(alpha)), 0.0, float(np.sin(alpha))]
    print("Angle of attack {0} deg, V_inf = {1}".format(alpha_deg, base["flow"]["V_inf"]))

    # convergence file names from input.json, moved under OUTPUT_FOLDER
    output = base.get("output", {})
    table_file = output.get("convergence_file", "output/convergence.csv").replace("output/", OUTPUT_FOLDER + "/")
    figure_file = output.get("convergence_figure", "output/convergence.png").replace("output/", OUTPUT_FOLDER + "/")
    summary_file = output.get("convergence_summary_file",
                              "output/convergence_summary.csv").replace("output/", OUTPUT_FOLDER + "/")

    rows = []
    for name in names:
        rows.append(run_one_mesh(name, base))
        # rewrite the convergence files after every mesh (needs at least two meshes)
        if len(rows) > 1:
            rows = postprocess.convergence_table(rows)
            postprocess.write_convergence(rows, table_file, figure_file, summary_file)


if __name__ == "__main__":
    main()
