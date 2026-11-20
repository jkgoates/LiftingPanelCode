import os
import json
import numpy as np

from mesh import Mesh
from solver import Solver
import postprocess


def main():

    # Welcome message
    print("Lifting Panel Code")
    print("MAE 6500 Fall 2026")
    print("Authors: Mason Anderson, Jeremy Goates, and Nguyen Phung")

    print("Starting main function...")

    # Load json input file
    with open("input.json", "r") as f:
        input_data = json.load(f)

    mesh_cfg = input_data["mesh"]
    flow_cfg = input_data["flow"]
    output_cfg = input_data.get("output", {})

    # Load mesh from input file
    mesh = Mesh(mesh_cfg["filename"])
    print("Mesh loaded with {0} panels and {1} vertices.".format(mesh.N, mesh.vertices.shape[0]))

    flow_direction = flow_cfg["V_inf"]

    # Find kutta edges
    kutta_edges = mesh.find_kutta_edges(flow_direction)
    print("Found {0} kutta edges.".format(len(kutta_edges)))

    # Initialize wake
    wake = mesh.initialize_wake(kutta_edges, flow_direction)
    print("Wake initialized with {0} horseshoe vortices.".format(len(wake)))

    # Initialize solver with full freestream vector and reference geometry from input.json
    mySolver = Solver(
        mesh,
        wake,
        V_inf=flow_direction,
        S_ref=mesh_cfg.get("S_ref"),
        c_ref=mesh_cfg.get("c_ref"),
        b_ref=mesh_cfg.get("b_ref"),
        r_ref=mesh_cfg.get("r_ref"),
        rho_inf=flow_cfg.get("rho_inf", 1.225),
    )

    # Calculate Influence Matrix [A] and velocity influence tensor
    mySolver.calculate_influence_matrix()

    # Solve for panel & wake circulations, surface Cp, surface forces, and Trefftz-plane lift and drag
    mySolver.solve_panel_strengths(verbose=True)
    solution = mySolver.get_solution_dict()

    # Evaluate off-body velocities (e.g. for streamlines / off_body_velocity_points.csv) if provided
    off_body_file = output_cfg.get("off_body_velocity_points")
    if off_body_file and os.path.exists(off_body_file):
        pts = np.loadtxt(off_body_file, delimiter=",", ndmin=2)
        off_body_velocities = mySolver.evaluate_velocities(pts)
        solution["off_body_points"] = pts
        solution["off_body_velocities"] = off_body_velocities
        print("Evaluated velocity at {0} off-body points.".format(len(pts)))

    # Hand off `solution` (Cp, V_surf, surface forces, Trefftz-plane lift and drag, gamma_panels, gamma_wake,
    # gamma_shed)
    # to post-processing/export (report.json, body.vtk, wake.vtk) and streamline plots.
    postprocess.export_solution(input_data, mySolver, solution)

    # Grid convergence study over the meshes listed under "convergence" in input.json
    postprocess.run_convergence(input_data)


if __name__ == "__main__":
    main()
