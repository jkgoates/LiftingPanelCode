import json

from mesh import Mesh
from solver import Solver


def main():

    # Welcome message
    print("Lifting Panel Code")
    print("MAE 6500 Fall 2026")
    print("Authors: Mason Anderson, Jeremy Goates, and Nguyen Phung")

    print("Starting main function...")

    # Load json input file
    with open("input.json", "r") as f:
        input_data = json.load(f)

    # Load mesh from input file
    mesh = Mesh(input_data["mesh_file"])
    print("Mesh loaded with {0} panels and {1} vertices.".format(mesh.N, mesh.vertices.shape[0]))

    # Find kutta edges
    kutta_edges = mesh.find_kutta_edges()
    print("Found {0} kutta edges.".format(len(kutta_edges)))

    # Initialize wake
    wake = mesh.initialize_wake(kutta_edges)
    print("Wake initialized with {0} vortex filaments.".format(len(wake)))

    # Initialize solver
    mySolver = Solver(mesh, wake)

    # Calculate Influence Matrix
    mySolver.calculate_influence_matrix()

    # Solve for panel strengths
    mySolver.solve_panel_strengths()

    # So on and so forth


if __name__ == "__main__":
    main()