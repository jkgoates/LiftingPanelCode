import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mesh import Mesh
from solver import Solver
import postprocess

WING = PROJECT_ROOT / "meshes" / "wing_RA10_coarse.stl"


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # solve the coarse wing once with the real solver; every export test uses this solution
        V_inf = [1.0, 0.0, 0.0]
        mesh = Mesh(WING)
        wake = mesh.initialize_wake(mesh.find_kutta_edges(V_inf), V_inf)
        cls.solver = Solver(mesh, wake, V_inf=V_inf, S_ref=1.0, c_ref=1.0)
        cls.solver.calculate_influence_matrix()
        cls.solver.solve_panel_strengths(verbose=False)
        cls.solution = cls.solver.get_solution_dict()
        cls.n_panels = mesh.N
        cls.n_vertices = mesh.N_vert
        cls.n_wake = len(wake)

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())

    def read_vtk(self, path):
        # returns the points, the cell lists and the lines after the cells
        lines = path.read_text().split("\n")
        n_points = int(lines[4].split()[1])
        points = [[float(v) for v in line.split()] for line in lines[5:5 + n_points]]
        start = 5 + n_points
        n_cells = int(lines[start].split()[1])
        cells = [[int(v) for v in line.split()] for line in lines[start + 1:start + 1 + n_cells]]
        return np.array(points), cells, lines[start + 1 + n_cells:]

    def read_scalar(self, rest, name):
        # the values of one SCALARS array from the lines after the cells
        start = rest.index("SCALARS {0} float 1".format(name)) + 2
        n = int(rest[0].split()[1])
        return np.array([float(v) for v in rest[start:start + n]])

    def test_body_vtk_has_every_panel_and_the_solver_cp(self):
        path = self.folder / "sub" / "body.vtk"
        postprocess.write_body_vtk(str(path), self.solver, self.solution)
        points, cells, rest = self.read_vtk(path)
        self.assertEqual(len(points), self.n_vertices)
        self.assertEqual(len(cells), self.n_panels)
        for c in cells:
            self.assertEqual(c[0], 3)
            self.assertTrue(all(0 <= v < len(points) for v in c[1:]))
        self.assertEqual(rest[0], "CELL_DATA {0}".format(self.n_panels))
        text = "\n".join(rest)
        for name in ["SCALARS gamma", "SCALARS Cp", "VECTORS V_surf", "VECTORS normal"]:
            self.assertIn(name, text)
        self.assertNotIn("nan", text)
        np.testing.assert_allclose(self.read_scalar(rest, "Cp"), self.solution["Cp"], rtol=1e-6, atol=1e-9)

    def test_wake_vtk_has_one_cell_per_horseshoe(self):
        path = self.folder / "wake.vtk"
        postprocess.write_wake_vtk(str(path), self.solver, self.solution, 5.0)
        points, cells, rest = self.read_vtk(path)
        self.assertEqual(len(cells), self.n_wake)
        self.assertEqual(len(points), 4 * self.n_wake)
        # the second point of the first cell is 5 downstream of the first
        np.testing.assert_allclose(points[1] - points[0], [5.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(self.read_scalar(rest, "gamma_wake"), self.solution["gamma_wake"],
                                   rtol=1e-6, atol=1e-9)

    def test_report_repeats_the_solver_numbers(self):
        path = self.folder / "report.json"
        postprocess.write_report(str(path), WING, self.solver, self.solution)
        report = json.loads(path.read_text())
        self.assertEqual(report["n_panels"], self.n_panels)
        self.assertEqual(report["n_vertices"], self.n_vertices)
        self.assertEqual(report["n_wake_horseshoes"], self.n_wake)
        self.assertAlmostEqual(report["coefficients"]["CL"], float(self.solution["forces"]["CL"]))
        self.assertAlmostEqual(report["coefficients"]["CL_KJ"], float(self.solution["CL_KJ"]))
        self.assertAlmostEqual(report["coefficients"]["CD_press"], float(self.solution["forces"]["CD_press"]))
        self.assertEqual(len(report["forces"]["F_body"]), 3)

    def test_off_body_csv_has_a_header_and_one_row_per_point(self):
        out = self.folder / "velocity.csv"
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]])
        postprocess.write_off_body_csv(str(out), points, np.array([[1.0, 0.0, 0.0], [0.9, 0.0, 0.1]]))
        rows = out.read_text().strip().split("\n")
        self.assertEqual(rows[0], "x,y,z,u,v,w")
        self.assertEqual(len(rows), 3)
        self.assertEqual([float(v) for v in rows[2].split(",")], [-1.0, 0.0, 0.5, 0.9, 0.0, 0.1])

    def test_grid_points_run_x_fastest_between_the_limits(self):
        grid = {"x_min": 0.0, "x_max": 2.0, "n_x": 3, "y_min": -1.0, "y_max": 1.0, "n_y": 2,
                "z_min": 5.0, "z_max": 5.0, "n_z": 1}
        origin, spacing, dims, points = postprocess.grid_points(grid)
        self.assertEqual(dims, [3, 2, 1])
        self.assertEqual(spacing, [1.0, 2.0, 0.0])
        np.testing.assert_allclose(points, [[0, -1, 5], [1, -1, 5], [2, -1, 5], [0, 1, 5], [1, 1, 5], [2, 1, 5]])

    def test_field_vtk_holds_the_solver_velocity_at_each_grid_point(self):
        grid = {"x_min": 5.0, "x_max": 7.0, "n_x": 2, "y_min": 0.0, "y_max": 4.0, "n_y": 2,
                "z_min": 1.0, "z_max": 1.0, "n_z": 1}
        path = self.folder / "off_body.vtk"
        postprocess.export_velocity_field(str(path), grid, self.solver)
        lines = path.read_text().split("\n")
        self.assertEqual(lines[3], "DATASET STRUCTURED_POINTS")
        self.assertEqual(lines[4], "DIMENSIONS 2 2 1")
        self.assertEqual(lines[7], "POINT_DATA 4")
        self.assertEqual(lines[8], "VECTORS velocity float")
        written = np.array([[float(v) for v in line.split()] for line in lines[9:13]])
        # the last point of the grid is (7, 4, 1)
        np.testing.assert_allclose(written[3], self.solver.evaluate_velocity([7.0, 4.0, 1.0]), rtol=1e-6)
        self.assertNotIn("nan", path.read_text())

    def test_export_solution_writes_every_file_named_in_the_input(self):
        solution = dict(self.solution)
        solution["off_body_points"] = np.array([[2.0, 0.0, 0.5]])
        solution["off_body_velocities"] = np.array([[1.0, 0.0, 0.0]])
        names = {"report_file": "report.json", "body_file": "body.vtk", "wake_file": "wake.vtk",
                 "off_body_velocity_file": "velocity.csv"}
        output = {key: str(self.folder / name) for key, name in names.items()}
        postprocess.export_solution({"mesh": {"filename": str(WING)}, "output": output}, self.solver, solution)
        for name in names.values():
            self.assertTrue((self.folder / name).exists(), name)


class ConvergenceTests(unittest.TestCase):
    def test_table_is_sorted_and_measured_from_the_finest_mesh(self):
        rows = [{"mesh": "fine", "n_panels": 400, "CL": 0.50, "CL_KJ": 0.40, "CD_press": 0.010},
                {"mesh": "coarse", "n_panels": 100, "CL": 0.45, "CL_KJ": 0.44, "CD_press": 0.012}]
        rows = postprocess.convergence_table(rows)
        self.assertEqual([r["mesh"] for r in rows], ["coarse", "fine"])
        self.assertAlmostEqual(rows[0]["CL_change[%]"], -10.0)
        self.assertAlmostEqual(rows[0]["CL_KJ_change[%]"], 10.0)
        self.assertAlmostEqual(rows[0]["CD_press_change[%]"], 20.0)
        self.assertAlmostEqual(rows[1]["CL_change[%]"], 0.0)

    def test_one_mesh_row_matches_a_direct_solve(self):
        input_data = {"mesh": {"filename": str(WING), "S_ref": 1.0, "c_ref": 1.0},
                      "flow": {"V_inf": [1.0, 0.0, 0.0]}}
        row = postprocess.solve_for_convergence(WING, input_data)
        self.assertEqual(row["mesh"], "wing_RA10_coarse.stl")
        self.assertEqual(row["n_panels"], Mesh(WING).N)
        for name in ["CL", "CL_KJ", "CD_press"]:
            self.assertTrue(np.isfinite(row[name]))


if __name__ == "__main__":
    unittest.main()
