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
        coefficients = report["coefficients"]
        self.assertAlmostEqual(coefficients["CL_surface"], float(self.solution["forces"]["CL"]))
        self.assertAlmostEqual(coefficients["CD_surface"], float(self.solution["forces"]["CD_press"]))
        self.assertAlmostEqual(coefficients["CL_trefftz"], float(self.solution["CL_KJ"]))
        self.assertAlmostEqual(coefficients["CD_trefftz"], float(self.solution["CD_KJ"]))
        self.assertAlmostEqual(report["forces"]["Drag_trefftz"], float(self.solution["Drag_KJ"]))
        # only surface / Trefftz-plane names in the report
        self.assertNotIn("KJ", json.dumps(report))
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

    def test_export_solution_skips_the_streamline_field_when_disabled(self):
        grid = {"enabled": False, "x_min": 5.0, "x_max": 7.0, "n_x": 2, "y_min": 0.0, "y_max": 4.0, "n_y": 2,
                "z_min": 1.0, "z_max": 1.0, "n_z": 1}
        output = {"report_file": str(self.folder / "report.json"), "off_body_file": str(self.folder / "field.vtk"),
                  "body_file": str(self.folder / "body.vtk"), "wake_file": str(self.folder / "wake.vtk")}
        input_data = {"mesh": {"filename": str(WING)}, "off_body_grid": grid, "output": output}
        postprocess.export_solution(input_data, self.solver, self.solution)
        self.assertFalse((self.folder / "field.vtk").exists())
        self.assertTrue((self.folder / "body.vtk").exists())
        # switched back on, the same settings write the field
        grid["enabled"] = True
        postprocess.export_solution(input_data, self.solver, self.solution)
        self.assertTrue((self.folder / "field.vtk").exists())


def mesh_row(mesh, n_panels, CL_surface, CL_trefftz, CD_surface, CD_trefftz, **extra):
    # one row of the convergence table, as the solver runs produce it
    row = {"mesh": mesh, "n_panels": n_panels, "CL_surface": CL_surface, "CL_trefftz": CL_trefftz,
           "CD_surface": CD_surface, "CD_trefftz": CD_trefftz}
    row.update(extra)
    return row


class ConvergenceTests(unittest.TestCase):
    def test_quantities_are_surface_and_trefftz_lift_and_drag(self):
        self.assertEqual(postprocess.CONVERGENCE_QUANTITIES, ["CL_surface", "CL_trefftz", "CD_surface", "CD_trefftz"])

    def test_table_is_sorted_and_measured_from_the_finest_mesh(self):
        rows = [mesh_row("fine", 400, 0.50, 0.40, 0.010, 0.0080),
                mesh_row("coarse", 100, 0.45, 0.44, 0.012, 0.0070)]
        rows = postprocess.convergence_table(rows)
        self.assertEqual([r["mesh"] for r in rows], ["coarse", "fine"])
        self.assertAlmostEqual(rows[0]["CL_surface_change[%]"], -10.0)
        self.assertAlmostEqual(rows[0]["CL_trefftz_change[%]"], 10.0)
        self.assertAlmostEqual(rows[0]["CD_surface_change[%]"], 20.0)
        self.assertAlmostEqual(rows[0]["CD_trefftz_change[%]"], -12.5)
        self.assertAlmostEqual(rows[1]["CL_surface_change[%]"], 0.0)

    def test_one_mesh_row_matches_a_direct_solve(self):
        input_data = {"mesh": {"filename": str(WING), "S_ref": 1.0, "c_ref": 1.0},
                      "flow": {"V_inf": [1.0, 0.0, 0.0]}}
        row = postprocess.solve_for_convergence(WING, input_data)
        self.assertEqual(row["mesh"], "wing_RA10_coarse.stl")
        self.assertEqual(row["n_panels"], Mesh(WING).N)
        for name in postprocess.CONVERGENCE_QUANTITIES:
            self.assertTrue(np.isfinite(row[name]))
        self.assertGreater(row["solve_time[s]"], 0.0)

    def test_coefficient_row_takes_surface_and_trefftz_values_from_the_solver(self):
        class FakeSolver:
            forces = {"CL": 0.45, "CD_press": 0.009}
            CL_KJ = 0.46
            CD_KJ = 0.0068
        row = postprocess.coefficient_row(FakeSolver())
        self.assertEqual(row, {"CL_surface": 0.45, "CL_trefftz": 0.46, "CD_surface": 0.009, "CD_trefftz": 0.0068})


class OrderOfConvergenceTests(unittest.TestCase):
    def test_matches_the_worked_example_in_the_docs(self):
        # docs/postprocessing.md Sec. 6.7: p = 1.14, extrapolated CL = 0.471, GCI = 2.9%
        result = postprocess.order_of_convergence([272, 864, 3344], [0.4272, 0.44856, 0.46088])
        self.assertEqual(result["behaviour"], "monotonic")
        self.assertAlmostEqual(result["r21"], 1.967, places=3)
        self.assertAlmostEqual(result["r32"], 1.782, places=3)
        self.assertAlmostEqual(result["p"], 1.14, places=2)
        self.assertAlmostEqual(result["extrapolated"], 0.471, places=3)
        self.assertAlmostEqual(result["GCI_fine[%]"], 2.9, places=1)

    def test_recovers_second_order_from_an_exact_power_law(self):
        # f = 0.5 + 3 h^2 with h = 1/sqrt(N): p must be 2 and the extrapolation 0.5, even with unequal ratios
        n = [272, 864, 3344]
        values = [0.5 + 3.0 * (1.0 / np.sqrt(N))**2 for N in n]
        result = postprocess.order_of_convergence(n, values)
        self.assertEqual(result["behaviour"], "monotonic")
        self.assertAlmostEqual(result["p"], 2.0, places=8)
        self.assertAlmostEqual(result["extrapolated"], 0.5, places=10)

    def test_oscillating_and_diverging_results_get_no_extrapolation(self):
        oscillating = postprocess.order_of_convergence([864, 3344, 13104], [0.4594, 0.4595, 0.4594])
        self.assertEqual(oscillating["behaviour"], "oscillating")
        diverging = postprocess.order_of_convergence([864, 3344, 13104], [0.437, 0.457, 0.400])
        self.assertEqual(diverging["behaviour"], "oscillating")
        growing = postprocess.order_of_convergence([864, 3344, 13104], [0.40, 0.41, 0.43])
        self.assertEqual(growing["behaviour"], "diverging")
        for result in [oscillating, diverging, growing]:
            self.assertIsNone(result["p"])
            self.assertIsNone(result["extrapolated"])
            self.assertIsNone(result["GCI_fine[%]"])

    def test_equal_steps_are_not_converging_even_with_round_off(self):
        # 0.05 -> 0.03 -> 0.01 changes by the same amount each time (R = 1), but in floating point
        # R comes out as 0.9999999999999998, which used to give p = 0 and a huge extrapolation
        result = postprocess.order_of_convergence([272, 864, 3344], [0.05, 0.03, 0.01])
        self.assertEqual(result["behaviour"], "diverging")
        self.assertIsNone(result["p"])
        self.assertIsNone(result["extrapolated"])

    def test_summary_uses_the_three_finest_meshes(self):
        rows = [mesh_row("a", 272, 0.38, 0.46, 0.05, 0.0078),
                mesh_row("b", 864, 0.42, 0.459, 0.03, 0.0068),
                mesh_row("c", 3344, 0.45, 0.4595, 0.01, 0.00675),
                mesh_row("d", 13104, 0.46, 0.4600, 0.008, 0.0068)]
        summary = postprocess.convergence_summary(rows)
        self.assertEqual([s["quantity"] for s in summary], ["CL_surface", "CL_trefftz", "CD_surface", "CD_trefftz"])
        for s in summary:
            self.assertEqual(s["meshes"], "b, c, d")
        # fewer than three meshes: no summary
        self.assertEqual(postprocess.convergence_summary(rows[:2]), [])

    def test_write_convergence_writes_the_table_summary_and_plot(self):
        folder = Path(tempfile.mkdtemp())
        rows = [mesh_row("a", 272, 0.4272, 0.46, 0.05, 0.0078, **{"solve_time[s]": 1.0}),
                mesh_row("b", 864, 0.44856, 0.459, 0.03, 0.0070, **{"solve_time[s]": 9.0}),
                mesh_row("c", 3344, 0.46088, 0.4595, 0.01, 0.0069, **{"solve_time[s]": 99.0})]
        rows = postprocess.convergence_table(rows)
        postprocess.write_convergence(rows, str(folder / "conv.csv"), str(folder / "conv.png"),
                                      str(folder / "summary.csv"))
        table = (folder / "conv.csv").read_text().strip().split("\n")
        header = table[0].split(",")
        self.assertEqual(header[:6], ["mesh", "n_panels", "CL_surface", "CL_trefftz", "CD_surface", "CD_trefftz"])
        self.assertIn("CD_trefftz_change[%]", header)
        self.assertIn("solve_time[s]", header)
        self.assertNotIn("KJ", table[0])
        self.assertEqual(len(table), 4)
        summary = (folder / "summary.csv").read_text().strip().split("\n")
        self.assertEqual(summary[0].split(",")[:2], ["quantity", "meshes"])
        self.assertEqual(len(summary), 5)
        self.assertTrue(summary[1].startswith("CL_surface,"))
        self.assertIn("monotonic", summary[1])
        self.assertTrue(summary[4].startswith("CD_trefftz,"))
        self.assertTrue((folder / "conv.png").exists())

    def test_plot_skips_extrapolations_with_order_below_one_half(self):
        # p = 0.18 gave CD_surface extrapolated to -0.0246 on the uniform wing: not drawn.
        # oscillating/diverging results (p None) are not drawn either; p = 0.5 and above are
        summary = [{"quantity": "CL_surface", "p": 0.81, "extrapolated": 0.465},
                   {"quantity": "CL_trefftz", "p": None, "extrapolated": None},
                   {"quantity": "CD_surface", "p": 0.18, "extrapolated": -0.0246},
                   {"quantity": "CD_trefftz", "p": 0.5, "extrapolated": 0.0074}]
        drawn = postprocess.plotted_extrapolations(summary)
        self.assertEqual([s["quantity"] for s in drawn], ["CL_surface", "CD_trefftz"])
        self.assertEqual(postprocess.MIN_ORDER_TO_PLOT, 0.5)

    def test_rows_from_a_case_table_rebuild_the_convergence_files(self):
        # all_cases.csv from the wing-family runs: one family per row, KJ column names from the solver
        folder = Path(tempfile.mkdtemp())
        cases = folder / "all_cases.csv"
        cases.write_text("family,mesh,n_panels,CL,CL_KJ,CD_press,CD_KJ,Cm_y,solve_time[s]\n"
                         "clustered,w1.stl,224,0.4267,0.4684,0.0238,0.0078,-0.11,0.05\n"
                         "clustered,w2.stl,728,0.4534,0.4692,0.0130,0.0068,-0.12,0.33\n"
                         "uniform,w1.stl,224,0.4025,0.4744,0.0406,0.0064,-0.11,0.04\n")
        families = postprocess.rows_from_case_table(str(cases))
        self.assertEqual(sorted(families), ["clustered", "uniform"])
        self.assertEqual(len(families["clustered"]), 2)
        self.assertEqual(families["clustered"][1], {"mesh": "w2.stl", "n_panels": 728, "CL_surface": 0.4534,
                                                    "CL_trefftz": 0.4692, "CD_surface": 0.0130,
                                                    "CD_trefftz": 0.0068, "solve_time[s]": 0.33})


if __name__ == "__main__":
    unittest.main()
