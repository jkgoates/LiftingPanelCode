import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mesh import Mesh


class MeshImportTests(unittest.TestCase):
    def test_imports_provided_mesh(self):
        imported_mesh = Mesh(PROJECT_ROOT / "meshes" / "wing_RA10_coarse.stl")

        self.assertEqual(imported_mesh.N, 232)
        self.assertEqual(imported_mesh.N_vert, 118)
        self.assertEqual(imported_mesh.panels.shape, (232,))
        self.assertEqual(imported_mesh.cp.shape, (232, 3))
        self.assertEqual(imported_mesh.n.shape, (232, 3))
        self.assertEqual(imported_mesh.dA.shape, (232,))
        self.assertTrue(np.all(imported_mesh.dA > 0.0))
        np.testing.assert_allclose(
            np.linalg.norm(imported_mesh.n, axis=1), 1.0, rtol=1e-7, atol=1e-7
        )

        panel_vertices = np.asarray(
            [panel.vertices for panel in imported_mesh.panels]
        )
        cross_products = np.cross(
            panel_vertices[:, 1] - panel_vertices[:, 0],
            panel_vertices[:, 2] - panel_vertices[:, 0],
        )
        expected_areas = np.linalg.norm(cross_products, axis=1) / 2.0
        expected_normals = cross_products / np.linalg.norm(
            cross_products, axis=1
        )[:, np.newaxis]
        expected_centroids = np.mean(panel_vertices, axis=1)
        np.testing.assert_allclose(imported_mesh.n, expected_normals)
        np.testing.assert_allclose(imported_mesh.dA, expected_areas)
        np.testing.assert_allclose(imported_mesh.cp, expected_centroids)

    def test_panel_indices_reference_the_imported_vertices(self):
        imported_mesh = Mesh(PROJECT_ROOT / "meshes" / "wing_RA10_coarse.stl")

        for panel, indices in zip(
            imported_mesh.panels, imported_mesh._panel_vertex_indices
        ):
            self.assertEqual(indices[0], 3)
            np.testing.assert_array_equal(
                imported_mesh.vertices[indices[1:]], panel.vertices
            )


class WakeInitializationTests(unittest.TestCase):
    def test_finds_kutta_edges_and_initializes_wake(self):
        mesh = Mesh(PROJECT_ROOT / "meshes" / "wing_RA10_coarse.stl")
        V_inf = [1.0, 0.0, 0.0]

        kutta_edges = mesh.find_kutta_edges(V_inf)
        wake = mesh.initialize_wake(kutta_edges, V_inf)

        self.assertEqual(len(kutta_edges), 8)
        self.assertEqual(len(wake), 8)
        self.assertEqual(
            len({vertex for edge in kutta_edges for vertex in edge}), 9
        )

        for edge, horseshoe in zip(kutta_edges, wake):
            np.testing.assert_array_equal(horseshoe.v0, mesh.vertices[edge[0]])
            np.testing.assert_array_equal(horseshoe.v1, mesh.vertices[edge[1]])
            np.testing.assert_array_equal(horseshoe.direction, V_inf)

    def test_rejects_zero_flow_velocity(self):
        mesh = Mesh(PROJECT_ROOT / "meshes" / "wing_RA10_coarse.stl")

        with self.assertRaises(ValueError):
            mesh.find_kutta_edges([0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
