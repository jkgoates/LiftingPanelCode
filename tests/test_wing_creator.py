import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "dev" / "helper_scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mesh import Mesh
from wing_creator import create_wing, export_wing_to_stl


class WingCreatorTests(unittest.TestCase):
    def setUp(self):
        self.wing = create_wing(
            RA=10.0,
            RT=0.5,
            b=10.0,
            series="0010",
            n_span=4,
            n_chord=8,
            sweep=0.0,
        )

    def test_creates_expected_full_span_closed_mesh(self):
        self.assertEqual(self.wing.shape, (196, 3, 3))
        points = self.wing.reshape(-1, 3)
        np.testing.assert_allclose(points[:, 1].min(), -5.0)
        np.testing.assert_allclose(points[:, 1].max(), 5.0)
        np.testing.assert_allclose(points[:, 0].min(), 0.0, atol=1e-14)
        expected_root_chord = 2.0 * 10.0 / (10.0 * (1.0 + 0.5))
        np.testing.assert_allclose(
            points[:, 0].max(), expected_root_chord, rtol=1e-12
        )
        double_areas = np.linalg.norm(
            np.cross(
                self.wing[:, 1] - self.wing[:, 0],
                self.wing[:, 2] - self.wing[:, 0],
            ),
            axis=1,
        )
        self.assertTrue(np.all(double_areas > 0.0))

    def test_export_is_watertight_and_solver_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wing.stl"
            export_wing_to_stl(self.wing, output)
            imported = Mesh(output)

        self.assertEqual(imported.N, 196)
        edge_counts = {}
        for _, a, b, c in imported._panel_vertex_indices:
            for edge in ((a, b), (b, c), (c, a)):
                edge = tuple(sorted(edge))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        self.assertTrue(all(count == 2 for count in edge_counts.values()))
        self.assertGreater(len(imported.find_kutta_edges([1.0, 0.0, 0.0])), 0)

    def test_clustered_spacing_changes_vertices_not_panel_count(self):
        clustered = create_wing(
            10.0, 0.5, 10.0, "0010", 4, 8, True, True, sweep=0.0
        )
        self.assertEqual(clustered.shape, self.wing.shape)
        self.assertFalse(np.allclose(clustered, self.wing))

    def test_can_leave_both_tips_open(self):
        open_wing = create_wing(
            10.0,
            0.5,
            10.0,
            "0010",
            4,
            8,
            close_tips=False,
        )
        self.assertEqual(open_wing.shape, (168, 3, 3))

        vertices, inverse = np.unique(
            open_wing.reshape(-1, 3), axis=0, return_inverse=True
        )
        indices = inverse.reshape(-1, 3)
        edges = np.sort(
            np.concatenate(
                (indices[:, [0, 1]], indices[:, [1, 2]], indices[:, [2, 0]])
            ),
            axis=1,
        )
        unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[edge_counts == 1]
        boundary_y = vertices[boundary_edges, 1]
        self.assertEqual(len(boundary_edges), 28)
        self.assertTrue(np.all(np.isclose(np.abs(boundary_y), 5.0)))


if __name__ == "__main__":
    unittest.main()
