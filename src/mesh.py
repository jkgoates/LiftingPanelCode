import warnings

import numpy as np
from stl import mesh as stl_mesh

from panels import Horseshoe, Tri, Vertex


class Mesh:
    """Triangular surface mesh loaded from an STL file."""

    def __init__(self, filename, multi_file=False):
        self._load_stl(filename, multi_file=multi_file)

    def _load_stl(self, stl_file, multi_file=False):
        """Load one STL mesh, or all solids in an STL when ``multi_file`` is set."""
        if multi_file:
            raw_meshes = list(
                stl_mesh.Mesh.from_multi_file(stl_file, remove_empty_areas=True)
            )
        else:
            raw_meshes = [
                stl_mesh.Mesh.from_file(stl_file, remove_empty_areas=True)
            ]

        if not raw_meshes:
            raise ValueError(f"No meshes were found in STL file: {stl_file}")

        vectors = np.concatenate([raw_mesh.vectors for raw_mesh in raw_meshes])
        # Check the geometry itself instead of trusting normals stored in the STL.
        cross_products = np.cross(
            vectors[:, 1] - vectors[:, 0], vectors[:, 2] - vectors[:, 0]
        )
        valid_facets = np.linalg.norm(cross_products, axis=1) > 0.0

        skipped_facets = np.flatnonzero(~valid_facets)
        for facet_index in skipped_facets:
            warnings.warn(
                f"Panel {facet_index} has zero area. Skipping...",
                RuntimeWarning,
                stacklevel=2,
            )

        vectors = vectors[valid_facets]
        if vectors.shape[0] == 0:
            raise ValueError(f"STL file contains no non-degenerate facets: {stl_file}")

        self.N = vectors.shape[0]
        self.panels = np.array(
            [Tri(v0=facet[0], v1=facet[1], v2=facet[2]) for facet in vectors],
            dtype=object,
        )

        self.cp = np.zeros((self.N, 3))
        self.n = np.zeros((self.N, 3))
        self.dA = np.zeros(self.N)
        for i, panel in enumerate(self.panels):
            self.n[i], self.dA[i], self.cp[i] = panel.get_info()

        self.vertices, inverse_indices = np.unique(
            vectors.reshape(-1, 3), return_inverse=True, axis=0
        )
        panel_vertex_indices = inverse_indices.reshape(self.N, 3)
        self._panel_vertex_indices = [
            [3, *indices.tolist()] for indices in panel_vertex_indices
        ]

        self.N_vert = len(self.vertices)
        self.vertex_objects = np.array(
            [Vertex(vertex, self.N_vert) for vertex in self.vertices], dtype=object
        )

    def _get_flow_direction(self, V_inf):
        direction = np.asarray(V_inf, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("V_inf must contain three finite values")

        magnitude = np.linalg.norm(direction)
        if magnitude == 0.0:
            raise ValueError("V_inf cannot be zero")

        return direction / magnitude

    def find_kutta_edges(self, V_inf):
        # Find the panels and opposite vertices belonging to each mesh edge
        edge_data = {}
        for panel_index, indices in enumerate(self._panel_vertex_indices):
            a, b, c = indices[1:]
            panel_edges = [(a, b, c), (b, c, a), (c, a, b)]
            for v0, v1, opposite_vertex in panel_edges:
                edge = tuple(sorted((v0, v1)))
                if edge not in edge_data:
                    edge_data[edge] = []
                edge_data[edge].append((panel_index, opposite_vertex))

        direction = self._get_flow_direction(V_inf)
        vertex_position = self.vertices @ direction
        kutta_edges = []

        for edge, adjacent_panels in edge_data.items():
            # A Kutta edge must join two panels
            if len(adjacent_panels) != 2:
                continue

            p0 = adjacent_panels[0][0]
            p1 = adjacent_panels[1][0]
            angle = np.degrees(
                np.arccos(np.clip(np.dot(self.n[p0], self.n[p1]), -1.0, 1.0))
            )

            # The trailing edge is sharp and downstream of both adjacent panels
            opposite_vertices = [item[1] for item in adjacent_panels]
            if angle >= 150.0 and max(vertex_position[opposite_vertices]) < min(vertex_position[list(edge)]):
                kutta_edges.append(edge)

        return kutta_edges

    def initialize_wake(self, kutta_edges, V_inf):
        direction = self._get_flow_direction(V_inf)
        wake = []

        for edge in kutta_edges:
            v0 = self.vertices[edge[0]]
            v1 = self.vertices[edge[1]]
            wake.append(Horseshoe(v0, v1, direction))

        return wake
