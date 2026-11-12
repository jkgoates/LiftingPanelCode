import numpy as np
import stl

from panels import Tri, Vertex
from helpers import norm


class Mesh:
    def __init__(self, filename):
        self._load_stl(filename)

    def _load_stl(self, stl_file):
        # Loads mesh from an stl file

        self.panels = []
        v0 = []
        v1 = []
        v2 = []


        # Load one mesh from file
        raw_mesh = stl.mesh.Mesh.from_file(stl_file, remove_empty_areas=True)

        # Initialize storage
        N = raw_mesh.v0.shape[0]
        self.N = N
        bad_facets = []

        # Loop through panels and initialize objects
        for i in range(N):

            # Check for finite area
            if norm(raw_mesh.normals[i]) == 0.0:
                self.N -= 1
                #warnings.warn("Panel {0} has zero area. Skipping...".format(i))
                bad_facets.append(i)
                continue

            # Initialize
            panel = Tri(v0=raw_mesh.v0[i],
                        v1=raw_mesh.v1[i],
                        v2=raw_mesh.v2[i])

            self.panels.append(panel)

        self.panels = np.array(self.panels)

        # Store panel information
        self.cp = np.zeros((self.N, 3))
        self.n = np.zeros((self.N, 3))
        self.dA = np.zeros(self.N)
        for i in range(self.N):
            self.n[i], self.dA[i], self.cp[i] = self.panels[i].get_info()

        # Return vertices belonging to existent panels
        good_facets = [i for i in range(N) if i not in bad_facets]

        raw_vertices = np.concatenate((raw_mesh.v0[good_facets], raw_mesh.v1[good_facets], raw_mesh.v2[good_facets]))

        # Get vertex list
        self.vertices, inverse_indices = np.unique(raw_vertices, return_inverse=True, axis=0)
        self._panel_vertex_indices = []
        for i in range(self.N):
            self._panel_vertex_indices.append([3, *inverse_indices[i::self.N]])

    def _load_stl(self, stl_file, multi_file):
        # Loads mesh from an stl file

        self.panels = []
        v0 = []
        v1 = []
        v2 = []

        # Load stl file
        if multi_file:

            # Load multiple meshes from file
            raw_meshes = stl.mesh.Mesh.from_multi_file(stl_file, remove_empty_areas=True)

            # Initialize storage
            bad_facets = []

            # Loop through panels and initialize objects
            i = -1
            N = 0
            for raw_mesh in raw_meshes:

                # Update number of panels
                N += raw_mesh.v0.shape[0]

                # Store vertices
                v0.append(raw_mesh.v0)
                v1.append(raw_mesh.v1)
                v2.append(raw_mesh.v2)
                for j in range(raw_mesh.v0.shape[0]):
                    i += 1

                    # Check for finite area
                    if norm(raw_mesh.normals[j]) == 0.0:
                        N -= 1
                        warnings.warn("Panel {0} has zero area. Skipping...".format(i))
                        bad_facets.append(i)
                        continue

                    # Initialize
                    panel = Tri(v0=raw_mesh.v0[j],
                                v1=raw_mesh.v1[j],
                                v2=raw_mesh.v2[j])

                    self.panels.append(panel)

            self.N = N

        else:

            # Load one mesh from file
            raw_mesh = stl.mesh.Mesh.from_file(stl_file, remove_empty_areas=True)

            # Initialize storage
            N = raw_mesh.v0.shape[0]
            self.N = N
            bad_facets = []

            # Loop through panels and initialize objects
            for i in range(N):

                # Check for finite area
                if norm(raw_mesh.normals[i]) == 0.0:
                    self.N -= 1
                    warnings.warn("Panel {0} has zero area. Skipping...".format(i))
                    bad_facets.append(i)
                    continue

                # Initialize
                panel = Tri(v0=raw_mesh.v0[i],
                            v1=raw_mesh.v1[i],
                            v2=raw_mesh.v2[i])

                self.panels.append(panel)

        self.panels = np.array(self.panels)

        # Store panel information
        self.cp = np.zeros((self.N, 3))
        self.n = np.zeros((self.N, 3))
        self.dA = np.zeros(self.N)
        for i in range(self.N):
            self.n[i], self.dA[i], self.cp[i] = self.panels[i].get_info()

        # Return vertices belonging to existent panels
        good_facets = [i for i in range(N) if i not in bad_facets]
        if multi_file:
            v0 = np.concatenate(v0, axis=0)
            v1 = np.concatenate(v1, axis=0)
            v2 = np.concatenate(v2, axis=0)
            raw_vertices = np.concatenate((v0[good_facets], v1[good_facets], v2[good_facets]))

        else:
            raw_vertices = np.concatenate((raw_mesh.v0[good_facets], raw_mesh.v1[good_facets], raw_mesh.v2[good_facets]))

        # Get vertex list
        self.vertices, inverse_indices = np.unique(raw_vertices, return_inverse=True, axis=0)
        self._panel_vertex_indices = []
        for i in range(self.N):
            self._panel_vertex_indices.append([3, *inverse_indices[i::self.N]])

        # Initialize vertices (same for all types)
        self.N_vert = len(self.vertices)
        self.vertex_objects = np.empty(self.N_vert, dtype=Vertex)
        for i, vertex in enumerate(self.vertices):
            self.vertex_objects[i] = Vertex(vertex, self.N_vert)