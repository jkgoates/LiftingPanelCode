import numpy as np
from helpers import cross, norm

class Vertex:
    def __init__(self, vertex, N_vert):
        self.vertex = vertex
        self.N_vert = N_vert

class Panel:

    def __init__(self):
        pass

    def get_info(self):

        # get panel normal
        n = self.get_normal()
        dA = self.get_area()
        cp = self.get_centroid()

        return n, dA, cp

    def get_normal(self):
        n = cross(self.v1 - self.v0, self.v2 - self.v0)
        return n

    def get_area(self):
        n = self.get_normal()
        dA = 0.5 * norm(n)
        return dA

    def get_centroid(self):
        cp = (self.v0 + self.v1 + self.v2) / 3
        return cp


class Tri(Panel):
    def __init__(self, v0, v1, v2):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.N = 3
        self.vertices = [v0, v1, v2]




