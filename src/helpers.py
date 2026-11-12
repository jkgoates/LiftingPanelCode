def norm(v):
    return (v[0]**2 + v[1]**2 + v[2]**2)**0.5

def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]