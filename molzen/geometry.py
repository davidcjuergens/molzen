"""Geomtry utils"""

import numpy as np


def np_dihedral(a, b, c, d):
    """Compute dihedral angle for four points a, b, c, d

    Args:
        a, b, c, d: np.ndarray of shape (3,)"""

    v1_1 = b - a
    v1_2 = c - b

    v2_1 = v1_2  # same as c - b
    v2_2 = d - c

    # Compute the normal vectors for the two planes
    n1 = np.cross(v1_1, v1_2)
    n2 = np.cross(v2_1, v2_2)

    # Compute the dihedral angle
    angle = np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))

    return angle
