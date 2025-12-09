import numpy as np


def np_kabsch(A, B):
    """
    Numpy version of kabsch algorithm. Superimposes B onto A

    Parameters:
        (A,B) np.array - shape (N,3) arrays of xyz crds of points


    Returns:
        rms - rmsd between A and B
        R - rotation matrix to superimpose B onto A
        rB - the rotated B coordinates
    """
    A = np.copy(A)
    B = np.copy(B)

    def centroid(X):
        # return the mean X,Y,Z down the atoms
        return np.mean(X, axis=0, keepdims=True)

    def rmsd(V, W, eps=0):
        # First sum down atoms, then sum down xyz
        N = V.shape[-2]
        return np.sqrt(np.sum((V - W) * (V - W), axis=(-2, -1)) / N + eps)

    N, ndim = A.shape

    # move to centroid
    A = A - centroid(A)
    B = B - centroid(B)

    # computation of the covariance matrix
    C = np.matmul(A.T, B)

    # compute optimal rotation matrix using SVD
    U, S, Vt = np.linalg.svd(C)

    # ensure right handed coordinate system
    d = np.eye(3)
    d[-1, -1] = np.sign(np.linalg.det(Vt.T @ U.T))

    # construct rotation matrix
    R = Vt.T @ d @ U.T

    # get rotated coords
    rB = B @ R

    # calculate rmsd
    rms = rmsd(A, rB)

    return rms, rB, R


def rotate_around_dihedral(
    xyz: np.ndarray, dih_idx: list, atom_idx: list, angle_degrees: float
):
    """
    Rotate the specified atoms around the dihedral defined by dih_idx by angle_degrees.

    Args:
        xyz: Array of atomic coords (N, 3)
        dih_idx: Four atom indices defining the dihedral (a, b, c, d)
        atom_idx: Indices of atoms to rigidly rotate around the dihedral
        angle_degrees: Angle in degrees to rotate
    """
    angle_radians = np.radians(angle_degrees)
    a, b, c, d = [xyz[i] for i in dih_idx]

    # Define the rotation axis
    axis = c - b
    axis /= np.linalg.norm(axis)
    x, y, z = axis

    # Translate atoms so that point b is at the origin

    translated_atoms = xyz[atom_idx] - b

    # convert axis-angle to rotation matrix (https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle)
    c = np.cos(angle_radians)
    C = 1 - c
    s = np.sin(angle_radians)

    # fmt: off
    R = np.array([[C*x*x + c,    C*x*y - z*s,  C*x*z + y*s],
                  [C*y*x + z*s,  C*y*y + c,    C*y*z - x*s],
                  [C*z*x - y*s,  C*z*y + x*s,  C*z*z + c    ]])
    # fmt: on

    # rotate
    rotated_atoms = translated_atoms @ R.T

    # Translate back
    rotated_atoms += b

    # Update the original xyz array
    new_xyz = xyz.copy()
    new_xyz[atom_idx] = rotated_atoms

    return new_xyz
