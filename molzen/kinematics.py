import numpy as np 

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
