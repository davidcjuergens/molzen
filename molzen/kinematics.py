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
    _, pivot, axis_end, _ = [xyz[i] for i in dih_idx]

    # Translate atoms so that the second dihedral atom is at the origin.
    translated_atoms = xyz[atom_idx] - pivot
    rotated_atoms = (
        translated_atoms @ rotation_matrix(axis_end - pivot, angle_degrees).T
    )

    # Translate back
    rotated_atoms += pivot

    # Update the original xyz array
    new_xyz = xyz.copy()
    new_xyz[atom_idx] = rotated_atoms

    return new_xyz


def rotation_matrix(axis, angle_degrees):
    """Return a right-handed rotation matrix for a column vector.

    Row coordinates transform as ``points @ R.T``. ``angle_degrees`` follows
    the same sign convention as ``rotate_around_dihedral``.

    Args:
        axis: Rotation axis. Length is ignored; a zero axis is rejected.
        angle_degrees: Rotation angle in degrees.

    Raises:
        ValueError: If the axis or angle is not a finite, usable value.
    """
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("axis must be a finite length-3 vector.")
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("axis must have nonzero length.")
    x, y, z = axis / norm
    angle = float(angle_degrees)
    if not np.isfinite(angle):
        raise ValueError("angle_degrees must be finite.")
    cosine = np.cos(np.radians(angle))
    sine = np.sin(np.radians(angle))
    complement = 1.0 - cosine
    return np.array(
        [
            [
                complement * x * x + cosine,
                complement * x * y - z * sine,
                complement * x * z + y * sine,
            ],
            [
                complement * y * x + z * sine,
                complement * y * y + cosine,
                complement * y * z - x * sine,
            ],
            [
                complement * z * x - y * sine,
                complement * z * y + x * sine,
                complement * z * z + cosine,
            ],
        ],
        dtype=float,
    )


def apply_rigid_motion(coords, rotation, *, origin=None, translation=None):
    """Apply one lab-frame rigid motion to coordinates of shape (..., 3).

    The column-vector ``rotation`` is applied about ``origin``, then
    ``translation`` is added:
    ``(coords - origin) @ rotation.T + origin + translation``.
    """
    points = np.asarray(coords, dtype=float)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("coords must have shape (..., 3).")
    rot = np.asarray(rotation, dtype=float)
    if rot.shape != (3, 3) or not np.isfinite(rot).all():
        raise ValueError("rotation must be a finite 3x3 matrix.")
    if origin is None:
        origin_vec = np.zeros(3, dtype=float)
    else:
        origin_vec = np.asarray(origin, dtype=float)
        if origin_vec.shape != (3,) or not np.isfinite(origin_vec).all():
            raise ValueError("origin must be a finite length-3 vector.")
    if translation is None:
        translation_vec = np.zeros(3, dtype=float)
    else:
        translation_vec = np.asarray(translation, dtype=float)
        if translation_vec.shape != (3,) or not np.isfinite(translation_vec).all():
            raise ValueError("translation must be a finite length-3 vector.")
    return (points - origin_vec) @ rot.T + origin_vec + translation_vec


def best_fit_frame(points):
    """Return the centroid and a right-handed frame for a point set.

    Args:
        points: Coordinates with shape (N, 3). At least three non-collinear
            points are required.

    Returns:
        A centroid of shape (3,) and a (3, 3) basis whose columns are
        ``(axis_1, axis_2, normal)``. ``normal`` is the best-fit plane normal.
    """
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if len(coords) < 3:
        raise ValueError("At least three atoms are required to fit a plane.")
    if not np.isfinite(coords).all():
        raise ValueError("points must be finite.")
    centroid = coords.mean(axis=0)
    _, singular, vt = np.linalg.svd(coords - centroid, full_matrices=False)
    if singular[0] == 0.0 or singular[1] <= 1e-8 * singular[0]:
        raise ValueError("Selected atoms do not span a plane.")
    axis_1 = vt[0]
    normal = np.cross(axis_1, vt[1])
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0.0:
        raise ValueError("Selected atoms do not span a plane.")
    normal = normal / normal_norm
    axis_2 = np.cross(normal, axis_1)
    return centroid, np.column_stack((axis_1, axis_2, normal))


def _frame_from_normal(normal):
    """Return a right-handed basis whose third column is ``normal``."""
    direction = np.asarray(normal, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("normal must be a finite length-3 vector.")
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("normal must have nonzero length.")
    direction = direction / norm
    helper = (
        np.array([1.0, 0.0, 0.0])
        if abs(direction[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    axis_1 = np.cross(helper, direction)
    axis_1 /= np.linalg.norm(axis_1)
    axis_2 = np.cross(direction, axis_1)
    return np.column_stack((axis_1, axis_2, direction))


def plane_alignment(points, normal):
    """Return the rigid motion that centers a plane and aligns its normal.

    Args:
        points: Coordinates with shape (N, 3) used to fit the plane.
        normal: Requested lab-frame normal. It is normalized.

    Returns:
        ``origin``, column-vector ``rotation``, and ``translation``. Applying
        them with ``apply_rigid_motion`` puts the centroid at the origin and
        the best-fit normal along ``normal``.
    """
    origin, source = best_fit_frame(points)
    rotation = _frame_from_normal(normal) @ source.T
    return origin, rotation, -origin
