"""Geometry measurements shared by the viewers."""

from typing import Any

import numpy as np

# Selections closer than this are treated as coincident, in angstroms.
_MIN_LENGTH = 1e-6


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


def _as_points(points: np.ndarray) -> np.ndarray:
    """Return finite coordinates with shape (N, 3)."""
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("points must be finite and have shape (N, 3).")
    return coords


def _length(vector: np.ndarray) -> float:
    """Return a vector length, rejecting coincident endpoints."""
    length = float(np.linalg.norm(vector))
    if length < _MIN_LENGTH:
        raise ValueError("Geometry is undefined for coincident points.")
    return length


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the distance between two points, in angstroms."""
    points = _as_points(np.vstack([a, b]))
    return _length(points[1] - points[0])


def bond_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle at ``b`` formed by ``a-b-c``, in degrees."""
    points = _as_points(np.vstack([a, b, c]))
    toward_a = points[0] - points[1]
    toward_c = points[2] - points[1]
    cosine = np.dot(toward_a, toward_c) / (_length(toward_a) * _length(toward_c))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def dihedral_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Return the signed ``a-b-c-d`` dihedral, in degrees.

    The sign follows the right-hand rule about the ``b -> c`` axis. The result
    is in the range (-180, 180].
    """
    points = _as_points(np.vstack([a, b, c, d]))
    toward_a = points[0] - points[1]
    axis = points[2] - points[1]
    toward_d = points[3] - points[2]
    axis = axis / _length(axis)
    plane_a = toward_a - np.dot(toward_a, axis) * axis
    plane_d = toward_d - np.dot(toward_d, axis) * axis
    _length(plane_a)
    _length(plane_d)
    sine = float(np.dot(np.cross(axis, plane_a), plane_d))
    cosine = float(np.dot(plane_a, plane_d))
    return float(np.degrees(np.atan2(sine, cosine)))


def describe_geometry(points: np.ndarray) -> dict[str, Any]:
    """Describe one to four points for the live geometry panel.

    Args:
        points: Coordinates in selection order, shape ``(N, 3)``, in angstroms.

    Returns:
        A dict with ``kind`` and ``value``. Kinds are ``coordinates``,
        ``distance``, ``angle``, and ``dihedral``. Distances are angstroms and
        angles are degrees. Degenerate input returns ``kind`` ``undefined``.

    Raises:
        ValueError: If ``points`` does not contain one to four finite positions.
    """
    coords = _as_points(points)
    if not 1 <= len(coords) <= 4:
        raise ValueError("Geometry display accepts 1 to 4 points.")
    try:
        if len(coords) == 1:
            return {"kind": "coordinates", "value": coords[0].tolist()}
        if len(coords) == 2:
            return {"kind": "distance", "value": distance(coords[0], coords[1])}
        if len(coords) == 3:
            return {
                "kind": "angle",
                "value": bond_angle(coords[0], coords[1], coords[2]),
            }
        return {
            "kind": "dihedral",
            "value": dihedral_angle(coords[0], coords[1], coords[2], coords[3]),
        }
    except ValueError as exc:
        if "undefined" not in str(exc):
            raise
        return {"kind": "undefined", "value": None}
