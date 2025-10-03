"""
Python translation of TracingTypes.h/.cpp and TracingUtils.h/.cpp (with PLUCKER_KERNEL and LEAN_PLUCKER assumed False),
revised to store all x/y/z data as (N,3) arrays for efficient vectorized NumPy operations.
"""
import numpy as np
from scipy.constants import golden
from typing import Tuple

EPS_EDGE = 1e-7         # edge-inclusive tolerance for ray hits
EPS_FACING = 1e-7       # ray-plane test perpendicular tolerance
EPS_PARALLEL = 1e-7    # ray-plane test parallel tolerance
EPS_ZFIGHT = 1e-5       # tie-breaker window for Z-fighting
# TODO: Use this, and also, fix its use in C++ (it currently checks against patch_id)
EPS_SELFHIT = 1e-3      # reject hits too close to the ray origin


class TriangleMesh:
    """
    Structure-of-Arrays (SoA) container for a triangle mesh used by the tracing kernels.

    The Möller–Trumbore algorithm stores (for each triangle i):
      - A: First vertex
      - edge1: B-A
      - edge2: C-A
      - n: Surface normal = edge1 x edge2
      - d0: plane offset so that dot(n, X) - d0 = 0 for all X on the triangle's plane
      - ID

    Notes:
      * The kernel enforces:
          - the triangle's normal faces the ray origin (dot(n,O)-d0 > EPS_FACING)
          - the intersection point lies inside the triangle (edges included)
        It does NOT enforce t > 0 (line–triangle, not ray–triangle).
      * In case of Z-fighting, the lower triangle index wins (mirrors the original logic).
    """

    def __init__(self, vertices: np.ndarray,
                 vert_triplets: np.ndarray,
                 patch_ids: np.ndarray):
        """
        Build the SoA from:
          - vertices:       (N,3) array of 3D coordinates (float)
          - vert_triplets:  (M,3) array of vertex indices forming M triangles (int)
          - patch_ids:      (M,)  array of patch IDs for each triangle (int)

        The winding order in `faces` determines the triangle normal orientation
        (right-hand rule): n = (B - A) x (C - A).
        """
        # Validate & coerce inputs
        V = np.asarray(vertices, dtype=float)
        F = np.asarray(vert_triplets, dtype=int)
        self.ID = np.asarray(patch_ids, dtype=int)

        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        if self.ID.ndim != 1 or self.ID.shape[0] != F.shape[0]:
            raise ValueError("patch_ids must have shape (M,) matching faces.shape[0]")

        if F.min() < 0 or F.max() >= V.shape[0]:
            raise IndexError("faces contain vertex indices out of range for `vertices`")

        self.v1 = V[F[:, 0]]
        self.edge1 = V[F[:, 1]] - self.v1
        self.edge2 = V[F[:, 2]] - self.v1
        self.n = np.cross(self.edge1, self.edge2)

        # Note: these work because self.n is not normalized yet
        self.d0 = np.einsum("ij,ij->i", self.n, self.v1)
        self.area = 0.5 * np.linalg.norm(self.n, axis=1)

        # Normalize normal
        self.n /= np.linalg.norm(self.n, axis=1)[:, None]

    def size(self) -> int:
        return int(self.v1.shape[0])


# TODO: In "hemisphere mode", add two options:
#       - Expose the fact that it's a hemisphere
#       - Hide the fact that it's a hemisphere (return duplicated, inverted results, like the C++ code does)
class RayBundle:
    """
    Class for tracing a bundle of rays with (possibly) separate origins and directions.
    All RayPencil functionality from the original code is folded into this class.
    """

    def __init__(self, O: np.ndarray, D: np.ndarray):
        """
        Basic, explicit constructor.
        It also allocates per-ray bookkeeping arrays on the instance.
        """
        self.O = np.asarray(O, dtype=float)
        self.D = np.asarray(D, dtype=float)
        # Normalize directions (avoid division by zero by clamping length to 1)
        self.D = self.D / np.linalg.norm(self.D, axis=1, keepdims=True)

        N = self.O.shape[0]
        # Per-ray state (initialized to defaults)
        self.radiance = np.ones(N)
        self.totalDistance = np.zeros(N)

        # TODO: Use currentTriangle to avoid self-hits
        self.currentTriangle = np.full(N, -1, dtype=int)

        self.frontDistance = np.full(N, np.nan, dtype=float)
        self.frontCosine = np.full(N, np.nan, dtype=float)
        self.frontPatch = np.full(N, -1, dtype=int)

        self.backDistance = np.full(N, np.nan, dtype=float)
        self.backCosine = np.full(N, np.nan, dtype=float)
        self.backPatch = np.full(N, -1, dtype=int)

    @classmethod
    def from_shared_origin(cls,
                           origin: np.ndarray,
                           directions: np.ndarray,
    ) -> "RayBundle":
        """
        Initialization with one shared origin and many directions.

        Parameters
        ----------
        origin : (3,) float ndarray
            The common origin for all rays.
        directions : (M,3) float ndarray
            Array of M direction vectors (not necessarily normalized).

        Notes
        -----
        Directions are normalized inside the RayBundle constructor.
        """
        O = np.asarray(origin, dtype=float)
        D = np.asarray(directions, dtype=float)

        if O.ndim != 1 or O.shape[0] != 3:
            raise ValueError("origin must have shape (3,)")

        if D.ndim != 2 or D.shape[1] != 3:
            raise ValueError("directions must have shape (M, 3)")

        # Broadcast origin to all rays
        O = np.repeat(O[None, :], D.shape[0], axis=0)

        return cls(O, D)

    # TODO: Allow using this method to construct several pencils.
    @classmethod
    def from_origins_and_directions(cls,
                                    origins: np.ndarray,
                                    directions: np.ndarray,
    ) -> "RayBundle":
        """
        Initialization with per-ray origins and directions.

        Parameters
        ----------
        origins : (M,3) float ndarray
            Per-ray origin points.
        directions : (M,3) float ndarray
            Per-ray direction vectors (not necessarily normalized).

        Notes
        -----
        Directions are normalized inside the RayBundle constructor.
        """
        O = np.asarray(origins, dtype=float)
        D = np.asarray(directions, dtype=float)

        if O.ndim != 2 or O.shape[1] != 3:
            raise ValueError("origins must have shape (M, 3)")
        if D.ndim != 2 or D.shape[1] != 3:
            raise ValueError("directions must have shape (M, 3)")
        if O.shape[0] != D.shape[0]:
            raise NotImplementedError("origins and directions must have the same number of rows (M)."
                                      " TODO: allow using this method to construct several pencils.")

        return cls(O, D)

    # TODO: Allow using this method to construct several pencils.
    @classmethod
    def sample_sphere(cls,
                      num_rays: int,
                      hemisphere_only: bool = False,
                      origin: np.ndarray = np.zeros(3),
                      north_pole: np.ndarray = np.array([0., 0., 1.]),
    ) -> "RayBundle":
        """
        Build a 'pencil' of directions sampled uniformly on a Fibonacci sphere,
        optionally restricted to the +Z hemisphere (matching the C++ RayPencil constructor).

        Parameters
        ----------
        num_rays : int
            Number of directions to generate.
        hemisphere_only : bool, default True
            If True, restrict to +Z hemisphere; else use the full sphere.
        origin : (3,) float ndarray, default zeros(3)
            Common origin for all rays.
        north_pole : (3,) float ndarray, default (0, 0, 1)
            Center of the +Z hemisphere.

        Notes
        -----
        Directions are normalized in the RayBundle constructor.
        """
        O = np.asarray(origin, dtype=float)
        if O.ndim != 1 or O.shape[0] != 3:
            raise ValueError("origin must have shape (3,)")
        north_pole = np.asarray(north_pole, dtype=float)
        if north_pole.ndim != 1 or north_pole.shape[0] != 3:
            raise ValueError("north_pole must have shape (3,)")
        if np.linalg.norm(north_pole) == 0:
            raise ValueError("north_pole must be non-zero")
        north_pole /= np.linalg.norm(north_pole)

        N = int(num_rays)
        if N <= 0:
            return cls(np.zeros((0, 3)), np.zeros((0, 3)))

        # Number of candidate points we generate (2N ensures we can take N points on +Z hemisphere)
        Nr = 2 * N if hemisphere_only else N
        i = np.arange(Nr)

        # Vogel/Fibonacci sphere parameters
        z = 1 - 2 * (i + 0.5) / Nr
        r = np.sqrt(np.maximum(0, 1 - z**2))
        phi = 2*np.pi * ((i / golden) % 1)

        if hemisphere_only:
            # Take the N points with z >= 0
            phi = phi[z >= 0]
            r = r[z >= 0]
            z = z[z >= 0]

            assert z.shape[0] == N

        D = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))

        # Broadcast origin to all rays
        O = np.repeat(O[None, :], D.shape[0], axis=0)

        # Rotate so +Z maps to north_pole
        pos_z = np.array([0.0, 0.0, 1.0])
        if np.allclose(north_pole, pos_z, atol=1e-7):
            # Already aligned, nothing to do
            return cls(O, D)
        elif np.allclose(north_pole, -pos_z, atol=1e-7):
            # Opposite: flip along Z
            D[:, 2] *= -1

            return cls(O, D)
        else:
            # N.B.: Using atol=1e-7 in the previous two checks means that the cross product's norm is guaranteed to be nonzero.
            c = np.dot(pos_z, north_pole)
            axis = np.cross(pos_z, north_pole)
            s = np.linalg.norm(axis)

            # Rodrigues' rotation formula
            # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
            ax, ay, az = axis / s
            K = np.array([[0, -az, ay],
                          [az, 0, -ax],
                          [-ay, ax, 0]])
            R = np.eye(3) + K * s + (K @ K) * (1 - c)

            D = D @ R.T

            return cls(O, D)

    def getNumRays(self) -> int:
        return int(self.D.shape[0])

    def getOrigins(self) -> np.ndarray:
        """Return Nx3 array of origins."""
        return self.O.copy()

    def getDirections(self) -> np.ndarray:
        """Return Nx3 array of directions."""
        return self.D.copy()

    def getTotalDistances(self) -> np.ndarray:
        """For each ray, returns the total travel distance in meters. NaN denotes invalid intersections."""
        return self.totalDistance.copy()

    def getDistances(self) -> Tuple[np.ndarray, np.ndarray]:
        """For each ray, returns the distance of the closest intersection (front and back). NaN denotes invalid intersections."""
        return self.frontDistance.copy(), self.backDistance.copy()

    def getCosines(self) -> Tuple[np.ndarray, np.ndarray]:
        """For each ray, returns the incidence cosine of the closest intersection (front and back). NaN denotes invalid intersections."""
        return self.frontCosine.copy(), self.backCosine.copy()

    def getIndices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (current, previous) patch indices per ray. -1 denotes invalid intersections."""
        return self.frontPatch.copy(), self.backPatch.copy()

    def getRadiance(self) -> np.ndarray:
        """Returns the per-ray radiance values."""
        return self.radiance.copy()

    def moveOrigins(self, origins: np.ndarray) -> None:
        """
        Change the rays' origins (if one origin is given, it is used for all rays).

        Parameters
        ----------
        origins : (M,3) float ndarray
            Per-ray origin points. If M==1, the value is used for all rays.
        """
        if origins.ndim < 1 or origins.ndim > 2 or origins.shape[-1] != 3:
            raise ValueError("origins must have shape (M, 3), and M must either be 1 or the number of rays")

        if origins.ndim == 1:
            self.O = np.repeat(origins, self.O.shape[0], axis=0)
        elif origins.shape[0] == 1:
            self.O = np.repeat(origins[0], self.O.shape[0], axis=0)
        elif origins.shape == self.O.shape:
            self.O = origins.copy()
        else:
            raise ValueError("origins must have shape (M, 3), and M must either be 1 or the number of rays.")

    def traceAll(self, triangles: TriangleMesh) -> None:
        """
        Find the next intersection point of each ray and update the intersected triangle indices,
        without advancing the rays.
        """
        M = self.getNumRays()
        N = triangles.size()
        if M == 0 or N == 0:
            return

        # 1) Facing test: faceNum = dot(n, O) - d0 -> (M,N)
        faceNum = np.einsum("nj,mj->mn", triangles.n, self.O) - triangles.d0[None, :]  # (M,N)
        face_ok = (faceNum > EPS_FACING)

        # 2) Möller–Trumbore (broadcasted over (M,N,3))
        D = self.D[:, None, :]  # (M,1,3)
        O = self.O[:, None, :]  # (M,1,3)
        v1 = triangles.v1[None, :, :]  # (1,N,3)
        edge1 = triangles.edge1[None, :, :]  # (1,N,3)
        edge2 = triangles.edge2[None, :, :]  # (1,N,3)

        pvec = np.cross(D, edge2)  # (M,N,3)
        det = np.einsum("mnj,mnj->mn", pvec, edge1)  # (M,N)

        tvec = O - v1  # (M,N,3)
        u_num = np.einsum("mnj,mnj->mn", tvec, pvec)  # (M,N)

        qvec = np.cross(tvec, edge1)  # (M,N,3)
        v_num = np.einsum("mj,mnj->mn", self.D, qvec)  # (M,N)

        w_num = det - (u_num + v_num)
        allNonNeg = (u_num >= -EPS_EDGE) & (v_num >= -EPS_EDGE) & (w_num >= -EPS_EDGE)
        allNonPos = (u_num <= +EPS_EDGE) & (v_num <= +EPS_EDGE) & (w_num <= +EPS_EDGE)
        edge_ok = (allNonNeg | allNonPos)

        not_parallel = (np.abs(det) > EPS_PARALLEL)
        valid = (face_ok & edge_ok & not_parallel)

        # Distances and cosines
        # TODO: This eigensum performs redundant operations.
        #       Invalid indices should be ignored BEFORE performing einsum, rather that ignoring invalid results.
        t_num = np.einsum("mnj,nj->mn", qvec, triangles.edge2)  # (M,N)
        dist = np.full((M, N), np.nan, dtype=float)
        dist[valid] = t_num[valid] / det[valid]

        cosv = np.full((M, N), np.nan, dtype=float)
        # |dot(n, D)| broadcast over (M,N)
        # TODO: This eigensum performs redundant operations.
        #       Invalid indices should be ignored BEFORE performing einsum, rather that ignoring invalid results.
        cosv[valid] = np.abs(np.einsum("nj,mj->mn", triangles.n, self.D)[valid])

        # TODO: The following section can be greatly simplified through a smart use of np.argwhere and np.argmin.
        idx = np.arange(N)[None, :].repeat(M, axis=0)  # (M,N)

        # FRONT selection: minimal positive distance; tie by lowest triangle index
        pos_mask = (dist > EPS_SELFHIT)
        pos_dist = np.where(pos_mask, dist, np.inf)
        min_pos = pos_dist.min(axis=1)  # (M,)
        tie_pos = pos_mask & (np.abs(dist - min_pos[:, None]) < EPS_ZFIGHT)
        cand_front = np.where(tie_pos, idx, N)
        i_front = cand_front.min(axis=1)  # (M,)

        # BACK selection: maximum negative distance (closest to 0); tie by lowest index
        neg_mask = (dist < -EPS_SELFHIT)
        neg_abs = np.where(neg_mask, -dist, np.inf)  # positive distances for negatives
        min_neg_abs = neg_abs.min(axis=1)            # (M,) equals smallest abs among negatives
        tie_back = neg_mask & (np.abs(-dist - min_neg_abs[:, None]) < EPS_ZFIGHT)
        cand_back = np.where(tie_back, idx, N)
        i_back = cand_back.min(axis=1)  # (M,)

        # TODO: This np.arange can be replaced by a smart use of np.take_along_axis. But would it be any faster?
        row = np.arange(M)

        front_ok = (i_front < N)
        self.frontPatch[front_ok] = triangles.ID[i_front[front_ok]]
        self.frontDistance[front_ok] = dist[row[front_ok], i_front[front_ok]]
        self.frontCosine[front_ok] = cosv[row[front_ok], i_front[front_ok]]
        self.frontPatch[~front_ok] = -1
        self.frontDistance[~front_ok] = np.nan
        self.frontCosine[~front_ok] = np.nan

        back_ok = (i_back < N)
        self.backPatch[back_ok] = triangles.ID[i_back[back_ok]]
        self.backDistance[back_ok] = -dist[row[back_ok], i_back[back_ok]]
        self.backCosine[back_ok] = cosv[row[back_ok], i_back[back_ok]]
        self.backPatch[~back_ok] = -1
        self.backDistance[~back_ok] = np.nan
        self.backCosine[~back_ok] = np.nan

    # TODO: Implement direction clustering

    # TODO: Implement "advance", "reflect", etc.
