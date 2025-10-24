"""
Python translation of TracingTypes.h/.cpp and TracingUtils.h/.cpp (with PLUCKER_KERNEL and LEAN_PLUCKER assumed False),
revised to store all x/y/z data as (N,3) arrays for efficient vectorized NumPy operations.
"""
import numpy as np
from scipy.constants import golden
from typing import Tuple

EPS_EDGE = 1e-5         # edge-inclusive tolerance for ray hits
EPS_FACING = 1e-5       # ray-plane test perpendicular tolerance
EPS_PARALLEL = 1e-5     # ray-plane test parallel tolerance
EPS_ZFIGHT = 1e-5       # tie-breaker window for Z-fighting
EPS_SELFHIT = 1e-7      # reject hits too close to the ray origin


class TriangleMesh:
    """
    Structure-of-Arrays (SoA) container for triangle meshes used by the tracing kernels.
    These are designed for using the Möller–Trumbore intersection algorithm.

    Each triangle i stores:
      - v1: first vertex A
      - edge1: B - A
      - edge2: C - A
      - n: unit surface normal = edge1 x edge2
      - d0: plane offset such that dot(n, X) - d0 = 0 on the triangle plane
      - ID: per-triangle patch identifier

    Notes
    -----
    - The intersection kernel enforces:
      dot(n, O) - d0 > EPS_FACING (triangle faces the ray origin), and
      barycentric coordinates within edges (edges inclusive).
      It does not enforce t > 0; the low-level test is line-triangle.
    - In case of near ties (Z-fighting), the lower triangle index wins.
    """

    def __init__(self, vertices: np.ndarray,
                 vert_triplets: np.ndarray,
                 patch_ids: np.ndarray):
        """
        Build the SoA representation from vertex and face lists.

        Parameters
        ----------
        vertices : (N, 3) array_like of float
            3D vertex coordinates.
        vert_triplets : (M, 3) array_like of int
            Vertex indices forming M triangles. Winding determines the normal
            orientation by the right-hand rule: n = (B - A) x (C - A).
        patch_ids : (M,) array_like of int
            Per-triangle patch identifier.

        Notes
        -----
        The stored normals are normalized to unit length. Triangle areas are
        stored in `area`, and d0 is computed as dot(n, v1).
        """
        # Validate inputs and force types
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
        nlen = np.linalg.norm(self.n, axis=1)
        if np.any(nlen == 0):
            raise ValueError("All faces must have nonzero area.")
        self.n /= nlen[:, None]

        self.area = 0.5 * nlen
        self.d0 = np.einsum("ij,ij->i", self.n, self.v1)

    def size(self) -> int:
        """
        Number of triangles in the mesh.

        Returns
        -------
        int
            The count of triangles (M).
        """
        return int(self.v1.shape[0])

    def sample_triangle(self, triangle_idx: int, points_per_square_meter: float) -> np.ndarray:
        """
        Quasi-Monte Carlo surface sampling of one triangle.

        A 2D lattice is constructed in a local orthonormal basis defined by
        two tangent vectors and the triangle normal. The lattice is rotated
        by 3*pi/8 (Rodrigues' formula) and tested against the target triangle
        in 2D; accepted samples are mapped back to 3D. If no lattice points
        fall inside, the centroid is used. This approach in inspired by the
        one proposed in: Kinjal Basu and Art B. Owen. "Low discrepancy
        constructions in the triangle." SIAM Journal on Numerical Analysis
        53.2 (2015): 743-761. However, here we test sample points against the
        target triangle. In the cited paper, the authors test sample points
        against the unit right triangle, and then apply a linear
        transformation to the target triangle.

        Parameters
        ----------
        triangle_idx : int
            Index of the triangle to sample.
        points_per_square_meter : float
            Target sampling density.

        Returns
        -------
        numpy.ndarray
            Array of shape (K, 3) containing 3D sample points on the triangle.
        """
        edge1_len = np.linalg.norm(self.edge1[triangle_idx])
        edge2_len = np.linalg.norm(self.edge2[triangle_idx])
        # The maximum extent for the 2D lattice (see below)
        grid_extent = max(edge1_len, edge2_len)

        # Start by using one edge of the triangle as a reference tangent vector
        tangent1 = self.edge1[triangle_idx] / edge1_len
        # Rotate the tangent by an irrational angle (3*pi/8) using Rodrigues' rotation formula
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
        theta = 3. * np.pi / 8.
        axis = self.n[triangle_idx]
        tangent1 = tangent1 * np.cos(theta) \
                   + np.cross(axis, tangent1) * np.sin(theta) \
                   + axis * np.dot(axis, tangent1) * (1 - np.cos(theta))

        # Find a second tangent vector orthogonal to the first
        tangent2 = np.cross(tangent1, self.n[triangle_idx])

        # Prepare the 2D lattice (this lives in the coordinate space defined by the two orthogonal tangents)
        sample_spacing = np.sqrt(1 / points_per_square_meter)
        lattice_2D = np.dstack(np.meshgrid(np.arange(-grid_extent, grid_extent, sample_spacing),
                                           np.arange(-grid_extent, grid_extent, sample_spacing)))
        lattice_2D = lattice_2D.reshape(-1, 2)

        # Translate the triangle's edges into the coordinate space defined by the two orthogonal tangents
        edge1_2D = np.array([tangent1.dot(self.edge1[triangle_idx]),
                             tangent2.dot(self.edge1[triangle_idx])])
        edge2_2D = np.array([tangent1.dot(self.edge2[triangle_idx]),
                             tangent2.dot(self.edge2[triangle_idx])])

        # Vectorized 2D point-in-triangle test (test whole lattice in one go)
        # https://stackoverflow.com/a/51479401
        #   x, y = lattice_2D
        #   ax, ay = (0, 0) (the reference vertex v1 is the origin of the lattice)
        #   bx, by = edge1_2D
        #   cx, cy = edge2_2D
        side_1 = np.cross(lattice_2D - edge1_2D, -edge1_2D)
        side_2 = np.cross(lattice_2D - edge2_2D, edge1_2D - edge2_2D)
        side_3 = np.cross(lattice_2D, edge2_2D)

        allNonNeg = (side_1 >= -EPS_EDGE) & (side_2 >= -EPS_EDGE) & (side_3 >= -EPS_EDGE)
        allNonPos = (side_1 <= +EPS_EDGE) & (side_2 <= +EPS_EDGE) & (side_3 <= +EPS_EDGE)
        within_edges = (allNonNeg | allNonPos)

        # Take only the lattice points which passed the test (inside triangle)
        sample_points_2D = list()
        for i, point in enumerate(lattice_2D):
            if within_edges[i]:
                sample_points_2D.append(point)
        if len(sample_points_2D) == 0:
            # No valid sample points: use the centroid.
            sample_points_2D = (edge1_2D + edge2_2D)[None] / 3
        else:
            sample_points_2D = np.array(sample_points_2D)

        # Translate back to 3D cartesian coordinates
        sample_points_3D = self.v1[triangle_idx] \
                           + sample_points_2D[:, 0, None] * tangent1[None, :] \
                           + sample_points_2D[:, 1, None] * tangent2[None, :]

        return sample_points_3D


# TODO: In "hemisphere mode", add two options:
#       - Expose the fact that it's a hemisphere (current behavior)
#       - Pretend it's a sphere, use hemisphere under the hood (like the C++ code does)
class RayBundle:
    """
    Bundle of rays with (possibly) separate per-ray origins and directions.

    Directions are normalized on construction. The instance stores per-ray
    bookkeeping used by the tracing kernel:
      - radiance
      - totalDistance
      - currentTriangle (for future self-hit handling)
      - frontDistance, frontCosine, frontPatch
      - backDistance, backCosine, backPatch

    Methods are provided to construct bundles, access internal arrays, move
    origins, and perform intersection queries against a TriangleMesh.
    """

    def __init__(self, O: np.ndarray, D: np.ndarray):
        """
        Construct a bundle from per-ray origins and directions.

        Parameters
        ----------
        O : (M, 3) array_like of float
            Per-ray origins.
        D : (M, 3) array_like of float
            Per-ray directions. They are normalized inside this constructor.
        """
        self.O = np.asarray(O, dtype=float)
        self.D = np.asarray(D, dtype=float)
        # Normalize directions
        self.D = self.D / np.linalg.norm(self.D, axis=1, keepdims=True)

        N = self.O.shape[0]
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
        Construct a bundle from one origin and many directions.

        Parameters
        ----------
        origin : (3,) array_like of float
            Shared origin for all rays.
        directions : (M, 3) array_like of float
            Ray directions (not necessarily normalized).

        Returns
        -------
        RayBundle
            A new bundle with repeated origins and normalized directions.

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
        Construct a bundle from per-ray origins and directions.

        Parameters
        ----------
        origins : (M, 3) array_like of float
            Per-ray origins.
        directions : (M, 3) array_like of float
            Per-ray directions (not necessarily normalized).

        Returns
        -------
        RayBundle
            A new bundle with normalized directions.

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
        Sample directions on a Fibonacci sphere and build a bundle.

        The generator creates `num_rays` approximately uniform directions.
        If `hemisphere_only` is True, it selects the +Z hemisphere before
        rotation. The +Z axis is then rotated so it aligns with `north_pole`
        using Rodrigues' formula. A shared `origin` is assigned to all rays.

        Parameters
        ----------
        num_rays : int
            Number of directions to generate.
        hemisphere_only : bool, default False
            If True, use only the +Z hemisphere before rotation.
        origin : (3,) array_like of float, default zeros(3)
            Shared origin for all rays.
        north_pole : (3,) array_like of float, default [0, 0, 1]
            Target direction for the +Z axis after rotation.

        Returns
        -------
        RayBundle
            A new bundle with normalized directions.

        Notes
        -----
        The number of generated directions is always `num_rays`, i.e.,
        the sampling density is doubled when `hemisphere_only` is True.
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

            # This should be ensured by the Fibonacci construction
            assert z.shape[0] == N

        D = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))

        # Broadcast origin to all rays
        O = np.repeat(O[None, :], D.shape[0], axis=0)

        # Rotate so +Z maps to north_pole
        pos_z = np.array([0.0, 0.0, 1.0])
        if np.allclose(north_pole, pos_z, atol=EPS_PARALLEL):
            # Already aligned, nothing to do
            return cls(O, D)
        elif np.allclose(north_pole, -pos_z, atol=EPS_PARALLEL):
            # Opposite: flip along Z
            D[:, 2] *= -1

            return cls(O, D)
        else:
            # N.B.: Using `atol` in the previous two checks means that the cross product's norm is guaranteed to be nonzero.
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
        """
        Number of rays in the bundle.

        Returns
        -------
        int
            The count of rays (M).
        """
        return int(self.D.shape[0])

    def getOrigins(self, copy: bool = True) -> np.ndarray:
        """
        Access current per-ray origins.

        Parameters
        ----------
        copy : bool, default True
            If True, return a copy; otherwise, return a view.

        Returns
        -------
        numpy.ndarray
            Array of shape (M, 3) with origins.
        """
        if copy:
            return self.O.copy()
        else:
            return self.O

    def getDirections(self, copy: bool = True) -> np.ndarray:
        """
        Access current per-ray directions.

        Parameters
        ----------
        copy : bool, default True
            If True, return a copy; otherwise, return a view.

        Returns
        -------
        numpy.ndarray
            Array of shape (M, 3) with directions.
        """
        if copy:
            return self.D.copy()
        else:
            return self.D

    def getRadiance(self, copy: bool = True) -> np.ndarray:
        """
        Access current per-ray radiance scalars.

        Parameters
        ----------
        copy : bool, default True
            If True, return a copy; otherwise, return a view.

        Returns
        -------
        numpy.ndarray
            Array of shape (M,) with radiance values.
        """
        if copy:
            return self.radiance.copy()

    def getTotalDistances(self, copy: bool = True) -> np.ndarray:
        """
        Access accumulated path lengths.

        Parameters
        ----------
        copy : bool, default True
            If True, return a copy; otherwise, return a view.

        Returns
        -------
        numpy.ndarray
            Array of shape (M,) with total distances.
        """
        if copy:
            return self.totalDistance.copy()
        else:
            return self.totalDistance

    def getDistances(self, copy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Access front and back intersection distances related to the latest trace.

        Parameters
        ----------
        copy : bool, default True
            If True, return copies; otherwise, return views.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Front and back distances, each of shape (M,).
            All distances are non-negative; back distances are stored as positive
            magnitudes for hits behind the origin.
        """
        if copy:
            return self.frontDistance.copy(), self.backDistance.copy()
        else:
            return self.frontDistance, self.backDistance

    def getCosines(self, copy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Access front and back departure cosines related to the latest trace.

        Parameters
        ----------
        copy : bool, default True
            If True, return copies; otherwise, return views.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Front and back cosines, each of shape (M,).
        """
        if copy:
            return self.frontCosine.copy(), self.backCosine.copy()
        else:
            return self.frontCosine, self.backCosine

    def getIndices(self, copy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Access front and back patch indices hit during the latest trace.

        Parameters
        ----------
        copy : bool, default True
            If True, return copies; otherwise, return views.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Front and back patch ids, each of shape (M,). A value of -1 marks no hit.
        """
        if copy:
            return self.frontPatch.copy(), self.backPatch.copy()
        else:
            return self.frontPatch, self.backPatch

    def moveOrigins(self, origins: np.ndarray) -> None:
        """
        Replace ray origins in bulk.

        If a single origin is provided, it is broadcast to all rays.

        Parameters
        ----------
        origins : array_like
            Either shape (3,) or (1, 3) to broadcast to all rays, or shape
            (M, 3) to set per-ray origins.
        """
        if origins.ndim < 1 or origins.ndim > 2 or origins.shape[-1] != 3:
            raise ValueError("origins must have shape (M, 3), and M must either be 1 or the number of rays")

        if origins.ndim == 1:
            self.O = np.repeat(origins[None, :], self.O.shape[0], axis=0)
        elif origins.shape[0] == 1:
            self.O = np.repeat(origins, self.O.shape[0], axis=0)
        elif origins.shape == self.O.shape:
            self.O = origins.copy()
        else:
            raise ValueError("origins must have shape (M, 3), and M must either be 1 or the number of rays.")

    def traceAll(self, triangles: TriangleMesh) -> None:
        """
        Trace all rays against a TriangleMesh and record nearest hits.

        For each ray, perform a facing test and a Möller-Trumbore triangle
        test against all triangles, with edge-inclusive tolerances. Update:
          - frontPatch/frontDistance/frontCosine with the minimal positive
            distance hit (ties resolved to the lowest triangle index),
          - backPatch/backDistance/backCosine with the negative-distance hit
            closest to the origin (again, lowest index on ties).
        Distances below EPS_SELFHIT are ignored.

        Parameters
        ----------
        triangles : TriangleMesh
            Mesh to intersect against.

        Notes
        -----
        This method does not advance rays or update totalDistance.
        """
        M = self.getNumRays()
        N = triangles.size()
        if M == 0 or N == 0:
            return

        # Facing test: faceNum = dot(n, O) - d0 > 0
        faceNum = np.einsum("nj,mj->mn", triangles.n, self.O) - triangles.d0[None, :]  # (M,N)
        face_ok = (faceNum > EPS_FACING)

        # Möller–Trumbore (broadcasted over (M,N,3))
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

        # TODO: The following section can probably be simplified through a smart use of np.argwhere and np.argmin.
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

        # TODO: This np.arange can probably be replaced by a smart use of np.take_along_axis. But would it be any faster?
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
