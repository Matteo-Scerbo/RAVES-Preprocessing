"""
Python translation of TracingTypes.h/.cpp and TracingUtils.h/.cpp (with PLUCKER_KERNEL and LEAN_PLUCKER assumed False),
revised to store all x/y/z data as (N,3) arrays for efficient vectorized NumPy operations.
"""
import numpy as np
from scipy.constants import golden
from typing import Tuple

EPS_EDGE = 1e-5  # edge-inclusive tolerance for ray hits
EPS_FACING = 1e-5  # ray-plane test perpendicular tolerance
EPS_PARALLEL = 1e-5  # ray-plane test parallel tolerance
EPS_ZFIGHT = 1e-5  # tie-breaker window for Z-fighting
EPS_SELFHIT = 1e-7  # reject hits too close to the ray origin


class TriangleMesh:
    """
    Structure-of-Arrays (SoA) container for triangle meshes used by the tracing kernels.
    These are designed for using the Möller–Trumbore intersection algorithm.

    Each triangle i stores:
      - v_1: first vertex A
      - edge_1: B - A
      - edge_2: C - A
      - n: unit surface normal = edge_1 x edge_2
      - d_0: plane offset such that dot(n, X) - d_0 = 0 on the triangle plane
      - patch_ids: per-triangle patch identifier

    Notes
    -----
    - The intersection kernel enforces:
      dot(n, origins) - d_0 > EPS_FACING (triangle faces the ray origin), and
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
        stored in `area`, and d_0 is computed as dot(n, v_1).
        """
        # Validate inputs and force types
        v = np.asarray(vertices, dtype=float)
        f = np.asarray(vert_triplets, dtype=int)
        self.patch_ids = np.asarray(patch_ids, dtype=int)

        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        if self.patch_ids.ndim != 1 or self.patch_ids.shape[0] != f.shape[0]:
            raise ValueError("patch_ids must have shape (M,) matching faces.shape[0]")

        if f.min() < 0 or f.max() >= v.shape[0]:
            raise IndexError("faces contain vertex indices out of range for `vertices`")

        self.v_1 = v[f[:, 0]]
        self.edge_1 = v[f[:, 1]] - self.v_1
        self.edge_2 = v[f[:, 2]] - self.v_1

        self.n = np.cross(self.edge_1, self.edge_2)
        nlen = np.linalg.norm(self.n, axis=1)
        if np.any(nlen == 0):
            raise ValueError("All faces must have nonzero area.")
        self.n /= nlen[:, None]

        self.area = 0.5 * nlen
        self.d_0 = np.einsum("ij,ij->i", self.n, self.v_1)

    def size(self) -> int:
        """
        Number of triangles in the mesh.

        Returns
        -------
        int
            The count of triangles (M).
        """
        return int(self.v_1.shape[0])

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
        edge_1_len = np.linalg.norm(self.edge_1[triangle_idx])
        edge_2_len = np.linalg.norm(self.edge_2[triangle_idx])
        # The maximum extent for the 2D lattice (see below)
        grid_extent = max(edge_1_len, edge_2_len)

        # Start by using one edge of the triangle as a reference tangent vector
        tangent1 = self.edge_1[triangle_idx] / edge_1_len
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
        edge_1_2D = np.array([tangent1.dot(self.edge_1[triangle_idx]),
                              tangent2.dot(self.edge_1[triangle_idx])])
        edge_2_2D = np.array([tangent1.dot(self.edge_2[triangle_idx]),
                              tangent2.dot(self.edge_2[triangle_idx])])

        # Vectorized 2D point-in-triangle test (test whole lattice in one go)
        # https://stackoverflow.com/a/51479401
        #   x, y = lattice_2D
        #   ax, ay = (0, 0) (the reference vertex v_1 is the origin of the lattice)
        #   bx, by = edge_1_2D
        #   cx, cy = edge_2_2D
        side_1 = np.cross(lattice_2D - edge_1_2D, -edge_1_2D)
        side_2 = np.cross(lattice_2D - edge_2_2D, edge_1_2D - edge_2_2D)
        side_3 = np.cross(lattice_2D, edge_2_2D)

        all_non_neg = (side_1 >= -EPS_EDGE) & (side_2 >= -EPS_EDGE) & (side_3 >= -EPS_EDGE)
        all_non_pos = (side_1 <= +EPS_EDGE) & (side_2 <= +EPS_EDGE) & (side_3 <= +EPS_EDGE)
        within_edges = (all_non_neg | all_non_pos)

        # Take only the lattice points which passed the test (inside triangle)
        sample_points_2D = list()
        for i, point in enumerate(lattice_2D):
            if within_edges[i]:
                sample_points_2D.append(point)
        if len(sample_points_2D) == 0:
            # No valid sample points: use the centroid.
            sample_points_2D = (edge_1_2D + edge_2_2D)[None] / 3
        else:
            sample_points_2D = np.array(sample_points_2D)

        # Translate back to 3D cartesian coordinates
        sample_points_3D = self.v_1[triangle_idx] \
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
      - total_distance
      - current_triangle (for future self-hit handling)
      - front_distance, front_cosine, front_patch
      - back_distance, back_cosine, back_patch

    Methods are provided to construct bundles, access internal arrays, move
    origins, and perform intersection queries against a TriangleMesh.
    """

    def __init__(self, origins: np.ndarray, directions: np.ndarray):
        """
        Construct a bundle from per-ray origins and directions.

        Parameters
        ----------
        origins : (M, 3) array_like of float
            Per-ray origins.
        directions : (M, 3) array_like of float
            Per-ray directions. They are normalized inside this constructor.
        """
        self.origins = np.asarray(origins, dtype=float)
        self.directions = np.asarray(directions, dtype=float)
        # Normalize directions
        self.directions = self.directions / np.linalg.norm(self.directions, axis=1, keepdims=True)

        n = self.origins.shape[0]
        self.radiance = np.ones(n)
        self.total_distance = np.zeros(n)

        # TODO: Use current_triangle to avoid self-hits
        self.current_triangle = np.full(n, -1, dtype=int)

        self.front_distance = np.full(n, np.nan, dtype=float)
        self.front_cosine = np.full(n, np.nan, dtype=float)
        self.front_patch = np.full(n, -1, dtype=int)

        self.back_distance = np.full(n, np.nan, dtype=float)
        self.back_cosine = np.full(n, np.nan, dtype=float)
        self.back_patch = np.full(n, -1, dtype=int)

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
        origins = np.asarray(origin, dtype=float)
        directions = np.asarray(directions, dtype=float)

        if origins.ndim != 1 or origins.shape[0] != 3:
            raise ValueError("origin must have shape (3,)")

        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions must have shape (M, 3)")

        # Broadcast origin to all rays
        origins = np.repeat(origins[None, :], directions.shape[0], axis=0)

        return cls(origins, directions)

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
        origins = np.asarray(origins, dtype=float)
        directions = np.asarray(directions, dtype=float)

        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins must have shape (M, 3)")
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions must have shape (M, 3)")
        if origins.shape[0] != directions.shape[0]:
            raise NotImplementedError("origins and directions must have the same number of rows (M)."
                                      " TODO: allow using this method to construct several pencils.")

        return cls(origins, directions)

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
        origins = np.asarray(origin, dtype=float)
        if origins.ndim != 1 or origins.shape[0] != 3:
            raise ValueError("origin must have shape (3,)")
        north_pole = np.asarray(north_pole, dtype=float)
        if north_pole.ndim != 1 or north_pole.shape[0] != 3:
            raise ValueError("north_pole must have shape (3,)")
        if np.linalg.norm(north_pole) == 0:
            raise ValueError("north_pole must be non-zero")
        north_pole /= np.linalg.norm(north_pole)

        n = int(num_rays)
        if n <= 0:
            return cls(np.zeros((0, 3)), np.zeros((0, 3)))

        # Number of candidate points we generate (2N ensures we can take N points on +Z hemisphere)
        n_z = 2 * n if hemisphere_only else n
        i = np.arange(n_z)

        # Vogel/Fibonacci sphere parameters
        z = 1 - 2 * (i + 0.5) / n_z
        r = np.sqrt(np.maximum(0, 1 - z ** 2))
        phi = 2 * np.pi * ((i / golden) % 1)

        if hemisphere_only:
            # Take the N points with z >= 0
            phi = phi[z >= 0]
            r = r[z >= 0]
            z = z[z >= 0]

            # This should be ensured by the Fibonacci construction
            assert z.shape[0] == n

        directions = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))

        # Broadcast origin to all rays
        origins = np.repeat(origins[None, :], directions.shape[0], axis=0)

        # Rotate so +Z maps to north_pole
        pos_z = np.array([0.0, 0.0, 1.0])
        if np.allclose(north_pole, pos_z, atol=EPS_PARALLEL):
            # Already aligned, nothing to do
            return cls(origins, directions)
        elif np.allclose(north_pole, -pos_z, atol=EPS_PARALLEL):
            # Opposite: flip along Z
            directions[:, 2] *= -1

            return cls(origins, directions)
        else:
            # N.B.: Using `atol` in the previous two checks means that the cross product's norm is guaranteed to be nonzero.
            c = np.dot(pos_z, north_pole)
            axis = np.cross(pos_z, north_pole)
            s = np.linalg.norm(axis)

            # Rodrigues' rotation formula
            # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
            ax, ay, az = axis / s
            k = np.array([[0, -az, ay],
                          [az, 0, -ax],
                          [-ay, ax, 0]])
            r = np.eye(3) + k * s + (k @ k) * (1 - c)

            directions = directions @ r.T

            return cls(origins, directions)

    def get_num_rays(self) -> int:
        """
        Number of rays in the bundle.

        Returns
        -------
        int
            The count of rays (M).
        """
        return int(self.directions.shape[0])

    def get_origins(self, copy: bool = True) -> np.ndarray:
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
            return self.origins.copy()
        else:
            return self.origins

    def get_directions(self, copy: bool = True) -> np.ndarray:
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
            return self.directions.copy()
        else:
            return self.directions

    def get_radiance(self, copy: bool = True) -> np.ndarray:
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

    def get_total_distances(self, copy: bool = True) -> np.ndarray:
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
            return self.total_distance.copy()
        else:
            return self.total_distance

    def get_distances(self, copy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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
            return self.front_distance.copy(), self.back_distance.copy()
        else:
            return self.front_distance, self.back_distance

    def get_cosines(self, copy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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
            return self.front_cosine.copy(), self.back_cosine.copy()
        else:
            return self.front_cosine, self.back_cosine

    def get_indices(self, copy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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
            return self.front_patch.copy(), self.back_patch.copy()
        else:
            return self.front_patch, self.back_patch

    def move_origins(self, origins: np.ndarray) -> None:
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
            self.origins = np.repeat(origins[None, :], self.origins.shape[0], axis=0)
        elif origins.shape[0] == 1:
            self.origins = np.repeat(origins, self.origins.shape[0], axis=0)
        elif origins.shape == self.origins.shape:
            self.origins = origins.copy()
        else:
            raise ValueError("origins must have shape (M, 3), and M must either be 1 or the number of rays.")

    def trace_all(self, triangles: TriangleMesh) -> None:
        """
        Trace all rays against a TriangleMesh and record nearest hits.

        For each ray, perform a facing test and a Möller-Trumbore triangle
        test against all triangles, with edge-inclusive tolerances. Update:
          - front_patch/front_distance/front_cosine with the minimal positive
            distance hit (ties resolved to the lowest triangle index),
          - back_patch/back_distance/back_cosine with the negative-distance hit
            closest to the origin (again, lowest index on ties).
        Distances below EPS_SELFHIT are ignored.

        Parameters
        ----------
        triangles : TriangleMesh
            Mesh to intersect against.

        Notes
        -----
        This method does not advance rays or update total_distance.
        """
        m = self.get_num_rays()
        n = triangles.size()
        if m == 0 or n == 0:
            return

        # Facing test: faceNum = dot(n, origins) - d_0 > 0
        face_num = np.einsum("nj,mj->mn", triangles.n, self.origins) - triangles.d_0[None, :]  # (M,N)
        face_ok = (face_num > EPS_FACING)

        # Möller–Trumbore (broadcasted over (M,N,3))
        directions = self.directions[:, None, :]  # (M,1,3)
        origins = self.origins[:, None, :]  # (M,1,3)
        v_1 = triangles.v_1[None, :, :]  # (1,N,3)
        edge_1 = triangles.edge_1[None, :, :]  # (1,N,3)
        edge_2 = triangles.edge_2[None, :, :]  # (1,N,3)

        pvec = np.cross(directions, edge_2)  # (M,N,3)
        det = np.einsum("mnj,mnj->mn", pvec, edge_1)  # (M,N)

        tvec = origins - v_1  # (M,N,3)
        u_num = np.einsum("mnj,mnj->mn", tvec, pvec)  # (M,N)

        qvec = np.cross(tvec, edge_1)  # (M,N,3)
        v_num = np.einsum("mj,mnj->mn", self.directions, qvec)  # (M,N)

        w_num = det - (u_num + v_num)
        all_non_neg = (u_num >= -EPS_EDGE) & (v_num >= -EPS_EDGE) & (w_num >= -EPS_EDGE)
        all_non_pos = (u_num <= +EPS_EDGE) & (v_num <= +EPS_EDGE) & (w_num <= +EPS_EDGE)
        edge_ok = (all_non_neg | all_non_pos)

        not_parallel = (np.abs(det) > EPS_PARALLEL)
        valid = (face_ok & edge_ok & not_parallel)

        # Distances and cosines
        # TODO: This eigensum performs redundant operations.
        #       Invalid indices should be ignored BEFORE performing einsum, rather that ignoring invalid results.
        t_num = np.einsum("mnj,nj->mn", qvec, triangles.edge_2)  # (M,N)
        dist = np.full((m, n), np.nan, dtype=float)
        dist[valid] = t_num[valid] / det[valid]

        cosv = np.full((m, n), np.nan, dtype=float)
        # |dot(n, directions)| broadcast over (M,N)
        # TODO: This eigensum performs redundant operations.
        #       Invalid indices should be ignored BEFORE performing einsum, rather that ignoring invalid results.
        cosv[valid] = np.abs(np.einsum("nj,mj->mn", triangles.n, self.directions)[valid])

        # TODO: The following section can probably be simplified through a smart use of np.argwhere and np.argmin.
        idx = np.arange(n)[None, :].repeat(m, axis=0)  # (M,N)

        # FRONT selection: minimal positive distance; tie by lowest triangle index
        pos_mask = (dist > EPS_SELFHIT)
        pos_dist = np.where(pos_mask, dist, np.inf)
        min_pos = pos_dist.min(axis=1)  # (M,)
        tie_pos = pos_mask & (np.abs(dist - min_pos[:, None]) < EPS_ZFIGHT)
        cand_front = np.where(tie_pos, idx, n)
        i_front = cand_front.min(axis=1)  # (M,)

        # BACK selection: maximum negative distance (closest to 0); tie by lowest index
        neg_mask = (dist < -EPS_SELFHIT)
        neg_abs = np.where(neg_mask, -dist, np.inf)  # positive distances for negatives
        min_neg_abs = neg_abs.min(axis=1)  # (M,) equals smallest abs among negatives
        tie_back = neg_mask & (np.abs(-dist - min_neg_abs[:, None]) < EPS_ZFIGHT)
        cand_back = np.where(tie_back, idx, n)
        i_back = cand_back.min(axis=1)  # (M,)

        # TODO: This np.arange can probably be replaced by a smart use of np.take_along_axis. But would it be any faster?
        row = np.arange(m)

        front_ok = (i_front < n)
        self.front_patch[front_ok] = triangles.patch_ids[i_front[front_ok]]
        self.front_distance[front_ok] = dist[row[front_ok], i_front[front_ok]]
        self.front_cosine[front_ok] = cosv[row[front_ok], i_front[front_ok]]
        self.front_patch[~front_ok] = -1
        self.front_distance[~front_ok] = np.nan
        self.front_cosine[~front_ok] = np.nan

        back_ok = (i_back < n)
        self.back_patch[back_ok] = triangles.patch_ids[i_back[back_ok]]
        self.back_distance[back_ok] = -dist[row[back_ok], i_back[back_ok]]
        self.back_cosine[back_ok] = cosv[row[back_ok], i_back[back_ok]]
        self.back_patch[~back_ok] = -1
        self.back_distance[~back_ok] = np.nan
        self.back_cosine[~back_ok] = np.nan

    # TODO: Implement direction clustering

    # TODO: Implement "advance", "reflect", etc.
