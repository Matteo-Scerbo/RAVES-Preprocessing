"""
@brief Python translation of UnitTest_TracingTypes.cpp and UnitTest_TracingClasses.cpp,
adapted to use the Python TriangleMesh and RayBundle in raytracing.py.
"""
import numpy as np
from scipy.constants import golden
import unittest

from .raves_io import load_mesh
from .raytracing import TriangleMesh, RayBundle, EPS_FACING, EPS_SELFHIT

EXPECTED_DIST_PAIR = np.array([
    [  # origin index 0
        [1., 1.],  # dir 0
        [1., np.nan],  # dir 1
        [1., 1.],  # dir 2
        [1., 1.],  # dir 3
        [np.nan, np.nan],  # dir 4
        [2., np.nan],  # dir 5
    ],
    [  # origin index 1
        [1., 1.],  # dir 0
        [1., np.nan],  # dir 1
        [1., 1.],  # dir 2
        [1., 1.],  # dir 3
        [np.nan, np.nan],  # dir 4
        [2., np.nan],  # dir 5
    ],
], dtype=float)

EXPECTED_IDX_PAIR = np.array([
    [  # origin index 0
        [0, 2],  # dir 0
        [0, -1],  # dir 1
        [0, 2],  # dir 2
        [2, 0],  # dir 3
        [-1, -1],  # dir 4
        [5, -1],  # dir 5
    ],
    [  # origin index 1
        [7, 9],  # dir 0
        [7, -1],  # dir 1
        [7, 9],  # dir 2
        [9, 7],  # dir 3
        [-1, -1],  # dir 4
        [12, -1],  # dir 5
    ],
], dtype=int)


def build_single_triangle(z, up_normal):
    # Right triangle in the z = const plane with +Z normal
    V = np.array([
        [1.0, 0.0, z],
        [0.0, 1.0, z],
        [0.0, 0.0, z],
    ])
    if up_normal:
        F = np.array([[0, 1, 2]], dtype=int)
    else:
        F = np.array([[0, 2, 1]], dtype=int)
    P = np.array([0], dtype=int)
    return TriangleMesh(V, F, P)


def build_unit_cube(outward):
    V = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 0..3  (z=0)
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # 4..7  (z=1)
    ])

    # 12 triangles, outward normals via right-hand rule
    F = np.array([
        # z = 0 (normal -Z)
        [0, 2, 1], [0, 3, 2],
        # z = 1 (normal +Z)
        [4, 5, 6], [4, 6, 7],
        # y = 0 (normal -Y)
        [0, 1, 5], [0, 5, 4],
        # y = 1 (normal +Y)
        [3, 7, 6], [3, 6, 2],
        # x = 0 (normal -X)
        [0, 4, 7], [0, 7, 3],
        # x = 1 (normal +X)
        [1, 2, 6], [1, 6, 5],
    ], dtype=int)

    if not outward:
        # flip winding of every triangle to invert normals
        F = F[:, [0, 2, 1]]

    P = np.arange(F.shape[0], dtype=int)
    return TriangleMesh(V, F, P)


def build_test_mesh():
    # Build a room containing an assortment of triangles (same vertex data as C++ tests)
    vertices = np.zeros((42, 3), dtype=float)
    vertices[0] = [0.0, 0.0, -1.0]
    vertices[1] = [1.0, 0.0, -1.0]
    vertices[2] = [0.0, 1.0, -1.0]
    vertices[3] = [0.0, 0.0, -2.0]
    vertices[4] = [2.0, 0.0, -2.0]
    vertices[5] = [0.0, 2.0, -2.0]
    vertices[6] = [0.0, 0.0, 1.0]
    vertices[7] = [0.0, 1.0, 1.0]
    vertices[8] = [1.0, 0.0, 1.0]
    vertices[9] = [0.0, 0.0, 2.0]
    vertices[10] = [0.0, 2.0, 2.0]
    vertices[11] = [2.0, 0.0, 2.0]
    vertices[12] = [0.0, 0.0, -3.0]
    vertices[13] = [1.0, 0.0, -3.0]
    vertices[14] = [0.0, 1.0, -3.0]
    vertices[15] = [1.2, 1.5, -0.7]
    vertices[16] = [2.0, 1.5, -0.7]
    vertices[17] = [1.2, 1.5, 1.3]
    vertices[18] = [0.0, 0.0, 0.0]
    vertices[19] = [1.0, 0.0, 0.0]
    vertices[20] = [0.0, 1.0, 0.0]
    vertices[21] = [0.0, 0.0, 9.0]
    vertices[22] = [1.0, 0.0, 9.0]
    vertices[23] = [0.0, 1.0, 9.0]
    vertices[24] = [0.0, 0.0, 8.0]
    vertices[25] = [2.0, 0.0, 8.0]
    vertices[26] = [0.0, 2.0, 8.0]
    vertices[27] = [0.0, 0.0, 11.0]
    vertices[28] = [0.0, 1.0, 11.0]
    vertices[29] = [1.0, 0.0, 11.0]
    vertices[30] = [0.0, 0.0, 12.0]
    vertices[31] = [0.0, 2.0, 12.0]
    vertices[32] = [2.0, 0.0, 12.0]
    vertices[33] = [0.0, 0.0, 7.0]
    vertices[34] = [1.0, 0.0, 7.0]
    vertices[35] = [0.0, 1.0, 7.0]
    vertices[36] = [1.2, 1.5, 9.3]
    vertices[37] = [2.0, 1.5, 9.3]
    vertices[38] = [1.2, 1.5, 11.3]
    vertices[39] = [0.0, 0.0, 10.0]
    vertices[40] = [1.0, 0.0, 10.0]
    vertices[41] = [0.0, 1.0, 10.0]

    vert_triplets = np.zeros((14, 3), dtype=int)
    patch_ids = np.zeros(14, dtype=int)
    for i in range(0, 40, 3):
        for j in range(3):
            vert_triplets[int(i/3), j] = i+j
        patch_ids[int(i/3)] = int(i/3)

    mesh = TriangleMesh(vertices, vert_triplets, patch_ids)
    assert mesh.size() == 14
    return mesh


# PlatonicVertices helper exactly as in C++ unit tests (ported)
def platonic_vertices(n):
    if n == 1:
        verts = np.array([[0.0, 0.0, 1.0]], dtype=float)
    elif n == 2:
        verts = np.array([[0.0, 0.0, 1.0],
                          [0.0, 0.0, -1.0]], dtype=float)
    elif n == 3:
        verts = np.zeros((3, 3), dtype=float)
        for k in range(3):
            theta = 2.0 * np.pi * k / 3.0
            verts[k] = [np.cos(theta), np.sin(theta), 0.0]
    elif n == 4:
        s = 1 / np.sqrt(3)
        verts = np.array([[s, s, s],
                          [s, -s, -s],
                          [-s, s, -s],
                          [-s, -s, s]], dtype=float)
    elif n == 6:
        verts = np.array([[1.0, 0.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, -1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, 0.0, -1.0]], dtype=float)
    elif n == 8:
        s = 1 / np.sqrt(3)
        verts = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    verts.append([sx * s, sy * s, sz * s])
        verts = np.asarray(verts, dtype=float)
    elif n == 12:
        r = np.sqrt(1.0 + golden * golden)
        s1 = 1.0 / r
        sP = golden / r
        verts = []
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append([0.0, sy * s1, sz * sP])
        for sx in (-1, 1):
            for sy in (-1, 1):
                verts.append([sx * s1, sy * sP, 0.0])
        for sx in (-1, 1):
            for sz in (-1, 1):
                verts.append([sx * sP, 0.0, sz * s1])
        verts = np.asarray(verts, dtype=float)
    elif n == 20:
        s = 1 / np.sqrt(3)
        verts = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    verts.append([sx * s, sy * s, sz * s])
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append([0.0, (1.0 / golden) * sy * s, golden * sz * s])
        for sx in (-1, 1):
            for sy in (-1, 1):
                verts.append([(1.0 / golden) * sx * s, golden * sy * s, 0.0])
        for sx in (-1, 1):
            for sz in (-1, 1):
                verts.append([golden * sx * s, 0.0, (1.0 / golden) * sz * s])
        verts = np.asarray(verts, dtype=float)
    else:
        verts = np.zeros((0, 3), dtype=float)

    return verts


class PlaneTests(unittest.TestCase):
    def test_plane_parameters_consistency(self):
        mesh, _, _ = load_mesh('../../example environments/AudioForGames_20_patches')

        # For any triangle, the plane identity dot(n, v1) - d0 must be ~0 if (n, d0) are consistent.
        residual = np.einsum("ij,ij->i", mesh.n, mesh.v1) - mesh.d0

        self.assertTrue(np.all(np.abs(residual) < EPS_FACING),
                        msg='Plane (n, d0) inconsistent; max residual = ' + str(np.max(np.abs(residual))))

    def test_origin_side_vs_direction(self):
        for plane_z in [-3., 0., 3.]:
            for triangle_normal in [1, -1]:
                test_triangle = build_single_triangle(z=plane_z,
                                                      up_normal=(triangle_normal > 0))
                for ray_origin_position in [1, -1]:
                    for ray_orientation in [1, -1]:
                        test_ray = RayBundle.from_shared_origin(origin=np.array([0.25, 0.25, plane_z + ray_origin_position]),
                                                                directions=ray_orientation * np.array([[0., 0., 1.]]))

                        test_ray.traceAll(test_triangle)
                        front, back = test_ray.getIndices()

                        msg = 'Triangle normal ' + str(triangle_normal) + \
                              ', ray origin ' + str(ray_origin_position) + \
                              ', ray orientation ' + str(ray_orientation)

                        if triangle_normal * ray_origin_position < 0:
                            self.assertEqual(front, -1, msg=msg + '. Ray should NOT hit in the front.')
                            self.assertEqual(back, -1, msg=msg + '. Ray should NOT hit in the back.')
                        elif ray_origin_position * ray_orientation > 0:
                            self.assertEqual(front, -1, msg=msg + '. Ray should NOT hit in the front.')
                            self.assertNotEqual(back, -1, msg=msg + '. Ray should hit in the back.')
                        else:
                            self.assertNotEqual(front, -1, msg=msg + '. Ray should NOT hit in the front.')
                            self.assertEqual(back, -1, msg=msg + '. Ray should hit in the back.')

    def test_cube_hits_depend_on_normal(self):
        """
        For a watertight inward-facing cube, tracing from the center should find intersections in both directions.
        If the cube is outward-facing, tracing from the center should find no intersections in either direction.
        """
        N = 1000

        # Inward normals
        cube_out = build_unit_cube(outward=False)
        test_rays = RayBundle.sample_sphere(N, origin=np.array([0.5, 0.5, 0.5]))
        test_rays.traceAll(cube_out)
        front, back = test_rays.getIndices()

        self.assertEqual(int(np.count_nonzero(front == -1)), 0,
                         msg='There should be no invalid front hits from within an inward-facing cube.')
        self.assertEqual(int(np.count_nonzero(back == -1)), 0,
                         msg='There should be no invalid back hits from within an inward-facing cube.')

        # Outward normals
        cube_out = build_unit_cube(outward=True)
        test_rays = RayBundle.sample_sphere(N, origin=np.array([0.5, 0.5, 0.5]))
        test_rays.traceAll(cube_out)
        front, back = test_rays.getIndices()

        self.assertEqual(int(np.count_nonzero(front == -1)), N,
                         msg='There should be no valid front hits from within an outward-facing cube.')
        self.assertEqual(int(np.count_nonzero(back == -1)), N,
                         msg='There should be no valid back hits from within an outward-facing cube.')


class TracingClassesTests(unittest.TestCase):
    def test_pencil_tracing(self):
        testMesh = build_test_mesh()

        self.assertTrue(np.allclose(np.linalg.norm(testMesh.n, axis=1), 1.0),
                        msg='\n' + str(np.linalg.norm(testMesh.n)))

        testDirections = np.array([
            [0.0, 0.0, -1.0],
            [0.4, -0.1, -1.0],
            [-0.1, -0.1, -1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [10.0, 10.0, 1.0],
        ], dtype=float)

        testOrigins = np.array([
            [0.1, 0.1, 1e-6],
            [0.1, 0.1, 10.0 + 1e-6],
        ], dtype=float)

        for oi in range(testOrigins.shape[0]):
            testPencil = RayBundle.from_shared_origin(testOrigins[oi], testDirections)

            # Check that directions are automatically normalized on construction.
            self.assertTrue(np.allclose(np.linalg.norm(testPencil.getDirections(), axis=1), 1.0),
                            msg='\n' + str(np.linalg.norm(testPencil.getDirections())))

            testPencil.traceAll(testMesh)
            frontDistances, backDistances = testPencil.getDistances()
            # TODO: frontCosines, backCosines = testPencil.getCosines()
            frontIndices, backIndices = testPencil.getIndices()

            # Either both or neither distance should be NaN.
            self.assertTrue(np.all(np.isnan(EXPECTED_DIST_PAIR[oi, :, 0]) == np.isnan(frontDistances)),
                            msg='\n' + str(EXPECTED_DIST_PAIR[oi, :, 0]) + '\n' + str(frontDistances))
            # Either both or neither distance should be finite.
            self.assertTrue(np.all(np.isnan(EXPECTED_DIST_PAIR[oi, :, 0]) == ~np.isfinite(frontDistances)),
                            msg='\n' + str(EXPECTED_DIST_PAIR[oi, :, 0]) + '\n' + str(frontDistances))
            # The distances have the same sign (if neither is NaN).
            self.assertTrue(np.all((np.sign(EXPECTED_DIST_PAIR[oi, :, 0]) == np.sign(frontDistances))
                                   | np.isnan(EXPECTED_DIST_PAIR[oi, :, 0])),
                            msg='\n' + str(EXPECTED_DIST_PAIR[oi, :, 0]) + '\n' + str(frontDistances))
            # The triangle indices should match expectations.
            self.assertTrue(np.all(EXPECTED_IDX_PAIR[oi, :, 0] == frontIndices),
                            msg='\n' + str(EXPECTED_IDX_PAIR[oi, :, 0]) + '\n' + str(frontIndices))
            self.assertTrue(np.all(EXPECTED_IDX_PAIR[oi, :, 1] == backIndices),
                            msg='\n' + str(EXPECTED_IDX_PAIR[oi, :, 1]) + '\n' + str(backIndices))

            # DONE: Test construction with different origins
            # TODO: Test moving to different origins

    def test_pencil_sphere(self):
        for numRays in np.logspace(2, 5, 4, dtype=int):
            for numClusters in [1, 2, 3, 4, 6, 8, 12, 20]:
                testClusters = platonic_vertices(numClusters)
                self.assertTrue(np.allclose(np.linalg.norm(testClusters, axis=1), 1.0),
                                msg='\n' + str(np.linalg.norm(testClusters)))

                testPencil = RayBundle.sample_sphere(numRays)

                effectiveNumRays = testPencil.getNumRays()
                self.assertEqual(numRays, effectiveNumRays, msg='\n' + str(numRays) + ' != ' + str(effectiveNumRays))

                # Check that directions are automatically normalized on construction.
                self.assertTrue(np.allclose(np.linalg.norm(testPencil.getDirections(), axis=1), 1.0),
                                msg='\n' + str(np.linalg.norm(testPencil.getDirections())))

                cosineSimilarities = np.einsum("nj,mj->nm", testClusters, testPencil.getDirections())
                _, clusters = np.unique(np.argmax(cosineSimilarities, axis=0), return_counts=True)

                # Check that the difference between the minimum and maximum cluster size is relatively small.
                self.assertTrue(np.max(clusters) - np.min(clusters) <= int(effectiveNumRays / 10),
                                msg='\n' + str(clusters))

    def test_pencil_hemisphere(self):
        for numRays in np.logspace(2, 5, 4, dtype=int):
            for northPole in platonic_vertices(20):
                testPencil = RayBundle.sample_sphere(numRays, hemisphere_only=True, north_pole=northPole)

                effectiveNumRays = testPencil.getNumRays()
                self.assertEqual(numRays, effectiveNumRays, msg='\n' + str(numRays) + ' != ' + str(effectiveNumRays))

                # Check that directions are automatically normalized on construction.
                self.assertTrue(np.allclose(np.linalg.norm(testPencil.getDirections(), axis=1), 1.0),
                                msg='\n' + str(np.linalg.norm(testPencil.getDirections())))

                # Test north_pole rotation
                cosineSimilarities = np.einsum("j,mj->m", northPole, testPencil.getDirections())
                self.assertTrue(np.all(cosineSimilarities >= 0.0),
                                msg='\n' + str(cosineSimilarities))

    def test_visibility_in_volume(self):
        """
        Load a triangle mesh which is known to be closed, trace rays (uniform sphere) from points inside it,
        and assert that all rays find valid intersections in the front and back.
        """
        mesh, _, _ = load_mesh('../../example environments/AudioForGames_20_patches')

        num_rays = 1000
        sphere_pencil = RayBundle.sample_sphere(num_rays)

        for point_in_bounds in [np.array([2.1, 1.9, 1.5]),
                                np.array([5.8, 4.1, 1.5]),
                                np.array([7.2, 6.5, 1.5])]:
            sphere_pencil.moveOrigins(point_in_bounds)
            sphere_pencil.traceAll(mesh)
            front_patch_ids, back_patch_ids = sphere_pencil.getIndices()

            num_front_misses = np.count_nonzero(front_patch_ids == -1)
            num_back_misses = np.count_nonzero(back_patch_ids == -1)

            self.assertEqual(int(num_front_misses), 0,
                             msg='Some rays had no valid front intersection from point ' + str(point_in_bounds))
            self.assertEqual(int(num_back_misses), 0,
                             msg='Some rays had no valid back intersection from point ' + str(point_in_bounds))

    def test_visibility_on_surface(self):
        """
        Load a triangle mesh which is known to be closed, trace rays (uniform hemisphere) from points on the surface,
        and assert that all rays find valid intersections in the front.
        Back intersections are ignored. In theory, back rays should all hit the triangle itself;
         in practice, the origin triangle will be ignored because it's too close to the ray origin (below EPS_SELFHIT).
        """
        mesh, _, _ = load_mesh('../../example environments/AudioForGames_20_patches')

        num_rays = 1000

        for triangle_idx in range(mesh.size()):
            centroid = mesh.v1[triangle_idx] + (mesh.edge1[triangle_idx] + mesh.edge2[triangle_idx]) / 3
            hemisphere_pencil = RayBundle.sample_sphere(num_rays, hemisphere_only=True,
                                                        origin=centroid,
                                                        north_pole=mesh.n[triangle_idx])

            hemisphere_pencil.traceAll(mesh)
            front_patch_ids, _ = hemisphere_pencil.getIndices()

            num_front_misses = np.count_nonzero(front_patch_ids == -1)

            self.assertEqual(int(num_front_misses), 0,
                             msg='Some rays had no valid front intersection from the centroid of triangle ' + str(triangle_idx+1))


if __name__ == "__main__":
    unittest.main()
