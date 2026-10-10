# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NanoVDB volumes built by the Metal GPU builder against the host builder, byte for byte."""

import contextlib
import os
import unittest
from fractions import Fraction

import numpy as np

import warp as wp
import warp.fem as fem

DEVICE = "metal:0"


def metal_available() -> bool:
    return getattr(wp, "is_metal_available", lambda: False)()


@contextlib.contextmanager
def gpu_volume(enabled: bool):
    saved = wp.config.metal_gpu_volume
    wp.config.metal_gpu_volume = enabled
    try:
        yield
    finally:
        wp.config.metal_gpu_volume = saved


def volume_bytes(volume) -> np.ndarray:
    wp.synchronize_device(DEVICE)
    return volume.array().numpy().copy()


def both(build):
    """Host-built and GPU-built results of ``build()``."""
    out = []
    for enabled in (False, True):
        with gpu_volume(enabled):
            out.append(build())
    return out


def f64_bits(x: float) -> int:
    return int(np.array([x], dtype=np.float64).view(np.uint64)[0])


def fma_reference(a: float, b: float, c: float) -> int:
    """Bits of the correctly rounded a*b + c (finite inputs, round to nearest even)."""
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        negative = (np.signbit(a) ^ np.signbit(b)) and np.signbit(c) and a * b == 0
        return f64_bits(-0.0 if negative else 0.0)
    try:
        return f64_bits(float(exact))
    except OverflowError:
        return f64_bits(np.inf if exact > 0 else -np.inf)


@unittest.skipUnless(metal_available(), "Requires an Apple GPU")
class TestVolumeMetal(unittest.TestCase):
    def assert_same_volume(self, host, gpu, msg):
        hb, gb = volume_bytes(host), volume_bytes(gpu)
        self.assertEqual(hb.shape, gb.shape, f"grid size differs: {msg}")
        diff = np.nonzero(hb != gb)[0]
        self.assertEqual(diff.size, 0, f"{diff.size} bytes differ (first at {diff[:8]}): {msg}")
        for get in ("get_voxels", "get_tiles"):
            results = []
            for volume, enabled in ((host, False), (gpu, True)):
                with gpu_volume(enabled):
                    results.append(getattr(volume, get)().numpy())
            np.testing.assert_array_equal(results[0], results[1], err_msg=f"{get}: {msg}")

    def assert_voxels_match(self, points, msg, **kwargs):
        host, gpu = both(lambda: wp.Volume.allocate_by_voxels(points, device=DEVICE, **kwargs))
        self.assert_same_volume(host, gpu, msg)

    def test_allocate_by_voxels_matches_host(self):
        rng = np.random.default_rng(1)
        world = {
            "random": rng.standard_normal((5000, 3)) * 2.0,
            "clustered": np.repeat(rng.standard_normal((40, 3)) * 3.0, 50, axis=0)
            + rng.standard_normal((2000, 3)) * 0.05,
            "far": np.concatenate([rng.standard_normal((500, 3)) * 1.0e3, rng.standard_normal((500, 3))]),
            "single": np.array([[0.3, -0.2, 0.1]]),
        }
        for name, pts in world.items():
            points = wp.array(pts.astype(np.float32), dtype=wp.vec3, device=DEVICE)
            for voxel_size in (0.1, 0.037, 1.0, (0.05, 0.2, 0.07)):
                for translation in ((0.0, 0.0, 0.0), (0.013, -0.5, 7.25)):
                    self.assert_voxels_match(
                        points,
                        f"world {name} voxel_size={voxel_size} translation={translation}",
                        voxel_size=voxel_size,
                        translation=translation,
                    )

        index_points = {
            "random": rng.integers(-3000, 3000, size=(20000, 3)),
            "duplicates": np.repeat(rng.integers(-50, 50, size=(300, 3)), 30, axis=0),
            "far": np.concatenate(
                [rng.integers(-(2**20), 2**20, size=(2000, 3)), rng.integers(-10, 10, size=(2000, 3))]
            ),
            "dense": np.stack(np.meshgrid(*[np.arange(-20, 20)] * 3, indexing="ij"), -1).reshape(-1, 3),
        }
        for name, pts in index_points.items():
            points = wp.array(pts.astype(np.int32), dtype=wp.vec3i, device=DEVICE)
            self.assert_voxels_match(points, f"index {name}", voxel_size=0.5)
            # 2D int32 layout
            points2d = wp.array(pts.astype(np.int32), dtype=wp.int32, device=DEVICE)
            self.assert_voxels_match(points2d, f"index 2d {name}", voxel_size=0.5)

    def test_rounding_on_voxel_boundaries(self):
        # points exactly half way between voxel centres: the GPU must round as nanovdb::Map::applyInverseMapF(p).round()
        rng = np.random.default_rng(2)
        for voxel_size in (0.1, 0.3, 0.0125, 1.0 / 3.0):
            for translation in ((0.0, 0.0, 0.0), (0.1, -0.7, 2.3)):
                ijk = rng.integers(-500, 500, size=(20000, 3)).astype(np.float64)
                pts = (ijk + 0.5) * voxel_size + np.array(translation)
                pts = pts.astype(np.float32)
                # and the float32 neighbours of the half-way points
                pts = np.concatenate(
                    [pts, np.nextafter(pts, np.float32(np.inf)), np.nextafter(pts, np.float32(-np.inf))]
                )
                points = wp.array(pts, dtype=wp.vec3, device=DEVICE)
                self.assert_voxels_match(
                    points, f"boundaries voxel_size={voxel_size}", voxel_size=voxel_size, translation=translation
                )

    def test_transform(self):
        rng = np.random.default_rng(3)
        points = wp.array((rng.standard_normal((3000, 3)) * 2.0).astype(np.float32), dtype=wp.vec3, device=DEVICE)
        angle = 0.3
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        for scale in (0.05, 0.2):
            self.assert_voxels_match(
                points, f"rotated scale={scale}", transform=rotation * scale, translation=(1.0, 2.0, 3.0)
            )

    def test_point_mask(self):
        rng = np.random.default_rng(4)
        points = wp.array(rng.integers(-300, 300, size=(20000, 3)).astype(np.int32), dtype=wp.vec3i, device=DEVICE)
        mask = wp.array((rng.random(20000) < 0.3).astype(np.int32), device=DEVICE)
        self.assert_voxels_match(points, "masked", voxel_size=0.5, point_mask=mask)
        empty_mask = wp.zeros(20000, dtype=wp.int32, device=DEVICE)
        self.assert_voxels_match(points, "all masked", voxel_size=0.5, point_mask=empty_mask)

    def test_large(self):
        rng = np.random.default_rng(5)
        points = wp.array((rng.standard_normal((300_000, 3)) * 0.3).astype(np.float32), dtype=wp.vec3, device=DEVICE)
        self.assert_voxels_match(points, "300k", voxel_size=0.01)

    def test_empty_points(self):
        points = wp.empty(0, dtype=wp.vec3, device=DEVICE)
        for enabled in (False, True):
            with gpu_volume(enabled), self.assertRaises(RuntimeError):
                wp.Volume.allocate_by_voxels(points, voxel_size=0.1, device=DEVICE)

    def test_allocate_by_tiles_matches_host(self):
        rng = np.random.default_rng(6)
        index_points = wp.array(
            rng.integers(-3000, 3000, size=(5000, 3)).astype(np.int32), dtype=wp.vec3i, device=DEVICE
        )
        world_points = wp.array((rng.standard_normal((5000, 3)) * 2.0).astype(np.float32), dtype=wp.vec3, device=DEVICE)
        values = [
            None,
            1.5,
            7,
            wp.int64(-3),
            wp.uint32(9),
            wp.float64(2.5),
            wp.vec3(1.0, 2.0, 3.0),
            wp.vec3d(1.0, 2.0, 3.0),
            wp.vec4(1.0, 2.0, 3.0, 4.0),
            wp.vec4d(1.0, 2.0, 3.0, 4.0),
        ]
        for bg_value in values:
            for name, points in (("index", index_points), ("world", world_points)):
                host, gpu = both(
                    lambda points=points, bg_value=bg_value: wp.Volume.allocate_by_tiles(
                        points, voxel_size=0.25, bg_value=bg_value, translation=(0.5, 0.0, -1.0), device=DEVICE
                    )
                )
                self.assert_same_volume(host, gpu, f"tiles {name} bg={bg_value!r}")

    def _rebuild_pair(self, allocate, rebuild_points, point_mask=None):
        """Two volumes allocated identically; one rebuilt by the host, one by the GPU."""
        results = []
        for enabled in (False, True):
            volume = allocate()
            status = wp.full(1, 12345, dtype=wp.uint32, device=DEVICE)
            with gpu_volume(enabled):
                volume.rebuild(rebuild_points, status=status, point_mask=point_mask)
            results.append((volume, int(status.numpy()[0])))
        return results

    def test_rebuild_voxels_matches_host(self):
        rng = np.random.default_rng(7)
        initial = wp.array((rng.standard_normal((4000, 3)) * 0.5).astype(np.float32), dtype=wp.vec3, device=DEVICE)

        def allocate():
            return wp.Volume.allocate_by_voxels(
                initial,
                voxel_size=0.05,
                rebuildable=True,
                max_active_voxels=20000,
                max_leaf_nodes=4000,
                max_lower_nodes=64,
                max_upper_nodes=8,
            )

        cases = {
            "moved": (rng.standard_normal((6000, 3)) * 0.6).astype(np.float32),
            "fewer": (rng.standard_normal((100, 3)) * 0.1).astype(np.float32),
            "leaf overflow": (rng.standard_normal((20000, 3)) * 3.0).astype(np.float32),
            "lower and upper overflow": (rng.standard_normal((20000, 3)) * 500.0).astype(np.float32),
            "voxel overflow": (rng.standard_normal((40000, 3)) * 1.0).astype(np.float32),
        }
        for name, pts in cases.items():
            points = wp.array(pts, dtype=wp.vec3, device=DEVICE)
            (host, host_status), (gpu, gpu_status) = self._rebuild_pair(allocate, points)
            self.assertEqual(host_status, gpu_status, name)
            self.assert_same_volume(host, gpu, f"rebuild voxels {name}")

        points = wp.array(cases["moved"], dtype=wp.vec3, device=DEVICE)
        mask = wp.array((rng.random(6000) < 0.5).astype(np.int32), device=DEVICE)
        (host, host_status), (gpu, gpu_status) = self._rebuild_pair(allocate, points, mask)
        self.assertEqual(host_status, gpu_status)
        self.assert_same_volume(host, gpu, "rebuild voxels masked")

        empty = wp.empty(0, dtype=wp.vec3, device=DEVICE)
        (host, host_status), (gpu, gpu_status) = self._rebuild_pair(allocate, empty)
        self.assertEqual(host_status, wp.Volume.REBUILD_INVALID_INPUT)
        self.assertEqual(gpu_status, wp.Volume.REBUILD_INVALID_INPUT)
        self.assert_same_volume(host, gpu, "rebuild voxels empty")

    def test_rebuild_tiles_matches_host(self):
        rng = np.random.default_rng(8)
        initial = wp.array(rng.integers(-200, 200, size=(500, 3)).astype(np.int32), dtype=wp.vec3i, device=DEVICE)
        for bg_value in (None, 2.5, wp.vec3(1.0, -1.0, 0.5), wp.vec4d(1.0, 2.0, 3.0, 4.0)):

            def allocate(bg_value=bg_value):
                return wp.Volume.allocate_by_tiles(
                    initial,
                    voxel_size=0.1,
                    bg_value=bg_value,
                    rebuildable=True,
                    max_tiles=600,
                    max_lower_nodes=64,
                    max_upper_nodes=8,
                )

            for name, pts in (
                ("moved", rng.integers(-250, 250, size=(700, 3))),
                ("lower overflow", rng.integers(-(2**16), 2**16, size=(500, 3))),
            ):
                points = wp.array(pts.astype(np.int32), dtype=wp.vec3i, device=DEVICE)
                (host, host_status), (gpu, gpu_status) = self._rebuild_pair(allocate, points)
                self.assertEqual(host_status, gpu_status, name)
                self.assert_same_volume(host, gpu, f"rebuild tiles bg={bg_value!r} {name}")

    def test_rebuild_in_graph_capture(self):
        rng = np.random.default_rng(9)
        initial = wp.array((rng.standard_normal((3000, 3)) * 0.5).astype(np.float32), dtype=wp.vec3, device=DEVICE)
        moved = (rng.standard_normal((3000, 3)) * 0.5).astype(np.float32)
        points = wp.array(initial.numpy(), dtype=wp.vec3, device=DEVICE)
        status = wp.zeros(1, dtype=wp.uint32, device=DEVICE)
        capacities = {"max_active_voxels": 10000, "max_leaf_nodes": 2000, "max_lower_nodes": 64, "max_upper_nodes": 8}
        gpu = wp.Volume.allocate_by_voxels(points, voxel_size=0.05, rebuildable=True, **capacities)
        with wp.ScopedCapture(device=DEVICE) as capture:
            gpu.rebuild(points, status=status)
        points.assign(moved)
        wp.capture_launch(capture.graph)

        host = wp.Volume.allocate_by_voxels(initial, voxel_size=0.05, rebuildable=True, **capacities)
        with gpu_volume(False):
            host.rebuild(wp.array(moved, dtype=wp.vec3, device=DEVICE))
        self.assertEqual(int(status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
        self.assert_same_volume(host, gpu, "rebuild replayed from a graph")

    def test_nanogrid_matches_host(self):
        rng = np.random.default_rng(10)
        points = wp.array((rng.standard_normal((20000, 3)) * 0.4).astype(np.float32), dtype=wp.vec3, device=DEVICE)
        grids = []
        for enabled in (False, True):
            with gpu_volume(enabled):
                volume = wp.Volume.allocate_by_voxels(points, voxel_size=0.02, device=DEVICE)
                grids.append(fem.Nanogrid(volume))
        host, gpu = grids
        self.assert_same_volume(host.cell_grid, gpu.cell_grid, "Nanogrid cell grid")
        self.assert_same_volume(host.vertex_grid, gpu.vertex_grid, "Nanogrid node grid")
        self.assertEqual(host.cell_count(), gpu.cell_count())
        self.assertEqual(host.vertex_count(), gpu.vertex_count())
        np.testing.assert_array_equal(host._cell_ijk.numpy(), gpu._cell_ijk.numpy())
        np.testing.assert_array_equal(host._node_ijk.numpy(), gpu._node_ijk.numpy())

    def test_rebuildable_nanogrid_matches_host(self):
        rng = np.random.default_rng(11)
        initial = wp.array((rng.standard_normal((5000, 3)) * 0.4).astype(np.float32), dtype=wp.vec3, device=DEVICE)
        moved = wp.array((rng.standard_normal((5000, 3)) * 0.45).astype(np.float32), dtype=wp.vec3, device=DEVICE)
        grids = []
        for enabled in (False, True):
            with gpu_volume(enabled):
                volume = wp.Volume.allocate_by_voxels(
                    initial,
                    voxel_size=0.02,
                    device=DEVICE,
                    rebuildable=True,
                    max_active_voxels=20000,
                    max_leaf_nodes=2000,
                    max_lower_nodes=16,
                    max_upper_nodes=4,
                )
                grid = fem.Nanogrid(volume, rebuildable=True)
                status = wp.zeros(1, dtype=wp.uint32, device=DEVICE)
                grid.rebuild(moved, status=status)
                grids.append((grid, int(status.numpy()[0])))
        (host, host_status), (gpu, gpu_status) = grids
        self.assertEqual(host_status, gpu_status)
        self.assert_same_volume(host.cell_grid, gpu.cell_grid, "rebuildable Nanogrid cell grid")
        self.assert_same_volume(host.vertex_grid, gpu.vertex_grid, "rebuildable Nanogrid node grid")
        np.testing.assert_array_equal(host._cell_ijk.numpy(), gpu._cell_ijk.numpy())
        np.testing.assert_array_equal(host._node_ijk.numpy(), gpu._node_ijk.numpy())

    def test_get_voxels_loaded_grids(self):
        assets = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        tested = 0
        for name in ("test_grid.nvdb", "test_index_grid.nvdb", "test_int32_grid.nvdb", "test_vec_grid.nvdb"):
            with open(os.path.join(assets, name), "rb") as f:
                data = f.read()
            if data.startswith(b"version https://git-lfs"):
                continue  # Git LFS pointer, asset not fetched
            results = []
            for enabled in (False, True):
                with gpu_volume(enabled):
                    volume = wp.Volume.load_from_nvdb(data, DEVICE)
                    results.append((volume.get_voxels().numpy(), volume.get_tiles().numpy()))
            np.testing.assert_array_equal(results[0][0], results[1][0], err_msg=f"get_voxels {name}")
            np.testing.assert_array_equal(results[0][1], results[1][1], err_msg=f"get_tiles {name}")
            tested += 1
        if not tested:
            self.skipTest("NanoVDB test assets not fetched (Git LFS)")

    def test_software_fma(self):
        from warp._src import metal_volume  # noqa: PLC0415

        rng = np.random.default_rng(12)
        triples = []
        for _ in range(4000):
            triples.append(rng.standard_normal(3) * 10.0 ** rng.integers(-8, 9, 3))
            x = rng.standard_normal()
            triples.append([x, 1.0 + rng.standard_normal() * 1e-15, -x * (1.0 + rng.standard_normal() * 1e-15)])
            triples.append(
                [float(rng.integers(-(2**31), 2**31)), float(np.float32(rng.random())), rng.standard_normal()]
            )
        triples += [
            [1e-300, 1e-10, 0.0],
            [1e-300, 1e-300, 1e-320],
            [1e300, 1e10, -1e308],
            [0.0, 5.0, -0.0],
            [-0.0, 5.0, -0.0],
            [2.0**-1074, 0.5, 0.0],
            [2.0**-1074, 0.75, 0.0],
            [3.0 * 2.0**-1074, 0.5, 0.0],
            [1.0, 1.0, -1.0],
        ]
        values = np.array(triples, dtype=np.float64)
        result = metal_volume._test_f64_fma(wp.get_device(DEVICE), values.view(np.uint64))
        expected = np.array([fma_reference(*t) for t in values], dtype=np.uint64)
        mismatches = np.nonzero(result != expected)[0]
        self.assertEqual(mismatches.size, 0, f"fma differs for {values[mismatches[:5]]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
