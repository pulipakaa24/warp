# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``wp.launch(..., dim_limit=...)``: a launch bounded by a device-side count (indirect dispatch on Metal)."""

import unittest

import numpy as np

import warp as wp
import warp._src.context as ctx

BLOCK = 256


def metal_available() -> bool:
    return getattr(wp, "is_metal_available", lambda: False)()


@wp.kernel(module="unique")
def mark(out: wp.array[int]):
    i = wp.tid()
    out[i] = i + 1


@wp.kernel(module="unique")
def mark2d(out: wp.array2d[int]):
    i, j = wp.tid()
    out[i, j] = i * 10 + j + 1


@wp.kernel(module="unique")
def set_count(count: wp.array[int], value: int):
    count[0] = value


@wp.kernel(module="unique")
def bump_count(count: wp.array[int], step: int):
    count[0] = count[0] + step


@wp.kernel(module="unique")
def axpy(a: float, x: wp.array[wp.vec3], y: wp.array[wp.vec3]):
    i = wp.tid()
    y[i] = y[i] + a * x[i]


@wp.kernel(module="unique")
def close_range(ranges: wp.array[wp.uint32]):
    ranges[1] = wp.uint32(0)


@wp.kernel(module="unique")
def add_one(out: wp.array[int]):
    i = wp.tid()
    out[i] = out[i] + 1


def ran(n, count, block=BLOCK):
    """Return the indices that must have run (below the count) and the first that must not (past its threadgroup)."""
    c = max(0, min(count, n))
    return c, min(n, -(-c // block) * block)


@unittest.skipUnless(metal_available(), "Requires a Metal device")
class TestMetalBoundedLaunch(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device("metal:0")

    def check(self, out, count, n, block=BLOCK):
        c, end = ran(n, count, block)
        a = out.numpy().reshape(n, -1)
        expected = np.arange(1, n + 1)
        np.testing.assert_array_equal(a[:c, 0], expected[:c])
        np.testing.assert_array_equal(a[end:], -1)

    def test_eager_counts(self):
        n = 5000
        for count in (0, -3, 1, 255, 256, 257, 1000, 4999, 5000, 7000):
            with self.subTest(count=count):
                out = wp.full(n, -1, dtype=int, device=self.device)
                limit = wp.array([count], dtype=int, device=self.device)
                wp.launch(mark, dim=n, inputs=[out], dim_limit=limit, device=self.device)
                self.check(out, count, n)

    def test_count_written_on_device(self):
        # the bound is produced by a kernel and read by the dispatch, with no host round trip in between
        n = 4096
        out = wp.full(n, -1, dtype=int, device=self.device)
        limit = wp.zeros(1, dtype=int, device=self.device)
        wp.launch(set_count, dim=1, inputs=[limit, 700], device=self.device)
        wp.launch(mark, dim=n, inputs=[out], dim_limit=limit, device=self.device)
        self.check(out, 700, n)

    def test_view_and_record_cmd(self):
        n = 3000
        out = wp.full(n, -1, dtype=int, device=self.device)
        counts = wp.array([5, 1234, 9], dtype=int, device=self.device)
        cmd = wp.launch(mark, dim=n, inputs=[out], dim_limit=counts[1:2], device=self.device, record_cmd=True)
        cmd.launch()
        self.check(out, 1234, n)

    def test_multidimensional_leading_bound(self):
        n, m = 1000, 3
        out = wp.full((n, m), -1, dtype=int, device=self.device)
        limit = wp.array([421], dtype=int, device=self.device)
        wp.launch(mark2d, dim=(n, m), inputs=[out], dim_limit=limit, device=self.device)
        a = out.numpy()
        ii, jj = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
        np.testing.assert_array_equal(a[:421], (ii * 10 + jj + 1)[:421])
        end = -(-421 * m // BLOCK) * BLOCK  # flat task index past the last threadgroup that ran
        np.testing.assert_array_equal(a.reshape(-1)[end:], -1)

    def test_bitwise_equal_to_full_launch_below_count(self):
        n, c = 100000, 37011
        rng = np.random.default_rng(0)
        x = wp.array(rng.standard_normal((n, 3)).astype(np.float32), dtype=wp.vec3, device=self.device)
        y0 = rng.standard_normal((n, 3)).astype(np.float32)
        y_full = wp.array(y0, dtype=wp.vec3, device=self.device)
        y_bnd = wp.array(y0, dtype=wp.vec3, device=self.device)
        limit = wp.array([c], dtype=int, device=self.device)
        wp.launch(axpy, dim=n, inputs=[0.37, x, y_full], device=self.device)
        wp.launch(axpy, dim=n, inputs=[0.37, x, y_bnd], device=self.device, dim_limit=limit)
        a, b = y_full.numpy(), y_bnd.numpy()
        self.assertTrue(np.array_equal(a[:c].view(np.uint32), b[:c].view(np.uint32)))
        _, end = ran(n, c)
        self.assertTrue(np.array_equal(b[end:], y0[end:]))

    def test_graph_rereads_count(self):
        n = 6000
        out = wp.full(n, -1, dtype=int, device=self.device)
        limit = wp.zeros(1, dtype=int, device=self.device)
        with wp.ScopedCapture(device=self.device) as cap:
            wp.launch(bump_count, dim=1, inputs=[limit, 1000], device=self.device)
            wp.launch(mark, dim=n, inputs=[out], dim_limit=limit, device=self.device)
        for k in range(1, 4):
            out.fill_(-1)
            wp.capture_launch(cap.graph)
            self.check(out, 1000 * k, n)

    def test_graph_bounded_between_plain_launches(self):
        n = 2048
        before = wp.zeros(n, dtype=int, device=self.device)
        after = wp.zeros(n, dtype=int, device=self.device)
        out = wp.full(n, -1, dtype=int, device=self.device)
        limit = wp.array([300], dtype=int, device=self.device)
        with wp.ScopedCapture(device=self.device) as cap:
            wp.launch(add_one, dim=n, inputs=[before], device=self.device)
            wp.launch(mark, dim=n, inputs=[out], dim_limit=limit, device=self.device)
            wp.launch(add_one, dim=n, inputs=[after], device=self.device)
        wp.capture_launch(cap.graph)
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(before.numpy(), 2)
        np.testing.assert_array_equal(after.numpy(), 2)
        self.check(out, 300, n)

    def test_bounded_inside_indirect_range(self):
        if not hasattr(ctx, "metal_capture_range_begin"):
            self.skipTest("no indirect execution ranges")
        n = 4096
        ranges = wp.zeros(4, dtype=wp.uint32, device=self.device)
        pre = wp.zeros(n, dtype=int, device=self.device)
        post = wp.zeros(n, dtype=int, device=self.device)
        out = wp.full(n, -1, dtype=int, device=self.device)
        limit = wp.array([513], dtype=int, device=self.device)
        with wp.ScopedCapture(device=self.device) as cap:
            ctx.metal_capture_range_begin(ranges, 0, device=self.device)
            wp.launch(add_one, dim=n, inputs=[pre], device=self.device)
            wp.launch(mark, dim=n, inputs=[out], dim_limit=limit, device=self.device)
            wp.launch(add_one, dim=n, inputs=[post], device=self.device)
            wp.launch(close_range, dim=1, inputs=[ranges], device=self.device)  # later replays skip the range
            ctx.metal_capture_range_end(device=self.device)
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(pre.numpy(), 1)
        np.testing.assert_array_equal(post.numpy(), 1)
        self.check(out, 513, n)
        # closed: nothing in the range runs, the bounded dispatch included
        out.fill_(-1)
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(pre.numpy(), 1)
        np.testing.assert_array_equal(post.numpy(), 1)
        np.testing.assert_array_equal(out.numpy(), -1)
        # reopened from the host
        r = ranges.numpy()
        r[1] = r[2]
        ranges.assign(r)
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(pre.numpy(), 2)
        np.testing.assert_array_equal(post.numpy(), 2)
        self.check(out, 513, n)

    def test_validation(self):
        out = wp.zeros(10, dtype=int, device=self.device)
        with self.assertRaises(TypeError):
            wp.launch(mark, dim=10, inputs=[out], dim_limit=wp.zeros(1, dtype=float, device=self.device))
        with self.assertRaises(ValueError):
            wp.launch(mark, dim=10, inputs=[out], dim_limit=wp.zeros(1, dtype=int, device="cpu"), device=self.device)


class TestBoundedLaunchCpu(unittest.TestCase):
    def test_cpu_runs_full_launch(self):
        n = 100
        out = wp.full(n, -1, dtype=int, device="cpu")
        limit = wp.array([10], dtype=int, device="cpu")
        wp.launch(mark, dim=n, inputs=[out], dim_limit=limit, device="cpu")
        np.testing.assert_array_equal(out.numpy(), np.arange(1, n + 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
