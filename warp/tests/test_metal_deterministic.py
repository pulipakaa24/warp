# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic mode on Metal: kernels with atomics in deterministic modules run as ordered launches."""

import unittest

import numpy as np

import warp as wp

RUN_TO_RUN = {"deterministic": wp.DeterministicMode.RUN_TO_RUN}


def metal_available() -> bool:
    return getattr(wp, "is_metal_available", lambda: False)()


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def ordered_append(stride: int, counter: wp.array[int], out: wp.array[int]):
    tid = wp.tid()
    if tid % stride == 0:
        slot = wp.atomic_add(counter, 0, 1)
        out[slot] = tid


@wp.kernel(module="unique")
def unordered_append(counter: wp.array[int], out: wp.array[int]):
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    out[slot] = tid


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def ordered_append_loop(counter: wp.array[int], out: wp.array[int]):
    tid = wp.tid()
    for k in range(3):
        slot = wp.atomic_add(counter, 0, 1)
        out[slot] = tid * 3 + k


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def ordered_sum(x: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    wp.atomic_add(out, tid % 3, x[tid])


@wp.func
def add_via_function(out: wp.array2d[float], i: int, j: int, v: float):
    out[i, j] += v


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def ordered_sum_2d_func(x: wp.array2d[float], out: wp.array2d[float]):
    i, j = wp.tid()
    add_via_function(out, 0, j % 2, x[i, j])


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def ordered_sum_fixed_scratch(x: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    local = wp.zeros(shape=(2,), dtype=float)
    local[0] = x[tid]
    local[1] = 2.0 * x[tid]
    wp.atomic_add(out, 0, local[0] + local[1])


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def mixed_max_add(x: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    if tid % 2 == 0:
        wp.atomic_max(out, 0, x[tid])
    else:
        wp.atomic_add(out, 1, x[tid])


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def no_atomics(x: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    out[tid] = 2.0 * x[tid]


def _hooks(kernel, device):
    device = wp.get_device(device)
    wp.load_module(kernel.module, device=device)
    return kernel.module.get_kernel_hooks(kernel, device)


@unittest.skipUnless(metal_available(), "Metal not available")
class TestMetalDeterministic(unittest.TestCase):
    device = "metal:0"

    def test_counter_slots_in_task_order(self):
        """Hand out counter slots in task order across many SIMD-groups and threadgroups."""
        for dim, stride in ((45, 1), (300, 1), (1000, 7), (20000, 3)):
            expected = np.arange(0, dim, stride)
            for _ in range(5):
                counter = wp.zeros(1, dtype=int, device=self.device)
                out = wp.full(dim, -1, dtype=int, device=self.device)
                wp.launch(ordered_append, dim=dim, inputs=[stride, counter, out], device=self.device)
                self.assertEqual(int(counter.numpy()[0]), expected.size)
                np.testing.assert_array_equal(out.numpy()[: expected.size], expected)

    def test_counter_loop_matches_simd_interleaving(self):
        """Interleave repeated atomics of one SIMD stride lane by lane, as one CUDA warp issues them."""
        dim = 50
        expected = []
        for start in range(0, dim, 32):
            lanes = range(start, min(start + 32, dim))
            for k in range(3):
                expected.extend(t * 3 + k for t in lanes)
        counter = wp.zeros(1, dtype=int, device=self.device)
        out = wp.full(3 * dim, -1, dtype=int, device=self.device)
        wp.launch(ordered_append_loop, dim=dim, inputs=[counter, out], device=self.device)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_float_sum_is_task_order_sum(self):
        """Accumulate float atomics in task order: bitwise equal to a sequential float32 sum."""
        x_np = (np.random.default_rng(0).standard_normal(5000) * 1.0e3).astype(np.float32)
        x = wp.array(x_np, device=self.device)
        expected = np.zeros(3, dtype=np.float32)
        for i, v in enumerate(x_np):
            expected[i % 3] = np.float32(expected[i % 3] + v)
        for _ in range(3):
            out = wp.zeros(3, dtype=float, device=self.device)
            wp.launch(ordered_sum, dim=x_np.size, inputs=[x, out], device=self.device)
            np.testing.assert_array_equal(out.numpy(), expected)

    def test_atomic_in_user_function_and_2d_launch(self):
        """Order atomics reached through a user function in a 2D launch (row-major task order)."""
        x_np = (np.random.default_rng(1).standard_normal((40, 6)) * 10.0).astype(np.float32)
        x = wp.array(x_np, device=self.device)
        expected = np.zeros((1, 2), dtype=np.float32)
        for i in range(40):
            for j in range(6):
                expected[0, j % 2] = np.float32(expected[0, j % 2] + x_np[i, j])
        out = wp.zeros((1, 2), dtype=float, device=self.device)
        wp.launch(ordered_sum_2d_func, dim=x_np.shape, inputs=[x, out], device=self.device)
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertIsNotNone(_hooks(ordered_sum_2d_func, self.device).metal_forward_ordered)

    def test_fixed_scratch_kernel(self):
        """Give every task of an ordered launch its own fixed-array scratch slot."""
        x_np = np.arange(100, dtype=np.float32)
        x = wp.array(x_np, device=self.device)
        out = wp.zeros(1, dtype=float, device=self.device)
        wp.launch(ordered_sum_fixed_scratch, dim=100, inputs=[x, out], device=self.device)
        expected = np.float32(0.0)
        for v in x_np:
            expected = np.float32(expected + np.float32(v + np.float32(2.0) * v))
        self.assertEqual(out.numpy()[0], expected)

    def test_only_deterministic_atomic_kernels_are_ordered(self):
        """Keep the parallel entry point for kernels without atomics and for non-deterministic modules."""
        self.assertIsNotNone(_hooks(ordered_append, self.device).metal_forward_ordered)
        self.assertIsNone(_hooks(no_atomics, self.device).metal_forward_ordered)
        self.assertIsNone(_hooks(unordered_append, self.device).metal_forward_ordered)
        x = wp.array(np.arange(64, dtype=np.float32), device=self.device)
        out = wp.zeros(64, dtype=float, device=self.device)
        wp.launch(no_atomics, dim=64, inputs=[x, out], device=self.device)
        np.testing.assert_array_equal(out.numpy(), 2.0 * np.arange(64, dtype=np.float32))
        counter = wp.zeros(1, dtype=int, device=self.device)
        slots = wp.full(64, -1, dtype=int, device=self.device)
        wp.launch(unordered_append, dim=64, inputs=[counter, slots], device=self.device)
        np.testing.assert_array_equal(np.sort(slots.numpy()), np.arange(64))

    def test_mixed_reductions_from_cache(self):
        """Launch a kernel mixing max and add atomics on one array, also when its module comes from the cache.

        CUDA's record-based deterministic mode rejects this mix; ordered launches do not need its metadata.
        """
        x_np = np.arange(70, dtype=np.float32)
        x = wp.array(x_np, device=self.device)
        expected = np.array([x_np[::2].max(), np.float32(0.0)], dtype=np.float32)
        for v in x_np[1::2]:
            expected[1] = np.float32(expected[1] + v)
        for _ in range(2):
            out = wp.zeros(2, dtype=float, device=self.device)
            wp.launch(mixed_max_add, dim=70, inputs=[x, out], device=self.device)
            np.testing.assert_array_equal(out.numpy(), expected)
            mixed_max_add.module.unload()  # the second launch loads the cached binary

    def test_graph_capture(self):
        """Replay an ordered launch from a captured graph."""
        counter = wp.zeros(1, dtype=int, device=self.device)
        out = wp.full(300, -1, dtype=int, device=self.device)
        wp.load_module(ordered_append.module, device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            counter.zero_()
            wp.launch(ordered_append, dim=300, inputs=[1, counter, out], device=self.device)
        for _ in range(3):
            out.fill_(-1)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(out.numpy(), np.arange(300))


if __name__ == "__main__":
    unittest.main(verbosity=2)
