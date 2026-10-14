# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic mode on Metal (warp/_src/metal_deterministic.py, warp/native/metal_deterministic.h).

Float accumulation atomics add into exact per-element integer accumulators and are rounded once after the kernel;
consumed-return counters run in two phases and hand out slots in thread order per counter element. The opt-in
module option ``metal_ordered_atomics`` keeps the earlier ordered launches (one SIMD-group in task order).
"""

import ctypes
import math
import unittest

import numpy as np

import warp as wp

RUN_TO_RUN = {"deterministic": wp.DeterministicMode.RUN_TO_RUN}
ORDERED = {"metal_ordered_atomics": True}


def metal_available() -> bool:
    return getattr(wp, "is_metal_available", lambda: False)()


def exact_f32(values) -> np.float32:
    """The exact sum of float32 values, rounded once to float32."""
    return np.float32(math.fsum(float(v) for v in values))


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def det_scatter(x: wp.array[float], idx: wp.array[int], out: wp.array[float]):
    tid = wp.tid()
    wp.atomic_add(out, idx[tid], x[tid])


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def det_scatter_vec(x: wp.array[wp.vec3], idx: wp.array[int], out: wp.array2d[wp.vec3]):
    tid = wp.tid()
    out[idx[tid] % 2, idx[tid]] += x[tid]


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def det_append(stride: int, counter: wp.array[int], out: wp.array[int]):
    tid = wp.tid()
    if tid % stride == 0:
        slot = wp.atomic_add(counter, 0, 1)
        out[slot] = tid


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def det_append_per_world(counter: wp.array[int], out: wp.array2d[int]):
    w, i = wp.tid()
    if (i * 7 + w) % 3 != 0:
        slot = wp.atomic_add(counter, w, 2)
        out[w, slot] = i
        out[w, slot + 1] = -i


@wp.kernel(module="unique", module_options=RUN_TO_RUN | {"deterministic_max_records": 4})
def det_append_loop(n: int, counter: wp.array[int], out: wp.array[int]):
    tid = wp.tid()
    for k in range(n):
        slot = wp.atomic_add(counter, 0, 1)
        out[slot] = tid * n + k


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def det_mixed_max_add(x: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    if tid % 2 == 0:
        wp.atomic_max(out, 0, x[tid])
    else:
        wp.atomic_add(out, 1, x[tid])


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def det_append_fixed_scratch(x: wp.array[float], counter: wp.array[int], out: wp.array[float]):
    tid = wp.tid()
    local = wp.zeros(shape=(2,), dtype=float)
    local[0] = x[tid]
    local[1] = 2.0 * x[tid]
    if local[0] + local[1] > 0.0:
        slot = wp.atomic_add(counter, 0, 1)
        out[slot] = local[0] + local[1]


@wp.kernel(module="unique", module_options=RUN_TO_RUN)
def det_append_after_scratch(
    x: wp.array[float], scratch: wp.array[float], counter: wp.array[int], out: wp.array[float]
):
    # unsupported: the counter call depends on a global store made earlier in the launch, which the counting pass
    # does not perform ("Side effects in the counting pass" in the deterministic execution guide)
    tid = wp.tid()
    scratch[tid] = x[tid]
    if scratch[tid] > 0.0:
        slot = wp.atomic_add(counter, 0, 1)
        out[slot] = scratch[tid]


@wp.kernel(module="unique", module_options=ORDERED)
def ordered_sum(x: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    wp.atomic_add(out, tid % 3, x[tid])


def _hooks(kernel, device):
    device = wp.get_device(device)
    wp.load_module(kernel.module, device=device)
    return kernel.module.get_kernel_hooks(kernel, device)


@unittest.skipUnless(metal_available(), "Metal not available")
class TestMetalDeterministic(unittest.TestCase):
    device = "metal:0"

    def test_float_sum_is_exactly_rounded(self):
        """Accumulate float atomics exactly and round once: equal to the correctly rounded sum, on every run."""
        rng = np.random.default_rng(0)
        n, m = 100_000, 37
        x_np = (rng.standard_normal(n) * 10.0 ** rng.integers(-4, 5, n)).astype(np.float32)
        idx_np = rng.integers(0, m, n).astype(np.int32)
        x, idx = wp.array(x_np, device=self.device), wp.array(idx_np, device=self.device)
        expected = np.array([exact_f32(x_np[idx_np == k]) for k in range(m)], dtype=np.float32)
        for _ in range(3):
            out = wp.zeros(m, dtype=float, device=self.device)
            wp.launch(det_scatter, dim=n, inputs=[x, idx, out], device=self.device)
            np.testing.assert_array_equal(out.numpy(), expected)

    def test_one_threadgroup_launch_finalizes_in_kernel(self):
        """Launches that are one threadgroup finalize their exact sums inside the kernel (one dispatch per target
        less): bitwise the separate finalize pass, for scalars and vectors, alone and after counter phases, and
        the bins are re-armed for the next launch."""
        from warp._src import metal_deterministic as md  # noqa: PLC0415

        rng = np.random.default_rng(3)
        hooks = _hooks(det_scatter, self.device)
        self.assertTrue(hooks.metal_det_tg_finalize)
        self.assertTrue(md._single_threadgroup(hooks, 200, 256))
        self.assertFalse(md._single_threadgroup(hooks, 300, 256))
        saved = md._TG_FINALIZE
        results = []
        try:
            for in_kernel in (True, False):
                md._TG_FINALIZE = in_kernel
                outs = []
                for n in (1, 37, 200, 256, 257, 5000):  # the last two are several threadgroups (separate pass)
                    x_np = (rng.standard_normal(n) * 10.0 ** rng.integers(-6, 6, n)).astype(np.float32)
                    idx_np = rng.integers(0, 9, n).astype(np.int32)
                    x, idx = wp.array(x_np, device=self.device), wp.array(idx_np, device=self.device)
                    out = wp.array(rng.standard_normal(9).astype(np.float32), device=self.device)
                    for _ in range(2):  # the second launch sees bins left by the first
                        wp.launch(det_scatter, dim=n, inputs=[x, idx, out], device=self.device)
                    xv = wp.array(rng.standard_normal((n, 3)).astype(np.float32), dtype=wp.vec3, device=self.device)
                    outv = wp.zeros((2, 9), dtype=wp.vec3, device=self.device)
                    wp.launch(det_scatter_vec, dim=n, inputs=[xv, idx, outv], device=self.device)
                    xm = wp.array(rng.standard_normal(n).astype(np.float32), device=self.device)
                    outm = wp.zeros(2, dtype=float, device=self.device)
                    wp.launch(det_mixed_max_add, dim=n, inputs=[xm, outm], device=self.device)
                    outs.append((out.numpy(), outv.numpy(), outm.numpy()))
                rng = np.random.default_rng(3)
                results.append(outs)
        finally:
            md._TG_FINALIZE = saved
        for a, b in zip(*results):
            for u, v in zip(a, b):
                np.testing.assert_array_equal(u, v)

    def test_float_sum_adds_to_destination(self):
        """Add the rounded sum to the destination's value, and leave untouched elements alone."""
        x = wp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32), device=self.device)
        idx = wp.array(np.array([1, 1, 3], dtype=np.int32), device=self.device)
        out = wp.array(np.array([5.0, 0.5, -1.0, 4.0], dtype=np.float32), device=self.device)
        wp.launch(det_scatter, dim=3, inputs=[x, idx, out], device=self.device)
        np.testing.assert_array_equal(out.numpy(), np.array([5.0, 3.5, -1.0, 7.0], dtype=np.float32))

    def test_float_sum_special_values(self):
        """Propagate infinities and NaN, and sum cancelling extremes exactly."""
        big = np.float32(3.0e38)
        cases = [
            ([np.inf, 1.0], np.inf),
            ([-np.inf, 1.0], -np.inf),
            ([np.inf, -np.inf], np.nan),
            ([np.nan, 1.0], np.nan),
            ([big, big, -big], big),
            ([1.0e-30, 3.0e-30], exact_f32(np.array([1.0e-30, 3.0e-30], dtype=np.float32))),
            ([1.0, 1.0e-30, -1.0], np.float32(1.0e-30)),
        ]
        for values, expected in cases:
            x = wp.array(np.array(values, dtype=np.float32), device=self.device)
            idx = wp.zeros(len(values), dtype=int, device=self.device)
            out = wp.zeros(1, dtype=float, device=self.device)
            wp.launch(det_scatter, dim=len(values), inputs=[x, idx, out], device=self.device)
            np.testing.assert_array_equal(out.numpy()[0], np.float32(expected), err_msg=str(values))

    def test_vector_sum_2d(self):
        """Accumulate vec3 values into a 2D array component by component."""
        rng = np.random.default_rng(1)
        n, m = 20_000, 50
        x_np = rng.standard_normal((n, 3)).astype(np.float32)
        idx_np = rng.integers(0, m, n).astype(np.int32)
        expected = np.zeros((2, m, 3), dtype=np.float32)
        for k in range(m):
            for c in range(3):
                expected[k % 2, k, c] = exact_f32(x_np[idx_np == k, c])
        x = wp.array(x_np, dtype=wp.vec3, device=self.device)
        idx = wp.array(idx_np, device=self.device)
        for _ in range(2):
            out = wp.zeros((2, m), dtype=wp.vec3, device=self.device)
            wp.launch(det_scatter_vec, dim=n, inputs=[x, idx, out], device=self.device)
            np.testing.assert_array_equal(out.numpy(), expected)

    def test_counter_slots_in_thread_order(self):
        """Hand out counter slots in thread order across many SIMD-groups and threadgroups."""
        for dim, stride in ((45, 1), (300, 1), (1000, 7), (20000, 3), (300_000, 1)):
            expected = np.arange(0, dim, stride)
            for _ in range(2):
                counter = wp.zeros(1, dtype=int, device=self.device)
                out = wp.full(dim, -1, dtype=int, device=self.device)
                wp.launch(det_append, dim=dim, inputs=[stride, counter, out], device=self.device)
                self.assertEqual(int(counter.numpy()[0]), expected.size)
                np.testing.assert_array_equal(out.numpy()[: expected.size], expected)

    def test_counter_starts_at_current_value(self):
        """Start slots at the counter's value before the launch and leave the total in it."""
        counter = wp.array([5], dtype=int, device=self.device)
        out = wp.full(100, -1, dtype=int, device=self.device)
        wp.launch(det_append, dim=50, inputs=[1, counter, out], device=self.device)
        self.assertEqual(int(counter.numpy()[0]), 55)
        np.testing.assert_array_equal(out.numpy()[5:55], np.arange(50))

    def test_per_element_counters(self):
        """Order each counter element's slots by thread, independently of the other elements."""
        # the second size spans more than 256 scan tiles of records
        for worlds, n in ((37, 500), (64, 20_000)):
            counter = wp.zeros(worlds, dtype=int, device=self.device)
            out = wp.full((worlds, 2 * n), 999999, dtype=int, device=self.device)
            wp.launch(det_append_per_world, dim=(worlds, n), inputs=[counter, out], device=self.device)
            out_np, counter_np = out.numpy(), counter.numpy()
            i = np.arange(n)
            for w in range(worlds):
                ids = i[(i * 7 + w) % 3 != 0]
                self.assertEqual(counter_np[w], 2 * ids.size)
                np.testing.assert_array_equal(out_np[w, : 2 * ids.size], np.stack([ids, -ids], axis=1).ravel())

    def test_counter_in_dynamic_loop(self):
        """Size records of a counter called in a dynamic loop with deterministic_max_records."""
        n = 3
        counter = wp.zeros(1, dtype=int, device=self.device)
        out = wp.full(50 * n, -1, dtype=int, device=self.device)
        wp.launch(det_append_loop, dim=50, inputs=[n, counter, out], device=self.device)
        np.testing.assert_array_equal(out.numpy(), np.arange(50 * n))

    def test_mixed_reductions(self):
        """Mix max and add atomics on one array (separate targets per reduction on Metal)."""
        x_np = np.random.default_rng(2).standard_normal(5001).astype(np.float32)
        x = wp.array(x_np, device=self.device)
        for _ in range(2):
            out = wp.array(np.array([-1.0e30, 0.0], dtype=np.float32), device=self.device)
            wp.launch(det_mixed_max_add, dim=5001, inputs=[x, out], device=self.device)
            np.testing.assert_array_equal(out.numpy(), [x_np[0::2].max(), exact_f32(x_np[1::2])])
            det_mixed_max_add.module.unload()  # the second launch loads the cached binary

    def test_phase0_keeps_fixed_scratch(self):
        """Keep stores to fixed-array scratch in the counting phase, so control flow matches the second phase."""
        x_np = np.random.default_rng(3).standard_normal(4000).astype(np.float32)
        x = wp.array(x_np, device=self.device)
        counter = wp.zeros(1, dtype=int, device=self.device)
        out = wp.zeros(4000, dtype=float, device=self.device)
        wp.launch(det_append_fixed_scratch, dim=4000, inputs=[x, counter, out], device=self.device)
        values = x_np + np.float32(2.0) * x_np
        expected = values[values > 0]
        self.assertEqual(int(counter.numpy()[0]), expected.size)
        np.testing.assert_array_equal(out.numpy()[: expected.size], expected)

    def _phase_mismatch_raised(self, x_np, scratch_init):
        from warp._src import metal_deterministic as md  # noqa: PLC0415

        state = md._state(wp.get_device(self.device))
        n = x_np.size
        x = wp.array(x_np, device=self.device)
        scratch = wp.full(n, scratch_init, dtype=float, device=self.device)
        counter = wp.zeros(1, dtype=int, device=self.device)
        out = wp.zeros(n, dtype=float, device=self.device)
        try:
            # the launch only peeks at the flag without waiting for the GPU; check it again after a sync
            wp.launch(det_append_after_scratch, dim=n, inputs=[x, scratch, counter, out], device=self.device)
            wp.synchronize_device(self.device)
            md._check_overflow(state, det_append_after_scratch)
        except RuntimeError as e:
            return "phase mismatch" in str(e)
        finally:
            ctypes.memset(state.overflow.ptr, 0, 4)
            state.overflow_reported = False
        return False

    def test_phase_mismatch_tripwire(self):
        """Flag a kernel whose execution pass makes counter calls its counting pass did not record (more calls:
        the counting pass reads the scratch as 0), or records calls the execution pass does not make (fewer)."""
        self.assertTrue(self._phase_mismatch_raised(np.ones(3000, dtype=np.float32), 0.0))
        self.assertTrue(self._phase_mismatch_raised(-np.ones(3000, dtype=np.float32), 1.0))

    def test_no_tripwire_when_phases_agree(self):
        """Supported counter kernels leave the flag clear (scratch already equal to the input in both passes)."""
        x_np = np.ones(3000, dtype=np.float32)
        self.assertFalse(self._phase_mismatch_raised(x_np, 1.0))

    def test_graph_capture(self):
        """Replay deterministic accumulation and counter launches from a captured graph."""
        rng = np.random.default_rng(4)
        x_np = rng.standard_normal(30_000).astype(np.float32)
        idx_np = rng.integers(0, 16, 30_000).astype(np.int32)
        x, idx = wp.array(x_np, device=self.device), wp.array(idx_np, device=self.device)
        acc = wp.zeros(16, dtype=float, device=self.device)
        counter = wp.zeros(1, dtype=int, device=self.device)
        out = wp.full(3000, -1, dtype=int, device=self.device)
        wp.launch(det_scatter, dim=30_000, inputs=[x, idx, acc], device=self.device)
        wp.launch(det_append, dim=3000, inputs=[1, counter, out], device=self.device)
        with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
            acc.zero_()
            counter.zero_()
            wp.launch(det_scatter, dim=30_000, inputs=[x, idx, acc], device=self.device)
            wp.launch(det_append, dim=3000, inputs=[1, counter, out], device=self.device)
        expected = np.array([exact_f32(x_np[idx_np == k]) for k in range(16)], dtype=np.float32)
        for _ in range(3):
            out.fill_(-1)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(acc.numpy(), expected)
            np.testing.assert_array_equal(out.numpy(), np.arange(3000))

    def test_ordered_launches_opt_in(self):
        """Run atomics in task order on one SIMD-group with the metal_ordered_atomics module option."""
        x_np = (np.random.default_rng(5).standard_normal(5000) * 1.0e3).astype(np.float32)
        x = wp.array(x_np, device=self.device)
        expected = np.zeros(3, dtype=np.float32)
        for i, v in enumerate(x_np):
            expected[i % 3] = np.float32(expected[i % 3] + v)
        out = wp.zeros(3, dtype=float, device=self.device)
        wp.launch(ordered_sum, dim=x_np.size, inputs=[x, out], device=self.device)
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertIsNotNone(_hooks(ordered_sum, self.device).metal_forward_ordered)
        self.assertIsNone(_hooks(det_scatter, self.device).metal_forward_ordered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
