# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal GPU radix sort, integer scan and run-length encoding against the host implementations, bit for bit."""

import contextlib
import unittest

import numpy as np

import warp as wp


def metal_available() -> bool:
    return getattr(wp, "is_metal_available", lambda: False)()


@contextlib.contextmanager
def gpu_utilities(enabled: bool):
    saved = wp.config.metal_gpu_utilities
    wp.config.metal_gpu_utilities = enabled
    try:
        yield
    finally:
        wp.config.metal_gpu_utilities = saved


KEY_TYPES = [
    (wp.int32, np.int32),
    (wp.uint32, np.uint32),
    (wp.float32, np.float32),
    (wp.int64, np.int64),
    (wp.uint64, np.uint64),
    (wp.float64, np.float64),
]

SMALL_SIZES = [1, 31, 32, 33, 1000, 2049]
LARGE_SIZES = [2**20 + 7, 2_000_000]


def make_keys(rng, np_type, size, pattern):
    """``size`` keys (both halves of the sort buffers) following ``pattern``."""
    width = np.dtype(np_type).itemsize * 8
    utype = np.uint32 if width == 32 else np.uint64
    if pattern == "random":
        if np.issubdtype(np_type, np.floating):
            keys = rng.standard_normal(size).astype(np_type)
            keys[::13] = 0.0
            keys[::17] = -0.0
            keys[::19] = np.inf
            keys[::23] = -np.inf
            keys[::29] = np.nan
            return keys
        bits = rng.integers(0, 2**width - 1, size=size, dtype=utype, endpoint=True)
        return bits.view(np_type)
    if pattern == "duplicates":
        return rng.integers(0, 5, size=size).astype(np_type)
    if pattern == "equal":
        return np.full(size, 3, dtype=np_type)
    if pattern == "sorted":
        return np.sort(rng.integers(-1000, 1000, size=size).astype(np_type))
    if pattern == "reverse":
        return np.sort(rng.integers(-1000, 1000, size=size).astype(np_type))[::-1].copy()
    if pattern.startswith("byte"):
        # only one byte varies: every other digit pass is skipped, which exercises the pass routing
        byte = int(pattern[4:])
        if byte * 8 >= width:
            byte = width // 8 - 1
        vals = rng.integers(0, 256, size=size).astype(utype) << utype(8 * byte)
        return (vals | utype(0x1234 if width == 32 else 0x0123_4567_0000_1234)).view(np_type)
    raise ValueError(pattern)


def sort_both(keys_np, values_np, wp_type, value_type, count, begin_bit, end_bit):
    """Full 2*count keys and values after a host sort and after a GPU sort."""
    results = []
    for enabled in (False, True):
        with gpu_utilities(enabled):
            keys = wp.array(keys_np, dtype=wp_type, device="metal:0")
            values = wp.array(values_np, dtype=value_type, device="metal:0")
            wp.utils.radix_sort_pairs(keys, values, count, begin_bit, end_bit)
            results.append((keys.numpy().view(np.uint8).copy(), values.numpy().view(np.uint8).copy()))
    return results


@unittest.skipUnless(metal_available(), "Requires an Apple GPU")
class TestMetalPrimitives(unittest.TestCase):
    def assert_sort_matches(self, keys_np, values_np, wp_type, value_type, count, begin_bit=0, end_bit=None, msg=""):
        (hk, hv), (gk, gv) = sort_both(keys_np, values_np, wp_type, value_type, count, begin_bit, end_bit)
        self.assertTrue(np.array_equal(hk, gk), f"keys differ: {msg}")
        self.assertTrue(np.array_equal(hv, gv), f"values differ: {msg}")

    def test_radix_sort_pairs_matches_host(self):
        rng = np.random.default_rng(42)
        patterns = ["random", "duplicates", "equal", "sorted", "reverse", "byte0", "byte1", "byte2", "byte3", "byte5"]
        for wp_type, np_type in KEY_TYPES:
            width = np.dtype(np_type).itemsize * 8
            bit_ranges = [(0, None), (0, 8), (3, 21), (8, 24), (16, width), (5, width - 3)]
            for n in SMALL_SIZES:
                for pattern in patterns:
                    keys_np = make_keys(rng, np_type, 2 * n, pattern)
                    for value_type, value_np in ((wp.int32, np.int32), (wp.int64, np.int64)):
                        values_np = rng.integers(-(2**31), 2**31 - 1, size=2 * n).astype(value_np)
                        for begin_bit, end_bit in bit_ranges:
                            self.assert_sort_matches(
                                keys_np,
                                values_np,
                                wp_type,
                                value_type,
                                n,
                                begin_bit,
                                end_bit,
                                f"{wp_type.__name__} n={n} {pattern} values={value_type.__name__} bits=[{begin_bit}, {end_bit})",
                            )

    def test_radix_sort_pairs_large(self):
        rng = np.random.default_rng(7)
        for wp_type, np_type in KEY_TYPES:
            for n in LARGE_SIZES:
                for pattern in ("random", "duplicates", "sorted", "byte1"):
                    keys_np = make_keys(rng, np_type, 2 * n, pattern)
                    values_np = np.arange(2 * n, dtype=np.int32)
                    self.assert_sort_matches(
                        keys_np, values_np, wp_type, wp.int32, n, msg=f"{wp_type.__name__} n={n} {pattern}"
                    )

    def test_radix_sort_pairs_empty_and_partial_buffers(self):
        rng = np.random.default_rng(3)
        keys_np = rng.integers(-100, 100, size=500).astype(np.int32)
        values_np = np.arange(500, dtype=np.int32)
        for count in (0, 1, 100, 250):
            # buffers longer than 2*count keep their tail
            self.assert_sort_matches(keys_np, values_np, wp.int32, wp.int32, count, msg=f"count={count}")
        self.assert_sort_matches(keys_np, values_np, wp.int32, wp.int32, 200, 4, 4, msg="empty bit range")

    def test_array_scan_matches_host(self):
        rng = np.random.default_rng(5)
        for wp_type, np_type in ((wp.int32, np.int32), (wp.int64, np.int64)):
            for n in [1, 31, 32, 33, 1000, 2049, 2**20 + 7, 2_000_000]:
                values_np = rng.integers(-(2**20), 2**20, size=n).astype(np_type)
                if n == 1000:
                    # wrap-around must match the host's
                    values_np[:] = np.iinfo(np_type).max // 7
                for inclusive in (True, False):
                    results = []
                    for enabled in (False, True):
                        with gpu_utilities(enabled):
                            a = wp.array(values_np, dtype=wp_type, device="metal:0")
                            out = wp.full(n, 99, dtype=wp_type, device="metal:0")
                            wp.utils.array_scan(a, out, inclusive)
                            wp.utils.array_scan(a, a, inclusive)  # in place
                            results.append((out.numpy(), a.numpy()))
                    msg = f"{wp_type.__name__} n={n} inclusive={inclusive}"
                    np.testing.assert_array_equal(results[0][0], results[1][0], err_msg=msg)
                    np.testing.assert_array_equal(results[0][1], results[1][1], err_msg=msg)

    def test_array_scan_vectors_and_strides(self):
        rng = np.random.default_rng(6)
        values_np = rng.integers(-1000, 1000, size=(3001, 3)).astype(np.int32)
        big_np = rng.integers(-1000, 1000, size=9000).astype(np.int64)
        results = []
        for enabled in (False, True):
            with gpu_utilities(enabled):
                a = wp.array(values_np, dtype=wp.vec3i, device="metal:0")
                out = wp.zeros_like(a)
                wp.utils.array_scan(a, out, True)
                big = wp.array(big_np, dtype=wp.int64, device="metal:0")
                strided_in = big[::3]
                strided_out = wp.zeros(6000, dtype=wp.int64, device="metal:0")[::2]
                wp.utils.array_scan(strided_in, strided_out, False)
                results.append((out.numpy(), strided_out.numpy()))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])

    def test_runlength_encode_matches_host(self):
        rng = np.random.default_rng(8)
        for n in [1, 2, 31, 32, 33, 1000, 2049, 2**20 + 7, 2_000_000]:
            for pattern in ("runs", "unique", "equal"):
                if pattern == "runs":
                    values_np = np.sort(rng.integers(-5, max(1, n // 3), size=n)).astype(np.int32)
                elif pattern == "unique":
                    values_np = rng.integers(-(2**31), 2**31 - 1, size=n).astype(np.int32)
                else:
                    values_np = np.full(n, 7, dtype=np.int32)
                results = []
                for enabled in (False, True):
                    with gpu_utilities(enabled):
                        values = wp.array(values_np, dtype=wp.int32, device="metal:0")
                        run_values = wp.full(n, -1, dtype=wp.int32, device="metal:0")
                        run_lengths = wp.full(n, -1, dtype=wp.int32, device="metal:0")
                        run_count = wp.zeros(1, dtype=wp.int32, device="metal:0")
                        wp.utils.runlength_encode(values, run_values, run_lengths, run_count)
                        host_count = wp.utils.runlength_encode(values, run_values, run_lengths)
                        partial = wp.utils.runlength_encode(values, run_values, run_lengths, value_count=n // 2)
                        results.append(
                            (run_values.numpy(), run_lengths.numpy(), run_count.numpy(), host_count, partial)
                        )
                msg = f"n={n} {pattern}"
                for h, g in zip(results[0], results[1], strict=True):
                    np.testing.assert_array_equal(h, g, err_msg=msg)

    def test_aliased_outputs(self):
        """Overlapping inputs and outputs (other than an exact in-place scan) give the host results."""
        rng = np.random.default_rng(10)
        n = 5000
        runs_np = np.sort(rng.integers(0, 300, size=n)).astype(np.int32)
        scan_np = rng.integers(-100, 100, size=n + 1).astype(np.int32)
        results = []
        for enabled in (False, True):
            with gpu_utilities(enabled):
                values = wp.array(runs_np, dtype=wp.int32, device="metal:0")
                run_lengths = wp.zeros(n, dtype=wp.int32, device="metal:0")
                count = wp.utils.runlength_encode(values, values, run_lengths)
                shifted = wp.array(scan_np, dtype=wp.int32, device="metal:0")
                wp.utils.array_scan(shifted[:n], shifted[1:], True)
                results.append((values.numpy(), run_lengths.numpy(), count, shifted.numpy()))
        for h, g in zip(results[0], results[1], strict=True):
            np.testing.assert_array_equal(h, g)

    def test_graph_capture(self):
        """Sort, run-length encode and scan recorded into a Metal graph replay on new inputs like the host."""
        device = wp.get_device("metal:0")
        rng = np.random.default_rng(9)
        n = 300_000

        def inputs():
            return rng.integers(0, 1000, size=2 * n).astype(np.int32), rng.integers(0, 2**31 - 1, size=2 * n).astype(
                np.int32
            )

        keys = wp.zeros(2 * n, dtype=wp.int32, device=device)
        values = wp.zeros(2 * n, dtype=wp.int32, device=device)
        run_values = wp.zeros(n, dtype=wp.int32, device=device)
        run_lengths = wp.zeros(n, dtype=wp.int32, device=device)
        run_count = wp.zeros(1, dtype=wp.int32, device=device)
        offsets = wp.zeros(n, dtype=wp.int32, device=device)

        def pipeline():
            wp.utils.radix_sort_pairs(keys, values, n)
            wp.utils.runlength_encode(keys, run_values, run_lengths, run_count, value_count=n)
            wp.utils.array_scan(run_lengths, offsets, False)

        def outputs():
            return [a.numpy().copy() for a in (keys, values, run_values, run_lengths, run_count, offsets)]

        with gpu_utilities(True):
            # a fresh device scratch pool grows inside the capture
            device.__dict__.pop("_metal_primitives", None)
            with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                pipeline()
            for _ in range(3):
                k, v = inputs()
                keys.assign(k)
                values.assign(v)
                run_values.zero_()
                run_lengths.zero_()
                wp.capture_launch(capture.graph)
                got = outputs()
                with gpu_utilities(False):
                    keys.assign(k)
                    values.assign(v)
                    run_values.zero_()
                    run_lengths.zero_()
                    pipeline()
                    want = outputs()
                for g, w in zip(got, want, strict=True):
                    np.testing.assert_array_equal(g, w)


if __name__ == "__main__":
    unittest.main(verbosity=2)
