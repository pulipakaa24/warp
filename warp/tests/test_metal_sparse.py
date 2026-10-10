# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal GPU BSR assembly (``bsr_set_from_triplets``, ``bsr_set_transpose``) against the host implementation.

The host path is the same Warp code with ``warp.config.metal_gpu_utilities`` off (the native host functions on the
Metal arrays). Topology is compared bit for bit. Values are compared bit for bit whenever the summation order of
duplicate triplets cannot matter (no run of three or more duplicates, or values whose sums are exact); otherwise the
GPU sum is compared bit for bit with a sequential sum in increasing triplet order, the order of the stable GPU sort
(the host's ``std::sort`` is not stable).
"""

import contextlib
import unittest

import numpy as np

import warp as wp
import warp.sparse as sparse
from warp._src.sparse import _bsr_ensure_fits, _bsr_set_from_triplets_native, _zero_value_masks


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


BLOCK_SHAPES = [(1, 1), (3, 3), (6, 6), (1, 3), (3, 1)]


def block_type(shape, scalar=wp.float32):
    return scalar if shape == (1, 1) else wp.types.matrix(shape=shape, dtype=scalar)


def make_triplets(
    rng, n, nrow, ncol, shape, dup_rows=None, invalid=0.0, zeros=0.0, integer=False, np_scalar=np.float32
):
    """``n`` triplets; ``dup_rows`` draws (row, column) from a small set so that duplicates are frequent."""
    if dup_rows is not None:
        rows = rng.integers(0, dup_rows, n)
        cols = rng.integers(0, min(dup_rows, ncol), n)
    else:
        rows = rng.integers(0, nrow, n)
        cols = rng.integers(0, ncol, n)
    if invalid > 0:
        bad = rng.random(n) < invalid
        which = rng.integers(0, 4, n)
        rows = np.where(bad & (which == 0), -1 - rng.integers(0, 3, n), rows)
        rows = np.where(bad & (which == 1), nrow + rng.integers(0, 3, n), rows)
        cols = np.where(bad & (which == 2), -1 - rng.integers(0, 3, n), cols)
        cols = np.where(bad & (which == 3), ncol + rng.integers(0, 3, n), cols)
    if integer:
        values = rng.integers(-8, 9, (n, *shape)).astype(np_scalar)
    else:
        values = rng.standard_normal((n, *shape)).astype(np_scalar)
    if zeros > 0:
        z = rng.random(n) < zeros
        values[z] = 0.0
        # negative zeros are zero blocks too (the prune mask drops the sign bit)
        nz = z & (rng.random(n) < 0.5)
        values[nz] = -0.0
    return rows.astype(np.int32), cols.astype(np.int32), values


def run_native(device, nrow, ncol, shape, rows, cols, values, count=None, prune=True, masked_by=None):
    """Call the topology builder like ``bsr_set_from_triplets`` does and return every output array."""
    n = len(rows)
    scalar = wp.float64 if values is not None and values.dtype == np.float64 else wp.float32
    dest = sparse.bsr_zeros(nrow, ncol, block_type(shape, scalar), device=device)
    if masked_by is not None:
        sparse.bsr_assign(dest=dest, src=masked_by)
        if dest.columns.size < n:
            # grow the storage without losing the mask's columns
            columns = wp.empty(n, dtype=wp.int32, device=device)
            wp.copy(columns, dest.columns, count=dest.columns.size)
            dest.columns = columns
    else:
        _bsr_ensure_fits(dest, nnz=n)
    w_rows = wp.array(rows, dtype=wp.int32, device=device)
    w_cols = wp.array(cols, dtype=wp.int32, device=device)
    w_values = wp.array(values, dtype=scalar, device=device) if values is not None else None
    w_count = wp.array([count], dtype=wp.int32, device=device) if count is not None else None
    summed_offsets = wp.full(n, -7, dtype=wp.int32, device=device)
    summed_indices = wp.full(n, -7, dtype=wp.int32, device=device)
    mask = _zero_value_masks.get(scalar, 0) if prune and values is not None else 0
    _bsr_set_from_triplets_native(
        dest,
        w_rows,
        w_cols,
        w_values,
        w_count,
        n,
        mask,
        masked_by is not None,
        summed_offsets,
        summed_indices,
        dest.offsets,
        dest.row_counts,
        dest.columns,
    )
    offsets = dest.offsets.numpy()[: nrow + 1].copy()
    nnz = int(offsets[nrow])
    return offsets, dest.columns.numpy()[:nnz].copy(), summed_offsets.numpy()[:nnz], summed_indices.numpy()


def assemble(device, nrow, ncol, shape, rows, cols, values, count=None, prune=True, topology=None, dest=None):
    scalar = wp.float64 if values.dtype == np.float64 else wp.float32
    if dest is None:
        dest = sparse.bsr_zeros(nrow, ncol, block_type(shape, scalar), device=device)
    w_values = wp.array(values, dtype=scalar, device=device)
    if shape == (1, 1):
        w_values = w_values.reshape((len(rows),))
    elif w_values.ndim == 3:
        w_values = wp.array(values, dtype=block_type(shape, scalar), device=device)
    sparse.bsr_set_from_triplets(
        dest,
        wp.array(rows, dtype=wp.int32, device=device),
        wp.array(cols, dtype=wp.int32, device=device),
        w_values,
        count=wp.array([count], dtype=wp.int32, device=device) if count is not None else None,
        prune_numerical_zeros=prune,
        topology=topology,
    )
    return dest


def matrix_arrays(m):
    nnz = m.nnz_sync()
    out = {"offsets": m.offsets.numpy()[: m.nrow + 1].copy(), "nnz": nnz}
    if m.row_counts is not None:
        out["row_counts"] = m.row_counts.numpy()[: m.nrow].copy()
        active = np.concatenate(
            [np.arange(o, o + c) for o, c in zip(out["offsets"][:-1], out["row_counts"], strict=True)]
            or [np.zeros(0, dtype=int)]
        ).astype(int)
    else:
        active = np.arange(nnz)
    out["columns"] = m.columns.numpy()[active].copy()
    out["values"] = m.scalar_values.numpy()[active].copy()
    return out


def stable_reference_values(nrow, ncol, shape, rows, cols, values, count, prune):
    """Block values summed sequentially in increasing triplet order, in (row, column) order."""
    n = len(rows) if count is None else count
    keep = (rows[:n] >= 0) & (rows[:n] < nrow) & (cols[:n] >= 0) & (cols[:n] < ncol)
    if prune:
        keep &= np.any(values[:n].reshape(n, -1) != 0, axis=1)
    idx = np.nonzero(keep)[0]
    order = idx[np.lexsort((idx, cols[idx], rows[idx]))]
    blocks = []
    prev = None
    for i in order:
        key = (rows[i], cols[i])
        if key != prev:
            blocks.append(values[i].copy())
            prev = key
        else:
            blocks[-1] = blocks[-1] + values[i]  # elementwise in the scalar type, as the kernel
    return np.array(blocks, dtype=values.dtype).reshape(len(blocks), *shape)


def max_multiplicity(nrow, ncol, rows, cols, n=None):
    n = len(rows) if n is None else n
    keep = (rows[:n] >= 0) & (rows[:n] < nrow) & (cols[:n] >= 0) & (cols[:n] < ncol)
    if not np.any(keep):
        return 0
    _, counts = np.unique(np.stack([rows[:n][keep], cols[:n][keep]]), axis=1, return_counts=True)
    return int(counts.max())


@unittest.skipUnless(metal_available(), "Requires an Apple GPU")
class TestMetalSparse(unittest.TestCase):
    device = None

    @classmethod
    def setUpClass(cls):
        cls.device = wp.get_device("metal:0")

    def both(self, fn):
        with gpu_utilities(False):
            host = fn()
        with gpu_utilities(True):
            gpu = fn()
        return host, gpu

    def assert_topology_matches(self, host, gpu, msg):
        h_off, h_cols, h_so, h_si = host
        g_off, g_cols, g_so, g_si = gpu
        np.testing.assert_array_equal(g_off, h_off, err_msg=f"offsets: {msg}")
        np.testing.assert_array_equal(g_cols, h_cols, err_msg=f"columns: {msg}")
        np.testing.assert_array_equal(g_so, h_so, err_msg=f"summed offsets: {msg}")
        # the same triplets in each segment; increasing triplet order on the GPU (stable sort)
        beg = 0
        for end in h_so:
            self.assertEqual(sorted(h_si[beg:end]), list(g_si[beg:end]), f"summed indices: {msg}")
            beg = end

    def test_triplet_topology_matches_host(self):
        """Offsets, columns and summed-triplet segments, for every block shape and pruning option."""
        rng = np.random.default_rng(1)
        cases = [
            # n, nrow, ncol, dup_rows, invalid, zeros
            (1, 4, 5, None, 0.0, 0.0),
            (37, 6, 4, 3, 0.0, 0.0),
            (1000, 50, 40, None, 0.1, 0.2),
            (5000, 300, 7, 6, 0.05, 0.3),
            (20000, 4000, 3000, None, 0.02, 0.1),
            (40000, 100, 100, 10, 0.2, 0.5),
        ]
        for shape in BLOCK_SHAPES:
            for n, nrow, ncol, dup, invalid, zeros in cases:
                rows, cols, values = make_triplets(rng, n, nrow, ncol, shape, dup, invalid, zeros)
                for prune in (True, False):
                    for count in (None, n // 2):
                        msg = f"shape={shape} n={n} prune={prune} count={count}"
                        host, gpu = self.both(
                            lambda rows=rows, cols=cols, values=values, prune=prune, count=count, nrow=nrow, ncol=ncol, shape=shape: (
                                run_native(self.device, nrow, ncol, shape, rows, cols, values, count, prune)
                            )
                        )
                        self.assert_topology_matches(host, gpu, msg)

    def test_triplet_values_match_host(self):
        """Assembled blocks bitwise equal to the host: exact sums with many duplicates, or at most two duplicates."""
        rng = np.random.default_rng(2)
        for shape in BLOCK_SHAPES:
            for n, nrow, ncol, dup, integer in [
                (3000, 40, 40, 5, True),  # long duplicate runs, sums exact in any order
                (30000, 200, 300, 20, True),
                (4000, 3000, 3000, None, False),  # few duplicates (at most two checked below)
            ]:
                rows, cols, values = make_triplets(rng, n, nrow, ncol, shape, dup, 0.1, 0.2, integer=integer)
                if not integer:
                    # keep at most two triplets per (row, column)
                    _, first = np.unique(np.stack([rows, cols]), axis=1, return_index=True)
                    keep = np.zeros(n, dtype=bool)
                    keep[first] = True
                    second = np.setdiff1d(np.arange(n), first)[::2]
                    keep[second] = True
                    rows, cols, values = rows[keep], cols[keep], values[keep]
                    self.assertLessEqual(max_multiplicity(nrow, ncol, rows, cols), 2)
                for prune in (True, False):
                    for topology in ("compact", "padded"):
                        msg = f"shape={shape} n={len(rows)} prune={prune} topology={topology}"

                        def fn(
                            rows=rows,
                            cols=cols,
                            values=values,
                            prune=prune,
                            topology=topology,
                            nrow=nrow,
                            ncol=ncol,
                            shape=shape,
                        ):
                            dest = None
                            if topology == "padded":
                                dest = sparse.bsr_zeros(
                                    nrow, ncol, block_type(shape), device=self.device, topology="padded", row_capacity=8
                                )
                            return matrix_arrays(
                                assemble(
                                    self.device, nrow, ncol, shape, rows, cols, values, None, prune, topology, dest
                                )
                            )

                        host, gpu = self.both(fn)
                        for key in host:
                            np.testing.assert_array_equal(gpu[key], host[key], err_msg=f"{key}: {msg}")

    def test_triplet_values_stable_order(self):
        """With long runs of inexact duplicates, the GPU sums each run in increasing triplet order, bit for bit."""
        rng = np.random.default_rng(3)
        for shape in BLOCK_SHAPES:
            for np_scalar in (np.float32,):  # no float64 on Metal
                nrow, ncol = 30, 20
                rows, cols, values = make_triplets(rng, 6000, nrow, ncol, shape, 8, 0.1, 0.1, np_scalar=np_scalar)
                for prune, count in ((True, None), (False, 4000)):
                    with gpu_utilities(True):
                        got = matrix_arrays(assemble(self.device, nrow, ncol, shape, rows, cols, values, count, prune))
                    want = stable_reference_values(nrow, ncol, shape, rows, cols, values, count, prune)
                    msg = f"shape={shape} {np_scalar.__name__} prune={prune} count={count}"
                    np.testing.assert_array_equal(got["values"].reshape(want.shape), want, err_msg=msg)
                    with gpu_utilities(False):
                        host = matrix_arrays(assemble(self.device, nrow, ncol, shape, rows, cols, values, count, prune))
                    np.testing.assert_array_equal(got["offsets"], host["offsets"], err_msg=msg)
                    np.testing.assert_array_equal(got["columns"], host["columns"], err_msg=msg)
                    np.testing.assert_allclose(got["values"], host["values"], rtol=1e-5, atol=1e-5, err_msg=msg)

    def test_triplet_edge_cases(self):
        """Empty input, every triplet discarded, a single row, wide (64-bit) keys and topology-only builds."""
        rng = np.random.default_rng(4)
        shape = (3, 3)
        # every triplet out of range, zero, or past the count
        rows = np.array([-1, 5, 0, 1], dtype=np.int32)
        cols = np.array([0, 0, 9, 1], dtype=np.int32)
        values = np.zeros((4, *shape), dtype=np.float32)
        for count in (None, 0):
            host, gpu = self.both(lambda count=count: run_native(self.device, 5, 5, shape, rows, cols, values, count))
            self.assert_topology_matches(host, gpu, f"all discarded count={count}")
            self.assertEqual(int(gpu[0][-1]), 0)
        # empty triplet arrays
        for topology in ("compact", "padded"):
            host, gpu = self.both(
                lambda topology=topology: matrix_arrays(
                    assemble(
                        self.device,
                        4,
                        4,
                        shape,
                        np.zeros(0, np.int32),
                        np.zeros(0, np.int32),
                        np.zeros((0, *shape), np.float32),
                        topology=topology,
                        dest=sparse.bsr_zeros(4, 4, block_type(shape), device=self.device, topology=topology)
                        if topology == "padded"
                        else None,
                    )
                )
            )
            for key in host:
                np.testing.assert_array_equal(gpu[key], host[key])
        # one row, one column
        rows, cols, values = make_triplets(rng, 500, 1, 1, (1, 1), None, 0.0, 0.0, integer=True)
        host, gpu = self.both(lambda: run_native(self.device, 1, 1, (1, 1), rows, cols, values))
        self.assert_topology_matches(host, gpu, "1x1 matrix")
        # nrow * ncol >= 2**32: 64-bit sort keys
        nrow, ncol = 100_000, 70_000
        rows, cols, values = make_triplets(rng, 300_000, nrow, ncol, (1, 1), None, 0.01, 0.1, integer=True)
        rows[::3] = rows[1::3][: len(rows[::3])]  # duplicate some (row, column) pairs
        cols[::3] = cols[1::3][: len(cols[::3])]
        host, gpu = self.both(lambda: run_native(self.device, nrow, ncol, (1, 1), rows, cols, values))
        self.assert_topology_matches(host, gpu, "64-bit keys")
        # topology only (no values), as bsr_assign and bsr_axpy use it
        host, gpu = self.both(lambda: run_native(self.device, nrow, ncol, (1, 1), rows, cols, None))
        self.assert_topology_matches(host, gpu, "no values")

    def test_triplet_masked_topology(self):
        """The masked builder keeps only triplets that are blocks of the mask (bsr_assign / bsr_axpy masked)."""
        rng = np.random.default_rng(5)
        nrow, ncol = 60, 50
        for shape in ((1, 1), (3, 3)):
            m_rows, m_cols, m_values = make_triplets(rng, 800, nrow, ncol, shape, None, 0.0, 0.0, integer=True)
            with gpu_utilities(False):
                mask = assemble(self.device, nrow, ncol, shape, m_rows, m_cols, m_values)
            rows, cols, values = make_triplets(rng, 3000, nrow, ncol, shape, None, 0.1, 0.2, integer=True)
            host, gpu = self.both(
                lambda rows=rows, cols=cols, values=values, mask=mask, shape=shape: run_native(
                    self.device, nrow, ncol, shape, rows, cols, values, masked_by=mask
                )
            )
            self.assert_topology_matches(host, gpu, f"masked shape={shape}")

            # through the public API
            def axpy(mask=mask, rows=rows, cols=cols, values=values, shape=shape):
                x = assemble(self.device, nrow, ncol, shape, rows, cols, values)
                y = sparse.bsr_copy(mask)
                sparse.bsr_axpy(x, y, alpha=2.0, beta=1.0, topology="masked")
                z = sparse.bsr_copy(mask)
                sparse.bsr_axpy(x, z, alpha=1.0, beta=-1.0)
                return [matrix_arrays(m) for m in (x, y, z)]

            host, gpu = self.both(axpy)
            for h, g in zip(host, gpu, strict=True):
                for key in h:
                    np.testing.assert_array_equal(g[key], h[key], err_msg=f"axpy {key} shape={shape}")

    def test_transpose_matches_host(self):
        """Compact and padded transposes of compact and padded sources, including capacity overflow."""
        rng = np.random.default_rng(6)
        for shape in BLOCK_SHAPES:
            for n, nrow, ncol, dup in [
                (1, 3, 4, None),
                (2000, 50, 70, None),
                (30000, 900, 600, None),
                (5000, 40, 30, 6),
            ]:
                rows, cols, values = make_triplets(rng, n, nrow, ncol, shape, dup, 0.05, 0.1, integer=True)
                for src_topology in ("compact", "padded"):
                    for dest_topology in ("compact", "padded"):
                        for row_capacity in (64, 2):
                            if dest_topology == "compact" and row_capacity != 64:
                                continue
                            msg = f"shape={shape} n={n} src={src_topology} dest={dest_topology} cap={row_capacity}"

                            def fn(
                                rows=rows,
                                cols=cols,
                                values=values,
                                src_topology=src_topology,
                                dest_topology=dest_topology,
                                row_capacity=row_capacity,
                                nrow=nrow,
                                ncol=ncol,
                                shape=shape,
                            ):
                                src_dest = None
                                if src_topology == "padded":
                                    src_dest = sparse.bsr_zeros(
                                        nrow,
                                        ncol,
                                        block_type(shape),
                                        device=self.device,
                                        topology="padded",
                                        row_capacity=96,
                                    )
                                src = assemble(
                                    self.device,
                                    nrow,
                                    ncol,
                                    shape,
                                    rows,
                                    cols,
                                    values,
                                    None,
                                    True,
                                    src_topology,
                                    src_dest,
                                )
                                t_shape = shape[::-1]
                                if dest_topology == "padded":
                                    dest = sparse.bsr_zeros(
                                        ncol,
                                        nrow,
                                        block_type(t_shape),
                                        device=self.device,
                                        topology="padded",
                                        row_capacity=row_capacity,
                                    )
                                    dest.clear_status()
                                else:
                                    dest = sparse.bsr_zeros(ncol, nrow, block_type(t_shape), device=self.device)
                                sparse.bsr_set_transpose(dest, src * 2.0, topology=dest_topology)
                                out = matrix_arrays(dest)
                                out["status"] = np.array([dest.status_sync()])
                                return out

                            host, gpu = self.both(fn)
                            for key in host:
                                np.testing.assert_array_equal(gpu[key], host[key], err_msg=f"{key}: {msg}")

    def test_transpose_raw_outputs(self):
        """The native transpose outputs (block map, columns past the active blocks) as on the host."""
        from warp._src.context import runtime  # noqa: PLC0415

        rng = np.random.default_rng(7)
        nrow, ncol = 200, 150
        rows, cols, values = make_triplets(rng, 3000, nrow, ncol, (1, 1), None, 0.0, 0.0, integer=True)
        for padded_src in (False, True):

            def fn(padded_src=padded_src):
                src_dest = (
                    sparse.bsr_zeros(nrow, ncol, wp.float32, device=self.device, topology="padded", row_capacity=40)
                    if padded_src
                    else None
                )
                src = assemble(
                    self.device,
                    nrow,
                    ncol,
                    (1, 1),
                    rows,
                    cols,
                    values,
                    None,
                    True,
                    "padded" if padded_src else None,
                    src_dest,
                )
                dest = sparse.bsr_zeros(ncol, nrow, wp.float32, device=self.device)
                nnz = src.nnz
                _bsr_ensure_fits(dest, nnz=nnz)
                dest.columns.fill_(-5)
                block_map = wp.full(2 * nnz, -9, dtype=int, device=self.device)
                from warp._src import metal_sparse  # noqa: PLC0415

                if metal_sparse.enabled(self.device):
                    metal_sparse.bsr_transpose(
                        nrow,
                        ncol,
                        nnz,
                        src.offsets,
                        src.row_counts,
                        src.columns,
                        dest.offsets,
                        None,
                        dest.columns,
                        block_map,
                        None,
                        2,
                    )
                else:
                    import ctypes  # noqa: PLC0415

                    wp.synchronize_device(self.device)
                    p = lambda a: ctypes.cast(a.ptr, ctypes.POINTER(ctypes.c_int32))  # noqa: E731
                    runtime.core.wp_bsr_transpose_host(
                        nrow,
                        ncol,
                        nnz,
                        p(src.offsets),
                        p(src.row_counts) if src.row_counts is not None else None,
                        p(src.columns),
                        p(dest.offsets),
                        None,
                        p(dest.columns),
                        p(block_map),
                        None,
                    )
                active = int(dest.offsets.numpy()[ncol])
                return (
                    dest.offsets.numpy()[: ncol + 1].copy(),
                    dest.columns.numpy()[:nnz].copy(),
                    block_map.numpy()[:active].copy(),
                )

            host, gpu = self.both(fn)
            for h, g in zip(host, gpu, strict=True):
                np.testing.assert_array_equal(g, h, err_msg=f"padded_src={padded_src}")

    def test_graph_capture(self):
        """Triplet assembly and transpose recorded into a Metal graph replay on new triplets like the host."""
        device = self.device
        rng = np.random.default_rng(8)
        n, nrow, ncol, shape = 50_000, 3000, 2000, (3, 3)

        w_rows = wp.zeros(n, dtype=wp.int32, device=device)
        w_cols = wp.zeros(n, dtype=wp.int32, device=device)
        w_values = wp.zeros(n, dtype=block_type(shape), device=device)
        a = sparse.bsr_zeros(nrow, ncol, block_type(shape), device=device)
        at = sparse.bsr_zeros(ncol, nrow, block_type(shape), device=device)

        def pipeline():
            sparse.bsr_set_from_triplets(a, w_rows, w_cols, w_values)
            sparse.bsr_set_transpose(at, a)

        def fill():
            rows, cols, values = make_triplets(rng, n, nrow, ncol, shape, None, 0.05, 0.1, integer=True)
            w_rows.assign(rows)
            w_cols.assign(cols)
            w_values.assign(values)

        def outputs():
            return [matrix_arrays(m) for m in (a, at)]

        with gpu_utilities(True):
            fill()
            pipeline()  # loads the modules, sizes the matrices
            with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                pipeline()
            for _ in range(3):
                fill()
                wp.capture_launch(capture.graph)
                got = outputs()
                with gpu_utilities(False):
                    pipeline()
                    want = outputs()
                for g, w in zip(got, want, strict=True):
                    # the graph keeps the capture-time nnz upper bound; compare the active blocks
                    for key in ("offsets", "columns", "values"):
                        np.testing.assert_array_equal(g[key], w[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
