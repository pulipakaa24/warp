# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior specific to the Metal backend, exercised through the public API."""

import gc
import unittest

import numpy as np

import warp as wp
import warp.sparse
from warp.tests.unittest_utils import StdOutCapture


def metal_available() -> bool:
    return getattr(wp, "is_metal_available", lambda: False)()


@wp.kernel
def increment_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = a[i] + 1.0


@wp.kernel
def scale_kernel(a: wp.array[float], s: float):
    i = wp.tid()
    a[i] = a[i] * s


@wp.kernel
def saxpy_kernel(x: wp.array[float], y: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = 2.0 * x[i] + y[i]


@wp.struct
class ArrayHolder:
    values: wp.array[float]
    offset: float


@wp.kernel
def holder_sum_kernel(holders: wp.array[ArrayHolder], out: wp.array[float]):
    i = wp.tid()
    h = holders[i]
    out[i] = h.values[0] + h.values[1] + h.offset


@wp.kernel
def print_first_kernel():
    print("metal first string")


@wp.kernel
def print_second_kernel():
    print("metal second string")


@wp.kernel
def mat44_products_twice(a: wp.array[wp.mat44], b: wp.array[wp.mat44], out: wp.array[float]):
    c = a[0] * b[0]
    d = a[0] * b[0]
    for i in range(4):
        for j in range(4):
            out[i * 4 + j] = 2.0 * c[i, j]
            out[16 + i * 4 + j] = 3.0 * d[i, j]


@wp.kernel
def mat44_products_chain(a: wp.array[wp.mat44], b: wp.array[wp.mat44], out: wp.array[float]):
    c = a[0] * b[0]
    d = (c * b[0]) * (a[0] * c)
    for i in range(4):
        for j in range(4):
            out[i * 4 + j] = 2.0 * c[i, j]
            out[16 + i * 4 + j] = 3.0 * d[i, j]


@wp.kernel
def mat44_products_inverse(a: wp.array[wp.mat44], b: wp.array[wp.mat44], out: wp.array[float]):
    c = wp.inverse(a[0])
    d = (c * a[0]) * b[0]
    for i in range(4):
        for j in range(4):
            out[i * 4 + j] = 2.0 * c[i, j]
            out[16 + i * 4 + j] = 3.0 * d[i, j]


def _make_cholesky_kernel(n: int):
    """Factor and solve one ``n x n`` system per thread block with the tile Cholesky builtins."""

    @wp.kernel(enable_backward=False, module="unique")
    def factor_and_solve(a: wp.array3d[float], b: wp.array2d[float], factor: wp.array3d[float], x: wp.array2d[float]):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n, n), storage="shared")
        wp.tile_cholesky_inplace(t, fill_mode="upper")
        wp.tile_store(factor[w], t)
        rhs = wp.tile_load(b[w], shape=(n,))
        wp.tile_store(x[w], wp.tile_cholesky_solve(t, rhs, fill_mode="upper"))

    return factor_and_solve


def _make_tile_ops_kernel(n: int):
    """Reductions, scan and sort on one shared ``n``-element tile per thread block."""

    @wp.kernel(enable_backward=False, module="unique")
    def tile_ops(
        a: wp.array2d[float],
        keys_in: wp.array2d[int],
        total: wp.array2d[float],
        lowest: wp.array2d[float],
        highest: wp.array2d[float],
        arg_lowest: wp.array2d[int],
        arg_highest: wp.array2d[int],
        running: wp.array2d[float],
        keys_out: wp.array2d[int],
        order: wp.array2d[int],
    ):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n,), storage="shared")
        wp.tile_store(total[w], wp.tile_sum(t))
        wp.tile_store(lowest[w], wp.tile_min(t))
        wp.tile_store(highest[w], wp.tile_max(t))
        wp.tile_store(arg_lowest[w], wp.tile_argmin(t))
        wp.tile_store(arg_highest[w], wp.tile_argmax(t))
        wp.tile_store(running[w], wp.tile_scan_inclusive(t))
        keys = wp.tile_load(keys_in[w], shape=(n,), storage="shared")
        values = wp.tile_arange(n, dtype=int, storage="shared")
        wp.tile_sort(keys, values)
        wp.tile_store(keys_out[w], keys)
        wp.tile_store(order[w], values)

    return tile_ops


def _make_tile_matmul_kernel(n: int):
    """Compute the returning and the accumulating matrix product of shared ``n x n`` tiles."""

    @wp.kernel(enable_backward=False, module="unique")
    def tile_products(a: wp.array3d[float], b: wp.array3d[float], product: wp.array3d[float], acc: wp.array3d[float]):
        w = wp.tid()
        ta = wp.tile_load(a[w], shape=(n, n), storage="shared")
        tb = wp.tile_load(b[w], shape=(n, n), storage="shared")
        wp.tile_store(product[w], wp.tile_matmul(ta, tb))
        sum_tile = wp.tile_load(a[w], shape=(n, n), storage="shared")
        wp.tile_matmul(ta, tb, sum_tile)
        wp.tile_store(acc[w], sum_tile)

    return tile_products


# Newton's VBD interval arithmetic (newton/_src/solvers/vbd/interval_arithmetic.py) calls nextafterf through the
# CPU branch of a native snippet on Metal; the Metal library has no nextafterf (air64 link error before the fix).
@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
return nextafterf(value, INFINITY);
#else
return __builtin_nextafterf(value, INFINITY);
#endif
"""
)
def next_float_up(value: float) -> float: ...


@wp.func_native("return __builtin_nextafterf(value, toward);")
def next_float_toward(value: float, toward: float) -> float: ...


@wp.kernel
def nextafter_kernel(x: wp.array[float], y: wp.array[float], up: wp.array[float], toward: wp.array[float]):
    i = wp.tid()
    up[i] = next_float_up(x[i])
    toward[i] = next_float_toward(x[i], y[i])


# ---------------------------------------------------------------------------------------------------
# Reference LBVH: a direct NumPy transcription of LinearBVHBuilderGPU::build and bvh_refit_kernel in
# warp/native/bvh.cu (Warp 1.16.0), for checking the tree a Metal device builds bit for bit.

_NODE_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("ib", "<u4")])


def _cuda_fmin(a, b):
    """``fminf`` as the GPU computes it: NaN is missing, -0.0 orders below +0.0."""
    out = np.fmin(a, b)
    zero = (a == 0) & (b == 0)
    return np.where(zero, np.where(np.signbit(a) | np.signbit(b), np.float32(-0.0), np.float32(0.0)), out)


def _cuda_fmax(a, b):
    out = np.fmax(a, b)
    zero = (a == 0) & (b == 0)
    return np.where(zero, np.where(np.signbit(a) & np.signbit(b), np.float32(-0.0), np.float32(0.0)), out)


def _part1by2(n):
    n = n.astype(np.uint32)
    n = (n ^ (n << np.uint32(16))) & np.uint32(0xFF0000FF)
    n = (n ^ (n << np.uint32(8))) & np.uint32(0x0300F00F)
    n = (n ^ (n << np.uint32(4))) & np.uint32(0x030C30C3)
    n = (n ^ (n << np.uint32(2))) & np.uint32(0x09249249)
    return n


def _clz64(x):
    return 64 if x == 0 else 64 - int(x).bit_length()


def _reference_lbvh(item_lowers, item_uppers, groups=None, leaf_size=1):
    lo = np.asarray(item_lowers, dtype=np.float32)
    up = np.asarray(item_uppers, dtype=np.float32)
    n = len(lo)
    m = 2 * n - 1

    # compute_total_bounds, compute_total_inv_edges
    total_lower = np.full(3, np.finfo(np.float32).max, dtype=np.float32)
    total_upper = -total_lower
    for i in range(n):
        total_lower = _cuda_fmin(total_lower, lo[i])
        total_upper = _cuda_fmax(total_upper, up[i])
    edges = (total_upper - total_lower) + np.float32(0.0001)
    inv_edges = np.float32(1.0) / edges

    # compute_morton_codes
    center = np.float32(0.5) * (lo + up)
    local = (center - total_lower) * inv_edges
    q = np.clip((local * np.float32(1024)).astype(np.int32), 0, 1023)
    morton = (_part1by2(q[:, 2]) << np.uint32(2)) | (_part1by2(q[:, 1]) << np.uint32(1)) | _part1by2(q[:, 0])
    group = np.zeros(n, dtype=np.uint64) if groups is None else np.asarray(groups).astype(np.uint32).astype(np.uint64)
    unsorted_keys = (group << np.uint64(32)) | morton.astype(np.uint64)

    # stable radix sort of (key, index) pairs
    indices = np.argsort(unsorted_keys, kind="stable").astype(np.int32)
    keys = [int(k) for k in unsorted_keys[indices]]
    prims = [int(i) for i in indices]
    deltas = [_clz64(keys[i] ^ keys[i + 1]) for i in range(n - 1)]

    # build_leaves
    lower_b = np.zeros((m, 3), dtype=np.float32)
    upper_b = np.zeros((m, 3), dtype=np.float32)
    lower_i = [0] * m
    upper_i = [0] * m
    leaf = [False] * m
    lower_b[:n] = lo[indices]
    upper_b[:n] = up[indices]
    for i in range(n):
        lower_i[i] = upper_i[i] = i
        leaf[i] = True
    range_lefts = list(range(n)) + [0] * (n - 1)
    range_rights = list(range(n)) + [0] * (n - 1)
    parents = [0] * m
    num_children = [0] * m
    root = -1

    # build_hierarchy
    for thread in range(n):
        index = thread
        while True:
            left, right = range_lefts[index], range_rights[index]
            if left == 0 and right == n - 1:
                root = index
                parents[index] = -1
                break
            if left == 0:
                parent_right = True
            else:
                decided = False
                if keys[left] >> 32 == keys[right] >> 32:
                    g = keys[left] >> 32
                    go_right = right < n - 1 and keys[right + 1] >> 32 == g
                    go_left = keys[left - 1] >> 32 == g
                    if go_right != go_left:
                        parent_right = go_right
                        decided = True
                if not decided:
                    if right != n - 1 and deltas[right] >= deltas[left - 1]:
                        if deltas[right] == deltas[left - 1]:
                            parent_right = bool((prims[left - 1] % 2) ^ (prims[right] % 2))
                        else:
                            parent_right = True
                    else:
                        parent_right = False
            if parent_right:
                parent = right + n
                lower_i[parent] = index
                range_lefts[parent] = left
            else:
                parent = left + n - 1
                upper_i[parent] = index
                range_rights[parent] = right
            parents[index] = parent
            num_children[parent] += 1
            if num_children[parent] != 2:
                break
            lc, rc = lower_i[parent], upper_i[parent]
            lower_b[parent] = _cuda_fmin(lower_b[lc], lower_b[rc])
            upper_b[parent] = _cuda_fmax(upper_b[lc], upper_b[rc])
            leaf[parent] = False
            index = parent

    # mark_packed_leaf_nodes
    for node in range(m):
        depth, p = 1, parents[node]
        while p != -1:
            p = parents[p]
            depth += 1
        left, right = range_lefts[node], range_rights[node] + 1
        if keys[left] >> 32 == keys[right - 1] >> 32 and (right - left <= leaf_size or depth >= 32):
            leaf[node] = True
            lower_i[node] = left
            upper_i[node] = right

    return {
        "lower_b": lower_b,
        "upper_b": upper_b,
        "lower_i": np.array(lower_i, dtype=np.uint32),
        "upper_i": np.array(upper_i, dtype=np.uint32),
        "leaf": np.array(leaf, dtype=bool),
        "parents": np.array(parents, dtype=np.int32),
        "prims": np.array(prims, dtype=np.int32),
        "root": root,
    }


def _reference_lbvh_refit(tree, item_lowers, item_uppers):
    """bvh_refit_kernel of warp/native/bvh.cu, one leaf thread after the other."""
    lo = np.asarray(item_lowers, dtype=np.float32)
    up = np.asarray(item_uppers, dtype=np.float32)
    t = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tree.items()}
    leaf, parents = t["leaf"], t["parents"]
    n = len(t["prims"])

    def leaf_bounds(node):
        b_lo = np.full(3, np.finfo(np.float32).max, dtype=np.float32)
        b_up = -b_lo
        for c in range(t["lower_i"][node], t["upper_i"][node]):
            p = t["prims"][c]
            b_lo = np.where(lo[p] < b_lo, lo[p], b_lo)  # bounds3::add_bounds (std_min / std_max)
            b_up = np.where(up[p] > b_up, up[p], b_up)
        t["lower_b"][node], t["upper_b"][node] = b_lo, b_up

    count = np.zeros(len(leaf), dtype=np.int32)
    for index in range(n):
        if not leaf[index]:
            continue
        parent = parents[index]
        if parent == -1 or not leaf[parent]:
            leaf_bounds(index)
        while True:
            parent = parents[index]
            if parent == -1:
                break
            count[parent] += 1
            if count[parent] != 2:
                break
            if leaf[parent]:
                if parents[parent] == -1 or not leaf[parents[parent]]:
                    leaf_bounds(parent)
            else:
                lc, rc = t["lower_i"][parent], t["upper_i"][parent]
                t["lower_b"][parent] = _cuda_fmin(t["lower_b"][lc], t["lower_b"][rc])
                t["upper_b"][parent] = _cuda_fmax(t["upper_b"][lc], t["upper_b"][rc])
            index = parent
    return t


def _read_tree(obj, is_mesh):
    """The tree arrays of a Bvh or Mesh, through ``wp_bvh_get_tree``."""
    core = wp._src.context.runtime.core
    on_device = 0 if obj.device.is_cpu else 1
    info = np.zeros(6, dtype=np.int32)
    if not core.wp_bvh_get_tree(obj.id, on_device, int(is_mesh), info.ctypes.data, None, None, None, None):
        raise RuntimeError("wp_bvh_get_tree failed")
    max_nodes, num_items = int(info[0]), int(info[1])
    lowers = np.zeros(max_nodes, dtype=_NODE_DTYPE)
    uppers = np.zeros(max_nodes, dtype=_NODE_DTYPE)
    parents = np.zeros(max_nodes, dtype=np.int32)
    prims = np.zeros(num_items, dtype=np.int32)
    core.wp_bvh_get_tree(
        obj.id,
        on_device,
        int(is_mesh),
        info.ctypes.data,
        lowers.ctypes.data,
        uppers.ctypes.data,
        parents.ctypes.data,
        prims.ctypes.data,
    )
    return {
        "lower_b": np.stack([lowers["x"], lowers["y"], lowers["z"]], axis=1),
        "upper_b": np.stack([uppers["x"], uppers["y"], uppers["z"]], axis=1),
        "lower_i": lowers["ib"] & np.uint32(0x7FFFFFFF),
        "upper_i": uppers["ib"] & np.uint32(0x7FFFFFFF),
        "leaf": (lowers["ib"] >> np.uint32(31)).astype(bool),
        "upper_flag": (uppers["ib"] >> np.uint32(31)).astype(bool),
        "parents": parents,
        "prims": prims,
        "root": int(info[2]),
        "num_leaf_nodes": int(info[3]),
        "constructor": int(info[4]),
    }


def _assert_same_tree(test, tree, ref, msg=""):
    test.assertEqual(tree["root"], ref["root"], msg)
    np.testing.assert_array_equal(tree["prims"], ref["prims"], err_msg=msg)
    np.testing.assert_array_equal(tree["parents"], ref["parents"], err_msg=msg)
    np.testing.assert_array_equal(tree["leaf"], ref["leaf"], err_msg=msg)
    test.assertFalse(tree["upper_flag"].any(), msg)
    np.testing.assert_array_equal(tree["lower_i"], ref["lower_i"], err_msg=msg)
    np.testing.assert_array_equal(tree["upper_i"], ref["upper_i"], err_msg=msg)
    # bit for bit, signed zeros included
    np.testing.assert_array_equal(tree["lower_b"].view(np.uint32), ref["lower_b"].view(np.uint32), err_msg=msg)
    np.testing.assert_array_equal(tree["upper_b"].view(np.uint32), ref["upper_b"].view(np.uint32), err_msg=msg)


def _box_mesh(res, half_extents=(0.5, 0.3, 0.2)):
    """A closed box surface with ``res`` x ``res`` quads per face (as two triangles each)."""
    points, tris = [], []
    hx = np.asarray(half_extents, dtype=np.float32)
    for axis in range(3):
        for side in (-1.0, 1.0):
            u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3
            base = len(points)
            for a in range(res + 1):
                for b in range(res + 1):
                    p = np.zeros(3, dtype=np.float32)
                    p[axis] = side * hx[axis]
                    p[u_axis] = (2.0 * a / res - 1.0) * hx[u_axis]
                    p[v_axis] = (2.0 * b / res - 1.0) * hx[v_axis]
                    points.append(p)
            for a in range(res):
                for b in range(res):
                    i0 = base + a * (res + 1) + b
                    i1, i2, i3 = i0 + 1, i0 + res + 1, i0 + res + 2
                    if side > 0:
                        tris += [i0, i2, i1, i1, i2, i3]
                    else:
                        tris += [i0, i1, i2, i1, i3, i2]
    return np.array(points, dtype=np.float32), np.array(tris, dtype=np.int32)


def _triangle_bounds(points, tris):
    t = points[tris.reshape(-1, 3)]
    lower = _cuda_fmin(_cuda_fmin(t[:, 0], t[:, 1]), t[:, 2])
    upper = _cuda_fmax(_cuda_fmax(t[:, 0], t[:, 1]), t[:, 2])
    return lower, upper


@wp.kernel
def mesh_closest_kernel(
    mesh: wp.uint64, pts: wp.array[wp.vec3], face: wp.array[int], dist: wp.array[float], max_dist: float
):
    i = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh, pts[i], max_dist)
    if q.result:
        face[i] = q.face
        dist[i] = wp.length(wp.mesh_eval_position(mesh, q.face, q.u, q.v) - pts[i])
    else:
        face[i] = -1
        dist[i] = -1.0


@wp.kernel
def mesh_winding_sign_kernel(mesh: wp.uint64, pts: wp.array[wp.vec3], sign: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, pts[i], 10.0)
    sign[i] = q.sign


def _lbvh_cases():
    """(name, lowers, uppers, groups) item bounds exercising the LBVH builder's branches."""
    rng = np.random.default_rng(7)
    cases = []
    c = rng.uniform(-1.0, 1.0, (1000, 3)).astype(np.float32)
    e = rng.uniform(0.001, 0.05, (1000, 3)).astype(np.float32)
    cases.append(("random", c - e, c + e, None))
    # many equal Morton keys: the parity tie-break and the depth limit of leaf packing
    c = (rng.integers(0, 4, (500, 3)) * 0.25).astype(np.float32)
    cases.append(("duplicates", c - np.float32(0.01), c + np.float32(0.01), None))
    same = np.zeros((200, 3), dtype=np.float32)
    cases.append(("identical", same - np.float32(0.5), same + np.float32(0.5), None))
    # unsorted group ids
    c = rng.uniform(-1.0, 1.0, (600, 3)).astype(np.float32)
    cases.append(("grouped", c - np.float32(0.02), c + np.float32(0.02), rng.integers(0, 5, 600).astype(np.int32)))
    # signed zeros in the bounds
    z = np.array([[-0.0, 0.0, -0.0], [0.0, -0.0, 0.0], [-0.0, -0.0, 0.0], [0.5, 0.0, -0.0]], dtype=np.float32)
    cases.append(("signed_zeros", z, z + np.array([0.0, 0.0, 0.0], dtype=np.float32), None))
    for n in (1, 2, 3):
        c = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
        cases.append((f"n={n}", c - np.float32(0.1), c + np.float32(0.1), None))
    return cases


class TestMetalLbvhHost(unittest.TestCase):
    """The host LBVH builder (used for Metal trees) against the CUDA reference, through the host entry point."""

    def test_host_lbvh_matches_cuda_build(self):
        core = wp._src.context.runtime.core
        for name, lowers, uppers, groups in _lbvh_cases():
            lo = np.ascontiguousarray(lowers, dtype=np.float32)
            up = np.ascontiguousarray(uppers, dtype=np.float32)
            gr = None if groups is None else np.ascontiguousarray(groups, dtype=np.int32)
            bvh_id = core.wp_bvh_create_host(
                lo.ctypes.data, up.ctypes.data, len(lo), 2, None if gr is None else gr.ctypes.data, 4
            )
            self.assertNotEqual(bvh_id, 0, name)
            try:
                holder = type("HostBvh", (), {"id": bvh_id, "device": wp.get_device("cpu")})()
                _assert_same_tree(self, _read_tree(holder, is_mesh=False), _reference_lbvh(lo, up, gr, 4), name)
            finally:
                core.wp_bvh_destroy_host(bvh_id)


@unittest.skipUnless(metal_available(), "Requires an Apple GPU")
class TestMetal(unittest.TestCase):
    device = "metal:0"

    def test_nextafterf(self):
        """nextafterf and __builtin_nextafterf in native snippets: bit-exact against numpy and the CPU device."""
        inf, nan = np.float32(np.inf), np.float32(np.nan)
        tiny = np.float32(1.4e-45)
        x = np.array([0.0, -0.0, 1.0, -1.0, 1.5e-38, -tiny, tiny, 3.4028235e38, -3.4028235e38, 0.1, -2.5, 1e-7, nan, inf, -inf, 7.0], np.float32)
        y = np.array([1.0, -1.0, 0.0, 0.0, -1.0, 1.0, -1.0, inf, -inf, 0.1, -inf, 1.0, 0.0, 0.0, 0.0, nan], np.float32)
        with np.errstate(over="ignore"):
            ref_up = np.nextafter(x, np.full_like(x, inf))
            ref_toward = np.nextafter(x, y)
        for device in (self.device, "cpu"):
            xs, ys = wp.array(x, device=device), wp.array(y, device=device)
            up, toward = wp.zeros_like(xs), wp.zeros_like(xs)
            wp.launch(nextafter_kernel, dim=len(x), inputs=[xs, ys, up, toward], device=device)
            np.testing.assert_array_equal(up.numpy().view(np.uint32)[~np.isnan(ref_up)], ref_up.view(np.uint32)[~np.isnan(ref_up)], err_msg=device)
            np.testing.assert_array_equal(np.isnan(up.numpy()), np.isnan(ref_up), err_msg=device)
            fin = ~np.isnan(ref_toward)
            np.testing.assert_array_equal(toward.numpy().view(np.uint32)[fin], ref_toward.view(np.uint32)[fin], err_msg=device)
            np.testing.assert_array_equal(np.isnan(toward.numpy()), np.isnan(ref_toward), err_msg=device)

    def test_host_memory_prefix_then_whole(self):
        """A host range that starts inside an imported range but extends past it is mapped completely."""
        page = 16384
        n = (3 * page) // 4  # three pages of floats: the prefix import covers only the first page
        data = np.zeros(n, dtype=np.float32)
        a = wp.array(data, dtype=float, device="cpu", copy=False)
        prefix = a[:4]
        wp.launch(increment_kernel, dim=4, inputs=[prefix], device=self.device)
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        expected = np.ones(n, dtype=np.float32)
        expected[:4] = 2.0
        np.testing.assert_array_equal(data, expected)

        # releasing the prefix must not take the mapping of the whole array with it
        del prefix
        gc.collect()
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        np.testing.assert_array_equal(data, expected + 1.0)

    def test_host_memory_inner_page_then_whole(self):
        """An import in the middle of a larger host array does not shadow the rest of that array."""
        page = 16384
        n = (4 * page) // 4
        data = np.zeros(n, dtype=np.float32)
        a = wp.array(data, dtype=float, device="cpu", copy=False)
        first = page // 4 + 8
        inner = a[first : first + 4]  # lies in the second page only
        wp.launch(increment_kernel, dim=4, inputs=[inner], device=self.device)
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        del inner
        gc.collect()
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        expected = np.full(n, 2.0, dtype=np.float32)
        expected[first : first + 4] = 3.0
        np.testing.assert_array_equal(data, expected)

    def test_bsr_topology_inside_capture_raises(self):
        """With the host BSR path (``metal_gpu_utilities`` off), BSR topology cannot be replayed by a Metal graph, so it
        refuses to run in a capture. The GPU path is capturable (``TestMetalSparse``)."""
        rows = wp.array([0, 1], dtype=int, device=self.device)
        cols = wp.array([0, 1], dtype=int, device=self.device)
        vals = wp.array([1.0, 2.0], dtype=float, device=self.device)
        m = wp.sparse.bsr_zeros(2, 2, block_type=float, device=self.device)
        saved = wp.config.metal_gpu_utilities
        wp.config.metal_gpu_utilities = False
        try:
            with self.assertRaisesRegex(RuntimeError, "inside a graph capture"):
                with wp.ScopedCapture(device=self.device):
                    wp.sparse.bsr_set_from_triplets(m, rows, cols, vals)
        finally:
            wp.config.metal_gpu_utilities = saved
        self.assertFalse(wp.get_device(self.device).is_capturing)
        wp.sparse.bsr_set_from_triplets(m, rows, cols, vals)  # fine outside a capture
        self.assertEqual(m.nnz_sync(), 2)

    def test_scalar_arguments_are_not_imported(self):
        """NumPy scalars passed by value are not treated as host arrays."""
        device = wp.get_device(self.device)
        a = wp.ones(8, dtype=float, device=self.device)
        wp.launch(scale_kernel, dim=8, inputs=[a, np.float32(3.0)], device=self.device)
        self.assertFalse(device.__dict__.get("_metal_host_memory_pending", False))
        np.testing.assert_array_equal(a.numpy(), np.full(8, 3.0, dtype=np.float32))

    def test_graph_keeps_capture_allocations(self):
        """Arrays allocated during a capture stay valid for replays after Python released them."""
        n = 1024
        x = wp.array(np.arange(n, dtype=np.float32), device=self.device)
        out = wp.zeros(n, dtype=float, device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            y = wp.ones(n, dtype=float, device=self.device)  # lives only inside the capture
            wp.launch(saxpy_kernel, dim=n, inputs=[x, y, out], device=self.device)
        del y
        # reuse whatever memory a released buffer would hand back
        junk = [wp.full(n, 7.0, dtype=float, device=self.device) for _ in range(8)]
        out.zero_()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(out.numpy(), 2.0 * np.arange(n, dtype=np.float32) + 1.0)
        del junk

    def test_capture_survives_failing_branch(self):
        """A branch body that raises leaves the device usable for normal launches and new captures."""
        cond = wp.ones(1, dtype=int, device=self.device)
        a = wp.zeros(4, dtype=float, device=self.device)

        def failing_body():
            raise ValueError("branch body failed")

        with self.assertRaises(ValueError):
            with wp.ScopedCapture(device=self.device):
                wp.capture_if(cond, on_true=failing_body)
        self.assertFalse(wp.get_device(self.device).is_capturing)
        wp.launch(increment_kernel, dim=4, inputs=[a], device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            wp.launch(increment_kernel, dim=4, inputs=[a], device=self.device)
        wp.capture_launch(capture.graph)
        # one eager launch plus one replay: launches inside a capture are recorded, not executed
        np.testing.assert_array_equal(a.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_replay_translates_nested_arrays_after_new_allocations(self):
        """Graph replays resolve array descriptors stored in device memory after the allocation set changed."""
        values = [wp.array([1.0 + i, 10.0 * (i + 1)], dtype=float, device=self.device) for i in range(4)]
        items = []
        for i, v in enumerate(values):
            h = ArrayHolder()
            h.values = v
            h.offset = float(i)
            items.append(h)
        holders = wp.array(items, dtype=ArrayHolder, device=self.device)
        out = wp.zeros(4, dtype=float, device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            wp.launch(holder_sum_kernel, dim=4, inputs=[holders, out], device=self.device)
        extra = [wp.zeros(4096, dtype=float, device=self.device) for _ in range(4)]  # changes the address table
        out.zero_()
        wp.capture_launch(capture.graph)
        expected = np.array([1.0 + i + 10.0 * (i + 1) + i for i in range(4)], dtype=np.float32)
        np.testing.assert_array_equal(out.numpy(), expected)
        del extra

    def test_print_strings_are_per_kernel(self):
        """String constants with the same generated name in different kernels print their own text."""
        capture = StdOutCapture()
        capture.begin()
        wp.launch(print_first_kernel, dim=1, inputs=[], device=self.device)
        wp.launch(print_second_kernel, dim=1, inputs=[], device=self.device)
        wp.synchronize_device(self.device)
        output = capture.end()
        self.assertIn("metal first string", output)
        self.assertIn("metal second string", output)

    def test_to_torch_attaches_gradient(self):
        try:
            import torch  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("Requires PyTorch")
        a = wp.ones(4, dtype=float, device=self.device, requires_grad=True)
        a.grad.fill_(2.0)
        t = wp.to_torch(a)
        self.assertIsNotNone(t.grad)
        self.assertEqual(t.grad.data_ptr(), a.grad.ptr)
        np.testing.assert_array_equal(t.grad.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_mat44_product_gradients_match_cpu(self):
        """Several 4x4 products in one kernel used to lose the gradient of one operand on Metal."""
        rng = np.random.default_rng(7)
        a_np = (rng.standard_normal((1, 4, 4)) + 3.0 * np.eye(4)).astype(np.float32)
        b_np = rng.standard_normal((1, 4, 4)).astype(np.float32)
        for kernel in (mat44_products_twice, mat44_products_chain, mat44_products_inverse):
            for selected in (0, 17, 31):
                gradients = {}
                for device in ("cpu", self.device):
                    a = wp.array(a_np, dtype=wp.mat44, requires_grad=True, device=device)
                    b = wp.array(b_np, dtype=wp.mat44, requires_grad=True, device=device)
                    out = wp.zeros(32, dtype=float, requires_grad=True, device=device)
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[a, b], outputs=[out], device=device)
                    seed = np.zeros(32, dtype=np.float32)
                    seed[selected] = 1.0
                    tape.backward(grads={out: wp.array(seed, dtype=float, device=device)})
                    gradients[device] = (tape.gradients[a].numpy(), tape.gradients[b].numpy())
                for on_cpu, on_metal in zip(gradients["cpu"], gradients[self.device], strict=True):
                    np.testing.assert_allclose(
                        on_metal, on_cpu, rtol=1e-3, atol=1e-4, err_msg=f"{kernel.key}[{selected}]"
                    )

    def test_tile_cholesky_larger_than_block(self):
        """A matrix larger than the launch block size gives each lane several columns of the factor.

        The register Cholesky used to zero the mirrored cell, which belongs to another lane's column, so
        the factor was wrong whenever ``n > block_dim`` (up to the size where the register path ends).
        MuJoCo Warp hits this with any model of more than 32 degrees of freedom.
        """
        rng = np.random.default_rng(0)
        worlds = 4
        for n in (27, 33, 35, 40):
            m = rng.standard_normal((worlds, n, n)).astype(np.float32)
            a_np = m @ m.transpose(0, 2, 1) + n * np.eye(n, dtype=np.float32)
            b_np = rng.standard_normal((worlds, n)).astype(np.float32)
            x_ref = np.linalg.solve(a_np.astype(np.float64), b_np.astype(np.float64)[..., None])[..., 0]
            u_ref = np.linalg.cholesky(a_np.astype(np.float64)).transpose(0, 2, 1)
            kernel = _make_cholesky_kernel(n)
            for block_dim in (16, 32, 64):
                a = wp.array(a_np, dtype=float, device=self.device)
                b = wp.array(b_np, dtype=float, device=self.device)
                factor = wp.zeros((worlds, n, n), dtype=float, device=self.device)
                x = wp.zeros((worlds, n), dtype=float, device=self.device)
                wp.launch_tiled(kernel, dim=[worlds], inputs=[a, b, factor, x], device=self.device, block_dim=block_dim)
                msg = f"n={n}, block_dim={block_dim}"
                np.testing.assert_allclose(factor.numpy(), u_ref, rtol=1e-4, atol=1e-4, err_msg=msg)
                np.testing.assert_allclose(x.numpy(), x_ref, rtol=1e-3, atol=1e-4, err_msg=msg)

    def test_tile_cholesky_register_bound_configurable(self):
        """``warp.config.metal_register_cholesky_max`` moves matrices above the default bound (40) onto the
        register path (MuJoCo Warp's G1 has 43 dofs); the factor and solve must still match numpy."""
        rng = np.random.default_rng(1)
        worlds = 4
        old = wp.config.metal_register_cholesky_max
        try:
            wp.config.metal_register_cholesky_max = 64
            for n in (41, 43, 48, 64):
                m = rng.standard_normal((worlds, n, n)).astype(np.float32)
                a_np = m @ m.transpose(0, 2, 1) + n * np.eye(n, dtype=np.float32)
                b_np = rng.standard_normal((worlds, n)).astype(np.float32)
                x_ref = np.linalg.solve(a_np.astype(np.float64), b_np.astype(np.float64)[..., None])[..., 0]
                u_ref = np.linalg.cholesky(a_np.astype(np.float64)).transpose(0, 2, 1)
                kernel = _make_cholesky_kernel(n)
                a = wp.array(a_np, dtype=float, device=self.device)
                b = wp.array(b_np, dtype=float, device=self.device)
                factor = wp.zeros((worlds, n, n), dtype=float, device=self.device)
                x = wp.zeros((worlds, n), dtype=float, device=self.device)
                wp.launch_tiled(kernel, dim=[worlds], inputs=[a, b, factor, x], device=self.device, block_dim=32)
                np.testing.assert_allclose(factor.numpy(), u_ref, rtol=1e-4, atol=1e-4, err_msg=f"n={n}")
                np.testing.assert_allclose(x.numpy(), x_ref, rtol=1e-3, atol=1e-4, err_msg=f"n={n}")
        finally:
            wp.config.metal_register_cholesky_max = old

    def test_tile_ops_across_block_boundary(self):
        """Tile sizes just below, at and above the block size, and above two blocks.

        Operations with a Metal-specific implementation split a tile over the lanes of a block, so the
        interesting sizes are the ones where the number of elements per lane changes.
        """
        rng = np.random.default_rng(2)
        worlds = 3
        for block_dim in (16, 32, 64):
            for n in (block_dim - 1, block_dim, block_dim + 1, 2 * block_dim + 1):
                a_np = rng.standard_normal((worlds, n)).astype(np.float32)
                keys_np = rng.permutation(worlds * n).reshape(worlds, n).astype(np.int32)
                expected = {
                    "total": a_np.sum(1, keepdims=True),
                    "lowest": a_np.min(1, keepdims=True),
                    "highest": a_np.max(1, keepdims=True),
                    "arg_lowest": a_np.argmin(1)[:, None],
                    "arg_highest": a_np.argmax(1)[:, None],
                    "running": np.cumsum(a_np.astype(np.float64), 1),
                    "keys_out": np.sort(keys_np, 1),
                    "order": np.argsort(keys_np, 1),
                }
                out = {
                    name: wp.zeros(ref.shape, dtype=int if ref.dtype.kind == "i" else float, device=self.device)
                    for name, ref in expected.items()
                }
                wp.launch_tiled(
                    _make_tile_ops_kernel(n),
                    dim=[worlds],
                    inputs=[
                        wp.array(a_np, dtype=float, device=self.device),
                        wp.array(keys_np, dtype=int, device=self.device),
                        *out.values(),
                    ],
                    device=self.device,
                    block_dim=block_dim,
                )
                for name, ref in expected.items():
                    np.testing.assert_allclose(
                        out[name].numpy(), ref, rtol=1e-4, atol=1e-4, err_msg=f"{name}, n={n}, block_dim={block_dim}"
                    )

    def test_tile_matmul_across_block_boundary(self):
        rng = np.random.default_rng(3)
        worlds = 3
        for block_dim in (16, 32):
            for n in (block_dim - 1, block_dim, block_dim + 1):
                a_np = rng.standard_normal((worlds, n, n)).astype(np.float32)
                b_np = rng.standard_normal((worlds, n, n)).astype(np.float32)
                product = wp.zeros((worlds, n, n), dtype=float, device=self.device)
                acc = wp.zeros((worlds, n, n), dtype=float, device=self.device)
                wp.launch_tiled(
                    _make_tile_matmul_kernel(n),
                    dim=[worlds],
                    inputs=[
                        wp.array(a_np, dtype=float, device=self.device),
                        wp.array(b_np, dtype=float, device=self.device),
                        product,
                        acc,
                    ],
                    device=self.device,
                    block_dim=block_dim,
                )
                ref = a_np.astype(np.float64) @ b_np
                msg = f"n={n}, block_dim={block_dim}"
                np.testing.assert_allclose(product.numpy(), ref, rtol=1e-3, atol=1e-3, err_msg=msg)
                np.testing.assert_allclose(acc.numpy(), a_np + ref, rtol=1e-3, atol=1e-3, err_msg=msg)

    def test_fem_dispatch_bilinear_wide_blocks(self):
        """Dispatched bilinear assembly with 6x6 node blocks (Newton's implicit MPM).

        The dispatch kernel runs with one lane per block off CUDA. Its tile reduction must be compiled out
        then, or it reserves 256 lanes of 6x6 tile scratch (about 75 KB) and the launch exceeds the 32 KB
        threadgroup limit.
        """
        import warp.fem as fem  # noqa: PLC0415

        @fem.integrand
        def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
            return wp.dot(u(s), v(s))

        results = {}
        for device in (self.device, "cpu"):
            with wp.ScopedDevice(device):
                geo = fem.Grid3D(res=wp.vec3i(2, 2, 2))
                space = fem.make_polynomial_space(geo, degree=1, dtype=wp.types.vector(6, wp.float32))
                domain = fem.Cells(geo)
                u = fem.make_trial(space, domain=domain)
                v = fem.make_test(space, domain=domain)
                mat = fem.integrate(
                    mass_form,
                    fields={"u": u, "v": v},
                    assembly="dispatch",
                    accumulate_dtype=wp.float32,
                    output_dtype=wp.float32,
                )
                results[device] = warp.sparse.bsr_get_diag(mat).numpy()
        np.testing.assert_allclose(results[self.device], results["cpu"], rtol=1e-6, atol=1e-7)
        self.assertGreater(np.abs(results["cpu"]).max(), 0.0)

    def test_lbvh_tree_matches_cuda_build(self):
        """A Metal LBVH is bitwise the tree the CUDA LBVH kernels build (reference: ``_reference_lbvh``)."""
        for name, lowers, uppers, groups in _lbvh_cases():
            for leaf_size in (1, 4):
                msg = f"{name}, leaf_size={leaf_size}"
                with self.subTest(case=name, leaf_size=leaf_size):
                    lo = wp.array(lowers, dtype=wp.vec3, device=self.device)
                    up = wp.array(uppers, dtype=wp.vec3, device=self.device)
                    gr = None if groups is None else wp.array(groups, dtype=wp.int32, device=self.device)
                    bvh = wp.Bvh(lo, up, groups=gr, leaf_size=leaf_size)  # default constructor: LBVH, as on CUDA
                    tree = _read_tree(bvh, is_mesh=False)
                    self.assertEqual(tree["constructor"], 2, msg)
                    self.assertEqual(tree["num_leaf_nodes"], len(lowers), msg)
                    _assert_same_tree(self, tree, _reference_lbvh(lowers, uppers, groups, leaf_size), msg)

    def test_lbvh_refit_and_rebuild_match_cuda(self):
        rng = np.random.default_rng(3)
        _, lowers, uppers, _ = _lbvh_cases()[0]
        moved_lo = (lowers + rng.normal(0.0, 0.05, lowers.shape)).astype(np.float32)
        moved_up = np.maximum(moved_lo, (uppers + rng.normal(0.0, 0.05, uppers.shape)).astype(np.float32))

        lo = wp.array(lowers, dtype=wp.vec3, device=self.device)
        up = wp.array(uppers, dtype=wp.vec3, device=self.device)
        bvh = wp.Bvh(lo, up, constructor="lbvh", leaf_size=4)
        ref = _reference_lbvh(lowers, uppers, None, 4)
        lo.assign(moved_lo)
        up.assign(moved_up)
        bvh.refit()
        _assert_same_tree(self, _read_tree(bvh, is_mesh=False), _reference_lbvh_refit(ref, moved_lo, moved_up), "refit")

        # in-place rebuild is LBVH, as on CUDA, also for a tree built with SAH
        lo.assign(lowers)
        up.assign(uppers)
        sah = wp.Bvh(lo, up, constructor="sah", leaf_size=4)
        self.assertEqual(_read_tree(sah, is_mesh=False)["constructor"], 0)
        lo.assign(moved_lo)
        up.assign(moved_up)
        for tree_bvh in (bvh, sah):
            tree_bvh.rebuild()
            tree = _read_tree(tree_bvh, is_mesh=False)
            self.assertEqual(tree["constructor"], 2)
            _assert_same_tree(self, tree, _reference_lbvh(moved_lo, moved_up, None, 4), "rebuild")

    def test_lbvh_mesh_matches_cuda_build(self):
        points, tris = _box_mesh(8)
        pts = wp.array(points, dtype=wp.vec3, device=self.device)
        mesh = wp.Mesh(points=pts, indices=wp.array(tris, dtype=wp.int32, device=self.device))  # default: LBVH
        tree = _read_tree(mesh, is_mesh=True)
        self.assertEqual(tree["constructor"], 2)
        ref = _reference_lbvh(*_triangle_bounds(points, tris), None, 4)
        _assert_same_tree(self, tree, ref, "mesh")

        # closest-point queries agree with a CPU (SAH) mesh in distance
        rng = np.random.default_rng(5)
        queries = rng.uniform(-0.8, 0.8, (4096, 3)).astype(np.float32)
        dists = {}
        for device, m in (
            (self.device, mesh),
            (
                "cpu",
                wp.Mesh(
                    points=wp.array(points, dtype=wp.vec3, device="cpu"),
                    indices=wp.array(tris, dtype=wp.int32, device="cpu"),
                ),
            ),
        ):
            q = wp.array(queries, dtype=wp.vec3, device=device)
            face = wp.zeros(len(queries), dtype=int, device=device)
            dist = wp.zeros(len(queries), dtype=float, device=device)
            wp.launch(mesh_closest_kernel, dim=len(queries), inputs=[m.id, q, face, dist, 10.0], device=device)
            self.assertTrue((face.numpy() >= 0).all())
            dists[device] = dist.numpy()
        np.testing.assert_allclose(dists[self.device], dists["cpu"], rtol=0.0, atol=2e-6)

        # winding-number signs (solid-angle refit from the LBVH root)
        signs = {}
        for device in (self.device, "cpu"):
            m = wp.Mesh(
                points=wp.array(points, dtype=wp.vec3, device=device),
                indices=wp.array(tris, dtype=wp.int32, device=device),
                support_winding_number=True,
            )
            q = wp.array(queries, dtype=wp.vec3, device=device)
            sign = wp.zeros(len(queries), dtype=float, device=device)
            wp.launch(mesh_winding_sign_kernel, dim=len(queries), inputs=[m.id, q, sign], device=device)
            signs[device] = sign.numpy()
        inside = np.all(np.abs(queries) < np.array([0.5, 0.3, 0.2], dtype=np.float32), axis=1)
        np.testing.assert_array_equal(signs["cpu"], np.where(inside, -1.0, 1.0))
        np.testing.assert_array_equal(signs[self.device], signs["cpu"])

        # refit after moving the points
        moved = (points * np.float32(1.25) + np.float32(0.1)).astype(np.float32)
        pts.assign(moved)
        mesh.refit()
        _assert_same_tree(
            self, _read_tree(mesh, is_mesh=True), _reference_lbvh_refit(ref, *_triangle_bounds(moved, tris)), "refit"
        )

    def test_lbvh_explicit_sah_and_median_still_build(self):
        _, lowers, uppers, _ = _lbvh_cases()[0]
        for constructor, code in (("sah", 0), ("median", 1)):
            bvh = wp.Bvh(
                wp.array(lowers, dtype=wp.vec3, device=self.device),
                wp.array(uppers, dtype=wp.vec3, device=self.device),
                constructor=constructor,
            )
            self.assertEqual(_read_tree(bvh, is_mesh=False)["constructor"], code)

    def test_fuse_fp_contracts_multiply_add(self):
        """The fuse_fp module option (default on) contracts a * b + c into an FMA, as --fmad=true does on CUDA.

        Inputs are chosen with c close to -a * b, so a single rounding of a * b + c differs from rounding the product
        first for most elements. With fuse_fp off the result is the twice-rounded one.
        """
        rng = np.random.default_rng(0)
        n = 4096
        a = rng.uniform(-2.0, 2.0, n).astype(np.float32)
        b = rng.uniform(-2.0, 2.0, n).astype(np.float32)
        c = (-(a.astype(np.float64) * b) + rng.uniform(-1e-6, 1e-6, n)).astype(np.float32)
        fused = (a.astype(np.float64) * b + c).astype(np.float32)  # exact product, one rounding
        separate = (a * b).astype(np.float32) + c
        self.assertGreater(np.mean(fused != separate), 0.5)
        for fuse_fp, expected in ((True, fused), (False, separate)):

            @wp.kernel(module="unique", module_options={"fuse_fp": fuse_fp})
            def madd(a: wp.array[float], b: wp.array[float], c: wp.array[float], out: wp.array[float]):
                i = wp.tid()
                out[i] = a[i] * b[i] + c[i]

            out = wp.zeros(n, dtype=float, device=self.device)
            wp.launch(madd, n, inputs=[wp.array(a, device=self.device), wp.array(b, device=self.device),
                                       wp.array(c, device=self.device), out], device=self.device)
            np.testing.assert_array_equal(out.numpy(), expected, err_msg=f"fuse_fp={fuse_fp}")

    def test_fast_math_approximates_division_and_sqrt_only(self):
        """The fast_math module option makes float32 division and sqrt approximate, as --use_fast_math does on CUDA.

        CUDA's --use_fast_math is --prec-div=false --prec-sqrt=false --ftz=true --fmad=true: division becomes
        x * rcp(y) (__fdividef, within 2 ulp) and sqrt is approximate, without reassociation. Without the option both
        are correctly rounded. Neither mode may reassociate (x + 2^24) - 2^24 into x.
        """
        rng = np.random.default_rng(0)
        n = 1 << 14
        a = rng.uniform(-2.0, 2.0, n).astype(np.float32)
        x = rng.uniform(1.0e-3, 10.0, n).astype(np.float32)
        div_ref = a.astype(np.float64) / x
        sqrt_ref = np.sqrt(x.astype(np.float64))

        def ulp_error(r, ref):
            rr = ref.astype(np.float32)
            return np.abs(r.astype(np.float64) - ref) / (np.spacing(np.abs(rr)).astype(np.float64))

        for fast in (False, True):

            @wp.kernel(module="unique", module_options={"fast_math": fast})
            def ops(a: wp.array[float], x: wp.array[float], q: wp.array[float], s: wp.array[float],
                    r: wp.array[float]):
                i = wp.tid()
                q[i] = a[i] / x[i]
                s[i] = wp.sqrt(x[i])
                r[i] = (x[i] + 16777216.0) - 16777216.0

            arrays = [wp.zeros(n, dtype=float, device=self.device) for _ in range(3)]
            wp.launch(ops, n, inputs=[wp.array(a, device=self.device), wp.array(x, device=self.device), *arrays],
                      device=self.device)
            q, s, r = (arr.numpy() for arr in arrays)
            reassociated = (x + np.float32(16777216.0)) - np.float32(16777216.0)
            np.testing.assert_array_equal(r, reassociated, err_msg=f"fast_math={fast} reassociated")
            if not fast:
                np.testing.assert_array_equal(q, div_ref.astype(np.float32))
                np.testing.assert_array_equal(s, sqrt_ref.astype(np.float32))
            else:
                self.assertLess(np.mean(q == div_ref.astype(np.float32)), 0.95)
                self.assertLess(np.mean(s == sqrt_ref.astype(np.float32)), 0.95)
                self.assertLessEqual(ulp_error(q, div_ref).max(), 2.5)
                self.assertLessEqual(ulp_error(s, sqrt_ref).max(), 2.5)


class TestMetalInlineBudget(unittest.TestCase):
    """Forced inlining is bounded by the fully inlined size (pure codegen logic, no device needed)."""

    def test_budget_counts_callees(self):
        from warp._src import codegen  # noqa: PLC0415

        budget = codegen._METAL_FORCE_INLINE_MAX_LINES
        names = ("wp_test_inline_small", "wp_test_inline_big", "wp_test_inline_caller")
        try:
            self.assertTrue(codegen._metal_force_inline(names[0], "x = 1;\n" * 10))
            self.assertFalse(codegen._metal_force_inline(names[1], "x = 1;\n" * (budget + 1)))
            # two lines of its own, but inlining its callee would exceed the budget
            self.assertFalse(codegen._metal_force_inline(names[2], f"y = {names[1]}(a);\nreturn y;\n"))
            self.assertTrue(codegen._metal_force_inline(names[2], f"y = {names[0]}(a);\nreturn y;\n"))
        finally:
            for name in names:
                codegen._metal_inlined_lines.pop(name, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
