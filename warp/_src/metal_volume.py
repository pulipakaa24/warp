# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NanoVDB volume construction on the Metal GPU (``Volume.allocate_by_tiles`` / ``allocate_by_voxels``, ``rebuild``).

On Metal the native volume entry points drain the GPU and run the host builder (``warp/native/volume_builder.cpp``)
on unified memory. This module runs the same key-based algorithm as GPU kernels, written in Metal Shading Language and
launched like the utilities of :mod:`warp._src.metal_primitives` (no native rebuild):

1. one 64-bit upper-node key per point (world points are mapped to index space with the arithmetic of
   ``nanovdb::Map::applyInverseMapF(p).round()``: explicit ``fma`` and ``floor(x + 0.5f)``), keys-only LSD radix
   sort, unique;
2. one 64-bit hierarchy key per point (upper-node rank by binary search, upper / lower / leaf offsets), radix sort,
   unique of the keys shifted by 0 (active voxels), 9 (leaves) and 21 (lower nodes), counts on the GPU;
3. every tree level filled with one thread per node: node ``i`` of a level is the ``i``-th unique key, parents are
   found by binary search, child / value masks are set with 32-bit atomic ORs on the half word that holds the bit
   (exact: OR is bitwise), bounding boxes with int32 atomic min / max, per-leaf voxel counts with atomic adds and
   an exclusive scan. These operations commute, so the result does not depend on thread scheduling;
4. the world bounding box in double precision by a software ``fma`` on 64-bit integers (Metal has no ``double``),
   with the expression order of ``nanovdb::Map::applyMap``.

The grid depends only on the set of occupied voxels, so the buffer is byte for byte the host builder's (the
header's map and voxel size are taken from a one-point host build with the same transform, cached).

Exact (non-rebuildable) allocations read the node counts back to size the grid, as on CUDA; with the GPU idle at
that point, the grid and tree headers are written from the host and the volume is registered
(``wp_volume_create_device``) before the population kernels are enqueued, so nothing else waits. ``Volume.rebuild`` runs entirely on the GPU: capacity checks and the
status word are computed by a kernel and the population kernels are skipped on overflow, without a host
synchronization, so rebuilds can be recorded into graph captures. ``get_voxels`` / ``get_tiles`` are GPU kernels
too. Rebuildable *allocation* stays on the host builder (it runs once).
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np

import warp as wp
from warp._src import metal_primitives as _prim

_TG = _prim._TG
_SORT_TILE = _prim._SORT_TILE
_SCAN_TILE = _prim._SCAN_TILE
_SORT_GRID = _prim._SORT_GRID
_MAX_GRID_THREADS = 1 << 20  # grid-stride loops over data-dependent extents

_INVALID_KEY = (1 << 64) - 1

GRID_TYPE_FLOAT = 1
GRID_TYPE_DOUBLE = 2
GRID_TYPE_INT32 = 4
GRID_TYPE_INT64 = 5
GRID_TYPE_VEC3F = 6
GRID_TYPE_VEC3D = 7
GRID_TYPE_UINT32 = 10
GRID_TYPE_VEC4F = 17
GRID_TYPE_VEC4D = 18
GRID_TYPE_INDEX = 19
GRID_TYPE_ONINDEX = 20

# Rows of ``pnanovdb_grid_type_constants`` (warp/native/nanovdb/PNanoVDB.h) for the grid types the builder makes:
# root_off_background, root_size, value_stride_bits, table_stride, root_tile_size, upper_off_table, upper_size,
# lower_off_table, lower_size, leaf_off_table, leaf_size.
_GRID_CONSTANTS = {
    GRID_TYPE_FLOAT: (28, 64, 32, 8, 32, 8256, 270400, 1088, 33856, 96, 2144),
    GRID_TYPE_DOUBLE: (32, 96, 64, 8, 32, 8256, 270400, 1088, 33856, 128, 4224),
    GRID_TYPE_INT32: (28, 64, 32, 8, 32, 8256, 270400, 1088, 33856, 96, 2144),
    GRID_TYPE_INT64: (32, 96, 64, 8, 32, 8256, 270400, 1088, 33856, 128, 4224),
    GRID_TYPE_VEC3F: (28, 96, 96, 16, 32, 8256, 532544, 1088, 66624, 128, 6272),
    GRID_TYPE_VEC3D: (32, 128, 192, 24, 64, 8288, 794720, 1120, 99424, 160, 12448),
    GRID_TYPE_UINT32: (28, 64, 32, 8, 32, 8256, 270400, 1088, 33856, 96, 2144),
    GRID_TYPE_VEC4F: (28, 96, 128, 16, 64, 8288, 532576, 1120, 66656, 128, 8320),
    GRID_TYPE_VEC4D: (32, 160, 256, 32, 64, 8320, 1056896, 1152, 132224, 160, 16544),
    GRID_TYPE_INDEX: (32, 96, 0, 8, 32, 8256, 270400, 1088, 33856, 80, 96),
    GRID_TYPE_ONINDEX: (32, 96, 0, 8, 32, 8256, 270400, 1088, 33856, 80, 96),
}

# value type names of Volume._supported_allocation_types -> (grid type, value size in bytes)
_VALUE_TYPES = {
    "int32": (GRID_TYPE_INT32, 4),
    "uint32": (GRID_TYPE_UINT32, 4),
    "int64": (GRID_TYPE_INT64, 8),
    "float": (GRID_TYPE_FLOAT, 4),
    "double": (GRID_TYPE_DOUBLE, 8),
    "Vec3f": (GRID_TYPE_VEC3F, 12),
    "Vec3d": (GRID_TYPE_VEC3D, 24),
    "Vec4f": (GRID_TYPE_VEC4F, 16),
    "Vec4d": (GRID_TYPE_VEC4D, 32),
}

# grid type names as printed by nanovdb::toStr (Volume.get_grid_info().type_str) -> grid type, for get_voxels
_TYPE_STR = {
    "float": GRID_TYPE_FLOAT,
    "double": GRID_TYPE_DOUBLE,
    "int32": GRID_TYPE_INT32,
    "int64": GRID_TYPE_INT64,
    "Vec3f": GRID_TYPE_VEC3F,
    "Vec3d": GRID_TYPE_VEC3D,
    "uint32": GRID_TYPE_UINT32,
    "Vec4f": GRID_TYPE_VEC4F,
    "Vec4d": GRID_TYPE_VEC4D,
    "Index": GRID_TYPE_INDEX,
    "OnIndex": GRID_TYPE_ONINDEX,
}

_GRID_HEADER_SIZE = 672  # sizeof(nanovdb::GridData)
_TREE_SIZE = 64  # sizeof(nanovdb::TreeData)
_ROOT_OFFSET = _GRID_HEADER_SIZE + _TREE_SIZE

_VB_SOURCE = r"""
// ===============================================================================================================
// NanoVDB key-based builder (warp/native/volume_builder.cpp on the GPU)

#pragma clang fp contract(off)

#define VB_INVALID 0xfffffffffffffffful
#define VB_GT_INDEX 19u
#define VB_GT_ONINDEX 20u
#define VB_GRID_SIZE 672ul
#define VB_TREE 672ul
#define VB_ROOT 736ul
#define VB_UPPER_CHILD_MASK 4128ul
#define VB_LOWER_CHILD_MASK 544ul
#define VB_LEAF_VALUE_MASK 16ul
#define VB_INT_MAX 0x7fffffff
#define VB_INT_MIN (-0x7fffffff - 1)

struct vb_args_t {
    device const uchar* points;   // world-space vec3f or index-space vec3i points
    device const int* mask;       // optional point mask
    device ulong* keys;           // emitted / sorted keys
    device ulong* upper_keys;     // unique keys of each level
    device ulong* leaf_keys;
    device ulong* lower_keys;
    device ulong* voxel_keys;
    device uint* counts;          // [0] leaves [1] lower [2] upper [3] voxels [4] populate flag [5] status
    device uint* status;          // caller's status word (optional)
    device uint* grid;            // grid buffer
    device const uint* header;    // grid header template (map, voxel size, class, type)
    device int* leaf_counts;      // active voxels per leaf, then their exclusive scan
    device int* tile_sums;        // unique: [num_out][num_tiles]
    device int* out_coords;       // get_voxels / get_tiles output (vec3i)
    ulong grid_size;
    ulong off_upper;
    ulong off_lower;
    ulong off_leaf;
    ulong cap_voxel;
    ulong desc_voxel_count;
    uint n;
    uint num_tiles;
    uint world;
    uint grid_type;
    uint active_voxels;
    uint check_capacity;
    uint cap_leaf;
    uint cap_lower;
    uint cap_upper;
    uint value_size;
    uint num_out;
    uint desc_leaf_count;
    uint out_shift[3];
    uint out_slot[3];
    uint bg[8];
    uint root_off_background;
    uint root_size;
    uint value_stride_bits;
    uint table_stride;
    uint root_tile_size;
    uint upper_off_table;
    uint upper_size;
    uint lower_off_table;
    uint lower_size;
    uint leaf_off_table;
    uint leaf_size;
    uint pad;
};

inline uint vb_rd32(device const uint* g, ulong off) { return g[off >> 2]; }
inline void vb_wr32(device uint* g, ulong off, uint v) { g[off >> 2] = v; }
inline ulong vb_rd64(device const uint* g, ulong off) { return ulong(g[off >> 2]) | (ulong(g[(off >> 2) + 1]) << 32); }
inline void vb_wr64(device uint* g, ulong off, ulong v)
{
    g[off >> 2] = uint(v);
    g[(off >> 2) + 1] = uint(v >> 32);
}
inline int3 vb_rd_coord(device const uint* g, ulong off)
{
    return int3(int(g[off >> 2]), int(g[(off >> 2) + 1]), int(g[(off >> 2) + 2]));
}
inline void vb_wr_coord(device uint* g, ulong off, int3 c)
{
    g[off >> 2] = uint(c.x);
    g[(off >> 2) + 1] = uint(c.y);
    g[(off >> 2) + 2] = uint(c.z);
}
inline void vb_wr_bytes(device uint* g, ulong off, constant uint* words, uint byte_count)
{
    for (uint i = 0; i < byte_count / 4u; ++i)
        g[(off >> 2) + i] = words[i];
}

// pnanovdb_root_tile / upper / lower handles
inline ulong vb_upper(constant vb_args_t& a, uint i) { return a.off_upper + ulong(i) * a.upper_size; }
inline ulong vb_lower(constant vb_args_t& a, uint i) { return a.off_lower + ulong(i) * a.lower_size; }
inline ulong vb_leaf(constant vb_args_t& a, uint i) { return a.off_leaf + ulong(i) * a.leaf_size; }
inline bool vb_regular(uint gt) { return gt != VB_GT_INDEX && gt != VB_GT_ONINDEX; }

// rebuild_set_mask_on with a 32-bit atomic OR on the half word holding the bit
inline void vb_set_mask_on(device uint* g, ulong node, ulong mask_offset, uint bit)
{
    const ulong off = node + mask_offset + 8ul * (bit >> 6) + 4ul * ((bit >> 5) & 1u);
    atomic_fetch_or_explicit((device atomic_uint*)(g + (off >> 2)), 1u << (bit & 31u), memory_order_relaxed);
}

inline void vb_expand_bbox(device uint* g, ulong node, int3 mn, int3 mx)
{
    device atomic_int* b = (device atomic_int*)(g + (node >> 2));
    atomic_fetch_min_explicit(b + 0, mn.x, memory_order_relaxed);
    atomic_fetch_min_explicit(b + 1, mn.y, memory_order_relaxed);
    atomic_fetch_min_explicit(b + 2, mn.z, memory_order_relaxed);
    atomic_fetch_max_explicit(b + 3, mx.x, memory_order_relaxed);
    atomic_fetch_max_explicit(b + 4, mx.y, memory_order_relaxed);
    atomic_fetch_max_explicit(b + 5, mx.z, memory_order_relaxed);
}

inline void vb_set_invalid_bbox(device uint* g, ulong node)
{
    vb_wr_coord(g, node, int3(VB_INT_MAX));
    vb_wr_coord(g, node + 12ul, int3(VB_INT_MIN));
}

// first index in keys[0, count) with keys[i] >= key
inline uint vb_lower_bound(device const ulong* keys, uint count, ulong key)
{
    uint lo = 0, hi = count;
    while (lo < hi) {
        uint mid = lo + ((hi - lo) >> 1);
        if (keys[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// rebuild_point_to_coord: nanovdb::Map::applyInverseMapF(p).round() for world points
inline int3 vb_point_coord(constant vb_args_t& a, uint i)
{
    if (a.world) {
        device const float* p = (device const float*)a.points + 3ul * i;
        device const float* h = (device const float*)a.header;
        device const float* m = h + (296 + 36) / 4;  // map.mInvMatF
        device const float* t = h + (296 + 72) / 4;  // map.mVecF
        const float d0 = p[0] - t[0], d1 = p[1] - t[1], d2 = p[2] - t[2];
        const float x = fma(d0, m[0], fma(d1, m[1], d2 * m[2]));
        const float y = fma(d0, m[3], fma(d1, m[4], d2 * m[5]));
        const float z = fma(d0, m[6], fma(d1, m[7], d2 * m[8]));
        return int3(int(floor(x + 0.5f)), int(floor(y + 0.5f)), int(floor(z + 0.5f)));
    }
    device const int* c = (device const int*)a.points + 3ul * i;
    return int3(c[0], c[1], c[2]);
}

inline bool vb_masked(constant vb_args_t& a, uint i) { return a.mask && a.mask[i] == 0; }

// rebuild_upper_key_from_coord
inline ulong vb_upper_key(int3 c)
{
    const long off = 1l << 31;
    return ulong(uint(long(c.z) + off) >> 12) | (ulong(uint(long(c.y) + off) >> 12) << 21)
        | (ulong(uint(long(c.x) + off) >> 12) << 42);
}

// rebuild_upper_key_to_coord
inline int3 vb_upper_key_to_coord(ulong key)
{
    const long off = 1l << 31;
    const ulong mask = (1ul << 21) - 1ul;
    return int3(int(long(((key >> 42) & mask) << 12) - off), int(long(((key >> 21) & mask) << 12) - off),
        int(long((key & mask) << 12) - off));
}

// pnanovdb_coord_to_key
inline ulong vb_root_key(int3 c)
{
    return ulong(uint(c.z) >> 12) | (ulong(uint(c.y) >> 12) << 21) | (ulong(uint(c.x) >> 12) << 42);
}

// rebuild_hierarchy_key
inline ulong vb_hierarchy_key(uint upper_id, int3 c)
{
    const uint uo = (uint((c.x & 4095) >> 7) << 10) + (uint((c.y & 4095) >> 7) << 5) + uint((c.z & 4095) >> 7);
    const uint lo = (uint((c.x & 127) >> 3) << 8) + (uint((c.y & 127) >> 3) << 4) + uint((c.z & 127) >> 3);
    const uint lf = (uint(c.x & 7) << 6) + (uint(c.y & 7) << 3) + uint(c.z & 7);
    return (ulong(upper_id) << 36) | (ulong(uo) << 21) | (ulong(lo) << 9) | ulong(lf);
}

kernel void vb_emit_upper_keys(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= a.n)
        return;
    a.keys[i] = vb_masked(a, i) ? VB_INVALID : vb_upper_key(vb_point_coord(a, i));
}

kernel void vb_emit_hierarchy_keys(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= a.n)
        return;
    ulong key = VB_INVALID;
    if (!vb_masked(a, i)) {
        const int3 c = vb_point_coord(a, i);
        const ulong uk = vb_upper_key(c);
        const uint count = a.counts[2];
        const uint id = vb_lower_bound(a.upper_keys, count, uk);
        if (id < count && a.upper_keys[id] == uk)
            key = vb_hierarchy_key(id, c);
    }
    a.keys[i] = key;
}

// ---------------------------------------------------------------------------------------------------------------
// keys-only stable scatter of one radix pass (sort_scatter without values)

kernel void vb_sort_scatter_keys(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant sort_args_t& a [[buffer(1)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    device const uint* plan = a.plan + 4 * a.pass;
    if (!plan[0])
        return;
    threadgroup uint hist[NSG * 256];
    threadgroup uint offset[256];
    threadgroup uint sm[NSG + 1];

    uint total;
    const uint digit_start = tg_exclusive_scan(a.digit_total[lid], sm, lane, sg, total);
    device const ulong* keys = (device const ulong*)a.keys[plan[1]];
    device ulong* keys_out = (device ulong*)a.keys[plan[2]];
    const bool second = plan[3] != 3;
    device ulong* keys_out2 = (device ulong*)a.keys[second ? plan[3] : plan[2]];
    const ulong mask = (1ul << a.bits) - 1ul;

    for (uint tile = tg; tile < a.num_tiles; tile += ntg) {
    for (uint i = lid; i < NSG * 256; i += TG)
        hist[i] = 0;
    offset[lid] = digit_start + a.tile_hist[lid * a.num_tiles + tile];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = tile * SORT_TILE + sg * 32u * SORT_KPT + lane;
    threadgroup uint* row = hist + sg * 256;
    ulong key[SORT_KPT];
    ushort rank[SORT_KPT];
    uchar dig[SORT_KPT];
    for (uint j = 0; j < SORT_KPT; ++j) {
        uint idx = base + j * 32u;
        bool valid = idx < a.n;
        key[j] = valid ? keys[idx] : 0ul;
        uint d = valid ? uint((key[j] >> a.shift) & mask) : 0u;
        uint m = match_digit(d, a.bits, valid);
        uint lt = popcount(m & ((1u << lane) - 1u));
        uint old = row[d];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (valid && lt == 0)
            row[d] = old + popcount(m);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        rank[j] = ushort(old + lt);
        dig[j] = uchar(d);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint running = offset[lid];
        for (uint s = 0; s < NSG; ++s) {
            uint c = hist[s * 256 + lid];
            hist[s * 256 + lid] = running;
            running += c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < SORT_KPT; ++j) {
        uint idx = base + j * 32u;
        if (idx < a.n) {
            uint pos = row[dig[j]] + rank[j];
            keys_out[pos] = key[j];
            if (second)
                keys_out2[pos] = key[j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// unique of sorted keys (shifted right by out_shift[o]; INVALID keys, sorted last, are dropped)

inline device ulong* vb_slot(constant vb_args_t& a, uint slot)
{
    return slot == 0 ? a.leaf_keys : slot == 1 ? a.lower_keys : slot == 2 ? a.upper_keys : a.voxel_keys;
}

inline int vb_head(constant vb_args_t& a, uint i, uint s)
{
    if (i >= a.n)
        return 0;
    const ulong k = a.keys[i];
    if (k == VB_INVALID)
        return 0;
    return (i == 0 || (k >> s) != (a.keys[i - 1] >> s)) ? 1 : 0;
}

kernel void vb_unique_count(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup int sm[NSG + 1];
    for (uint o = 0; o < a.num_out; ++o) {
        int c = 0;
        for (uint j = 0; j < SCAN_IPT; ++j)
            c += vb_head(a, tile * SCAN_TILE + j * TG + lid, a.out_shift[o]);
        int total;
        tg_exclusive_scan(c, sm, lane, sg, total);
        if (lid == 0)
            a.tile_sums[o * a.num_tiles + tile] = total;
    }
}

kernel void vb_unique_scatter(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup int sm[NSG + 1];
    const uint first = tile * SCAN_TILE + lid * SCAN_IPT;
    for (uint o = 0; o < a.num_out; ++o) {
        const uint s = a.out_shift[o];
        device ulong* out = vb_slot(a, a.out_slot[o]);
        int h[SCAN_IPT];
        int c = 0;
        for (uint j = 0; j < SCAN_IPT; ++j) {
            h[j] = vb_head(a, first + j, s);
            c += h[j];
        }
        int total;
        int run = tg_exclusive_scan(c, sm, lane, sg, total) + a.tile_sums[o * a.num_tiles + tile];
        for (uint j = 0; j < SCAN_IPT; ++j) {
            const uint i = first + j;
            if (h[j]) {
                out[run] = a.keys[i] >> s;
                ++run;
            }
            if (i == a.n - 1)
                a.counts[a.out_slot[o]] = uint(run);
        }
    }
}

// counts, capacity status and the populate flag (rebuild_capacity_status)
kernel void vb_finalize_counts(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i != 0)
        return;
    device uint* c = a.counts;
    if (!a.active_voxels)
        c[3] = c[0] * 512u;
    uint status = 0;
    if (a.check_capacity) {
        if (c[0] > a.cap_leaf)
            status |= 1u;
        if (c[1] > a.cap_lower)
            status |= 2u;
        if (c[2] > a.cap_upper)
            status |= 4u;
        if (ulong(c[3]) > a.cap_voxel)
            status |= 8u;
    }
    c[4] = status == 0 ? 1u : 0u;
    c[5] = status;
    if (a.status)
        *a.status = status;
}

// ---------------------------------------------------------------------------------------------------------------
// populate (rebuild_populate_grid); every kernel is skipped when the counts did not fit the capacities

kernel void vb_zero(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]],
    uint grid [[threads_per_grid]])
{
    if (!a.counts[4])
        return;
    device uint4* g = (device uint4*)a.grid;
    const ulong n = a.grid_size >> 4;
    for (ulong j = i; j < n; j += grid)
        g[j] = uint4(0u);
    for (uint j = i; j < a.cap_leaf; j += grid)
        a.leaf_counts[j] = 0;
}

// rebuild_init_grid_tree_root
kernel void vb_init_grid(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i != 0 || !a.counts[4])
        return;
    device uint* g = a.grid;
    device const uint* c = a.counts;
    for (uint w = 0; w < VB_GRID_SIZE / 4; ++w)
        g[w] = a.header[w];
    vb_wr64(g, 32, a.grid_size);   // grid_size
    vb_wr64(g, 640, a.grid_size);  // blind_metadata_offset
    vb_wr64(g, 656, vb_regular(a.grid_type) ? 0ul : ulong(c[3]) + 1ul);

    vb_wr64(g, VB_TREE + 0, a.off_leaf - VB_TREE);
    vb_wr64(g, VB_TREE + 8, a.off_lower - VB_TREE);
    vb_wr64(g, VB_TREE + 16, a.off_upper - VB_TREE);
    vb_wr64(g, VB_TREE + 24, VB_ROOT - VB_TREE);
    vb_wr32(g, VB_TREE + 32, c[0]);
    vb_wr32(g, VB_TREE + 36, c[1]);
    vb_wr32(g, VB_TREE + 40, c[2]);
    vb_wr32(g, VB_TREE + 44, c[0]);
    vb_wr32(g, VB_TREE + 48, c[1]);
    vb_wr32(g, VB_TREE + 52, c[2]);
    vb_wr64(g, VB_TREE + 56, ulong(c[3]));

    vb_set_invalid_bbox(g, VB_ROOT);
    vb_wr32(g, VB_ROOT + 24, c[2]);
    if (vb_regular(a.grid_type))
        vb_wr_bytes(g, VB_ROOT + a.root_off_background, a.bg, a.value_size);
}

// rebuild_build_upper_nodes
kernel void vb_build_upper(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (!a.counts[4] || i >= a.counts[2])
        return;
    device uint* g = a.grid;
    const ulong up = vb_upper(a, i);
    const int3 o = vb_upper_key_to_coord(a.upper_keys[i]);
    const ulong tile = VB_ROOT + a.root_size + ulong(i) * a.root_tile_size;
    vb_wr64(g, tile + 0, vb_root_key(o));
    vb_wr32(g, tile + 16, 0u);
    vb_wr64(g, tile + 8, ulong(long(up) - long(VB_ROOT)));
    vb_wr_coord(g, up, o);
    vb_wr_coord(g, up + 12, int3(VB_INT_MIN));
    vb_wr64(g, up + 24, 0ul);
}

// rebuild_set_{upper,lower}_background_values, rebuild_set_leaf_values (regular value grids)
inline void vb_fill_background(constant vb_args_t& a, uint count, uint per_node, uint i, uint grid, int level)
{
    if (!a.counts[4])
        return;
    const ulong total = ulong(count) * per_node;
    for (ulong j = i; j < total; j += grid) {
        const uint node = uint(j / per_node), n = uint(j % per_node);
        ulong off;
        if (level == 2)
            off = vb_upper(a, node) + a.upper_off_table + ulong(a.table_stride) * n;
        else if (level == 1)
            off = vb_lower(a, node) + a.lower_off_table + ulong(a.table_stride) * n;
        else
            off = vb_leaf(a, node) + a.leaf_off_table + ((ulong(a.value_stride_bits) * n) >> 3);
        vb_wr_bytes(a.grid, off, a.bg, a.value_size);
    }
}

kernel void vb_upper_background(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]],
    uint grid [[threads_per_grid]])
{
    vb_fill_background(a, a.counts[2], 32768u, i, grid, 2);
}

kernel void vb_lower_background(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]],
    uint grid [[threads_per_grid]])
{
    vb_fill_background(a, a.counts[1], 4096u, i, grid, 1);
}

kernel void vb_leaf_values(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]],
    uint grid [[threads_per_grid]])
{
    vb_fill_background(a, a.counts[0], 512u, i, grid, 0);
}

// rebuild_build_lower_nodes
kernel void vb_build_lower(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (!a.counts[4] || i >= a.counts[1])
        return;
    device uint* g = a.grid;
    const ulong key = a.lower_keys[i];
    const uint upper_id = uint(key >> 15);
    if (upper_id >= a.counts[2])
        return;
    const ulong up = vb_upper(a, upper_id);
    const ulong lo = vb_lower(a, i);
    const uint uo = uint(key & 32767ul);
    vb_set_mask_on(g, up, VB_UPPER_CHILD_MASK, uo);
    vb_wr64(g, up + a.upper_off_table + ulong(a.table_stride) * uo, ulong(long(lo) - long(up)));
    const int3 origin = vb_rd_coord(g, up)
        + int3(int((uo >> 10) & 31u) << 7, int((uo >> 5) & 31u) << 7, int(uo & 31u) << 7);
    vb_wr_coord(g, lo, origin);
    vb_wr_coord(g, lo + 12, int3(VB_INT_MIN));
    vb_wr64(g, lo + 24, 0ul);
}

// rebuild_build_leaf_nodes
kernel void vb_build_leaf(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (!a.counts[4] || i >= a.counts[0])
        return;
    device uint* g = a.grid;
    const ulong key = a.leaf_keys[i];
    const ulong lower_key = key >> 12;
    const uint lower_count = a.counts[1];
    const uint lower_id = vb_lower_bound(a.lower_keys, lower_count, lower_key);
    if (lower_id >= lower_count || a.lower_keys[lower_id] != lower_key)
        return;
    const ulong lo = vb_lower(a, lower_id);
    const ulong lf = vb_leaf(a, i);
    const uint off = uint(key & 4095ul);
    vb_set_mask_on(g, lo, VB_LOWER_CHILD_MASK, off);
    vb_wr64(g, lo + a.lower_off_table + ulong(a.table_stride) * off, ulong(long(lf) - long(lo)));
    const int3 origin = vb_rd_coord(g, lo)
        + int3(int((off >> 8) & 15u) << 3, int((off >> 4) & 15u) << 3, int(off & 15u) << 3);
    vb_wr_coord(g, lf, origin);
    vb_wr32(g, lf + 12, 2u << 24);
    const ulong table = lf + a.leaf_off_table;
    if (a.grid_type == VB_GT_INDEX) {
        vb_wr64(g, table, ulong(i) * 512ul + 1ul);
        vb_wr64(g, table + 8, 0ul);
        for (uint w = 0; w < 8; ++w)
            vb_wr64(g, lf + VB_LEAF_VALUE_MASK + 8ul * w, VB_INVALID);
        vb_wr32(g, lf + 12, 0x00070707u | (2u << 24));
    } else if (a.grid_type == VB_GT_ONINDEX) {
        vb_wr64(g, table, 1ul);
        vb_wr64(g, table + 8, 0ul);
    } else if (!a.active_voxels) {
        for (uint w = 0; w < 8; ++w)
            vb_wr64(g, lf + VB_LEAF_VALUE_MASK + 8ul * w, VB_INVALID);
        vb_wr32(g, lf + 12, 0x00070707u | (2u << 24));
    }
}

// rebuild_set_active_voxels
kernel void vb_set_active_voxels(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]],
    uint grid [[threads_per_grid]])
{
    if (!a.counts[4])
        return;
    const uint voxel_count = a.counts[3];
    const uint leaf_count = a.counts[0];
    for (uint j = i; j < voxel_count; j += grid) {
        const ulong key = a.voxel_keys[j];
        const ulong leaf_key = key >> 9;
        const uint leaf_id = vb_lower_bound(a.leaf_keys, leaf_count, leaf_key);
        if (leaf_id >= leaf_count || a.leaf_keys[leaf_id] != leaf_key)
            continue;
        vb_set_mask_on(a.grid, vb_leaf(a, leaf_id), VB_LEAF_VALUE_MASK, uint(key & 511ul));
        atomic_fetch_add_explicit((device atomic_int*)(a.leaf_counts + leaf_id), 1, memory_order_relaxed);
    }
}

inline uint vb_find_lowest_on(ulong w) { return w == 0 ? 64u : uint(ctz(w)); }
inline uint vb_find_highest_on(ulong w) { return w == 0 ? 64u : 63u - uint(clz(w)); }

// rebuild_update_leaf_bbox
inline void vb_update_leaf_bbox(device uint* g, ulong leaf)
{
    ulong union_word = 0;
    uint x_min = 8u, x_max = 8u;
    for (uint i = 0; i < 8u; ++i) {
        const ulong word = vb_rd64(g, leaf + VB_LEAF_VALUE_MASK + 8ul * i);
        if (word) {
            union_word |= word;
            if (x_min == 8u)
                x_min = i;
            x_max = i;
        }
    }
    uint flags = vb_rd32(g, leaf + 12) >> 24;
    if (!union_word) {
        vb_wr32(g, leaf + 12, (flags & ~2u) << 24);
        return;
    }
    int3 mn = vb_rd_coord(g, leaf);
    mn.x = (mn.x & ~7) + int(x_min);
    mn.y = (mn.y & ~7) + int(vb_find_lowest_on(union_word) >> 3);
    const uint word32 = uint(union_word) | uint(union_word >> 32);
    const uint word16 = (word32 & 0xffffu) | (word32 >> 16);
    const uint byte = (word16 & 0xffu) | (word16 >> 8);
    mn.z = (mn.z & ~7) + int(vb_find_lowest_on(ulong(byte)));
    vb_wr_coord(g, leaf, mn);
    const uint x_dif = x_max - x_min;
    const uint y_dif = (vb_find_highest_on(union_word) >> 3) - (vb_find_lowest_on(union_word) >> 3);
    const uint z_dif = vb_find_highest_on(ulong(byte)) - vb_find_lowest_on(ulong(byte));
    flags |= 2u;
    vb_wr32(g, leaf + 12, x_dif | (y_dif << 8) | (z_dif << 16) | (flags << 24));
}

// rebuild_finalize_onindex_leaves (leaf_counts holds the exclusive scan of the per-leaf voxel counts)
kernel void vb_finalize_onindex_leaves(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (!a.counts[4] || i >= a.counts[0])
        return;
    device uint* g = a.grid;
    const ulong lf = vb_leaf(a, i);
    const ulong table = lf + a.leaf_off_table;
    vb_wr64(g, table, 1ul + ulong(uint(a.leaf_counts[i])));
    ulong prefix = ulong(popcount(vb_rd64(g, lf + VB_LEAF_VALUE_MASK)));
    ulong sum = prefix;
    for (uint n = 9; n < 55; n += 9) {
        sum += ulong(popcount(vb_rd64(g, lf + VB_LEAF_VALUE_MASK + 8ul * (n / 9))));
        prefix |= sum << n;
    }
    vb_wr64(g, table + 8, prefix);
    vb_update_leaf_bbox(g, lf);
}

// rebuild_reset_bboxes
kernel void vb_reset_bboxes(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (!a.counts[4])
        return;
    if (i == 0)
        vb_set_invalid_bbox(a.grid, VB_ROOT);
    if (i < a.counts[1])
        vb_set_invalid_bbox(a.grid, vb_lower(a, i));
    if (i < a.counts[2])
        vb_set_invalid_bbox(a.grid, vb_upper(a, i));
}

// rebuild_propagate_leaf_bboxes
kernel void vb_propagate_leaf_bboxes(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (!a.counts[4] || i >= a.counts[0])
        return;
    device uint* g = a.grid;
    const ulong lf = vb_leaf(a, i);
    if (a.grid_type != VB_GT_ONINDEX)
        vb_update_leaf_bbox(g, lf);
    const ulong lower_key = a.leaf_keys[i] >> 12;
    const uint lower_count = a.counts[1];
    const uint lower_id = vb_lower_bound(a.lower_keys, lower_count, lower_key);
    if (lower_id < lower_count && a.lower_keys[lower_id] == lower_key) {
        const int3 mn = vb_rd_coord(g, lf);
        const uint dif = vb_rd32(g, lf + 12);
        const int3 mx = mn + int3(int(dif & 0xffu), int((dif >> 8) & 0xffu), int((dif >> 16) & 0xffu));
        vb_expand_bbox(g, vb_lower(a, lower_id), mn, mx);
    }
}

// rebuild_propagate_lower_bboxes
kernel void vb_propagate_lower_bboxes(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (!a.counts[4] || i >= a.counts[1])
        return;
    const ulong upper_key = a.lower_keys[i] >> 15;
    if (upper_key >= ulong(a.counts[2]))
        return;
    const ulong lo = vb_lower(a, i);
    vb_expand_bbox(a.grid, vb_upper(a, uint(upper_key)), vb_rd_coord(a.grid, lo), vb_rd_coord(a.grid, lo + 12));
}

// rebuild_propagate_upper_bboxes
kernel void vb_propagate_upper_bboxes(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (!a.counts[4] || i >= a.counts[2])
        return;
    const ulong up = vb_upper(a, i);
    vb_expand_bbox(a.grid, VB_ROOT, vb_rd_coord(a.grid, up), vb_rd_coord(a.grid, up + 12));
}

// ---------------------------------------------------------------------------------------------------------------
// IEEE binary64 on 64-bit integers (Metal has no double): fused multiply-add rounded to nearest even, comparison

#define F64_FRAC 0x000ffffffffffffful
#define F64_HIDDEN 0x0010000000000000ul
#define F64_INF 0x7ff0000000000000ul
#define F64_NAN 0x7ff8000000000000ul

struct vb_u128 {
    ulong hi, lo;
};

inline vb_u128 u128_shl(vb_u128 x, uint s)
{
    if (s == 0)
        return x;
    if (s >= 128)
        return { 0ul, 0ul };
    if (s >= 64)
        return { x.lo << (s - 64), 0ul };
    return { (x.hi << s) | (x.lo >> (64 - s)), x.lo << s };
}

// right shift; the bits shifted out are ORed into bit 0 (sticky)
inline vb_u128 u128_shr_sticky(vb_u128 x, uint s)
{
    if (s == 0)
        return x;
    if (s >= 128)
        return { 0ul, (x.hi | x.lo) ? 1ul : 0ul };
    vb_u128 r;
    ulong lost;
    if (s >= 64) {
        lost = x.lo | (s > 64 ? (x.hi << (128 - s)) : 0ul);
        r = { 0ul, x.hi >> (s - 64) };
    } else {
        lost = x.lo << (64 - s);
        r = { x.hi >> s, (x.lo >> s) | (x.hi << (64 - s)) };
    }
    if (lost)
        r.lo |= 1ul;
    return r;
}

inline vb_u128 u128_add(vb_u128 a, vb_u128 b)
{
    const ulong lo = a.lo + b.lo;
    return { a.hi + b.hi + (lo < a.lo ? 1ul : 0ul), lo };
}

inline vb_u128 u128_sub(vb_u128 a, vb_u128 b)
{
    return { a.hi - b.hi - (a.lo < b.lo ? 1ul : 0ul), a.lo - b.lo };
}

inline bool u128_lt(vb_u128 a, vb_u128 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }

inline int u128_msb(vb_u128 a)
{
    if (a.hi)
        return 127 - int(clz(a.hi));
    if (a.lo)
        return 63 - int(clz(a.lo));
    return -1;
}

inline bool u128_bit(vb_u128 a, int b) { return b >= 64 ? ((a.hi >> (b - 64)) & 1ul) != 0 : ((a.lo >> b) & 1ul) != 0; }

// any bit below position b
inline bool u128_any_below(vb_u128 a, int b)
{
    if (b <= 0)
        return false;
    if (b >= 128)
        return (a.hi | a.lo) != 0;
    if (b >= 64)
        return a.lo != 0 || (b > 64 && (a.hi & ((1ul << (b - 64)) - 1ul)) != 0);
    return (a.lo & ((1ul << b) - 1ul)) != 0;
}

// fma(a, b, c) on binary64 bit patterns, round to nearest even
inline ulong f64_fma(ulong a, ulong b, ulong c)
{
    const uint ea = uint(a >> 52) & 0x7ffu, eb = uint(b >> 52) & 0x7ffu, ec = uint(c >> 52) & 0x7ffu;
    const ulong sa = a >> 63, sb = b >> 63, sc = c >> 63;
    const ulong sp = sa ^ sb;
    ulong ma = a & F64_FRAC, mb = b & F64_FRAC, mc = c & F64_FRAC;
    if (ea == 0x7ffu || eb == 0x7ffu || ec == 0x7ffu) {
        if ((ea == 0x7ffu && ma) || (eb == 0x7ffu && mb) || (ec == 0x7ffu && mc))
            return F64_NAN;
        const bool a_zero = ea == 0 && ma == 0, b_zero = eb == 0 && mb == 0;
        if (ea == 0x7ffu || eb == 0x7ffu) {
            if (a_zero || b_zero)
                return F64_NAN;
            if (ec == 0x7ffu && sc != sp)
                return F64_NAN;
            return (sp << 63) | F64_INF;
        }
        return c;
    }
    int Ea = ea ? int(ea) - 1075 : -1074;
    int Eb = eb ? int(eb) - 1075 : -1074;
    int Ec = ec ? int(ec) - 1075 : -1074;
    if (ea)
        ma |= F64_HIDDEN;
    if (eb)
        mb |= F64_HIDDEN;
    if (ec)
        mc |= F64_HIDDEN;

    vb_u128 P = { mulhi(ma, mb), ma * mb };
    const bool p_zero = (P.hi | P.lo) == 0;
    if (p_zero && mc == 0)
        return (sp & sc) << 63;
    if (p_zero)
        return c;

    int Ep = Ea + Eb;
    const int mp = u128_msb(P);
    P = u128_shl(P, uint(125 - mp));
    Ep -= 125 - mp;

    vb_u128 X = P, Y = { 0ul, 0ul };
    int Ex = Ep, Ey = Ep;
    ulong sx = sp, sy = sp;
    if (mc) {
        const int mcb = 63 - int(clz(mc));
        vb_u128 C = u128_shl(vb_u128 { 0ul, mc }, uint(125 - mcb));
        Ec -= 125 - mcb;
        if (Ep >= Ec) {
            Y = C;
            Ey = Ec;
            sy = sc;
        } else {
            X = C;
            Ex = Ec;
            sx = sc;
            Y = P;
            Ey = Ep;
            sy = sp;
        }
    }
    Y = u128_shr_sticky(Y, uint(min(Ex - Ey, 200)));

    vb_u128 S;
    ulong s;
    if (sx == sy) {
        S = u128_add(X, Y);
        s = sx;
    } else if (u128_lt(X, Y)) {
        S = u128_sub(Y, X);
        s = sy;
    } else {
        S = u128_sub(X, Y);
        s = sx;
    }
    if ((S.hi | S.lo) == 0)
        return 0ul;

    const int E = Ex;  // value = S * 2^E
    const int msb = u128_msb(S);  // at most 126
    int lsb = max(msb - 52, -1074 - E);
    if (lsb > msb + 1)
        return s << 63;  // below half the smallest subnormal
    ulong mant;
    if (lsb <= 0) {
        mant = u128_shl(S, uint(-lsb)).lo;
    } else {
        mant = lsb >= 64 ? S.hi >> (lsb - 64) : ((S.lo >> lsb) | (S.hi << (64 - lsb)));
        const bool round_bit = u128_bit(S, lsb - 1);
        const bool rest = u128_any_below(S, lsb - 1);
        if (round_bit && (rest || (mant & 1ul)))
            ++mant;
        if (mant == (1ul << 53)) {
            mant >>= 1;
            ++lsb;
        }
    }
    const int e_lsb = lsb + E;
    if (mant >= F64_HIDDEN) {
        const int ef = e_lsb + 1075;
        if (ef >= 2047)
            return (s << 63) | F64_INF;
        return (s << 63) | (ulong(ef) << 52) | (mant & F64_FRAC);
    }
    return (s << 63) | mant;
}

// exact for |v| < 2^53
inline ulong f64_from_long(long v)
{
    if (v == 0)
        return 0ul;
    const ulong s = v < 0 ? 1ul : 0ul;
    const ulong m = s ? ulong(-v) : ulong(v);
    const int msb = 63 - int(clz(m));
    return (s << 63) | (ulong(msb + 1023) << 52) | ((m << (52 - msb)) & F64_FRAC);
}

// a < b for non-NaN binary64 bit patterns (-0 == +0)
inline bool f64_lt(ulong a, ulong b)
{
    if (((a | b) << 1) == 0)
        return false;
    const ulong ka = (a >> 63) ? ~a : (a | (1ul << 63));
    const ulong kb = (b >> 63) ? ~b : (b | (1ul << 63));
    return ka < kb;
}

struct vb_vec3d {
    ulong x, y, z;
};

// nanovdb::Map::applyMap: matMult(mMatD, mVecD, xyz)
inline vb_vec3d vb_apply_map(device const uint* h, vb_vec3d v)
{
    ulong m[9], t[3];
    for (uint k = 0; k < 9; ++k)
        m[k] = vb_rd64(h, 296 + 88 + 8 * k);
    for (uint k = 0; k < 3; ++k)
        t[k] = vb_rd64(h, 296 + 232 + 8 * k);
    vb_vec3d r;
    r.x = f64_fma(v.x, m[0], f64_fma(v.y, m[1], f64_fma(v.z, m[2], t[0])));
    r.y = f64_fma(v.x, m[3], f64_fma(v.y, m[4], f64_fma(v.z, m[5], t[1])));
    r.z = f64_fma(v.x, m[6], f64_fma(v.y, m[7], f64_fma(v.z, m[8], t[2])));
    return r;
}

inline void vb_expand(thread vb_vec3d& mn, thread vb_vec3d& mx, vb_vec3d p)
{
    if (f64_lt(p.x, mn.x))
        mn.x = p.x;
    if (f64_lt(p.y, mn.y))
        mn.y = p.y;
    if (f64_lt(p.z, mn.z))
        mn.z = p.z;
    if (f64_lt(mx.x, p.x))
        mx.x = p.x;
    if (f64_lt(mx.y, p.y))
        mx.y = p.y;
    if (f64_lt(mx.z, p.z))
        mx.z = p.z;
}

// rebuild_finalize_world_bbox
kernel void vb_world_bbox(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i != 0 || !a.counts[4])
        return;
    device uint* g = a.grid;
    const ulong dbl_max = 0x7feffffffffffffful;
    vb_vec3d mn = { dbl_max, dbl_max, dbl_max };
    vb_vec3d mx = { dbl_max | (1ul << 63), dbl_max | (1ul << 63), dbl_max | (1ul << 63) };
    if (a.counts[2] > 0) {
        const int3 rmin = vb_rd_coord(g, VB_ROOT);
        const int3 rmax = vb_rd_coord(g, VB_ROOT + 12);
        const ulong x0 = f64_from_long(rmin.x), y0 = f64_from_long(rmin.y), z0 = f64_from_long(rmin.z);
        const ulong x1 = f64_from_long(long(rmax.x) + 1), y1 = f64_from_long(long(rmax.y) + 1),
                    z1 = f64_from_long(long(rmax.z) + 1);
        device const uint* h = a.header;
        const vb_vec3d p0 = vb_apply_map(h, vb_vec3d { x0, y0, z0 });
        mn = p0;
        mx = p0;
        vb_expand(mn, mx, vb_apply_map(h, vb_vec3d { x0, y0, z1 }));
        vb_expand(mn, mx, vb_apply_map(h, vb_vec3d { x0, y1, z0 }));
        vb_expand(mn, mx, vb_apply_map(h, vb_vec3d { x1, y0, z0 }));
        vb_expand(mn, mx, vb_apply_map(h, vb_vec3d { x1, y1, z0 }));
        vb_expand(mn, mx, vb_apply_map(h, vb_vec3d { x1, y0, z1 }));
        vb_expand(mn, mx, vb_apply_map(h, vb_vec3d { x0, y1, z1 }));
        vb_expand(mn, mx, vb_apply_map(h, vb_vec3d { x1, y1, z1 }));
    }
    vb_wr64(g, 560, mn.x);
    vb_wr64(g, 568, mn.y);
    vb_wr64(g, 576, mn.z);
    vb_wr64(g, 584, mx.x);
    vb_wr64(g, 592, mx.y);
    vb_wr64(g, 600, mx.z);
}

// test entry: out[i] = fma(in[3i], in[3i+1], in[3i+2]) on binary64 bit patterns
kernel void vb_test_f64_fma(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= a.n)
        return;
    device const ulong* in = (device const ulong*)a.points;
    a.keys[i] = f64_fma(in[3 * i], in[3 * i + 1], in[3 * i + 2]);
}

// ---------------------------------------------------------------------------------------------------------------
// get_tiles / get_voxels (wp_volume_get_tiles_host / wp_volume_get_voxels_host); grid layout read from the buffer

kernel void vb_get_tiles(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    device const uint* g = a.grid;
    const uint leaf_count = min(a.desc_leaf_count, vb_rd32(g, VB_TREE + 32));
    if (i >= leaf_count)
        return;
    const ulong lf = VB_TREE + vb_rd64(g, VB_TREE + 0) + ulong(i) * a.leaf_size;
    const int3 o = vb_rd_coord(g, lf) & int3(~7);
    a.out_coords[3 * i] = o.x;
    a.out_coords[3 * i + 1] = o.y;
    a.out_coords[3 * i + 2] = o.z;
}

kernel void vb_get_voxels(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant vb_args_t& a [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint grid [[threads_per_grid]])
{
    device const uint* g = a.grid;
    const uint actual_leaf_count = vb_rd32(g, VB_TREE + 32);
    const uint leaf_count = min(a.desc_leaf_count, actual_leaf_count);
    const ulong actual_voxel_count
        = a.grid_type == VB_GT_ONINDEX ? vb_rd64(g, VB_TREE + 56) : ulong(actual_leaf_count) * 512ul;
    const ulong first_leaf = VB_TREE + vb_rd64(g, VB_TREE + 0);
    for (ulong j = tid; j < ulong(leaf_count) * 512ul; j += grid) {
        const uint leaf = uint(j >> 9), n = uint(j & 511ul);
        const ulong lf = first_leaf + ulong(leaf) * a.leaf_size;
        const int3 ijk = (vb_rd_coord(g, lf) & int3(~7)) + int3(int((n >> 6) & 7u), int((n >> 3) & 7u), int(n & 7u));
        const ulong value = lf + a.leaf_off_table;
        ulong index;
        if (a.grid_type == VB_GT_INDEX) {
            index = vb_rd64(g, value) + n - 1ul;
        } else if (a.grid_type == VB_GT_ONINDEX) {
            const uint word = n >> 6, bit = n & 63u;
            const ulong mask = vb_rd64(g, lf + VB_LEAF_VALUE_MASK + 8ul * word);
            ulong value_index = 0;
            if ((mask >> bit) & 1ul) {
                uint sum = popcount(mask & ((1ul << bit) - 1ul));
                if (word > 0)
                    sum += uint(vb_rd64(g, value + 8) >> (9u * (word - 1u))) & 511u;
                value_index = vb_rd64(g, value) + sum;
            }
            index = value_index - 1ul;
        } else {
            index = ulong(leaf) * 512ul + n;
        }
        if (index < a.desc_voxel_count && index < actual_voxel_count) {
            a.out_coords[3 * index] = ijk.x;
            a.out_coords[3 * index + 1] = ijk.y;
            a.out_coords[3 * index + 2] = ijk.z;
        }
    }
}
"""


class _VbArgs(ctypes.Structure):
    _fields_ = [
        ("points", ctypes.c_uint64),
        ("mask", ctypes.c_uint64),
        ("keys", ctypes.c_uint64),
        ("upper_keys", ctypes.c_uint64),
        ("leaf_keys", ctypes.c_uint64),
        ("lower_keys", ctypes.c_uint64),
        ("voxel_keys", ctypes.c_uint64),
        ("counts", ctypes.c_uint64),
        ("status", ctypes.c_uint64),
        ("grid", ctypes.c_uint64),
        ("header", ctypes.c_uint64),
        ("leaf_counts", ctypes.c_uint64),
        ("tile_sums", ctypes.c_uint64),
        ("out_coords", ctypes.c_uint64),
        ("grid_size", ctypes.c_uint64),
        ("off_upper", ctypes.c_uint64),
        ("off_lower", ctypes.c_uint64),
        ("off_leaf", ctypes.c_uint64),
        ("cap_voxel", ctypes.c_uint64),
        ("desc_voxel_count", ctypes.c_uint64),
        ("n", ctypes.c_uint32),
        ("num_tiles", ctypes.c_uint32),
        ("world", ctypes.c_uint32),
        ("grid_type", ctypes.c_uint32),
        ("active_voxels", ctypes.c_uint32),
        ("check_capacity", ctypes.c_uint32),
        ("cap_leaf", ctypes.c_uint32),
        ("cap_lower", ctypes.c_uint32),
        ("cap_upper", ctypes.c_uint32),
        ("value_size", ctypes.c_uint32),
        ("num_out", ctypes.c_uint32),
        ("desc_leaf_count", ctypes.c_uint32),
        ("out_shift", ctypes.c_uint32 * 3),
        ("out_slot", ctypes.c_uint32 * 3),
        ("bg", ctypes.c_uint32 * 8),
        ("root_off_background", ctypes.c_uint32),
        ("root_size", ctypes.c_uint32),
        ("value_stride_bits", ctypes.c_uint32),
        ("table_stride", ctypes.c_uint32),
        ("root_tile_size", ctypes.c_uint32),
        ("upper_off_table", ctypes.c_uint32),
        ("upper_size", ctypes.c_uint32),
        ("lower_off_table", ctypes.c_uint32),
        ("lower_size", ctypes.c_uint32),
        ("leaf_off_table", ctypes.c_uint32),
        ("leaf_size", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


_VB_PTR_OFFSETS = (ctypes.c_size_t * 14)(*(8 * i for i in range(14)))
_SORT_PTR_OFFSETS = _prim._SORT_PTR_OFFSETS
_SCAN_PTR_OFFSETS = _prim._SCAN_PTR_OFFSETS

# unique output slots (counts index)
_SLOT_LEAF, _SLOT_LOWER, _SLOT_UPPER, _SLOT_VOXEL = 0, 1, 2, 3


class _State(_prim._DeviceState):
    """Compiled builder library, kernel handles, scratch pool and header templates of one Metal device."""

    def __init__(self, device):
        from warp._src.context import runtime  # noqa: PLC0415

        self.device = device
        self.core = runtime.core
        source = _prim._MSL_SOURCE + _VB_SOURCE
        self.library = self.core.wp_metal_load_library(device.metal_ordinal, source.encode("utf-8"))
        if not self.library:
            raise RuntimeError(f"Failed to compile the Metal volume builder kernels:\n{runtime.get_error_string()}")
        self.kernels = {}
        self.scratch = []  # grow-only; captured graphs may reference outgrown buffers
        self.templates = {}


def _state(device) -> _State:
    state = device.__dict__.get("_metal_volume")
    if state is None:
        state = _State(device)
        device.__dict__["_metal_volume"] = state
    return state


def _align(x, a=256):
    return (x + a - 1) // a * a


def _groups(threads):
    return (threads + _TG - 1) // _TG


# ---------------------------------------------------------------------------------------------------------------
# header template: a one-point host build with the same transform and type gives the map and voxel size bytes


def _header_template(state, kind, transform_buf, translation_buf, value_type=None, cvalue=None):
    """672-byte grid header of a host-built grid of this kind / type / transform: (device array, bytes)."""
    key = (
        kind,
        value_type,
        bytes(cvalue) if cvalue is not None else None,
        bytes(transform_buf),
        bytes(translation_buf),
    )
    cached = state.templates.get(key)
    if cached is not None:
        return cached

    core = state.core
    point = (ctypes.c_int32 * 3)(0, 0, 0)
    point_ptr = ctypes.cast(point, ctypes.c_void_p)
    if kind == "voxels":
        host_id = core.wp_volume_from_active_voxels_host(
            point_ptr, 1, None, transform_buf, translation_buf, False, False, 0, 0, 0, 0, None
        )
    elif value_type is None:
        host_id = core.wp_volume_index_from_tiles_host(
            point_ptr, 1, None, transform_buf, translation_buf, False, False, 0, 0, 0, None
        )
    else:
        host_id = core.wp_volume_from_tiles_host(
            point_ptr,
            1,
            None,
            transform_buf,
            translation_buf,
            False,
            ctypes.pointer(cvalue),
            ctypes.sizeof(cvalue),
            value_type.encode("ascii") if isinstance(value_type, str) else value_type,
            False,
            0,
            0,
            0,
            None,
        )
    if not host_id:
        raise RuntimeError("Failed to build the volume header template")
    buf = ctypes.c_void_p(0)
    size = ctypes.c_uint64(0)
    core.wp_volume_get_buffer_info(host_id, ctypes.byref(buf), ctypes.byref(size))
    data = ctypes.string_at(buf.value, _GRID_HEADER_SIZE)
    core.wp_volume_destroy_host(host_id)

    # a new Metal allocation is host-visible (unified memory) and not yet used by any GPU work: fill it from the
    # host without a synchronization (also during a graph capture)
    header = wp.empty(_GRID_HEADER_SIZE, dtype=wp.uint8, device=state.device)
    ctypes.memmove(header.ptr, data, _GRID_HEADER_SIZE)
    state.templates[key] = (header, data)
    return header, data


# ---------------------------------------------------------------------------------------------------------------
# build


class _Layout:
    """Byte offsets of the node arrays of a grid with the given capacities (rebuildable_grid_size)."""

    def __init__(self, grid_type, cap_leaf, cap_lower, cap_upper):
        c = _GRID_CONSTANTS[grid_type]
        self.constants = c
        root_size, root_tile_size, upper_size, lower_size, leaf_size = c[1], c[4], c[6], c[8], c[10]
        self.off_upper = _ROOT_OFFSET + root_size + root_tile_size * cap_upper
        self.off_lower = self.off_upper + upper_size * cap_upper
        self.off_leaf = self.off_lower + lower_size * cap_lower
        self.grid_size = self.off_leaf + leaf_size * cap_leaf


def _fill_constants(args, grid_type):
    c = _GRID_CONSTANTS[grid_type]
    (
        args.root_off_background,
        args.root_size,
        args.value_stride_bits,
        args.table_stride,
        args.root_tile_size,
        args.upper_off_table,
        args.upper_size,
        args.lower_off_table,
        args.lower_size,
        args.leaf_off_table,
        args.leaf_size,
    ) = c


def _sort_keys(state, b0, b1, b2, count, hist, misc):
    """Keys-only radix sort of ``count`` 64-bit keys at ``b0`` (result in ``b0``; ``b1`` / ``b2`` scratch)."""
    num_tiles = (count + _SORT_TILE - 1) // _SORT_TILE
    passes = [(s, 8) for s in range(0, 64, 8)]
    args = _prim._SortArgs()
    args.keys[:] = (b0, b1, b2)
    args.values[:] = (0, 0, 0)
    args.tile_hist, args.digit_total, args.diff, args.plan = hist, misc, misc + 1024, misc + 1024 + 8
    args.n, args.num_tiles, args.mode = count, num_tiles, 0
    # m = last pass: buffer 0 ends with the sorted keys whichever passes are skipped
    args.num_passes, args.m = len(passes), len(passes) - 1
    for i, (shift, bits) in enumerate(passes):
        args.shifts[i], args.widths[i] = shift, bits

    grid = min(num_tiles, _SORT_GRID)
    state.launch("sort_diff_clear", 0, args, _SORT_PTR_OFFSETS, threads=2)
    state.launch("sort_diff_k8", grid, args, _SORT_PTR_OFFSETS)
    state.launch("sort_plan", 0, args, _SORT_PTR_OFFSETS, threads=1)

    copy_args = _prim._SortArgs.from_buffer_copy(args)
    copy_args.shift, copy_args.bits = count * 2, 0
    copy_threads = min(count * 2, _SORT_GRID * _TG)
    copy_args.pass_ = 0
    state.launch("sort_copy", 0, copy_args, _SORT_PTR_OFFSETS, threads=copy_threads)
    for i, (shift, bits) in enumerate(passes):
        args.pass_, args.shift, args.bits = i, shift, bits
        state.launch("sort_upsweep_k8", grid, args, _SORT_PTR_OFFSETS)
        state.launch("sort_scan_tiles", 256, args, _SORT_PTR_OFFSETS)
        state.launch("vb_sort_scatter_keys", grid, args, _SORT_PTR_OFFSETS)
    copy_args.pass_ = 1
    state.launch("sort_copy", 0, copy_args, _SORT_PTR_OFFSETS, threads=copy_threads)


def _unique(state, args, outputs, tile_sums):
    """Unique of the sorted ``args.keys`` shifted by each ``(shift, slot)`` of ``outputs``; counts on the GPU."""
    num_tiles = (args.n + _SCAN_TILE - 1) // _SCAN_TILE
    args.num_tiles = num_tiles
    args.num_out = len(outputs)
    for o, (shift, slot) in enumerate(outputs):
        args.out_shift[o], args.out_slot[o] = shift, slot
    args.tile_sums = tile_sums
    state.launch("vb_unique_count", num_tiles, args, _VB_PTR_OFFSETS)
    for o in range(len(outputs)):
        scan_args = _prim._ScanArgs(0, 0, tile_sums + 4 * o * num_tiles, 0, 0, 0, 0, 4, 4, 0, num_tiles, 0, 0)
        state.launch("scan_spine_i4", 1, scan_args, _SCAN_PTR_OFFSETS)
    state.launch("vb_unique_scatter", num_tiles, args, _VB_PTR_OFFSETS)


class _Build:
    """Scratch layout and arguments of one build over ``n`` points."""

    def __init__(self, state, n, cap_leaf_hint):
        self.state = state
        self.n = n
        keys_bytes = _align(8 * n)
        sort_tiles = (n + _SORT_TILE - 1) // _SORT_TILE
        scan_tiles = (n + _SCAN_TILE - 1) // _SCAN_TILE
        leaf_scan_tiles = (max(cap_leaf_hint, 1) + _SCAN_TILE - 1) // _SCAN_TILE
        sizes = [
            ("k0", keys_bytes),
            ("k1", keys_bytes),
            ("k2", keys_bytes),
            ("upper", keys_bytes),
            ("leaf", keys_bytes),
            ("lower", keys_bytes),
            ("voxel", keys_bytes),
            ("counts", 256),
            ("hist", _align(256 * 4 * sort_tiles)),
            ("misc", _align(1024 + 8 + 48 * 4)),
            ("tile_sums", _align(3 * 4 * scan_tiles)),
            ("leaf_counts", _align(4 * max(cap_leaf_hint, 1))),
            ("leaf_scan", _align(4 * leaf_scan_tiles)),
        ]
        total = sum(s for _, s in sizes)
        base = state.scratch_ptr(total)
        self.ptr = {}
        for name, size in sizes:
            self.ptr[name] = base
            base += size


def _count(state, build, points_ptr, world, mask_ptr, header_ptr, active_voxels):
    """Sorted unique keys of every level and their counts (``counts[0..3]``) on the GPU."""
    p = build.ptr
    n = build.n
    args = _VbArgs()
    args.points, args.mask, args.keys = points_ptr, mask_ptr, p["k0"]
    args.upper_keys, args.leaf_keys, args.lower_keys, args.voxel_keys = p["upper"], p["leaf"], p["lower"], p["voxel"]
    args.counts, args.header = p["counts"], header_ptr
    args.n, args.world, args.active_voxels = n, int(world), int(active_voxels)

    groups = _groups(n)
    state.launch("vb_emit_upper_keys", groups, args, _VB_PTR_OFFSETS)
    _sort_keys(state, p["k0"], p["k1"], p["k2"], n, p["hist"], p["misc"])
    _unique(state, args, [(0, _SLOT_UPPER)], p["tile_sums"])

    state.launch("vb_emit_hierarchy_keys", groups, args, _VB_PTR_OFFSETS)
    _sort_keys(state, p["k0"], p["k1"], p["k2"], n, p["hist"], p["misc"])
    outputs = [(9, _SLOT_LEAF), (21, _SLOT_LOWER)]
    if active_voxels:
        outputs.append((0, _SLOT_VOXEL))
    _unique(state, args, outputs, p["tile_sums"])
    return args


def _populate(state, build, args, grid_ptr, layout, grid_type, caps, status_ptr, check_capacity, bg_words, value_size):
    """Capacity status, then the grid (skipped on the GPU when the counts exceed the capacities)."""
    p = build.ptr
    cap_leaf, cap_lower, cap_upper, cap_voxel = caps
    args.status = status_ptr
    args.grid = grid_ptr
    args.leaf_counts = p["leaf_counts"]
    args.grid_size = layout.grid_size
    args.off_upper, args.off_lower, args.off_leaf = layout.off_upper, layout.off_lower, layout.off_leaf
    args.cap_voxel = cap_voxel
    args.cap_leaf, args.cap_lower, args.cap_upper = cap_leaf, cap_lower, cap_upper
    args.check_capacity = int(check_capacity)
    args.grid_type = grid_type
    args.value_size = value_size
    if bg_words is not None:
        args.bg[: len(bg_words)] = bg_words
    _fill_constants(args, grid_type)
    regular = grid_type not in (GRID_TYPE_INDEX, GRID_TYPE_ONINDEX)

    one = 1
    state.launch("vb_finalize_counts", 0, args, _VB_PTR_OFFSETS, threads=one)
    zero_threads = max(min(max(layout.grid_size // 16, cap_leaf), _MAX_GRID_THREADS), 1)
    state.launch("vb_zero", 0, args, _VB_PTR_OFFSETS, threads=zero_threads)
    state.launch("vb_init_grid", 0, args, _VB_PTR_OFFSETS, threads=one)
    state.launch("vb_build_upper", 0, args, _VB_PTR_OFFSETS, threads=max(cap_upper, 1))
    if regular:
        threads = max(min(cap_upper * 32768, _MAX_GRID_THREADS), 1)
        state.launch("vb_upper_background", 0, args, _VB_PTR_OFFSETS, threads=threads)
    state.launch("vb_build_lower", 0, args, _VB_PTR_OFFSETS, threads=max(cap_lower, 1))
    if regular:
        threads = max(min(cap_lower * 4096, _MAX_GRID_THREADS), 1)
        state.launch("vb_lower_background", 0, args, _VB_PTR_OFFSETS, threads=threads)
    state.launch("vb_build_leaf", 0, args, _VB_PTR_OFFSETS, threads=max(cap_leaf, 1))
    if args.active_voxels and grid_type == GRID_TYPE_ONINDEX:
        threads = max(min(cap_voxel, _MAX_GRID_THREADS), 1)
        state.launch("vb_set_active_voxels", 0, args, _VB_PTR_OFFSETS, threads=threads)
        # exclusive scan of the per-leaf voxel counts, in place (zeros past the leaf count)
        leaf_n = max(cap_leaf, 1)
        num_tiles = (leaf_n + _SCAN_TILE - 1) // _SCAN_TILE
        scan_args = _prim._ScanArgs(
            p["leaf_counts"], p["leaf_counts"], p["leaf_scan"], 0, 0, 0, 0, 4, 4, leaf_n, num_tiles, 0, 0
        )
        state.launch("scan_reduce_i4", num_tiles, scan_args, _SCAN_PTR_OFFSETS)
        state.launch("scan_spine_i4", 1, scan_args, _SCAN_PTR_OFFSETS)
        state.launch("scan_down_i4", num_tiles, scan_args, _SCAN_PTR_OFFSETS)
        state.launch("vb_finalize_onindex_leaves", 0, args, _VB_PTR_OFFSETS, threads=max(cap_leaf, 1))
    elif regular:
        threads = max(min(cap_leaf * 512, _MAX_GRID_THREADS), 1)
        state.launch("vb_leaf_values", 0, args, _VB_PTR_OFFSETS, threads=threads)
    state.launch("vb_reset_bboxes", 0, args, _VB_PTR_OFFSETS, threads=max(cap_lower, cap_upper, 1))
    state.launch("vb_propagate_leaf_bboxes", 0, args, _VB_PTR_OFFSETS, threads=max(cap_leaf, 1))
    state.launch("vb_propagate_lower_bboxes", 0, args, _VB_PTR_OFFSETS, threads=max(cap_lower, 1))
    state.launch("vb_propagate_upper_bboxes", 0, args, _VB_PTR_OFFSETS, threads=max(cap_upper, 1))
    state.launch("vb_world_bbox", 0, args, _VB_PTR_OFFSETS, threads=one)


def _bg_words(cvalue, value_size):
    if cvalue is None:
        return None, 8 if value_size is None else value_size
    raw = bytes(cvalue)[:value_size]
    raw = raw + b"\0" * (-len(raw) % 4)
    return [int(w) for w in np.frombuffer(raw, dtype=np.uint32)], value_size


def allocate(
    device,
    points,
    num_points,
    point_mask,
    transform_buf,
    translation_buf,
    in_world_space,
    kind,
    value_type=None,
    cvalue=None,
):
    """Exact build of a non-rebuildable volume; returns ``(volume id, grid array)`` (id 0 when the build fails).

    ``kind`` is ``"voxels"`` (on-index grid of the active voxels) or ``"tiles"`` (index grid when ``value_type`` is
    ``None``, otherwise a value grid of ``value_type`` with background ``cvalue``).
    """
    if num_points <= 0:
        return 0, None  # the host builder rejects empty point sets
    state = _state(device)
    active_voxels = kind == "voxels"
    if active_voxels:
        grid_type, value_size = GRID_TYPE_ONINDEX, 8
    elif value_type is None:
        grid_type, value_size = GRID_TYPE_INDEX, 8
    else:
        grid_type, value_size = _VALUE_TYPES[value_type]
    header, header_bytes = _header_template(state, kind, transform_buf, translation_buf, value_type, cvalue)

    # the leaf count is at most the point count
    build = _Build(state, num_points, num_points)
    args = _count(
        state,
        build,
        points.ptr,
        in_world_space,
        point_mask.ptr if point_mask is not None else 0,
        header.ptr,
        active_voxels,
    )
    # exact sizes: read the counts back (as on CUDA)
    wp.synchronize_device(device)
    counts = np.frombuffer(ctypes.string_at(build.ptr["counts"], 16), dtype=np.uint32)
    leaf, lower, upper = int(counts[0]), int(counts[1]), int(counts[2])
    voxel = int(counts[3]) if active_voxels else leaf * 512
    layout = _Layout(grid_type, leaf, lower, upper)
    grid = wp.empty(shape=(layout.grid_size,), dtype=wp.uint8, device=device)

    # The GPU is idle after the readback and the allocation is new: write the grid and tree headers (all the volume
    # descriptor reads) from the host and register the volume now, so that wp_volume_create_device's synchronization
    # has nothing to wait for and the population below stays asynchronous. The kernels rewrite the same headers
    # (and the world bounding box, which the descriptor does not use).
    head = bytearray(header_bytes)
    head[32:40] = layout.grid_size.to_bytes(8, "little")
    head[640:648] = layout.grid_size.to_bytes(8, "little")
    head[656:664] = (0 if grid_type not in (GRID_TYPE_INDEX, GRID_TYPE_ONINDEX) else voxel + 1).to_bytes(8, "little")
    tree = struct.pack(
        "<QQQQIIIIIIQ",
        layout.off_leaf - _GRID_HEADER_SIZE,
        layout.off_lower - _GRID_HEADER_SIZE,
        layout.off_upper - _GRID_HEADER_SIZE,
        _ROOT_OFFSET - _GRID_HEADER_SIZE,
        leaf,
        lower,
        upper,
        leaf,
        lower,
        upper,
        voxel,
    )
    ctypes.memmove(grid.ptr, bytes(head) + tree, _ROOT_OFFSET)
    volume_id = state.core.wp_volume_create_device(
        device.context, ctypes.c_void_p(grid.ptr), layout.grid_size, False, False
    )
    if not volume_id:
        return 0, None

    bg_words, value_size = _bg_words(cvalue, value_size)
    _populate(
        state,
        build,
        args,
        grid.ptr,
        layout,
        grid_type,
        (leaf, lower, upper, voxel),
        0,
        False,
        bg_words,
        value_size,
    )
    return volume_id, grid


def rebuild(
    volume,
    points,
    num_points,
    point_mask,
    transform_buf,
    translation_buf,
    in_world_space,
    status,
    value_type=None,
    cvalue=None,
):
    """In-place rebuild of a rebuildable volume on the GPU (no host synchronization).

    The status word is written by a kernel; the grid is left untouched when the counts exceed the capacities.
    """
    device = volume.device
    state = _state(device)
    info = volume.get_rebuild_info()
    kind = info.kind
    active_voxels = kind == "voxels"
    if active_voxels:
        grid_type, value_size = GRID_TYPE_ONINDEX, 8
    elif value_type is None:
        grid_type, value_size = GRID_TYPE_INDEX, 8
    else:
        grid_type, value_size = _VALUE_TYPES[
            value_type.decode("ascii") if isinstance(value_type, bytes) else value_type
        ]
    if active_voxels:
        caps = (info.max_leaf_node_count, info.max_lower_node_count, info.max_upper_node_count, info.max_voxel_count)
    else:
        caps = (
            info.max_leaf_node_count,
            info.max_lower_node_count,
            info.max_upper_node_count,
            info.max_leaf_node_count * 512,
        )

    if num_points <= 0 or caps[0] == 0 or caps[1] == 0 or caps[2] == 0:
        if status is not None:
            status.fill_(wp.uint32(16))  # REBUILD_INVALID_INPUT, grid untouched
        return

    buf = ctypes.c_void_p(0)
    size = ctypes.c_uint64(0)
    state.core.wp_volume_get_buffer_info(volume.id, ctypes.byref(buf), ctypes.byref(size))
    layout = _Layout(grid_type, caps[0], caps[1], caps[2])
    if layout.grid_size != size.value:
        raise RuntimeError("Unexpected rebuildable volume buffer size")

    header, _ = _header_template(
        state,
        kind,
        transform_buf,
        translation_buf,
        value_type.decode("ascii") if isinstance(value_type, bytes) else value_type,
        cvalue,
    )
    build = _Build(state, num_points, caps[0])
    args = _count(
        state,
        build,
        points.ptr,
        in_world_space,
        point_mask.ptr if point_mask is not None else 0,
        header.ptr,
        active_voxels,
    )
    bg_words, value_size = _bg_words(cvalue, value_size)
    _populate(
        state,
        build,
        args,
        buf.value,
        layout,
        grid_type,
        caps,
        status.ptr if status is not None else 0,
        True,
        bg_words,
        value_size,
    )


# ---------------------------------------------------------------------------------------------------------------
# get_tiles / get_voxels


def grid_type_of(volume):
    """Grid type of ``volume`` if the GPU ``get_voxels`` / ``get_tiles`` support it, else ``None``."""
    return _TYPE_STR.get(volume.get_grid_info().type_str)


def _grid_args(volume, grid_type, out):
    state = _state(volume.device)
    buf = ctypes.c_void_p(0)
    size = ctypes.c_uint64(0)
    state.core.wp_volume_get_buffer_info(volume.id, ctypes.byref(buf), ctypes.byref(size))
    leaf_count = ctypes.c_uint32(0)
    voxel_count = ctypes.c_uint64(0)
    state.core.wp_volume_get_tile_and_voxel_count(volume.id, ctypes.byref(leaf_count), ctypes.byref(voxel_count))
    args = _VbArgs()
    args.grid = buf.value
    args.out_coords = out.ptr or 0
    args.grid_type = grid_type
    args.desc_leaf_count = leaf_count.value
    args.desc_voxel_count = voxel_count.value
    _fill_constants(args, grid_type)
    return state, args


def get_tiles(volume, grid_type, out):
    state, args = _grid_args(volume, grid_type, out)
    if args.desc_leaf_count:
        state.launch("vb_get_tiles", 0, args, _VB_PTR_OFFSETS, threads=args.desc_leaf_count)


def get_voxels(volume, grid_type, out):
    state, args = _grid_args(volume, grid_type, out)
    if args.desc_leaf_count and args.desc_voxel_count:
        threads = min(args.desc_leaf_count * 512, _MAX_GRID_THREADS)
        state.launch("vb_get_voxels", 0, args, _VB_PTR_OFFSETS, threads=threads)


# ---------------------------------------------------------------------------------------------------------------
# test hook


def _test_f64_fma(device, triples: np.ndarray) -> np.ndarray:
    """``fma`` of the binary64 bit patterns ``triples[:, 0..2]`` by the GPU software implementation."""
    state = _state(device)
    n = triples.shape[0]
    inp = wp.array(np.ascontiguousarray(triples, dtype=np.uint64).reshape(-1), dtype=wp.uint64, device=device)
    out = wp.empty(n, dtype=wp.uint64, device=device)
    args = _VbArgs()
    args.points = inp.ptr
    args.keys = out.ptr
    args.n = n
    state.launch("vb_test_f64_fma", 0, args, _VB_PTR_OFFSETS, threads=n)
    return out.numpy()
