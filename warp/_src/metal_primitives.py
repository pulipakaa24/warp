# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU versions of the parallel utilities (radix sort, integer scan, run-length encode) on Metal.

The CUDA backend runs these utilities through CUB; on Metal they used to run on the host (Metal arrays are
unified memory) after draining the GPU. The kernels here are hand-written Metal Shading Language, compiled once
per device through the same native entry points as Warp's generated Metal modules (``wp_metal_load_library``,
``wp_metal_get_kernel``) and dispatched with ``wp_metal_launch_kernel``, so they are ordered with other kernel
launches without a host synchronization and are recorded into Metal graph captures like any kernel dispatch.

Every result is bitwise identical to the host implementation in ``warp/native/{sort,scan,runlength_encode}.cpp``:

- ``radix_sort_pairs``: a stable LSD radix sort with 8-bit digits and three reduce-then-scan dispatches per digit
  (tile histograms, a scan of the histogram table, a stable scatter). No dispatch waits on another threadgroup
  (no decoupled look-back), so the sort needs no forward-progress guarantee between threadgroups, which Apple GPUs
  do not give. Ranks within a tile come from SIMD ballots (deterministic, no atomics), so equal keys keep their
  input order. A stable sort by a given bit range has exactly one result; the host sorts 16 bits per pass and
  ping-pongs between the two halves of the ``2*count`` buffers, and the GPU does each 16-bit host pass as two 8-bit
  passes through a scratch buffer, so both halves of the keys and values buffers end up exactly as on the host.
- ``array_scan`` for ``int32`` / ``int64`` scalars and vectors (integer addition is associative, so the parallel
  scan is exact). Floating-point scans stay on the host: a parallel scan reassociates the additions.
- ``runlength_encode`` (``int32``): head flags, a tile scan and a scatter; the run count is written to the
  ``run_count`` device array without a readback.

Dispatch shapes depend only on the element count, never on the data, and no step reads data back on the host.
Scratch memory is a per-device pool that only grows; outgrown buffers stay alive because captured graphs may
still reference them.
"""

from __future__ import annotations

import ctypes
import os

import warp as wp
from warp._src.types import type_repr

_TG = 256  # threads per threadgroup (the sort kernels give one thread to each of the 256 digits)
_SORT_KPT = 8  # keys per thread in a sort tile
_SORT_TILE = _TG * _SORT_KPT
# Threadgroups of the sort's tile loops: a fixed grid (a few per GPU core) keeps the cost of a skipped pass (every
# threadgroup returns at once) small; one threadgroup per tile made a skipped pass cost ~50 us at 9M keys.
_SORT_GRID = int(os.environ.get("WP_METAL_SORT_GRID", "320"))
_SCAN_IPT = 8  # elements per thread in a scan / run-length tile
_SCAN_TILE = _TG * _SCAN_IPT

_MSL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define TG 256u
#define NSG (TG / 32u)
#define SORT_KPT 8u
#define SORT_TILE (TG * SORT_KPT)
#define SCAN_IPT 8u
#define SCAN_TILE (TG * SCAN_IPT)

struct wp_prim_bounds_t { ulong unused; };

// ---------------------------------------------------------------------------------------------------------------
// threadgroup scans (TG threads, NSG SIMD groups of 32)

inline int simd_excl(int v, uint lane) { return simd_prefix_exclusive_sum(v); }
inline uint simd_excl(uint v, uint lane) { return simd_prefix_exclusive_sum(v); }
inline long simd_excl(long v, uint lane)
{
    long x = v;
    for (ushort o = 1; o < 32; o <<= 1) {
        uint2 p = as_type<uint2>(x);
        uint2 q = uint2(simd_shuffle_up(p.x, o), simd_shuffle_up(p.y, o));
        if (lane >= o)
            x += as_type<long>(q);
    }
    return x - v;
}

// Exclusive scan of one value per thread over the threadgroup; total receives the sum of all values.
// sm must hold NSG + 1 values. Ends with a barrier, so sm can be reused right away.
template <typename T>
inline T tg_exclusive_scan(T v, threadgroup T* sm, uint lane, uint sg, thread T& total)
{
    T x = simd_excl(v, lane);
    if (lane == 31)
        sm[sg] = x + v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        T s = lane < NSG ? sm[lane] : T(0);
        T e = simd_excl(s, lane);
        if (lane < NSG)
            sm[lane] = e;
        if (lane == NSG - 1)
            sm[NSG] = e + s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T r = x + sm[sg];
    total = sm[NSG];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// ---------------------------------------------------------------------------------------------------------------
// radix sort

// Order-preserving maps to unsigned integers, as in warp/native/sort.cpp (mode 0: unsigned, 1: signed, 2: float).
inline uint radix_of(uint k, int mode)
{
    if (mode == 1)
        return k ^ 0x80000000u;
    if (mode == 2)
        return k ^ (uint(-int(k >> 31)) | 0x80000000u);
    return k;
}

inline ulong radix_of(ulong k, int mode)
{
    if (mode == 1)
        return k ^ 0x8000000000000000ul;
    if (mode == 2)
        return k ^ (ulong(-long(k >> 63)) | 0x8000000000000000ul);
    return k;
}

// Buffers 0 and 1 are the two halves of the caller's 2*count keys / values arrays, buffer 2 a scratch buffer.
// The plan (written by sort_plan on the GPU) routes each 8-bit pass between them:
//   plan[4p + 0..3] = active, source, destination, second destination (3: none) of pass p
//   plan[40..42] = active, source, destination of the copy before the passes; plan[44..46] the copy after them
struct sort_args_t {
    device uchar* keys[3];
    device uchar* values[3];
    device uint* tile_hist;     // [256][num_tiles], digit-major
    device uint* digit_total;   // [256]
    device atomic_uint* diff;   // [2]: OR over all keys of radix(key) ^ radix(key[0]) (low, high word)
    device uint* plan;          // [48]
    uint n;
    uint num_tiles;
    uint shift;                 // this pass's digit; sort_copy: number of 4-byte key words
    uint bits;                  // sort_copy: number of 4-byte value words
    int mode;
    uint pass;                  // this pass's index; sort_copy: 0 before the passes, 1 after
    uint num_passes;
    int m;                      // the pass after which the host's second half holds its final contents
    uint shifts[8];
    uint widths[8];
    uint key_words;             // 4-byte words per key / value (sort_copy with a device bound)
    uint value_words;
    device const int* n_dev;    // optional device bound: only the first min(*n_dev, n) pairs are sorted
};

// Pairs sorted: n, or min(*n_dev, n) with a device bound (the pairs past it are left in place).
inline uint sort_count(constant sort_args_t& a)
{
    if (!a.n_dev)
        return a.n;
    const int c = *a.n_dev;
    return c <= 0 ? 0u : min(uint(c), a.n);
}
inline uint sort_tiles(uint n) { return (n + SORT_TILE - 1u) / SORT_TILE; }

kernel void sort_diff_clear(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant sort_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i < 2)
        atomic_store_explicit(&a.diff[i], 0u, memory_order_relaxed);
}

inline ulong as_ulong(uint x) { return ulong(x); }
inline ulong as_ulong(ulong x) { return x; }

// diff |= radix(key) ^ radix(key[0]) over all keys: bit b of diff is clear iff every key has the same bit b.
template <typename K>
kernel void sort_diff(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant sort_args_t& a [[buffer(1)]],
    uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint sm[2 * NSG];
    device const K* keys = (device const K*)a.keys[0];
    const K first = radix_of(keys[0], a.mode);
    const uint n = sort_count(a);
    K d = K(0);
    for (uint idx = tg * TG + lid; idx < n; idx += ntg * TG)
        d |= radix_of(keys[idx], a.mode) ^ first;
    ulong x = as_ulong(d);
    uint lo = simd_or(uint(x)), hi = simd_or(uint(x >> 32));
    if (lane == 0) {
        sm[sg] = lo;
        sm[NSG + sg] = hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 2) {
        uint r = 0;
        for (uint s = 0; s < NSG; ++s)
            r |= sm[lid * NSG + s];
        if (r)
            atomic_fetch_or_explicit(&a.diff[lid], r, memory_order_relaxed);
    }
}

// A pass whose digit is the same for every key leaves the order unchanged (the sort is stable), so it is skipped.
// The remaining passes are routed so that buffer 0 ends with the sorted pairs and buffer 1 with the pairs as they
// stand after pass m (what the host's second half holds), with at most one copy.
kernel void sort_plan(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant sort_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i != 0)
        return;
    device uint* plan = a.plan;
    device const uint* dw = (device const uint*)a.diff;
    const ulong diff = ulong(dw[0]) | (ulong(dw[1]) << 32);
    const int P = int(a.num_passes);
    bool act[8];
    int e_last = -1, e_m = -1;
    for (int p = 0; p < P; ++p) {
        act[p] = ((diff >> a.shifts[p]) & ((1ul << a.widths[p]) - 1ul)) != 0;
        if (act[p]) {
            e_last = p;
            if (p <= a.m)
                e_m = p;
        }
        plan[4 * p] = 0;
        plan[4 * p + 3] = 3;
    }
    plan[40] = 0;
    plan[44] = 0;

    // passes after e_m: buffer 1 holds their input; alternate between 0 and 2, ending in 0
    uint w = 0;
    for (int p = P - 1; p > e_m; --p) {
        if (act[p]) {
            plan[4 * p] = 1;
            plan[4 * p + 2] = w;
            w = 2 - w;
        }
    }
    if (e_m < 0) {
        // no pass up to m changes the order: buffer 1 is the input
        plan[40] = 1;
        plan[41] = 0;
        plan[42] = 1;
    }
    uint cur = 1;
    for (int p = e_m + 1; p < P; ++p) {
        if (act[p]) {
            plan[4 * p + 1] = cur;
            cur = plan[4 * p + 2];
        }
    }
    if (e_m >= 0) {
        // passes before e_m alternate between 2 and 1, ending in 2; e_m writes buffer 1 (and 0 if it is the last)
        uint w2 = 2;
        for (int p = e_m - 1; p >= 0; --p) {
            if (act[p]) {
                plan[4 * p] = 1;
                plan[4 * p + 2] = w2;
                w2 = 3 - w2;
            }
        }
        uint c = 0;
        for (int p = 0; p < e_m; ++p) {
            if (act[p]) {
                plan[4 * p + 1] = c;
                c = plan[4 * p + 2];
            }
        }
        plan[4 * e_m] = 1;
        plan[4 * e_m + 1] = c;
        plan[4 * e_m + 2] = 1;
        if (e_m == e_last) {
            if (c != 0) {
                plan[4 * e_m + 3] = 0;
            } else {
                // the only pass reads buffer 0: copy its result into buffer 0 afterwards
                plan[44] = 1;
                plan[45] = 1;
                plan[46] = 0;
            }
        }
    }
}

// Copies 4-byte words of keys and of values over a fixed grid.
kernel void sort_copy(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant sort_args_t& a [[buffer(1)]],
    uint i [[thread_position_in_grid]],
    uint grid [[threads_per_grid]])
{
    device const uint* c = a.plan + (a.pass ? 44 : 40);
    if (!c[0])
        return;
    device const uint* ks = (device const uint*)a.keys[c[1]];
    device uint* kd = (device uint*)a.keys[c[2]];
    device const uint* vs = (device const uint*)a.values[c[1]];
    device uint* vd = (device uint*)a.values[c[2]];
    uint kw = a.shift, vw = a.bits;
    if (a.n_dev) {
        const uint n = sort_count(a);
        kw = n * a.key_words;
        vw = n * a.value_words;
    }
    for (uint j = i; j < max(kw, vw); j += grid) {
        if (j < kw)
            kd[j] = ks[j];
        if (j < vw)
            vd[j] = vs[j];
    }
}

// Lanes of the SIMD group whose (valid) digit equals this lane's, from one ballot per digit bit.
inline uint match_digit(uint d, uint bits, bool valid)
{
    // fast path (uniform branch): every lane valid with the same digit, common for clustered keys
    if (simd_all(valid && d == simd_broadcast_first(d)))
        return 0xffffffffu;
    uint m = uint(ulong(simd_ballot(valid)));
    for (uint b = 0; b < bits; ++b) {
        bool s = ((d >> b) & 1u) != 0;
        uint v = uint(ulong(simd_ballot(s)));
        m &= s ? v : ~v;
    }
    return m;
}

// Tile histogram: SIMD group sg counts its 32*SORT_KPT contiguous keys into its own row (leader lanes add the
// popcount of their digit's lanes, no atomics), then the rows are summed into the [digit][tile] table.
template <typename K>
kernel void sort_upsweep(
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
    device const K* keys = (device const K*)a.keys[plan[1]];
    const K mask = (K(1) << a.bits) - K(1);
    const uint n = sort_count(a);
    const uint num_tiles = sort_tiles(n);
    // a fixed grid loops over the tiles, so that a skipped pass costs few threadgroups
    for (uint tile = tg; tile < num_tiles; tile += ntg) {
    for (uint i = lid; i < NSG * 256; i += TG)
        hist[i] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = tile * SORT_TILE + sg * 32u * SORT_KPT + lane;
    threadgroup uint* row = hist + sg * 256;
    for (uint j = 0; j < SORT_KPT; ++j) {
        uint idx = base + j * 32u;
        bool valid = idx < n;
        uint d = valid ? uint((radix_of(keys[idx], a.mode) >> a.shift) & mask) : 0u;
        uint m = match_digit(d, a.bits, valid);
        uint lt = popcount(m & ((1u << lane) - 1u));
        uint old = row[d];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (valid && lt == 0)
            row[d] = old + popcount(m);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sum = 0;
    for (uint s = 0; s < NSG; ++s)
        sum += hist[s * 256 + lid];
    a.tile_hist[lid * a.num_tiles + tile] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One threadgroup per digit: exclusive scan of the digit's row of tile counts, in place; the row total goes to
// digit_total.
kernel void sort_scan_tiles(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant sort_args_t& a [[buffer(1)]],
    uint digit [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    if (!a.plan[4 * a.pass])
        return;
    threadgroup uint sm[NSG + 1];
    device uint* row = a.tile_hist + digit * a.num_tiles;  // rows keep the full tile count as stride
    const uint num_tiles = sort_tiles(sort_count(a));
    uint carry = 0;
    for (uint start = 0; start < num_tiles; start += TG) {
        uint i = start + lid;
        uint v = i < num_tiles ? row[i] : 0u;
        uint total;
        uint e = tg_exclusive_scan(v, sm, lane, sg, total);
        if (i < num_tiles)
            row[i] = carry + e;
        carry += total;
    }
    if (lid == 0)
        a.digit_total[digit] = carry;
}

// Stable scatter of one tile: rank each key among the keys of the tile with the same digit (SIMD ballots in index
// order within a SIMD group, then the SIMD groups in order), add the tile's global offset for the digit.
template <typename K, typename V>
kernel void sort_scatter(
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

    // global start of each digit: the digits before it, over all tiles
    uint total;
    const uint digit_start = tg_exclusive_scan(a.digit_total[lid], sm, lane, sg, total);
    device const K* keys = (device const K*)a.keys[plan[1]];
    device const V* values = (device const V*)a.values[plan[1]];
    device K* keys_out = (device K*)a.keys[plan[2]];
    device V* values_out = (device V*)a.values[plan[2]];
    const bool second = plan[3] != 3;
    device K* keys_out2 = (device K*)a.keys[second ? plan[3] : plan[2]];
    device V* values_out2 = (device V*)a.values[second ? plan[3] : plan[2]];
    const K mask = (K(1) << a.bits) - K(1);
    const uint n = sort_count(a);
    const uint num_tiles = sort_tiles(n);

    // a fixed grid loops over the tiles, so that a skipped pass costs few threadgroups
    for (uint tile = tg; tile < num_tiles; tile += ntg) {
    for (uint i = lid; i < NSG * 256; i += TG)
        hist[i] = 0;
    // plus this digit in earlier tiles
    offset[lid] = digit_start + a.tile_hist[lid * a.num_tiles + tile];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = tile * SORT_TILE + sg * 32u * SORT_KPT + lane;
    threadgroup uint* row = hist + sg * 256;
    K key[SORT_KPT];
    ushort rank[SORT_KPT];
    uchar dig[SORT_KPT];
    for (uint j = 0; j < SORT_KPT; ++j) {
        uint idx = base + j * 32u;
        bool valid = idx < n;
        key[j] = valid ? keys[idx] : K(0);
        uint d = valid ? uint((radix_of(key[j], a.mode) >> a.shift) & mask) : 0u;
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

    // per digit (thread lid), turn the SIMD-group counts into exclusive offsets including the tile's global offset
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
        if (idx < n) {
            uint pos = row[dig[j]] + rank[j];
            V v = values[idx];
            keys_out[pos] = key[j];
            values_out[pos] = v;
            if (second) {
                keys_out2[pos] = key[j];
                values_out2[pos] = v;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("sort_diff_k4")]] kernel void sort_diff<uint>(
    constant wp_prim_bounds_t&, constant sort_args_t&, uint, uint, uint, uint, uint);
template [[host_name("sort_diff_k8")]] kernel void sort_diff<ulong>(
    constant wp_prim_bounds_t&, constant sort_args_t&, uint, uint, uint, uint, uint);
template [[host_name("sort_upsweep_k4")]] kernel void sort_upsweep<uint>(
    constant wp_prim_bounds_t&, constant sort_args_t&, uint, uint, uint, uint, uint);
template [[host_name("sort_upsweep_k8")]] kernel void sort_upsweep<ulong>(
    constant wp_prim_bounds_t&, constant sort_args_t&, uint, uint, uint, uint, uint);
template [[host_name("sort_scatter_k4_v4")]] kernel void sort_scatter<uint, uint>(
    constant wp_prim_bounds_t&, constant sort_args_t&, uint, uint, uint, uint, uint);
template [[host_name("sort_scatter_k4_v8")]] kernel void sort_scatter<uint, ulong>(
    constant wp_prim_bounds_t&, constant sort_args_t&, uint, uint, uint, uint, uint);
template [[host_name("sort_scatter_k8_v4")]] kernel void sort_scatter<ulong, uint>(
    constant wp_prim_bounds_t&, constant sort_args_t&, uint, uint, uint, uint, uint);
template [[host_name("sort_scatter_k8_v8")]] kernel void sort_scatter<ulong, ulong>(
    constant wp_prim_bounds_t&, constant sort_args_t&, uint, uint, uint, uint, uint);

// ---------------------------------------------------------------------------------------------------------------
// integer scan (reduce, spine scan, downsweep) and run-length encoding

struct scan_args_t {
    device const uchar* in;
    device uchar* out;
    device uchar* tile_sums;    // [num_tiles] of T
    device uchar* starts;       // run-length encoding: start index of each run
    device uchar* run_values;
    device uchar* run_lengths;
    device int* run_count;
    long in_stride;             // bytes
    long out_stride;            // bytes
    uint n;
    uint num_tiles;
    int inclusive;
    uint pad;
    device const int* n_dev;    // run-length encoding: optional device bound on the number of values
};

// Values encoded by the run-length encoding: n, or min(*n_dev, n) with a device bound (scans: n).
inline uint rle_count_of(constant scan_args_t& a)
{
    if (!a.n_dev)
        return a.n;
    const int c = *a.n_dev;
    return c <= 0 ? 0u : min(uint(c), a.n);
}

template <typename T>
inline T load_elem(constant scan_args_t& a, uint i) { return *(device const T*)(a.in + long(i) * a.in_stride); }

// Sum of each tile (any order: integer addition is associative).
template <typename T>
kernel void scan_reduce(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant scan_args_t& a [[buffer(1)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup T sm[NSG + 1];
    T s = T(0);
    for (uint j = 0; j < SCAN_IPT; ++j) {
        uint i = tile * SCAN_TILE + j * TG + lid;
        if (i < a.n)
            s += load_elem<T>(a, i);
    }
    T total;
    tg_exclusive_scan(s, sm, lane, sg, total);
    if (lid == 0)
        ((device T*)a.tile_sums)[tile] = total;
}

// Exclusive scan of the tile sums in place, by one threadgroup.
template <typename T>
kernel void scan_spine(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant scan_args_t& a [[buffer(1)]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup T sm[NSG + 1];
    device T* sums = (device T*)a.tile_sums;
    const uint num_tiles = a.n_dev ? (rle_count_of(a) + SCAN_TILE - 1u) / SCAN_TILE : a.num_tiles;
    T carry = T(0);
    for (uint start = 0; start < num_tiles; start += TG) {
        uint i = start + lid;
        T v = i < num_tiles ? sums[i] : T(0);
        T total;
        T e = tg_exclusive_scan(v, sm, lane, sg, total);
        if (i < num_tiles)
            sums[i] = carry + e;
        carry += total;
    }
}

// Each thread scans SCAN_IPT consecutive elements; in place is safe since every element is read before its own
// output is written and no thread writes another thread's elements.
template <typename T>
kernel void scan_down(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant scan_args_t& a [[buffer(1)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup T sm[NSG + 1];
    const uint first = tile * SCAN_TILE + lid * SCAN_IPT;
    T v[SCAN_IPT];
    T s = T(0);
    for (uint j = 0; j < SCAN_IPT; ++j) {
        uint i = first + j;
        v[j] = i < a.n ? load_elem<T>(a, i) : T(0);
        s += v[j];
    }
    T total;
    T run = tg_exclusive_scan(s, sm, lane, sg, total) + ((device const T*)a.tile_sums)[tile];
    for (uint j = 0; j < SCAN_IPT; ++j) {
        uint i = first + j;
        if (a.inclusive)
            run += v[j];
        if (i < a.n)
            *(device T*)(a.out + long(i) * a.out_stride) = run;
        if (!a.inclusive)
            run += v[j];
    }
}

template [[host_name("scan_reduce_i4")]] kernel void scan_reduce<int>(
    constant wp_prim_bounds_t&, constant scan_args_t&, uint, uint, uint, uint);
template [[host_name("scan_reduce_i8")]] kernel void scan_reduce<long>(
    constant wp_prim_bounds_t&, constant scan_args_t&, uint, uint, uint, uint);
template [[host_name("scan_spine_i4")]] kernel void scan_spine<int>(
    constant wp_prim_bounds_t&, constant scan_args_t&, uint, uint, uint);
template [[host_name("scan_spine_i8")]] kernel void scan_spine<long>(
    constant wp_prim_bounds_t&, constant scan_args_t&, uint, uint, uint);
template [[host_name("scan_down_i4")]] kernel void scan_down<int>(
    constant wp_prim_bounds_t&, constant scan_args_t&, uint, uint, uint, uint);
template [[host_name("scan_down_i8")]] kernel void scan_down<long>(
    constant wp_prim_bounds_t&, constant scan_args_t&, uint, uint, uint, uint);

inline int rle_head(uint n, device const int* v, uint i)
{
    return (i < n && (i == 0 || v[i] != v[i - 1])) ? 1 : 0;
}

// Number of run heads in each tile.
kernel void rle_count(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant scan_args_t& a [[buffer(1)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup int sm[NSG + 1];
    device const int* v = (device const int*)a.in;
    const uint n = rle_count_of(a);
    int s = 0;
    for (uint j = 0; j < SCAN_IPT; ++j)
        s += rle_head(n, v, tile * SCAN_TILE + j * TG + lid);
    int total;
    tg_exclusive_scan(s, sm, lane, sg, total);
    if (lid == 0)
        ((device int*)a.tile_sums)[tile] = total;
}

// Run index of every head from the scanned tile counts: write the run's value and start; the thread holding the
// last element writes the run count.
kernel void rle_scatter(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant scan_args_t& a [[buffer(1)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup int sm[NSG + 1];
    device const int* v = (device const int*)a.in;
    const uint n = rle_count_of(a);
    const uint first = tile * SCAN_TILE + lid * SCAN_IPT;
    int h[SCAN_IPT];
    int s = 0;
    for (uint j = 0; j < SCAN_IPT; ++j) {
        h[j] = rle_head(n, v, first + j);
        s += h[j];
    }
    int total;
    int run = tg_exclusive_scan(s, sm, lane, sg, total) + ((device const int*)a.tile_sums)[tile];
    for (uint j = 0; j < SCAN_IPT; ++j) {
        uint i = first + j;
        if (h[j]) {
            ((device int*)a.run_values)[run] = v[i];
            ((device int*)a.starts)[run] = int(i);
            ++run;
        }
        if (i == n - 1)
            *a.run_count = run;
    }
    if (n == 0 && tile == 0 && lid == 0)
        *a.run_count = 0;
}

// Run lengths from consecutive run starts (thread j < run count).
kernel void rle_lengths(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant scan_args_t& a [[buffer(1)]],
    uint j [[thread_position_in_grid]])
{
    int count = *a.run_count;
    if (int(j) >= count)
        return;
    device const int* starts = (device const int*)a.starts;
    int end = int(j) + 1 < count ? starts[j + 1] : int(rle_count_of(a));
    ((device int*)a.run_lengths)[j] = end - starts[j];
}

// ---------------------------------------------------------------------------------------------------------------
// strided / indexed array copy (CUDA: array_copy_{1,2,3,4}d_kernel in warp.cu), one thread per element

struct array_copy_args_t {
    device uchar* dst;
    device const uchar* src;
    device const int* dst_indices[4];
    device const int* src_indices[4];
    long dst_strides[4];        // bytes
    long src_strides[4];        // bytes
    uint shape[4];
    uint ndim;
    uint elem_size;             // bytes
    ulong n;
};

kernel void array_copy(
    constant wp_prim_bounds_t& bounds [[buffer(0)]],
    constant array_copy_args_t& a [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (ulong(tid) >= a.n)
        return;
    long so = 0, do_ = 0;
    ulong rest = tid;
    for (int d = int(a.ndim) - 1; d >= 0; --d) {
        const uint i = uint(rest % a.shape[d]);
        rest /= a.shape[d];
        const long si = a.src_indices[d] ? long(a.src_indices[d][i]) : long(i);
        const long di = a.dst_indices[d] ? long(a.dst_indices[d][i]) : long(i);
        so += si * a.src_strides[d];
        do_ += di * a.dst_strides[d];
    }
    device const uchar* p = a.src + so;
    device uchar* q = a.dst + do_;
    if ((a.elem_size & 3u) == 0u && ((ulong(p) | ulong(q)) & 3ul) == 0ul) {
        for (uint w = 0; w < a.elem_size / 4u; ++w)
            ((device uint*)q)[w] = ((device const uint*)p)[w];
    } else {
        for (uint b = 0; b < a.elem_size; ++b)
            q[b] = p[b];
    }
}
"""


class _SortArgs(ctypes.Structure):
    _fields_ = [
        ("keys", ctypes.c_uint64 * 3),
        ("values", ctypes.c_uint64 * 3),
        ("tile_hist", ctypes.c_uint64),
        ("digit_total", ctypes.c_uint64),
        ("diff", ctypes.c_uint64),
        ("plan", ctypes.c_uint64),
        ("n", ctypes.c_uint32),
        ("num_tiles", ctypes.c_uint32),
        ("shift", ctypes.c_uint32),
        ("bits", ctypes.c_uint32),
        ("mode", ctypes.c_int32),
        ("pass_", ctypes.c_uint32),
        ("num_passes", ctypes.c_uint32),
        ("m", ctypes.c_int32),
        ("shifts", ctypes.c_uint32 * 8),
        ("widths", ctypes.c_uint32 * 8),
        ("key_words", ctypes.c_uint32),
        ("value_words", ctypes.c_uint32),
        ("n_dev", ctypes.c_uint64),
    ]


class _ScanArgs(ctypes.Structure):
    _fields_ = [
        ("in_", ctypes.c_uint64),
        ("out", ctypes.c_uint64),
        ("tile_sums", ctypes.c_uint64),
        ("starts", ctypes.c_uint64),
        ("run_values", ctypes.c_uint64),
        ("run_lengths", ctypes.c_uint64),
        ("run_count", ctypes.c_uint64),
        ("in_stride", ctypes.c_int64),
        ("out_stride", ctypes.c_int64),
        ("n", ctypes.c_uint32),
        ("num_tiles", ctypes.c_uint32),
        ("inclusive", ctypes.c_int32),
        ("pad", ctypes.c_uint32),
        ("n_dev", ctypes.c_uint64),
    ]


class _ArrayCopyArgs(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.c_uint64),
        ("src", ctypes.c_uint64),
        ("dst_indices", ctypes.c_uint64 * 4),
        ("src_indices", ctypes.c_uint64 * 4),
        ("dst_strides", ctypes.c_int64 * 4),
        ("src_strides", ctypes.c_int64 * 4),
        ("shape", ctypes.c_uint32 * 4),
        ("ndim", ctypes.c_uint32),
        ("elem_size", ctypes.c_uint32),
        ("n", ctypes.c_uint64),
    ]


def _pointer_offsets(struct_type, count):
    offsets = [getattr(struct_type, name).offset for name, _ in struct_type._fields_[:count]]
    return (ctypes.c_size_t * len(offsets))(*offsets)


# the leading pointer fields, plus the device bound (n_dev, last so that positional constructions stay valid)
_SORT_PTR_OFFSETS = (ctypes.c_size_t * 11)(*(8 * i for i in range(10)), _SortArgs.n_dev.offset)
_SCAN_PTR_OFFSETS = (ctypes.c_size_t * 8)(*_pointer_offsets(_ScanArgs, 7), _ScanArgs.n_dev.offset)
_COPY_PTR_OFFSETS = (ctypes.c_size_t * 10)(*(8 * i for i in range(10)))
_BOUNDS = ctypes.c_uint64(0)


def enabled(device) -> bool:
    """Whether the utilities run as GPU kernels on the Metal ``device`` (``warp.config.metal_gpu_utilities``)."""
    return device.is_metal and wp.config.metal_gpu_utilities


class _DeviceState:
    """Compiled library, kernel handles and scratch pool of one Metal device."""

    def __init__(self, device):
        from warp._src.context import runtime  # noqa: PLC0415

        self.device = device
        self.core = runtime.core
        self.library = self.core.wp_metal_load_library(device.metal_ordinal, _MSL_SOURCE.encode("utf-8"))
        if not self.library:
            raise RuntimeError(f"Failed to compile the Metal utility kernels:\n{runtime.get_error_string()}")
        self.kernels = {}
        self.scratch = []  # grow-only; captured graphs may reference outgrown buffers

    def kernel(self, name):
        handle = self.kernels.get(name)
        if handle is None:
            handle = self.core.wp_metal_get_kernel(self.library, name.encode("utf-8"))
            if not handle:
                from warp._src.context import runtime  # noqa: PLC0415

                raise RuntimeError(f"Metal utility kernel '{name}' is unavailable: {runtime.get_error_string()}")
            self.kernels[name] = handle
        return handle

    def scratch_ptr(self, nbytes):
        """Address of a device scratch buffer of at least ``nbytes`` (contents undefined)."""
        if not self.scratch or self.scratch[-1].size < nbytes:
            size = max(nbytes, 2 * self.scratch[-1].size if self.scratch else 0, 1 << 16)
            self.scratch.append(wp.empty(size, dtype=wp.uint8, device=self.device))
        return self.scratch[-1].ptr

    def launch_bounded(self, name, limit, args, pointer_offsets, count, divisor, per_thread=False):
        """Dispatch of ``limit`` threadgroups (or threads, ``per_thread``) of which only the first
        ``ceil(count[0] / divisor)`` run, sized on the GPU (``wp_metal_launch_kernel_bounded``)."""
        from warp._src.context import runtime  # noqa: PLC0415

        if limit <= 0:
            return
        inner = 1 if per_thread else _TG
        if self.core.wp_metal_launch_kernel_bounded(
            self.device.metal_ordinal,
            self.kernel(name),
            limit * inner,
            _TG,
            ctypes.byref(_BOUNDS),
            ctypes.sizeof(_BOUNDS),
            ctypes.byref(args),
            ctypes.sizeof(args),
            pointer_offsets,
            len(pointer_offsets),
            0,
            ctypes.c_void_p(count.ptr),
            limit,
            inner,
            divisor,
        ):
            raise RuntimeError(f"Failed to launch Metal utility kernel '{name}': {runtime.get_error_string()}")

    def launch(self, name, num_groups, args, pointer_offsets, threads=None):
        from warp._src.context import runtime  # noqa: PLC0415

        if num_groups <= 0 and not threads:
            return
        if self.core.wp_metal_launch_kernel(
            self.device.metal_ordinal,
            self.kernel(name),
            threads if threads else num_groups * _TG,
            _TG,
            ctypes.byref(_BOUNDS),
            ctypes.sizeof(_BOUNDS),
            ctypes.byref(args),
            ctypes.sizeof(args),
            pointer_offsets,
            len(pointer_offsets),
            0,
        ):
            raise RuntimeError(f"Failed to launch Metal utility kernel '{name}': {runtime.get_error_string()}")


def _state(device) -> _DeviceState:
    state = device.__dict__.get("_metal_primitives")
    if state is None:
        state = _DeviceState(device)
        device.__dict__["_metal_primitives"] = state
    return state


def _align(x, a=256):
    return (x + a - 1) // a * a


_KEY_MODES = {
    wp.uint32: (4, 0),
    wp.int32: (4, 1),
    wp.float32: (4, 2),
    wp.uint64: (8, 0),
    wp.int64: (8, 1),
    wp.float64: (8, 2),
}


def radix_sort_pairs(keys, values, count: int, begin_bit: int, end_bit: int, value_size: int, count_limit=None) -> None:
    """GPU radix sort of the first ``count`` pairs; same contract and final buffer contents as the host sort.

    ``count_limit`` (internal, a 1-element ``int32`` device array): sort only the first ``n = min(count_limit[0],
    count)`` pairs, read on the GPU. When the pairs past ``n`` have keys not less than every key before it and are
    already in key order (e.g. a sentinel maximum key), the first half of the buffers then ends exactly as with the
    full sort (those pairs stay in place); the second half is written up to ``n`` only.
    """
    if keys.dtype not in _KEY_MODES:
        raise RuntimeError(
            f"Unsupported keys and values data types: {type_repr(keys.dtype)}, {type_repr(values.dtype)}"
        )
    if count >= 2**31:
        raise RuntimeError("radix_sort_pairs() on Metal supports fewer than 2**31 elements")
    key_size, mode = _KEY_MODES[keys.dtype]
    state = _state(keys.device)

    num_tiles = (count + _SORT_TILE - 1) // _SORT_TILE

    # host passes of up to 16 bits (warp/native/sort.cpp), each one or two 8-bit GPU passes; after the host passes
    # the first half holds the sorted pairs and the second half the pairs after host pass host_passes - 2 (all
    # passes when host_passes is odd: the host copies the second half into the first)
    host_passes = (end_bit - begin_bit + 15) // 16
    passes = []  # (shift, bits)
    m = -1
    for p in range(host_passes):
        shift = begin_bit + 16 * p
        bits = min(16, end_bit - shift)
        passes.append((shift, min(bits, 8)))
        if bits > 8:
            passes.append((shift + 8, bits - 8))
        if p == host_passes - 2:
            m = len(passes) - 1
    if host_passes % 2 == 1:
        m = len(passes) - 1

    hist_bytes = _align(256 * num_tiles * 4)
    misc_bytes = _align(256 * 4 + 8 + 48 * 4)
    temp_k_bytes = _align(count * key_size)
    temp_v_bytes = _align(count * value_size)
    base = state.scratch_ptr(hist_bytes + misc_bytes + temp_k_bytes + temp_v_bytes)
    tile_hist = base
    digit_total = base + hist_bytes
    diff = digit_total + 256 * 4
    plan = diff + 8
    temp_k = digit_total + misc_bytes
    temp_v = temp_k + temp_k_bytes

    args = _SortArgs()
    args.keys[:] = (keys.ptr, keys.ptr + count * key_size, temp_k)
    args.values[:] = (values.ptr, values.ptr + count * value_size, temp_v)
    args.tile_hist, args.digit_total, args.diff, args.plan = tile_hist, digit_total, diff, plan
    args.n, args.num_tiles, args.mode = count, num_tiles, mode
    args.num_passes, args.m = len(passes), m
    args.n_dev = count_limit.ptr if count_limit is not None else 0
    args.key_words, args.value_words = key_size // 4, value_size // 4
    for i, (shift, bits) in enumerate(passes):
        args.shifts[i], args.widths[i] = shift, bits

    # which key bits differ, then the routing of the passes that change the order
    state.launch("sort_diff_clear", 0, args, _SORT_PTR_OFFSETS, threads=2)
    grid = min(num_tiles, _SORT_GRID)
    state.launch(f"sort_diff_k{key_size}", grid, args, _SORT_PTR_OFFSETS)
    state.launch("sort_plan", 0, args, _SORT_PTR_OFFSETS, threads=1)

    copy_args = _SortArgs.from_buffer_copy(args)
    copy_args.shift, copy_args.bits = count * key_size // 4, count * value_size // 4
    copy_threads = min(max(copy_args.shift, copy_args.bits), _SORT_GRID * _TG)
    copy_args.pass_ = 0
    state.launch("sort_copy", 0, copy_args, _SORT_PTR_OFFSETS, threads=copy_threads)

    upsweep = f"sort_upsweep_k{key_size}"
    scatter = f"sort_scatter_k{key_size}_v{value_size}"
    for i, (shift, bits) in enumerate(passes):
        args.pass_, args.shift, args.bits = i, shift, bits
        state.launch(upsweep, grid, args, _SORT_PTR_OFFSETS)
        state.launch("sort_scan_tiles", 256, args, _SORT_PTR_OFFSETS)
        state.launch(scatter, grid, args, _SORT_PTR_OFFSETS)

    copy_args.pass_ = 1
    state.launch("sort_copy", 0, copy_args, _SORT_PTR_OFFSETS, threads=copy_threads)


def _scan_pass(state, suffix, in_ptr, out_ptr, n, in_stride, out_stride, inclusive, elem_size):
    num_tiles = (n + _SCAN_TILE - 1) // _SCAN_TILE
    tile_sums = state.scratch_ptr(_align(num_tiles * elem_size))
    args = _ScanArgs(in_ptr, out_ptr, tile_sums, 0, 0, 0, 0, in_stride, out_stride, n, num_tiles, int(inclusive), 0)
    state.launch(f"scan_reduce_{suffix}", num_tiles, args, _SCAN_PTR_OFFSETS)
    state.launch(f"scan_spine_{suffix}", 1, args, _SCAN_PTR_OFFSETS)
    state.launch(f"scan_down_{suffix}", num_tiles, args, _SCAN_PTR_OFFSETS)


def _extent(ptr, count, stride, elem_size):
    """Byte range [begin, end) of ``count`` elements of ``elem_size`` bytes, ``stride`` bytes apart, from ``ptr``."""
    if count <= 0:
        return ptr, ptr
    last = ptr + (count - 1) * stride
    return min(ptr, last), max(ptr, last) + elem_size


def scan_aliasing_ok(in_array, out_array, in_stride, out_stride, elem_size) -> bool:
    """Whether the GPU scan may run: separate arrays, or exactly in place (other overlaps use the host scan)."""
    if in_array.ptr == out_array.ptr and in_stride == out_stride:
        return True
    a = _extent(in_array.ptr, in_array.size, in_stride, elem_size)
    b = _extent(out_array.ptr, out_array.size, out_stride, elem_size)
    return a[1] <= b[0] or b[1] <= a[0]


def overlaps(values, count, outputs) -> bool:
    """Whether the first ``count`` elements of contiguous ``values`` share memory with any of the ``outputs``."""
    a = (values.ptr, values.ptr + count * values.strides[0])
    for out in outputs:
        b = (out.ptr, out.ptr + out.capacity)
        if a[0] < b[1] and b[0] < a[1]:
            return True
    return False


def supports_scan(scalar_type) -> bool:
    return scalar_type in (wp.int32, wp.int64)


def array_scan(in_array, out_array, scalar_type, type_length, in_stride, out_stride, inclusive) -> None:
    """GPU scan of an integer array (scalars, or vectors component-wise), exact like the host scan."""
    n = in_array.size
    if n >= 2**31:
        raise RuntimeError("array_scan() on Metal supports fewer than 2**31 elements")
    state = _state(in_array.device)
    elem_size = 4 if scalar_type == wp.int32 else 8
    suffix = "i4" if elem_size == 4 else "i8"
    for c in range(type_length):
        _scan_pass(
            state,
            suffix,
            in_array.ptr + c * elem_size,
            out_array.ptr + c * elem_size,
            n,
            in_stride,
            out_stride,
            inclusive,
            elem_size,
        )


def runlength_encode(values, run_values, run_lengths, run_count, value_count: int, count_limit=None) -> None:
    """GPU run-length encoding of ``value_count`` ``int32`` values; ``run_count`` receives the number of runs.

    ``count_limit`` (internal, a 1-element ``int32`` device array): encode only the first
    ``min(count_limit[0], value_count)`` values; the dispatches are sized on the GPU from that count.
    """
    if value_count >= 2**31:
        raise RuntimeError("runlength_encode() on Metal supports fewer than 2**31 elements")
    state = _state(values.device)
    n = value_count
    num_tiles = (n + _SCAN_TILE - 1) // _SCAN_TILE
    sums_bytes = _align(num_tiles * 4)
    base = state.scratch_ptr(sums_bytes + _align(n * 4))
    limit_ptr = count_limit.ptr if count_limit is not None else 0
    args = _ScanArgs(
        values.ptr,
        0,
        base,
        base + sums_bytes,
        run_values.ptr,
        run_lengths.ptr,
        run_count.ptr,
        4,
        4,
        n,
        num_tiles,
        1,
        0,
        limit_ptr,
    )
    if count_limit is None:  # (callers pass a bound only with warp.config.metal_bounded_launch)
        state.launch("rle_count", num_tiles, args, _SCAN_PTR_OFFSETS)
        state.launch("scan_spine_i4", 1, args, _SCAN_PTR_OFFSETS)
        state.launch("rle_scatter", num_tiles, args, _SCAN_PTR_OFFSETS)
        state.launch("rle_lengths", 0, args, _SCAN_PTR_OFFSETS, threads=n)
        return
    if n == 0:
        run_count.zero_()
        return
    # tile dispatches sized from the device count (one threadgroup per SCAN_TILE values); run_count is cleared
    # first, as no rle_scatter threadgroup runs when the bounded count is 0
    run_count[:1].zero_()
    state.launch_bounded("rle_count", num_tiles, args, _SCAN_PTR_OFFSETS, count_limit, _SCAN_TILE)
    state.launch("scan_spine_i4", 1, args, _SCAN_PTR_OFFSETS)
    state.launch_bounded("rle_scatter", num_tiles, args, _SCAN_PTR_OFFSETS, count_limit, _SCAN_TILE)
    state.launch_bounded("rle_lengths", n, args, _SCAN_PTR_OFFSETS, count_limit, 1, per_thread=True)


def _copy_operand(a):
    """(data pointer, byte strides, index pointers) of a Warp array or indexed array, or None if unsupported."""
    if isinstance(a, wp.array):
        return a.ptr, list(a.strides), [0] * 4
    if isinstance(a, wp.indexedarray):
        if not isinstance(a.data, wp.array):
            return None
        indices = [0] * 4
        for d, idx in enumerate(a.indices):
            if idx is not None:
                if idx.device != a.data.device or not idx.is_contiguous or idx.dtype != wp.int32:
                    return None
                indices[d] = idx.ptr
        return a.data.ptr, list(a.data.strides), indices
    return None


def array_copy(dest, src, elem_size: int) -> bool:
    """GPU copy between strided and / or indexed arrays of one Metal device, element by element (``elem_size``
    bytes each), with the indexing of CUDA's ``wp_array_copy_device``: ``dest[dst_index(i)] = src[src_index(i)]``
    over the common shape, where an indexed dimension maps ``i`` through its index array.

    Returns ``False`` (nothing launched) for operands it does not handle (Fabric arrays); the caller then copies on
    the host. Ordered with kernel launches and recordable into graph captures (no host synchronization).
    """
    d = _copy_operand(dest)
    s = _copy_operand(src)
    if d is None or s is None or dest.ndim != src.ndim or tuple(dest.shape) != tuple(src.shape):
        return False
    n = 1
    for extent in src.shape:
        n *= extent
    if n == 0:
        return True
    if n >= 2**32:
        return False
    args = _ArrayCopyArgs()
    args.dst, args.src = d[0], s[0]
    args.dst_indices[:] = d[2]
    args.src_indices[:] = s[2]
    ndim = src.ndim
    args.dst_strides[:] = d[1] + [0] * (4 - ndim)
    args.src_strides[:] = s[1] + [0] * (4 - ndim)
    args.shape[:] = list(src.shape) + [1] * (4 - ndim)
    args.ndim, args.elem_size, args.n = ndim, elem_size, n
    state = _state(dest.device)
    state.launch("array_copy", 0, args, _COPY_PTR_OFFSETS, threads=n)
    return True
