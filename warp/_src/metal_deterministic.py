# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic launches on Metal (``wp.DeterministicMode.RUN_TO_RUN`` and ``GPU_TO_GPU``).

The code generator lowers atomics as on CUDA (``warp/_src/deterministic.py``); the device helpers are in
``warp/native/metal_deterministic.h``. This module runs the launch:

- Accumulation atomics (return value unused) with floating-point values add, during the kernel, into an exact
  integer accumulator per destination component (nine 32-bit bins of int64 sums spanning float32's whole range).
  Integer addition commutes, so the accumulated bits do not depend on thread order. One pass after the kernel
  rounds each touched destination's sum to float once, adds it to the destination and clears the bins. No
  records, no sort, no capacity: the cost is a few 32-bit atomics per update and one pass over the touched
  destinations. Min/max atomics stay native (total-order compare-and-swap, see ``metal_atomics.h``).
- Consumed-return counter atomics (``slot = wp.atomic_add(counter, i, n)`` on ``int32``) run in two phases as on
  CUDA. Phase 0 runs the kernel with global stores suppressed and stores each call's (destination, count) at
  record index ``thread * records_per_thread + call``. The records are keyed by destination and stably radix
  sorted (the Metal sort of ``metal_primitives``), so each destination's calls stay in thread order; a segmented
  scan turns them into slots. Phase 1 reruns the kernel, each call returns its slot, and the counter totals are
  written back. Slots are thus handed out in thread order per destination, as on CUDA.

Every pass is dispatched with a shape that depends only on launch sizes and reads counts from device memory, so
launches need no host synchronization and can be captured. State that must start at zero (bins, flags, touched
markers, cursors) lives in a grow-only per-device pool that every launch leaves zeroed again; transient buffers
(records, sort keys, scan carries) share a second grow-only pool.

Results are independent of thread scheduling and of what else runs in the launch: an accumulated sum depends
only on the set of values added to that element, and a counter slot only on the calls to that counter element
by lower-numbered threads.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

import warp as wp
from warp._src.types import array_t, type_size_in_bytes

_TG = 256
_IPT = 8
_TILE = _TG * _IPT
_MAX_COUNTER_TARGETS = 8
_NBINS = 9
_FINALIZE_GRID = int(os.environ.get("WP_METAL_DET_FINALIZE_GRID", "16384"))
_BUILD_GRID = int(os.environ.get("WP_METAL_DET_BUILD_GRID", str(320 * _TG)))

_SCALAR_CODES = {"float16": 0, "float32": 1, "bfloat16": 7}

_MSL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define TG 256u
#define IPT 8u
#define TILE (TG * IPT)
#define MAXT 8
#define NBINS 9

struct wp_det_bounds_t { ulong unused; };

// ---------------------------------------------------------------------------------------------------------------
// accumulation targets: round each touched destination's exact sum to float once and add it

struct det_finalize_args_t {
    device uint* bins;            // [dest_size * C][NBINS][lo, hi]
    device uint* flags;           // [dest_size * C]
    device uint* touched;         // [dest_size]
    device int* touched_list;     // [dest_size]
    device atomic_int* counts;    // [0] touched destinations, [1] threads done
    device uchar* dest;           // element 0 of the destination
    uint value_size;              // bytes per destination element
    uint scalar_size;             // bytes per component
    uint components;
    int scalar_code;              // 0 half, 1 float, 7 bfloat16
};

inline float bins_to_float(device uint* w, uint flag)
{
    if ((flag & 1u) || (flag & 6u) == 6u)
        return NAN;
    if (flag & 2u)
        return INFINITY;
    if (flag & 4u)
        return -INFINITY;
    // value = sum_b bin_b * 2^(32 b - 149): propagate carries into 32-bit limbs, the last two from the carry out
    uint limb[NBINS + 2];
    long carry = 0;
    for (uint b = 0; b < NBINS; ++b) {
        const long v = long((ulong(w[2 * b + 1]) << 32) | ulong(w[2 * b]));
        const long s = v + carry;
        limb[b] = uint(ulong(s));
        carry = s >> 32;
    }
    limb[NBINS] = uint(ulong(carry));
    limb[NBINS + 1] = uint(ulong(carry) >> 32);
    const bool negative = carry < 0;
    if (negative) {
        uint borrow = 1u;
        for (uint b = 0; b < NBINS + 2; ++b) {
            const uint t = ~limb[b] + borrow;
            borrow = (borrow != 0u && t == 0u) ? 1u : 0u;
            limb[b] = t;
        }
    }
    int top = NBINS + 1;
    while (top >= 0 && limb[top] == 0u)
        --top;
    if (top < 0)
        return 0.0f;
    ulong window;
    int lsb;
    if (top == 0) {
        window = ulong(limb[0]);
        lsb = -149;
    } else {
        window = (ulong(limb[top]) << 32) | ulong(limb[top - 1]);
        lsb = 32 * (top - 1) - 149;
        bool sticky = false;
        for (int b = 0; b < top - 1; ++b)
            sticky = sticky || limb[b] != 0u;
        if (sticky)
            window |= 1ul;  // the window keeps at least 33 significant bits, so this only decides ties
    }
    const float r = ldexp(float(window), lsb);
    return negative ? -r : r;
}

inline float bf16_to_float(ushort u) { return as_type<float>(uint(u) << 16); }
inline ushort float_to_bf16(float f)
{
    uint u = as_type<uint>(f);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu))
        return ushort((u >> 16) | 0x40u);  // quiet NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return ushort(u >> 16);
}

kernel void det_scatter_finalize(
    constant wp_det_bounds_t& bounds [[buffer(0)]],
    constant det_finalize_args_t& a [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]])
{
    const int n = atomic_load_explicit(&a.counts[0], memory_order_relaxed);
    for (int i = int(tid); i < n; i += int(nthreads)) {
        const int d = a.touched_list[i];
        for (uint c = 0; c < a.components; ++c) {
            const ulong slot = ulong(d) * a.components + c;
            device uint* w = a.bins + slot * (2 * NBINS);
            const float r = bins_to_float(w, a.flags[slot]);
            for (uint k = 0; k < 2 * NBINS; ++k)
                w[k] = 0u;
            a.flags[slot] = 0u;
            device uchar* p = a.dest + ulong(d) * a.value_size + ulong(c) * a.scalar_size;
            if (a.scalar_code == 1) {
                device float* f = (device float*)p;
                *f = *f + r;
            } else if (a.scalar_code == 0) {
                device half* h = (device half*)p;
                *h = half(float(*h) + r);
            } else {
                device ushort* h = (device ushort*)p;
                *h = float_to_bf16(bf16_to_float(*h) + r);
            }
        }
        a.touched[d] = 0u;
    }
    // the last thread to finish (every thread has read the count by then) re-arms the target
    if (atomic_fetch_add_explicit(&a.counts[1], 1, memory_order_relaxed) == int(nthreads) - 1) {
        atomic_store_explicit(&a.counts[0], 0, memory_order_relaxed);
        atomic_store_explicit(&a.counts[1], 0, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// counter targets: sort the phase-0 records by destination and scan them into slots

struct det_counter_target_t {
    device int* dest_of;       // [dim * records_per_thread]
    device int* values;
    device int* prefixes;
    device int* cursors;       // [2 * dim]
    device int* counter;       // element 0 of the counter array
    int records_per_thread;
    int dim;
    int target_size;
    int key_base;
    int record_base;
    int cursor_base;
};

struct det_counter_args_t {
    det_counter_target_t t[MAXT];
    device uint* keys;         // sort keys: key_base + destination, or num_keys for an empty record
    device uint* payload;      // record index
    device int* carry_k;       // [num_tiles] scan carry: key of the run
    device int* carry_s;       // [num_tiles] sum of the run so far
    device int* carry_u;       // [num_tiles] whole tile is one run
    device int* totals;        // [num_keys]
    device uint* touched;      // [num_keys]
    int num_targets;
    uint n;                    // records
    uint num_keys;
    uint num_tiles;
    uint num_cursors;
    uint pad;
};

inline int target_of_record(constant det_counter_args_t& a, uint r)
{
    int t = 0;
    while (t + 1 < a.num_targets && int(r) >= a.t[t + 1].record_base)
        ++t;
    return t;
}

kernel void det_counter_build(
    constant wp_det_bounds_t& bounds [[buffer(0)]],
    constant det_counter_args_t& a [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]])
{
    for (uint r = tid; r < a.n; r += nthreads) {
        const int t = target_of_record(a, r);
        const uint local = r - uint(a.t[t].record_base);
        const uint rpt = uint(a.t[t].records_per_thread);
        const uint thread_index = local / rpt;
        const uint call = local - thread_index * rpt;
        uint key = a.num_keys;
        if (int(call) < a.t[t].cursors[thread_index]) {
            const int d = a.t[t].dest_of[local];
            if (d >= 0 && d < a.t[t].target_size)
                key = uint(a.t[t].key_base + d);
            else
                a.t[t].prefixes[local] = 0;
        }
        a.keys[r] = key;
        a.payload[r] = r;
    }
}

struct segp {
    int k;  // key of the last element (-1: empty)
    int s;  // sum over the trailing run of that key
    int u;  // all elements share the key
};

inline segp seg_identity() { return segp{ -1, 0, 1 }; }

inline segp seg_combine(segp a, segp b)
{
    if (a.k < 0)
        return b;
    if (b.k < 0)
        return a;
    segp r;
    r.k = b.k;
    r.u = (a.u != 0 && b.u != 0 && a.k == b.k) ? 1 : 0;
    r.s = (b.u != 0 && b.k == a.k) ? a.s + b.s : b.s;
    return r;
}

inline int record_value(constant det_counter_args_t& a, uint r)
{
    const int t = target_of_record(a, r);
    return a.t[t].values[r - uint(a.t[t].record_base)];
}

inline segp seg_element(constant det_counter_args_t& a, uint i)
{
    const uint key = a.keys[i];
    const int v = key < a.num_keys ? record_value(a, a.payload[i]) : 0;
    return segp{ int(key), v, 1 };
}

// Inclusive scan of one pair per thread over the threadgroup (Hillis-Steele); returns the thread's exclusive prefix
// and leaves the total in sm[TG - 1].
inline segp tg_seg_scan(segp v, threadgroup segp* sm, uint lid)
{
    sm[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1; off < TG; off <<= 1) {
        const segp mine = sm[lid];
        const segp prev = lid >= off ? sm[lid - off] : seg_identity();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sm[lid] = seg_combine(prev, mine);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return lid > 0 ? sm[lid - 1] : seg_identity();
}

kernel void det_seg_reduce(
    constant wp_det_bounds_t& bounds [[buffer(0)]],
    constant det_counter_args_t& a [[buffer(1)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup segp sm[TG];
    segp acc = seg_identity();
    for (uint j = 0; j < IPT; ++j) {
        const uint i = tile * TILE + lid * IPT + j;
        if (i < a.n)
            acc = seg_combine(acc, seg_element(a, i));
    }
    tg_seg_scan(acc, sm, lid);
    if (lid == 0) {
        const segp total = sm[TG - 1];
        a.carry_k[tile] = total.k;
        a.carry_s[tile] = total.s;
        a.carry_u[tile] = total.u;
    }
}

// One threadgroup turns the tile totals into exclusive carries, in place, 256 tiles at a time.
kernel void det_seg_spine(
    constant wp_det_bounds_t& bounds [[buffer(0)]],
    constant det_counter_args_t& a [[buffer(1)]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup segp sm[TG];
    segp carry = seg_identity();
    for (uint start = 0; start < a.num_tiles; start += TG) {
        const uint t = start + lid;
        const segp total = t < a.num_tiles ? segp{ a.carry_k[t], a.carry_s[t], a.carry_u[t] } : seg_identity();
        const segp excl = seg_combine(carry, tg_seg_scan(total, sm, lid));
        const segp chunk = sm[TG - 1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < a.num_tiles) {
            a.carry_k[t] = excl.k;
            a.carry_s[t] = excl.s;
            a.carry_u[t] = excl.u;
        }
        carry = seg_combine(carry, chunk);
    }
}

kernel void det_seg_down(
    constant wp_det_bounds_t& bounds [[buffer(0)]],
    constant det_counter_args_t& a [[buffer(1)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup segp sm[TG];
    segp acc = seg_identity();
    for (uint j = 0; j < IPT; ++j) {
        const uint i = tile * TILE + lid * IPT + j;
        if (i < a.n)
            acc = seg_combine(acc, seg_element(a, i));
    }
    const segp excl = tg_seg_scan(acc, sm, lid);
    segp run = seg_combine(segp{ a.carry_k[tile], a.carry_s[tile], a.carry_u[tile] }, excl);
    for (uint j = 0; j < IPT; ++j) {
        const uint i = tile * TILE + lid * IPT + j;
        if (i >= a.n)
            break;
        const segp e = seg_element(a, i);
        const uint key = uint(e.k);
        if (key < a.num_keys) {
            const uint r = a.payload[i];
            const int t = target_of_record(a, r);
            const uint local = r - uint(a.t[t].record_base);
            const int d = int(key) - a.t[t].key_base;
            const int before = run.k == e.k ? run.s : 0;
            const int base = a.t[t].counter[d];
            a.t[t].prefixes[local] = base + before;
            if (i + 1 == a.n || a.keys[i + 1] != key) {
                a.totals[key] = base + before + e.s;
                a.touched[key] = 1u;
            }
        }
        run = seg_combine(run, e);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// GPU_TO_GPU accumulation targets: records sorted by destination, summed in thread order

struct det_records_args_t {
    device int* rec_dest;      // [n]
    device uchar* rec_vals;    // [n] values
    device int* rec_cursors;   // [dim]
    device uint* keys;
    device uint* payload;
    device uchar* dest;        // element 0 of the destination
    uint n;
    int rec_per_thread;
    int dest_size;
    uint value_size;
    uint scalar_size;
    uint components;
    int scalar_code;
    uint dim;
};

kernel void det_records_build(
    constant wp_det_bounds_t& bounds [[buffer(0)]],
    constant det_records_args_t& a [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]])
{
    for (uint r = tid; r < a.n; r += nthreads) {
        const uint thread_index = r / uint(a.rec_per_thread);
        const uint call = r - thread_index * uint(a.rec_per_thread);
        uint key = uint(a.dest_size);
        if (int(call) < a.rec_cursors[thread_index])
            key = uint(a.rec_dest[r]);
        a.keys[r] = key;
        a.payload[r] = r;
    }
}

inline float load_component(constant det_records_args_t& a, device const uchar* p)
{
    if (a.scalar_code == 1)
        return *(device const float*)p;
    if (a.scalar_code == 0)
        return float(*(device const half*)p);
    return bf16_to_float(*(device const ushort*)p);
}

// rounds to the destination's scalar type, as each step of the CUDA reduction does in that type
inline float round_component(constant det_records_args_t& a, float x)
{
    if (a.scalar_code == 1)
        return x;
    if (a.scalar_code == 0)
        return float(half(x));
    return bf16_to_float(float_to_bf16(x));
}

inline void store_component(constant det_records_args_t& a, device uchar* p, float x)
{
    if (a.scalar_code == 1)
        *(device float*)p = x;
    else if (a.scalar_code == 0)
        *(device half*)p = half(x);
    else
        *(device ushort*)p = float_to_bf16(x);
}

// The first record of each destination sums the destination's records in sorted (thread) order and adds the sum.
kernel void det_records_reduce(
    constant wp_det_bounds_t& bounds [[buffer(0)]],
    constant det_records_args_t& a [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]])
{
    for (uint i = tid; i < a.n; i += nthreads) {
        const uint key = a.keys[i];
        if (key >= uint(a.dest_size) || (i > 0 && a.keys[i - 1] == key))
            continue;
        device uchar* d = a.dest + ulong(key) * a.value_size;
        for (uint c = 0; c < a.components; ++c) {
            const ulong offset = ulong(c) * a.scalar_size;
            float acc = load_component(a, a.rec_vals + ulong(a.payload[i]) * a.value_size + offset);
            for (uint j = i + 1; j < a.n && a.keys[j] == key; ++j)
                acc = round_component(a, acc + load_component(a, a.rec_vals + ulong(a.payload[j]) * a.value_size + offset));
            store_component(a, d + offset, round_component(a, load_component(a, d + offset) + acc));
        }
    }
}

kernel void det_records_reset(
    constant wp_det_bounds_t& bounds [[buffer(0)]],
    constant det_records_args_t& a [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]])
{
    for (uint t = tid; t < a.dim; t += nthreads)
        a.rec_cursors[t] = 0;
}

// After phase 1: publish the totals of the counters that were used, and zero every thread's call cursors.
kernel void det_counter_writeback(
    constant wp_det_bounds_t& bounds [[buffer(0)]],
    constant det_counter_args_t& a [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]])
{
    const uint n = max(a.num_keys, a.num_cursors);
    for (uint k = tid; k < n; k += nthreads) {
        if (k < a.num_keys && a.touched[k] != 0u) {
            int t = 0;
            while (t + 1 < a.num_targets && int(k) >= a.t[t + 1].key_base)
                ++t;
            a.t[t].counter[int(k) - a.t[t].key_base] = a.totals[k];
            a.touched[k] = 0u;
        }
        if (k < a.num_cursors) {
            int t = 0;
            while (t + 1 < a.num_targets && int(k) >= a.t[t + 1].cursor_base)
                ++t;
            const int j = int(k) - a.t[t].cursor_base;
            a.t[t].cursors[j] = 0;
            a.t[t].cursors[a.t[t].dim + j] = 0;
        }
    }
}
"""


# ---------------------------------------------------------------------------------------------------------------
# ctypes mirrors of the device structs (metal_deterministic.h and _MSL_SOURCE)


class DetScatterBuf(ctypes.Structure):
    _fields_ = [
        ("bins", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("touched", ctypes.c_uint64),
        ("touched_list", ctypes.c_uint64),
        ("touched_count", ctypes.c_uint64),
        ("rec_dest", ctypes.c_uint64),
        ("rec_vals", ctypes.c_uint64),
        ("rec_cursors", ctypes.c_uint64),
        ("dest_size", ctypes.c_int32),
        ("rec_per_thread", ctypes.c_int32),
    ]
    _wp_pointer_fields = (
        "bins", "flags", "touched", "touched_list", "touched_count", "rec_dest", "rec_vals", "rec_cursors"
    )


class DetCounterBuf(ctypes.Structure):
    _fields_ = [
        ("dest_of", ctypes.c_uint64),
        ("values", ctypes.c_uint64),
        ("prefixes", ctypes.c_uint64),
        ("cursors", ctypes.c_uint64),
        ("capacity", ctypes.c_int32),
        ("records_per_thread", ctypes.c_int32),
        ("dim", ctypes.c_int32),
        ("target_size", ctypes.c_int32),
        ("target", array_t),
    ]
    _wp_pointer_fields = ("dest_of", "values", "prefixes", "cursors")


class _FinalizeArgs(ctypes.Structure):
    _fields_ = [
        ("bins", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("touched", ctypes.c_uint64),
        ("touched_list", ctypes.c_uint64),
        ("counts", ctypes.c_uint64),
        ("dest", ctypes.c_uint64),
        ("value_size", ctypes.c_uint32),
        ("scalar_size", ctypes.c_uint32),
        ("components", ctypes.c_uint32),
        ("scalar_code", ctypes.c_int32),
    ]


class _RecordsArgs(ctypes.Structure):
    _fields_ = [
        ("rec_dest", ctypes.c_uint64),
        ("rec_vals", ctypes.c_uint64),
        ("rec_cursors", ctypes.c_uint64),
        ("keys", ctypes.c_uint64),
        ("payload", ctypes.c_uint64),
        ("dest", ctypes.c_uint64),
        ("n", ctypes.c_uint32),
        ("rec_per_thread", ctypes.c_int32),
        ("dest_size", ctypes.c_int32),
        ("value_size", ctypes.c_uint32),
        ("scalar_size", ctypes.c_uint32),
        ("components", ctypes.c_uint32),
        ("scalar_code", ctypes.c_int32),
        ("dim", ctypes.c_uint32),
    ]


class _CounterTarget(ctypes.Structure):
    _fields_ = [
        ("dest_of", ctypes.c_uint64),
        ("values", ctypes.c_uint64),
        ("prefixes", ctypes.c_uint64),
        ("cursors", ctypes.c_uint64),
        ("counter", ctypes.c_uint64),
        ("records_per_thread", ctypes.c_int32),
        ("dim", ctypes.c_int32),
        ("target_size", ctypes.c_int32),
        ("key_base", ctypes.c_int32),
        ("record_base", ctypes.c_int32),
        ("cursor_base", ctypes.c_int32),
    ]


class _CounterArgs(ctypes.Structure):
    _fields_ = [
        ("t", _CounterTarget * _MAX_COUNTER_TARGETS),
        ("keys", ctypes.c_uint64),
        ("payload", ctypes.c_uint64),
        ("carry_k", ctypes.c_uint64),
        ("carry_s", ctypes.c_uint64),
        ("carry_u", ctypes.c_uint64),
        ("totals", ctypes.c_uint64),
        ("touched", ctypes.c_uint64),
        ("num_targets", ctypes.c_int32),
        ("n", ctypes.c_uint32),
        ("num_keys", ctypes.c_uint32),
        ("num_tiles", ctypes.c_uint32),
        ("num_cursors", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


def _struct_pointer_offsets(struct_type, base=0):
    offsets = []
    for name, field_type in struct_type._fields_:
        offset = base + getattr(struct_type, name).offset
        if field_type is ctypes.c_uint64 and name in getattr(struct_type, "_wp_pointer_fields", _ALL_POINTERS):
            offsets.append(offset)
        elif isinstance(field_type, type) and issubclass(field_type, ctypes.Array):
            elem = field_type._type_
            if issubclass(elem, ctypes.Structure):
                for i in range(field_type._length_):
                    offsets.extend(_struct_pointer_offsets(elem, offset + i * ctypes.sizeof(elem)))
        elif isinstance(field_type, type) and issubclass(field_type, ctypes.Structure):
            offsets.extend(_struct_pointer_offsets(field_type, offset))
    return offsets


class _AllPointers:
    def __contains__(self, name):
        return True


_ALL_POINTERS = _AllPointers()
for _t in (_FinalizeArgs, _CounterTarget, _RecordsArgs):
    _t._wp_pointer_fields = tuple(name for name, ftype in _t._fields_ if ftype is ctypes.c_uint64)
_CounterArgs._wp_pointer_fields = ("keys", "payload", "carry_k", "carry_s", "carry_u", "totals", "touched")


def _offsets_array(struct_type):
    offsets = _struct_pointer_offsets(struct_type)
    return (ctypes.c_size_t * len(offsets))(*offsets)


_FINALIZE_OFFSETS = _offsets_array(_FinalizeArgs)
_COUNTER_OFFSETS = _offsets_array(_CounterArgs)
_RECORDS_OFFSETS = _offsets_array(_RecordsArgs)
_BOUNDS = ctypes.c_uint64(0)


def det_args_type(det_meta) -> type:
    """ctypes mirror of the ``wp_det_args_<kernel>`` struct that codegen appends to a kernel's arguments."""
    cached = getattr(det_meta, "_metal_det_args_type", None)
    if cached is not None:
        return cached
    fields = [("phase", ctypes.c_int32), ("debug", ctypes.c_int32), ("overflow", ctypes.c_uint64)]
    fields += [(t.helper_name, DetCounterBuf) for t in det_meta.counter_targets]
    fields += [(t.helper_name, DetScatterBuf) for t in det_meta.scatter_targets]
    struct_type = type("wp_det_args", (ctypes.Structure,), {"_fields_": fields, "_wp_pointer_fields": ("overflow",)})
    det_meta._metal_det_args_type = struct_type
    return struct_type


def det_args_pointer_offsets(struct_type, base: int) -> list[int]:
    """Byte offsets (from ``base``) of the device pointers inside a ``det_args_type`` struct."""
    return _struct_pointer_offsets(struct_type, base)


def metal_struct_source(name: str, det_meta, counter_ctype, scatter_ctype) -> str:
    """MSL definition of ``wp_det_args_<name>`` (field order and types match ``det_args_type``)."""
    lines = [f"struct wp_det_args_{name} {{", "    int phase;", "    int debug;", "    device metal::atomic_int* overflow;"]
    lines += [f"    {counter_ctype(t)} {t.helper_name};" for t in det_meta.counter_targets]
    lines += [f"    {scatter_ctype(t)} {t.helper_name};" for t in det_meta.scatter_targets]
    lines.append("};")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# per-device state


class _Pool:
    """Grow-only device buffer; outgrown buffers stay alive (plans and captured graphs may reference them)."""

    def __init__(self, device, zeroed: bool):
        self.device = device
        self.zeroed = zeroed
        self.buffers = []

    def get(self, nbytes: int):
        if not self.buffers or self.buffers[-1].size < nbytes:
            size = max(nbytes, 2 * self.buffers[-1].size if self.buffers else 0, 1 << 16)
            self.buffers.append(_host_zeroed(size, wp.uint8, self.device))
        return self.buffers[-1]


def _host_zeroed(n, dtype, device):
    """A new device array zeroed from the host (Metal memory is shared), so this works during graph capture."""
    arr = wp.empty(n, dtype=dtype, device=device)
    ctypes.memset(arr.ptr, 0, n * type_size_in_bytes(dtype))
    return arr


class _DeviceState:
    def __init__(self, device):
        from warp._src.context import runtime  # noqa: PLC0415

        self.device = device
        self.core = runtime.core
        self.library = self.core.wp_metal_load_library(device.metal_ordinal, _MSL_SOURCE.encode("utf-8"))
        if not self.library:
            raise RuntimeError(f"Failed to compile the Metal deterministic-mode kernels:\n{runtime.get_error_string()}")
        self.kernels = {}
        # state that every launch returns to zero (bins, flags, touched markers, cursors, done counters)
        self.zero_pool = _Pool(device, zeroed=True)
        # transient buffers (records, sort keys and scratch, scan carries, totals)
        self.scratch_pool = _Pool(device, zeroed=False)
        # sticky overflow flag (counter records beyond records_per_thread)
        self.overflow = _host_zeroed(1, wp.int32, device)
        self.overflow_reported = False

    def kernel(self, name):
        handle = self.kernels.get(name)
        if handle is None:
            handle = self.core.wp_metal_get_kernel(self.library, name.encode("utf-8"))
            if not handle:
                from warp._src.context import runtime  # noqa: PLC0415

                raise RuntimeError(f"Metal deterministic kernel '{name}' is unavailable: {runtime.get_error_string()}")
            self.kernels[name] = handle
        return handle

    def launch(self, name, threads, args, offsets, group=_TG):
        from warp._src.context import runtime  # noqa: PLC0415

        if threads <= 0:
            return
        if self.core.wp_metal_launch_kernel(
            self.device.metal_ordinal,
            self.kernel(name),
            threads,
            group,
            ctypes.byref(_BOUNDS),
            ctypes.sizeof(_BOUNDS),
            ctypes.byref(args),
            ctypes.sizeof(args),
            offsets,
            len(offsets),
            0,
        ):
            raise RuntimeError(f"Failed to launch Metal deterministic kernel '{name}': {runtime.get_error_string()}")


def _state(device) -> _DeviceState:
    state = device.__dict__.get("_metal_deterministic")
    if state is None:
        state = _DeviceState(device)
        device.__dict__["_metal_deterministic"] = state
    return state


def _align(x, a=256):
    return (x + a - 1) // a * a


class _Layout:
    """Byte layout of one plan's regions inside a pool."""

    def __init__(self):
        self.size = 0

    def take(self, nbytes):
        offset = self.size
        self.size = _align(offset + max(int(nbytes), 1))
        return offset


# ---------------------------------------------------------------------------------------------------------------
# launch plans


def _resolve(arg, attr_path):
    for attr in attr_path:
        if arg is None:
            return None
        arg = getattr(arg, attr)
    return arg


class _Sort:
    """Stable LSD radix sort (8-bit digits) of uint32 keys with uint32 payloads, on metal_primitives' kernels,
    over a fixed number of elements; keys run 0..max_key."""

    def __init__(self, scratch, n, max_key):
        self.n = n
        self.key_bits = max(1, int(max_key).bit_length())
        self.num_passes = (self.key_bits + 7) // 8
        self.num_tiles = (n + _TILE - 1) // _TILE
        self.regions = {
            "keys0": scratch.take(n * 4),
            "keys1": scratch.take(n * 4),
            "pay0": scratch.take(n * 4),
            "pay1": scratch.take(n * 4),
            "tile_hist": scratch.take(256 * self.num_tiles * 4),
            "digit_total": scratch.take(256 * 4),
            "diff": scratch.take(8),
        }

    def bind(self, state, sb):
        from warp._src import metal_primitives  # noqa: PLC0415

        r = self.regions
        # ping-pong between buffers 0 and 2 of the sort's buffer table; the result ends in buffer `final`
        self.plan_buf = _host_zeroed(48, wp.uint32, state.device)
        plan = [0] * 48
        src = 0
        for p in range(self.num_passes):
            dst = 2 if src == 0 else 0
            plan[4 * p : 4 * p + 4] = [1, src, dst, 3]
            src = dst
        ctypes.memmove(self.plan_buf.ptr, (ctypes.c_uint32 * 48)(*plan), 48 * 4)
        keys = (sb + r["keys0"], sb + r["keys0"], sb + r["keys1"])
        pays = (sb + r["pay0"], sb + r["pay0"], sb + r["pay1"])
        self.input = (keys[0], pays[0])
        self.output = (keys[src], pays[src])
        args = metal_primitives._SortArgs()
        args.keys[:] = keys
        args.values[:] = pays
        args.tile_hist = sb + r["tile_hist"]
        args.digit_total = sb + r["digit_total"]
        args.diff = sb + r["diff"]
        args.plan = self.plan_buf.ptr
        args.n = self.n
        args.num_tiles = self.num_tiles
        args.mode = 0
        args.num_passes = self.num_passes
        args.m = -1
        self.grid = min(self.num_tiles, metal_primitives._SORT_GRID)
        self.passes = []
        for p in range(self.num_passes):
            pa = metal_primitives._SortArgs.from_buffer_copy(args)
            pa.pass_, pa.shift, pa.bits = p, 8 * p, min(8, self.key_bits - 8 * p)
            self.passes.append(pa)

    def run(self, state):
        from warp._src import metal_primitives  # noqa: PLC0415

        prim = metal_primitives._state(state.device)
        offsets = metal_primitives._SORT_PTR_OFFSETS
        for pa in self.passes:
            prim.launch("sort_upsweep_k4", self.grid, pa, offsets)
            prim.launch("sort_scan_tiles", 256, pa, offsets)
            prim.launch("sort_scatter_k4_v4", self.grid, pa, offsets)


class _Plan:
    """Buffers and pass arguments of one (kernel, launch size, target arrays) combination."""

    def __init__(self, state, kernel, det_meta, dim, scatter, counters, debug, records):
        self.state = state
        self.det_type = det_args_type(det_meta)
        self.det = self.det_type()
        self.det.debug = debug
        self.det.overflow = state.overflow.ptr
        self.finalize = []  # (args, threads): exact accumulation targets
        self.records = []  # (args, sort, threads, reset_threads): GPU_TO_GPU accumulation targets
        self.counter_args = None
        self.counter_sort = None
        self.dim = dim

        zero = _Layout()
        scratch = _Layout()

        # accumulation targets
        scatter_regions = []
        for target, dest in scatter:
            if dest is None:
                continue
            dest_size = _dest_size_elements(dest)
            comps = getattr(target.value_dtype, "_length_", 1)
            scalar_name = getattr(target.scalar_dtype, "__name__", str(target.scalar_dtype))
            if scalar_name not in _SCALAR_CODES:
                raise RuntimeError(
                    f"Deterministic mode on Metal does not support accumulating {scalar_name} values "
                    f"(target '{target.target_label}')"
                )
            if any(s < 0 for s in dest.strides):
                raise RuntimeError(
                    "Deterministic mode on Metal does not support negative-stride destinations for "
                    f"accumulation or composite slot reductions (target '{target.target_label}')"
                )
            if records:
                rpt = max(det_meta.scatter_records_per_thread.get(target, 1), det_meta.max_records, 1)
                n = dim * rpt
                if n >= 2**31:
                    raise RuntimeError(
                        f"Deterministic scatter records exceed 2^31 in kernel '{kernel.key}' "
                        f"({dim} threads x {rpt} records per thread)"
                    )
                regions = {
                    "rec_dest": scratch.take(n * 4),
                    "rec_vals": scratch.take(n * type_size_in_bytes(target.value_dtype)),
                    "rec_cursors": zero.take(dim * 4),
                    "sort": _Sort(scratch, n, dest_size),
                    "rpt": rpt,
                }
            else:
                regions = {
                    "bins": zero.take(dest_size * comps * _NBINS * 8),
                    "flags": zero.take(dest_size * comps * 4),
                    "touched": zero.take(dest_size * 4),
                    "counts": zero.take(8),
                    "list": scratch.take(dest_size * 4),
                }
            scatter_regions.append((target, dest, dest_size, comps, scalar_name, regions))

        # counter targets
        counter_regions = []
        record_base = key_base = cursor_base = 0
        for target, arr, rpt in counters:
            if arr is None:
                continue
            target_size = _dest_size_elements(arr)
            capacity = dim * rpt
            if capacity >= 2**31 or record_base + capacity >= 2**31:
                raise RuntimeError(
                    f"Deterministic counter records exceed 2^31 in kernel '{kernel.key}' "
                    f"({dim} threads x {rpt} records per thread)"
                )
            regions = {
                "dest_of": scratch.take(capacity * 4),
                "values": scratch.take(capacity * 4),
                "prefixes": scratch.take(capacity * 4),
                "cursors": zero.take(2 * dim * 4),
            }
            counter_regions.append((target, arr, rpt, target_size, capacity, record_base, key_base, cursor_base, regions))
            record_base += capacity
            key_base += target_size
            cursor_base += dim
        if len(counter_regions) > _MAX_COUNTER_TARGETS:
            raise RuntimeError(
                f"Deterministic mode on Metal supports at most {_MAX_COUNTER_TARGETS} consumed-return counter "
                f"targets per kernel ('{kernel.key}' has {len(counter_regions)})"
            )
        if counter_regions:
            n = record_base
            num_keys = key_base
            num_tiles = (n + _TILE - 1) // _TILE
            self.counter_sort = _Sort(scratch, n, num_keys)  # keys run 0..num_keys (num_keys: empty record)
            scan_regions = {
                "carry_k": scratch.take(num_tiles * 4),
                "carry_s": scratch.take(num_tiles * 4),
                "carry_u": scratch.take(num_tiles * 4),
                "totals": scratch.take(num_keys * 4),
                "touched": zero.take(num_keys * 4),
            }

        self.zero_buf = state.zero_pool.get(zero.size)
        self.scratch_buf = state.scratch_pool.get(scratch.size)
        zb = self.zero_buf.ptr
        sb = self.scratch_buf.ptr
        self.sorts = []

        for target, dest, dest_size, comps, scalar_name, r in scatter_regions:
            buf = getattr(self.det, target.helper_name)
            buf.dest_size = dest_size
            if records:
                buf.rec_dest = sb + r["rec_dest"]
                buf.rec_vals = sb + r["rec_vals"]
                buf.rec_cursors = zb + r["rec_cursors"]
                buf.rec_per_thread = r["rpt"]
                sort = r["sort"]
                sort.bind(state, sb)
                self.sorts.append(sort)
                ra = _RecordsArgs()
                ra.rec_dest, ra.rec_vals, ra.rec_cursors = buf.rec_dest, buf.rec_vals, buf.rec_cursors
                ra.dest = dest.ptr
                ra.n, ra.rec_per_thread, ra.dest_size, ra.dim = sort.n, r["rpt"], dest_size, dim
                ra.value_size = type_size_in_bytes(target.value_dtype)
                ra.scalar_size = type_size_in_bytes(target.scalar_dtype)
                ra.components = comps
                ra.scalar_code = _SCALAR_CODES[scalar_name]
                build = _RecordsArgs.from_buffer_copy(ra)
                build.keys, build.payload = sort.input
                reduce = _RecordsArgs.from_buffer_copy(ra)
                reduce.keys, reduce.payload = sort.output
                self.records.append(
                    (build, reduce, sort, min(_align(sort.n, _TG), _BUILD_GRID), min(_align(dim, _TG), _BUILD_GRID))
                )
                continue
            buf.bins = zb + r["bins"]
            buf.flags = zb + r["flags"]
            buf.touched = zb + r["touched"]
            buf.touched_list = sb + r["list"]
            buf.touched_count = zb + r["counts"]
            fin = _FinalizeArgs()
            fin.bins, fin.flags, fin.touched = buf.bins, buf.flags, buf.touched
            fin.touched_list, fin.counts, fin.dest = buf.touched_list, buf.touched_count, dest.ptr
            fin.value_size = type_size_in_bytes(target.value_dtype)
            fin.scalar_size = type_size_in_bytes(target.scalar_dtype)
            fin.components = comps
            fin.scalar_code = _SCALAR_CODES[scalar_name]
            threads = min(_align(dest_size, _TG), _FINALIZE_GRID)
            self.finalize.append((fin, threads))

        if counter_regions:
            args = _CounterArgs()
            for i, (target, arr, rpt, target_size, capacity, rbase, kbase, cbase, r) in enumerate(counter_regions):
                buf = getattr(self.det, target.helper_name)
                buf.dest_of = sb + r["dest_of"]
                buf.values = sb + r["values"]
                buf.prefixes = sb + r["prefixes"]
                buf.cursors = zb + r["cursors"]
                buf.capacity = capacity
                buf.records_per_thread = rpt
                buf.dim = dim
                buf.target_size = target_size
                buf.target = arr.__ctype__()
                t = args.t[i]
                t.dest_of, t.values, t.prefixes, t.cursors = buf.dest_of, buf.values, buf.prefixes, buf.cursors
                t.counter = arr.ptr
                t.records_per_thread, t.dim, t.target_size = rpt, dim, target_size
                t.key_base, t.record_base, t.cursor_base = kbase, rbase, cbase
            args.num_targets = len(counter_regions)
            args.n, args.num_keys, args.num_tiles, args.num_cursors = n, num_keys, num_tiles, cursor_base
            args.carry_k = sb + scan_regions["carry_k"]
            args.carry_s = sb + scan_regions["carry_s"]
            args.carry_u = sb + scan_regions["carry_u"]
            args.totals = sb + scan_regions["totals"]
            args.touched = zb + scan_regions["touched"]
            self.counter_sort.bind(state, sb)
            self.sorts.append(self.counter_sort)
            self.build_args = _CounterArgs.from_buffer_copy(args)
            self.build_args.keys, self.build_args.payload = self.counter_sort.input
            self.counter_args = _CounterArgs.from_buffer_copy(args)
            self.counter_args.keys, self.counter_args.payload = self.counter_sort.output
            self.num_tiles = num_tiles
            self.build_threads = min(_align(n, _TG), _BUILD_GRID)
            self.writeback_threads = min(_align(max(num_keys, cursor_base), _TG), _BUILD_GRID)

    def run_counter_passes(self):
        st = self.state
        st.launch("det_counter_build", self.build_threads, self.build_args, _COUNTER_OFFSETS)
        self.counter_sort.run(st)
        st.launch("det_seg_reduce", self.num_tiles * _TG, self.counter_args, _COUNTER_OFFSETS)
        st.launch("det_seg_spine", _TG, self.counter_args, _COUNTER_OFFSETS)
        st.launch("det_seg_down", self.num_tiles * _TG, self.counter_args, _COUNTER_OFFSETS)

    def run_writeback(self):
        self.state.launch("det_counter_writeback", self.writeback_threads, self.counter_args, _COUNTER_OFFSETS)

    def run_finalize(self):
        st = self.state
        for fin, threads in self.finalize:
            st.launch("det_scatter_finalize", threads, fin, _FINALIZE_OFFSETS)
        for build, reduce, sort, threads, reset_threads in self.records:
            st.launch("det_records_build", threads, build, _RECORDS_OFFSETS)
            sort.run(st)
            st.launch("det_records_reduce", threads, reduce, _RECORDS_OFFSETS)
            st.launch("det_records_reset", reset_threads, build, _RECORDS_OFFSETS)


def _dest_size_elements(arr) -> int:
    from warp._src.deterministic import _det_dest_size_elements  # noqa: PLC0415

    return _det_dest_size_elements(arr)


def _target_arg_index(kernel, label):
    for j, arg in enumerate(kernel.adj.args):
        if arg.label == label:
            return j
    return None


def _resolve_target(kernel, target, fwd_args, adj_args):
    j = _target_arg_index(kernel, target.array_var_label)
    if j is None:
        return None
    if getattr(target, "adjoint", False):
        arr = _resolve(adj_args[j], target.attr_path) if j < len(adj_args) else None
        if arr is None:
            primal = _resolve(fwd_args[j], target.attr_path)
            arr = getattr(primal, "grad", None)
        return arr
    return _resolve(fwd_args[j], target.attr_path)


def _usable(arr):
    return arr is not None and getattr(arr, "ptr", None) and getattr(arr, "size", 0) > 0


def launch(kernel, hooks, params, bounds, device, det_meta, fwd_args, adj_args, adjoint, block_dim) -> None:
    """Run one deterministic Metal launch (see the module docstring)."""
    from warp._src.context import invoke_metal  # noqa: PLC0415
    from warp._src.deterministic import REDUCE_OP_ADD  # noqa: PLC0415

    from warp._src.deterministic import ArrayStoreTarget, _det_array_byte_bounds  # noqa: PLC0415

    adj_args = list(adj_args or [])
    dim = bounds.size

    # Backward reductions that a store in the same kernel can overwrite must happen in program order: allowed only
    # for single-thread launches, which then use native atomics (as on CUDA, see deterministic.launch_deterministic)
    check_component_order = adjoint and det_meta.has_component_updates
    ordered_backward_targets = list(det_meta.ordered_backward_targets) if check_component_order else []
    if check_component_order:
        store_bounds = []
        for store in det_meta.store_targets_seen:
            for is_adjoint in (True,) if store.adjoint else (False, True):
                storage = ArrayStoreTarget(store.array_var_label, store.attr_path, is_adjoint)
                interval = _det_array_byte_bounds(_resolve_target(kernel, storage, fwd_args, adj_args))
                if interval is not None:
                    store_bounds.append(interval)
        for target in det_meta.scatter_targets:
            if not target.use_backward or target in ordered_backward_targets:
                continue
            interval = _det_array_byte_bounds(_resolve_target(kernel, target, fwd_args, adj_args))
            if interval is not None and any(interval[0] < end and start < interval[1] for start, end in store_bounds):
                ordered_backward_targets.append(target)
    serial_backward = bool(ordered_backward_targets)
    if serial_backward and dim > 1:
        names = ", ".join(t.target_label for t in ordered_backward_targets)
        raise RuntimeError(
            f"Deterministic mode does not support parallel backward launches in kernel '{kernel.key}' with "
            f"array mutation and gradient reduction on target(s): {names}. "
            "Use separate mutation and reduction kernels, use a single-thread launch, "
            "or disable deterministic mode for this kernel."
        )

    scatter = []
    for target in det_meta.scatter_targets:
        active = target.use_backward if adjoint else target.use_forward
        if not active or target.reduce_op != REDUCE_OP_ADD or serial_backward:
            scatter.append((target, None))  # inactive, or min/max (native total-order atomics)
            continue
        arr = _resolve_target(kernel, target, fwd_args, adj_args)
        scatter.append((target, arr if _usable(arr) else None))

    counters = []
    for target in det_meta.counter_targets:
        active = target.use_backward if adjoint else target.use_forward
        arr = _resolve_target(kernel, target, fwd_args, adj_args) if active else None
        # one record per call site; a call inside a dynamic loop can repeat, so its target takes the
        # module's deterministic_max_records bound (CUDA applies that bound to every target)
        rpt = max(det_meta.counter_records_per_thread.get(target, 1), 1)
        if target in det_meta.counter_dynamic:
            rpt = max(rpt, det_meta.max_records)
        counters.append((target, arr if _usable(arr) else None, rpt))

    if any(a is not None for _, a in scatter) and dim > (1 << 32):
        raise RuntimeError(
            "Deterministic scatter atomics support launch sizes up to 2^32 threads because thread indices are "
            "32-bit."
        )
    if any(a is not None for _, a, _ in counters) and dim > ((1 << 31) - 1):
        raise RuntimeError(
            "Deterministic consumed-return counter atomics support launch sizes up to 2^31 - 1 threads because "
            "counter prefix buffers store per-thread contributions as int32 values."
        )

    state = _state(device)
    key = (
        adjoint,
        dim,
        tuple((t.helper_name, a.ptr, a.size, a.strides) if a is not None else None for t, a in scatter),
        tuple((t.helper_name, a.ptr, a.size, a.strides, r) if a is not None else None for t, a, r in counters),
    )
    plans = hooks.__dict__.setdefault("_metal_det_plans", {})
    plan = plans.get(key)
    if plan is None:
        if len(plans) > 64:
            plans.clear()  # plans stay valid while referenced; a bounded cache keeps memory in check
        plan = _Plan(
            state,
            kernel,
            det_meta,
            dim,
            [(t, a) for t, a in scatter],
            [(t, a, r) for t, a, r in counters],
            int(wp.config.deterministic_debug),
            records=det_meta.determinism_mode == wp.DeterministicMode.GPU_TO_GPU,
        )
        plans[key] = plan
    # counter arrays are read through the plan's args (their host pointers are fixed by the key); the
    # destination array descriptors inside the det struct do not change either

    if plan.counter_args is not None:
        plan.det.phase = 0
        invoke_metal(kernel, hooks, params, device, block_dim, adjoint, det_args=plan.det)
        plan.run_counter_passes()
        plan.det.phase = 1
        invoke_metal(kernel, hooks, params, device, block_dim, adjoint, det_args=plan.det)
        plan.run_writeback()
    else:
        plan.det.phase = 1
        invoke_metal(kernel, hooks, params, device, block_dim, adjoint, det_args=plan.det)
    plan.run_finalize()

    if device.is_capturing:
        from warp._src.context import runtime  # noqa: PLC0415

        graph = runtime._metal_graph
        if graph is not None:
            # the graph replays with these buffers after this call returns
            graph._deterministic_buffer_refs.extend(
                b
                for b in (plan.zero_buf, plan.scratch_buf, state.overflow, *(s.plan_buf for s in plan.sorts))
                if b is not None
            )
    else:
        _check_overflow(state, kernel)


def _check_overflow(state, kernel):
    # The flag lives in shared memory: peek without waiting for the GPU. An overflow is reported by the first
    # deterministic launch that sees it (possibly a later one than the launch that overflowed).
    if state.overflow_reported:
        return
    value = ctypes.c_int32.from_address(state.overflow.ptr).value
    if value:
        state.overflow_reported = True
        raise RuntimeError(
            "Deterministic counter overflow on Metal: a thread made more consumed-return atomic_add calls on "
            f"one counter array than its records_per_thread bound (detected at kernel '{kernel.key}'). "
            "Increase the 'deterministic_max_records' module option for the kernel that overflowed."
        )


def create_launch(**kwargs) -> Any:
    """``wp.launch(..., record_cmd=True)`` object that replays through :func:`launch`."""
    from warp._src.context import Launch  # noqa: PLC0415

    class MetalDeterministicLaunch(Launch):
        def __init__(self, det_meta, **kw):
            super().__init__(**kw)
            self.det_meta = det_meta
            self.fwd_args = list(kw.get("fwd_args") or [])
            self.adj_args = list(kw.get("adj_args") or [])

        def _is_deterministic_target_arg(self, index: int) -> bool:
            if index < 0 or index >= len(self.kernel.adj.args):
                return False
            label = self.kernel.adj.args[index].label
            targets = (*self.det_meta.scatter_targets, *self.det_meta.counter_targets, *self.det_meta.store_targets_seen)
            return any(t.array_var_label == label for t in targets)

        def set_param_at_index(self, index: int, value, adjoint: bool = False):
            super().set_param_at_index(index, value, adjoint)
            args = self.adj_args if adjoint else self.fwd_args
            if index < len(args):
                args[index] = value

        def set_param_at_index_from_ctype(self, index: int, value, adjoint: bool = False):
            if self._is_deterministic_target_arg(index):
                raise RuntimeError(
                    "Updating deterministic target argument "
                    f"'{self.kernel.adj.args[index].label}' from a raw ctypes value is not supported. "
                    "Use set_param_at_index() with a Warp array or struct object instead."
                )
            super().set_param_at_index_from_ctype(index, value, adjoint)
            args = self.adj_args if adjoint else self.fwd_args
            if index < len(args):
                args[index] = value

        def launch(self, stream=None) -> None:
            launch(
                self.kernel,
                self.hooks,
                self.params,
                self.bounds,
                self.device,
                self.det_meta,
                self.fwd_args,
                self.adj_args,
                self.adjoint,
                self.block_dim,
            )

    det_meta = kwargs.pop("det_meta")
    return MetalDeterministicLaunch(det_meta, **kwargs)
