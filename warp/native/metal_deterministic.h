// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Deterministic mode on Metal (included by deterministic.h for Metal kernels).
//
// The code generator lowers atomics exactly as for CUDA (see warp/_src/deterministic.py); only the helpers
// below and the launch-time passes in warp/_src/metal_deterministic.py differ.
//
// Accumulation atomics (return value unused) with a floating-point value are added, in phase 1, into an exact
// integer accumulator per destination component: the value's 24-bit significand is placed at its binary
// exponent in a 288-bit fixed-point number (bit 0 = 2^-149, the smallest float32 subnormal) held as nine
// 32-bit bins, each bin an int64 sum of 32-bit slices (two's complement, built from 32-bit atomics with carry
// propagation). Integer addition commutes, so the bins end up identical whatever order the threads add in; a
// pass after the kernel rounds each accumulated sum to float once and adds it to the destination. No records
// are written and no sort is needed. Min/max accumulation keeps native atomics (Metal's float min/max are
// total-order compare-and-swap loops, see metal_atomics.h, so their result does not depend on order either).
//
// Consumed-return counter atomics (int32 atomic_add whose result is used) run in two phases as on CUDA. Phase 0
// stores each call's (destination, value) at the record index thread * records_per_thread + call number of the
// thread, with all other global side effects suppressed. The launch then sorts the records by destination
// (stable, so thread order is kept within a destination) and scans them; phase 1 returns the scanned slot.

namespace wp {

#define WP_DET_NBINS 9

// One accumulation target (an array, or the gradient of one): bins and flags per destination component.
template <typename T> struct det_scatter_buf_t {
    WP_DEVICE metal::atomic_uint* bins;  // [dest_size * components][WP_DET_NBINS][lo, hi]
    WP_DEVICE metal::atomic_uint* flags;  // [dest_size * components]: 1 NaN, 2 +inf, 4 -inf
    WP_DEVICE metal::atomic_uint* touched;  // [dest_size]: destination has contributions
    WP_DEVICE int* touched_list;  // [dest_size]: destinations to finalize, in any order
    WP_DEVICE metal::atomic_int* touched_count;  // [1]
    // GPU_TO_GPU: record each update at thread * rec_per_thread + call instead (bins are null); the launch sorts
    // the records by destination and sums each destination's records in thread order, as CUDA does
    WP_DEVICE int* rec_dest;  // [dim * rec_per_thread]
    WP_DEVICE char* rec_vals;  // [dim * rec_per_thread] of T
    WP_DEVICE int* rec_cursors;  // [dim]
    int dest_size;
    int rec_per_thread;
    // element 0 of the destination (float scalars): read by the in-kernel finalize of single-threadgroup launches
    WP_DEVICE char* dest;
};

// One consumed-return counter target.
struct det_counter_buf_t {
    WP_DEVICE int* dest_of;  // [capacity] phase 0: destination of each call
    WP_DEVICE int* values;  // [capacity] phase 0: requested count of each call
    WP_DEVICE int* prefixes;  // [capacity] phase 1: slot returned by each call
    WP_DEVICE int* cursors;  // [2 * dim]: calls made so far by each thread in phase 0, then in phase 1
    int capacity;
    int records_per_thread;
    int dim;
    int target_size;
    array_t<int> target;
};

struct det_ctx {
    int phase;
    int debug;
    size_t idx;
    WP_DEVICE metal::atomic_int* overflow;
    WP_THREAD array_t<int>* counter_targets;
    int counter_target_count;
    // per-thread fixed-array scratch (wp.zeros in kernels): thread-private, so phase 0 keeps its stores
    WP_DEVICE char* fixed_lo;
    WP_DEVICE char* fixed_hi;
};

template <typename T> struct det_scalar {
    typedef T type;
};
template <unsigned Length, typename Type> struct det_scalar<vec_t<Length, Type>> {
    typedef Type type;
};
template <unsigned Rows, unsigned Cols, typename Type> struct det_scalar<mat_t<Rows, Cols, Type>> {
    typedef Type type;
};
template <typename Type> struct det_scalar<quat_t<Type>> {
    typedef Type type;
};
template <typename Type> struct det_scalar<transform_t<Type>> {
    typedef Type type;
};

namespace deterministic {

inline float to_float(float x) { return x; }
inline float to_float(half x) { return half_to_float(x); }
#ifndef WP_NO_BFLOAT16
inline float to_float(wp_bfloat16 x) { return bfloat16_to_float(x); }
#endif

inline void flag_overflow(thread det_ctx& ctx)
{
    if (ctx.overflow)
        metal::atomic_store_explicit(ctx.overflow, 1, metal::memory_order_relaxed);
}

// Adds a signed 64-bit value to the bin at b (two 32-bit words, low first): exact modulo 2^64 in any order.
inline void bin_add(WP_DEVICE metal::atomic_uint* b, long v)
{
    const uint lo = uint(ulong(v));
    const uint hi = uint(ulong(v) >> 32);
    const uint old = metal::atomic_fetch_add_explicit(b, lo, metal::memory_order_relaxed);
    const uint carry = (old + lo < old) ? 1u : 0u;
    const uint add_hi = hi + carry;
    if (add_hi)
        metal::atomic_fetch_add_explicit(b + 1, add_hi, metal::memory_order_relaxed);
}

// Exact accumulation of one float into the bins of one destination component.
inline void deposit(WP_DEVICE metal::atomic_uint* bins, WP_DEVICE metal::atomic_uint* flag, float x)
{
    const uint bits = as_type<uint>(x);
    const uint e = (bits >> 23) & 0xffu;
    uint m = bits & 0x7fffffu;
    const bool negative = (bits >> 31) != 0u;
    if (e == 0xffu) {
        metal::atomic_fetch_or_explicit(flag, m ? 1u : (negative ? 4u : 2u), metal::memory_order_relaxed);
        return;
    }
    uint pos = 0u;  // x = m * 2^(pos - 149)
    if (e == 0u) {
        if (m == 0u)
            return;
    } else {
        m |= 0x800000u;
        pos = e - 1u;
    }
    const ulong wide = ulong(m) << (pos & 31u);
    const uint b = pos >> 5;
    const long lo = long(wide & 0xffffffffull);
    const long hi = long(wide >> 32);
    bin_add(bins + 2u * b, negative ? -lo : lo);
    if (hi)
        bin_add(bins + 2u * (b + 1u), negative ? -hi : hi);
}

template <typename T>
inline void scatter(thread det_ctx& ctx, thread const det_scatter_buf_t<T>& buf, int dest, T value)
{
    typedef typename det_scalar<T>::type S;
    constexpr int C = int(sizeof(T) / sizeof(S));
    if (dest < 0 || dest >= buf.dest_size)
        return;
    if (buf.rec_dest != nullptr) {
        WP_DEVICE int* cursor = buf.rec_cursors + ctx.idx;
        const int call = *cursor;  // only this thread touches its cursor
        *cursor = call + 1;
        if (call >= buf.rec_per_thread) {
            flag_overflow(ctx);
            return;
        }
        const size_t record = ctx.idx * size_t(buf.rec_per_thread) + size_t(call);
        buf.rec_dest[record] = dest;
        reinterpret_cast<WP_DEVICE T*>(buf.rec_vals)[record] = value;
        return;
    }
    thread const S* s = reinterpret_cast<thread const S*>(&value);
    for (int c = 0; c < C; ++c) {
        const size_t slot = size_t(dest) * C + c;
        deposit(buf.bins + slot * (2 * WP_DET_NBINS), buf.flags + slot, to_float(s[c]));
    }
    if (metal::atomic_exchange_explicit(buf.touched + dest, 1u, metal::memory_order_relaxed) == 0u) {
        const int i = metal::atomic_fetch_add_explicit(buf.touched_count, 1, metal::memory_order_relaxed);
        buf.touched_list[i] = dest;
    }
}

inline int counter_add(thread det_ctx& ctx, thread const det_counter_buf_t& buf, int dest, int value)
{
    WP_DEVICE int* cursor = buf.cursors + (ctx.phase == 0 ? size_t(0) : size_t(buf.dim)) + ctx.idx;
    const int call = *cursor;  // only this thread touches its cursor
    *cursor = call + 1;
    if (call >= buf.records_per_thread) {
        flag_overflow(ctx);
        return 0;
    }
    const long record = long(ctx.idx) * buf.records_per_thread + call;
    if (record < 0 || record >= buf.capacity) {
        flag_overflow(ctx);
        return 0;
    }
    if (ctx.phase == 0) {
        buf.dest_of[record] = dest;
        buf.values[record] = value;
        return 0;
    }
    // the base slot of this reservation; values greater than one reserve [slot, slot + value)
    return buf.prefixes[record];
}

inline bool is_reachable_counter_element(thread const array_t<int>& target, WP_DEVICE const void* ptr)
{
    if (target.data == nullptr || target.ndim <= 0)
        return false;

    const long offset = long(
        reinterpret_cast<WP_DEVICE const char*>(ptr) - reinterpret_cast<WP_DEVICE const char*>(target.data)
    );
    long min_offset = 0;
    long max_offset = 0;
    for (int i = 0; i < target.ndim; ++i) {
        const int shape = target.shape[i];
        if (shape <= 0)
            return false;
        const long extent = long(shape - 1) * long(target.strides[i]);
        if (extent < 0)
            min_offset += extent;
        else
            max_offset += extent;
    }
    if (offset < min_offset || offset > max_offset)
        return false;

    int dims[4] = { 0, 1, 2, 3 };
    for (int i = 0; i < target.ndim; ++i) {
        for (int j = i + 1; j < target.ndim; ++j) {
            const long ls = long(target.strides[dims[i]]);
            const long rs = long(target.strides[dims[j]]);
            const long lhs = ls < 0 ? -ls : ls;
            const long rhs = rs < 0 ? -rs : rs;
            if (rhs > lhs) {
                const int tmp = dims[i];
                dims[i] = dims[j];
                dims[j] = tmp;
            }
        }
    }
    long remaining = offset;
    for (int k = 0; k < target.ndim; ++k) {
        const int dim = dims[k];
        const long stride = target.strides[dim];
        if (stride == 0)
            continue;
        const long index = remaining / stride;
        if (index < 0 || index >= target.shape[dim])
            return false;
        remaining -= index * stride;
    }
    return remaining == 0;
}

inline bool is_counter_store_target(thread const det_ctx& ctx, WP_DEVICE const void* ptr)
{
    for (int i = 0; i < ctx.counter_target_count; ++i) {
        if (is_reachable_counter_element(ctx.counter_targets[i], ptr))
            return true;
    }
    return false;
}

// Whether phase 0 may perform a store to ptr: always in phase 1; in phase 0 only to thread-private memory
// (registers, the fixed-array scratch) and to counter targets (as on CUDA).
inline bool store_allowed(thread const det_ctx& ctx, WP_DEVICE const void* ptr)
{
    if (ctx.phase != 0)
        return true;
    WP_DEVICE const char* p = reinterpret_cast<WP_DEVICE const char*>(ptr);
    if (p >= ctx.fixed_lo && p < ctx.fixed_hi)
        return true;
    return is_counter_store_target(ctx, ptr);
}
inline bool store_allowed(thread const det_ctx& ctx, WP_THREAD const void* ptr) { return true; }
inline bool store_allowed(thread const det_ctx& ctx, WP_THREADGROUP const void* ptr) { return true; }

template <template <typename> class A, typename T>
inline void array_store_if_active(thread det_ctx& ctx, thread const A<T>& buf, int i, T value)
{
    if (store_allowed(ctx, wp::address(buf, i)))
        wp::array_store(buf, i, value);
}

template <template <typename> class A, typename T>
inline void array_store_if_active(thread det_ctx& ctx, thread const A<T>& buf, int i, int j, T value)
{
    if (store_allowed(ctx, wp::address(buf, i, j)))
        wp::array_store(buf, i, j, value);
}

template <template <typename> class A, typename T>
inline void array_store_if_active(thread det_ctx& ctx, thread const A<T>& buf, int i, int j, int k, T value)
{
    if (store_allowed(ctx, wp::address(buf, i, j, k)))
        wp::array_store(buf, i, j, k, value);
}

template <template <typename> class A, typename T>
inline void array_store_if_active(thread det_ctx& ctx, thread const A<T>& buf, int i, int j, int k, int l, T value)
{
    if (store_allowed(ctx, wp::address(buf, i, j, k, l)))
        wp::array_store(buf, i, j, k, l, value);
}

template <typename P, typename V> inline void slot_store_if_active(thread det_ctx& ctx, P ptr, V value)
{
    if (store_allowed(ctx, ptr))
        wp::store(ptr, value);
}

// The exact sum held by one destination component's bins, rounded to float once (det_scatter_finalize's
// bins_to_float in warp/_src/metal_deterministic.py, the same operations).
inline float bins_to_float(WP_DEVICE uint* w, uint flag)
{
    if ((flag & 1u) || (flag & 6u) == 6u)
        return NAN;
    if (flag & 2u)
        return INFINITY;
    if (flag & 4u)
        return -INFINITY;
    uint limb[WP_DET_NBINS + 2];
    long carry = 0;
    for (uint b = 0; b < WP_DET_NBINS; ++b) {
        const long v = long((ulong(w[2 * b + 1]) << 32) | ulong(w[2 * b]));
        const long s = v + carry;
        limb[b] = uint(ulong(s));
        carry = s >> 32;
    }
    limb[WP_DET_NBINS] = uint(ulong(carry));
    limb[WP_DET_NBINS + 1] = uint(ulong(carry) >> 32);
    const bool negative = carry < 0;
    if (negative) {
        uint borrow = 1u;
        for (uint b = 0; b < WP_DET_NBINS + 2; ++b) {
            const uint t = ~limb[b] + borrow;
            borrow = (borrow != 0u && t == 0u) ? 1u : 0u;
            limb[b] = t;
        }
    }
    int top = WP_DET_NBINS + 1;
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
            window |= 1ul;
    }
    const float r = metal::ldexp(float(window), lsb);
    return negative ? -r : r;
}

// det_scatter_finalize inside the kernel, for a launch that is one threadgroup: after a device-memory barrier, the
// threadgroup's first `stride` threads round each touched destination's sum once, add it to the destination and
// clear the bins, then the touched count is re-armed. Every thread of the threadgroup must call it (barriers).
template <typename T> inline void finalize_tg(thread const det_scatter_buf_t<T>& buf, uint lid, uint stride)
{
    typedef typename det_scalar<T>::type S;
    constexpr int C = int(sizeof(T) / sizeof(S));
    if (buf.touched_count == nullptr || buf.bins == nullptr)
        return;  // target not accumulated exactly in this launch (uniform)
    const int n = metal::atomic_load_explicit(buf.touched_count, metal::memory_order_relaxed);
    if (lid < stride) {
        for (int i = int(lid); i < n; i += int(stride)) {
            const int d = buf.touched_list[i];
            for (int c = 0; c < C; ++c) {
                const size_t slot = size_t(d) * C + c;
                WP_DEVICE uint* w = reinterpret_cast<WP_DEVICE uint*>(buf.bins + slot * (2 * WP_DET_NBINS));
                WP_DEVICE uint* flag = reinterpret_cast<WP_DEVICE uint*>(buf.flags + slot);
                const float r = bins_to_float(w, *flag);
                for (uint k = 0; k < 2 * WP_DET_NBINS; ++k)
                    w[k] = 0u;
                *flag = 0u;
                WP_DEVICE float* f = reinterpret_cast<WP_DEVICE float*>(buf.dest) + slot;
                *f = *f + r;
            }
            reinterpret_cast<WP_DEVICE uint*>(buf.touched)[d] = 0u;
        }
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_device);  // every thread has read the count
    if (lid == 0)
        metal::atomic_store_explicit(buf.touched_count, 0, metal::memory_order_relaxed);
}

}  // namespace deterministic
}  // namespace wp

#define WP_DET_SCATTER_OR_FALLBACK(det_ctx, helper, flat_idx, value, cpu_expr) \
    do { \
        if ((det_ctx).phase != 0) { \
            if ((helper).bins != nullptr || (helper).rec_dest != nullptr) { \
                wp::deterministic::scatter((det_ctx), (helper), static_cast<int>(flat_idx), (value)); \
            } else { \
                cpu_expr; \
            } \
        } \
    } while (0)

#define WP_DET_COUNTER_OR_FALLBACK(out, det_ctx, helper, flat_idx, value, cpu_expr) \
    do { \
        (out) = ((helper).dest_of != nullptr) \
            ? wp::deterministic::counter_add((det_ctx), (helper), static_cast<int>(flat_idx), (value)) \
            : 0; \
    } while (0)

#define WP_DET_STORE_IF_ACTIVE(det_ctx, ...) \
    do { \
        wp::deterministic::array_store_if_active((det_ctx), __VA_ARGS__); \
    } while (0)

#define WP_DET_SLOT_STORE_IF_ACTIVE(det_ctx, slot_lvalue, value_expr) \
    do { \
        wp::deterministic::slot_store_if_active((det_ctx), &(slot_lvalue), (value_expr)); \
    } while (0)

#define WP_DET_SIDE_EFFECT_IF_ACTIVE(det_ctx, ...) \
    do { \
        if ((det_ctx).phase != 0) { \
            __VA_ARGS__ \
        } \
    } while (0)
