// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device-memory atomics for the Metal backend, used by the wp::atomic_* builtins.
// Metal has native atomics for 32-bit integers, floats (add/exchange/CAS only) and,
// on Apple9 GPUs and later, unsigned 64-bit minimum and maximum. 16-bit and 64-bit adds are
// built from those; everything else is a plain read-modify-write (see docs/user_guide/metal.rst).

#include <metal_stdlib>

#define WP_METAL_INT_ATOMICS(T, ATOMIC_T)                                                                     \
    inline T wp_metal_atomic_add(device T* p, T v)                                                            \
    {                                                                                                         \
        return metal::atomic_fetch_add_explicit((device ATOMIC_T*)p, v, metal::memory_order_relaxed);         \
    }                                                                                                         \
    inline T wp_metal_atomic_min(device T* p, T v)                                                            \
    {                                                                                                         \
        return metal::atomic_fetch_min_explicit((device ATOMIC_T*)p, v, metal::memory_order_relaxed);         \
    }                                                                                                         \
    inline T wp_metal_atomic_max(device T* p, T v)                                                            \
    {                                                                                                         \
        return metal::atomic_fetch_max_explicit((device ATOMIC_T*)p, v, metal::memory_order_relaxed);         \
    }                                                                                                         \
    inline T wp_metal_atomic_and(device T* p, T v)                                                            \
    {                                                                                                         \
        return metal::atomic_fetch_and_explicit((device ATOMIC_T*)p, v, metal::memory_order_relaxed);         \
    }                                                                                                         \
    inline T wp_metal_atomic_or(device T* p, T v)                                                             \
    {                                                                                                         \
        return metal::atomic_fetch_or_explicit((device ATOMIC_T*)p, v, metal::memory_order_relaxed);          \
    }                                                                                                         \
    inline T wp_metal_atomic_xor(device T* p, T v)                                                            \
    {                                                                                                         \
        return metal::atomic_fetch_xor_explicit((device ATOMIC_T*)p, v, metal::memory_order_relaxed);         \
    }                                                                                                         \
    inline T wp_metal_atomic_exch(device T* p, T v)                                                           \
    {                                                                                                         \
        return metal::atomic_exchange_explicit((device ATOMIC_T*)p, v, metal::memory_order_relaxed);          \
    }                                                                                                         \
    inline T wp_metal_atomic_cas(device T* p, T compare, T v)                                                 \
    {                                                                                                         \
        T expected = compare;                                                                                 \
        while (!metal::atomic_compare_exchange_weak_explicit(                                                 \
                   (device ATOMIC_T*)p, &expected, v, metal::memory_order_relaxed, metal::memory_order_relaxed \
               )                                                                                              \
               && expected == compare) {                                                                      \
        }                                                                                                     \
        return expected;                                                                                      \
    }

WP_METAL_INT_ATOMICS(int, metal::atomic_int)
WP_METAL_INT_ATOMICS(unsigned int, metal::atomic_uint)
#undef WP_METAL_INT_ATOMICS

// Metal only offers 64-bit atomic min/max (Apple9 GPUs and later). Signed values are
// ordered as unsigned by flipping the sign bit; the other 64-bit operations use the
// non-atomic fallbacks below.
#if defined(__HAVE_ATOMIC_ULONG_MIN_MAX__)
// The 64-bit min/max atomics do not return the previous value; the value read before the
// update is returned instead, which may be stale under contention.
inline unsigned long wp_metal_atomic_min(device unsigned long* p, unsigned long v)
{
    unsigned long old = *p;
    metal::atomic_min_explicit((device metal::atomic_ulong*)p, v, metal::memory_order_relaxed);
    return old;
}
inline unsigned long wp_metal_atomic_max(device unsigned long* p, unsigned long v)
{
    unsigned long old = *p;
    metal::atomic_max_explicit((device metal::atomic_ulong*)p, v, metal::memory_order_relaxed);
    return old;
}
// Metal has no signed 64-bit atomic min/max and no 64-bit compare-exchange, so these are plain
// read-modify-write updates: correct without concurrent writers to the same element only.
inline long wp_metal_atomic_min(device long* p, long v)
{
    const long old = *p;
    if (v < old)
        *p = v;
    return old;
}
inline long wp_metal_atomic_max(device long* p, long v)
{
    const long old = *p;
    if (v > old)
        *p = v;
    return old;
}
#endif

inline float wp_metal_atomic_add(device float* p, float v)
{
    return metal::atomic_fetch_add_explicit((device metal::atomic_float*)p, v, metal::memory_order_relaxed);
}

inline float wp_metal_atomic_exch(device float* p, float v)
{
    return metal::atomic_exchange_explicit((device metal::atomic_float*)p, v, metal::memory_order_relaxed);
}

// Float compare-and-swap on the bit pattern, so NaN payloads compare exactly.
inline float wp_metal_atomic_cas(device float* p, float compare, float v)
{
    unsigned int result
        = wp_metal_atomic_cas((device unsigned int*)p, as_type<unsigned int>(compare), as_type<unsigned int>(v));
    return as_type<float>(result);
}

// Min/max without a native atomic: CAS loops on the bit pattern, with fmin()/fmax() semantics
// like CPU and CUDA: a NaN operand never replaces a number, and a number replaces a NaN slot.
// Zeros are ordered -0 < +0, so the result is the same whatever order the threads arrive in.
#define WP_METAL_CAS_MINMAX(T, BITS, ATOMIC_T, NAME, OP, ZERO_WINS)                                       \
    inline T wp_metal_atomic_##NAME(device T* p, T v)                                                     \
    {                                                                                                     \
        device ATOMIC_T* a = (device ATOMIC_T*)p;                                                         \
        BITS old_bits = metal::atomic_load_explicit(a, metal::memory_order_relaxed);                      \
        while (v OP as_type<T>(old_bits) || (metal::isnan(as_type<T>(old_bits)) && !metal::isnan(v))      \
               || (v == T(0) && as_type<T>(old_bits) == T(0) && ZERO_WINS)) {                             \
            if (metal::atomic_compare_exchange_weak_explicit(                                             \
                    a, &old_bits, as_type<BITS>(v), metal::memory_order_relaxed,                   \
                    metal::memory_order_relaxed                                                           \
                ))                                                                                        \
                break;                                                                                    \
        }                                                                                                 \
        return as_type<T>(old_bits);                                                               \
    }
WP_METAL_CAS_MINMAX(float, unsigned int, metal::atomic_uint, min, <, (metal::signbit(v) && !metal::signbit(as_type<float>(old_bits))))
WP_METAL_CAS_MINMAX(float, unsigned int, metal::atomic_uint, max, >, (!metal::signbit(v) && metal::signbit(as_type<float>(old_bits))))
#undef WP_METAL_CAS_MINMAX

// 64-bit integer add: the two 32-bit halves are added atomically with carry propagation; every add
// commutes, so the final value is exact even though a concurrent reader may see a torn intermediate.
inline unsigned long wp_metal_atomic_add(device unsigned long* p, unsigned long v)
{
    device metal::atomic_uint* lo = (device metal::atomic_uint*)p;
    device metal::atomic_uint* hi = lo + 1;  // little-endian
    const unsigned int old_lo = metal::atomic_fetch_add_explicit(lo, (unsigned int)v, metal::memory_order_relaxed);
    const unsigned int carry = (old_lo + (unsigned int)v) < old_lo ? 1u : 0u;
    const unsigned int old_hi
        = metal::atomic_fetch_add_explicit(hi, (unsigned int)(v >> 32) + carry, metal::memory_order_relaxed);
    return ((unsigned long)old_hi << 32) | old_lo;
}
inline long wp_metal_atomic_add(device long* p, long v)
{
    return (long)wp_metal_atomic_add((device unsigned long*)p, (unsigned long)v);
}

// 16-bit adds (half, bfloat16, int16, uint16): compare-and-swap on the aligned 32-bit word that holds the value.
template <typename T> inline T wp_metal_atomic_add16(device T* p, T v)
{
    device metal::atomic_uint* word = (device metal::atomic_uint*)((device char*)p - ((unsigned long)p & 2));
    const unsigned int shift = ((unsigned long)p & 2) * 8;
    unsigned int expected = metal::atomic_load_explicit(word, metal::memory_order_relaxed);
    for (;;) {
        const unsigned short bits = (unsigned short)(expected >> shift);
        const T old = as_type<T>(bits);
        const unsigned short sum = as_type<unsigned short>(T(old + v));
        const unsigned int desired = (expected & ~(0xFFFFu << shift)) | ((unsigned int)sum << shift);
        if (metal::atomic_compare_exchange_weak_explicit(
                word, &expected, desired, metal::memory_order_relaxed, metal::memory_order_relaxed
            ))
            return old;
    }
}
inline short wp_metal_atomic_add(device short* p, short v) { return wp_metal_atomic_add16(p, v); }
inline unsigned short wp_metal_atomic_add(device unsigned short* p, unsigned short v)
{
    return wp_metal_atomic_add16(p, v);
}
// half / bfloat16 keep their bits in `u`; the sum is computed in float through their operator+.
template <typename H> inline H wp_metal_atomic_add_half16(device H* p, H v)
{
    device metal::atomic_uint* word = (device metal::atomic_uint*)((device char*)p - ((unsigned long)p & 2));
    const unsigned int shift = ((unsigned long)p & 2) * 8;
    unsigned int expected = metal::atomic_load_explicit(word, metal::memory_order_relaxed);
    for (;;) {
        H old;
        old.u = (unsigned short)(expected >> shift);
        H sum = old + v;
        const unsigned int desired = (expected & ~(0xFFFFu << shift)) | ((unsigned int)sum.u << shift);
        if (metal::atomic_compare_exchange_weak_explicit(
                word, &expected, desired, metal::memory_order_relaxed, metal::memory_order_relaxed
            ))
            return old;
    }
}

// Fallbacks for types without hardware atomics (8-bit integers). Not atomic.
template <typename T> inline T wp_metal_atomic_add(device T* p, T v)
{
    static_assert(sizeof(T) == 1, "wp_metal_atomic_add: only 8-bit types may use the non-atomic fallback");
    T old = *p;
    *p = old + v;
    return old;
}
template <typename T> inline T wp_metal_atomic_min(device T* p, T v)
{
    T old = *p;
    *p = v < old ? v : old;
    return old;
}
template <typename T> inline T wp_metal_atomic_max(device T* p, T v)
{
    T old = *p;
    *p = v > old ? v : old;
    return old;
}
template <typename T> inline T wp_metal_atomic_exch(device T* p, T v)
{
    T old = *p;
    *p = v;
    return old;
}
template <typename T> inline T wp_metal_atomic_cas(device T* p, T compare, T v)
{
    T old = *p;
    if (old == compare)
        *p = v;
    return old;
}
template <typename T> inline T wp_metal_atomic_and(device T* p, T v)
{
    T old = *p;
    *p = old & v;
    return old;
}
template <typename T> inline T wp_metal_atomic_or(device T* p, T v)
{
    T old = *p;
    *p = old | v;
    return old;
}
template <typename T> inline T wp_metal_atomic_xor(device T* p, T v)
{
    T old = *p;
    *p = old ^ v;
    return old;
}
