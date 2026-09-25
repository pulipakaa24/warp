// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// C runtime subset for kernel modules compiled by the Metal shading language
// compiler. Included from crt.h when __METAL_VERSION__ is defined. Metal has
// no double, no 64-bit `long long`, no printf and no host callbacks, so this
// header maps the CRT names used by builtin.h onto the Metal standard library
// and turns the unavailable pieces into no-ops.

#include "metal_atomics.h"

#include <metal_logging>
#include <metal_stdlib>

#define WP_NO_FLOAT64

// `half` is a builtin Metal type; Warp's software half lives under another name here.
#define half wp_half

// Host->GPU address translation for pointers read from memory (see refresh_table in metal.mm). The function
// constant holds the GPU address of a slot that points at the current sorted table {count, pad, {host, length,
// gpu}...}.
constant unsigned long wp_metal_table_slot [[function_constant(0)]];
struct wp_metal_range {
    unsigned long host, length, gpu;
};
inline unsigned long wp_metal_translate(unsigned long host)
{
    if (host == 0 || wp_metal_table_slot == 0)
        return host;
    const unsigned long table = *(device const unsigned long*)wp_metal_table_slot;
    if (table == 0)
        return host;
    device const unsigned long* header = (device const unsigned long*)table;
    const unsigned long count = header[0];
    device const wp_metal_range* ranges = (device const wp_metal_range*)(header + 2);
    unsigned long lo = 0, hi = count;
    while (lo < hi) {
        const unsigned long mid = (lo + hi) / 2;
        if (ranges[mid].host <= host)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == 0)
        return host;
    const wp_metal_range r = ranges[lo - 1];
    return host - r.host <= r.length ? r.gpu + (host - r.host) : host;
}
#define WP_METAL_TRANSLATE(x) wp_metal_translate((unsigned long)(x))

// string.h: byte copies between thread-local objects (used for bit casts).
inline void memcpy(thread void* dst, thread const void* src, size_t n)
{
    thread char* d = (thread char*)dst;
    thread const char* s = (thread const char*)src;
    for (size_t i = 0; i < n; ++i)
        d[i] = s[i];
}
inline void memset(thread void* dst, int value, size_t n)
{
    thread char* d = (thread char*)dst;
    for (size_t i = 0; i < n; ++i)
        d[i] = char(value);
}

// Metal has no heap; these exist only so that CPU-only code paths still parse.
inline thread void* malloc(size_t) { return nullptr; }
inline void free(thread void*) { }

// printf goes through Metal shader logging (macOS 14+): the format must be a string literal and
// %s is not supported; the runtime forwards the lines to stdout (see metal.mm log handler).
#define printf(...) metal::os_log_default.log_info(__VA_ARGS__)

#define assert(x) ((void)0)

/// float.h
#ifndef FLT_RADIX
#define FLT_RADIX 2
#endif
#ifndef FLT_MANT_DIG
#define FLT_MANT_DIG 24
#endif
#ifndef FLT_DIG
#define FLT_DIG 6
#endif
#ifndef FLT_MIN_EXP
#define FLT_MIN_EXP -125
#endif
#ifndef FLT_MAX_EXP
#define FLT_MAX_EXP 128
#endif
#ifndef FLT_MAX
#define FLT_MAX 3.4028234e38f
#endif
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209289e-7f
#endif
#ifndef FLT_MIN
#define FLT_MIN 1.1754943e-38f
#endif

/// limits.h
#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif
#ifndef SCHAR_MIN
#define SCHAR_MIN (-128)
#endif
#ifndef SCHAR_MAX
#define SCHAR_MAX 127
#endif
#ifndef UCHAR_MAX
#define UCHAR_MAX 255
#endif
#ifndef SHRT_MIN
#define SHRT_MIN (-32768)
#endif
#ifndef SHRT_MAX
#define SHRT_MAX 32767
#endif
#ifndef USHRT_MAX
#define USHRT_MAX 65535
#endif
#ifndef INT_MAX
#define INT_MAX 2147483647
#endif
#ifndef INT_MIN
#define INT_MIN (-INT_MAX - 1)
#endif
#ifndef UINT_MAX
#define UINT_MAX 4294967295U
#endif
#ifndef LONG_MAX
#define LONG_MAX 9223372036854775807L
#endif
#ifndef LONG_MIN
#define LONG_MIN (-LONG_MAX - 1L)
#endif
#ifndef ULONG_MAX
#define ULONG_MAX 18446744073709551615UL
#endif
#ifndef LLONG_MAX
#define LLONG_MAX LONG_MAX
#endif
#ifndef LLONG_MIN
#define LLONG_MIN LONG_MIN
#endif
#ifndef ULLONG_MAX
#define ULLONG_MAX ULONG_MAX
#endif

/// math.h: builtin.h calls the C names; forward them to the Metal library.
inline int abs(int x) { return metal::abs(x); }
inline long llabs(long x) { return metal::abs(x); }
inline float fmodf(float x, float y) { return metal::fmod(x, y); }
inline float logf(float x) { return metal::log(x); }
inline float log2f(float x) { return metal::log2(x); }
inline float log10f(float x) { return metal::log10(x); }
inline float expf(float x) { return metal::exp(x); }
inline float exp2f(float x) { return metal::exp2(x); }
inline float ldexpf(float x, int e) { return metal::ldexp(x, e); }
inline float sqrtf(float x) { return metal::sqrt(x); }
inline float cbrtf(float x) { return metal::copysign(metal::pow(metal::abs(x), 1.0f / 3.0f), x); }
// exact results for small integer exponents (x*x etc.); metal::pow(6, 2) is 36.000004 even in precise mode
inline float wp_metal_pow(float x, float y)
{
    if (y == metal::rint(y) && metal::abs(y) <= 8.0f) {
        int n = int(metal::abs(y));
        float r = 1.0f, b = x;
        for (; n; n >>= 1, b *= b)
            if (n & 1)
                r *= b;
        return y < 0.0f ? 1.0f / r : r;
    }
    return metal::pow(x, y);
}
inline float powf(float x, float y) { return wp_metal_pow(x, y); }
inline float floorf(float x) { return metal::floor(x); }
inline float ceilf(float x) { return metal::ceil(x); }
inline float fabsf(float x) { return metal::abs(x); }
inline float roundf(float x) { return metal::round(x); }
inline float truncf(float x) { return metal::trunc(x); }
inline float rintf(float x) { return metal::rint(x); }
inline float acosf(float x) { return metal::acos(x); }
inline float asinf(float x) { return metal::asin(x); }
inline float atanf(float x) { return metal::atan(x); }
inline float atan2f(float y, float x) { return metal::atan2(y, x); }
inline float cosf(float x) { return metal::cos(x); }
inline float sinf(float x) { return metal::sin(x); }
inline float tanf(float x) { return metal::tan(x); }
inline float sinhf(float x) { return metal::sinh(x); }
inline float coshf(float x) { return metal::cosh(x); }
inline float tanhf(float x) { return metal::tanh(x); }
inline float fmaf(float x, float y, float z) { return metal::fma(x, y, z); }
inline float fminf(float x, float y) { return metal::fmin(x, y); }
inline float fmaxf(float x, float y) { return metal::fmax(x, y); }
inline float copysignf(float x, float y) { return metal::copysign(x, y); }
// C99 nextafterf: the Metal library has none (air64 link error "Undefined symbol nextafterf" when native
// snippets call it, e.g. Newton's VBD interval arithmetic). Bit-exact IEEE semantics, computed on the bit
// patterns only: Apple GPUs flush float32 subnormals in comparisons, so a float compare would treat +-1 ULP
// subnormals as zero. NaN in -> NaN, x == y -> y, +-0 -> smallest subnormal of y's sign, else one ULP toward y.
inline int wp_metal_float_order(unsigned int u)
{
    return (u & 0x80000000u) ? -int(u & 0x7fffffffu) : int(u);
}
inline float nextafterf(float x, float y)
{
    const unsigned int ux = as_type<unsigned int>(x), uy = as_type<unsigned int>(y);
    if ((ux & 0x7fffffffu) > 0x7f800000u || (uy & 0x7fffffffu) > 0x7f800000u)
        return x + y;
    const int kx = wp_metal_float_order(ux), ky = wp_metal_float_order(uy);
    if (kx == ky)
        return y;
    if ((ux & 0x7fffffffu) == 0u)
        return as_type<float>((ky > 0 ? 0u : 0x80000000u) | 1u);
    const bool away_from_zero = (kx < ky) == ((ux & 0x80000000u) == 0u);
    return as_type<float>(away_from_zero ? ux + 1u : ux - 1u);
}
// clang lowers __builtin_nextafterf to a libcall of nextafterf, which does not exist for air64; route the
// builtin spelling (used by CPU branches of native snippets that Metal also takes) to the function above.
#define __builtin_nextafterf(x, y) nextafterf((x), (y))
inline float rsqrtf(float x) { return metal::rsqrt(x); }

// Metal has no error functions. Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
// for erf, and Giles' single-precision approximation for the inverse.
inline float erff(float x)
{
    const float t = 1.0f / (1.0f + 0.3275911f * metal::abs(x));
    const float poly
        = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    const float y = 1.0f - poly * metal::exp(-x * x);
    return metal::copysign(y, x);
}
inline float erfcf(float x) { return 1.0f - erff(x); }
inline float erfinvf(float x)
{
    float w = -metal::log((1.0f - x) * (1.0f + x));
    float p;
    if (w < 5.0f) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = metal::sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}
inline float erfcinvf(float x) { return erfinvf(1.0f - x); }

// Unsuffixed C names used with wide_float (float on Metal).
inline float fabs(float x) { return metal::abs(x); }
inline float sqrt(float x) { return metal::sqrt(x); }
inline float floor(float x) { return metal::floor(x); }
inline float ceil(float x) { return metal::ceil(x); }
inline float round(float x) { return metal::round(x); }
inline float trunc(float x) { return metal::trunc(x); }
inline float exp(float x) { return metal::exp(x); }
inline float log(float x) { return metal::log(x); }
inline float pow(float x, float y) { return wp_metal_pow(x, y); }
inline float fmin(float x, float y) { return metal::fmin(x, y); }
inline float fmax(float x, float y) { return metal::fmax(x, y); }
inline float fma(float x, float y, float z) { return metal::fma(x, y, z); }
inline float copysign(float x, float y) { return metal::copysign(x, y); }
inline float sin(float x) { return metal::sin(x); }
inline float cos(float x) { return metal::cos(x); }
inline float tan(float x) { return metal::tan(x); }
inline float asin(float x) { return metal::asin(x); }
inline float acos(float x) { return metal::acos(x); }
inline float atan(float x) { return metal::atan(x); }
inline float atan2(float y, float x) { return metal::atan2(y, x); }
inline float sinh(float x) { return metal::sinh(x); }
inline float cosh(float x) { return metal::cosh(x); }
inline float tanh(float x) { return metal::tanh(x); }
inline float log2(float x) { return metal::log2(x); }
inline float log10(float x) { return metal::log10(x); }
inline float exp2(float x) { return metal::exp2(x); }
inline float rint(float x) { return metal::rint(x); }
inline float fmod(float x, float y) { return metal::fmod(x, y); }

// Host assertion callback used by builtin.h; assertions are disabled on Metal.
// Assertion messages are literals at every call site, so they can be logged directly.
#define _wp_assert(msg, file, line) metal::os_log_default.log_error(msg " (assertion failed, line %u)\n", unsigned(line))

// cmath
inline bool isfinite(float x) { return metal::isfinite(x); }
inline bool isnan(float x) { return metal::isnan(x); }
inline bool isinf(float x) { return metal::isinf(x); }
