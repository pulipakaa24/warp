// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "initializer_array.h"

namespace wp {

template <unsigned Length, typename Type> struct vec_t {
    Type c[Length < 1 ? 1 : Length];

    inline CUDA_CALLABLE vec_t()
        : c()
    {
    }

    inline CUDA_CALLABLE vec_t(Type s)
    {
        for (unsigned i = 0; i < Length; ++i) {
            c[i] = s;
        }
    }

    template <typename OtherType> inline explicit CUDA_CALLABLE vec_t(const vec_t<Length, OtherType> WP_THREAD& other)
    {
        for (unsigned i = 0; i < Length; ++i) {
            c[i] = static_cast<Type>(other[i]);
        }
    }

    inline CUDA_CALLABLE vec_t(Type x, Type y)
    {
        assert(Length == 2);
        c[0] = x;
        c[1] = y;
    }

    inline CUDA_CALLABLE vec_t(Type x, Type y, Type z)
    {
        assert(Length == 3);
        c[0] = x;
        c[1] = y;
        c[2] = z;
    }


    inline CUDA_CALLABLE vec_t(Type x, Type y, Type z, Type w)
    {
        assert(Length == 4);
        c[0] = x;
        c[1] = y;
        c[2] = z;
        c[3] = w;
    }

    inline CUDA_CALLABLE vec_t(const initializer_array<Length, Type> WP_THREAD& l)
    {
        for (unsigned i = 0; i < Length; ++i) {
            c[i] = l[i];
        }
    }

    // special screw vector constructor for spatial_vectors:
    inline CUDA_CALLABLE vec_t(vec_t<3, Type> w, vec_t<3, Type> v)
    {
        c[0] = w[0];
        c[1] = w[1];
        c[2] = w[2];
        c[3] = v[0];
        c[4] = v[1];
        c[5] = v[2];
    }

    inline CUDA_CALLABLE Type operator[](int index) const
    {
        assert(index < Length);
        return c[index];
    }

    inline CUDA_CALLABLE Type WP_THREAD& operator[](int index)
    {
        assert(index < Length);
        return c[index];
    }

    // Mutable component reference with Python-style negative indexing.
    inline CUDA_CALLABLE Type WP_THREAD& component_ref(int index)
    {
        if (index < 0) {
            index += Length;
        }
        assert(index >= 0 && index < (int)Length);
        return c[index];
    }
#if defined(__METAL_VERSION__)
    // Metal member functions are bound to an address space; array elements live in device memory
    // and shared tiles in threadgroup memory, so mirror the accessors for those spaces.
    inline Type operator[](int index) const device { return c[index]; }
    inline Type WP_DEVICE& operator[](int index) device { return c[index]; }
    inline Type operator[](int index) const threadgroup { return c[index]; }
    inline Type threadgroup& operator[](int index) threadgroup { return c[index]; }
    inline Type WP_DEVICE& component_ref(int index) device { return c[index < 0 ? index + (int)Length : index]; }
    inline Type threadgroup& component_ref(int index) threadgroup { return c[index < 0 ? index + (int)Length : index]; }
#endif
};

using vec2b = vec_t<2, int8>;
using vec3b = vec_t<3, int8>;
using vec4b = vec_t<4, int8>;
using vec2ub = vec_t<2, uint8>;
using vec3ub = vec_t<3, uint8>;
using vec4ub = vec_t<4, uint8>;

using vec2s = vec_t<2, int16>;
using vec3s = vec_t<3, int16>;
using vec4s = vec_t<4, int16>;
using vec2us = vec_t<2, uint16>;
using vec3us = vec_t<3, uint16>;
using vec4us = vec_t<4, uint16>;

using vec2i = vec_t<2, int32>;
using vec3i = vec_t<3, int32>;
using vec4i = vec_t<4, int32>;
using vec2ui = vec_t<2, uint32>;
using vec3ui = vec_t<3, uint32>;
using vec4ui = vec_t<4, uint32>;

using vec2l = vec_t<2, int64>;
using vec3l = vec_t<3, int64>;
using vec4l = vec_t<4, int64>;
using vec2ul = vec_t<2, uint64>;
using vec3ul = vec_t<3, uint64>;
using vec4ul = vec_t<4, uint64>;

using vec2h = vec_t<2, half>;
using vec3h = vec_t<3, half>;
using vec4h = vec_t<4, half>;

using vec2 = vec_t<2, float>;
using vec3 = vec_t<3, float>;
using vec4 = vec_t<4, float>;

using vec2f = vec_t<2, float>;
using vec3f = vec_t<3, float>;
using vec4f = vec_t<4, float>;

#if !defined(WP_NO_FLOAT64)
using vec2d = vec_t<2, double>;
using vec3d = vec_t<3, double>;
using vec4d = vec_t<4, double>;
#endif  // !WP_NO_FLOAT64

// Type trait to detect if a type is a vec_t
template <typename T> struct is_vector {
    static WP_CONSTANT constexpr bool value = false;
};

template <unsigned Length, typename Type> struct is_vector<vec_t<Length, Type>> {
    static WP_CONSTANT constexpr bool value = true;
};

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator-(const vec_t<Length, Type> WP_THREAD& x)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = -x[i];
    }

    return ret;
}

template <unsigned Length, typename Type>
CUDA_CALLABLE inline vec_t<Length, Type> pos(const vec_t<Length, Type> WP_THREAD& x)
{
    return x;
}

template <unsigned Length, typename Type>
CUDA_CALLABLE inline vec_t<Length, Type> neg(const vec_t<Length, Type> WP_THREAD& x)
{
    return -x;
}

template <typename Type> CUDA_CALLABLE inline vec_t<3, Type> neg(const vec_t<3, Type> WP_THREAD& x)
{
    return vec_t<3, Type>(-x.c[0], -x.c[1], -x.c[2]);
}

template <typename Type> CUDA_CALLABLE inline vec_t<2, Type> neg(const vec_t<2, Type> WP_THREAD& x)
{
    return vec_t<2, Type>(-x.c[0], -x.c[1]);
}

template <unsigned Length, typename Type>
CUDA_CALLABLE inline void adj_neg(
    const vec_t<Length, Type> WP_THREAD& x,
    vec_t<Length, Type> WP_THREAD& adj_x,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    adj_x -= adj_ret;
}

// equality:
template <unsigned Length, typename Type>
inline CUDA_CALLABLE bool operator==(const vec_t<Length, Type> WP_THREAD& a, const vec_t<Length, Type> WP_THREAD& b)
{
    for (unsigned i = 0; i < Length; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

// scalar multiplication:
template <unsigned Length, typename Type> inline CUDA_CALLABLE vec_t<Length, Type> mul(vec_t<Length, Type> a, Type s)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] * s;
    }
    return ret;
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> mul(vec_t<3, Type> a, Type s)
{
    return vec_t<3, Type>(a.c[0] * s, a.c[1] * s, a.c[2] * s);
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> mul(vec_t<2, Type> a, Type s)
{
    return vec_t<2, Type>(a.c[0] * s, a.c[1] * s);
}

template <unsigned Length, typename Type> inline CUDA_CALLABLE vec_t<Length, Type> mul(Type s, vec_t<Length, Type> a)
{
    return mul(a, s);
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator*(Type s, vec_t<Length, Type> a)
{
    return mul(a, s);
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator*(vec_t<Length, Type> a, Type s)
{
    return mul(a, s);
}


// component wise multiplication:
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> cw_mul(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] * b[i];
    }
    return ret;
}

// division
template <unsigned Length, typename Type> inline CUDA_CALLABLE vec_t<Length, Type> div(vec_t<Length, Type> a, Type s)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = WP_FDIV(a[i], s);
    }
    return ret;
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> div(vec_t<3, Type> a, Type s)
{
    return vec_t<3, Type>(WP_FDIV(a.c[0], s), WP_FDIV(a.c[1], s), WP_FDIV(a.c[2], s));
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> div(vec_t<2, Type> a, Type s)
{
    return vec_t<2, Type>(WP_FDIV(a.c[0], s), WP_FDIV(a.c[1], s));
}

template <unsigned Length, typename Type> inline CUDA_CALLABLE vec_t<Length, Type> div(Type s, vec_t<Length, Type> a)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = WP_FDIV(s, a[i]);
    }
    return ret;
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> div(Type s, vec_t<3, Type> a)
{
    return vec_t<3, Type>(WP_FDIV(s, a.c[0]), WP_FDIV(s, a.c[1]), WP_FDIV(s, a.c[2]));
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> div(Type s, vec_t<2, Type> a)
{
    return vec_t<2, Type>(WP_FDIV(s, a.c[0]), WP_FDIV(s, a.c[1]));
}

// approximate division
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> approx_div(vec_t<Length, Type> a, Type s)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = approx_div(a[i], s);
    }
    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> approx_div(Type s, vec_t<Length, Type> a)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = approx_div(s, a[i]);
    }
    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator/(vec_t<Length, Type> a, Type s)
{
    return div(a, s);
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator/(Type s, vec_t<Length, Type> a)
{
    return div(s, a);
}

// component wise division
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> cw_div(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = WP_FDIV(a[i], b[i]);
    }
    return ret;
}

// addition
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> add(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] + b[i];
    }
    return ret;
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> add(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return vec_t<2, Type>(a.c[0] + b.c[0], a.c[1] + b.c[1]);
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> add(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return vec_t<3, Type>(a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2]);
}

// subtraction
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> sub(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = Type(a[i] - b[i]);
    }
    return ret;
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> sub(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return vec_t<2, Type>(a.c[0] - b.c[0], a.c[1] - b.c[1]);
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> sub(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return vec_t<3, Type>(a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2]);
}

// modulo
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> mod(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = mod(a[i], b[i]);
    }

    return ret;
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> mod(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return vec_t<2, Type>(mod(a.c[0], b.c[0]), mod(a.c[1], b.c[1]));
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> mod(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return vec_t<3, Type>(mod(a.c[0], b.c[0]), mod(a.c[1], b.c[1]), mod(a.c[2], b.c[2]));
}

// bitwise AND
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> bit_and(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = Type(a[i] & b[i]);
    }
    return ret;
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> bit_and(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return vec_t<2, Type>(a.c[0] & b.c[0], a.c[1] & b.c[1]);
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> bit_and(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return vec_t<3, Type>(a.c[0] & b.c[0], a.c[1] & b.c[1], a.c[2] & b.c[2]);
}

// bitwise OR
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> bit_or(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = Type(a[i] | b[i]);
    }
    return ret;
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> bit_or(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return vec_t<2, Type>(a.c[0] | b.c[0], a.c[1] | b.c[1]);
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> bit_or(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return vec_t<3, Type>(a.c[0] | b.c[0], a.c[1] | b.c[1], a.c[2] | b.c[2]);
}

// bitwise XOR
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> bit_xor(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = Type(a[i] ^ b[i]);
    }
    return ret;
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> bit_xor(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return vec_t<2, Type>(a.c[0] ^ b.c[0], a.c[1] ^ b.c[1]);
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> bit_xor(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return vec_t<3, Type>(a.c[0] ^ b.c[0], a.c[1] ^ b.c[1], a.c[2] ^ b.c[2]);
}

// left shift
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> lshift(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = Type(a[i] << b[i]);
    }
    return ret;
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> lshift(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return vec_t<2, Type>(a.c[0] << b.c[0], a.c[1] << b.c[1]);
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> lshift(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return vec_t<3, Type>(a.c[0] << b.c[0], a.c[1] << b.c[1], a.c[2] << b.c[2]);
}

// right shift
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> rshift(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = Type(a[i] >> b[i]);
    }
    return ret;
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> rshift(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return vec_t<2, Type>(a.c[0] >> b.c[0], a.c[1] >> b.c[1]);
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> rshift(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return vec_t<3, Type>(a.c[0] >> b.c[0], a.c[1] >> b.c[1], a.c[2] >> b.c[2]);
}

// invert
template <unsigned Length, typename Type> inline CUDA_CALLABLE vec_t<Length, Type> invert(vec_t<Length, Type> v)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = ~v[i];
    }

    return ret;
}

// dot product:
template <unsigned Length, typename Type> inline CUDA_CALLABLE Type dot(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    Type ret(0);
    for (unsigned i = 0; i < Length; ++i) {
        ret += a[i] * b[i];
    }
    return ret;
}

template <typename Type> inline CUDA_CALLABLE Type dot(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return a.c[0] * b.c[0] + a.c[1] * b.c[1];
}

template <typename Type> inline CUDA_CALLABLE Type dot(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return a.c[0] * b.c[0] + a.c[1] * b.c[1] + a.c[2] * b.c[2];
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE Type tensordot(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    // corresponds to `np.tensordot()` with all axes being contracted
    return dot(a, b);
}


#if defined(__METAL_VERSION__)
// Reads element idx through a chain of static indices. The Metal compiler (GPUCompiler 32023)
// miscompiles inlined functions that scan a by-value vector with a dynamically indexed loop that
// exits on a data-dependent condition and then return the vector (warp/tests/test_vec_scan.py);
// static indices sidestep it.
template <unsigned I, unsigned Length, typename Type>
inline Type vec_get_at(const vec_t<Length, Type> WP_THREAD& v, int idx)
{
    if constexpr (I + 1 < Length) {
        if (idx == (int)I)
            return v.c[I];
        return vec_get_at<I + 1>(v, idx);
    } else {
        return v.c[I];
    }
}
#endif

template <unsigned Length, typename Type>
inline CUDA_CALLABLE Type extract(const vec_t<Length, Type> WP_THREAD& a, int idx)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

#if defined(__METAL_VERSION__)
    return vec_get_at<0>(a, idx);
#else
    return a[idx];
#endif
}

template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<SliceLength, Type> extract(const vec_t<Length, Type> WP_THREAD& a, slice_t slice)
{
    vec_t<SliceLength, Type> ret;

    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = slice.start; is_reversed ? (i > slice.stop) : (i < slice.stop); i += slice.step) {
        ret[ii] = a[i];
        ++ii;
    }

    assert(ii == SliceLength);
    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE Type WP_THREAD* index(vec_t<Length, Type> WP_THREAD& v, int idx)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    return &v[idx];
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE Type WP_THREAD* indexref(vec_t<Length, Type> WP_THREAD* v, int idx)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec store %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    return &((*v)[idx]);
}

#if defined(__METAL__)
// element references into device memory (array elements and struct members)
template <unsigned Length, typename Type> inline Type WP_DEVICE* index(vec_t<Length, Type> WP_DEVICE& v, int idx)
{
    return &v[idx < 0 ? idx + (int)Length : idx];
}

template <unsigned Length, typename Type> inline Type WP_DEVICE* indexref(vec_t<Length, Type> WP_DEVICE* v, int idx)
{
    return &((*v)[idx < 0 ? idx + (int)Length : idx]);
}
#endif

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_index(
    vec_t<Length, Type> WP_THREAD& v,
    int idx,
    vec_t<Length, Type> WP_THREAD& adj_v,
    int adj_idx,
    const Type WP_THREAD& adj_value
)
{
    // nop
}


template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_indexref(
    vec_t<Length, Type> WP_THREAD* v,
    int idx,
    vec_t<Length, Type> WP_THREAD& adj_v,
    int adj_idx,
    const Type WP_THREAD& adj_value
)
{
    // nop
}


template <unsigned Length, typename Type>
inline CUDA_CALLABLE void add_inplace(vec_t<Length, Type> WP_THREAD& v, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    v[idx] += value;
}


template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE void
add_inplace(vec_t<Length, Type> WP_THREAD& v, slice_t slice, const vec_t<SliceLength, Type> WP_THREAD& a)
{
    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = slice.start; is_reversed ? (i > slice.stop) : (i < slice.stop); i += slice.step) {
        v[i] += a[ii];
        ++ii;
    }

    assert(ii == SliceLength);
}


template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    vec_t<Length, Type> WP_THREAD& v,
    int idx,
    Type value,
    vec_t<Length, Type> WP_THREAD& adj_v,
    int adj_idx,
    Type WP_THREAD& adj_value
)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    adj_value += adj_v[idx];
}


template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    const vec_t<Length, Type> WP_THREAD& v,
    slice_t slice,
    const vec_t<SliceLength, Type> WP_THREAD& a,
    vec_t<Length, Type> WP_THREAD& adj_v,
    slice_t WP_THREAD& adj_slice,
    vec_t<SliceLength, Type> WP_THREAD& adj_a
)
{
    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = slice.start; is_reversed ? (i > slice.stop) : (i < slice.stop); i += slice.step) {
        adj_a[ii] += adj_v[i];
        ++ii;
    }

    assert(ii == SliceLength);
}


template <unsigned Length, typename Type>
inline CUDA_CALLABLE void sub_inplace(vec_t<Length, Type> WP_THREAD& v, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    v[idx] -= value;
}


template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE void
sub_inplace(vec_t<Length, Type> WP_THREAD& v, slice_t slice, const vec_t<SliceLength, Type> WP_THREAD& a)
{
    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = slice.start; is_reversed ? (i > slice.stop) : (i < slice.stop); i += slice.step) {
        v[i] -= a[ii];
        ++ii;
    }

    assert(ii == SliceLength);
}


template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    vec_t<Length, Type> WP_THREAD& v,
    int idx,
    Type value,
    vec_t<Length, Type> WP_THREAD& adj_v,
    int adj_idx,
    Type WP_THREAD& adj_value
)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    adj_value -= adj_v[idx];
}


template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    const vec_t<Length, Type> WP_THREAD& v,
    slice_t slice,
    const vec_t<SliceLength, Type> WP_THREAD& a,
    vec_t<Length, Type> WP_THREAD& adj_v,
    slice_t WP_THREAD& adj_slice,
    vec_t<SliceLength, Type> WP_THREAD& adj_a
)
{
    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = slice.start; is_reversed ? (i > slice.stop) : (i < slice.stop); i += slice.step) {
        adj_a[ii] -= adj_v[i];
        ++ii;
    }

    assert(ii == SliceLength);
}


template <unsigned Length, typename Type>
inline CUDA_CALLABLE void bit_and_inplace(vec_t<Length, Type> WP_THREAD& v, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    v[idx] &= value;
}


template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE void
bit_and_inplace(vec_t<Length, Type> WP_THREAD& v, slice_t slice, const vec_t<SliceLength, Type> WP_THREAD& a)
{
    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = slice.start; is_reversed ? (i > slice.stop) : (i < slice.stop); i += slice.step) {
        v[i] &= a[ii];
        ++ii;
    }

    assert(ii == SliceLength);
}


template <unsigned Length, typename Type>
inline CUDA_CALLABLE void bit_or_inplace(vec_t<Length, Type> WP_THREAD& v, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    v[idx] |= value;
}


template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE void
bit_or_inplace(vec_t<Length, Type> WP_THREAD& v, slice_t slice, const vec_t<SliceLength, Type> WP_THREAD& a)
{
    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = slice.start; is_reversed ? (i > slice.stop) : (i < slice.stop); i += slice.step) {
        v[i] |= a[ii];
        ++ii;
    }

    assert(ii == SliceLength);
}


template <unsigned Length, typename Type>
inline CUDA_CALLABLE void bit_xor_inplace(vec_t<Length, Type> WP_THREAD& v, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    v[idx] ^= value;
}


template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE void
bit_xor_inplace(vec_t<Length, Type> WP_THREAD& v, slice_t slice, const vec_t<SliceLength, Type> WP_THREAD& a)
{
    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = slice.start; is_reversed ? (i > slice.stop) : (i < slice.stop); i += slice.step) {
        v[i] ^= a[ii];
        ++ii;
    }

    assert(ii == SliceLength);
}


template <unsigned Length, typename Type>
inline CUDA_CALLABLE void assign_inplace(vec_t<Length, Type> WP_THREAD& v, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    v[idx] = value;
}

template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE void
assign_inplace(vec_t<Length, Type> WP_THREAD& v, slice_t slice, const vec_t<SliceLength, Type> WP_THREAD& a)
{
    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = slice.start; is_reversed ? (i > slice.stop) : (i < slice.stop); i += slice.step) {
        v[i] = a[ii];
        ++ii;
    }

    assert(ii == SliceLength);
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    vec_t<Length, Type> WP_THREAD& v,
    int idx,
    Type value,
    vec_t<Length, Type> WP_THREAD& adj_v,
    int WP_THREAD& adj_idx,
    Type WP_THREAD& adj_value
)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    adj_value += adj_v[idx];
}

template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    const vec_t<Length, Type> WP_THREAD& v,
    slice_t slice,
    const vec_t<SliceLength, Type> WP_THREAD& a,
    vec_t<Length, Type> WP_THREAD& adj_v,
    slice_t WP_THREAD& adj_slice,
    vec_t<SliceLength, Type> WP_THREAD& adj_a
)
{
    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = slice.start; is_reversed ? (i > slice.stop) : (i < slice.stop); i += slice.step) {
        adj_a[ii] += adj_v[i];
        ++ii;
    }

    assert(ii == SliceLength);
}


template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> assign_copy(vec_t<Length, Type> WP_THREAD& v, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    vec_t<Length, Type> ret(v);
    ret[idx] = value;
    return ret;
}

template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type>
assign_copy(vec_t<Length, Type> WP_THREAD& v, slice_t slice, const vec_t<SliceLength, Type> WP_THREAD& a)
{
    vec_t<Length, Type> ret(v);
    assign_inplace<SliceLength>(ret, slice, a);
    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    vec_t<Length, Type> WP_THREAD& v,
    int idx,
    Type value,
    vec_t<Length, Type> WP_THREAD& adj_v,
    int WP_THREAD& adj_idx,
    Type WP_THREAD& adj_value,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    adj_value += adj_ret[idx];
    for (unsigned i = 0; i < Length; ++i) {
        if (i != idx)
            adj_v[i] += adj_ret[i];
    }
}

template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    vec_t<Length, Type> WP_THREAD& v,
    slice_t slice,
    const vec_t<SliceLength, Type> WP_THREAD& a,
    vec_t<Length, Type> WP_THREAD& adj_v,
    slice_t WP_THREAD& adj_slice,
    vec_t<SliceLength, Type> WP_THREAD& adj_a,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = 0; i < Length; ++i) {
        bool in_slice = is_reversed ? (i <= slice.start && i > slice.stop && (slice.start - i) % (-slice.step) == 0)
                                    : (i >= slice.start && i < slice.stop && (i - slice.start) % slice.step == 0);

        if (!in_slice) {
            adj_v[i] += adj_ret[i];
        } else {
            adj_a[ii] += adj_ret[i];
            ++ii;
        }
    }

    assert(ii == SliceLength);
}

template <unsigned Length, typename Type> inline CUDA_CALLABLE Type length(vec_t<Length, Type> a)
{
    return sqrt(dot(a, a));
}

template <unsigned Length, typename Type> inline CUDA_CALLABLE Type length_sq(vec_t<Length, Type> a)
{
    return dot(a, a);
}


template <typename Type> inline CUDA_CALLABLE Type length(vec_t<2, Type> a)
{
    return sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1]);
}

template <typename Type> inline CUDA_CALLABLE Type length(vec_t<3, Type> a)
{
    return sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2]);
}

template <unsigned Length, typename Type> inline CUDA_CALLABLE vec_t<Length, Type> normalize(vec_t<Length, Type> a)
{
    Type l = length(a);
    if (l > Type(kEps))
        return div(a, l);
    else
        return vec_t<Length, Type>();
}

template <typename Type> inline CUDA_CALLABLE vec_t<2, Type> normalize(vec_t<2, Type> a)
{
    Type l = sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1]);
    if (l > Type(kEps))
        return vec_t<2, Type>(WP_FDIV(a.c[0], l), WP_FDIV(a.c[1], l));
    else
        return vec_t<2, Type>();
}

template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> normalize(vec_t<3, Type> a)
{
    Type l = sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2]);
    if (l > Type(kEps))
        return vec_t<3, Type>(WP_FDIV(a.c[0], l), WP_FDIV(a.c[1], l), WP_FDIV(a.c[2], l));
    else
        return vec_t<3, Type>();
}


template <typename Type> inline CUDA_CALLABLE vec_t<3, Type> cross(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return { Type(a[1] * b[2] - a[2] * b[1]), Type(a[2] * b[0] - a[0] * b[2]), Type(a[0] * b[1] - a[1] * b[0]) };
}


template <unsigned Length, typename Type> inline bool CUDA_CALLABLE isfinite(vec_t<Length, Type> x)
{
    for (unsigned i = 0; i < Length; ++i) {
        if (!isfinite(x[i])) {
            return false;
        }
    }
    return true;
}

template <unsigned Length, typename Type> inline bool CUDA_CALLABLE isnan(vec_t<Length, Type> x)
{
    for (unsigned i = 0; i < Length; ++i) {
        if (isnan(x[i])) {
            return true;
        }
    }
    return false;
}

template <unsigned Length, typename Type> inline bool CUDA_CALLABLE isinf(vec_t<Length, Type> x)
{
    for (unsigned i = 0; i < Length; ++i) {
        if (isinf(x[i])) {
            return true;
        }
    }
    return false;
}

// Element-wise vec min/max. Forwards to scalar min/max, which on float types
// uses C fmin/fmax semantics (NaN-as-missing).
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> min(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = min(a[i], b[i]);
    }
    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> max(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = max(a[i], b[i]);
    }
    return ret;
}

// Reduction min/max over a vector. On float types each step uses fmin/fmax,
// so NaN slots are silently skipped; if every element is NaN, the result is
// NaN (equal to v[0]).
template <unsigned Length, typename Type> inline CUDA_CALLABLE Type min(vec_t<Length, Type> v)
{
    Type ret = v[0];
    for (unsigned i = 1; i < Length; ++i) {
        ret = min(ret, v[i]);
    }
    return ret;
}

template <unsigned Length, typename Type> inline CUDA_CALLABLE Type max(vec_t<Length, Type> v)
{
    Type ret = v[0];
    for (unsigned i = 1; i < Length; ++i) {
        ret = max(ret, v[i]);
    }
    return ret;
}

// argmin / argmax skip NaN slots and return the index of the first non-NaN
// extremum. When every element is NaN, return 0 -- matching the forward
// reduction's choice of v[0]. For integer Type the `::isnan(float(...))`
// check is always false, so the loop degrades to the natural argmin/argmax.
template <unsigned Length, typename Type> inline CUDA_CALLABLE unsigned argmin(vec_t<Length, Type> v)
{
    unsigned ret = 0;
    while (ret < Length && ::isnan(float(v[ret])))
        ++ret;
    if (ret == Length)
        return 0;
    for (unsigned i = ret + 1; i < Length; ++i) {
        if (::isnan(float(v[i])))
            continue;
        if (v[i] < v[ret])
            ret = i;
    }
    return ret;
}

template <unsigned Length, typename Type> inline CUDA_CALLABLE unsigned argmax(vec_t<Length, Type> v)
{
    unsigned ret = 0;
    while (ret < Length && ::isnan(float(v[ret])))
        ++ret;
    if (ret == Length)
        return 0;
    for (unsigned i = ret + 1; i < Length; ++i) {
        if (::isnan(float(v[i])))
            continue;
        if (v[i] > v[ret])
            ret = i;
    }
    return ret;
}

template <unsigned Length, typename Type> inline CUDA_CALLABLE vec_t<Length, Type> abs(vec_t<Length, Type> v)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = abs(v[i]);
    }

    return ret;
}

template <unsigned Length, typename Type> inline CUDA_CALLABLE vec_t<Length, Type> sign(vec_t<Length, Type> v)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = v[i] < Type(0) ? Type(-1) : Type(1);
    }

    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_sign(
    const vec_t<Length, Type> WP_THREAD& v,
    vec_t<Length, Type> WP_THREAD& adj_v,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    // MISSINGADJOINT: element-wise subgradient; gradient is zero almost everywhere with subgradient
    // at v[i] = 0
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void expect_near(
    const vec_t<Length, Type> WP_THREAD& actual,
    const vec_t<Length, Type> WP_THREAD& expected,
    const Type WP_THREAD& tolerance
)
{
    Type diff(0);
    for (size_t i = 0; i < Length; ++i) {
        diff = max(diff, abs(actual[i] - expected[i]));
    }
    if (diff > tolerance) {
        printf("Error, expect_near() failed with tolerance ");
        print(tolerance);
        printf("    Expected: ");
        print(expected);
        printf("    Actual: ");
        print(actual);
        printf("    Max absolute difference: ");
        print(diff);
    }
}

// adjoint for the initializer_array constructor:
template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_vec_t(
    const initializer_array<Length, Type> WP_THREAD& cmps,
    const initializer_array<Length, Type WP_THREAD*> WP_THREAD& adj_cmps,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    for (unsigned i = 0; i < Length; ++i) {
        *(adj_cmps[i]) += adj_ret[i];
    }
}


// adjoint for the component constructors:
template <typename Type>
inline CUDA_CALLABLE void adj_vec_t(
    Type cmpx, Type cmpy, Type WP_THREAD& adj_cmpx, Type WP_THREAD& adj_cmpy, const vec_t<2, Type> WP_THREAD& adj_ret
)
{
    adj_cmpx += adj_ret.c[0];
    adj_cmpy += adj_ret.c[1];
}

template <typename Type>
inline CUDA_CALLABLE void adj_vec_t(
    Type cmpx,
    Type cmpy,
    Type cmpz,
    Type WP_THREAD& adj_cmpx,
    Type WP_THREAD& adj_cmpy,
    Type WP_THREAD& adj_cmpz,
    const vec_t<3, Type> WP_THREAD& adj_ret
)
{
    adj_cmpx += adj_ret.c[0];
    adj_cmpy += adj_ret.c[1];
    adj_cmpz += adj_ret.c[2];
}

template <typename Type>
inline CUDA_CALLABLE void adj_vec_t(
    Type cmpx,
    Type cmpy,
    Type cmpz,
    Type cmpw,
    Type WP_THREAD& adj_cmpx,
    Type WP_THREAD& adj_cmpy,
    Type WP_THREAD& adj_cmpz,
    Type WP_THREAD& adj_cmpw,
    const vec_t<4, Type> WP_THREAD& adj_ret
)
{
    adj_cmpx += adj_ret.c[0];
    adj_cmpy += adj_ret.c[1];
    adj_cmpz += adj_ret.c[2];
    adj_cmpw += adj_ret.c[3];
}

// adjoint for the constant constructor:
template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_vec_t(Type s, Type WP_THREAD& adj_s, const vec_t<Length, Type> WP_THREAD& adj_ret)
{
    for (unsigned i = 0; i < Length; ++i) {
        adj_s += adj_ret[i];
    }
}

// adjoint for the casting constructor
template <unsigned Length, typename Type, typename OtherType>
inline CUDA_CALLABLE void adj_vec_t(
    const vec_t<Length, OtherType> WP_THREAD& other,
    vec_t<Length, OtherType> WP_THREAD& adj_other,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    for (unsigned i = 0; i < Length; ++i) {
        adj_other[i] += static_cast<OtherType>(adj_ret[i]);
    }
}

template <typename Type>
CUDA_CALLABLE inline void adj_vec_t(
    const vec_t<3, Type> WP_THREAD& w,
    const vec_t<3, Type> WP_THREAD& v,
    vec_t<3, Type> WP_THREAD& adj_w,
    vec_t<3, Type> WP_THREAD& adj_v,
    const vec_t<6, Type> WP_THREAD& adj_ret
)
{
    adj_w[0] += adj_ret[0];
    adj_w[1] += adj_ret[1];
    adj_w[2] += adj_ret[2];
    adj_v[0] += adj_ret[3];
    adj_v[1] += adj_ret[4];
    adj_v[2] += adj_ret[5];
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_mul(
    vec_t<Length, Type> a,
    Type s,
    vec_t<Length, Type> WP_THREAD& adj_a,
    Type WP_THREAD& adj_s,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    for (unsigned i = 0; i < Length; ++i) {
        adj_a[i] += s * adj_ret[i];
    }

    adj_s += dot(a, adj_ret);

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret)) {
        // \TODO: How shall we implement this error message?
        // printf("adj_mul((%f %f %f %f), %f, (%f %f %f %f), %f, (%f %f %f %f)\n", a.x, a.y, a.z, a.w, s, adj_a.x,
        // adj_a.y, adj_a.z, adj_a.w, adj_s, adj_ret.x, adj_ret.y, adj_ret.z, adj_ret.w);
        assert(0);
    }
#endif
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_mul(
    Type s,
    vec_t<Length, Type> a,
    Type WP_THREAD& adj_s,
    vec_t<Length, Type> WP_THREAD& adj_a,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    adj_mul(a, s, adj_a, adj_s, adj_ret);
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_cw_mul(
    vec_t<Length, Type> a,
    vec_t<Length, Type> b,
    vec_t<Length, Type> WP_THREAD& adj_a,
    vec_t<Length, Type> WP_THREAD& adj_b,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    adj_a += cw_mul(b, adj_ret);
    adj_b += cw_mul(a, adj_ret);
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_div(
    vec_t<Length, Type> a,
    Type s,
    vec_t<Length, Type> WP_THREAD& adj_a,
    Type WP_THREAD& adj_s,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{

    adj_s -= dot(a, adj_ret) / (s * s);  // - a / s^2

    for (unsigned i = 0; i < Length; ++i) {
        adj_a[i] += adj_ret[i] / s;
    }

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret)) {
        // \TODO: How shall we implement this error message?
        // printf("adj_div((%f %f %f %f), %f, (%f %f %f %f), %f, (%f %f %f %f)\n", a.x, a.y, a.z, a.w, s, adj_a.x,
        // adj_a.y, adj_a.z, adj_a.w, adj_s, adj_ret.x, adj_ret.y, adj_ret.z, adj_ret.w);
        assert(0);
    }
#endif
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_div(
    Type s,
    vec_t<Length, Type> a,
    Type WP_THREAD& adj_s,
    vec_t<Length, Type> WP_THREAD& adj_a,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{

    for (unsigned i = 0; i < Length; ++i) {
        Type inv = Type(1) / a[i];
        adj_a[i] -= s * adj_ret[i] * inv * inv;
        adj_s += adj_ret[i] * inv;
    }

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret)) {
        // \TODO: How shall we implement this error message?
        // printf("adj_div((%f %f %f %f), %f, (%f %f %f %f), %f, (%f %f %f %f)\n", a.x, a.y, a.z, a.w, s, adj_a.x,
        // adj_a.y, adj_a.z, adj_a.w, adj_s, adj_ret.x, adj_ret.y, adj_ret.z, adj_ret.w);
        assert(0);
    }
#endif
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_approx_div(
    vec_t<Length, Type> a,
    Type s,
    vec_t<Length, Type> WP_THREAD& adj_a,
    Type WP_THREAD& adj_s,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    adj_s -= approx_div(dot(a, adj_ret), (s * s));

    for (unsigned i = 0; i < Length; ++i) {
        adj_a[i] += approx_div(adj_ret[i], s);
    }
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_approx_div(
    Type s,
    vec_t<Length, Type> a,
    Type WP_THREAD& adj_s,
    vec_t<Length, Type> WP_THREAD& adj_a,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    for (unsigned i = 0; i < Length; ++i) {
        Type inv = approx_rcp(a[i]);
        adj_a[i] -= s * adj_ret[i] * inv * inv;
        adj_s += adj_ret[i] * inv;
    }
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_cw_div(
    vec_t<Length, Type> a,
    vec_t<Length, Type> b,
    vec_t<Length, Type> WP_THREAD& ret,
    vec_t<Length, Type> WP_THREAD& adj_a,
    vec_t<Length, Type> WP_THREAD& adj_b,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    adj_a += cw_div(adj_ret, b);
    adj_b -= cw_mul(adj_ret, cw_div(ret, b));
}

// 5-arg overloads for tile operations (vec/scalar mixed types).
// These forward to the existing 5-arg adj_div for vec/scalar which doesn't need ret.
// The 6-arg adj_cw_div above is for vec/vec which needs ret for the gradient computation.
template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_cw_div(
    vec_t<Length, Type> a,
    Type b,
    vec_t<Length, Type> WP_THREAD& adj_a,
    Type WP_THREAD& adj_b,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    adj_div(a, b, adj_a, adj_b, adj_ret);
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_cw_div(
    Type a,
    vec_t<Length, Type> b,
    Type WP_THREAD& adj_a,
    vec_t<Length, Type> WP_THREAD& adj_b,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    adj_div(a, b, adj_a, adj_b, adj_ret);
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_add(
    vec_t<Length, Type> a,
    vec_t<Length, Type> b,
    vec_t<Length, Type> WP_THREAD& adj_a,
    vec_t<Length, Type> WP_THREAD& adj_b,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    adj_a += adj_ret;
    adj_b += adj_ret;
}

template <typename Type>
inline CUDA_CALLABLE void adj_add(
    vec_t<2, Type> a,
    vec_t<2, Type> b,
    vec_t<2, Type> WP_THREAD& adj_a,
    vec_t<2, Type> WP_THREAD& adj_b,
    const vec_t<2, Type> WP_THREAD& adj_ret
)
{
    adj_a.c[0] += adj_ret.c[0];
    adj_a.c[1] += adj_ret.c[1];
    adj_b.c[0] += adj_ret.c[0];
    adj_b.c[1] += adj_ret.c[1];
}

template <typename Type>
inline CUDA_CALLABLE void adj_add(
    vec_t<3, Type> a,
    vec_t<3, Type> b,
    vec_t<3, Type> WP_THREAD& adj_a,
    vec_t<3, Type> WP_THREAD& adj_b,
    const vec_t<3, Type> WP_THREAD& adj_ret
)
{
    adj_a.c[0] += adj_ret.c[0];
    adj_a.c[1] += adj_ret.c[1];
    adj_a.c[2] += adj_ret.c[2];
    adj_b.c[0] += adj_ret.c[0];
    adj_b.c[1] += adj_ret.c[1];
    adj_b.c[2] += adj_ret.c[2];
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_sub(
    vec_t<Length, Type> a,
    vec_t<Length, Type> b,
    vec_t<Length, Type> WP_THREAD& adj_a,
    vec_t<Length, Type> WP_THREAD& adj_b,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    adj_a += adj_ret;
    adj_b -= adj_ret;
}

template <typename Type>
inline CUDA_CALLABLE void adj_sub(
    vec_t<2, Type> a,
    vec_t<2, Type> b,
    vec_t<2, Type> WP_THREAD& adj_a,
    vec_t<2, Type> WP_THREAD& adj_b,
    const vec_t<2, Type> WP_THREAD& adj_ret
)
{
    adj_a.c[0] += adj_ret.c[0];
    adj_a.c[1] += adj_ret.c[1];
    adj_b.c[0] -= adj_ret.c[0];
    adj_b.c[1] -= adj_ret.c[1];
}

template <typename Type>
inline CUDA_CALLABLE void adj_sub(
    vec_t<3, Type> a,
    vec_t<3, Type> b,
    vec_t<3, Type> WP_THREAD& adj_a,
    vec_t<3, Type> WP_THREAD& adj_b,
    const vec_t<3, Type> WP_THREAD& adj_ret
)
{
    adj_a.c[0] += adj_ret.c[0];
    adj_a.c[1] += adj_ret.c[1];
    adj_a.c[2] += adj_ret.c[2];
    adj_b.c[0] -= adj_ret.c[0];
    adj_b.c[1] -= adj_ret.c[1];
    adj_b.c[2] -= adj_ret.c[2];
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_mod(
    vec_t<Length, Type> a,
    vec_t<Length, Type> b,
    vec_t<Length, Type> WP_THREAD& adj_a,
    vec_t<Length, Type> WP_THREAD& adj_b,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    // MISSINGADJOINT: element-wise: adj_a[i] += adj_ret[i]; adj_b[i] -= floor(a[i] / b[i]) *
    // adj_ret[i]
}

template <typename Type>
inline CUDA_CALLABLE void adj_mod(
    vec_t<2, Type> a,
    vec_t<2, Type> b,
    vec_t<2, Type> WP_THREAD& adj_a,
    vec_t<2, Type> WP_THREAD& adj_b,
    const vec_t<2, Type> WP_THREAD& adj_ret
)
{
    // MISSINGADJOINT: element-wise: adj_a[i] += adj_ret[i]; adj_b[i] -= floor(a[i] / b[i]) *
    // adj_ret[i]
}

template <typename Type>
inline CUDA_CALLABLE void adj_mod(
    vec_t<3, Type> a,
    vec_t<3, Type> b,
    vec_t<3, Type> WP_THREAD& adj_a,
    vec_t<3, Type> WP_THREAD& adj_b,
    const vec_t<3, Type> WP_THREAD& adj_ret
)
{
    // MISSINGADJOINT: element-wise: adj_a[i] += adj_ret[i]; adj_b[i] -= floor(a[i] / b[i]) *
    // adj_ret[i]
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_dot(
    vec_t<Length, Type> a,
    vec_t<Length, Type> b,
    vec_t<Length, Type> WP_THREAD& adj_a,
    vec_t<Length, Type> WP_THREAD& adj_b,
    const Type adj_ret
)
{
    adj_a += b * adj_ret;
    adj_b += a * adj_ret;

#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || !isfinite(adj_a) || !isfinite(adj_b) || !isfinite(adj_ret)) {
        // \TODO: How shall we implement this error message?
        // printf("adj_dot((%f %f %f %f), (%f %f %f %f), (%f %f %f %f), (%f %f %f %f), %f)\n", a.x, a.y, a.z, a.w, b.x,
        // b.y, b.z, b.w, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_b.x, adj_b.y, adj_b.z, adj_b.w, adj_ret);
        assert(0);
    }
#endif
}


template <typename Type>
inline CUDA_CALLABLE void adj_dot(
    vec_t<2, Type> a,
    vec_t<2, Type> b,
    vec_t<2, Type> WP_THREAD& adj_a,
    vec_t<2, Type> WP_THREAD& adj_b,
    const Type adj_ret
)
{
    adj_a.c[0] += b.c[0] * adj_ret;
    adj_a.c[1] += b.c[1] * adj_ret;

    adj_b.c[0] += a.c[0] * adj_ret;
    adj_b.c[1] += a.c[1] * adj_ret;
}

template <typename Type>
inline CUDA_CALLABLE void adj_dot(
    vec_t<3, Type> a,
    vec_t<3, Type> b,
    vec_t<3, Type> WP_THREAD& adj_a,
    vec_t<3, Type> WP_THREAD& adj_b,
    const Type adj_ret
)
{
    adj_a.c[0] += b.c[0] * adj_ret;
    adj_a.c[1] += b.c[1] * adj_ret;
    adj_a.c[2] += b.c[2] * adj_ret;

    adj_b.c[0] += a.c[0] * adj_ret;
    adj_b.c[1] += a.c[1] * adj_ret;
    adj_b.c[2] += a.c[2] * adj_ret;
}


template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_extract(
    const vec_t<Length, Type> WP_THREAD& a,
    int idx,
    vec_t<Length, Type> WP_THREAD& adj_a,
    int adj_idx,
    Type WP_THREAD& adj_ret
)
{
#ifndef NDEBUG
    if (idx < -(int)Length || idx >= (int)Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0) {
        idx += Length;
    }

    adj_a[idx] += adj_ret;
}

template <unsigned SliceLength, unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_extract(
    const vec_t<Length, Type> WP_THREAD& a,
    slice_t slice,
    vec_t<Length, Type> WP_THREAD& adj_a,
    slice_t WP_THREAD& adj_slice,
    const vec_t<SliceLength, Type> WP_THREAD& adj_ret
)
{
    assert(slice.start >= 0 && slice.start <= (int)Length);
    assert(slice.stop >= -1 && slice.stop <= (int)Length);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = slice.start; is_reversed ? (i > slice.stop) : (i < slice.stop); i += slice.step) {
        adj_a[i] += adj_ret[ii];
        ++ii;
    }

    assert(ii == SliceLength);
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void
adj_length(vec_t<Length, Type> a, Type ret, vec_t<Length, Type> WP_THREAD& adj_a, const Type adj_ret)
{
    if (ret > Type(kEps)) {
        adj_a += div(a, ret) * adj_ret;
    }

#if FP_CHECK
    if (!isfinite(adj_a)) {
        // \TODO: How shall we implement this error message?
        // printf("%s:%d - adj_length((%f %f %f %f), (%f %f %f %f), (%f))\n", __FILE__, __LINE__, a.x, a.y, a.z, a.w,
        // adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_ret);
        assert(0);
    }
#endif
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_length_sq(vec_t<Length, Type> a, vec_t<Length, Type> WP_THREAD& adj_a, const Type adj_ret)
{
    adj_a += Type(2.0) * a * adj_ret;

#if FP_CHECK
    if (!isfinite(adj_a)) {
        // \TODO: How shall we implement this error message?
        // printf("%s:%d - adj_length((%f %f %f %f), (%f %f %f %f), (%f))\n", __FILE__, __LINE__, a.x, a.y, a.z, a.w,
        // adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_ret);
        assert(0);
    }
#endif
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_normalize(
    vec_t<Length, Type> a,
    vec_t<Length, Type> WP_THREAD& ret,
    vec_t<Length, Type> WP_THREAD& adj_a,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    Type d = length(a);

    if (d > Type(kEps)) {
        Type invd = Type(1.0f) / d;

        adj_a += (adj_ret * invd - ret * (dot(ret, adj_ret)) * invd);

#if FP_CHECK
        if (!isfinite(adj_a)) {
            // \TODO: How shall we implement this error message?
            // printf("%s:%d - adj_normalize((%f %f %f %f), (%f %f %f %f), (%f, %f, %f, %f))\n", __FILE__, __LINE__,
            // a.x, a.y, a.z, a.w, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_ret.x, adj_ret.y, adj_ret.z, adj_ret.w);
            assert(0);
        }
#endif
    }
}

template <typename Type>
inline CUDA_CALLABLE void adj_cross(
    vec_t<3, Type> a,
    vec_t<3, Type> b,
    vec_t<3, Type> WP_THREAD& adj_a,
    vec_t<3, Type> WP_THREAD& adj_b,
    const vec_t<3, Type> WP_THREAD& adj_ret
)
{
    // todo: sign check
    adj_a += cross(b, adj_ret);
    adj_b -= cross(a, adj_ret);
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_min(
    const vec_t<Length, Type> WP_THREAD& a,
    const vec_t<Length, Type> WP_THREAD& b,
    vec_t<Length, Type> WP_THREAD& adj_a,
    vec_t<Length, Type> WP_THREAD& adj_b,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    for (unsigned i = 0; i < Length; ++i) {
        adj_min(a[i], b[i], adj_a[i], adj_b[i], adj_ret[i]);
    }
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_max(
    const vec_t<Length, Type> WP_THREAD& a,
    const vec_t<Length, Type> WP_THREAD& b,
    vec_t<Length, Type> WP_THREAD& adj_a,
    vec_t<Length, Type> WP_THREAD& adj_b,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    for (unsigned i = 0; i < Length; ++i) {
        adj_max(a[i], b[i], adj_a[i], adj_b[i], adj_ret[i]);
    }
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void
adj_min(const vec_t<Length, Type> WP_THREAD& v, vec_t<Length, Type> WP_THREAD& adj_v, const Type WP_THREAD& adj_ret)
{
    unsigned i = argmin(v);
    adj_v[i] += adj_ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void
adj_max(const vec_t<Length, Type> WP_THREAD& v, vec_t<Length, Type> WP_THREAD& adj_v, const Type WP_THREAD& adj_ret)
{
    unsigned i = argmax(v);
    adj_v[i] += adj_ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_abs(
    const vec_t<Length, Type> WP_THREAD& v,
    vec_t<Length, Type> WP_THREAD& adj_v,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    for (unsigned i = 0; i < Length; ++i) {
        if (v[i] < Type(0)) {
            adj_v[i] -= adj_ret[i];
        } else {
            adj_v[i] += adj_ret[i];
        }
    }
}

// Do I need to specialize these for different lengths?
template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_add(vec_t<Length, Type> WP_DEVICE* addr, vec_t<Length, Type> value)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_add(&(addr->c[i]), value[i]);
    }

    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_min(vec_t<Length, Type> WP_DEVICE* addr, vec_t<Length, Type> value)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_min(&(addr->c[i]), value[i]);
    }

    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_max(vec_t<Length, Type> WP_DEVICE* addr, vec_t<Length, Type> value)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_max(&(addr->c[i]), value[i]);
    }

    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_and(vec_t<Length, Type> WP_DEVICE* addr, vec_t<Length, Type> value)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_and(&(addr->c[i]), value[i]);
    }

    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_or(vec_t<Length, Type> WP_DEVICE* addr, vec_t<Length, Type> value)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_or(&(addr->c[i]), value[i]);
    }

    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_xor(vec_t<Length, Type> WP_DEVICE* addr, vec_t<Length, Type> value)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_xor(&(addr->c[i]), value[i]);
    }

    return ret;
}

template <unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_atomic_minmax(
    vec_t<Length, Type> WP_DEVICE* addr,
    vec_t<Length, Type> WP_DEVICE* adj_addr,
    const vec_t<Length, Type> WP_THREAD& value,
    vec_t<Length, Type> WP_THREAD& adj_value
)
{
    for (unsigned i = 0; i < Length; ++i)
        adj_atomic_minmax(&(addr->c[i]), &(adj_addr->c[i]), value[i], adj_value[i]);
}

// ok, the original implementation of this didn't take the absolute values.
// I wouldn't consider this expected behavior. It looks like it's only
// being used for bounding boxes at the moment, where this doesn't matter,
// but you often use it for ray tracing where it does. Not sure if the
// fabs() incurs a performance hit...
template <unsigned Length, typename Type> CUDA_CALLABLE inline int longest_axis(const vec_t<Length, Type> WP_THREAD& v)
{
    Type lmax = abs(v[0]);
    int ret(0);
    for (unsigned i = 1; i < Length; ++i) {
        Type l = abs(v[i]);
        if (l > lmax) {
            ret = i;
            lmax = l;
        }
    }
    return ret;
}

template <unsigned Length, typename Type>
CUDA_CALLABLE inline vec_t<Length, Type>
lerp(const vec_t<Length, Type> WP_THREAD& a, const vec_t<Length, Type> WP_THREAD& b, Type t)
{
    return a * (Type(1) - t) + b * t;
}

template <unsigned Length, typename Type>
CUDA_CALLABLE inline void adj_lerp(
    const vec_t<Length, Type> WP_THREAD& a,
    const vec_t<Length, Type> WP_THREAD& b,
    Type t,
    vec_t<Length, Type> WP_THREAD& adj_a,
    vec_t<Length, Type> WP_THREAD& adj_b,
    Type WP_THREAD& adj_t,
    const vec_t<Length, Type> WP_THREAD& adj_ret
)
{
    adj_a += adj_ret * (Type(1) - t);
    adj_b += adj_ret * t;
    adj_t += tensordot(b, adj_ret) - tensordot(a, adj_ret);
}

// for integral types we do not accumulate gradients
template <unsigned Length>
CUDA_CALLABLE inline void adj_atomic_add(vec_t<Length, int8> WP_DEVICE* buf, const vec_t<Length, int8> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_add(vec_t<Length, uint8> WP_DEVICE* buf, const vec_t<Length, uint8> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_add(vec_t<Length, int16> WP_DEVICE* buf, const vec_t<Length, int16> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_add(vec_t<Length, uint16> WP_DEVICE* buf, const vec_t<Length, uint16> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_add(vec_t<Length, int32> WP_DEVICE* buf, const vec_t<Length, int32> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_add(vec_t<Length, uint32> WP_DEVICE* buf, const vec_t<Length, uint32> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_add(vec_t<Length, int64> WP_DEVICE* buf, const vec_t<Length, int64> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_add(vec_t<Length, uint64> WP_DEVICE* buf, const vec_t<Length, uint64> WP_THREAD& value)
{
}

// for bitwise operations we do not accumulate gradients
template <unsigned Length>
CUDA_CALLABLE inline void adj_atomic_and(vec_t<Length, int8> WP_DEVICE* buf, const vec_t<Length, int8> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_and(vec_t<Length, uint8> WP_DEVICE* buf, const vec_t<Length, uint8> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_and(vec_t<Length, int16> WP_DEVICE* buf, const vec_t<Length, int16> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_and(vec_t<Length, uint16> WP_DEVICE* buf, const vec_t<Length, uint16> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_and(vec_t<Length, int32> WP_DEVICE* buf, const vec_t<Length, int32> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_and(vec_t<Length, uint32> WP_DEVICE* buf, const vec_t<Length, uint32> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_and(vec_t<Length, int64> WP_DEVICE* buf, const vec_t<Length, int64> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_and(vec_t<Length, uint64> WP_DEVICE* buf, const vec_t<Length, uint64> WP_THREAD& value)
{
}

template <unsigned Length>
CUDA_CALLABLE inline void adj_atomic_or(vec_t<Length, int8> WP_DEVICE* buf, const vec_t<Length, int8> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_or(vec_t<Length, uint8> WP_DEVICE* buf, const vec_t<Length, uint8> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_or(vec_t<Length, int16> WP_DEVICE* buf, const vec_t<Length, int16> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_or(vec_t<Length, uint16> WP_DEVICE* buf, const vec_t<Length, uint16> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_or(vec_t<Length, int32> WP_DEVICE* buf, const vec_t<Length, int32> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_or(vec_t<Length, uint32> WP_DEVICE* buf, const vec_t<Length, uint32> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_or(vec_t<Length, int64> WP_DEVICE* buf, const vec_t<Length, int64> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_or(vec_t<Length, uint64> WP_DEVICE* buf, const vec_t<Length, uint64> WP_THREAD& value)
{
}

template <unsigned Length>
CUDA_CALLABLE inline void adj_atomic_xor(vec_t<Length, int8> WP_DEVICE* buf, const vec_t<Length, int8> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_xor(vec_t<Length, uint8> WP_DEVICE* buf, const vec_t<Length, uint8> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_xor(vec_t<Length, int16> WP_DEVICE* buf, const vec_t<Length, int16> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_xor(vec_t<Length, uint16> WP_DEVICE* buf, const vec_t<Length, uint16> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_xor(vec_t<Length, int32> WP_DEVICE* buf, const vec_t<Length, int32> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_xor(vec_t<Length, uint32> WP_DEVICE* buf, const vec_t<Length, uint32> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_xor(vec_t<Length, int64> WP_DEVICE* buf, const vec_t<Length, int64> WP_THREAD& value)
{
}
template <unsigned Length>
CUDA_CALLABLE inline void
adj_atomic_xor(vec_t<Length, uint64> WP_DEVICE* buf, const vec_t<Length, uint64> WP_THREAD& value)
{
}


// adjoints for some of the constructors, used in intersect.h
inline CUDA_CALLABLE void
adj_vec2(float x, float y, float WP_THREAD& adj_x, float WP_THREAD& adj_y, const vec2 WP_THREAD& adj_ret)
{
    adj_x += adj_ret[0];
    adj_y += adj_ret[1];
}

inline CUDA_CALLABLE void adj_vec3(
    float x,
    float y,
    float z,
    float WP_THREAD& adj_x,
    float WP_THREAD& adj_y,
    float WP_THREAD& adj_z,
    const vec3 WP_THREAD& adj_ret
)
{
    adj_x += adj_ret[0];
    adj_y += adj_ret[1];
    adj_z += adj_ret[2];
}

inline CUDA_CALLABLE void adj_vec4(
    float x,
    float y,
    float z,
    float w,
    float WP_THREAD& adj_x,
    float WP_THREAD& adj_y,
    float WP_THREAD& adj_z,
    float WP_THREAD& adj_w,
    const vec4 WP_THREAD& adj_ret
)
{
    adj_x += adj_ret[0];
    adj_y += adj_ret[1];
    adj_z += adj_ret[2];
    adj_w += adj_ret[3];
}

inline CUDA_CALLABLE void adj_vec3(float s, float WP_THREAD& adj_s, const vec3 WP_THREAD& adj_ret)
{
    adj_vec_t(s, adj_s, adj_ret);
}

inline CUDA_CALLABLE void adj_vec4(float s, float WP_THREAD& adj_s, const vec4 WP_THREAD& adj_ret)
{
    adj_vec_t(s, adj_s, adj_ret);
}

template <unsigned Length, typename Type> CUDA_CALLABLE inline int len(const vec_t<Length, Type> WP_THREAD& x)
{
    return Length;
}

}  // namespace wp
