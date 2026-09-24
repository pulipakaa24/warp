// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

#if defined(__METAL_VERSION__)
// Struct values loaded from device memory carry host pointers in their array members; generated structs with
// such members get a global overload (codegen_struct) and wp::load() calls it. Everything else is a no-op.
template <typename T> inline void wp_metal_fixup(T WP_THREAD&) { }
#endif

namespace wp {

#if FP_CHECK

#define FP_ASSERT_FWD(value) \
    print(value); \
    printf(")\n"); \
    assert(0);

#define FP_ASSERT_ADJ(value, adj_value) \
    print(value); \
    printf(", "); \
    print(adj_value); \
    printf(")\n"); \
    assert(0);

#define FP_VERIFY_FWD(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(addr", __FILE__, __LINE__, __FUNCTION__); \
        FP_ASSERT_FWD(value) \
    }

#define FP_VERIFY_FWD_1(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d) ", __FILE__, __LINE__, __FUNCTION__, i); \
        FP_ASSERT_FWD(value) \
    }

#define FP_VERIFY_FWD_2(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j); \
        FP_ASSERT_FWD(value) \
    }

#define FP_VERIFY_FWD_3(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k); \
        FP_ASSERT_FWD(value) \
    }

#define FP_VERIFY_FWD_4(value) \
    if (!isfinite(value)) { \
        printf("%s:%d - %s(arr, %d, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k, l); \
        FP_ASSERT_FWD(value) \
    }

#define FP_VERIFY_ADJ(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(addr",  __FILE__, __LINE__, __FUNCTION__); \
        FP_ASSERT_ADJ(value, adj_value); \
    }

#define FP_VERIFY_ADJ_1(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d) ",  __FILE__, __LINE__, __FUNCTION__, i); \
        FP_ASSERT_ADJ(value, adj_value); \
    }

#define FP_VERIFY_ADJ_2(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d, %d) ",  __FILE__, __LINE__, __FUNCTION__, i, j); \
        FP_ASSERT_ADJ(value, adj_value); \
    }

#define FP_VERIFY_ADJ_3(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k); \
        FP_ASSERT_ADJ(value, adj_value); \
    }

#define FP_VERIFY_ADJ_4(value, adj_value) \
    if (!isfinite(value) || !isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, %d, %d, %d, %d) ", __FILE__, __LINE__, __FUNCTION__, i, j, k, l); \
        FP_ASSERT_ADJ(value, adj_value); \
    }

// Slot-only variant: forward ``value`` is not captured by the slot-level
// adjoint path, so only ``adj_value`` is checked.
#define FP_VERIFY_ADJ_SLOT(adj_value) \
    if (!isfinite(adj_value)) \
    { \
        printf("%s:%d - %s(arr, ...) ",  __FILE__, __LINE__, __FUNCTION__); \
        print(adj_value); \
        printf(")\n"); \
        assert(0); \
    }


#else

#define FP_VERIFY_FWD(value) {}
#define FP_VERIFY_FWD_1(value) {}
#define FP_VERIFY_FWD_2(value) {}
#define FP_VERIFY_FWD_3(value) {}
#define FP_VERIFY_FWD_4(value) {}

#define FP_VERIFY_ADJ(value, adj_value) {}
#define FP_VERIFY_ADJ_1(value, adj_value) {}
#define FP_VERIFY_ADJ_2(value, adj_value) {}
#define FP_VERIFY_ADJ_3(value, adj_value) {}
#define FP_VERIFY_ADJ_4(value, adj_value) {}
#define FP_VERIFY_ADJ_SLOT(adj_value) {}

#endif  // WP_FP_CHECK


template <size_t... Is> struct index_sequence { };

// Written without inheritance because Metal does not support derived classes.
template <size_t N> struct make_index_sequence_impl;
template <> struct make_index_sequence_impl<0> {
    using type = index_sequence<>;
};
template <> struct make_index_sequence_impl<1> {
    using type = index_sequence<0>;
};
template <> struct make_index_sequence_impl<2> {
    using type = index_sequence<0, 1>;
};
template <> struct make_index_sequence_impl<3> {
    using type = index_sequence<0, 1, 2>;
};
template <> struct make_index_sequence_impl<4> {
    using type = index_sequence<0, 1, 2, 3>;
};

template <size_t N> using make_index_sequence = typename make_index_sequence_impl<N>::type;


static WP_CONSTANT const int ARRAY_MAX_DIMS = 4;  // must match constant in types.py

// must match constants in types.py
static WP_CONSTANT const int ARRAY_TYPE_REGULAR = 0;
static WP_CONSTANT const int ARRAY_TYPE_INDEXED = 1;
static WP_CONSTANT const int ARRAY_TYPE_FABRIC = 2;
static WP_CONSTANT const int ARRAY_TYPE_FABRIC_INDEXED = 3;

static WP_CONSTANT constexpr uint16_t ARRAY_FLAG_RETAIN_GRAD = 1 << 0;

struct shape_t {
    int dims[ARRAY_MAX_DIMS];

    CUDA_CALLABLE inline shape_t()
        : dims()
    {
    }

    CUDA_CALLABLE inline int operator[](int i) const
    {
        assert(i < ARRAY_MAX_DIMS);
        return dims[i];
    }

    CUDA_CALLABLE inline int WP_THREAD& operator[](int i)
    {
        assert(i < ARRAY_MAX_DIMS);
        return dims[i];
    }
};

CUDA_CALLABLE inline int extract(const shape_t WP_THREAD& s, int i) { return s.dims[i]; }

inline CUDA_CALLABLE void print(shape_t s)
{
    // todo: only print valid dims, currently shape has a fixed size
    // but we don't know how many dims are valid (e.g.: 1d, 2d, etc)
    // should probably store ndim with shape
    printf("(%d, %d, %d, %d)\n", s.dims[0], s.dims[1], s.dims[2], s.dims[3]);
}
inline CUDA_CALLABLE void adj_print(shape_t s, shape_t WP_THREAD& adj_s) { /* nop: shape_t has no gradient */ }


#if defined(__METAL_VERSION__)
#define WP_ARRAY_PTR(x) WP_METAL_TRANSLATE(x)  // raw addresses come from the host (array.ptr)
#else
#define WP_ARRAY_PTR(x) (x)
#endif

template <typename T> struct array_t {
    CUDA_CALLABLE inline array_t()
        : data(nullptr)
        , grad(nullptr)
        , shape()
        , strides()
        , ndim(0)
        , flags(0)
    {
    }

    CUDA_CALLABLE array_t(T WP_DEVICE* data, int size, T WP_DEVICE* grad = nullptr)
        : data(data)
        , grad(grad)
    {
        // constructor for 1d array
        shape.dims[0] = size;
        shape.dims[1] = 0;
        shape.dims[2] = 0;
        shape.dims[3] = 0;
        ndim = 1;
        flags = 0;
        strides[0] = sizeof(T);
        strides[1] = 0;
        strides[2] = 0;
        strides[3] = 0;
    }
    CUDA_CALLABLE array_t(T WP_DEVICE* data, int dim0, int dim1, T WP_DEVICE* grad = nullptr)
        : data(data)
        , grad(grad)
    {
        // constructor for 2d array
        shape.dims[0] = dim0;
        shape.dims[1] = dim1;
        shape.dims[2] = 0;
        shape.dims[3] = 0;
        ndim = 2;
        flags = 0;
        strides[0] = dim1 * sizeof(T);
        strides[1] = sizeof(T);
        strides[2] = 0;
        strides[3] = 0;
    }
    CUDA_CALLABLE array_t(T WP_DEVICE* data, int dim0, int dim1, int dim2, T WP_DEVICE* grad = nullptr)
        : data(data)
        , grad(grad)
    {
        // constructor for 3d array
        shape.dims[0] = dim0;
        shape.dims[1] = dim1;
        shape.dims[2] = dim2;
        shape.dims[3] = 0;
        ndim = 3;
        flags = 0;
        strides[0] = dim1 * dim2 * sizeof(T);
        strides[1] = dim2 * sizeof(T);
        strides[2] = sizeof(T);
        strides[3] = 0;
    }
    CUDA_CALLABLE array_t(T WP_DEVICE* data, int dim0, int dim1, int dim2, int dim3, T WP_DEVICE* grad = nullptr)
        : data(data)
        , grad(grad)
    {
        // constructor for 4d array
        shape.dims[0] = dim0;
        shape.dims[1] = dim1;
        shape.dims[2] = dim2;
        shape.dims[3] = dim3;
        ndim = 4;
        flags = 0;
        strides[0] = dim1 * dim2 * dim3 * sizeof(T);
        strides[1] = dim2 * dim3 * sizeof(T);
        strides[2] = dim3 * sizeof(T);
        strides[3] = sizeof(T);
    }

    CUDA_CALLABLE array_t(uint64 data, int size, uint64 grad = 0)
        : array_t((T WP_DEVICE*)(WP_ARRAY_PTR(data)), size, (T WP_DEVICE*)(WP_ARRAY_PTR(grad)))
    {
    }

    CUDA_CALLABLE array_t(uint64 data, int dim0, int dim1, uint64 grad = 0)
        : array_t((T WP_DEVICE*)(WP_ARRAY_PTR(data)), dim0, dim1, (T WP_DEVICE*)(WP_ARRAY_PTR(grad)))
    {
    }

    CUDA_CALLABLE array_t(uint64 data, int dim0, int dim1, int dim2, uint64 grad = 0)
        : array_t((T WP_DEVICE*)(WP_ARRAY_PTR(data)), dim0, dim1, dim2, (T WP_DEVICE*)(WP_ARRAY_PTR(grad)))
    {
    }

    CUDA_CALLABLE array_t(uint64 data, int dim0, int dim1, int dim2, int dim3, uint64 grad = 0)
        : array_t((T WP_DEVICE*)(WP_ARRAY_PTR(data)), dim0, dim1, dim2, dim3, (T WP_DEVICE*)(WP_ARRAY_PTR(grad)))
    {
    }

    CUDA_CALLABLE inline bool empty() const { return !data; }

    T WP_DEVICE* data;
    T WP_DEVICE* grad;
    shape_t shape;
    int strides[ARRAY_MAX_DIMS];
    uint16_t ndim;
    uint16_t flags;

    CUDA_CALLABLE inline operator T WP_DEVICE*() const { return data; }
};


// Required when compiling adjoints.
template <typename T> inline CUDA_CALLABLE array_t<T> add(const array_t<T> WP_THREAD& a, const array_t<T> WP_THREAD& b)
{
    return array_t<T>();
}


#if !defined(__METAL_VERSION__)  // TODO(metal): fixedarray_t derives from array_t; Metal has no inheritance
// Stack‑allocated counterpart to `array_t<T>`.
// Useful for small buffers that have their shape known at compile-time,
// and that gain from having array semantics instead of vectors.
template <int Size, typename T> struct fixedarray_t : array_t<T> {
    using Base = array_t<T>;

    static_assert(Size > 0, "Expected Size > 0");

    CUDA_CALLABLE inline fixedarray_t()
        : Base(storage, Size)
        , storage()
    {
    }

    CUDA_CALLABLE fixedarray_t(int dim0, T WP_DEVICE* grad = nullptr)
        : Base(storage, dim0, grad)
        , storage()
    {
        assert(Size == dim0);
    }

    CUDA_CALLABLE fixedarray_t(int dim0, int dim1, T WP_DEVICE* grad = nullptr)
        : Base(storage, dim0, dim1, grad)
        , storage()
    {
        assert(Size == dim0 * dim1);
    }

    CUDA_CALLABLE fixedarray_t(int dim0, int dim1, int dim2, T WP_DEVICE* grad = nullptr)
        : Base(storage, dim0, dim1, dim2, grad)
        , storage()
    {
        assert(Size == dim0 * dim1 * dim2);
    }

    CUDA_CALLABLE fixedarray_t(int dim0, int dim1, int dim2, int dim3, T WP_DEVICE* grad = nullptr)
        : Base(storage, dim0, dim1, dim2, dim3, grad)
        , storage()
    {
        assert(Size == dim0 * dim1 * dim2 * dim3);
    }

    CUDA_CALLABLE fixedarray_t<Size, T> WP_THREAD& operator=(const fixedarray_t<Size, T> WP_THREAD& other)
    {
        for (unsigned int i = 0; i < Size; ++i) {
            this->storage[i] = other.storage[i];
        }

        this->data = this->storage;
        this->grad = nullptr;
        this->shape = other.shape;

        for (unsigned int i = 0; i < ARRAY_MAX_DIMS; ++i) {
            this->strides[i] = other.strides[i];
        }

        this->ndim = other.ndim;
        this->flags = other.flags;

        return *this;
    }

    T storage[Size];
};


// Required when compiling adjoints.
template <int Size, typename T>
inline CUDA_CALLABLE fixedarray_t<Size, T>
add(const fixedarray_t<Size, T> WP_THREAD& a, const fixedarray_t<Size, T> WP_THREAD& b)
{
    return fixedarray_t<Size, T>();
}
#else  // __METAL_VERSION__
// Metal has no derived classes, and a stack array cannot back an `array_t`, whose `data` is a
// `device` pointer (MSL address spaces are disjoint, so no cast reaches thread memory). Fixed-size
// arrays are therefore plain `array_t`s over a per-thread slice of a device scratch buffer:
// codegen gives each `wp.zeros()` / copy site a static offset in its function's frame, and every
// generated function receives the thread's scratch pointer as a hidden argument (`_wp_fixed`,
// see WP_FUNC_PARAM in tile.h), advanced past the caller's frame for each call. All array
// builtins, `.ptr` and `wp.array(ptr=...)` views of the storage then work unchanged.
// Differences from CPU/CUDA: the storage of a fixed array returned from a function lives in the
// callee's frame (valid until the next call at that depth), and it is device memory, not registers.
template <int Size, typename T> using fixedarray_t = array_t<T>;

template <int Size, typename T> inline T WP_DEVICE* fixedarray_zero_storage(WP_DEVICE char* storage)
{
    static_assert(Size > 0, "Expected Size > 0");
    WP_DEVICE uint32_t* words = (WP_DEVICE uint32_t*)storage;
    const int bytes = int(Size * sizeof(T));
    for (int i = 0; i < bytes / 4; ++i)
        words[i] = 0u;
    for (int i = bytes & ~3; i < bytes; ++i)
        storage[i] = 0;
    return (T WP_DEVICE*)storage;
}

template <int Size, typename T, int Offset> inline array_t<T> fixedarray_zeros(WP_DEVICE char* scratch, int dim0)
{
    return array_t<T>(fixedarray_zero_storage<Size, T>(scratch + Offset), dim0);
}
template <int Size, typename T, int Offset>
inline array_t<T> fixedarray_zeros(WP_DEVICE char* scratch, int dim0, int dim1)
{
    return array_t<T>(fixedarray_zero_storage<Size, T>(scratch + Offset), dim0, dim1);
}
template <int Size, typename T, int Offset>
inline array_t<T> fixedarray_zeros(WP_DEVICE char* scratch, int dim0, int dim1, int dim2)
{
    return array_t<T>(fixedarray_zero_storage<Size, T>(scratch + Offset), dim0, dim1, dim2);
}
template <int Size, typename T, int Offset>
inline array_t<T> fixedarray_zeros(WP_DEVICE char* scratch, int dim0, int dim1, int dim2, int dim3)
{
    return array_t<T>(fixedarray_zero_storage<Size, T>(scratch + Offset), dim0, dim1, dim2, dim3);
}

// Value copy (`b = a` of a fixed array): new storage at the copy site's frame offset, same shape.
template <int Size, typename T, int Offset>
inline array_t<T> fixedarray_copy(WP_DEVICE char* scratch, const array_t<T> WP_THREAD& src)
{
    T WP_DEVICE* dst = (T WP_DEVICE*)(scratch + Offset);
    for (int i = 0; i < Size; ++i)
        dst[i] = src.data[i];
    array_t<T> out = src;
    out.data = dst;
    out.grad = nullptr;
    return out;
}
#endif  // !__METAL_VERSION__


// TODO:
// - templated index type?
// - templated dimensionality? (also for array_t to save space when passing arrays to kernels)
template <typename T> struct indexedarray_t {
    CUDA_CALLABLE inline indexedarray_t()
        : arr()
        , indices()
        , shape()
    {
    }

    CUDA_CALLABLE inline bool empty() const { return !arr.data; }

    array_t<T> arr;
    int WP_DEVICE* indices[ARRAY_MAX_DIMS];  // index array per dimension (can be NULL)
    shape_t shape;  // element count per dimension (num. indices if indexed, array dim if not)
};


// return stride (in bytes) of the given index
template <typename T> CUDA_CALLABLE inline size_t stride(const array_t<T> WP_THREAD& a, int dim)
{
    return size_t(a.strides[dim]);
}

template <typename T>
CUDA_CALLABLE inline T WP_DEVICE* data_at_byte_offset(const array_t<T> WP_THREAD& a, size_t byte_offset)
{
    return reinterpret_cast<T WP_DEVICE*>(reinterpret_cast<char WP_DEVICE*>(a.data) + byte_offset);
}

template <typename T>
CUDA_CALLABLE inline T WP_DEVICE* grad_at_byte_offset(const array_t<T> WP_THREAD& a, size_t byte_offset)
{
    return reinterpret_cast<T WP_DEVICE*>(reinterpret_cast<char WP_DEVICE*>(a.grad) + byte_offset);
}

template <typename T> CUDA_CALLABLE inline size_t byte_offset(const array_t<T> WP_THREAD& arr, int i)
{
    assert(i >= 0 && i < arr.shape[0]);

    return i * stride(arr, 0);
}

template <typename T> CUDA_CALLABLE inline size_t byte_offset(const array_t<T> WP_THREAD& arr, int i, int j)
{
    // if (i < 0 || i >= arr.shape[0])
    //     printf("i: %d > arr.shape[0]: %d\n", i, arr.shape[0]);

    // if (j < 0 || j >= arr.shape[1])
    //     printf("j: %d > arr.shape[1]: %d\n", j, arr.shape[1]);


    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);

    return i * stride(arr, 0) + j * stride(arr, 1);
}

template <typename T> CUDA_CALLABLE inline size_t byte_offset(const array_t<T> WP_THREAD& arr, int i, int j, int k)
{
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);
    assert(k >= 0 && k < arr.shape[2]);

    return i * stride(arr, 0) + j * stride(arr, 1) + k * stride(arr, 2);
}

template <typename T>
CUDA_CALLABLE inline size_t byte_offset(const array_t<T> WP_THREAD& arr, int i, int j, int k, int l)
{
    assert(i >= 0 && i < arr.shape[0]);
    assert(j >= 0 && j < arr.shape[1]);
    assert(k >= 0 && k < arr.shape[2]);
    assert(l >= 0 && l < arr.shape[3]);

    return i * stride(arr, 0) + j * stride(arr, 1) + k * stride(arr, 2) + l * stride(arr, 3);
}

template <typename T> CUDA_CALLABLE inline T WP_DEVICE& index(const array_t<T> WP_THREAD& arr, int i)
{
    assert(arr.ndim == 1);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);

    if (i < 0) {
        i += arr.shape[0];
    }

    T WP_DEVICE& result = *data_at_byte_offset(arr, byte_offset(arr, i));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T WP_DEVICE& index(const array_t<T> WP_THREAD& arr, int i, int j)
{
    assert(arr.ndim == 2);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }

    T WP_DEVICE& result = *data_at_byte_offset(arr, byte_offset(arr, i, j));
    FP_VERIFY_FWD_2(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T WP_DEVICE& index(const array_t<T> WP_THREAD& arr, int i, int j, int k)
{
    assert(arr.ndim == 3);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);
    assert(k >= -arr.shape[2] && k < arr.shape[2]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }
    if (k < 0) {
        k += arr.shape[2];
    }

    T WP_DEVICE& result = *data_at_byte_offset(arr, byte_offset(arr, i, j, k));
    FP_VERIFY_FWD_3(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T WP_DEVICE& index(const array_t<T> WP_THREAD& arr, int i, int j, int k, int l)
{
    assert(arr.ndim == 4);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);
    assert(k >= -arr.shape[2] && k < arr.shape[2]);
    assert(l >= -arr.shape[3] && l < arr.shape[3]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }
    if (k < 0) {
        k += arr.shape[2];
    }
    if (l < 0) {
        l += arr.shape[3];
    }

    T WP_DEVICE& result = *data_at_byte_offset(arr, byte_offset(arr, i, j, k, l));
    FP_VERIFY_FWD_4(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T WP_DEVICE& index_grad(const array_t<T> WP_THREAD& arr, int i)
{
    assert(arr.ndim == 1);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);

    if (i < 0) {
        i += arr.shape[0];
    }

    T WP_DEVICE& result = *grad_at_byte_offset(arr, byte_offset(arr, i));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T WP_DEVICE& index_grad(const array_t<T> WP_THREAD& arr, int i, int j)
{
    assert(arr.ndim == 2);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }

    T WP_DEVICE& result = *grad_at_byte_offset(arr, byte_offset(arr, i, j));
    FP_VERIFY_FWD_2(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T WP_DEVICE& index_grad(const array_t<T> WP_THREAD& arr, int i, int j, int k)
{
    assert(arr.ndim == 3);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);
    assert(k >= -arr.shape[2] && k < arr.shape[2]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }
    if (k < 0) {
        k += arr.shape[2];
    }

    T WP_DEVICE& result = *grad_at_byte_offset(arr, byte_offset(arr, i, j, k));
    FP_VERIFY_FWD_3(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T WP_DEVICE& index_grad(const array_t<T> WP_THREAD& arr, int i, int j, int k, int l)
{
    assert(arr.ndim == 4);
    assert(i >= -arr.shape[0] && i < arr.shape[0]);
    assert(j >= -arr.shape[1] && j < arr.shape[1]);
    assert(k >= -arr.shape[2] && k < arr.shape[2]);
    assert(l >= -arr.shape[3] && l < arr.shape[3]);

    if (i < 0) {
        i += arr.shape[0];
    }
    if (j < 0) {
        j += arr.shape[1];
    }
    if (k < 0) {
        k += arr.shape[2];
    }
    if (l < 0) {
        l += arr.shape[3];
    }

    T WP_DEVICE& result = *grad_at_byte_offset(arr, byte_offset(arr, i, j, k, l));
    FP_VERIFY_FWD_4(result)

    return result;
}


template <typename T> CUDA_CALLABLE inline T WP_DEVICE& index(const indexedarray_t<T> WP_THREAD& iarr, int i)
{
    assert(iarr.arr.ndim == 1);
    assert(i >= -iarr.shape[0] && i < iarr.shape[0]);

    if (i < 0) {
        i += iarr.shape[0];
    }

    if (iarr.indices[0]) {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }

    T WP_DEVICE& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T> CUDA_CALLABLE inline T WP_DEVICE& index(const indexedarray_t<T> WP_THREAD& iarr, int i, int j)
{
    assert(iarr.arr.ndim == 2);
    assert(i >= -iarr.shape[0] && i < iarr.shape[0]);
    assert(j >= -iarr.shape[1] && j < iarr.shape[1]);

    if (i < 0) {
        i += iarr.shape[0];
    }
    if (j < 0) {
        j += iarr.shape[1];
    }

    if (iarr.indices[0]) {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }
    if (iarr.indices[1]) {
        j = iarr.indices[1][j];
        assert(j >= 0 && j < iarr.arr.shape[1]);
    }

    T WP_DEVICE& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i, j));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T WP_DEVICE& index(const indexedarray_t<T> WP_THREAD& iarr, int i, int j, int k)
{
    assert(iarr.arr.ndim == 3);
    assert(i >= -iarr.shape[0] && i < iarr.shape[0]);
    assert(j >= -iarr.shape[1] && j < iarr.shape[1]);
    assert(k >= -iarr.shape[2] && k < iarr.shape[2]);

    if (i < 0) {
        i += iarr.shape[0];
    }
    if (j < 0) {
        j += iarr.shape[1];
    }
    if (k < 0) {
        k += iarr.shape[2];
    }

    if (iarr.indices[0]) {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }
    if (iarr.indices[1]) {
        j = iarr.indices[1][j];
        assert(j >= 0 && j < iarr.arr.shape[1]);
    }
    if (iarr.indices[2]) {
        k = iarr.indices[2][k];
        assert(k >= 0 && k < iarr.arr.shape[2]);
    }

    T WP_DEVICE& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i, j, k));
    FP_VERIFY_FWD_1(result)

    return result;
}

template <typename T>
CUDA_CALLABLE inline T WP_DEVICE& index(const indexedarray_t<T> WP_THREAD& iarr, int i, int j, int k, int l)
{
    assert(iarr.arr.ndim == 4);
    assert(i >= -iarr.shape[0] && i < iarr.shape[0]);
    assert(j >= -iarr.shape[1] && j < iarr.shape[1]);
    assert(k >= -iarr.shape[2] && k < iarr.shape[2]);
    assert(l >= -iarr.shape[3] && l < iarr.shape[3]);

    if (i < 0) {
        i += iarr.shape[0];
    }
    if (j < 0) {
        j += iarr.shape[1];
    }
    if (k < 0) {
        k += iarr.shape[2];
    }
    if (l < 0) {
        l += iarr.shape[3];
    }

    if (iarr.indices[0]) {
        i = iarr.indices[0][i];
        assert(i >= 0 && i < iarr.arr.shape[0]);
    }
    if (iarr.indices[1]) {
        j = iarr.indices[1][j];
        assert(j >= 0 && j < iarr.arr.shape[1]);
    }
    if (iarr.indices[2]) {
        k = iarr.indices[2][k];
        assert(k >= 0 && k < iarr.arr.shape[2]);
    }
    if (iarr.indices[3]) {
        l = iarr.indices[3][l];
        assert(l >= 0 && l < iarr.arr.shape[3]);
    }

    T WP_DEVICE& result = *data_at_byte_offset(iarr.arr, byte_offset(iarr.arr, i, j, k, l));
    FP_VERIFY_FWD_1(result)

    return result;
}


// Unlike the variadic slice overload, these default-construct their result.
// See 7e5fb05b for the sm_89 NVCC miscompile behind that difference.
template <typename T> CUDA_CALLABLE inline array_t<T> view(array_t<T> WP_THREAD& src, int i)
{
    assert(src.ndim > 1);
    assert(i >= -src.shape[0] && i < src.shape[0]);

    if (i < 0) {
        i += src.shape[0];
    }

    array_t<T> a;
    size_t offset = byte_offset(src, i);
    a.data = data_at_byte_offset(src, offset);
    if (src.grad)
        a.grad = grad_at_byte_offset(src, offset);
    a.shape[0] = src.shape[1];
    a.shape[1] = src.shape[2];
    a.shape[2] = src.shape[3];
    a.strides[0] = src.strides[1];
    a.strides[1] = src.strides[2];
    a.strides[2] = src.strides[3];
    a.shape[3] = 0;  // unused dimensions read as zero (not left to the default constructor)
    a.strides[3] = 0;
    a.ndim = src.ndim - 1;

    return a;
}

template <typename T> CUDA_CALLABLE inline array_t<T> view(array_t<T> WP_THREAD& src, int i, int j)
{
    assert(src.ndim > 2);
    assert(i >= -src.shape[0] && i < src.shape[0]);
    assert(j >= -src.shape[1] && j < src.shape[1]);

    if (i < 0) {
        i += src.shape[0];
    }
    if (j < 0) {
        j += src.shape[1];
    }

    array_t<T> a;
    size_t offset = byte_offset(src, i, j);
    a.data = data_at_byte_offset(src, offset);
    if (src.grad)
        a.grad = grad_at_byte_offset(src, offset);
    a.shape[0] = src.shape[2];
    a.shape[1] = src.shape[3];
    a.strides[0] = src.strides[2];
    a.strides[1] = src.strides[3];
    a.shape[2] = a.shape[3] = 0;  // unused dimensions read as zero
    a.strides[2] = a.strides[3] = 0;
    a.ndim = src.ndim - 2;

    return a;
}

template <typename T> CUDA_CALLABLE inline array_t<T> view(array_t<T> WP_THREAD& src, int i, int j, int k)
{
    assert(src.ndim > 3);
    assert(i >= -src.shape[0] && i < src.shape[0]);
    assert(j >= -src.shape[1] && j < src.shape[1]);
    assert(k >= -src.shape[2] && k < src.shape[2]);

    if (i < 0) {
        i += src.shape[0];
    }
    if (j < 0) {
        j += src.shape[1];
    }
    if (k < 0) {
        k += src.shape[2];
    }

    array_t<T> a;
    size_t offset = byte_offset(src, i, j, k);
    a.data = data_at_byte_offset(src, offset);
    if (src.grad)
        a.grad = grad_at_byte_offset(src, offset);
    a.shape[0] = src.shape[3];
    a.strides[0] = src.strides[3];
    a.shape[1] = a.shape[2] = a.shape[3] = 0;  // unused dimensions read as zero
    a.strides[1] = a.strides[2] = a.strides[3] = 0;
    a.ndim = src.ndim - 3;

    return a;
}


CUDA_CALLABLE inline slice_t view_arg_as_slice(int index) { return { index, index, 1 }; }

CUDA_CALLABLE inline slice_t view_arg_as_slice(const slice_t WP_THREAD& slice) { return slice; }

CUDA_CALLABLE inline bool view_arg_is_slice(int) { return false; }

CUDA_CALLABLE inline bool view_arg_is_slice(const slice_t WP_THREAD&) { return true; }


template <typename T, size_t... Idxs>
inline CUDA_CALLABLE size_t
byte_offset_helper(array_t<T> WP_THREAD& src, const slice_t WP_THREAD* slices, index_sequence<Idxs...>)
{
    return byte_offset(src, slices[Idxs].start...);
}


template <typename T, typename... Slices>
CUDA_CALLABLE inline array_t<T> view(array_t<T> WP_THREAD& src, const Slices WP_THREAD&... slice_args)
{
    constexpr int N = sizeof...(Slices);
    static_assert(N >= 1 && N <= 4, "view supports 1 to 4 slices");
    assert(src.ndim >= N);

    slice_t slices[N] = { view_arg_as_slice(slice_args)... };
    bool is_slice_arg[N] = { view_arg_is_slice(slice_args)... };
    int slice_idxs[N];
    int slice_count = 0;

    for (int i = 0; i < N; ++i) {
        if (!is_slice_arg[i]) {
            // We have an integer index.
            if (slices[i].start < 0) {
                slices[i].start += src.shape[i];
            }
        } else {
            slices[i] = slice_adjust_indices(slices[i], src.shape[i]);
            slice_idxs[slice_count] = i;
            ++slice_count;
        }
    }

    size_t offset = byte_offset_helper(src, slices, make_index_sequence<N> {});

    // Copy-construct rather than default-construct to work around an NVCC
    // miscompile observed on older architectures (sm_89).
    array_t<T> out(src);
    out.flags = 0;

    out.data = data_at_byte_offset(src, offset);
    if (src.grad) {
        out.grad = grad_at_byte_offset(src, offset);
    }

    int dim = 0;
    for (; dim < slice_count; ++dim) {
        int idx = slice_idxs[dim];
        out.shape[dim] = slice_get_length_unchecked(slices[idx]);
        out.strides[dim] = src.strides[idx] * slices[idx].step;
    }
    for (; dim < slice_count + 4 - N; ++dim) {
        out.shape[dim] = src.shape[dim - slice_count + N];
        out.strides[dim] = src.strides[dim - slice_count + N];
    }
    for (; dim < 4; ++dim) {
        out.shape[dim] = 0;
        out.strides[dim] = 0;
    }

    out.ndim = src.ndim + slice_count - N;
    return out;
}

template <typename T> CUDA_CALLABLE inline indexedarray_t<T> view(indexedarray_t<T> WP_THREAD& src, int i)
{
    assert(src.arr.ndim > 1);

    if (src.indices[0]) {
        assert(i >= -src.shape[0] && i < src.shape[0]);
        if (i < 0) {
            i += src.shape[0];
        }
        i = src.indices[0][i];
    }

    indexedarray_t<T> a;
    a.arr = view(src.arr, i);
    a.indices[0] = src.indices[1];
    a.indices[1] = src.indices[2];
    a.indices[2] = src.indices[3];
    a.shape[0] = src.shape[1];
    a.shape[1] = src.shape[2];
    a.shape[2] = src.shape[3];

    return a;
}

template <typename T> CUDA_CALLABLE inline indexedarray_t<T> view(indexedarray_t<T> WP_THREAD& src, int i, int j)
{
    assert(src.arr.ndim > 2);

    if (src.indices[0]) {
        assert(i >= -src.shape[0] && i < src.shape[0]);
        if (i < 0) {
            i += src.shape[0];
        }
        i = src.indices[0][i];
    }
    if (src.indices[1]) {
        assert(j >= -src.shape[1] && j < src.shape[1]);
        if (j < 0) {
            j += src.shape[1];
        }
        j = src.indices[1][j];
    }

    indexedarray_t<T> a;
    a.arr = view(src.arr, i, j);
    a.indices[0] = src.indices[2];
    a.indices[1] = src.indices[3];
    a.shape[0] = src.shape[2];
    a.shape[1] = src.shape[3];

    return a;
}

template <typename T> CUDA_CALLABLE inline indexedarray_t<T> view(indexedarray_t<T> WP_THREAD& src, int i, int j, int k)
{
    assert(src.arr.ndim > 3);

    if (src.indices[0]) {
        assert(i >= -src.shape[0] && i < src.shape[0]);
        if (i < 0) {
            i += src.shape[0];
        }
        i = src.indices[0][i];
    }
    if (src.indices[1]) {
        assert(j >= -src.shape[1] && j < src.shape[1]);
        if (j < 0) {
            j += src.shape[1];
        }
        j = src.indices[1][j];
    }
    if (src.indices[2]) {
        assert(k >= -src.shape[2] && k < src.shape[2]);
        if (k < 0) {
            k += src.shape[2];
        }
        k = src.indices[2][k];
    }

    indexedarray_t<T> a;
    a.arr = view(src.arr, i, j, k);
    a.indices[0] = src.indices[3];
    a.shape[0] = src.shape[3];

    return a;
}

template <template <typename> class A1, template <typename> class A2, template <typename> class A3, typename T>
inline CUDA_CALLABLE void
adj_view(A1<T> WP_THREAD& src, int i, A2<T> WP_THREAD& adj_src, int adj_i, A3<T> WP_THREAD& adj_ret)
{
    // nop: view aliases the underlying array's storage; gradients flow through
    // subsequent operations on the view via the underlying array's .grad
}
template <template <typename> class A1, template <typename> class A2, template <typename> class A3, typename T>
inline CUDA_CALLABLE void
adj_view(A1<T> WP_THREAD& src, int i, int j, A2<T> WP_THREAD& adj_src, int adj_i, int adj_j, A3<T> WP_THREAD& adj_ret)
{
    // nop: view aliases the underlying array's storage; gradients flow through
    // subsequent operations on the view via the underlying array's .grad
}
template <template <typename> class A1, template <typename> class A2, template <typename> class A3, typename T>
inline CUDA_CALLABLE void adj_view(
    A1<T> WP_THREAD& src,
    int i,
    int j,
    int k,
    A2<T> WP_THREAD& adj_src,
    int adj_i,
    int adj_j,
    int adj_k,
    A3<T> WP_THREAD& adj_ret
)
{
    // nop: view aliases the underlying array's storage; gradients flow through
    // subsequent operations on the view via the underlying array's .grad
}

// Fallback overload for unsupported view signatures; intentionally empty.
#if defined(__METAL_VERSION__)
template <typename... Args> CUDA_CALLABLE inline void adj_view(Args...) { }  // MSL: no forwarding references
#else
template <typename... Args> CUDA_CALLABLE inline void adj_view(Args&&...) { }
#endif

// TODO: lower_bound() for indexed arrays?

template <typename T>
CUDA_CALLABLE inline int lower_bound(const array_t<T> WP_THREAD& arr, int arr_begin, int arr_end, T value)
{
    assert(arr.ndim == 1);

    int lower = arr_begin;
    int upper = arr_end - 1;

    while (lower < upper) {
        int mid = lower + (upper - lower) / 2;

        if (arr[mid] < value) {
            lower = mid + 1;
        } else {
            upper = mid;
        }
    }

    return lower;
}

template <typename T> CUDA_CALLABLE inline int lower_bound(const array_t<T> WP_THREAD& arr, T value)
{
    return lower_bound(arr, 0, arr.shape[0], value);
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_add(const A<T> WP_THREAD& buf, int i, T value)
{
    return atomic_add(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_add(const A<T> WP_THREAD& buf, int i, int j, T value)
{
    return atomic_add(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_add(const A<T> WP_THREAD& buf, int i, int j, int k, T value)
{
    return atomic_add(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_add(const A<T> WP_THREAD& buf, int i, int j, int k, int l, T value)
{
    return atomic_add(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_sub(const A<T> WP_THREAD& buf, int i, T value)
{
    return atomic_add(&index(buf, i), -value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_sub(const A<T> WP_THREAD& buf, int i, int j, T value)
{
    return atomic_add(&index(buf, i, j), -value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_sub(const A<T> WP_THREAD& buf, int i, int j, int k, T value)
{
    return atomic_add(&index(buf, i, j, k), -value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_sub(const A<T> WP_THREAD& buf, int i, int j, int k, int l, T value)
{
    return atomic_add(&index(buf, i, j, k, l), -value);
}

// SlotType is the value type of the referenced component returned by access(),
// such as a scalar vector component or a matrix row.
template <typename T, typename SlotType, typename Accessor, typename... Ints>
inline CUDA_CALLABLE SlotType
array_atomic_add_slot(const array_t<T> WP_THREAD& buf, SlotType value, Accessor access, Ints... indices)
{
    SlotType old = atomic_add(&access(index(buf, indices...)), value);
    FP_VERIFY_FWD(old + value)
    return old;
}

template <typename T, typename SlotType, typename Accessor, typename... Ints>
inline CUDA_CALLABLE SlotType
array_atomic_sub_slot(const array_t<T> WP_THREAD& buf, SlotType value, Accessor access, Ints... indices)
{
    SlotType old = atomic_add(&access(index(buf, indices...)), -value);
    FP_VERIFY_FWD(old - value)
    return old;
}

template <typename T, typename SlotType, typename Accessor, typename... Ints>
inline CUDA_CALLABLE SlotType
array_atomic_and_slot(const array_t<T> WP_THREAD& buf, SlotType value, Accessor access, Ints... indices)
{
    return atomic_and(&access(index(buf, indices...)), value);
}

template <typename T, typename SlotType, typename Accessor, typename... Ints>
inline CUDA_CALLABLE SlotType
array_atomic_or_slot(const array_t<T> WP_THREAD& buf, SlotType value, Accessor access, Ints... indices)
{
    return atomic_or(&access(index(buf, indices...)), value);
}

template <typename T, typename SlotType, typename Accessor, typename... Ints>
inline CUDA_CALLABLE SlotType
array_atomic_xor_slot(const array_t<T> WP_THREAD& buf, SlotType value, Accessor access, Ints... indices)
{
    return atomic_xor(&access(index(buf, indices...)), value);
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_min(const A<T> WP_THREAD& buf, int i, T value)
{
    return atomic_min(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_min(const A<T> WP_THREAD& buf, int i, int j, T value)
{
    return atomic_min(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_min(const A<T> WP_THREAD& buf, int i, int j, int k, T value)
{
    return atomic_min(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_min(const A<T> WP_THREAD& buf, int i, int j, int k, int l, T value)
{
    return atomic_min(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_max(const A<T> WP_THREAD& buf, int i, T value)
{
    return atomic_max(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_max(const A<T> WP_THREAD& buf, int i, int j, T value)
{
    return atomic_max(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_max(const A<T> WP_THREAD& buf, int i, int j, int k, T value)
{
    return atomic_max(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_max(const A<T> WP_THREAD& buf, int i, int j, int k, int l, T value)
{
    return atomic_max(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_cas(const A<T> WP_THREAD& buf, int i, T old_value, T new_value)
{
    return atomic_cas(&index(buf, i), old_value, new_value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_cas(const A<T> WP_THREAD& buf, int i, int j, T old_value, T new_value)
{
    return atomic_cas(&index(buf, i, j), old_value, new_value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_cas(const A<T> WP_THREAD& buf, int i, int j, int k, T old_value, T new_value)
{
    return atomic_cas(&index(buf, i, j, k), old_value, new_value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_cas(const A<T> WP_THREAD& buf, int i, int j, int k, int l, T old_value, T new_value)
{
    return atomic_cas(&index(buf, i, j, k, l), old_value, new_value);
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_exch(const A<T> WP_THREAD& buf, int i, T value)
{
    return atomic_exch(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_exch(const A<T> WP_THREAD& buf, int i, int j, T value)
{
    return atomic_exch(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_exch(const A<T> WP_THREAD& buf, int i, int j, int k, T value)
{
    return atomic_exch(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_exch(const A<T> WP_THREAD& buf, int i, int j, int k, int l, T value)
{
    return atomic_exch(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_and(const A<T> WP_THREAD& buf, int i, T value)
{
    return atomic_and(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_and(const A<T> WP_THREAD& buf, int i, int j, T value)
{
    return atomic_and(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_and(const A<T> WP_THREAD& buf, int i, int j, int k, T value)
{
    return atomic_and(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_and(const A<T> WP_THREAD& buf, int i, int j, int k, int l, T value)
{
    return atomic_and(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_or(const A<T> WP_THREAD& buf, int i, T value)
{
    return atomic_or(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_or(const A<T> WP_THREAD& buf, int i, int j, T value)
{
    return atomic_or(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_or(const A<T> WP_THREAD& buf, int i, int j, int k, T value)
{
    return atomic_or(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_or(const A<T> WP_THREAD& buf, int i, int j, int k, int l, T value)
{
    return atomic_or(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_xor(const A<T> WP_THREAD& buf, int i, T value)
{
    return atomic_xor(&index(buf, i), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_xor(const A<T> WP_THREAD& buf, int i, int j, T value)
{
    return atomic_xor(&index(buf, i, j), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_xor(const A<T> WP_THREAD& buf, int i, int j, int k, T value)
{
    return atomic_xor(&index(buf, i, j, k), value);
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T atomic_xor(const A<T> WP_THREAD& buf, int i, int j, int k, int l, T value)
{
    return atomic_xor(&index(buf, i, j, k, l), value);
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE T WP_DEVICE* address(const A<T> WP_THREAD& buf, int i)
{
    return &index(buf, i);  // cppcheck-suppress returnDanglingLifetime
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T WP_DEVICE* address(const A<T> WP_THREAD& buf, int i, int j)
{
    return &index(buf, i, j);  // cppcheck-suppress returnDanglingLifetime
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T WP_DEVICE* address(const A<T> WP_THREAD& buf, int i, int j, int k)
{
    return &index(buf, i, j, k);  // cppcheck-suppress returnDanglingLifetime
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE T WP_DEVICE* address(const A<T> WP_THREAD& buf, int i, int j, int k, int l)
{
    return &index(buf, i, j, k, l);  // cppcheck-suppress returnDanglingLifetime
}

template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T> WP_THREAD& buf, int i, T value)
{
    FP_VERIFY_FWD_1(value)

    index(buf, i) = value;
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T> WP_THREAD& buf, int i, int j, T value)
{
    FP_VERIFY_FWD_2(value)

    index(buf, i, j) = value;
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T> WP_THREAD& buf, int i, int j, int k, T value)
{
    FP_VERIFY_FWD_3(value)

    index(buf, i, j, k) = value;
}
template <template <typename> class A, typename T>
inline CUDA_CALLABLE void array_store(const A<T> WP_THREAD& buf, int i, int j, int k, int l, T value)
{
    FP_VERIFY_FWD_4(value)

    index(buf, i, j, k, l) = value;
}

template <typename T> inline CUDA_CALLABLE void store(T WP_DEVICE* address, T value)
{
    FP_VERIFY_FWD(value)

    *address = value;
}

template <typename T> inline CUDA_CALLABLE T load(T WP_DEVICE* address)
{
    T value = *address;
    FP_VERIFY_FWD(value)
#if defined(__METAL_VERSION__)
    wp_metal_fixup(value);  // struct values with array members: translate the host pointers they carry
#endif
    return value;
}

#if defined(__METAL_VERSION__)
// References to thread-local values (locals, struct fields) share the load/store builtins.
template <typename T> inline void store(thread T* address, T value) { *address = value; }
template <typename T> inline T load(thread T* address) { return *address; }
#endif

// where() overload for array condition - returns a if array.data is non-null, otherwise returns b
template <typename T1, typename T2>
CUDA_CALLABLE inline T2 where(const array_t<T1> WP_THREAD& arr, const T2 WP_THREAD& a, const T2 WP_THREAD& b)
{
    return arr.data ? a : b;
}

template <typename T1, typename T2>
CUDA_CALLABLE inline void adj_where(
    const array_t<T1> WP_THREAD& arr,
    const T2 WP_THREAD& a,
    const T2 WP_THREAD& b,
    const array_t<T1> WP_THREAD& adj_cond,
    T2 WP_THREAD& adj_a,
    T2 WP_THREAD& adj_b,
    const T2 WP_THREAD& adj_ret
)
{
    if (arr.data)
        adj_a += adj_ret;
    else
        adj_b += adj_ret;
}

// stub for the case where we have an nested array inside a struct and
// atomic add the whole struct onto an array (e.g.: during backwards pass)
template <typename T> CUDA_CALLABLE inline void atomic_add(array_t<T> WP_DEVICE*, array_t<T>) { }

// stub for the case where we have an indexed array inside a struct and
// atomic add the whole struct onto an array (e.g.: during backwards pass)
template <typename T> CUDA_CALLABLE inline void atomic_add(indexedarray_t<T> WP_DEVICE*, indexedarray_t<T>) { }

// for float and vector types this is just an alias for an atomic add
template <typename T> CUDA_CALLABLE inline void adj_atomic_add(T WP_DEVICE* buf, T value) { atomic_add(buf, value); }


// for integral types we do not accumulate gradients
CUDA_CALLABLE inline void adj_atomic_add(int8 WP_DEVICE* buf, int8 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint8 WP_DEVICE* buf, uint8 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int16 WP_DEVICE* buf, int16 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint16 WP_DEVICE* buf, uint16 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int32 WP_DEVICE* buf, int32 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint32 WP_DEVICE* buf, uint32 value) { }
CUDA_CALLABLE inline void adj_atomic_add(int64 WP_DEVICE* buf, int64 value) { }
CUDA_CALLABLE inline void adj_atomic_add(uint64 WP_DEVICE* buf, uint64 value) { }

CUDA_CALLABLE inline void adj_atomic_add(bool WP_DEVICE* buf, bool value) { }

// only generate gradients for T types
template <typename T>
inline CUDA_CALLABLE void adj_address(
    const array_t<T> WP_THREAD& buf,
    int i,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    const T WP_THREAD& adj_output
)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i), adj_output);
}
template <typename T>
inline CUDA_CALLABLE void adj_address(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    const T WP_THREAD& adj_output
)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i, j), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i, j), adj_output);
}
template <typename T>
inline CUDA_CALLABLE void adj_address(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    const T WP_THREAD& adj_output
)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i, j, k), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i, j, k), adj_output);
}
template <typename T>
inline CUDA_CALLABLE void adj_address(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    const T WP_THREAD& adj_output
)
{
    if (adj_buf.data)
        adj_atomic_add(&index(adj_buf, i, j, k, l), adj_output);
    else if (buf.grad)
        adj_atomic_add(&index_grad(buf, i, j, k, l), adj_output);
}

template <typename T>
inline CUDA_CALLABLE void adj_array_store(
    const array_t<T> WP_THREAD& buf,
    int i,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_value
)
{
    if (adj_buf.data) {
        T WP_DEVICE& g = index(adj_buf, i);
        adj_value += T(g);  // thread-local copy: g may live in device memory

        // Only zero if adj_buf aliases buf.grad (standard Warp Tape case)
        // and retain_grad is not set on the forward array.
        // Skip zeroing for external gradient buffers passed via adj_inputs,
        // since the caller owns them and may need to preserve accumulated gradients.
        if (buf.grad && adj_buf.data == buf.grad && !(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    } else if (buf.grad) {
        // No explicit adjoint passed (adj_buf is null), fall back to buf.grad.
        T WP_DEVICE& g = index_grad(buf, i);
        adj_value += T(g);  // thread-local copy: g may live in device memory
        if (!(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    }

    FP_VERIFY_ADJ_1(value, adj_value)
}

// Slot-level array-store adjoint for composite-component writes.
// ``access`` maps an array element to the overwritten slot.
template <typename T, typename Accessor, typename AdjSlot, typename... Ints>
inline CUDA_CALLABLE void adj_array_store_slot(
    const array_t<T> WP_THREAD& buf,
    const array_t<T> WP_THREAD& adj_buf,
    AdjSlot WP_THREAD& adj_value,
    Accessor access,
    Ints... indices
)
{
    if (adj_buf.data) {
        AdjSlot WP_DEVICE& slot = access(index(adj_buf, indices...));
        adj_value += AdjSlot(slot);
        if (buf.grad && adj_buf.data == buf.grad && !(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            slot = AdjSlot {};
    } else if (buf.grad) {
        AdjSlot WP_DEVICE& slot = access(index_grad(buf, indices...));
        adj_value += AdjSlot(slot);
        if (!(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            slot = AdjSlot {};
    }

    FP_VERIFY_ADJ_SLOT(adj_value)
}

template <typename T, typename Accessor, typename AdjSlot, typename... Ints>
inline CUDA_CALLABLE void adj_array_atomic_add_slot(
    const array_t<T> WP_THREAD& buf,
    const array_t<T> WP_THREAD& adj_buf,
    AdjSlot WP_THREAD& adj_value,
    Accessor access,
    Ints... indices
)
{
    if (adj_buf.data) {
        adj_value += AdjSlot(access(index(adj_buf, indices...)));
    } else if (buf.grad) {
        adj_value += AdjSlot(access(index_grad(buf, indices...)));
    }

    FP_VERIFY_ADJ_SLOT(adj_value)
}

template <typename T, typename Accessor, typename AdjSlot, typename... Ints>
inline CUDA_CALLABLE void adj_array_atomic_sub_slot(
    const array_t<T> WP_THREAD& buf,
    const array_t<T> WP_THREAD& adj_buf,
    AdjSlot WP_THREAD& adj_value,
    Accessor access,
    Ints... indices
)
{
    if (adj_buf.data) {
        adj_value -= AdjSlot(access(index(adj_buf, indices...)));
    } else if (buf.grad) {
        adj_value -= AdjSlot(access(index_grad(buf, indices...)));
    }

    FP_VERIFY_ADJ_SLOT(adj_value)
}

template <typename T, typename Accessor, typename AdjSlot, typename... Ints>
inline CUDA_CALLABLE void adj_array_atomic_and_slot(
    const array_t<T> WP_THREAD& buf,
    const array_t<T> WP_THREAD& adj_buf,
    AdjSlot WP_THREAD& adj_value,
    Accessor access,
    Ints... indices
)
{
    // Bitwise integer atomics are intentionally non-differentiable.
}

template <typename T, typename Accessor, typename AdjSlot, typename... Ints>
inline CUDA_CALLABLE void adj_array_atomic_or_slot(
    const array_t<T> WP_THREAD& buf,
    const array_t<T> WP_THREAD& adj_buf,
    AdjSlot WP_THREAD& adj_value,
    Accessor access,
    Ints... indices
)
{
    // Bitwise integer atomics are intentionally non-differentiable.
}

template <typename T, typename Accessor, typename AdjSlot, typename... Ints>
inline CUDA_CALLABLE void adj_array_atomic_xor_slot(
    const array_t<T> WP_THREAD& buf,
    const array_t<T> WP_THREAD& adj_buf,
    AdjSlot WP_THREAD& adj_value,
    Accessor access,
    Ints... indices
)
{
    // Bitwise integer atomics are intentionally non-differentiable.
}

template <typename T>
inline CUDA_CALLABLE void adj_array_store(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value
)
{
    if (adj_buf.data) {
        T WP_DEVICE& g = index(adj_buf, i, j);
        adj_value += T(g);  // thread-local copy: g may live in device memory
        if (buf.grad && adj_buf.data == buf.grad && !(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    } else if (buf.grad) {
        T WP_DEVICE& g = index_grad(buf, i, j);
        adj_value += T(g);  // thread-local copy: g may live in device memory
        if (!(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    }

    FP_VERIFY_ADJ_2(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_array_store(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value
)
{
    if (adj_buf.data) {
        T WP_DEVICE& g = index(adj_buf, i, j, k);
        adj_value += T(g);  // thread-local copy: g may live in device memory
        if (buf.grad && adj_buf.data == buf.grad && !(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    } else if (buf.grad) {
        T WP_DEVICE& g = index_grad(buf, i, j, k);
        adj_value += T(g);  // thread-local copy: g may live in device memory
        if (!(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    }

    FP_VERIFY_ADJ_3(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_array_store(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value
)
{
    if (adj_buf.data) {
        T WP_DEVICE& g = index(adj_buf, i, j, k, l);
        adj_value += T(g);  // thread-local copy: g may live in device memory
        if (buf.grad && adj_buf.data == buf.grad && !(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    } else if (buf.grad) {
        T WP_DEVICE& g = index_grad(buf, i, j, k, l);
        adj_value += T(g);  // thread-local copy: g may live in device memory
        if (!(buf.flags & ARRAY_FLAG_RETAIN_GRAD))
            g = T {};
    }

    FP_VERIFY_ADJ_4(value, adj_value)
}

template <typename T>
inline CUDA_CALLABLE void
adj_store(const T WP_DEVICE* address, T value, const T WP_THREAD& adj_address, T WP_THREAD& adj_value)
{
    // nop; generic store() operations are not differentiable, only array_store() is
    FP_VERIFY_ADJ(value, adj_value)
}

template <typename T>
inline CUDA_CALLABLE void adj_load(const T WP_DEVICE* address, const T WP_THREAD& adj_address, T WP_THREAD& adj_value)
{
    // nop; generic load() operations are not differentiable
}

template <typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const array_t<T> WP_THREAD& buf,
    int i,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_value += T(index(adj_buf, i));
    else if (buf.grad)
        adj_value += T(index_grad(buf, i));

    FP_VERIFY_ADJ_1(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_value += T(index(adj_buf, i, j));
    else if (buf.grad)
        adj_value += T(index_grad(buf, i, j));

    FP_VERIFY_ADJ_2(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_value += T(index(adj_buf, i, j, k));
    else if (buf.grad)
        adj_value += T(index_grad(buf, i, j, k));

    FP_VERIFY_ADJ_3(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_value += T(index(adj_buf, i, j, k, l));
    else if (buf.grad)
        adj_value += T(index_grad(buf, i, j, k, l));

    FP_VERIFY_ADJ_4(value, adj_value)
}

template <typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const array_t<T> WP_THREAD& buf,
    int i,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_value -= T(index(adj_buf, i));
    else if (buf.grad)
        adj_value -= T(index_grad(buf, i));

    FP_VERIFY_ADJ_1(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_value -= T(index(adj_buf, i, j));
    else if (buf.grad)
        adj_value -= T(index_grad(buf, i, j));

    FP_VERIFY_ADJ_2(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_value -= T(index(adj_buf, i, j, k));
    else if (buf.grad)
        adj_value -= T(index_grad(buf, i, j, k));

    FP_VERIFY_ADJ_3(value, adj_value)
}
template <typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const array_t<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const array_t<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_value -= T(index(adj_buf, i, j, k, l));
    else if (buf.grad)
        adj_value -= T(index_grad(buf, i, j, k, l));

    FP_VERIFY_ADJ_4(value, adj_value)
}

// generic array types that do not support gradient computation (indexedarray, etc.)
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void
adj_address(const A1<T> WP_THREAD& buf, int i, const A2<T> WP_THREAD& adj_buf, int adj_i, const T WP_THREAD& adj_output)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_address(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    const T WP_THREAD& adj_output
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_address(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    const T WP_THREAD& adj_output
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_address(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    const T WP_THREAD& adj_output
)
{
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_array_store(
    const A1<T> WP_THREAD& buf, int i, T value, const A2<T> WP_THREAD& adj_buf, int adj_i, T WP_THREAD& adj_value
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_array_store(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_array_store(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_array_store(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value
)
{
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const A1<T> WP_THREAD& buf,
    int i,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_add(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const A1<T> WP_THREAD& buf,
    int i,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_sub(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}

// generic handler for scalar values
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_min(
    const A1<T> WP_THREAD& buf,
    int i,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i), &index(adj_buf, i), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i), &index_grad(buf, i), value, adj_value);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_min(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j), &index(adj_buf, i, j), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j), &index_grad(buf, i, j), value, adj_value);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_min(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k), &index(adj_buf, i, j, k), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k), &index_grad(buf, i, j, k), value, adj_value);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_min(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index(adj_buf, i, j, k, l), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index_grad(buf, i, j, k, l), value, adj_value);

    FP_VERIFY_ADJ_4(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_max(
    const A1<T> WP_THREAD& buf,
    int i,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i), &index(adj_buf, i), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i), &index_grad(buf, i), value, adj_value);

    FP_VERIFY_ADJ_1(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_max(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j), &index(adj_buf, i, j), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j), &index_grad(buf, i, j), value, adj_value);

    FP_VERIFY_ADJ_2(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_max(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k), &index(adj_buf, i, j, k), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k), &index_grad(buf, i, j, k), value, adj_value);

    FP_VERIFY_ADJ_3(value, adj_value)
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_max(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index(adj_buf, i, j, k, l), value, adj_value);
    else if (buf.grad)
        adj_atomic_minmax(&index(buf, i, j, k, l), &index_grad(buf, i, j, k, l), value, adj_value);

    FP_VERIFY_ADJ_4(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_cas(
    const A1<T> WP_THREAD& buf,
    int i,
    T compare,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_compare,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_cas(&index(buf, i), compare, value, &index(adj_buf, i), adj_compare, adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_cas(&index(buf, i), compare, value, &index_grad(buf, i), adj_compare, adj_value, adj_ret);

    FP_VERIFY_ADJ_1(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_cas(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    T compare,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_compare,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_cas(&index(buf, i, j), compare, value, &index(adj_buf, i, j), adj_compare, adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_cas(&index(buf, i, j), compare, value, &index_grad(buf, i, j), adj_compare, adj_value, adj_ret);

    FP_VERIFY_ADJ_2(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_cas(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T compare,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_compare,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_cas(&index(buf, i, j, k), compare, value, &index(adj_buf, i, j, k), adj_compare, adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_cas(
            &index(buf, i, j, k), compare, value, &index_grad(buf, i, j, k), adj_compare, adj_value, adj_ret
        );

    FP_VERIFY_ADJ_3(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_cas(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T compare,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_compare,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_cas(
            &index(buf, i, j, k, l), compare, value, &index(adj_buf, i, j, k, l), adj_compare, adj_value, adj_ret
        );
    else if (buf.grad)
        adj_atomic_cas(
            &index(buf, i, j, k, l), compare, value, &index_grad(buf, i, j, k, l), adj_compare, adj_value, adj_ret
        );

    FP_VERIFY_ADJ_4(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_exch(
    const A1<T> WP_THREAD& buf,
    int i,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_exch(&index(buf, i), value, &index(adj_buf, i), adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_exch(&index(buf, i), value, &index_grad(buf, i), adj_value, adj_ret);

    FP_VERIFY_ADJ_1(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_exch(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_exch(&index(buf, i, j), value, &index(adj_buf, i, j), adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_exch(&index(buf, i, j), value, &index_grad(buf, i, j), adj_value, adj_ret);

    FP_VERIFY_ADJ_2(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_exch(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_exch(&index(buf, i, j, k), value, &index(adj_buf, i, j, k), adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_exch(&index(buf, i, j, k), value, &index_grad(buf, i, j, k), adj_value, adj_ret);

    FP_VERIFY_ADJ_3(value, adj_value)
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_exch(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
    if (adj_buf.data)
        adj_atomic_exch(&index(buf, i, j, k, l), value, &index(adj_buf, i, j, k, l), adj_value, adj_ret);
    else if (buf.grad)
        adj_atomic_exch(&index(buf, i, j, k, l), value, &index_grad(buf, i, j, k, l), adj_value, adj_ret);

    FP_VERIFY_ADJ_4(value, adj_value)
}

// for bitwise operations we do not accumulate gradients
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_and(
    const A1<T> WP_THREAD& buf,
    int i,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_and(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_and(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_and(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_or(
    const A1<T> WP_THREAD& buf,
    int i,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_or(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_or(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_or(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}

template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_xor(
    const A1<T> WP_THREAD& buf,
    int i,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_xor(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_xor(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}
template <template <typename> class A1, template <typename> class A2, typename T>
inline CUDA_CALLABLE void adj_atomic_xor(
    const A1<T> WP_THREAD& buf,
    int i,
    int j,
    int k,
    int l,
    T value,
    const A2<T> WP_THREAD& adj_buf,
    int adj_i,
    int adj_j,
    int adj_k,
    int adj_l,
    T WP_THREAD& adj_value,
    const T WP_THREAD& adj_ret
)
{
}


template <template <typename> class A, typename T> CUDA_CALLABLE inline int len(const A<T> WP_THREAD& a)
{
    return a.shape[0];
}

}  // namespace wp

#include "fabric.h"

#if defined(__METAL_VERSION__)
// Descriptors read from device memory (array fields of structs stored in arrays) hold host pointers:
// copy them to thread memory and translate before use.
namespace wp {
template <typename T> inline array_t<T> metal_load_array(const array_t<T> WP_DEVICE& src)
{
    array_t<T> a = src;
    a.data = (T WP_DEVICE*)WP_METAL_TRANSLATE(a.data);
    a.grad = (T WP_DEVICE*)WP_METAL_TRANSLATE(a.grad);
    return a;
}
template <typename T> inline array_t<T> metal_load_array(const array_t<T> WP_THREAD& src)
{
    array_t<T> a = src;
    a.data = (T WP_DEVICE*)WP_METAL_TRANSLATE(a.data);
    a.grad = (T WP_DEVICE*)WP_METAL_TRANSLATE(a.grad);
    return a;
}
template <typename T> inline void wp_metal_fixup(array_t<T> WP_THREAD& a) { a = metal_load_array(a); }
template <typename T, typename... I> inline decltype(auto) index(const array_t<T> WP_DEVICE& buf, I... i)
{
    array_t<T> local = metal_load_array(buf);
    return index(local, i...);
}
template <typename T, typename... I> inline decltype(auto) address(const array_t<T> WP_DEVICE& buf, I... i)
{
    array_t<T> local = metal_load_array(buf);
    return address(local, i...);
}
template <typename T, typename... I> inline decltype(auto) view(const array_t<T> WP_DEVICE& buf, I... i)
{
    array_t<T> local = metal_load_array(buf);
    return view(local, i...);
}
template <typename T, typename... I> inline void array_store(const array_t<T> WP_DEVICE& buf, I... i)
{
    array_t<T> local = metal_load_array(buf);
    array_store(local, i...);
}
template <typename T, typename... I> inline decltype(auto) atomic_add(const array_t<T> WP_DEVICE& buf, I... i)
{
    array_t<T> local = metal_load_array(buf);
    return atomic_add(local, i...);
}
template <typename T, typename... I> inline decltype(auto) atomic_sub(const array_t<T> WP_DEVICE& buf, I... i)
{
    array_t<T> local = metal_load_array(buf);
    return atomic_sub(local, i...);
}
template <typename T, typename... I> inline decltype(auto) atomic_min(const array_t<T> WP_DEVICE& buf, I... i)
{
    array_t<T> local = metal_load_array(buf);
    return atomic_min(local, i...);
}
template <typename T, typename... I> inline decltype(auto) atomic_max(const array_t<T> WP_DEVICE& buf, I... i)
{
    array_t<T> local = metal_load_array(buf);
    return atomic_max(local, i...);
}
template <typename T> inline int len(const array_t<T> WP_DEVICE& buf) { return buf.shape[0]; }
template <typename T> inline indexedarray_t<T> metal_load_array(const indexedarray_t<T> WP_DEVICE& src)
{
    indexedarray_t<T> a = src;
    a.arr = metal_load_array(src.arr);
    for (int k = 0; k < ARRAY_MAX_DIMS; ++k)
        a.indices[k] = (int WP_DEVICE*)WP_METAL_TRANSLATE(a.indices[k]);
    return a;
}
template <typename T> inline indexedarray_t<T> metal_load_array(const indexedarray_t<T> WP_THREAD& src)
{
    indexedarray_t<T> a = src;
    a.arr = metal_load_array(a.arr);
    for (int k = 0; k < ARRAY_MAX_DIMS; ++k)
        a.indices[k] = (int WP_DEVICE*)WP_METAL_TRANSLATE(a.indices[k]);
    return a;
}
template <typename T> inline void wp_metal_fixup(indexedarray_t<T> WP_THREAD& a) { a = metal_load_array(a); }
template <typename T, typename... I> inline decltype(auto) index(const indexedarray_t<T> WP_DEVICE& buf, I... i)
{
    indexedarray_t<T> local = metal_load_array(buf);
    return index(local, i...);
}
template <typename T, typename... I> inline decltype(auto) address(const indexedarray_t<T> WP_DEVICE& buf, I... i)
{
    indexedarray_t<T> local = metal_load_array(buf);
    return address(local, i...);
}
template <typename T, typename... I> inline void array_store(const indexedarray_t<T> WP_DEVICE& buf, I... i)
{
    indexedarray_t<T> local = metal_load_array(buf);
    array_store(local, i...);
}
}  // namespace wp
#endif
