// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// tile_solve: cooperative scalar triangular solves (forward / back
// substitution) and the tile_lower_solve / tile_upper_solve entry templates
// (and inplace variants).
//
// Performance: cooperative scalar vs cuSolverDx (L40, sm_89, fp64,
// block_dim=32, m_rhs=8, batch=64)
//
//   tile_lower_solve  (matrix RHS)
//     n   mathdx (us)   scalar (us)   scalar/mathdx
//     4         16.2          16.6         1.02x
//     8         16.2          18.2         1.12x
//    16         16.9          22.9         1.35x
//    32         19.3          36.8         1.90x
//    64         26.8          83.3         3.11x
//
//   tile_cholesky_solve  (full LL^T solve = forward + back substitution)
//     n   mathdx (us)   scalar (us)   scalar/mathdx
//     4         16.1          17.2         1.07x
//     8         16.7          21.1         1.27x
//    16         18.4          30.3         1.65x
//    32         22.4          58.0         2.60x
//    64         34.7         153.3         4.42x
//
// cuSolverDx wins at every size we measured. The gap is within noise at n<=8,
// modest at n=16 (~1.4x for trsm, ~1.7x for cholesky_solve), and grows
// quickly past that. Triangular solves are inherently serial in the row
// dimension; cooperative parallelism only helps the inner dot product (vector
// RHS) or the outer column dimension (matrix RHS), neither of which catches
// the cuSolverDx LTO at the sizes measured here.
//
// The cooperative scalar path is therefore primarily a correctness fallback
// for builds without libmathdx and for users who want to skip the slow LTO
// compilation cost during development. Users can route a kernel through the
// scalar path on a libmathdx-enabled build by setting the module option
// `enable_mathdx_solver=False` (or globally via
// `wp.config.enable_mathdx_solver = False`).

#pragma once

#include "tile.h"

#ifdef __clang__
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif  // __clang__

namespace wp {

namespace partitioned_gemm {

// Writes into X (the result vector or matrix).
//
// Two paths gated on the rank of TileY:
//   - Vector RHS (Shape::N == 1): the outer i loop is sequential (each x[i]
//     depends on x[0..i-1]); the only plausible parallelization opportunity
//     is the inner dot product. We intentionally keep that sequential too --
//     gating the whole path on a thread-0 + WP_TILE_SYNC -- to avoid the
//     per-row reduction overhead (n syncs) that does not obviously pay back
//     at typical tile sizes (n <= 64, block_dim = 32). Defer a benchmark-
//     gated reduction follow-up if profiling shows vector-RHS solves are a
//     bottleneck on GPU-without-mathdx.
//   - Matrix RHS (Shape::N == 2): the outer k loop iterates over m
//     independent columns; distribute that loop across threads. Each thread
//     runs the existing inner i loop sequentially.
//
// On CPU, WP_TILE_BLOCK_DIM == 1 collapses the matrix-RHS thread-strided loop
// to plain sequential, the vector-RHS thread-0 gate executes on the only
// thread, and WP_TILE_SYNC() is a no-op -- behaviour matches the prior
// single-threaded scalar fallback.
template <bool Upper, typename TileA, typename TileX, typename TileY>
inline CUDA_CALLABLE void
scalar_cholesky_forward_substitution(TileA WP_THREAD& A, TileX WP_THREAD& X, TileY WP_THREAD& Y)
{
    using T = typename TileA::Type;

    auto idx = [](int row, int col) { return Upper ? tile_coord(col, row) : tile_coord(row, col); };

    if constexpr (TileY::Layout::Shape::N == 1) {
        constexpr int n = TileA::Layout::Shape::dim(1);

        // Column sweep: once x[i] is known every thread eliminates it from its share of the
        // remaining right-hand side, so the O(n^2) work is split across the block. X doubles as
        // the working copy of Y (callers may alias them).
        for (int i = WP_TILE_THREAD_IDX; i < n; i += WP_TILE_BLOCK_DIM)
            X.data(tile_coord(i)) = Y.data(tile_coord(i));
        WP_TILE_SYNC();
        for (int i = 0; i < n; ++i) {
            if (WP_TILE_THREAD_IDX == 0) {
                T diag = A.data(idx(i, i));
                if (diag != T(0.0f))
                    X.data(tile_coord(i)) /= diag;
            }
            WP_TILE_SYNC();
            const T xi = X.data(tile_coord(i));
            for (int k = i + 1 + WP_TILE_THREAD_IDX; k < n; k += WP_TILE_BLOCK_DIM)
                X.data(tile_coord(k)) -= A.data(idx(k, i)) * xi;
            WP_TILE_SYNC();
        }
    } else if constexpr (TileY::Layout::Shape::N == 2) {
        constexpr int n = TileA::Layout::Shape::dim(1);
        constexpr int m = TileY::Layout::Shape::dim(1);

        for (int k = WP_TILE_THREAD_IDX; k < m; k += WP_TILE_BLOCK_DIM) {
            for (int i = 0; i < n; ++i) {
                T s = Y.data(tile_coord(i, k));

                for (int j = 0; j < i; ++j)
                    s -= A.data(idx(i, j)) * X.data(tile_coord(j, k));

                T diag = A.data(idx(i, i));
                X.data(tile_coord(i, k)) = (diag != T(0.0f)) ? s / diag : s;
            }
        }
        WP_TILE_SYNC();
    }
}

// Reads and writes X.
//
// Same cooperative split as scalar_cholesky_forward_substitution: vector RHS
// is fully gated on thread 0 (the outer i loop is sequential and the inner
// dot product alone doesn't pay for the reduction overhead at typical n);
// matrix RHS distributes the outer k loop across threads.
template <bool Upper, typename TileA, typename TileX>
inline CUDA_CALLABLE void scalar_cholesky_back_substitution(TileA WP_THREAD& A, TileX WP_THREAD& X)
{
    using T = typename TileA::Type;

    auto idx = [](int row, int col) { return Upper ? tile_coord(row, col) : tile_coord(col, row); };

    if constexpr (TileX::Layout::Shape::N == 1) {
        constexpr int n = TileA::Layout::Shape::dim(1);

        // Column sweep in reverse (see scalar_cholesky_forward_substitution).
        for (int i = n - 1; i >= 0; --i) {
            if (WP_TILE_THREAD_IDX == 0) {
                T diag = A.data(idx(i, i));
                if (diag != T(0.0f))
                    X.data(tile_coord(i)) /= diag;
            }
            WP_TILE_SYNC();
            const T xi = X.data(tile_coord(i));
            for (int k = WP_TILE_THREAD_IDX; k < i; k += WP_TILE_BLOCK_DIM)
                X.data(tile_coord(k)) -= A.data(idx(k, i)) * xi;
            WP_TILE_SYNC();
        }
    } else if constexpr (TileX::Layout::Shape::N == 2) {
        constexpr int n = TileA::Layout::Shape::dim(1);
        constexpr int m = TileX::Layout::Shape::dim(1);

        for (int k = WP_TILE_THREAD_IDX; k < m; k += WP_TILE_BLOCK_DIM) {
            for (int i = n - 1; i >= 0; --i) {
                T s = X.data(tile_coord(i, k));

                for (int j = i + 1; j < n; ++j)
                    s -= A.data(idx(i, j)) * X.data(tile_coord(j, k));

                T diag = A.data(idx(i, i));
                X.data(tile_coord(i, k)) = (diag != T(0.0f)) ? s / diag : s;
            }
        }
        WP_TILE_SYNC();
    }
}

#if defined(__METAL_VERSION__)
#ifndef WP_METAL_REGISTER_CHOLESKY_MAX
#define WP_METAL_REGISTER_CHOLESKY_MAX 40
#endif
#ifndef WP_METAL_REGISTER_SOLVE
#define WP_METAL_REGISTER_SOLVE 1
#endif
// Register triangular solves for blocks of at most one SIMD group (MetalSim). Lane l holds columns
// l, l+BD, ... of L (= U^T when the factor is stored upper) in registers, as metal_register_cholesky
// does, together with the right-hand side and solution entries of the same indices.
//   forward, L y = b:   row i's dot product is a per-lane partial over the lane's columns jc < i and a
//                       SIMD-group sum; the owner of column i divides by L[i, i].
//   backward, L^T x = y: from the last row down, x_i is broadcast from its owner and every lane
//                       subtracts L[i, jc] x_i from the right-hand sides of its columns jc < i.
// No threadgroup memory traffic and no barriers inside the sweeps; the cooperative path above pays
// two barriers per row per sweep. The row loops are unrolled by template recursion so that every
// register-array index is a compile-time constant. Summation order differs from the cooperative
// path (float noise). Disabled per module with WP_METAL_REGISTER_SOLVE 0 (warp.config.metal_register_solve).
template <int I, int N, int CPL, int BD, typename T>
inline WP_FORCE_INLINE void
metal_register_forward_step(thread const T (&col)[CPL][N], thread T (&y)[CPL], thread const T (&b)[CPL], int lane)
{
    if constexpr (I < N) {
        constexpr int owner = I % BD;
        constexpr int ci = I / BD;
        T partial = T {};
#pragma clang loop unroll(full)
        for (int c = 0; c < CPL; ++c) {
            const int jc = lane + c * BD;
            if (jc < I)
                partial += col[c][I] * y[c];
        }
        const T total = metal::simd_sum(partial);
        if (lane == owner) {
            const T diag = col[ci][I];
            const T v = b[ci] - total;
            y[ci] = (diag != T(0)) ? v / diag : v;
        }
        metal_register_forward_step<I + 1, N, CPL, BD, T>(col, y, b, lane);
    }
}

template <int I, int N, int CPL, int BD, typename T>
inline WP_FORCE_INLINE void
metal_register_backward_step(thread const T (&col)[CPL][N], thread T (&acc)[CPL], thread T (&x)[CPL], int lane)
{
    if constexpr (I >= 0) {
        constexpr int owner = I % BD;
        constexpr int ci = I / BD;
        T v = T {};
        if (lane == owner) {
            const T diag = col[ci][I];
            v = (diag != T(0)) ? acc[ci] / diag : acc[ci];
            x[ci] = v;
        }
        const T xi = metal::simd_shuffle(v, ushort(owner));
#pragma clang loop unroll(full)
        for (int c = 0; c < CPL; ++c) {
            const int jc = lane + c * BD;
            if (jc < I)
                acc[c] -= col[c][I] * xi;
        }
        metal_register_backward_step<I - 1, N, CPL, BD, T>(col, acc, x, lane);
    }
}

// Two columns per lane, compact (see metal_register_cholesky2): c1 holds rows i >= BD of column lane + BD.
template <int I, int N, int BD, typename T>
inline WP_FORCE_INLINE void
metal_register_forward_step2(thread const T (&c0)[N], thread const T (&c1)[N - BD], thread T (&y)[2], thread const T (&b)[2], int lane)
{
    if constexpr (I < N) {
        constexpr int owner = I % BD;
        constexpr int ci = I / BD;
        T partial = T {};
        if (lane < I)
            partial += c0[I] * y[0];
        if constexpr (I > BD) {
            if (lane + BD < I)
                partial += c1[I - BD] * y[1];
        }
        const T total = metal::simd_sum(partial);
        if (lane == owner) {
            const T diag = (ci == 0) ? c0[I] : c1[(I >= BD) ? (I - BD) : 0];
            const T v = b[ci] - total;
            y[ci] = (diag != T(0)) ? v / diag : v;
        }
        metal_register_forward_step2<I + 1, N, BD, T>(c0, c1, y, b, lane);
    }
}

template <int I, int N, int BD, typename T>
inline WP_FORCE_INLINE void
metal_register_backward_step2(thread const T (&c0)[N], thread const T (&c1)[N - BD], thread T (&acc)[2], thread T (&x)[2], int lane)
{
    if constexpr (I >= 0) {
        constexpr int owner = I % BD;
        constexpr int ci = I / BD;
        T v = T {};
        if (lane == owner) {
            const T diag = (ci == 0) ? c0[I] : c1[(I >= BD) ? (I - BD) : 0];
            v = (diag != T(0)) ? acc[ci] / diag : acc[ci];
            x[ci] = v;
        }
        const T xi = metal::simd_shuffle(v, ushort(owner));
        if (lane < I)
            acc[0] -= c0[I] * xi;
        if constexpr (I > BD) {
            if (lane + BD < I)
                acc[1] -= c1[I - BD] * xi;
        }
        metal_register_backward_step2<I - 1, N, BD, T>(c0, c1, acc, x, lane);
    }
}

template <bool Upper, typename TileA, typename TileX, typename TileY>
inline WP_FORCE_INLINE void metal_register_cholesky_solve2(TileA WP_THREAD& A, TileX WP_THREAD& X, TileY WP_THREAD& Y)
{
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);
    constexpr int BD = WP_TILE_BLOCK_DIM;
    const int lane = WP_TILE_THREAD_IDX;
    const int jc1 = lane + BD;

    T c0[n];
    T c1[n - BD];
    T b[2];
    T y[2];
    T x[2];
    b[0] = Y.data(tile_coord(lane));
    b[1] = (jc1 < n) ? Y.data(tile_coord(jc1)) : T {};
    y[0] = y[1] = x[0] = x[1] = T {};
#pragma clang loop unroll(full)
    for (int i = 0; i < n; ++i)
        c0[i] = (i >= lane) ? (Upper ? A.data(tile_coord(lane, i)) : A.data(tile_coord(i, lane))) : T {};
#pragma clang loop unroll(full)
    for (int i = BD; i < n; ++i)
        c1[i - BD] = (jc1 < n && i >= jc1) ? (Upper ? A.data(tile_coord(jc1, i)) : A.data(tile_coord(i, jc1))) : T {};
    WP_TILE_SYNC();  // X may alias Y

    metal_register_forward_step2<0, n, BD, T>(c0, c1, y, b, lane);
    metal_register_backward_step2<n - 1, n, BD, T>(c0, c1, y, x, lane);

    X.data(tile_coord(lane)) = x[0];
    if (jc1 < n)
        X.data(tile_coord(jc1)) = x[1];
    WP_TILE_SYNC();
}

#ifndef WP_METAL_COMPACT_REGISTER_CHOLESKY
#define WP_METAL_COMPACT_REGISTER_CHOLESKY 0
#endif

template <bool Upper, typename TileA, typename TileX, typename TileY>
inline WP_FORCE_INLINE void metal_register_cholesky_solve(TileA WP_THREAD& A, TileX WP_THREAD& X, TileY WP_THREAD& Y)
{
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);
    constexpr int BD = WP_TILE_BLOCK_DIM;
    constexpr int CPL = (n + BD - 1) / BD;  // columns per lane
    if constexpr (WP_METAL_COMPACT_REGISTER_CHOLESKY && CPL == 2) {
        metal_register_cholesky_solve2<Upper>(A, X, Y);
        return;
    }
    const int lane = WP_TILE_THREAD_IDX;

    // L[i, jc] for i >= jc is A(jc, i) when A holds U (Upper), A(i, jc) otherwise
    T col[CPL][n];
    T b[CPL];
    T y[CPL];
    T x[CPL];
#pragma clang loop unroll(full)
    for (int c = 0; c < CPL; ++c) {
        const int jc = lane + c * BD;
        b[c] = (jc < n) ? Y.data(tile_coord(jc)) : T {};
        y[c] = T {};
        x[c] = T {};
#pragma clang loop unroll(full)
        for (int i = 0; i < n; ++i)
            col[c][i] = (jc < n && i >= jc) ? (Upper ? A.data(tile_coord(jc, i)) : A.data(tile_coord(i, jc))) : T {};
    }
    WP_TILE_SYNC();  // X may alias Y: every lane has read its right-hand side before any lane writes

    metal_register_forward_step<0, n, CPL, BD, T>(col, y, b, lane);
    metal_register_backward_step<n - 1, n, CPL, BD, T>(col, y, x, lane);

#pragma clang loop unroll(full)
    for (int c = 0; c < CPL; ++c) {
        const int jc = lane + c * BD;
        if (jc < n)
            X.data(tile_coord(jc)) = x[c];
    }
    WP_TILE_SYNC();
}
#endif  // __METAL_VERSION__

template <bool Upper, typename TileA, typename TileX, typename TileY>
inline CUDA_CALLABLE void scalar_cholesky_solve(TileA WP_THREAD& A, TileX WP_THREAD& X, TileY WP_THREAD& Y)
{
#if defined(__METAL_VERSION__)
    if constexpr (WP_METAL_REGISTER_SOLVE && TileY::Layout::Shape::N == 1 && WP_TILE_BLOCK_DIM > 1 && WP_TILE_BLOCK_DIM <= 32
                  && TileA::Layout::Shape::dim(1) <= WP_METAL_REGISTER_CHOLESKY_MAX) {
        metal_register_cholesky_solve<Upper>(A, X, Y);
        return;
    }
#endif
    scalar_cholesky_forward_substitution<Upper>(A, X, Y);
    scalar_cholesky_back_substitution<Upper>(A, X);
}


}  // namespace partitioned_gemm


template <typename Fwd, typename Bkwd, typename TileL, typename TileY, typename TileZ>
TileZ WP_THREAD&
tile_lower_solve(Fwd fun_forward, Bkwd fun_bkwd, TileL WP_THREAD& L, TileY WP_THREAD& y, TileZ WP_THREAD& z)
{
    // Copy y to z
    z = y;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_forward_substitution<false>(L, z, y);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_forward_substitution<false>(L, z, y);
    } else {
        WP_TILE_SYNC();
        fun_forward(L.data.ptr, z.data.ptr);
        WP_TILE_SYNC();
    }
#endif

    return z;
}

template <typename Fwd, typename TileL, typename TileY>
void tile_lower_solve_inplace(Fwd fun_forward, TileL WP_THREAD& L, TileY WP_THREAD& y)
{
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_forward_substitution<false>(L, y, y);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_forward_substitution<false>(L, y, y);
    } else {
        WP_TILE_SYNC();
        fun_forward(L.data.ptr, y.data.ptr);
        WP_TILE_SYNC();
    }
#endif
}

// Adjoint of the out-of-place lower solve L z = y.
//
// Two leading func params (Fwd, Bkwd) mirror the two-func-var dispatch tuple in
// builtins.py (var_fwd, var_bkwd); Bkwd is the backward TRSM that solves the
// transposed system L^T w = adj_ret.
template <
    typename Fwd,
    typename Bkwd,
    typename TileL,
    typename TileY,
    typename TileZ,
    typename AdjFwd,
    typename AdjBkwd,
    typename AdjTileL,
    typename AdjTileY,
    typename AdjTileZ,
    typename AdjRet>
CUDA_CALLABLE void adj_tile_lower_solve(
    Fwd fun_forward,
    Bkwd fun_bkwd,
    TileL WP_THREAD& L,
    TileY WP_THREAD& y,
    TileZ WP_THREAD& z,
    AdjFwd adj_fun_forward,
    AdjBkwd adj_fun_bkwd,
    AdjTileL WP_THREAD& adj_L,
    AdjTileY WP_THREAD& adj_y,
    AdjTileZ WP_THREAD& adj_z,
    AdjRet WP_THREAD& adj_ret
)
{
    WP_TILE_ARENA_NULL
    using T = typename AdjRet::Type;

    // n = matrix dimension, nrhs = number of right-hand sides (1 for a vector RHS).
    constexpr int n = TileL::Layout::Shape::dim(1);
    constexpr int nrhs = (TileZ::Layout::Shape::N == 1) ? 1 : TileZ::Layout::Shape::dim(1);

    // Raw scratch for the transposed solve L^T W = adj_ret. Cooperative CPU
    // fibers must share this storage across their thread-strided phases.
#if defined(__CUDA_ARCH__)
    __shared__ T W[n * nrhs];
#else
    T W_local[WP_TILE_BLOCK_DIM == 1 ? n * nrhs : 1];
    T WP_THREAD* W;
    if constexpr (WP_TILE_BLOCK_DIM == 1)
        W = W_local;
    else
        W = (T WP_TILE_SHARED*)WP_TILE_ALLOC(int(sizeof(T) * n * nrhs));
#endif

    // Give the scalar fallback tile indexing without allocating tile storage.
    using WLayout = tile_layout_strided_t<typename TileZ::Layout::Shape>;
    tile_shared_t<T, WLayout, false> W_tile(W);

    // Preload the incoming gradient adj_ret into W (row-major, contiguous).
    for (int idx = WP_TILE_THREAD_IDX; idx < n * nrhs; idx += WP_TILE_BLOCK_DIM) {
        if constexpr (TileZ::Layout::Shape::N == 1) {
            W[idx] = adj_ret.grad(tile_coord(idx));
        } else {
            W[idx] = adj_ret.grad(tile_coord(idx / nrhs, idx % nrhs));
        }
    }
    WP_TILE_SYNC();

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    // scalar path
    partitioned_gemm::scalar_cholesky_back_substitution<false>(L, W_tile);
#else
    if constexpr (wp_is_null_func<Bkwd>::value) {
        partitioned_gemm::scalar_cholesky_back_substitution<false>(L, W_tile);
    } else {
        // MathDx path: in-place TRSM into the raw scratch.
        fun_bkwd(L.data.ptr, W);
        WP_TILE_SYNC();
    }
#endif

    // Accumulate the rhs gradient: adj_y += W (element-wise, any rank).
    for (int idx = WP_TILE_THREAD_IDX; idx < n * nrhs; idx += WP_TILE_BLOCK_DIM) {
        if constexpr (TileZ::Layout::Shape::N == 1) {
            adj_y.grad(tile_coord(idx)) += W[idx];
        } else {
            adj_y.grad(tile_coord(idx / nrhs, idx % nrhs)) += W[idx];
        }
    }
    WP_TILE_SYNC();

    // Accumulate the factor gradient: adj_L -= tril(W Z^T), lower triangle only.
    // Vector RHS (nrhs == 1) recovers adj_L[i, j] -= W[i] * z[j].
    for (int idx = WP_TILE_THREAD_IDX; idx < n * n; idx += WP_TILE_BLOCK_DIM) {
        int row = idx / n;
        int col = idx % n;
        if (row >= col) {
            T s = T(0);
            for (int k = 0; k < nrhs; ++k) {
                if constexpr (TileZ::Layout::Shape::N == 1) {
                    s += W[row] * z.data(tile_coord(col));
                } else {
                    s += W[row * nrhs + k] * z.data(tile_coord(col, k));
                }
            }
            adj_L.grad(tile_coord(row, col)) -= s;
        }
    }
    WP_TILE_SYNC();

#if !defined(__CUDA_ARCH__)
    if constexpr (WP_TILE_BLOCK_DIM > 1)
        WP_TILE_ALLOC(-int(sizeof(T) * n * nrhs));
#endif
}

template <typename Fwd, typename TileL, typename TileY, typename AdjFwd, typename AdjTileL, typename AdjTileY>
void adj_tile_lower_solve_inplace(
    Fwd fun_forward,
    TileL WP_THREAD& L,
    TileY WP_THREAD& y,
    AdjFwd adj_fun_forward,
    AdjTileL WP_THREAD& adj_L,
    AdjTileY WP_THREAD& adj_y
)
{
    // MISSINGADJOINT: same math as adj_tile_lower_solve but operating in place on
    // adj_y; adj_L -= outer(adj_y_new, y_pre_solve)
}


template <typename Fwd, typename TileU, typename TileZ, typename TileX>
TileX WP_THREAD& tile_upper_solve(Fwd fun_forward, TileU WP_THREAD& U, TileZ WP_THREAD& z, TileX WP_THREAD& x)
{
    // Copy z to x
    x = z;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    {
        auto L = tile_transpose(U);
        partitioned_gemm::scalar_cholesky_back_substitution<false>(L, x);
    }
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        auto L = tile_transpose(U);
        partitioned_gemm::scalar_cholesky_back_substitution<false>(L, x);
    } else {
        WP_TILE_SYNC();
        fun_forward(U.data.ptr, x.data.ptr);
        WP_TILE_SYNC();
    }
#endif

    return x;
}

template <typename Fwd, typename TileU, typename TileZ>
void tile_upper_solve_inplace(Fwd fun_forward, TileU WP_THREAD& U, TileZ WP_THREAD& z)
{
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    {
        auto L = tile_transpose(U);
        partitioned_gemm::scalar_cholesky_back_substitution<false>(L, z);
    }
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        auto L = tile_transpose(U);
        partitioned_gemm::scalar_cholesky_back_substitution<false>(L, z);
    } else {
        WP_TILE_SYNC();
        fun_forward(U.data.ptr, z.data.ptr);
        WP_TILE_SYNC();
    }
#endif
}

template <
    typename Fwd,
    typename TileU,
    typename TileZ,
    typename TileX,
    typename AdjFwd,
    typename AdjTileU,
    typename AdjTileZ,
    typename AdjTileX,
    typename AdjRet>
void adj_tile_upper_solve(
    Fwd fun_forward,
    TileU WP_THREAD& U,
    TileZ WP_THREAD& z,
    TileX WP_THREAD& x,
    AdjFwd adj_fun_forward,
    AdjTileU WP_THREAD& adj_U,
    AdjTileZ WP_THREAD& adj_z,
    AdjTileX WP_THREAD& adj_x,
    AdjRet WP_THREAD& adj_ret
)
{
    // MISSINGADJOINT: adjoint is the transposed (lower) solve U^T y = adj_ret; then adj_z
    // += y and adj_U -= outer(x, y)
}

template <typename Fwd, typename TileU, typename TileZ, typename AdjFwd, typename AdjTileU, typename AdjTileZ>
void adj_tile_upper_solve_inplace(
    Fwd fun_forward,
    TileU WP_THREAD& U,
    TileZ WP_THREAD& z,
    AdjFwd adj_fun_forward,
    AdjTileU WP_THREAD& adj_U,
    AdjTileZ WP_THREAD& adj_z
)
{
    // MISSINGADJOINT: same math as adj_tile_upper_solve but operating in place on
    // adj_z; adj_U -= outer(z_post_solve, adj_z_new)
}


template <bool Upper, typename Fwd, typename TileA, typename TileY, typename TileX>
TileX WP_THREAD& tile_cholesky_solve(Fwd fun_forward, TileA WP_THREAD& A, TileY WP_THREAD& Y, TileX WP_THREAD& X)
{
    // Copy y to x

    X = Y;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_solve<Upper>(A, X, Y);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_solve<Upper>(A, X, Y);
    } else {
        WP_TILE_SYNC();
        fun_forward(A.data.ptr, X.data.ptr);
        WP_TILE_SYNC();
    }
#endif

    return X;
}

template <bool Upper, typename Fwd, typename TileA, typename TileY>
void tile_cholesky_solve_inplace(Fwd fun_forward, TileA WP_THREAD& A, TileY WP_THREAD& Y)
{
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_solve<Upper>(A, Y, Y);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_solve<Upper>(A, Y, Y);
    } else {
        WP_TILE_SYNC();
        fun_forward(A.data.ptr, Y.data.ptr);
        WP_TILE_SYNC();
    }
#endif
}

template <
    bool Upper,
    typename Fwd,
    typename TileA,
    typename TileY,
    typename TileX,
    typename AdjFwd,
    typename AdjTileA,
    typename AdjTileY,
    typename AdjTileX,
    typename AdjRet>
void adj_tile_cholesky_solve(
    Fwd fun_forward,
    TileA WP_THREAD& A,
    TileY WP_THREAD& Y,
    TileX WP_THREAD& X,
    AdjFwd adj_fun_forward,
    AdjTileA WP_THREAD& adj_A,
    AdjTileY WP_THREAD& adj_Y,
    AdjTileX WP_THREAD& adj_X,
    AdjRet WP_THREAD& adj_ret
)
{
    // MISSINGADJOINT: implicit differentiation through A X = Y: solve A Z = adj_ret,
    // then adj_Y += Z and adj_A -= sym(outer(Z, X))
}

template <
    bool Upper,
    typename Fwd,
    typename TileA,
    typename TileY,
    typename AdjFwd,
    typename AdjTileA,
    typename AdjTileY>
void adj_tile_cholesky_solve_inplace(
    Fwd fun_forward,
    TileA WP_THREAD& A,
    TileY WP_THREAD& Y,
    AdjFwd adj_fun_forward,
    AdjTileA WP_THREAD& adj_A,
    AdjTileY WP_THREAD& adj_Y
)
{
    // MISSINGADJOINT: same math as adj_tile_cholesky_solve operating in place
    // on adj_Y; adj_A -= sym(outer(adj_Y_new, Y_post_solve))
}


}  // namespace wp

#ifdef __clang__
#pragma clang diagnostic pop
#endif
