// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// tile_cholesky: cooperative scalar Cholesky factorization, solve helper, and
// adjoint, plus the tile_cholesky / tile_cholesky_inplace entry templates and
// the tile_cholesky adjoint dispatch.
//
// Performance: cooperative scalar vs cuSolverDx (L40, sm_89, fp64, block_dim=32)
//
//   tile_cholesky  (factorization)
//     n   mathdx (us)   scalar (us)   scalar/mathdx
//     4         15.0          15.9         1.06x
//     8         16.1          19.0         1.18x
//    16         18.1          27.5         1.52x
//    32         25.3          53.9         2.13x
//    64         54.3         159.4         2.94x
//
//   adj_tile_cholesky  (Murray 2016 derivative, gemm + 2 trsm composition)
//     measured indirectly via tile_cholesky_solve (n=8, m_rhs=8, batch=64):
//     mathdx 16.7us, scalar 21.1us, 1.27x; widens to 4.4x at n=64.
//
// cuSolverDx wins at every size we measured. The gap is within noise at n<=8,
// modest at n=16 (~1.5x), and grows quickly past that. Cholesky has more
// inherent serialization than matmul (sequential outer column loop in
// factorization, two sequential triangular solves in the adjoint), so the
// matmul-style "scalar wins at small tiles" pattern does not appear here.
//
// The cooperative scalar path is therefore primarily a correctness fallback
// for builds without libmathdx and for users who want to skip the slow LTO
// compilation cost during development. Users can route a kernel through the
// scalar path on a libmathdx-enabled build by setting the module option
// `enable_mathdx_solver=False` (or globally via
// `wp.config.enable_mathdx_solver = False`).

#pragma once

#include "tile.h"
#include "tile_matmul.h"
#include "tile_solve.h"

#ifdef __clang__
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif  // __clang__

namespace wp {

namespace partitioned_gemm {


// Scalar Cholesky factorization, cooperative across WP_TILE_BLOCK_DIM threads.
//
// Upper=false: A = L L^T, L is lower triangular
// Upper=true:  A = U^T U, U is upper triangular
//
// Cooperative structure:
//   - Outer j loop (column index): SEQUENTIAL -- column j depends on
//     columns 0..j-1. All threads execute it in lockstep.
//   - Diagonal element L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2):
//     ALL threads compute s and invS = 1/s redundantly into local registers.
//     Reads of A[j,j] and previously-written L[j,k] (k<j) are race-free since
//     this iteration is past the WP_TILE_SYNC at the end of iteration j-1.
//     Only thread 0 writes Out[j,j] = s. No shared-mem cell, no extra sync.
//
//     Tradeoff: a thread-0-broadcast variant via __shared__ T s_cell would
//     skip the redundant compute but plumb a shared-mem cell and add an extra
//     sync per j. A parallel-reduction variant of the diagonal sum is also
//     possible. Redundant compute is O(n^2/2) ops per factorization (~2000 ops
//     at n=64) -- negligible vs the row-update work, so we pay it for code
//     simplicity. Revisit if a benchmark shows the diagonal compute is hot.
//   - Inner i = j+1..n loop (row updates): distributed over threads. Each
//     thread owns a strided subset of i values.
//   - "Zero opposite triangle" pass: distributed over threads.
//   - WP_TILE_SYNC() at the end of each j so the next column sees finished
//     writes from this j.
//
// On CPU, WP_TILE_BLOCK_DIM == 1 collapses the thread-strided inner loops to
// plain sequential, the thread-0 diagonal write executes on the only thread,
// and WP_TILE_SYNC() is a no-op -- behaviour matches the prior single-threaded
// scalar fallback.
#if defined(__METAL_VERSION__)
// Largest matrix dimension factored by the register path below (larger ones use the cooperative
// scalar path with a threadgroup barrier per column). Set per module from warp.config.
// metal_register_cholesky_max (codegen defines it; the Python scratch sizing reads the same value).
#ifndef WP_METAL_REGISTER_CHOLESKY_MAX
#define WP_METAL_REGISTER_CHOLESKY_MAX 40
#endif
// Register Cholesky for blocks of at most one SIMD group: lane l owns columns l, l+BD, ... of the
// factor in registers; the owner of column j scales it, the column is broadcast with SIMD shuffles
// and every lane applies the rank-1 update to its own columns. No threadgroup memory and no
// barriers during the factorization, so a world's latency no longer depends on how many
// threadgroups fit in a core. The column loop is unrolled by template recursion so that every
// register-array index is a compile-time constant (dynamic indexing would spill to thread memory).
template <int J, int N, int CPL, int BD, typename T>
inline WP_FORCE_INLINE void metal_register_cholesky_step(thread T (&col)[CPL][N], int lane)
{
    if constexpr (J < N) {
        constexpr int owner = J % BD;
        constexpr int cj = J / BD;
        if (lane == owner) {
            const T d = wp::sqrt(col[cj][J]);
            const T inv = T(1) / d;
            col[cj][J] = d;
#pragma clang loop unroll(full)
            for (int i = J + 1; i < N; ++i)
                col[cj][i] *= inv;
        }
        T ljc[CPL];
#pragma clang loop unroll(full)
        for (int c = 0; c < CPL; ++c)
            ljc[c] = T {};
#pragma clang loop unroll(full)
        for (int i = J; i < N; ++i) {
            const T lij = metal::simd_shuffle(col[cj][i], ushort(owner));
#pragma clang loop unroll(full)
            for (int c = 0; c < CPL; ++c) {
                const int jc = lane + c * BD;
                if (i == jc)
                    ljc[c] = lij;  // L[jc, J], reached before any row i > jc of column jc
                if (jc > J && i >= jc)
                    col[c][i] -= lij * ljc[c];
            }
        }
        metal_register_cholesky_step<J + 1, N, CPL, BD, T>(col, lane);
    }
}

// Two columns per lane (33..64 rows in a 32-lane group), compact: the lane's second column jc1 = lane + BD
// only has rows i >= BD, so it is held as c1[i - BD] (N - BD registers instead of N) and the unrolled
// update loops skip the rows it cannot have. Same operations on the same elements in the same order as the
// generic form above (bitwise). Rows are compile-time (template recursion over J, unrolled i).
template <int J, int N, int BD, typename T>
inline WP_FORCE_INLINE void metal_register_cholesky_step2(thread T (&c0)[N], thread T (&c1)[N - BD], int lane)
{
    if constexpr (J < N) {
        constexpr int owner = J % BD;
        constexpr bool second = J >= BD;
        if (lane == owner) {
            if constexpr (second) {
                const T d = wp::sqrt(c1[J - BD]);
                const T inv = T(1) / d;
                c1[J - BD] = d;
#pragma clang loop unroll(full)
                for (int i = J + 1; i < N; ++i)
                    c1[i - BD] *= inv;
            } else {
                const T d = wp::sqrt(c0[J]);
                const T inv = T(1) / d;
                c0[J] = d;
#pragma clang loop unroll(full)
                for (int i = J + 1; i < N; ++i)
                    c0[i] *= inv;
            }
        }
        const int jc0 = lane;
        const int jc1 = lane + BD;
        T l0 = T {};
        T l1 = T {};
#pragma clang loop unroll(full)
        for (int i = J; i < N; ++i) {
            T lij;
            if constexpr (second)
                lij = metal::simd_shuffle(c1[i - BD], ushort(owner));
            else
                lij = metal::simd_shuffle(c0[i], ushort(owner));
            if (i == jc0)
                l0 = lij;
            if (jc0 > J && i >= jc0)
                c0[i] -= lij * l0;
            if (i >= BD) {
                if (i == jc1)
                    l1 = lij;
                if (jc1 > J && i >= jc1)
                    c1[i - BD] -= lij * l1;
            }
        }
        metal_register_cholesky_step2<J + 1, N, BD, T>(c0, c1, lane);
    }
}

template <bool Upper, typename TileA, typename TileOut>
inline WP_FORCE_INLINE void metal_register_cholesky2(TileA WP_THREAD& A, TileOut WP_THREAD& Out)
{
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);
    constexpr int BD = WP_TILE_BLOCK_DIM;
    constexpr int N1 = n - BD;
    const int lane = WP_TILE_THREAD_IDX;
    const int jc1 = lane + BD;

    auto idx = [](int row, int col) { return Upper ? tile_coord(col, row) : tile_coord(row, col); };

    T c0[n];
    T c1[N1];
#pragma clang loop unroll(full)
    for (int i = 0; i < n; ++i)
        c0[i] = (i >= lane) ? (Upper ? A.data(tile_coord(lane, i)) : A.data(tile_coord(i, lane))) : T {};
#pragma clang loop unroll(full)
    for (int i = BD; i < n; ++i)
        c1[i - BD] = (jc1 < n && i >= jc1) ? (Upper ? A.data(tile_coord(jc1, i)) : A.data(tile_coord(i, jc1))) : T {};

    metal_register_cholesky_step2<0, n, BD, T>(c0, c1, lane);

    WP_TILE_SYNC();  // in-place callers alias A and Out: all reads are done before any write
#pragma clang loop unroll(full)
    for (int i = 0; i < n; ++i)
        Out.data(idx(i, lane)) = (i >= lane) ? c0[i] : T {};
    if (jc1 < n) {
#pragma clang loop unroll(full)
        for (int i = 0; i < n; ++i)
            Out.data(idx(i, jc1)) = (i >= jc1) ? c1[(i >= BD) ? (i - BD) : 0] : T {};
    }
    WP_TILE_SYNC();
}

#ifndef WP_METAL_COMPACT_REGISTER_CHOLESKY
#define WP_METAL_COMPACT_REGISTER_CHOLESKY 0
#endif

template <bool Upper, typename TileA, typename TileOut>
inline WP_FORCE_INLINE void metal_register_cholesky(TileA WP_THREAD& A, TileOut WP_THREAD& Out)
{
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);
    constexpr int BD = WP_TILE_BLOCK_DIM;
    constexpr int CPL = (n + BD - 1) / BD;  // columns per lane
    if constexpr (WP_METAL_COMPACT_REGISTER_CHOLESKY && CPL == 2) {
        metal_register_cholesky2<Upper>(A, Out);
        return;
    }
    const int lane = WP_TILE_THREAD_IDX;

    auto idx = [](int row, int col) { return Upper ? tile_coord(col, row) : tile_coord(row, col); };

    T col[CPL][n];
#pragma clang loop unroll(full)
    for (int c = 0; c < CPL; ++c) {
        const int jc = lane + c * BD;
#pragma clang loop unroll(full)
        for (int i = 0; i < n; ++i)
            col[c][i] = (jc < n && i >= jc) ? (Upper ? A.data(tile_coord(jc, i)) : A.data(tile_coord(i, jc))) : T {};
    }

    metal_register_cholesky_step<0, n, CPL, BD, T>(col, lane);

    WP_TILE_SYNC();  // in-place callers alias A and Out: all reads are done before any write
#pragma clang loop unroll(full)
    for (int c = 0; c < CPL; ++c) {
        const int jc = lane + c * BD;
        if (jc < n) {
#pragma clang loop unroll(full)
            // Every lane writes only the cells of its own columns: L[i, jc] for i >= jc and the zeros
            // above it. Zeroing the mirrored cell instead would hit a column owned by another lane, and
            // with more than one column per lane that zero lands after the owner wrote its value.
            for (int i = 0; i < n; ++i)
                Out.data(idx(i, jc)) = (i >= jc) ? col[c][i] : T {};
        }
    }
    WP_TILE_SYNC();
}
#endif  // __METAL_VERSION__

template <bool Upper, typename TileA, typename TileOut>
inline WP_FORCE_INLINE CUDA_CALLABLE void scalar_cholesky_impl(TileA WP_THREAD& A, TileOut WP_THREAD& Out)
{
#if defined(__METAL_VERSION__)
    if constexpr (WP_TILE_BLOCK_DIM > 1 && WP_TILE_BLOCK_DIM <= 32 && TileA::Layout::Shape::dim(1) <= WP_METAL_REGISTER_CHOLESKY_MAX) {
        metal_register_cholesky<Upper>(A, Out);
        return;
    }
#endif
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);

    // Helper: index into the output triangle.
    // Lower: Out(row, col), Upper: Out(col, row)
    auto idx = [](int row, int col) { return Upper ? tile_coord(col, row) : tile_coord(row, col); };

    for (int j = 0; j < n; ++j) {
        // Diagonal: redundant compute on all threads.
        T s = A.data(tile_coord(j, j));

        // In-place callers alias A and Out. Unlike GPU lockstep execution,
        // one CPU fiber could otherwise overwrite the diagonal before its
        // peers have read the original value.
        WP_TILE_SYNC();

        for (int k = 0; k < j; ++k) {
            T r = Out.data(idx(j, k));
            s -= r * r;
        }

        s = wp::sqrt(s);
        T invS = 1.0 / s;

        // Only thread 0 writes the diagonal.
        if (WP_TILE_THREAD_IDX == 0) {
            Out.data(idx(j, j)) = s;
        }

        // Row updates below the diagonal -- distributed across threads.
        for (int i = j + 1 + WP_TILE_THREAD_IDX; i < n; i += WP_TILE_BLOCK_DIM) {
            T s_i = Upper ? A.data(tile_coord(j, i)) : A.data(tile_coord(i, j));

            for (int k = 0; k < j; ++k) {
                s_i -= Out.data(idx(i, k)) * Out.data(idx(j, k));
            }

            Out.data(idx(i, j)) = s_i * invS;
        }

        // Zero out the opposite triangle in column j -- distributed.
        for (int k = j + 1 + WP_TILE_THREAD_IDX; k < n; k += WP_TILE_BLOCK_DIM) {
            Out.data(idx(j, k)) = T {};
        }

        WP_TILE_SYNC();
    }
}


// Cooperative scalar Cholesky adjoint, mirrors the libmathdx adjoint
// structure but uses scalar inner kernels. Replaces the previous
// single-threaded scalar_cholesky_adj_impl (which gated to thread 0 at the
// call site -- correct but underutilized block_dim - 1 threads).
//
// Algorithm: Murray (2016) symmetric-matrix Cholesky derivative.
// Six thread-strided phases over __shared__ scratch buffers, mirroring
// adj_tile_cholesky_impl's libmathdx LTO path verbatim:
//   1. gemm into W1   = adj_Out @ Out^T (Upper) or Out^T @ adj_Out (Lower)
//   2. symmetrize W1, copy into W2 (mirror the stored triangle)
//   3. first triangular solve in-place on W2:  L^T X = W2 (Lower) or U X = W2 (Upper)
//   4. transpose W2 -> W1
//   5. second triangular solve in-place on W1:  L^T B = W1 (Lower) or U B = W1 (Upper)
//   6. accumulate W1 into adj_A.grad on the stored triangle, halving the diagonal.
//
// Each phase ends with WP_TILE_SYNC(). Intra-phase, every thread writes to a
// unique address.
//
// CPU blocks share W1 and W2 through the tile arena. The one-lane
// specialization retains the original local stack arrays.
//
// Upper=false: A = L L^T, Upper=true: A = U^T U
template <bool Upper, typename TileA, typename TileOut>
inline CUDA_CALLABLE void
cooperative_scalar_cholesky_adj(TileA WP_THREAD& adj_A, TileOut WP_THREAD& adj_Out, TileOut WP_THREAD& Out)
{
    WP_TILE_ARENA_NULL
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);

    // Helper: index into the output triangle.
    // Lower: Out(row, col), Upper: Out(col, row)
    auto idx = [](int row, int col) { return Upper ? tile_coord(col, row) : tile_coord(row, col); };

#if defined(__CUDA_ARCH__)
    __shared__ T W1[n * n];
    __shared__ T W2[n * n];
#else
    T W1_local[WP_TILE_BLOCK_DIM == 1 ? n * n : 1];
    T W2_local[WP_TILE_BLOCK_DIM == 1 ? n * n : 1];
    T WP_THREAD* W1;
    T WP_THREAD* W2;
    if constexpr (WP_TILE_BLOCK_DIM == 1) {
        W1 = W1_local;
        W2 = W2_local;
    } else {
        W1 = (T WP_TILE_SHARED*)WP_TILE_ALLOC(int(sizeof(T) * n * n));
        W2 = (T WP_TILE_SHARED*)WP_TILE_ALLOC(int(sizeof(T) * n * n));
    }
#endif

    // Phase 1: gemm into W1.
    //   Upper: W1[i,j] = sum_k adj_Out[i,k] * Out[j,k]
    //   Lower: W1[i,j] = sum_k Out[k,i]    * adj_Out[k,j]
    for (int ij = WP_TILE_THREAD_IDX; ij < n * n; ij += WP_TILE_BLOCK_DIM) {
        int i = ij / n;
        int j = ij % n;
        T s = T(0);
        for (int k = 0; k < n; ++k) {
            if constexpr (Upper)
                s += adj_Out.grad(tile_coord(i, k)) * Out.data(tile_coord(j, k));
            else
                s += Out.data(tile_coord(k, i)) * adj_Out.grad(tile_coord(k, j));
        }
        W1[ij] = s;
    }
    WP_TILE_SYNC();

    // Phase 2: symmetrize W1 and copy into W2.
    //   Upper: keep triu, mirror to lower (W2[i,j] = W1[j,i] for i > j; else W1[i,j])
    //   Lower: keep tril, mirror to upper (W2[i,j] = W1[j,i] for i < j; else W1[i,j])
    for (int ij = WP_TILE_THREAD_IDX; ij < n * n; ij += WP_TILE_BLOCK_DIM) {
        int row = ij / n;
        int col = ij % n;
        bool mirror = Upper ? (row > col) : (row < col);
        W2[ij] = mirror ? W1[col * n + row] : W1[ij];
    }
    WP_TILE_SYNC();

    // Phase 3: solve L^T X = W2 (Lower) or U X = W2 (Upper) in-place into W2.
    //   Distribute over k columns; sequential descending i within column.
    for (int k = WP_TILE_THREAD_IDX; k < n; k += WP_TILE_BLOCK_DIM) {
        for (int i = n - 1; i >= 0; --i) {
            T s = W2[i * n + k];
            for (int j = i + 1; j < n; ++j)
                s -= Out.data(idx(j, i)) * W2[j * n + k];
            T diag = Out.data(tile_coord(i, i));
            W2[i * n + k] = (diag != T(0.0f)) ? s / diag : s;
        }
    }
    WP_TILE_SYNC();

    // Phase 4: transpose W2 into W1.
    for (int ij = WP_TILE_THREAD_IDX; ij < n * n; ij += WP_TILE_BLOCK_DIM) {
        int row = ij / n;
        int col = ij % n;
        W1[ij] = W2[col * n + row];
    }
    WP_TILE_SYNC();

    // Phase 5: solve L^T B = W1 (Lower) or U B = W1 (Upper) in-place into W1.
    for (int k = WP_TILE_THREAD_IDX; k < n; k += WP_TILE_BLOCK_DIM) {
        for (int i = n - 1; i >= 0; --i) {
            T s = W1[i * n + k];
            for (int j = i + 1; j < n; ++j)
                s -= Out.data(idx(j, i)) * W1[j * n + k];
            T diag = Out.data(tile_coord(i, i));
            W1[i * n + k] = (diag != T(0.0f)) ? s / diag : s;
        }
    }
    WP_TILE_SYNC();

    // Phase 6: accumulate W1 into adj_A.grad on the stored triangle.
    // Diagonal halved because B = A_bar + A_bar^T double-counts it. Writes via
    // tile_coord(row, col) directly (no Upper-swap) -- W1 and adj_A share the
    // same layout, so the gradient lands at the correct indices for both
    // Upper and Lower (the in_triangle predicate selects which half is
    // populated).
    for (int ij = WP_TILE_THREAD_IDX; ij < n * n; ij += WP_TILE_BLOCK_DIM) {
        int row = ij / n;
        int col = ij % n;
        bool in_triangle = Upper ? (row <= col) : (row >= col);
        if (in_triangle) {
            T scale = (row == col) ? T(0.5) : T(1);
            adj_A.grad(tile_coord(row, col)) += scale * W1[row * n + col];
        }
    }
    WP_TILE_SYNC();

#if !defined(__CUDA_ARCH__)
    if constexpr (WP_TILE_BLOCK_DIM > 1) {
        WP_TILE_ALLOC(-int(sizeof(T) * n * n));
        WP_TILE_ALLOC(-int(sizeof(T) * n * n));
    }
#endif
}


}  // namespace partitioned_gemm


// Cholesky factorization (out-of-place) implementation.
// Upper=false: produces lower-triangular L s.t. A = L L^T, zeros upper triangle.
// Upper=true:  produces upper-triangular U s.t. A = U^T U, zeros lower triangle.
template <bool Upper, typename Fwd, typename TileA, typename TileOut>
WP_FORCE_INLINE CUDA_CALLABLE TileOut WP_THREAD&
tile_cholesky_impl(Fwd fun_forward, TileA WP_THREAD& A, TileOut WP_THREAD& Out)
{
    static_assert(TileA::Layout::Shape::N == 2, "Expected TileA::Layout::Shape::N == 2");
    static_assert(TileOut::Layout::Shape::N == 2, "Expected TileOut::Layout::Shape::N == 2");

    static_assert(TileA::Layout::Shape::dim(0) == TileA::Layout::Shape::dim(1), "Expected TileA to be square");
    static_assert(TileOut::Layout::Shape::dim(0) == TileOut::Layout::Shape::dim(1), "Expected TileOut to be square");
    static_assert(
        TileA::Layout::Shape::dim(0) == TileOut::Layout::Shape::dim(0),
        "Expected A and Out to have the same number of rows"
    );
    static_assert(
        TileA::Layout::Shape::dim(1) == TileOut::Layout::Shape::dim(1),
        "Expected A and Out to have the same number of columns"
    );

    Out = A;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_impl<Upper>(A, Out);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_impl<Upper>(A, Out);
    } else {
        // TODO: for batched Cholesky, need one info per batch
        __shared__ int info[1];

        if (WP_TILE_THREAD_IDX == 0) {
            info[0] = 0;
        }

        WP_TILE_SYNC();

        fun_forward(Out.data.ptr, info);

        WP_TILE_SYNC();

        // TODO: for batched Cholesky, check all batches
#if defined(_DEBUG)
        if (WP_TILE_THREAD_IDX == 0 && info[0] != 0) {
            printf("Non-zero status in Cholesky factorization, got %d\n", info[0]);
        }
#endif

        // Zero-out the opposite triangular part
        WP_PRAGMA_UNROLL
        for (int i = WP_TILE_THREAD_IDX; i < TileOut::Layout::Size; i += WP_TILE_BLOCK_DIM) {
            auto c = TileOut::Layout::coord_from_linear(i);

            if (Upper ? (c[0] > c[1]) : (c[0] < c[1]))
                Out.data(c) = 0.0;
        }

        WP_TILE_SYNC();
    }
#endif

    return Out;
}


template <bool Upper, typename BkwdGemm, typename BkwdTrsm, typename TileA, typename TileOut>
CUDA_CALLABLE void adj_tile_cholesky_impl(
    BkwdGemm fun_bkwd_gemm,
    BkwdTrsm fun_bkwd_trsm,
    TileOut WP_THREAD& Out,
    TileA WP_THREAD& adj_A,
    TileOut WP_THREAD& adj_Out
)
{
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    // CPU (block_dim == 1) or GPU-without-mathdx: cooperative scalar adjoint.
    // Leading sync mirrors the libmathdx branch below: be defensive against
    // upstream hazards on Out / adj_Out before Phase 1 reads from them.
    WP_TILE_SYNC();
    partitioned_gemm::cooperative_scalar_cholesky_adj<Upper>(adj_A, adj_Out, Out);

#else

    if constexpr (wp_is_null_func<BkwdGemm>::value) {
        // GPU with mathdx build but backward LTOs absent (e.g. enable_backward
        // disabled, or enable_mathdx_solver=False at the module level):
        // cooperative scalar adjoint.
        WP_TILE_SYNC();
        partitioned_gemm::cooperative_scalar_cholesky_adj<Upper>(adj_A, adj_Out, Out);
    } else {
        __shared__ T W1[n * n];
        __shared__ T W2[n * n];

        T alpha_one = T(1);
        T beta_zero = T(0);
        WP_TILE_SYNC();

        // P = adj_Out @ Out^T (upper) or Out^T @ adj_Out (lower)
        if constexpr (Upper) {
            fun_bkwd_gemm(&alpha_one, adj_Out.grad.ptr, Out.data.ptr, &beta_zero, W1);
        } else {
            fun_bkwd_gemm(&alpha_one, Out.data.ptr, adj_Out.grad.ptr, &beta_zero, W1);
        }
        WP_TILE_SYNC();

        // Symmetrize P: mirror the stored triangle to the other side (preserving the diagonal).
        // Upper: keep triu, mirror to lower; Lower: keep tril, mirror to upper.
        for (int idx = WP_TILE_THREAD_IDX; idx < n * n; idx += WP_TILE_BLOCK_DIM) {
            int row = idx / n;
            int col = idx % n;
            bool mirror = Upper ? (row > col) : (row < col);
            if (mirror)
                W2[idx] = W1[col * n + row];
            else
                W2[idx] = W1[idx];
        }
        WP_TILE_SYNC();

        // Solve L^T X = S (lower) or U X = S (upper), in-place into W2
        fun_bkwd_trsm(Out.data.ptr, W2);
        WP_TILE_SYNC();

        // Transpose X into W1
        for (int idx = WP_TILE_THREAD_IDX; idx < n * n; idx += WP_TILE_BLOCK_DIM) {
            int row = idx / n;
            int col = idx % n;
            W1[idx] = W2[col * n + row];
        }
        WP_TILE_SYNC();

        // Solve L^T B = X^T (lower) or U B = X^T (upper), in-place into W1
        fun_bkwd_trsm(Out.data.ptr, W1);
        WP_TILE_SYNC();

        // Accumulate B into adj_A.grad (upper or lower triangle only).
        // Diagonal halved because B = A_bar + A_bar^T double-counts it.
        // W1 and adj_A share same layout so gradient accumulates at correct indices.
        for (int idx = WP_TILE_THREAD_IDX; idx < n * n; idx += WP_TILE_BLOCK_DIM) {
            int row = idx / n;
            int col = idx % n;
            bool in_triangle = Upper ? (row <= col) : (row >= col);
            if (in_triangle) {
                T scale = (row == col) ? T(0.5) : T(1);
                adj_A.grad(tile_coord(row, col)) += scale * W1[row * n + col];
            }
        }
    }

#endif

    WP_TILE_SYNC();
}

// Cholesky factorization (inplace) implementation.
template <bool Upper, typename Fwd, typename TileA>
WP_FORCE_INLINE CUDA_CALLABLE void tile_cholesky_inplace_impl(Fwd fun_forward, TileA WP_THREAD& A)
{
    static_assert(TileA::Layout::Shape::N == 2, "Expected TileA::Layout::Shape::N == 2");
    static_assert(TileA::Layout::Shape::dim(0) == TileA::Layout::Shape::dim(1), "Expected TileA to be square");

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_impl<Upper>(A, A);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_impl<Upper>(A, A);
    } else {
        // TODO: for batched Cholesky, need one info per batch
        __shared__ int info[1];

        if (WP_TILE_THREAD_IDX == 0) {
            info[0] = 0;
        }

        WP_TILE_SYNC();

        fun_forward(A.data.ptr, info);

        WP_TILE_SYNC();

        // TODO: for batched Cholesky, check all batches
#if defined(_DEBUG)
        if (WP_TILE_THREAD_IDX == 0 && info[0] != 0) {
            printf("Non-zero status in Cholesky factorization, got %d\n", info[0]);
        }
#endif

        // Zero-out the opposite triangular part
        WP_PRAGMA_UNROLL
        for (int i = WP_TILE_THREAD_IDX; i < TileA::Layout::Size; i += WP_TILE_BLOCK_DIM) {
            auto c = TileA::Layout::coord_from_linear(i);

            if (Upper ? (c[0] > c[1]) : (c[0] < c[1]))
                A.data(c) = 0.0;
        }

        WP_TILE_SYNC();
    }
#endif
}

// Cholesky (out-of-place): tile_cholesky<false>(...) for lower, tile_cholesky<true>(...) for upper
template <bool Upper, typename Fwd, typename BkwdGemm, typename BkwdTrsm, typename TileA, typename TileOut>
CUDA_CALLABLE TileOut WP_THREAD& tile_cholesky(
    Fwd fun_forward, BkwdGemm fun_bkwd_gemm, BkwdTrsm fun_bkwd_trsm, TileA WP_THREAD& A, TileOut WP_THREAD& Out
)
{
    return tile_cholesky_impl<Upper>(fun_forward, A, Out);
}

// Adjoint of Cholesky (out-of-place, Murray 2016, "Differentiation of the Cholesky decomposition"):
// adj_tile_cholesky<false>(...) for lower, adj_tile_cholesky<true>(...) for upper
template <bool Upper, typename Fwd, typename BkwdGemm, typename BkwdTrsm, typename TileA, typename TileOut>
CUDA_CALLABLE void adj_tile_cholesky(
    Fwd fun_forward,
    BkwdGemm fun_bkwd_gemm,
    BkwdTrsm fun_bkwd_trsm,
    TileA WP_THREAD& A,
    TileOut WP_THREAD& Out,
    Fwd adj_fun_forward,
    BkwdGemm adj_fun_bkwd_gemm,
    BkwdTrsm adj_fun_bkwd_trsm,
    TileA WP_THREAD& adj_A,
    TileOut WP_THREAD& adj_Out,
    TileOut WP_THREAD& adj_ret
)
{
    adj_tile_cholesky_impl<Upper>(fun_bkwd_gemm, fun_bkwd_trsm, Out, adj_A, adj_Out);
}

// Cholesky (inplace): tile_cholesky_inplace<false>(...) for lower, tile_cholesky_inplace<true>(...) for upper
template <bool Upper, typename Fwd, typename TileA>
CUDA_CALLABLE void tile_cholesky_inplace(Fwd fun_forward, TileA WP_THREAD& A)
{
    tile_cholesky_inplace_impl<Upper>(fun_forward, A);
}

template <bool Upper, typename Fwd, typename TileA, typename AdjFwd, typename AdjTileA>
void adj_tile_cholesky_inplace(Fwd fun_forward, TileA WP_THREAD& A, AdjFwd adj_fun_forward, AdjTileA WP_THREAD& adj_A)
{
    // MISSINGADJOINT: apply Murray 2016 derivative in place; on entry A holds L
    // (lower) or U (upper), adj_A holds adj_L/adj_U; on exit adj_A holds adj of the original symmetric input
}


}  // namespace wp

#ifdef __clang__
#pragma clang diagnostic pop
#endif
