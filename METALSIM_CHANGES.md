# MetalSim changes on this branch (`metalsim`)

This branch is the Warp build that [MetalSim](https://github.com/pulipakaa24/MetalSim) runs on (see the
section "What was built where" in its [README](https://github.com/pulipakaa24/MetalSim/blob/main/README.md)).

Lineage: [NVIDIA/warp](https://github.com/NVIDIA/warp) → [innate-inc/warp](https://github.com/innate-inc/warp)
(Metal backend, David Dobas / innate-inc; base commit `ce15f6bb`, innate-inc `main` on 2026-09-20) → this branch.

Licence: Apache License 2.0, as upstream. `LICENSE.md` and `licenses/` are unchanged from NVIDIA/warp and
innate-inc/warp (same blob). The changes below modify Apache-2.0 code and are offered under the same licence.

## Commits (on top of innate-inc/warp `ce15f6bb`)

| commit | change |
|---|---|
| `b39fee16` | Metal interop entry points: device, queue and buffer handles, foreign `MTLEvent`/`MTLSharedEvent` wait and signal, so a renderer or PyTorch MPS can alias Warp arrays without copies and order work without host synchronization. |
| `b1ec949e` | Captured graphs replay through a Metal indirect command buffer (one `executeCommandsInBuffer` instead of re-encoding each dispatch on the CPU); runtime counters (`wp_metal_counters`: host waits, flushes, host ops, dispatches). `WP_METAL_ICB=0` restores per-dispatch re-encoding. |
| `7b5c828a` | Deferred frees are released per completed command buffer instead of at the next host synchronize (long eager loops at 4096 worlds ran out of GPU memory). Knobs `WP_METAL_INFLIGHT` (bound on committed, unfinished command buffers) and `WP_METAL_ICB_BATCH` (graph replay in chunks). |
| `786cdae3` | Fixed-size arrays (`wp.zeros` inside kernels) on Metal, as array views of a per-thread slice of a device scratch buffer; `thread` qualification of reference casts in native snippets. |
| `9dcb1406` | `segmented_sort_pairs` inside a graph capture (no segment-validation readback while capturing; MuJoCo Warp's flex SAP sort). Also on `metalsim-flex`. |
| `b9557cb9` | Configurable register-Cholesky bound, `warp.config.metal_register_cholesky_max` (default 40, as before). MetalSim sets 48 for every scene, which puts MuJoCo Warp's 43-dof G1 on the single-SIMD-group register path. |
| `9050cb54` | `nextafterf` (and `__builtin_nextafterf`) in the Metal kernel runtime, bit-exact on float32 bit patterns (Newton's VBD solver needs it). |
| `9ebfad43` | Merge of `metalsim-flex` (`9dcb1406`). |
| `f194006a` | Merge of `9050cb54` into the merge above; the tree MetalSim's results were produced with. |
| `27e63fd1`, `a5b65da1`, `0c9a1fb5` | Metal: register triangular solves in `tile_cholesky_solve` (vector RHS, blocks of at most one SIMD group, matrices up to `metal_register_cholesky_max`): lanes own the factor's columns as in the register Cholesky, forward substitution reduces each row with `simd_sum`, back substitution broadcasts with `simd_shuffle`, no barriers inside the sweeps (the cooperative path pays two per row per sweep). Solve 0.52 -> 0.34 ms per 4096 at n = 43; MuJoCo Warp's G1 step +5.2 %. Summation order only (1.3-3.4e-7 relative). `warp.config.metal_register_solve = False` / `WP_METAL_REGISTER_SOLVE=0` restores the cooperative path. |
| `c44a3f16`, `139385a8` | Metal: compact two-column register layout for the register Cholesky and solve at 33-64 rows (bitwise the generic form). Measured 5-8 % slower at n = 43 / 48; off, `warp.config.metal_compact_register_cholesky` / `WP_METAL_COMPACT_REGISTER_CHOLESKY=1`. |
| `2869a2f7` | Metal: rolled (runtime-loop) form of the register Cholesky, bitwise the unrolled form; 3.4x slower at n = 43 (thread-memory arrays); off, `warp.config.metal_rolled_cholesky` / `WP_METAL_ROLLED_CHOLESKY=<min n>`. |
| `e29950ee` | `tile_cholesky_update_inplace(A, X, count, fill_mode)`: rank-1 Cholesky updates in place (the `mju_cholUpdate` recurrence, additions only), a Metal register form (lanes own rows, SIMD broadcasts) and a scalar path; no adjoint. Used by MuJoCo Warp's archived `MJW_ELLIPTIC_CONE_UPDATE` path. |
| `2834cf31` | Metal: 64-lane register Cholesky (`metal_register_cholesky64`, two SIMD groups per matrix, one column per lane, the tile's own column storage as the per-column exchange with one barrier per column) and a `block_dim` 64 path for the register solve (group 0 computes). Bitwise the 32-lane form at n = 16..48; no faster at n = 43 (factor 0.998 vs 0.989 ms per 4096, solve 0.239 vs 0.326), so MuJoCo Warp keeps 32 lanes; `WP_METAL_CHOL64=0` removes the form. |
| `9df9acee`, `61cf6a4a`, this commit | Metal: split-loop form of the register Cholesky step (`metal_register_cholesky_step_split`: scalar column pivots, the second column's update guarded by the compile-time row index; bitwise the generic step). In a native-snippet context it is 7x faster than the generic step (0.60 vs 4.16 ms per 4096 at n = 43, `scripts/diagnostics/metal_cholesky_residency.py` in MetalSim), on the tile path it measures the same (0.971 vs 0.973 ms), so it is off: `warp.config.metal_chol_split` / `WP_METAL_CHOL_SPLIT=1`. The same benchmark showed that residency (1-8 worlds per threadgroup) does not change the cost. |
| `6bceb39d` | Metal graph replay: indirect execution ranges (`wp_metal_capture_range_begin(range_host)` / `_end`, Python `warp._src.context.metal_capture_range_begin(ranges, index)` / `_end`). The launches recorded between begin and end are replayed with `executeCommandsInBuffer:indirectBuffer:`, the ICB execution range read by the GPU from a 16-byte Metal allocation (uint32 location, length, full length, pad; filled in at the range's end, the `MTLBuffer` resolved at begin because capture-time allocations are retained by the graph). A kernel earlier in the same replay may zero the length and the range's launches are skipped: MuJoCo Warp's exact per-world early exit of its Newton loop at `opt.iterations` on Metal (`MJW_METAL_ICB_EARLY_EXIT`), where `capture_while` has no conditional node. Plain ranges keep the batched `withRange:` replay; `WP_METAL_ICB=0` runs every dispatch as before. |

The same seven code commits are exported as `patches/warp/0001-0007` in MetalSim; applying them to `ce15f6bb`
gives the tree of `f194006a`. Tests for the Metal changes are in `warp/tests/test_metal.py`.
