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

The same seven code commits are exported as `patches/warp/0001-0007` in MetalSim; applying them to `ce15f6bb`
gives the tree of `f194006a`. Tests for the Metal changes are in `warp/tests/test_metal.py`.
