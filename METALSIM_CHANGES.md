# MetalSim changes on this branch (`metalsim-flex`)

Working branch for [MetalSim](https://github.com/pulipakaa24/MetalSim)'s deformable (MuJoCo Warp flex) work; see
the section "What was built where" in its [README](https://github.com/pulipakaa24/MetalSim/blob/main/README.md).
It has been merged into `metalsim` (merge `9ebfad43`); use `metalsim` unless you need this exact history.

Lineage: [NVIDIA/warp](https://github.com/NVIDIA/warp) → [innate-inc/warp](https://github.com/innate-inc/warp)
(Metal backend; base commit `ce15f6bb`) → this branch.

Licence: Apache License 2.0, as upstream. `LICENSE.md` and `licenses/` are unchanged from NVIDIA/warp and
innate-inc/warp. The changes below modify Apache-2.0 code and are offered under the same licence.

## Commits (on top of innate-inc/warp `ce15f6bb`)

| commit | change |
|---|---|
| `b39fee16` | Metal interop entry points (device, queue, buffer handles; foreign event wait and signal). |
| `b1ec949e` | Graph replay through Metal indirect command buffers; runtime counters. |
| `7b5c828a` | Deferred frees released per completed command buffer; `WP_METAL_INFLIGHT`, `WP_METAL_ICB_BATCH`. |
| `786cdae3` | Fixed-size arrays (`wp.zeros` in kernels) on Metal. |
| `9dcb1406` | `segmented_sort_pairs` inside a graph capture (no segment-validation readback while capturing). |

The first four are the same commits as on `metalsim`; see that branch's `METALSIM_CHANGES.md` for the full list.
