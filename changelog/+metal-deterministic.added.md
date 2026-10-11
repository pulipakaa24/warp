Support deterministic mode (`wp.config.deterministic`, the `"deterministic"` module option) on Metal devices
at any launch size. Floating-point accumulation atomics add into exact per-element integer accumulators and
are rounded once after the kernel; consumed-return counters run in two phases with a GPU sort and scan and hand
out slots in thread order. Deterministic launches need no host synchronization and can be captured. The
`"metal_ordered_atomics"` module option keeps ordered single-SIMD-group launches.
