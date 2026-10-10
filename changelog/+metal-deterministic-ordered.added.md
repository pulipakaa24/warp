Support deterministic mode (`wp.config.deterministic`, the `"deterministic"` module option) on Metal devices.
Kernels of a deterministic module that use atomics run as ordered launches: one SIMD-group strides over the
tasks, so counter slots are handed out and floating-point atomics accumulate in task order, bit for bit the same
on every run. Previously the option was silently ignored on Metal.
