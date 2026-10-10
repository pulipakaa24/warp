Build the BSR topology of `wp.sparse.bsr_set_from_triplets` (and so of `bsr_from_triplets`, `bsr_assign`,
`bsr_axpy` and `warp.fem` matrix assembly) and of `wp.sparse.bsr_set_transpose` with GPU kernels on Metal devices,
as on CUDA, instead of draining the GPU and running the host implementations. The calls no longer synchronize the
device and are recorded into Metal graph captures. Offsets, columns, padded row counts, transpose block maps and
status are bit for bit those of the host. Duplicate triplets are summed in increasing triplet order (a stable sort,
as on CUDA); the host sort is not stable, so with three or more duplicates of a block the floating-point sum can
round differently from the host. `warp.config.metal_gpu_utilities = False` (or `WP_METAL_GPU_UTILITIES=0`) restores
the host path.
