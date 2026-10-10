Run `wp.utils.radix_sort_pairs`, `wp.utils.runlength_encode` and integer (`int32` / `int64`) `wp.utils.array_scan`
as GPU kernels on Metal devices instead of draining the GPU and running the host implementations. Results are bit
for bit those of the host implementations (for the sort, both halves of the `2*count` keys and values buffers),
the calls no longer synchronize the device, and they are recorded into Metal graph captures as kernel dispatches.
Floating-point scans stay on the host. `warp.config.metal_gpu_utilities = False` (or `WP_METAL_GPU_UTILITIES=0`)
restores the host path.
