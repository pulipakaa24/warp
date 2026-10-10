Build NanoVDB volumes as GPU kernels on Metal devices: `wp.Volume.allocate_by_voxels` and `wp.Volume.allocate_by_tiles`
(non-rebuildable), `wp.Volume.rebuild`, `wp.Volume.get_voxels` and `wp.Volume.get_tiles` no longer drain the GPU and
run the host builder. The grids are byte for byte those of the host builder. Rebuilds need no host synchronization
and can be recorded into Metal graph captures; exact allocations read the node counts back once, as on CUDA.
Rebuildable allocation still uses the host builder. `warp.config.metal_gpu_volume = False` (or
`WP_METAL_GPU_VOLUME=0`) restores the host path.
