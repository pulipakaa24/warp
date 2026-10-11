Add `wp.launch(..., dim_limit=...)`: a 1-element `int32` device array that bounds the first launch dimension.
Tasks past the bound may be skipped, so fixed-capacity buffers whose used part is only known on the device can be
processed without host synchronization. On Metal only the threadgroups below the bound are dispatched (an
indirect dispatch sized on the GPU, also inside captured graphs and indirect execution ranges); other devices
run the full `dim`. `warp.config.metal_bounded_launch` (`WP_METAL_BOUNDED_LAUNCH=0`) turns it off.
