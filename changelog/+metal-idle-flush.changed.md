Commit the open Metal command buffer every `WP_METAL_IDLE_FLUSH` (default 16) eager dispatches when the GPU has nothing else to execute, so loops that wait on the GPU often keep it busy.
