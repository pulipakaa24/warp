Copy strided and indexed arrays within a Metal device on the GPU (`wp.copy`, `array.assign`), as
`wp_array_copy_device` does on CUDA, instead of draining the GPU and copying on the host. Such copies are now
ordered with kernel launches and can be recorded into graph captures.
