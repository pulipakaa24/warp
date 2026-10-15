Add `wp.preload_launches()`: a context manager that records the kernels a block of code launches instead of
launching them, then compiles and loads their modules (and, on Metal, creates their pipelines) on several threads.
It warms up code whose modules `wp.force_load()` cannot name in advance, such as kernel factories, by running one
step under it; `LaunchPreload.add()` records kernels known from elsewhere. On Metal, library and pipeline
compilation now run outside the runtime lock, so parallel loads overlap the compiles (`wp.force_load(max_workers=...)`
too). `warp.config.metal_maximize_concurrent_compilation` (off by default) makes a parallel preload set
`MTLDevice.shouldMaximizeConcurrentCompilation`.
