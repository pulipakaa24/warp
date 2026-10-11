Apple GPUs (Metal)
==================

.. currentmodule:: warp

Warp can run kernels on Apple Silicon GPUs through a Metal backend. On a Mac with a Metal-enabled build,
the GPU appears as the device ``"metal:0"`` and is the default device:

.. code-block:: python

    import warp as wp


    @wp.kernel
    def scale(a: wp.array[float], s: float):
        i = wp.tid()
        a[i] = a[i] * s


    a = wp.array([1.0, 2.0, 3.0], dtype=float, device="metal:0")
    wp.launch(scale, dim=3, inputs=[a, 2.0], device="metal:0")
    print(a.numpy())  # [2. 4. 6.]

:func:`wp.is_metal_available() <warp.is_metal_available>` reports whether a Metal device was found, and
``Device.is_metal`` identifies one. Code that selects a device with ``wp.get_device()``
or ``wp.ScopedDevice`` needs no changes. Code that tests ``device.is_cuda`` to decide whether it runs on a GPU
should test ``not device.is_cpu`` instead.

Requirements
------------

* Apple Silicon and macOS 15 or newer.
* A build of Warp that contains the Metal runtime. The macOS wheels on PyPI do not. Build from source with
  ``build_lib.py`` or CMake (see :ref:`building-from-source`); both compile ``warp/native/metal.mm`` on macOS.
  Xcode is not needed at run time: kernels are compiled from source by the Metal framework when a module loads,
  and the result is kept in the kernel cache.

Supported Features
------------------

Kernels and user functions, structs, tile operations, backward kernels and :class:`wp.Tape <Tape>`,
graph capture and replay including :func:`wp.capture_if() <capture_if>` and :func:`wp.capture_while() <capture_while>`,
meshes, BVHs, hash grids, NanoVDB volumes, textures, atomic operations, ``wp.ref`` parameters,
``print()`` and assertions inside kernels, and DLPack, NumPy and PyTorch interoperability.

Memory Model
------------

Apple GPUs share memory with the CPU. Arrays on a Metal device live in unified memory, so their ``ptr`` is a valid
host pointer:

* ``array.numpy()`` and ``__array_interface__`` return a view, not a copy. Warp synchronizes the device before
  handing out the pointer.
* :func:`wp.to_torch() <warp.to_torch>` returns a zero-copy PyTorch **CPU** tensor, with ``grad`` attached when the
  array has one. PyTorch's ``mps`` tensors cannot alias Warp arrays: PyTorch does not expose their buffers, and it
  schedules work on its own command queue. Move data to ``mps`` explicitly when a network should run on the GPU.
* NumPy arrays and PyTorch CPU tensors can be passed to a Metal kernel directly. Warp maps their pages into the
  GPU address space without copying and synchronizes after such a launch. The mapping is released when the object
  is garbage collected; for objects that cannot be weakly referenced it stays until the process exits. Mapping is
  page granular, so the GPU can address the whole pages around the data, and overlapping mappings are merged.
  The memory must stay mapped in the process for as long as the object lives: memory from a custom allocator
  that is unmapped while the object is still alive leaves the GPU with a dangling mapping.
* Copies between ``"cpu"`` and ``"metal:0"`` are plain memory copies.

Limitations
-----------

* **No** ``float64``. Apple GPUs have no double-precision type. A kernel that uses ``wp.float64``, or a vector,
  matrix or struct built on it, raises when it is launched on Metal. The rest of its module still loads.
* Apple's GPU compiler can miscompile kernels in which several fully unrolled products of 4x4 or larger matrices
  are inlined together: values computed later in the kernel come out as zero. Warp's own matrix product avoids
  this on Metal, which fixed zero gradients of ``wp.mat44`` products and inverses. Kernels with hand-written,
  fully unrolled loop nests of that size can meet the same condition, so check their results and gradients
  against the CPU device.
* Cooperative tile operations need full thread blocks, as on CUDA: launch tile kernels with
  :func:`wp.launch_tiled() <launch_tiled>`. A plain ``wp.launch()`` whose size is not a multiple of
  ``block_dim`` leaves the last block partial; operations on shared tiles in that block return undefined
  values (full blocks are unaffected, and nothing hangs).
* **Tile kernels are forward only.** Kernels that use tile operations have no adjoint on Metal.
* If the adjoint of a module fails to compile on Metal, Warp warns and rebuilds the module forward-only, so the
  forward pass keeps working. Launching one of its backward kernels raises.
* Tile kernels are limited by threadgroup memory, 32 KB on current Apple GPUs. A kernel whose tiles need more
  raises at launch. Keeping block-local tiles small, or accumulating into an output tile
  (``wp.tile_matmul(a, b, out)``) instead of creating temporaries, stays below the limit.
* Not every atomic operation is atomic on Metal, which only has 32-bit atomics and, from the Apple9 GPU family
  (M3, A17) on, unsigned 64-bit minimum and maximum. All operations on 32-bit integers and floats are atomic, and
  so is ``wp.atomic_add()`` on 16-bit types, including ``wp.float16`` and ``wp.bfloat16``. The rest is as follows:

  * ``wp.atomic_add()`` and the bitwise operations (``wp.atomic_and()``, ``wp.atomic_or()``, ``wp.atomic_xor()``)
    on 64-bit integers update the two 32-bit halves with separate atomics. The final value is exact, but a kernel
    that reads the element while others update it, or uses the returned previous value, can see a half-updated
    (torn) value.
  * ``wp.atomic_min()`` and ``wp.atomic_max()`` on ``wp.uint64`` are atomic on Apple9 and later, where the returned
    previous value is read before the update and can be stale under contention. On older GPUs they do not compile.
  * Not available: Apple GPUs have no 64-bit compare-and-swap, so ``wp.atomic_cas()`` and ``wp.atomic_exch()`` on
    ``wp.int64`` / ``wp.uint64`` and ``wp.atomic_min()`` / ``wp.atomic_max()`` on ``wp.int64`` (also as vector or
    matrix components) cannot be made atomic. A kernel that uses one raises at launch on Metal (the other kernels of
    its module are unaffected); native code (``wp.func_native``) that reaches one fails to compile. Build such
    operations from 32-bit atomics instead, e.g. claim a slot with a 32-bit compare-and-swap on one half.
  * Plain read-modify-write, correct only without concurrent writers to the same element: every atomic operation
    on 8-bit integers; minimum, maximum, exchange, compare-and-swap and the bitwise operations (``&=``, ``|=``,
    ``^=``) on 16-bit types.
* Spinlocks built from ``wp.atomic_cas()`` can hang: Apple GPUs do not guarantee forward progress between
  SIMD lanes.
* ``wp.fixedarray``, fabric arrays and saveable (APIC) captures are not supported.
* Deterministic mode (``wp.config.deterministic`` or the ``"deterministic"`` module option) lowers atomics as on
  CUDA and gives the same guarantee: results do not depend on thread scheduling. Two parts differ from CUDA:

  * Floating-point accumulation atomics (``wp.atomic_add()``/``wp.atomic_sub()`` or ``+=`` whose result is unused)
    add into an exact per-element integer accumulator, built from 32-bit atomics, instead of writing records that
    are sorted after the kernel. One pass after the kernel rounds each element's exact sum to float once and adds it
    to the element, so the result is the correctly rounded sum of the contributions (it can differ in the last bit
    from CUDA's sort-based reduction). The accumulator takes 72 bytes per destination scalar while the kernel runs.
    ``wp.atomic_min()``/``wp.atomic_max()`` stay native: on floats they order ``-0.0`` below ``+0.0``, so they do not
    depend on order either, and different reductions may target the same array.
  * Consumed-return counters (``slot = wp.atomic_add(counter, i, n)`` on ``int32``) run in two phases as on CUDA and
    hand out slots in thread order for each counter element. Records are sized by the number of call sites per
    thread; a call inside a dynamic loop takes the ``"deterministic_max_records"`` bound. A thread that makes more
    calls than its bound raises an error at a later deterministic launch (the overflow flag is read without
    waiting for the GPU).

  No pass reads data back on the host, so deterministic launches can be captured. The module option
  ``"metal_ordered_atomics"`` instead launches every kernel of the module that uses atomics as one SIMD-group
  striding over the tasks, which reproduces the order one CUDA warp issues atomics in (suits single small launches;
  a kernel that waits on a later task can hang under it).
* Host-side utilities that are not recorded into a graph, such as building or changing the topology of a
  :class:`warp.sparse.BsrMatrix`, raise when called during a graph capture on Metal. Call them outside the
  capture.
* Native snippets (:func:`wp.func_native() <func_native>`) must be valid Metal Shading Language. Pointer casts need
  an address space, for which Warp defines ``WP_THREAD`` and ``WP_DEVICE`` (both empty on CPU and CUDA), and
  ``long long`` is not available.
* Transcendental functions can differ from their CPU results in the last bit.

Performance Notes
-----------------

* Launches are recorded into a command buffer and run asynchronously. Reading an array on the host synchronizes.
* Graph capture removes most of the per-launch cost and is worth using for many small kernels.
* ``wp.capture_while()`` evaluates its condition on the host between iterations on Metal. A fixed iteration count
  is usually faster for short loops.
* Tile kernels run one tile per threadgroup. Their throughput is bounded by threadgroup memory, so they scale less
  well with problem size than they do on CUDA.
