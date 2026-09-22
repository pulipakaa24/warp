// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"

#include <cstddef>

#include <stddef.h>
#include <stdint.h>

// Native Metal runtime (Apple GPUs). Implemented in metal.mm on Apple platforms and stubbed out in
// metal_stub.cpp elsewhere. Device ordinals index the list of Metal devices with unified memory.
//
// Memory model: wp_alloc_metal() returns the host address of a shared-storage MTLBuffer. Kernel argument
// structs hold such host pointers; wp_metal_launch_kernel() rewrites them to GPU virtual addresses using
// the byte offsets given in pointer_offsets, so kernels can dereference them directly.
extern "C" {
WP_API int wp_is_metal_enabled();

WP_API int wp_metal_device_get_count();
WP_API const char* wp_metal_device_get_name(int ordinal);
WP_API int wp_metal_device_get_max_threadgroup_memory(int ordinal);
WP_API int wp_metal_device_get_max_threads_per_threadgroup(int ordinal);
WP_API int wp_metal_device_get_core_count(int ordinal);  // 0 if unknown
WP_API int wp_metal_device_has_unified_memory(int ordinal);

// Returns a host pointer into a shared MTLBuffer, or null on failure. A zero size allocates a minimal buffer.
WP_API void* wp_alloc_metal(int ordinal, size_t size);
// Safe to call while GPU work using the buffer is pending; the release is deferred until that work completes.
WP_API void wp_free_metal(int ordinal, void* ptr);
// GPU virtual address of a pointer into a Metal allocation (what kernels dereference), or 0 with an error set.
WP_API uint64_t wp_metal_gpu_address(int ordinal, const void* ptr);
// Makes foreign host memory (NumPy, Torch) addressable by kernels in place; refcounted per page range.
// Returns 0 on failure, 1 when the pointer already is Metal memory, 2 when an import was registered (release it).
WP_API int wp_metal_import_host_memory(int ordinal, const void* ptr, size_t size);
// 1 if ptr lies in memory allocated by wp_alloc_metal on this device (imported foreign pages excluded).
WP_API int wp_metal_owns_pointer(int ordinal, const void* ptr);
WP_API void wp_metal_release_host_memory(int ordinal, const void* ptr, size_t size);

// Compiles MSL source at runtime. Returns null and sets the error string (with compiler diagnostics) on failure.
WP_API void* wp_metal_load_library(int ordinal, const char* source);
WP_API void wp_metal_unload_library(void* library);

// Returns a compute pipeline state for the named kernel function, or null with an error string.
// The handle stays valid until the owning library is unloaded.
WP_API void* wp_metal_get_kernel(void* library, const char* name);
WP_API int wp_metal_get_kernel_thread_execution_width(void* kernel);
WP_API int wp_metal_get_kernel_max_threads_per_threadgroup(void* kernel);

// Enqueues a dispatch. bounds is bound at buffer index 0, a copy of args (with pointers translated) at index 1.
// When threadgroup_memory_bytes > 0, whole threadgroups are dispatched (ceil(num_threads / threads_per_threadgroup))
// and the memory is bound at threadgroup index 0; otherwise exactly num_threads threads are dispatched.
// Returns 0 on success, non-zero with an error string otherwise.
WP_API int wp_metal_launch_kernel(
    int ordinal,
    void* kernel,
    size_t num_threads,
    int threads_per_threadgroup,
    const void* bounds,
    size_t bounds_size,
    const void* args,
    size_t args_size,
    const size_t* pointer_offsets,
    int num_pointer_offsets,
    size_t threadgroup_memory_bytes
);

// Commits pending work and waits for all GPU work on the device. Returns 0 on success, non-zero if any
// command buffer failed (the error string describes the failure).
// Asynchronous memory operations on Metal allocations, ordered with kernel launches.
// wp_metal_memtile reads the pattern from host memory at call time; the others take Metal pointers.
WP_API int wp_metal_memset(int ordinal, void* dst, int value, size_t n);
WP_API int wp_metal_memtile(int ordinal, void* dst, const void* src, size_t src_size, size_t n);
WP_API int wp_metal_memcpy(int ordinal, void* dst, const void* src, size_t n);

WP_API int wp_metal_synchronize(int ordinal);
// Graph capture: launches and memory operations between begin and end are recorded instead of
// executed (arrays must stay allocated and in place); wp_metal_graph_launch replays them.
WP_API int wp_metal_capture_begin(int ordinal);
WP_API void* wp_metal_capture_end(int ordinal);  // NULL with an error set on failure
WP_API int wp_metal_graph_launch(int ordinal, void* graph);
WP_API int wp_metal_capture_push(int ordinal);
WP_API void* wp_metal_capture_pop(int ordinal);
WP_API int wp_metal_capture_conditional(
    int ordinal, int is_loop, const int* condition, void* on_true, void* on_false, int own_true, int own_false
);
// Records a host function call (up to 8 integer/pointer args) to replay with the graph; 0 when not capturing.
WP_API int wp_metal_capture_host_call(int ordinal, void* fn, const unsigned long long* args, int nargs);
WP_API void wp_metal_graph_destroy(int ordinal, void* graph);
// Commits pending work without waiting. Returns 0 on success.
WP_API int wp_metal_flush(int ordinal);
// Per-kernel GPU times collected since the last call (WP_METAL_PROFILE=1), sorted, one line each.
WP_API const char* wp_metal_profile_report();

// Textures sampled by Metal kernels through the software path in texture.h (see texture.cpp).
WP_API uint64_t wp_texture_create_metal(
    int ordinal,
    int ndim,
    int num_mip_levels,
    int* mip_widths,
    int* mip_heights,
    int* mip_depths,
    int num_channels,
    int dtype,
    int filter_mode,
    int mip_filter_mode,
    int* address_modes,
    bool use_normalized_coords,
    void** mip_data_ptrs_out
);
WP_API void wp_texture_destroy_metal(uint64_t id);
}

#if defined(__cplusplus)
#include <functional>
// Records a host-side operation into the graph being captured (run in order at graph launch, after the
// GPU work recorded before it). Returns false when no capture is active, so the caller runs it now.
bool wp_metal_capture_host_op(int ordinal, std::function<bool()> op);
// Keeps the current error string for the next wp_metal_synchronize(); for entry points that return void.
void wp_metal_defer_error(int ordinal);
#endif

// Interop with other users of the same Metal device (renderers, PyTorch MPS). Handles are unretained
// Objective-C object pointers (id<MTLDevice>, id<MTLCommandQueue>, id<MTLBuffer>, id<MTLEvent>/id<MTLSharedEvent>).
extern "C" {
WP_API void* wp_metal_device_handle(int ordinal);
WP_API void* wp_metal_queue_handle(int ordinal);
// The MTLBuffer behind a pointer into Metal memory of this device (own allocation or imported host memory) and
// the byte offset of ptr within it. Null if ptr is not Metal memory. The caller must keep the array alive.
WP_API void* wp_metal_buffer_handle(int ordinal, const void* ptr, size_t* offset_out);
// The event that orders this device's command buffers and the value the most recently committed work signals.
WP_API void* wp_metal_event_handle(int ordinal);
WP_API uint64_t wp_metal_event_value(int ordinal);
// Commits pending work; when it completes, `event` (MTLEvent or MTLSharedEvent) is signaled with `value`.
// Returns 0 on success. Not available during graph capture (signal after wp_metal_graph_launch instead).
WP_API int wp_metal_signal_event(int ordinal, void* event, uint64_t value);
// Commits pending work; everything launched afterwards waits until `event` reaches `value`. Returns 0 on success.
WP_API int wp_metal_wait_event(int ordinal, void* event, uint64_t value);
// Diagnostics: out[0..3] = host waits for the GPU, command buffers committed, host ops run during
// graph replay, kernel dispatches encoded (cumulative). Lets callers assert a loop never blocks.
WP_API int wp_metal_counters(int ordinal, uint64_t* out, int n);
}
