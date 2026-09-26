// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Stubs for platforms without Metal support (see metal.mm for the Apple implementation).

#include "error.h"
#include "metal.h"

namespace {
int unavailable()
{
    wp::set_error_string("Metal is not available on this platform");
    return -1;
}
}  // anonymous namespace

int wp_is_metal_enabled() { return 0; }

int wp_metal_device_get_count() { return 0; }

const char* wp_metal_device_get_name(int) { return unavailable(), nullptr; }

int wp_metal_device_get_max_threadgroup_memory(int) { return unavailable(), 0; }

int wp_metal_device_get_max_threads_per_threadgroup(int) { return unavailable(), 0; }

int wp_metal_device_get_core_count(int) { return 0; }

int wp_metal_device_has_unified_memory(int) { return unavailable(), 0; }

void* wp_alloc_metal(int, size_t) { return unavailable(), nullptr; }

void wp_free_metal(int, void*) { unavailable(); }
int wp_metal_import_host_memory(int, const void*, size_t) { return unavailable(), 0; }
int wp_metal_owns_pointer(int, const void*) { return unavailable(), 0; }
int wp_metal_capture_host_call(int, void*, const unsigned long long*, int) { return unavailable(), 0; }
void wp_metal_release_host_memory(int, const void*, size_t) { unavailable(); }

uint64_t wp_metal_gpu_address(int, const void*) { return unavailable(), 0; }

const char* wp_metal_profile_report() { return ""; }

int wp_metal_capture_begin(int) { return unavailable(); }

void* wp_metal_capture_end(int) { return unavailable(), nullptr; }

int wp_metal_capture_push(int) { return unavailable(); }

void* wp_metal_capture_pop(int) { return unavailable(), nullptr; }

int wp_metal_capture_conditional(int, int, const int*, void*, void*, int, int) { return unavailable(); }

int wp_metal_capture_range_begin(int, void*) { return unavailable(); }

int wp_metal_capture_range_end(int) { return unavailable(); }

bool wp_metal_capture_host_op(int, std::function<bool()>) { return false; }

void wp_metal_defer_error(int) { }

uint64_t wp_texture_create_metal(int, int, int, int*, int*, int*, int, int, int, int, int*, bool, void**)
{
    return unavailable(), 0;
}

void wp_texture_destroy_metal(uint64_t) { }

int wp_metal_graph_launch(int, void*) { return unavailable(); }

void wp_metal_graph_destroy(int, void*) { }

void* wp_metal_load_library(int, const char*) { return unavailable(), nullptr; }

void wp_metal_unload_library(void*) { }

void* wp_metal_get_kernel(void*, const char*) { return unavailable(), nullptr; }

int wp_metal_get_kernel_thread_execution_width(void*) { return unavailable(), 0; }

int wp_metal_get_kernel_max_threads_per_threadgroup(void*) { return unavailable(), 0; }

int wp_metal_launch_kernel(
    int, void*, size_t, int, const void*, size_t, const void*, size_t, const size_t*, int, size_t
)
{
    return unavailable();
}

int wp_metal_memset(int, void*, int, size_t) { return unavailable(); }

int wp_metal_memtile(int, void*, const void*, size_t, size_t) { return unavailable(); }

int wp_metal_memcpy(int, void*, const void*, size_t) { return unavailable(); }

int wp_metal_synchronize(int) { return unavailable(); }

int wp_metal_flush(int) { return unavailable(); }
