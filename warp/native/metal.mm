// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Metal runtime for Apple GPUs. Compiled as Objective-C++ with ARC enabled.
//
// All device memory lives in shared-storage MTLBuffers so the host pointer returned by wp_alloc_metal() is
// directly readable and writable by the CPU. Kernels receive their arguments through a bump-allocated ring buffer
// in which every host pointer has been rewritten to the GPU virtual address of the same location; this requires
// every buffer to be resident, which is guaranteed by a single MTLResidencySet attached to the command queue.
// Because kernels dereference raw addresses, Metal cannot track hazards between dispatches, so all dispatches go
// through serial compute encoders which execute in order.

#include "alloc_tracker.h"
#include "error.h"
#include "metal.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#include <unistd.h>

// Metal support requires macOS 15 (MTLResidencySet). Older systems report zero devices, so the newer API is
// never reached at runtime and the availability diagnostics are unnecessary.
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

namespace {

constexpr size_t kArgsRingBytes = 16 << 20;
constexpr size_t kArgsAlignment = 256;
constexpr size_t kThreadgroupMemoryAlignment = 16;
// Dispatches per command buffer before it is committed. Smaller batches let the GPU start while
// Python is still encoding a step, but every boundary drains the GPU (buffers are chained with
// an event, see Device::event). WP_METAL_BATCH overrides the default for tuning.
// WP_METAL_ICB=0 disables indirect-command-buffer graph replay (per-dispatch re-encoding instead).
bool icb_replay_enabled()
{
    static const bool value = [] {
        const char* env = std::getenv("WP_METAL_ICB");
        return !env || atoi(env) != 0;
    }();
    return value;
}

int max_dispatches_per_command_buffer()
{
    static const int value = [] {
        const char* env = std::getenv("WP_METAL_BATCH");
        return env ? std::max(1, atoi(env)) : 128;
    }();
    return value;
}

// WP_METAL_ICB_BATCH > 0 replays graphs in chunks of that many dispatches per command buffer.
int icb_batch()
{
    static const int value = [] {
        const char* env = std::getenv("WP_METAL_ICB_BATCH");
        return env ? std::max(0, atoi(env)) : 0;
    }();
    return value;
}

// WP_METAL_INFLIGHT bounds the committed-but-unfinished command buffers (see flush()).
int max_command_buffers_in_flight()
{
    static const int value = [] {
        const char* env = std::getenv("WP_METAL_INFLIGHT");
        return env ? std::max(1, atoi(env)) : 64;
    }();
    return value;
}

// Warp may call into the runtime from several Python threads; the device state is not thread-safe.
std::recursive_mutex& runtime_mutex()
{
    static std::recursive_mutex mutex;
    return mutex;
}
#define WP_METAL_LOCK() std::lock_guard<std::recursive_mutex> lock(runtime_mutex())

// A recorded sequence of dispatches (see wp_metal_capture_begin): every argument block is
// copied, with host pointers already translated, into one buffer owned by the graph.
struct Graph {
    struct Dispatch {
        id<MTLComputePipelineState> pipeline;
        size_t num_threads;
        size_t group_size;
        size_t threadgroup_bytes;
        size_t bounds_offset, bounds_size;  // bounds_size 0: memory kernel (arguments only)
        size_t args_offset, args_size;
    };
    struct HostOp {
        size_t before_dispatch;  // runs once the dispatches before this index have completed
        std::function<bool()> run;
    };
    std::vector<Dispatch> dispatches;
    std::vector<HostOp> host_ops;  // host-side rebuilds recorded during capture (hash grids, mesh refits)
    std::vector<std::pair<Graph*, bool>> children;  // conditional branch bodies (owned when captured inline)
    std::vector<char> bytes;  // bounds and argument blocks, kArgsAlignment-aligned
    std::vector<std::pair<size_t, size_t>> fixups;  // (field offset, data offset): field = data GPU address
    id<MTLBuffer> buffer;  // bytes uploaded at capture end
    std::vector<id<MTLBuffer>> retained;  // freed during capture but referenced by the recording
    // Replay: the dispatches encoded once into an indirect command buffer (with a barrier between
    // consecutive commands), executed with one call per launch instead of re-encoding every dispatch.
    // Built lazily on the first launch; graphs with host ops keep the per-dispatch path.
    id<MTLIndirectCommandBuffer> icb;
    bool icb_tried = false;
};

struct Device {
    Graph* capture = nullptr;  // graph being recorded, if any
    std::vector<Graph*> capture_stack;  // enclosing captures while a conditional branch body is recorded
    id<MTLDevice> device;
    std::string name;
    int core_count = 0;

    // created lazily by get_device()
    id<MTLCommandQueue> queue;
    id<MTLResidencySet> residency_set;
    bool residency_dirty = false;

    std::map<uintptr_t, id<MTLBuffer>> allocations;  // keyed by host base address
    std::vector<id<MTLBuffer>> deferred_frees;  // freed while GPU work may still use them
    // deferred frees attached to the command buffer committed after them: released when it completes,
    // so an eager loop that allocates temporaries per step (MuJoCo Warp does) does not accumulate them
    // until the next synchronize (which exhausted GPU memory at 4096 worlds)
    std::vector<std::pair<id<MTLCommandBuffer>, std::vector<id<MTLBuffer>>>> pending_frees;

    // Foreign host memory (NumPy, Torch) wrapped page-aligned with newBufferWithBytesNoCopy so kernels can
    // address it in place; keyed by page base, released when the last importing array is freed.
    // Imports are kept DISJOINT so every lookup by address (here, find_gpu_address and the GPU-side table) is
    // unambiguous: a new range that overlaps existing imports is merged with them into one import over the
    // union, which carries the summed references and keeps the absorbed buffers alive, since their GPU
    // addresses may be baked into recorded graphs or in-flight argument blocks.
    struct Import {
        id<MTLBuffer> buffer;
        int refs = 0;
        std::vector<id<MTLBuffer>> absorbed;
    };
    std::map<uintptr_t, Import> imports;

    // Host->GPU address translation table for pointers kernels read from memory (array descriptors stored in
    // arrays of structs, raw addresses passed as integers): `table_slot` holds the GPU address of the current
    // sorted range table and is baked into every kernel pipeline as a function constant; a rebuild writes a
    // new table and swaps the slot, so in-flight kernels keep a consistent table.
    id<MTLBuffer> table_slot;
    id<MTLBuffer> table;
    bool table_dirty = true;

    // Buffers allocated while a capture was open: the recording bakes their GPU addresses into its argument
    // blocks, so the graph keeps them alive when the array is freed later (like a CUDA graph's memory pool).
    std::map<uintptr_t, Graph*> capture_owned;
    // A GPU failure observed where it cannot be returned (freeing memory); raised by the next synchronize.
    std::string deferred_error;

    id<MTLBuffer> args_ring;
    size_t args_ring_offset = 0;

    // built-in kernels for asynchronous memory operations (compiled on first use)
    id<MTLComputePipelineState> memset_kernel;
    id<MTLComputePipelineState> memtile_kernel;
    id<MTLLogState> log_state;  // shader logging (printf) delivered to stdout
    std::string kernel_assertion;  // first error-level kernel log since the last report (see retire_completed)
    id<MTLComputePipelineState> memcpy_kernel;

    id<MTLCommandBuffer> command_buffer;  // open command buffer, if any
    // Buffers are untracked (kernels use raw GPU addresses), so consecutive command buffers would
    // overlap on the GPU; each one waits for the previous one through this event.
    id<MTLEvent> event;
    uint64_t event_value = 0;
    id<MTLComputeCommandEncoder> encoder;  // open encoder on command_buffer, if any
    int num_dispatches = 0;
    // Interop diagnostics (wp_metal_counters): how often the runtime waited for the GPU, committed a
    // command buffer, ran a recorded host op during graph replay, or encoded a dispatch.
    uint64_t n_sync = 0, n_flush = 0, n_host_ops = 0, n_dispatch = 0, n_backpressure = 0;
    std::vector<id<MTLCommandBuffer>> in_flight;  // committed but not yet known to be complete

    ~Device()
    {
        // Metal asserts if an encoder is released while still encoding (e.g. at interpreter exit).
        if (encoder)
            [encoder endEncoding];
    }
};

// WP_METAL_PRINTF: 2 = compile with logging and attach a log state to every command buffer (kernel
// printf works), 1 = compile with logging only, 0 = off. Measured to size the cost of shader logging.
static int printf_mode()
{
    static int mode = [] {
        const char* env = getenv("WP_METAL_PRINTF");
        return env ? atoi(env) : 2;
    }();
    return mode;
}

// Command buffers carry the device's log state so kernel printf lines reach the handler.
static id<MTLCommandBuffer> new_command_buffer(Device& dev)
{
    if (printf_mode() < 2)
        return [dev.queue commandBuffer];
    if (!dev.log_state) {
        MTLLogStateDescriptor* descriptor = [MTLLogStateDescriptor new];
        descriptor.level = MTLLogLevelDebug;
        descriptor.bufferSize = 4 << 20;
        NSError* error = nil;
        dev.log_state = [dev.device newLogStateWithDescriptor:descriptor error:&error];
        if (dev.log_state) {
            Device* device = &dev;
            [dev.log_state addLogHandler:^(NSString*, NSString*, MTLLogLevel level, NSString* message) {
                // raw write after flushing: keeps ordering with buffered stdout and never leaves the
                // FILE in an error state if the descriptor was redirected meanwhile
                fflush(stdout);
                const char* text = message.UTF8String;
                (void)!write(fileno(stdout), text, strlen(text));
                clearerr(stdout);
                // kernel assertions (_wp_assert) log at error level; the next synchronize reports them
                if (level == MTLLogLevelError && device->kernel_assertion.empty())
                    device->kernel_assertion = text;
            }];
        }
    }
    MTLCommandBufferDescriptor* descriptor = [MTLCommandBufferDescriptor new];
    descriptor.logState = dev.log_state;
    return [dev.queue commandBufferWithDescriptor:descriptor];
}

struct Library {
    id<MTLLibrary> library;
    std::vector<id<MTLComputePipelineState>> kernels;  // owned by the library, handed out as raw handles
    int ordinal = 0;
};

size_t align_up(size_t value, size_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }

const char* error_description(NSError* error)
{
    return error ? error.localizedDescription.UTF8String : "unknown error";
}

// Compiler output with the warning blocks removed, so that errors fit in Warp's error buffer.
std::string compile_errors(NSError* error)
{
    std::string result;
    bool keep = true;
    for (NSString* line in [@(error_description(error)) componentsSeparatedByString:@"\n"]) {
        std::string text = line.UTF8String;
        if (text.find(": warning:") != std::string::npos)
            keep = false;
        else if (text.find(": error:") != std::string::npos || text.find(": note:") != std::string::npos)
            keep = true;
        if (keep)
            result += text + "\n";
    }
    if (result.find(": error:") == std::string::npos)  // nothing recognisable survived: show everything
        return error_description(error);
    return result;
}

// Metal does not expose the GPU core count; the accelerator's IORegistry entry does.
int gpu_core_count(id<MTLDevice> device)
{
    int count = 0;
    io_iterator_t iterator;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iterator)
        != KERN_SUCCESS)
        return count;
    for (io_object_t entry = IOIteratorNext(iterator); entry; entry = IOIteratorNext(iterator)) {
        CFTypeRef value = IORegistryEntryCreateCFProperty(entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (value) {
            if (CFGetTypeID(value) == CFNumberGetTypeID())
                CFNumberGetValue((CFNumberRef)value, kCFNumberIntType, &count);
            CFRelease(value);
        }
        IOObjectRelease(entry);
        if (count)
            break;
    }
    IOObjectRelease(iterator);
    (void)device;
    return count;
}

std::vector<Device>& all_devices()
{
    static std::vector<Device> devices = [] {
        std::vector<Device> result;
        if (@available(macOS 15.0, *)) {
            @autoreleasepool {
                for (id<MTLDevice> device in MTLCopyAllDevices()) {
                    if (!device.hasUnifiedMemory)
                        continue;
                    Device entry;
                    entry.device = device;
                    entry.name = device.name.UTF8String;
                    entry.core_count = gpu_core_count(device);
                    result.push_back(std::move(entry));
                }
            }
        }
        return result;
    }();
    return devices;
}

// Returns the device for the ordinal with its queue, residency set and args ring created, or null with an error.
Device* get_device(int ordinal)
{
    std::vector<Device>& devices = all_devices();
    if (ordinal < 0 || ordinal >= int(devices.size())) {
        wp::set_error_string("Invalid Metal device ordinal %d (%zu devices available)", ordinal, devices.size());
        return nullptr;
    }
    Device& dev = devices[ordinal];
    if (dev.args_ring)
        return &dev;

    dev.queue = [dev.device newCommandQueue];
    dev.event = [dev.device newEvent];
    MTLResidencySetDescriptor* descriptor = [[MTLResidencySetDescriptor alloc] init];
    descriptor.label = @"Warp";
    descriptor.initialCapacity = 1024;
    NSError* error = nil;
    dev.residency_set = [dev.device newResidencySetWithDescriptor:descriptor error:&error];
    if (!dev.residency_set) {
        wp::set_error_string("Failed to create Metal residency set: %s", error_description(error));
        return nullptr;
    }
    [dev.queue addResidencySet:dev.residency_set];

    dev.args_ring = [dev.device newBufferWithLength:kArgsRingBytes options:MTLResourceStorageModeShared];
    dev.table_slot = [dev.device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    memset(dev.table_slot.contents, 0, 16);
    [dev.residency_set addAllocation:dev.table_slot];
    if (!dev.args_ring) {
        wp::set_error_string("Failed to allocate Metal kernel argument buffer");
        return nullptr;
    }
    dev.args_ring.label = @"Warp kernel args";
    [dev.residency_set addAllocation:dev.args_ring];
    dev.residency_dirty = true;
    return &dev;
}

void commit_residency(Device& dev)
{
    if (!dev.residency_dirty)
        return;
    [dev.residency_set commit];
    [dev.residency_set requestResidency];
    dev.residency_dirty = false;
}

bool has_pending_work(const Device& dev) { return dev.command_buffer != nil || !dev.in_flight.empty(); }

// Drops command buffers that have finished executing. Returns false if any of them failed.
void release_buffers(Device& dev, std::vector<id<MTLBuffer>>& buffers);

bool retire_completed(Device& dev)
{
    bool ok = true;
    auto finished = [&ok](id<MTLCommandBuffer> command_buffer) {
        if (command_buffer.status < MTLCommandBufferStatusCompleted)
            return false;
        if (command_buffer.error) {
            wp::set_error_string(
                "Metal command buffer%s%s failed: %s", command_buffer.label ? " " : "",
                command_buffer.label ? command_buffer.label.UTF8String : "", error_description(command_buffer.error)
            );
            ok = false;
        }
        return true;
    };
    std::vector<id<MTLCommandBuffer>>& in_flight = dev.in_flight;
    in_flight.erase(std::remove_if(in_flight.begin(), in_flight.end(), finished), in_flight.end());
    // frees whose command buffer completed can be released now (buffers are only ever referenced by
    // work committed before they were freed, and command buffers complete in order on the queue)
    for (size_t i = 0; i < dev.pending_frees.size();) {
        if (dev.pending_frees[i].first.status >= MTLCommandBufferStatusCompleted) {
            release_buffers(dev, dev.pending_frees[i].second);
            dev.pending_frees.erase(dev.pending_frees.begin() + i);
        } else {
            ++i;
        }
    }
    if (in_flight.empty() && !dev.kernel_assertion.empty()) {
        wp::set_error_string("Metal kernel assertion failed: %s", dev.kernel_assertion.c_str());
        dev.kernel_assertion.clear();  // reported once, like a CUDA trap surfacing at synchronize
        ok = false;
    }
    return ok;
}

// Removes the buffers from the residency set and releases them. Only valid when no GPU work can use them.
void release_buffers(Device& dev, std::vector<id<MTLBuffer>>& buffers)
{
    if (buffers.empty())
        return;
    for (id<MTLBuffer> buffer : buffers)
        [dev.residency_set removeAllocation:buffer];
    dev.residency_dirty = true;
    commit_residency(dev);
    buffers.clear();
}

// Commits the open command buffer, if any. Returns false if a previously committed command buffer failed.
bool flush(Device& dev);

// Per-kernel GPU time from command buffer timestamps, enabled with WP_METAL_PROFILE=1: each
// profiled dispatch gets its own command buffer, so use it for attribution, not absolute timing.
struct Profiler {
    const bool enabled = std::getenv("WP_METAL_PROFILE") != nullptr;
    std::mutex mutex;  // completion handlers run on Metal's threads
    std::map<std::string, std::pair<double, int>> totals;
    std::string report;
};
Profiler& profiler()
{
    static Profiler p;
    return p;
}

// Commits the open command buffer as a profiled unit attributed to name.
bool profile_dispatch(Device& dev, const char* name)
{
    if (!dev.command_buffer)
        return true;
    std::string key = name;
    dev.command_buffer.label = @(name);  // names the culprit if the GPU faults
    [dev.command_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        Profiler& p = profiler();
        std::lock_guard<std::mutex> lock(p.mutex);
        std::pair<double, int>& total = p.totals[key];
        total.first += buffer.GPUEndTime - buffer.GPUStartTime;
        total.second += 1;
    }];
    return flush(dev);
}

bool flush(Device& dev)
{
    if (dev.encoder) {
        [dev.encoder endEncoding];
        dev.encoder = nil;
    }
    if (dev.command_buffer) {
        commit_residency(dev);
        [dev.command_buffer encodeSignalEvent:dev.event value:++dev.event_value];
        [dev.command_buffer commit];
        ++dev.n_flush;
        dev.in_flight.push_back(dev.command_buffer);
        if (!dev.deferred_frees.empty()) {
            dev.pending_frees.emplace_back(dev.command_buffer, std::move(dev.deferred_frees));
            dev.deferred_frees.clear();
        }
        dev.command_buffer = nil;
        dev.num_dispatches = 0;
    }
    // Back-pressure: the driver reserves per-dispatch scratch for every command buffer in flight, and
    // thousands of large dispatches outstanding fail with kIOGPUCommandBufferCallbackErrorOutOfMemory
    // (seen at 4096 MuJoCo Warp worlds). Waiting on the oldest buffer keeps the queue deep enough to
    // stay fed while bounding that reservation. Not counted as a sync: no host-visible data is read.
    const int limit = max_command_buffers_in_flight();
    while (int(dev.in_flight.size()) > limit) {
        [dev.in_flight.front() waitUntilCompleted];
        ++dev.n_backpressure;
        if (!retire_completed(dev))
            return false;
    }
    return retire_completed(dev);
}

// Commits and waits for all work, then recycles the args ring and deferred frees. Returns false on GPU errors.
bool synchronize(Device& dev)
{
    bool ok = flush(dev);
    if (!dev.in_flight.empty())
        ++dev.n_sync;
    for (id<MTLCommandBuffer> command_buffer : dev.in_flight)
        [command_buffer waitUntilCompleted];
    ok = retire_completed(dev) && ok;   // releases every pending free: all command buffers completed
    dev.args_ring_offset = 0;
    release_buffers(dev, dev.deferred_frees);
    return ok;
}

id<MTLComputeCommandEncoder> get_encoder(Device& dev)
{
    if (!dev.command_buffer) {
        dev.command_buffer = new_command_buffer(dev);
        if (dev.event_value > 0)
            [dev.command_buffer encodeWaitForEvent:dev.event value:dev.event_value];
    }
    if (!dev.encoder)
        dev.encoder = [dev.command_buffer computeCommandEncoder];
    return dev.encoder;
}

// Reserves a ring slot holding a copy of args and returns its offset, or SIZE_MAX with an error.
size_t stage_args(Device& dev, const void* args, size_t args_size)
{
    size_t slot_size = align_up(std::max(args_size, size_t(1)), kArgsAlignment);
    if (slot_size > kArgsRingBytes) {
        wp::set_error_string("Kernel arguments of %zu bytes exceed the Metal argument buffer", args_size);
        return SIZE_MAX;
    }
    if (dev.args_ring_offset + slot_size > kArgsRingBytes && !synchronize(dev))
        return SIZE_MAX;
    size_t offset = dev.args_ring_offset;
    if (args_size > 0)
        memcpy(static_cast<char*>(dev.args_ring.contents) + offset, args, args_size);
    dev.args_ring_offset += slot_size;
    return offset;
}

// Appends size bytes (aligned like ring slots) to the graph being recorded and returns their offset.
size_t graph_append(Graph& graph, const void* data, size_t size)
{
    size_t offset = align_up(graph.bytes.size(), kArgsAlignment);
    graph.bytes.resize(offset + std::max(size, size_t(1)));
    if (size)
        memcpy(graph.bytes.data() + offset, data, size);
    return offset;
}

// Records a dispatch into the graph being captured instead of encoding it.
void graph_record(
    Graph& graph,
    id<MTLComputePipelineState> pipeline,
    size_t num_threads,
    size_t group_size,
    size_t threadgroup_bytes,
    const void* bounds,
    size_t bounds_size,
    const void* args,
    size_t args_size
)
{
    Graph::Dispatch d;
    d.pipeline = pipeline;
    d.num_threads = num_threads;
    d.group_size = group_size;
    d.threadgroup_bytes = threadgroup_bytes;
    d.bounds_size = bounds_size;
    d.bounds_offset = bounds_size ? graph_append(graph, bounds, bounds_size) : 0;
    d.args_size = args_size;
    d.args_offset = graph_append(graph, args, args_size);
    graph.dispatches.push_back(d);
}

// Rewrites the translation table (sorted by host base) when allocations or imports changed.
void refresh_table(Device& dev)
{
    if (!dev.table_dirty)
        return;
    struct Range {
        uint64_t host, length, gpu;
    };
    std::vector<Range> ranges;
    ranges.reserve(dev.allocations.size() + dev.imports.size());
    for (const auto& kv : dev.allocations)
        ranges.push_back({ uint64_t(kv.first), uint64_t(kv.second.length), kv.second.gpuAddress });
    for (const auto& kv : dev.imports)
        ranges.push_back({ uint64_t(kv.first), uint64_t(kv.second.buffer.length), kv.second.buffer.gpuAddress });
    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) { return a.host < b.host; });
    const size_t bytes = 16 + sizeof(Range) * std::max(ranges.size(), size_t(1));
    id<MTLBuffer> table = [dev.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    uint64_t* header = static_cast<uint64_t*>(table.contents);
    header[0] = ranges.size();
    header[1] = 0;
    if (!ranges.empty())
        memcpy(header + 2, ranges.data(), sizeof(Range) * ranges.size());
    [dev.residency_set addAllocation:table];
    dev.residency_dirty = true;
    commit_residency(dev);
    *static_cast<uint64_t*>(dev.table_slot.contents) = table.gpuAddress;  // atomic 64-bit swap
    if (dev.table)
        dev.deferred_frees.push_back(dev.table);
    dev.table = table;
    dev.table_dirty = false;
}

// Returns the GPU virtual address of a host pointer into a Metal allocation, or 0 if there is none.
// The buffer whose range contains host_address (end exclusive, or inclusive for the one-past-the-end
// address of an empty view), with its host base; nil if none.
template <typename Map, typename GetBuffer>
static id<MTLBuffer>
buffer_containing(const Map& map, uint64_t host_address, bool allow_end, uintptr_t& base, GetBuffer get)
{
    auto it = map.upper_bound(uintptr_t(host_address));
    if (it == map.begin())
        return nil;
    --it;
    id<MTLBuffer> buffer = get(it->second);
    const uint64_t offset = host_address - it->first;
    if (offset < buffer.length || (allow_end && offset == buffer.length)) {
        base = it->first;
        return buffer;
    }
    return nil;
}

uint64_t find_gpu_address(const Device& dev, uint64_t host_address)
{
    auto own = [](id<MTLBuffer> b) { return b; };
    auto imported = [](const Device::Import& i) { return i.buffer; };
    uintptr_t base = 0;
    // An address strictly inside a buffer wins; the end address of one buffer may be the start of another.
    for (bool allow_end : { false, true }) {
        if (id<MTLBuffer> b = buffer_containing(dev.allocations, host_address, allow_end, base, own))
            return b.gpuAddress + (host_address - base);
        if (id<MTLBuffer> b = buffer_containing(dev.imports, host_address, allow_end, base, imported))
            return b.gpuAddress + (host_address - base);
    }
    return 0;
}

// True if [ptr, ptr + size) lies entirely inside one of this device's own allocations.
static bool owns_range(const Device& dev, uintptr_t ptr, size_t size)
{
    uintptr_t base = 0;
    id<MTLBuffer> b = buffer_containing(dev.allocations, ptr, false, base, [](id<MTLBuffer> x) { return x; });
    return b && (ptr - base) + size <= b.length;
}

// Finds the import whose pages cover [ptr, ptr + size), or imports.end().
static std::map<uintptr_t, Device::Import>::iterator find_import(Device& dev, uintptr_t ptr, size_t size)
{
    auto it = dev.imports.upper_bound(ptr);
    if (it == dev.imports.begin())
        return dev.imports.end();
    --it;
    if (ptr + size > it->first + it->second.buffer.length)
        return dev.imports.end();
    return it;
}

// Like find_gpu_address, but sets an error when the pointer is not Metal memory.
uint64_t gpu_address(const Device& dev, uint64_t host_address)
{
    if (uint64_t address = find_gpu_address(dev, host_address))
        return address;
    wp::set_error_string(
        "Pointer 0x%llx is not Metal memory of device \"%s\"; memory used by Metal kernels must be allocated on "
        "that device",
        (unsigned long long)host_address, dev.name.c_str()
    );
    return 0;
}

// Rewrites the host pointer stored at args + offset to the corresponding GPU virtual address.
bool translate_pointer(const Device& dev, char* args, size_t args_size, size_t offset)
{
    if (offset + sizeof(uint64_t) > args_size) {
        wp::set_error_string(
            "Kernel argument pointer offset %zu lies outside the %zu-byte argument struct", offset, args_size
        );
        return false;
    }
    uint64_t host_address;
    memcpy(&host_address, args + offset, sizeof(host_address));
    if (host_address == 0)
        return true;
    uint64_t address = find_gpu_address(dev, host_address);
    if (!address) {
        // the offset identifies the argument (or the array member of a struct argument) at fault
        wp::set_error_string(
            "Pointer 0x%llx at kernel argument offset %zu is not Metal memory of device \"%s\"; memory used by "
            "Metal kernels must be allocated on that device",
            (unsigned long long)host_address, offset, dev.name.c_str()
        );
        return false;
    }
    memcpy(args + offset, &address, sizeof(address));
    return true;
}

// Byte-granular memory kernels so that memset/memtile/memcpy on Metal arrays are ordered with
// kernel launches instead of forcing a host synchronization.
constexpr const char* kMemoryKernelsSource = R"(
#include <metal_stdlib>
struct memset_args_t { device unsigned char* dst; unsigned int value; unsigned long n; };
struct memtile_args_t { device unsigned char* dst; device const unsigned char* src; unsigned long src_size; unsigned long n; };
struct memcpy_args_t { device unsigned char* dst; device const unsigned char* src; unsigned long n; };

// One thread per 16 bytes: whole uint4 words for the aligned bulk, bytes for the ragged head and tail.
constant unsigned int kWordBytes = 16;
kernel void wp_memset(constant memset_args_t& a [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    unsigned long begin = (unsigned long)i * kWordBytes;
    if (begin >= a.n) return;
    unsigned long end = metal::min(begin + kWordBytes, a.n);
    if (((unsigned long)a.dst & 15) == 0 && end - begin == kWordBytes) {
        unsigned int v = (unsigned int)a.value * 0x01010101u;
        ((device uint4*)a.dst)[i] = uint4(v, v, v, v);
    } else {
        for (unsigned long j = begin; j < end; ++j) a.dst[j] = (unsigned char)a.value;
    }
}
kernel void wp_memtile(constant memtile_args_t& a [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    unsigned long total = a.n * a.src_size;
    unsigned long begin = (unsigned long)i * kWordBytes;
    if (begin >= total) return;
    unsigned long end = metal::min(begin + kWordBytes, total);
    if (((unsigned long)a.dst & 15) == 0 && end - begin == kWordBytes && (kWordBytes % a.src_size) == 0) {
        // whole 16-byte word of a pattern that tiles it: build the word once, store it once
        uint4 word;
        thread unsigned char* bytes = (thread unsigned char*)&word;
        for (unsigned k = 0; k < kWordBytes; ++k) bytes[k] = a.src[k % a.src_size];
        ((device uint4*)a.dst)[i] = word;
        return;
    }
    unsigned long k = begin % a.src_size;
    for (unsigned long j = begin; j < end; ++j) {
        a.dst[j] = a.src[k];
        if (++k == a.src_size) k = 0;
    }
}
kernel void wp_memcpy(constant memcpy_args_t& a [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    unsigned long begin = (unsigned long)i * kWordBytes;
    if (begin >= a.n) return;
    unsigned long end = metal::min(begin + kWordBytes, a.n);
    if (((unsigned long)a.dst & 15) == 0 && ((unsigned long)a.src & 15) == 0 && end - begin == kWordBytes) {
        ((device uint4*)a.dst)[i] = ((device const uint4*)a.src)[i];
    } else {
        for (unsigned long j = begin; j < end; ++j) a.dst[j] = a.src[j];
    }
}
// Checks the assumption behind the shared-tile arena (tile.h): the threadgroup buffer bound at
// index 0 starts at threadgroup address 0. Writes through the constant address, reads through
// the bound pointer.
kernel void wp_check_arena(device int* out [[buffer(0)]], threadgroup char* smem [[threadgroup(0)]], uint tid [[thread_index_in_threadgroup]])
{
    ((threadgroup int*)0)[tid] = int(tid) + 12345;
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    out[tid] = ((threadgroup int*)smem)[(tid + 1) % 32] - int((tid + 1) % 32);
}
)";

bool load_memory_kernels(Device& dev)
{
    if (dev.memcpy_kernel)
        return true;
    NSError* error = nil;
    id<MTLLibrary> library = [dev.device newLibraryWithSource:@(kMemoryKernelsSource) options:nil error:&error];
    if (!library) {
        wp::set_error_string("Failed to compile Metal memory kernels:\n%s", compile_errors(error).c_str());
        return false;
    }
    auto pipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
        MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = [library newFunctionWithName:name];
        descriptor.label = name;
        descriptor.supportIndirectCommandBuffers = YES;
        return descriptor.computeFunction ? [dev.device newComputePipelineStateWithDescriptor:descriptor
                                                                                      options:MTLPipelineOptionNone
                                                                                   reflection:nil
                                                                                        error:&error]
                                          : nil;
    };
    dev.memset_kernel = pipeline(@"wp_memset");
    dev.memtile_kernel = pipeline(@"wp_memtile");
    dev.memcpy_kernel = pipeline(@"wp_memcpy");
    id<MTLComputePipelineState> check = pipeline(@"wp_check_arena");
    if (!dev.memset_kernel || !dev.memtile_kernel || !dev.memcpy_kernel || !check) {
        wp::set_error_string("Failed to create Metal memory kernel pipelines: %s", error_description(error));
        dev.memcpy_kernel = nil;
        return false;
    }
    // Tile kernels compute lane indices from 32-wide SIMD groups and address the shared arena at 0.
    id<MTLBuffer> result = [dev.device newBufferWithLength:32 * sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command_buffer = new_command_buffer(dev);
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:check];
    [encoder setBuffer:result offset:0 atIndex:0];
    [encoder setThreadgroupMemoryLength:32 * sizeof(int) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    bool arena_ok = true;
    for (int i = 0; i < 32; ++i)
        arena_ok = arena_ok && static_cast<int*>(result.contents)[i] == 12345;
    if (!arena_ok || check.threadExecutionWidth != 32) {
        wp::set_error_string(
            "Device \"%s\" does not support Warp's Metal tile model (arena at threadgroup address 0: %s, SIMD width "
            "%lu)",
            dev.name.c_str(), arena_ok ? "yes" : "no", (unsigned long)check.threadExecutionWidth
        );
        dev.memcpy_kernel = nil;
        return false;
    }
    return true;
}

// Translates the host address at args + offset to a GPU address, then dispatches one thread per 16 bytes.
bool dispatch_memory_kernel(
    Device& dev,
    id<MTLComputePipelineState> kernel,
    void* args,
    size_t args_size,
    std::initializer_list<size_t> pointer_offsets,
    size_t num_bytes
)
{
    if (num_bytes == 0)
        return true;
    for (size_t offset : pointer_offsets) {
        if (!translate_pointer(dev, static_cast<char*>(args), args_size, offset))
            return false;
    }
    if (dev.capture) {
        graph_record(*dev.capture, kernel, (num_bytes + 15) / 16, 256, 0, nullptr, 0, args, args_size);
        return true;
    }
    id<MTLComputeCommandEncoder> encoder = get_encoder(dev);
    [encoder setComputePipelineState:kernel];
    [encoder setBytes:args length:args_size atIndex:0];
    [encoder dispatchThreads:MTLSizeMake((num_bytes + 15) / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (profiler().enabled)
        return profile_dispatch(dev, kernel.label.UTF8String);
    return ++dev.num_dispatches < max_dispatches_per_command_buffer() || flush(dev);
}

}  // anonymous namespace

int wp_is_metal_enabled() { return 1; }

int wp_metal_device_get_count()
{
    WP_METAL_LOCK();
    return int(all_devices().size());
}

const char* wp_metal_device_get_name(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    return dev ? dev->name.c_str() : nullptr;
}

int wp_metal_device_get_max_threadgroup_memory(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    return dev ? int(dev->device.maxThreadgroupMemoryLength) : 0;
}

int wp_metal_device_get_max_threads_per_threadgroup(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    return dev ? int(dev->device.maxThreadsPerThreadgroup.width) : 0;
}

int wp_metal_device_get_core_count(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    return dev ? dev->core_count : 0;
}

int wp_metal_device_has_unified_memory(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    return dev ? int(dev->device.hasUnifiedMemory) : 0;
}

void* wp_alloc_metal(int ordinal, size_t size)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev)
            return nullptr;
        // Buffers are never bound to encoders (kernels use raw GPU addresses), so hazard tracking is pointless.
        MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        id<MTLBuffer> buffer = [dev->device newBufferWithLength:std::max(size, size_t(1)) options:options];
        if (!buffer) {
            wp::set_error_string(
                "Failed to allocate %zu bytes of Metal memory on device \"%s\"", size, dev->name.c_str()
            );
            return nullptr;
        }
        void* ptr = buffer.contents;
        dev->allocations[uintptr_t(ptr)] = buffer;
        if (dev->capture)
            dev->capture_owned[uintptr_t(ptr)] = dev->capture_stack.empty() ? dev->capture : dev->capture_stack.front();
        [dev->residency_set addAllocation:buffer];
        dev->residency_dirty = true;
        dev->table_dirty = true;
        if (g_alloc_tracker.enabled)
            g_alloc_tracker.record_alloc(ptr, size, ALLOC_KIND_DEVICE, ordinal);
        return ptr;
    }
}

void wp_free_metal(int ordinal, void* ptr)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !ptr)
            return;
        auto it = dev->allocations.find(uintptr_t(ptr));
        if (it == dev->allocations.end()) {
            wp::set_error_string("Attempting to free a pointer (%p) that is not a Metal allocation", ptr);
            return;
        }
        dev->table_dirty = true;
        if (g_alloc_tracker.enabled)
            g_alloc_tracker.record_free(ptr);
        auto owned = dev->capture_owned.find(uintptr_t(ptr));
        if (dev->capture || owned != dev->capture_owned.end()) {
            // a recording references this memory (freed during capture, or allocated during one and freed
            // later): the graph keeps it alive, like CUDA graph allocations
            Graph* owner = owned != dev->capture_owned.end() ? owned->second : dev->capture;
            owner->retained.push_back(it->second);
            if (dev->capture && dev->capture != owner)
                dev->capture->retained.push_back(it->second);  // the open recording may reference it as well
            if (owned != dev->capture_owned.end())
                dev->capture_owned.erase(owned);
            dev->allocations.erase(it);
            return;
        }
        dev->deferred_frees.push_back(it->second);
        dev->allocations.erase(it);
        if (!retire_completed(*dev))
            dev->deferred_error = wp::get_error_string();  // cannot be returned from a free
        if (!has_pending_work(*dev))
            release_buffers(*dev, dev->deferred_frees);
    }
}

void wp_metal_defer_error(int ordinal)
{
    WP_METAL_LOCK();
    if (Device* dev = get_device(ordinal))
        dev->deferred_error = wp::get_error_string();
}

int wp_metal_owns_pointer(int ordinal, const void* ptr)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    if (!dev || !ptr)
        return 0;
    return owns_range(*dev, uintptr_t(ptr), 1) ? 1 : 0;
}

int wp_metal_import_host_memory(int ordinal, const void* ptr, size_t size)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !ptr)
            return 0;
        auto it = find_import(*dev, uintptr_t(ptr), size);
        if (it != dev->imports.end()) {
            ++it->second.refs;
            return 2;
        }
        if (owns_range(*dev, uintptr_t(ptr), std::max(size, size_t(1))))
            return 1;  // the whole range already is Metal memory of this device: nothing to release later
        const uintptr_t page = uintptr_t(getpagesize());
        uintptr_t base = uintptr_t(ptr) & ~(page - 1);
        uintptr_t end = (uintptr_t(ptr) + std::max(size, size_t(1)) + page - 1) & ~(page - 1);
        // grow to the union with every import the page range overlaps (pages of those imports are valid too)
        std::vector<std::map<uintptr_t, Device::Import>::iterator> overlapping;
        for (auto o = dev->imports.begin(); o != dev->imports.end(); ++o) {
            const uintptr_t o_end = o->first + o->second.buffer.length;
            if (o->first < end && base < o_end)
                overlapping.push_back(o);
        }
        for (auto o : overlapping) {
            base = std::min(base, o->first);
            end = std::max(end, uintptr_t(o->first + o->second.buffer.length));
        }
        const size_t length = size_t(end - base);
        MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        id<MTLBuffer> buffer = [dev->device newBufferWithBytesNoCopy:(void*)base
                                                              length:length
                                                             options:options
                                                         deallocator:nil];
        if (!buffer) {
            wp::set_error_string(
                "Failed to map %zu bytes of host memory at %p for Metal device \"%s\"", size, ptr, dev->name.c_str()
            );
            return 0;
        }
        Device::Import merged { buffer, 1, {} };
        for (auto o : overlapping) {
            merged.refs += o->second.refs;
            merged.absorbed.push_back(o->second.buffer);  // stays resident until the union is released
            merged.absorbed.insert(merged.absorbed.end(), o->second.absorbed.begin(), o->second.absorbed.end());
            dev->imports.erase(o);
        }
        dev->imports[base] = std::move(merged);
        [dev->residency_set addAllocation:buffer];
        dev->residency_dirty = true;
        dev->table_dirty = true;
        return 2;
    }
}

void wp_metal_release_host_memory(int ordinal, const void* ptr, size_t size)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !ptr)
            return;
        auto it = find_import(*dev, uintptr_t(ptr), size);
        if (it == dev->imports.end() || --it->second.refs > 0)
            return;
        dev->table_dirty = true;
        std::vector<id<MTLBuffer>> buffers = it->second.absorbed;
        buffers.push_back(it->second.buffer);
        dev->imports.erase(it);
        std::vector<id<MTLBuffer>>& keep = dev->capture ? dev->capture->retained : dev->deferred_frees;
        keep.insert(keep.end(), buffers.begin(), buffers.end());
        if (dev->capture)
            return;  // the recording may reference this memory: the graph keeps it alive
        if (!retire_completed(*dev))
            dev->deferred_error = wp::get_error_string();  // cannot be returned from a release
        if (!has_pending_work(*dev))
            release_buffers(*dev, dev->deferred_frees);
    }
}

uint64_t wp_metal_gpu_address(int ordinal, const void* ptr)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    return dev && ptr ? gpu_address(*dev, (uint64_t)(uintptr_t)ptr) : 0;
}

void* wp_metal_load_library(int ordinal, const char* source)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !source)
            return nullptr;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.enableLogging = printf_mode() >= 1;  // wp.printf / assertions in kernels
        options.languageVersion = MTLLanguageVersion3_2;  // 64-bit atomics, os_log
        // IEEE-strict arithmetic (no reassociation) with precise math functions; Warp kernels are
        // compared against CPU results at tight tolerances.
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError* error = nil;
        id<MTLLibrary> library = [dev->device newLibraryWithSource:@(source) options:options error:&error];
        if (!library) {
            wp::set_error_string("Metal library compilation failed:\n%s", compile_errors(error).c_str());
            return nullptr;
        }
        return new Library { library, {}, ordinal };
    }
}

void wp_metal_unload_library(void* library)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        delete static_cast<Library*>(library);
    }
}

void* wp_metal_get_kernel(void* library, const char* name)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Library* lib = static_cast<Library*>(library);
        if (!lib || !name) {
            wp::set_error_string("Invalid Metal library or kernel name");
            return nullptr;
        }
        NSError* error = nil;
        Device* dev = get_device(lib->ordinal);
        MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
        uint64_t slot_address = dev ? dev->table_slot.gpuAddress : 0;
        [constants setConstantValue:&slot_address type:MTLDataTypeULong atIndex:0];
        id<MTLFunction> function = [lib->library newFunctionWithName:@(name) constantValues:constants error:&error];
        if (!function) {
            wp::set_error_string(
                "Kernel \"%s\" was not found in the Metal library: %s", name, error_description(error)
            );
            return nullptr;
        }
        MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        descriptor.label = @(name);
        descriptor.supportIndirectCommandBuffers = YES;  // graphs replay through indirect command buffers
        id<MTLComputePipelineState> pipeline =
            [lib->library.device newComputePipelineStateWithDescriptor:descriptor
                                                               options:MTLPipelineOptionNone
                                                            reflection:nil
                                                                 error:&error];
        if (!pipeline) {
            wp::set_error_string(
                "Failed to create Metal pipeline for kernel \"%s\": %s", name, error_description(error)
            );
            return nullptr;
        }
        lib->kernels.push_back(pipeline);
        return (__bridge void*)pipeline;
    }
}

int wp_metal_get_kernel_thread_execution_width(void* kernel)
{
    return kernel ? int(((__bridge id<MTLComputePipelineState>)kernel).threadExecutionWidth) : 0;
}

int wp_metal_get_kernel_max_threads_per_threadgroup(void* kernel)
{
    return kernel ? int(((__bridge id<MTLComputePipelineState>)kernel).maxTotalThreadsPerThreadgroup) : 0;
}

int wp_metal_launch_kernel(
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
)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev)
            return -1;
        if (!kernel) {
            wp::set_error_string("Invalid Metal kernel handle");
            return -1;
        }
        if (num_threads == 0)
            return 0;

        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)kernel;
        size_t max_threads = pipeline.maxTotalThreadsPerThreadgroup;
        size_t group_size = std::min(std::max(size_t(threads_per_threadgroup), size_t(1)), max_threads);

        // the first group_size words of the threadgroup buffer hold the shared-tile arena's per-lane offsets
        threadgroup_memory_bytes += align_up(group_size * sizeof(unsigned int), kThreadgroupMemoryAlignment);
        threadgroup_memory_bytes = align_up(threadgroup_memory_bytes, kThreadgroupMemoryAlignment);
        if (threadgroup_memory_bytes > dev->device.maxThreadgroupMemoryLength) {
            wp::set_error_string(
                "Kernel requests %zu bytes of threadgroup memory but device \"%s\" supports at most "
                "%zu bytes",
                threadgroup_memory_bytes, dev->name.c_str(), size_t(dev->device.maxThreadgroupMemoryLength)
            );
            return -1;
        }

        if (dev->capture) {
            std::vector<char> translated(static_cast<const char*>(args), static_cast<const char*>(args) + args_size);
            for (int i = 0; i < num_pointer_offsets; ++i) {
                if (!translate_pointer(*dev, translated.data(), args_size, pointer_offsets[i]))
                    return -1;
            }
            graph_record(
                *dev->capture, pipeline, num_threads, group_size, threadgroup_memory_bytes, bounds, bounds_size,
                translated.data(), args_size
            );
            return 0;
        }

        size_t args_offset = stage_args(*dev, args, args_size);
        if (args_offset == SIZE_MAX)
            return -1;
        char* staged_args = static_cast<char*>(dev->args_ring.contents) + args_offset;
        for (int i = 0; i < num_pointer_offsets; ++i) {
            if (!translate_pointer(*dev, staged_args, args_size, pointer_offsets[i]))
                return -1;
        }

        if (profiler().enabled && !flush(*dev))
            return -1;
        refresh_table(*dev);
        id<MTLComputeCommandEncoder> encoder = get_encoder(*dev);
        [encoder setComputePipelineState:pipeline];
        if (bounds_size > 0)
            [encoder setBytes:bounds length:bounds_size atIndex:0];
        [encoder setBuffer:dev->args_ring offset:args_offset atIndex:1];
        MTLSize group = MTLSizeMake(group_size, 1, 1);
        if (threadgroup_memory_bytes > 0) {
            // Tile kernels rely on every thread of a threadgroup running, so dispatch whole threadgroups.
            [encoder setThreadgroupMemoryLength:threadgroup_memory_bytes atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake((num_threads + group_size - 1) / group_size, 1, 1)
                    threadsPerThreadgroup:group];
        } else {
            [encoder dispatchThreads:MTLSizeMake(num_threads, 1, 1) threadsPerThreadgroup:group];
        }

        ++dev->n_dispatch;
        if (profiler().enabled)
            return profile_dispatch(*dev, pipeline.label.UTF8String) ? 0 : -1;
        // Bound latency by committing large batches; the next launch starts a fresh command buffer.
        if (++dev->num_dispatches >= max_dispatches_per_command_buffer() && !flush(*dev))
            return -1;
        return 0;
    }
}

int wp_metal_capture_begin(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    if (!dev)
        return -1;
    if (dev->capture) {
        wp::set_error_string("Graph capture already in progress on device \"%s\"", dev->name.c_str());
        return -1;
    }
    dev->capture = new Graph();
    return 0;
}

static void graph_destroy(int ordinal, Device* dev, Graph* graph);

// Stops tracking allocations for a graph that is going away.
static void forget_capture_owned(Device& dev, Graph* graph)
{
    for (auto it = dev.capture_owned.begin(); it != dev.capture_owned.end();)
        it = it->second == graph ? dev.capture_owned.erase(it) : std::next(it);
}

// Drops a recording that was never finalized (no argument buffer yet).
static void discard_recording(Device& dev, Graph* graph)
{
    forget_capture_owned(dev, graph);
    for (id<MTLBuffer> buffer : graph->retained)
        dev.deferred_frees.push_back(buffer);
    for (const auto& child : graph->children)
        if (child.second)
            graph_destroy(0, &dev, child.first);
    delete graph;
}

// Uploads a recorded graph's argument blocks; the graph is ready to launch afterwards.
static Graph* finalize_capture(Device& dev, Graph* graph)
{
    graph->buffer =
        [dev.device newBufferWithLength:std::max(graph->bytes.size(), size_t(1))
                                options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    if (!graph->buffer) {
        wp::set_error_string("Failed to allocate %zu bytes for a Metal graph", graph->bytes.size());
        discard_recording(dev, graph);
        return nullptr;
    }
    graph->buffer.label = @"Warp graph";
    for (const auto& fixup : graph->fixups) {
        uint64_t address = graph->buffer.gpuAddress + fixup.second;
        memcpy(graph->bytes.data() + fixup.first, &address, sizeof(address));
    }
    memcpy(graph->buffer.contents, graph->bytes.data(), graph->bytes.size());
    std::vector<char>().swap(graph->bytes);
    [dev.residency_set addAllocation:graph->buffer];
    dev.residency_dirty = true;
    return graph;
}

void* wp_metal_capture_end(int ordinal)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !dev->capture) {
            wp::set_error_string("No graph capture in progress");
            return nullptr;
        }
        if (!dev->capture_stack.empty()) {
            // a branch body raised: drop every open recording so the device leaves capture mode
            discard_recording(*dev, dev->capture);
            for (Graph* enclosing : dev->capture_stack)
                discard_recording(*dev, enclosing);
            dev->capture_stack.clear();
            dev->capture = nullptr;
            wp::set_error_string("Graph capture ended while a conditional branch capture was still open");
            return nullptr;
        }
        Graph* graph = dev->capture;
        dev->capture = nullptr;
        return finalize_capture(*dev, graph);
    }
}

// Conditional nodes: the branch bodies are captured into nested graphs and the conditional itself is a
// recorded host operation that reads the condition (unified memory, after the preceding work completed)
// and launches the chosen graph, repeatedly for a while loop.
int wp_metal_capture_push(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    if (!dev || !dev->capture) {
        wp::set_error_string("No graph capture in progress");
        return -1;
    }
    dev->capture_stack.push_back(dev->capture);
    dev->capture = new Graph();
    return 0;
}

void* wp_metal_capture_pop(int ordinal)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !dev->capture || dev->capture_stack.empty()) {
            wp::set_error_string("No conditional branch capture in progress");
            return nullptr;
        }
        Graph* graph = dev->capture;
        dev->capture = dev->capture_stack.back();
        dev->capture_stack.pop_back();
        return finalize_capture(*dev, graph);
    }
}

static int graph_launch(Device& dev, Graph* graph);

int wp_metal_capture_conditional(
    int ordinal, int is_loop, const int* condition, void* on_true, void* on_false, int own_true, int own_false
)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    if (!dev || !dev->capture || !condition) {
        wp::set_error_string("No graph capture in progress");
        return -1;
    }
    Graph* g_true = static_cast<Graph*>(on_true);
    Graph* g_false = static_cast<Graph*>(on_false);
    if (g_true)
        dev->capture->children.emplace_back(g_true, own_true != 0);
    if (g_false)
        dev->capture->children.emplace_back(g_false, own_false != 0);
    Device* device = dev;
    auto op = [=]() -> bool {
        if (is_loop) {
            while (*condition) {  // the previous iteration's work completed before the read
                if (g_true && graph_launch(*device, g_true) != 0)
                    return false;
                if (!synchronize(*device))
                    return false;
            }
            return true;
        }
        Graph* branch = *condition ? g_true : g_false;
        return !branch || graph_launch(*device, branch) == 0;
    };
    dev->capture->host_ops.push_back({ dev->capture->dispatches.size(), std::move(op) });
    return 0;
}

int wp_metal_graph_launch(int ordinal, void* handle)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        Graph* graph = static_cast<Graph*>(handle);
        if (!dev || !graph) {
            wp::set_error_string("Invalid Metal graph");
            return -1;
        }
        if (dev->capture) {
            wp::set_error_string("Launching a graph during capture is not supported on Metal");
            return -1;
        }
        return graph_launch(*dev, graph);
    }
}

// Encodes the graph's dispatches into an indirect command buffer. Returns false (and leaves icb nil)
// if the device refuses; the caller then falls back to per-dispatch encoding.
static bool graph_build_icb(Device& dev, Graph* graph)
{
    graph->icb_tried = true;
    const size_t count = graph->dispatches.size();
    if (count == 0 || !graph->buffer)
        return false;
    MTLIndirectCommandBufferDescriptor* descriptor = [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
    descriptor.inheritPipelineState = NO;
    descriptor.inheritBuffers = NO;
    descriptor.maxKernelBufferBindCount = 2;
    if ([descriptor respondsToSelector:@selector(setMaxKernelThreadgroupMemoryBindCount:)])
        descriptor.maxKernelThreadgroupMemoryBindCount = 1;
    id<MTLIndirectCommandBuffer> icb = [dev.device newIndirectCommandBufferWithDescriptor:descriptor
                                                                          maxCommandCount:count
                                                                                  options:MTLResourceStorageModeShared];
    if (!icb)
        return false;
    icb.label = @"Warp graph";
    for (size_t i = 0; i < count; ++i) {
        const Graph::Dispatch& d = graph->dispatches[i];
        id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:i];
        [cmd setComputePipelineState:d.pipeline];
        if (d.bounds_size) {
            [cmd setKernelBuffer:graph->buffer offset:d.bounds_offset atIndex:0];
            [cmd setKernelBuffer:graph->buffer offset:d.args_offset atIndex:1];
        } else {
            [cmd setKernelBuffer:graph->buffer offset:d.args_offset atIndex:0];
        }
        MTLSize group = MTLSizeMake(d.group_size, 1, 1);
        if (d.threadgroup_bytes) {
            [cmd setThreadgroupMemoryLength:d.threadgroup_bytes atIndex:0];
            [cmd concurrentDispatchThreadgroups:MTLSizeMake((d.num_threads + d.group_size - 1) / d.group_size, 1, 1)
                          threadsPerThreadgroup:group];
        } else {
            [cmd concurrentDispatchThreads:MTLSizeMake(d.num_threads, 1, 1) threadsPerThreadgroup:group];
        }
        [cmd setBarrier];  // recorded launches are sequential, as on a single CUDA stream
    }
    graph->icb = icb;
    return true;
}

static int graph_launch(Device& device, Graph* graph)
{
    Device* dev = &device;
    refresh_table(device);  // recorded kernels translate host pointers they read from memory, too
    if (graph->host_ops.empty() && icb_replay_enabled()) {
        if (!graph->icb_tried)
            graph_build_icb(device, graph);
        if (graph->icb) {
            // Replayed in chunks so each command buffer carries at most WP_METAL_BATCH dispatches (the
            // same bound as eager launches); command buffers on one queue execute in order.
            const size_t count = graph->dispatches.size();
            const int batch = icb_batch() > 0 ? icb_batch() : int(std::max(count, size_t(1)) + dev->num_dispatches);
            size_t offset = 0;
            while (offset < count) {
                int room = batch - dev->num_dispatches;
                if (room <= 0) {
                    if (!flush(*dev))
                        return -1;
                    continue;
                }
                const size_t len = std::min(count - offset, size_t(room));
                id<MTLComputeCommandEncoder> encoder = get_encoder(*dev);
                [encoder useResource:graph->buffer usage:MTLResourceUsageRead];
                [encoder executeCommandsInBuffer:graph->icb withRange:NSMakeRange(offset, len)];
                dev->n_dispatch += len;
                dev->num_dispatches += int(len);
                offset += len;
                if (dev->num_dispatches >= batch && !flush(*dev))
                    return -1;
            }
            return 0;
        }
    }
    {
        size_t next_host_op = 0;
        auto run_host_ops = [&](size_t before_dispatch) {
            for (; next_host_op < graph->host_ops.size()
                 && graph->host_ops[next_host_op].before_dispatch <= before_dispatch;
                 ++next_host_op) {
                if (!synchronize(*dev))  // the host op reads results of the dispatches recorded before it
                    return false;
                ++dev->n_host_ops;
                if (!graph->host_ops[next_host_op].run())
                    return false;
            }
            return true;
        };
        for (size_t i = 0; i < graph->dispatches.size(); ++i) {
            if (!run_host_ops(i))
                return -1;
            const Graph::Dispatch& d = graph->dispatches[i];
            id<MTLComputeCommandEncoder> encoder = get_encoder(*dev);
            [encoder setComputePipelineState:d.pipeline];
            if (d.bounds_size) {
                [encoder setBuffer:graph->buffer offset:d.bounds_offset atIndex:0];
                [encoder setBuffer:graph->buffer offset:d.args_offset atIndex:1];
            } else {
                [encoder setBuffer:graph->buffer offset:d.args_offset atIndex:0];
            }
            MTLSize group = MTLSizeMake(d.group_size, 1, 1);
            if (d.threadgroup_bytes) {
                [encoder setThreadgroupMemoryLength:d.threadgroup_bytes atIndex:0];
                [encoder dispatchThreadgroups:MTLSizeMake((d.num_threads + d.group_size - 1) / d.group_size, 1, 1)
                        threadsPerThreadgroup:group];
            } else {
                [encoder dispatchThreads:MTLSizeMake(d.num_threads, 1, 1) threadsPerThreadgroup:group];
            }
            ++dev->n_dispatch;
            if (++dev->num_dispatches >= max_dispatches_per_command_buffer() && !flush(*dev))
                return -1;
        }
        return run_host_ops(graph->dispatches.size()) ? 0 : -1;
    }
}

bool wp_metal_capture_host_op(int ordinal, std::function<bool()> op)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    if (!dev || !dev->capture)
        return false;  // no capture: the caller performs the operation now
    dev->capture->host_ops.push_back({ dev->capture->dispatches.size(), std::move(op) });
    return true;
}

// Records a call to a host function taking up to 8 integer/pointer arguments (the Python utilities such as
// array_scan and radix_sort_pairs run on the host for Metal); replayed in order with the graph's dispatches.
int wp_metal_capture_host_call(int ordinal, void* fn, const unsigned long long* args, int nargs)
{
    if (!fn || nargs < 0 || nargs > 8)
        return 0;
    std::vector<unsigned long long> a(args, args + nargs);
    typedef unsigned long long U;
    auto call = [fn, a]() -> bool {
        const U* v = a.data();
        switch (a.size()) {
        case 0:
            ((void (*)())fn)();
            break;
        case 1:
            ((void (*)(U))fn)(v[0]);
            break;
        case 2:
            ((void (*)(U, U))fn)(v[0], v[1]);
            break;
        case 3:
            ((void (*)(U, U, U))fn)(v[0], v[1], v[2]);
            break;
        case 4:
            ((void (*)(U, U, U, U))fn)(v[0], v[1], v[2], v[3]);
            break;
        case 5:
            ((void (*)(U, U, U, U, U))fn)(v[0], v[1], v[2], v[3], v[4]);
            break;
        case 6:
            ((void (*)(U, U, U, U, U, U))fn)(v[0], v[1], v[2], v[3], v[4], v[5]);
            break;
        case 7:
            ((void (*)(U, U, U, U, U, U, U))fn)(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
            break;
        default:
            ((void (*)(U, U, U, U, U, U, U, U))fn)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            break;
        }
        return true;
    };
    return wp_metal_capture_host_op(ordinal, call) ? 1 : 0;
}

static void graph_destroy(int ordinal, Device* dev, Graph* graph);
void wp_metal_graph_destroy(int ordinal, void* handle)
{
    WP_METAL_LOCK();
    graph_destroy(ordinal, get_device(ordinal), static_cast<Graph*>(handle));
}
static void graph_destroy(int ordinal, Device* dev, Graph* graph)
{
    if (!dev || !graph)
        return;
    forget_capture_owned(*dev, graph);
    if (graph->buffer)
        dev->deferred_frees.push_back(graph->buffer);  // released once no GPU work can use it
    for (id<MTLBuffer> buffer : graph->retained)
        dev->deferred_frees.push_back(buffer);
    for (const auto& child : graph->children)
        if (child.second)
            graph_destroy(ordinal, dev, child.first);
    delete graph;
}

const char* wp_metal_profile_report()
{
    Profiler& p = profiler();
    std::lock_guard<std::mutex> lock(p.mutex);
    std::vector<std::pair<std::string, std::pair<double, int>>> rows(p.totals.begin(), p.totals.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    p.report.clear();
    char line[512];
    for (const auto& row : rows) {
        snprintf(
            line, sizeof(line), "%10.3f ms %6d  %s\n", row.second.first * 1e3, row.second.second, row.first.c_str()
        );
        p.report += line;
    }
    p.totals.clear();
    return p.report.c_str();
}

int wp_metal_memset(int ordinal, void* dst, int value, size_t n)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !load_memory_kernels(*dev))
            return -1;
        if (!find_gpu_address(*dev, (uint64_t)dst)) {  // plain host memory (e.g. CPU texture storage)
            if (!synchronize(*dev))
                return -1;
            memset(dst, value, n);
            return 0;
        }
        struct {
            uint64_t dst;
            uint32_t value;
            uint64_t n;
        } args = { (uint64_t)dst, (uint32_t)(unsigned char)value, n };
        return dispatch_memory_kernel(*dev, dev->memset_kernel, &args, sizeof(args), { 0 }, n) ? 0 : -1;
    }
}

int wp_metal_memtile(int ordinal, void* dst, const void* src, size_t src_size, size_t n)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !load_memory_kernels(*dev))
            return -1;
        if (!find_gpu_address(*dev, (uint64_t)dst)) {  // plain host memory
            if (!synchronize(*dev))
                return -1;
            for (size_t i = 0; i < n; ++i)
                memcpy(static_cast<char*>(dst) + i * src_size, src, src_size);
            return 0;
        }
        struct {
            uint64_t dst;
            uint64_t src;
            uint64_t src_size;
            uint64_t n;
        } args = { (uint64_t)dst, 0, src_size, n };
        if (dev->capture) {
            // the pattern lives in the graph buffer; its address is patched in at capture end
            Graph& graph = *dev->capture;
            size_t pattern_offset = graph_append(graph, src, src_size);
            size_t args_offset = align_up(graph.bytes.size(), kArgsAlignment);
            graph.fixups.emplace_back(args_offset + offsetof(decltype(args), src), pattern_offset);
        } else {
            // the host-side pattern is staged in the argument ring so the kernel can read it
            size_t pattern_offset = stage_args(*dev, src, src_size);
            if (pattern_offset == SIZE_MAX)
                return -1;
            args.src = dev->args_ring.gpuAddress + pattern_offset;
        }
        return dispatch_memory_kernel(*dev, dev->memtile_kernel, &args, sizeof(args), { 0 }, n * src_size) ? 0 : -1;
    }
}

int wp_metal_memcpy(int ordinal, void* dst, const void* src, size_t n)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !load_memory_kernels(*dev))
            return -1;
        if (!find_gpu_address(*dev, (uint64_t)dst) || !find_gpu_address(*dev, (uint64_t)src)) {  // plain host memory
            if (!synchronize(*dev))
                return -1;
            memcpy(dst, src, n);
            return 0;
        }
        struct {
            uint64_t dst;
            uint64_t src;
            uint64_t n;
        } args = { (uint64_t)dst, (uint64_t)src, n };
        return dispatch_memory_kernel(*dev, dev->memcpy_kernel, &args, sizeof(args), { 0, 8 }, n) ? 0 : -1;
    }
}

int wp_metal_synchronize(int ordinal)
{
    WP_METAL_LOCK();
    if (Device* dev = get_device(ordinal); dev && dev->capture) {
        wp::set_error_string("Cannot synchronize device \"%s\" during graph capture", dev->name.c_str());
        return -1;
    }
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !synchronize(*dev))
            return -1;
        if (!dev->deferred_error.empty()) {
            wp::set_error_string("%s", dev->deferred_error.c_str());
            dev->deferred_error.clear();
            return -1;
        }
        return 0;
    }
}

int wp_metal_flush(int ordinal)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        return dev && flush(*dev) ? 0 : -1;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Interop: handles and cross-queue ordering for other Metal users on the same device (see metal.h).

void* wp_metal_device_handle(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    return dev ? (__bridge void*)dev->device : nullptr;
}

void* wp_metal_queue_handle(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    return dev ? (__bridge void*)dev->queue : nullptr;
}

void* wp_metal_buffer_handle(int ordinal, const void* ptr, size_t* offset_out)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    if (!dev || !ptr)
        return nullptr;
    auto own = [](id<MTLBuffer> b) { return b; };
    auto imported = [](const Device::Import& i) { return i.buffer; };
    uintptr_t base = 0;
    for (bool allow_end : { false, true }) {
        id<MTLBuffer> buffer = buffer_containing(dev->allocations, uint64_t(ptr), allow_end, base, own);
        if (!buffer)
            buffer = buffer_containing(dev->imports, uint64_t(ptr), allow_end, base, imported);
        if (buffer) {
            if (offset_out)
                *offset_out = size_t(uintptr_t(ptr) - base);
            return (__bridge void*)buffer;
        }
    }
    return nullptr;
}

void* wp_metal_event_handle(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    return dev ? (__bridge void*)dev->event : nullptr;
}

uint64_t wp_metal_event_value(int ordinal)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    return dev ? dev->event_value : 0;
}

int wp_metal_signal_event(int ordinal, void* event, uint64_t value)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !event) {
            wp::set_error_string("Invalid Metal device or event");
            return -1;
        }
        if (dev->capture) {
            wp::set_error_string("Signaling an event during graph capture is not supported on Metal");
            return -1;
        }
        if (dev->encoder) {
            [dev->encoder endEncoding];
            dev->encoder = nil;
        }
        if (!dev->command_buffer) {
            // nothing pending in this buffer: still order the signal after all previously committed work
            dev->command_buffer = new_command_buffer(*dev);
            if (dev->event_value > 0)
                [dev->command_buffer encodeWaitForEvent:dev->event value:dev->event_value];
        }
        [dev->command_buffer encodeSignalEvent:(__bridge id<MTLEvent>)event value:value];
        return flush(*dev) ? 0 : -1;
    }
}

int wp_metal_wait_event(int ordinal, void* event, uint64_t value)
{
    WP_METAL_LOCK();
    @autoreleasepool {
        Device* dev = get_device(ordinal);
        if (!dev || !event) {
            wp::set_error_string("Invalid Metal device or event");
            return -1;
        }
        if (dev->capture) {
            wp::set_error_string("Waiting for an event during graph capture is not supported on Metal");
            return -1;
        }
        // Work launched before the wait must not be held back by it: commit it first, then open the
        // command buffer that every later launch is encoded into with the wait at its head. Command
        // buffers chain through dev->event, so the ordering carries over to later buffers as well.
        if (!flush(*dev))
            return -1;
        dev->command_buffer = new_command_buffer(*dev);
        if (dev->event_value > 0)
            [dev->command_buffer encodeWaitForEvent:dev->event value:dev->event_value];
        [dev->command_buffer encodeWaitForEvent:(__bridge id<MTLEvent>)event value:value];
        return 0;
    }
}

// Diagnostics: n_sync (host waits for the GPU), n_flush (command buffers committed), n_host_ops (host
// operations run during graph replay, each preceded by a wait), n_dispatch (kernel dispatches encoded).
int wp_metal_counters(int ordinal, uint64_t* out, int n)
{
    WP_METAL_LOCK();
    Device* dev = get_device(ordinal);
    if (!dev || !out)
        return -1;
    uint64_t values[4] = { dev->n_sync, dev->n_flush, dev->n_host_ops, dev->n_dispatch };
    for (int i = 0; i < n && i < 4; ++i)
        out[i] = values[i];
    return 0;
}
