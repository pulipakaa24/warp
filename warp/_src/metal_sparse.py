# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU versions of the BSR topology builders (``bsr_set_from_triplets``, ``bsr_set_transpose``) on Metal.

On Metal these used to drain the GPU and run the host implementations in ``warp/native/sparse.cpp`` (Metal arrays
are unified memory). The versions here follow the CUDA implementation in ``warp/native/sparse.cu``: one key per
triplet (or source block) with the discarded entries mapped to a sentinel that sorts last, a stable radix sort of
(key, index) pairs (``warp._src.metal_primitives``), then segment heads, an integer scan and binary searches in the
sorted keys. Every step is an ordinary kernel dispatch: nothing is read back on the host, the dispatch shapes depend
only on the host-side upper bounds, and the whole build is recorded into Metal graph captures.

Outputs, against the host implementation:

- The topology (``offsets``, ``columns``; for the transpose also the block map, padded row counts and status) is
  bitwise that of the host: the (row, column) keys of the kept entries define it, whatever the sort order of equal
  keys.
- Within a run of duplicate (row, column) triplets, the summed-triplet indices are in increasing triplet order (the
  sort is stable, as on CUDA). The host ``std::sort`` is not stable, so for runs of three or more duplicates the order,
  and with it the floating-point sum of the values, can differ from the host; runs of one or two entries sum to the
  same bits in any order (IEEE addition is commutative).
"""

from __future__ import annotations

from typing import Any

import warp as wp

_SENTINEL_LIMIT_U32 = 2**32


@wp.func
def _key_lower_bound(keys: wp.array[Any], n: int, target: wp.int64) -> int:
    """First index in ``keys[0:n]`` (sorted) whose key is not less than ``target``."""
    lo = int(0)
    hi = n
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if wp.int64(keys[mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.func
def _column_in_mask(
    row: int,
    col: int,
    mask_offsets: wp.array[int],
    mask_row_counts: wp.array[int],
    mask_columns: wp.array[int],
) -> bool:
    beg = mask_offsets[row]
    end = mask_offsets[row + 1]
    if mask_row_counts:
        end = beg + mask_row_counts[row]
    # std::lower_bound over the row's (sorted) active columns
    lo = beg
    hi = end
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if mask_columns[mid] < col:
            lo = mid + 1
        else:
            hi = mid
    return lo < end and mask_columns[lo] == col


@wp.func
def _triplet_key(
    i: int,
    count: wp.array[int],
    row_count: int,
    col_count: int,
    rows: wp.array[int],
    columns: wp.array[int],
    block_size: int,
    values: wp.array[Any],
    zero_mask: wp.uint64,
    masked: int,
    mask_offsets: wp.array[int],
    mask_row_counts: wp.array[int],
    mask_columns: wp.array[int],
    sentinel: wp.int64,
) -> wp.int64:
    if count:
        if i >= count[0]:
            return sentinel
    row = rows[i]
    col = columns[i]
    if row < 0 or row >= row_count or col < 0 or col >= col_count:
        return sentinel
    if masked != 0:
        if not _column_in_mask(row, col, mask_offsets, mask_row_counts, mask_columns):
            return sentinel
    key = wp.int64(row) * wp.int64(col_count) + wp.int64(col)
    if zero_mask == wp.uint64(0):
        return key
    # pruned if every scalar is zero under the mask (bit patterns, as on the host)
    for k in range(block_size):
        if (wp.uint64(values[i * block_size + k]) & zero_mask) != wp.uint64(0):
            return key
    return sentinel


@wp.kernel(enable_backward=False)
def _triplet_keys_u32(
    count: wp.array[int],
    row_count: int,
    col_count: int,
    rows: wp.array[int],
    columns: wp.array[int],
    block_size: int,
    values: wp.array[Any],
    zero_mask: wp.uint64,
    masked: int,
    mask_offsets: wp.array[int],
    mask_row_counts: wp.array[int],
    mask_columns: wp.array[int],
    sentinel: wp.int64,
    keys: wp.array[wp.uint32],
    indices: wp.array[int],
):
    i = wp.tid()
    key = _triplet_key(
        i,
        count,
        row_count,
        col_count,
        rows,
        columns,
        block_size,
        values,
        zero_mask,
        masked,
        mask_offsets,
        mask_row_counts,
        mask_columns,
        sentinel,
    )
    keys[i] = wp.uint32(key)
    indices[i] = i


@wp.kernel(enable_backward=False)
def _triplet_keys_u64(
    count: wp.array[int],
    row_count: int,
    col_count: int,
    rows: wp.array[int],
    columns: wp.array[int],
    block_size: int,
    values: wp.array[Any],
    zero_mask: wp.uint64,
    masked: int,
    mask_offsets: wp.array[int],
    mask_row_counts: wp.array[int],
    mask_columns: wp.array[int],
    sentinel: wp.int64,
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
):
    i = wp.tid()
    key = _triplet_key(
        i,
        count,
        row_count,
        col_count,
        rows,
        columns,
        block_size,
        values,
        zero_mask,
        masked,
        mask_offsets,
        mask_row_counts,
        mask_columns,
        sentinel,
    )
    keys[i] = wp.uint64(key)
    indices[i] = i


@wp.kernel(enable_backward=False)
def _segment_heads(n: int, sentinel: wp.int64, keys: wp.array[Any], heads: wp.array[int]):
    i = wp.tid()
    key = wp.int64(keys[i])
    head = 0
    if key < sentinel:
        if i == 0:
            head = 1
        elif wp.int64(keys[i - 1]) != key:
            head = 1
    heads[i] = head


@wp.kernel(enable_backward=False)
def _triplet_segments(
    n: int,
    sentinel: wp.int64,
    keys: wp.array[Any],
    sorted_indices: wp.array[int],
    segment_ids: wp.array[int],
    columns: wp.array[int],
    write_summed: int,
    summed_offsets: wp.array[int],
    summed_indices: wp.array[int],
    bsr_columns: wp.array[int],
):
    i = wp.tid()
    idx = sorted_indices[i]
    key = wp.int64(keys[i])
    if write_summed != 0:
        summed_indices[i] = idx
    if key >= sentinel:
        return
    s = segment_ids[i] - 1
    if i == 0 or wp.int64(keys[i - 1]) != key:
        bsr_columns[s] = columns[idx]
    if write_summed != 0:
        if i == n - 1 or wp.int64(keys[i + 1]) != key:
            summed_offsets[s] = i + 1


@wp.kernel(enable_backward=False)
def _triplet_row_offsets(
    n: int,
    col_count: int,
    keys: wp.array[Any],
    segment_ids: wp.array[int],
    bsr_offsets: wp.array[int],
):
    row = wp.tid()
    # the first kept triplet of this row or a later one; the segments before it are the blocks of the earlier rows
    first = _key_lower_bound(keys, n, wp.int64(row) * wp.int64(col_count))
    if first == 0:
        bsr_offsets[row] = 0
    else:
        bsr_offsets[row] = segment_ids[first - 1]


@wp.func
def _transpose_key(
    i: int,
    row_count: int,
    bsr_offsets: wp.array[int],
    bsr_row_counts: wp.array[int],
    bsr_columns: wp.array[int],
    sentinel: wp.int64,
) -> wp.int64:
    if i >= bsr_offsets[row_count]:
        return sentinel
    # last row whose storage starts at or before block i
    lo = int(0)
    hi = row_count + 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if bsr_offsets[mid] < i + 1:
            lo = mid + 1
        else:
            hi = mid
    row = lo - 1
    if bsr_row_counts:
        if i >= bsr_offsets[row] + bsr_row_counts[row]:
            return sentinel
    # (column, row) order; row_count > 0 here
    return wp.int64(bsr_columns[i]) * wp.int64(row_count) + wp.int64(row)


@wp.kernel(enable_backward=False)
def _transpose_keys_u32(
    row_count: int,
    bsr_offsets: wp.array[int],
    bsr_row_counts: wp.array[int],
    bsr_columns: wp.array[int],
    sentinel: wp.int64,
    keys: wp.array[wp.uint32],
    indices: wp.array[int],
):
    i = wp.tid()
    keys[i] = wp.uint32(_transpose_key(i, row_count, bsr_offsets, bsr_row_counts, bsr_columns, sentinel))
    indices[i] = i


@wp.kernel(enable_backward=False)
def _transpose_keys_u64(
    row_count: int,
    bsr_offsets: wp.array[int],
    bsr_row_counts: wp.array[int],
    bsr_columns: wp.array[int],
    sentinel: wp.int64,
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
):
    i = wp.tid()
    keys[i] = wp.uint64(_transpose_key(i, row_count, bsr_offsets, bsr_row_counts, bsr_columns, sentinel))
    indices[i] = i


@wp.kernel(enable_backward=False)
def _transpose_offsets(
    n: int, row_count: int, sentinel: wp.int64, keys: wp.array[Any], transposed_offsets: wp.array[int]
):
    col = wp.tid()
    transposed_offsets[col] = _key_lower_bound(keys, n, wp.min(wp.int64(col) * wp.int64(row_count), sentinel))


@wp.kernel(enable_backward=False)
def _transpose_columns(
    n: int,
    row_count: int,
    sentinel: wp.int64,
    keys: wp.array[Any],
    sorted_indices: wp.array[int],
    write_columns: int,
    transposed_columns: wp.array[int],
    block_indices: wp.array[int],
):
    i = wp.tid()
    key = wp.int64(keys[i])
    if key < sentinel:
        block_indices[i] = sorted_indices[i]
        if write_columns != 0:
            transposed_columns[i] = int(key - (key // wp.int64(row_count)) * wp.int64(row_count))
    elif write_columns != 0:
        transposed_columns[i] = -1


@wp.kernel(enable_backward=False)
def _transpose_scatter_padded(
    n: int,
    row_count: int,
    sentinel: wp.int64,
    exceeded_status: int,
    keys: wp.array[Any],
    sorted_indices: wp.array[int],
    transposed_offsets: wp.array[int],
    transposed_row_counts: wp.array[int],
    transposed_columns: wp.array[int],
    block_indices: wp.array[int],
    status: wp.array[int],
):
    dest_row = wp.tid()
    beg = _key_lower_bound(keys, n, wp.min(wp.int64(dest_row) * wp.int64(row_count), sentinel))
    end = _key_lower_bound(keys, n, wp.min(wp.int64(dest_row + 1) * wp.int64(row_count), sentinel))
    block_count = end - beg

    dest_beg = transposed_offsets[dest_row]
    dest_capacity_end = transposed_offsets[dest_row + 1]

    if dest_beg + block_count > dest_capacity_end:
        transposed_row_counts[dest_row] = 0
        if status:
            wp.atomic_max(status, 0, exceeded_status)
        for block in range(dest_beg, dest_capacity_end):
            block_indices[block] = -1
        return

    transposed_row_counts[dest_row] = block_count
    for block in range(dest_beg, dest_capacity_end):
        local_block = block - dest_beg
        if local_block < block_count:
            key = wp.int64(keys[beg + local_block])
            transposed_columns[block] = int(key - (key // wp.int64(row_count)) * wp.int64(row_count))
            block_indices[block] = sorted_indices[beg + local_block]
        else:
            block_indices[block] = -1


def enabled(device) -> bool:
    """Whether BSR topology builds run as GPU kernels on ``device`` (Metal, ``warp.config.metal_gpu_utilities``)."""
    return device.is_metal and wp.config.metal_gpu_utilities


class _Scratch:
    """Grow-only per-device work buffers. Outgrown buffers stay alive: a captured graph may still reference them
    (as the scratch pool of ``warp._src.metal_primitives``)."""

    def __init__(self):
        self.buffers = {}
        self.outgrown = []

    def get(self, name, dtype, size, device):
        buf = self.buffers.get(name)
        if buf is None or buf.dtype != dtype or buf.size < size:
            if buf is not None:
                self.outgrown.append(buf)
            grown = 2 * buf.size if buf is not None and buf.dtype == dtype else 0
            buf = wp.empty(max(size, grown, 1024), dtype=dtype, device=device)
            self.buffers[name] = buf
        return buf


def _scratch(device) -> _Scratch:
    s = device.__dict__.get("_metal_sparse_scratch")
    if s is None:
        s = _Scratch()
        device.__dict__["_metal_sparse_scratch"] = s
    return s


def _sorted_keys(device, n: int, sentinel: int):
    """Key (2n) and index (2n) work buffers for ``n`` entries whose kept keys are below ``sentinel``."""
    scratch = _scratch(device)
    wide = sentinel >= _SENTINEL_LIMIT_U32
    keys = scratch.get("keys64" if wide else "keys32", wp.uint64 if wide else wp.uint32, 2 * n, device)
    indices = scratch.get("indices", wp.int32, 2 * n, device)
    return keys, indices, wide


def _sort(keys, indices, n, sentinel):
    from warp._src.utils import radix_sort_pairs  # noqa: PLC0415

    # every key is at most the sentinel, so its bits above the sentinel's are zero
    radix_sort_pairs(keys, indices, n, 0, max(int(sentinel).bit_length(), 1))


_ZERO_VIEW_TYPES = {1: wp.uint8, 2: wp.uint16, 4: wp.uint32, 8: wp.uint64}


def bsr_from_triplets(
    row_count: int,
    col_count: int,
    block_size: int,
    nnz: int,
    count,
    rows,
    columns,
    values,
    scalar_size: int,
    zero_value_mask: int,
    masked: bool,
    summed_offsets,
    summed_indices,
    bsr_offsets,
    bsr_row_counts,
    bsr_columns,
) -> None:
    """GPU counterpart of ``wp_bsr_matrix_from_triplets_host`` (same arguments and outputs, see module notes)."""
    device = rows.device
    n = nnz
    sentinel = row_count * col_count
    keys, indices, wide = _sorted_keys(device, n, sentinel)

    prune = values is not None and zero_value_mask != 0 and scalar_size in _ZERO_VIEW_TYPES
    if prune:
        raw_values = wp.array(
            ptr=values.ptr,
            dtype=_ZERO_VIEW_TYPES[scalar_size],
            shape=(n * block_size,),
            device=device,
            copy=False,
        )
    else:
        raw_values = wp.empty(0, dtype=wp.uint32, device=device)

    wp.launch(
        _triplet_keys_u64 if wide else _triplet_keys_u32,
        dim=n,
        device=device,
        inputs=[
            count,
            row_count,
            col_count,
            rows,
            columns,
            block_size,
            raw_values,
            wp.uint64(zero_value_mask if prune else 0),
            int(masked),
            bsr_offsets if masked else None,
            bsr_row_counts if masked else None,
            bsr_columns if masked else None,
            wp.int64(sentinel),
            keys,
            indices,
        ],
    )
    _sort(keys, indices, n, sentinel)

    from warp._src.utils import array_scan  # noqa: PLC0415

    segment_ids = _scratch(device).get("segments", wp.int32, n, device)
    segment_ids = segment_ids[:n]
    wp.launch(_segment_heads, dim=n, device=device, inputs=[n, wp.int64(sentinel), keys, segment_ids])
    array_scan(segment_ids, segment_ids, inclusive=True)

    write_summed = summed_offsets is not None and summed_indices is not None
    wp.launch(
        _triplet_segments,
        dim=n,
        device=device,
        inputs=[
            n,
            wp.int64(sentinel),
            keys,
            indices,
            segment_ids,
            columns,
            int(write_summed),
            summed_offsets if write_summed else None,
            summed_indices if write_summed else None,
            bsr_columns,
        ],
    )
    wp.launch(
        _triplet_row_offsets,
        dim=row_count + 1,
        device=device,
        inputs=[n, col_count, keys, segment_ids, bsr_offsets],
    )


def bsr_transpose(
    row_count: int,
    col_count: int,
    nnz: int,
    bsr_offsets,
    bsr_row_counts,
    bsr_columns,
    transposed_offsets,
    transposed_row_counts,
    transposed_columns,
    block_indices,
    status,
    exceeded_status: int,
) -> None:
    """GPU counterpart of ``wp_bsr_transpose_host`` (same arguments and outputs).

    ``transposed_row_counts`` selects the padded variant (``transposed_offsets`` then holds the destination row
    capacities and is read, not written).
    """
    device = bsr_columns.device
    n = nnz
    padded = transposed_row_counts is not None
    sentinel = col_count * row_count
    keys, indices, wide = _sorted_keys(device, n, sentinel)

    if n > 0:
        wp.launch(
            _transpose_keys_u64 if wide else _transpose_keys_u32,
            dim=n,
            device=device,
            inputs=[row_count, bsr_offsets, bsr_row_counts, bsr_columns, wp.int64(sentinel), keys, indices],
        )
        _sort(keys, indices, n, sentinel)
        # the sorted kept blocks first (the padded scatter below then overwrites the destination slots)
        wp.launch(
            _transpose_columns,
            dim=n,
            device=device,
            inputs=[
                n,
                row_count,
                wp.int64(sentinel),
                keys,
                indices,
                int(not padded),
                transposed_columns,
                block_indices,
            ],
        )

    if padded:
        wp.launch(
            _transpose_scatter_padded,
            dim=col_count,
            device=device,
            inputs=[
                n,
                row_count,
                wp.int64(sentinel),
                exceeded_status,
                keys,
                indices,
                transposed_offsets,
                transposed_row_counts,
                transposed_columns,
                block_indices,
                status,
            ],
        )
    else:
        wp.launch(
            _transpose_offsets,
            dim=col_count + 1,
            device=device,
            inputs=[n, row_count, wp.int64(sentinel), keys, transposed_offsets],
        )
