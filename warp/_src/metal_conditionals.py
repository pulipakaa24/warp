# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager :func:`warp.capture_while` / :func:`warp.capture_if` on Metal without reading the condition every iteration.

Outside a capture, CUDA and Metal run a ``capture_while`` loop from the host: read the condition back, run the
body, repeat; every iteration is a host wait for the device. On Metal the body is recorded once into a graph whose
dispatches form an indirect execution range (:func:`warp._src.context.metal_capture_range_begin`): a kernel at the
end of the body zeroes the range's length once the condition is 0, so replays past the end of the loop execute
nothing. The host submits a batch of replays (as many as the previous call of the loop submitted), reads the
condition, and doubles the batch while the loop goes on. ``capture_if`` likewise records the branch bodies as two
ranges selected on the device: no read at all. The kernels executed, their order and their arguments are those of
the host loop, so results are bitwise the same.

Bodies that record host operations (nested conditionals, host-side utilities) cannot be skipped by a range, and
bodies that synchronize or read back cannot be recorded; both keep the host loop. Opt-in: ``warp.config.metal_eager_conditionals = True`` (``WP_METAL_EAGER_CONDITIONALS=1``).
"""

from __future__ import annotations

import warp
from warp._src import context as _ctx


@warp.kernel(enable_backward=False)
def _while_open(condition: warp.array[warp.int32], ranges: warp.array[warp.uint32]):
    # before the first replay: run the body only if the condition holds (a while loop checks first)
    if condition[0] == 0:
        ranges[1] = warp.uint32(0)
    else:
        ranges[1] = ranges[2]


@warp.kernel(enable_backward=False)
def _while_gate(condition: warp.array[warp.int32], ranges: warp.array[warp.uint32]):
    # end of the body: once the condition is 0, later replays of the body execute nothing
    if condition[0] == 0:
        ranges[1] = warp.uint32(0)


@warp.kernel(enable_backward=False)
def _if_select(condition: warp.array[warp.int32], ranges: warp.array[warp.uint32]):
    # ranges[0:4]: on_true body, ranges[4:8]: on_false body
    if condition[0] != 0:
        ranges[1] = ranges[2]
        ranges[5] = warp.uint32(0)
    else:
        ranges[1] = warp.uint32(0)
        ranges[5] = ranges[6]


# replays the previous call of a loop body submitted (the next call's first batch)
_while_counts: dict = {}
_MAX_FIRST_BATCH = 256
# bodies that could not be recorded (they synchronize or read back): the host loop runs them
_unrecordable: set = set()


def _load(device):
    for k in (_while_open, _while_gate, _if_select):
        k.module.load(device)


def _record(device, bodies, kwargs, gate_condition=None):
    """Record ``bodies`` (callables) as consecutive indirect ranges of one graph; ``gate_condition`` adds the while
    gate inside the (single) range. Returns ``(graph, ranges)`` or ``None`` when the recording cannot be gated."""
    key = tuple(getattr(b, "__qualname__", None) or id(b) for b in bodies)
    if key in _unrecordable:
        return None
    ranges = warp.empty(4 * len(bodies), dtype=warp.uint32, device=device)
    try:
        with warp.ScopedCapture(device=device, force_module_load=False) as capture:
            for j, body in enumerate(bodies):
                _ctx.metal_capture_range_begin(ranges, j, device=device)
                if body is not None:
                    body(**kwargs)
                if gate_condition is not None:
                    warp.launch(_while_gate, dim=1, inputs=[gate_condition, ranges], device=device)
                _ctx.metal_capture_range_end(device=device)
    except Exception:
        # the body does something a recording cannot hold (a synchronize, a read back): nothing was executed on
        # the device; remember it and let the caller run the host loop
        _unrecordable.add(key)
        return None
    graph = capture.graph
    if not _ctx.runtime.core.wp_metal_graph_ranges_honoured(graph.metal_graph):
        return None
    return graph, ranges


def eager_while(device, condition, while_body, kwargs) -> bool:
    """Run ``while condition: while_body(**kwargs)`` on a Metal device; False if the caller must use the host loop."""
    _load(device)
    rec = _record(device, [while_body], kwargs, gate_condition=condition)
    if rec is None:
        return False
    graph, ranges = rec
    warp.launch(_while_open, dim=1, inputs=[condition, ranges], device=device)
    key = getattr(while_body, "__qualname__", None) or id(while_body)
    # first batch: as many replays as the previous call of this loop needed reads for (gated replays past the end
    # of the loop execute nothing); while the loop goes on, the batch doubles, so a read is paid per doubling
    batch = _while_counts.get(key, 1)
    replays = 0
    while True:
        for _ in range(batch):
            warp.capture_launch(graph)
        replays += batch
        if not _ctx._read_condition(condition, device, None):
            break
        batch *= 2
    _while_counts[key] = min(max(1, replays), _MAX_FIRST_BATCH)
    return True


def eager_if(device, condition, on_true, on_false, kwargs) -> bool:
    """Run ``on_true(**kwargs) if condition else on_false(**kwargs)`` on a Metal device without a host read."""
    _load(device)
    rec = _record(device, [on_true, on_false], kwargs)
    if rec is None:
        return False
    graph, ranges = rec
    warp.launch(_if_select, dim=1, inputs=[condition, ranges], device=device)
    warp.capture_launch(graph)
    return True
