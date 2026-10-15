# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``wp.preload_launches()``: record launches, then load their modules in parallel."""

import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices


def _make_scale_kernel(factor: float):
    """Build a kernel in its own module, like the specialized kernels MuJoCo Warp builds while it steps."""

    @wp.kernel(module="unique")
    def scale(a: wp.array[float], out: wp.array[float]):
        i = wp.tid()
        out[i] = a[i] * factor

    return scale


@wp.kernel
def _add_one(a: wp.array[float]):
    i = wp.tid()
    a[i] = a[i] + 1.0


@wp.kernel
def _generic_copy(src: wp.array[Any], dst: wp.array[Any]):
    i = wp.tid()
    dst[i] = src[i]


def _loaded_variants(kernel, device):
    return [key for key in kernel.module.execs if key[0] == device.context]


def _default_variant_loaded(kernel, device):
    """Return whether the module variant an ordinary launch uses is loaded.

    Other block_dim variants of a shared module may be loaded too (other tests launch kernels of this file's module
    with other block sizes).
    """
    block_dim = wp._src.context._resolve_launch_block_dim(device, None)
    return (device.context, block_dim) in kernel.module.execs


def _run_isolated(test, script, *args):
    """Run ``script`` in a fresh process.

    A deliberately broken kernel stays registered for the life of a process, and ``wp.force_load()`` of all modules
    (e.g. ``ScopedCapture(force_module_load=True)``) would then raise for it in unrelated tests.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "isolated_check.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent(script))
        warp_root = os.path.dirname(os.path.dirname(os.path.abspath(wp.__file__)))
        out = subprocess.run(
            [sys.executable, path, warp_root, os.path.join(tmp, "cache"), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    test.assertEqual(out.returncode, 0, out.stdout[-3000:] + out.stderr[-3000:])
    return out.stdout


_BUILD_ERROR_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])  # the Warp under test, not another installed copy
import numpy as np
import warp as wp
wp.config.kernel_cache_dir = sys.argv[2]
wp.config.quiet = True
wp.init()


@wp.kernel(module="unique")
def bad(a: wp.array[float]):
    i = wp.tid()
    a[i] = undefined_name_for_preload_test  # noqa: F821


@wp.kernel(module="unique")
def good(a: wp.array[float]):
    i = wp.tid()
    a[i] = a[i] + 1.0


@wp.kernel(module="unique")
def later(a: wp.array[float]):
    i = wp.tid()
    a[i] = a[i] * 2.0


for device in ["cpu"] + (["metal:0"] if wp.is_metal_available() else []):
    a = wp.zeros(4, dtype=float, device=device)
    with wp.preload_launches(device, max_workers=2) as preload:
        wp.launch(bad, dim=4, inputs=[a], device=device)
        wp.launch(good, dim=4, inputs=[a], device=device)
    assert len(preload.errors) == 1 and "bad" in preload.errors[0][0], preload.errors
    # unrelated launches after the failed preload: the recorded error belongs to bad's module only
    wp.launch(good, dim=4, inputs=[a], device=device)
    wp.launch(later, dim=4, inputs=[a], device=device)
    assert np.array_equal(a.numpy(), np.full(4, 2.0, dtype=np.float32)), a.numpy()
    wp.force_load(device=device, modules=[later.module, good.module])
    # a later preload of other kernels reports no error
    with wp.preload_launches(device, max_workers=2) as again:
        wp.launch(later, dim=4, inputs=[a], device=device)
    assert again.errors == [], again.errors
    # the bad kernel itself still raises its own error at launch
    try:
        wp.launch(bad, dim=4, inputs=[a], device=device)
    except Exception as e:
        assert "undefined_name_for_preload_test" in str(e), e
    else:
        raise AssertionError("the broken kernel launched")
print("ISOLATED_OK")
"""


_RECORDS_AND_LOADS_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])  # the Warp under test, not another installed copy
import numpy as np
import warp as wp
import warp._src.context as ctx
wp.config.kernel_cache_dir = sys.argv[2]
wp.config.quiet = True
wp.init()
device = wp.get_device(sys.argv[3])


def make(factor):
    # own module per kernel; constant-memory arguments on Metal, so one forward pipeline per kernel and no
    # small-launch entry point created lazily at a launch
    @wp.kernel(module="unique", module_options={"metal_device_args": False})
    def scale(a: wp.array[float], out: wp.array[float]):
        i = wp.tid()
        out[i] = a[i] * factor

    return scale


# count every compile a launch could trigger: module codegen + build (all devices), Metal library and pipeline
# compiles, CPU object loads and entry-point lookups
counts = {"compile": 0, "metal_library": 0, "metal_pipeline": 0, "cpu_load": 0, "cpu_lookup": 0}
orig_compile = ctx.Module._compile


def counting_compile(self, *args, **kwargs):
    counts["compile"] += 1
    return orig_compile(self, *args, **kwargs)


ctx.Module._compile = counting_compile


def wrap(lib, name, key):
    fn = getattr(lib, name, None)
    if fn is None:
        return

    def counted(*args):
        counts[key] += 1
        return fn(*args)

    setattr(lib, name, counted)


core, llvm = ctx.runtime.core, ctx.runtime.llvm
wrap(core, "wp_metal_load_library", "metal_library")
wrap(core, "wp_metal_get_kernel", "metal_pipeline")
if llvm is not None:
    wrap(llvm, "wp_load_obj", "cpu_load")
    wrap(llvm, "wp_lookup", "cpu_lookup")

kernels = [make(float(k) + 0.5) for k in range(6)]
a = wp.array(np.arange(8, dtype=np.float32), device=device)
outs = [wp.zeros(8, dtype=float, device=device) for _ in kernels]

wp.synchronize_device(device)
start = dict(counts)
with wp.preload_launches(device, max_workers=4) as preload:
    for kernel, out in zip(kernels, outs):
        wp.launch(kernel, dim=8, inputs=[a], outputs=[out], device=device)
    # inside the block the launches are only recorded: nothing compiled or loaded
    assert counts == start, (start, counts)
for out in outs:  # and nothing ran
    assert np.array_equal(out.numpy(), np.zeros(8, dtype=np.float32))
assert preload.errors == [], preload.errors
assert (preload.modules, preload.kernels, preload.workers) == (6, 6, 4), vars(preload)
assert counts["compile"] >= start["compile"] + 6, counts  # every module compiled by the preload

# the guarantee: launching a preloaded kernel compiles and loads nothing (counted before any readback)
before = dict(counts)
for kernel, out in zip(kernels, outs):
    wp.launch(kernel, dim=8, inputs=[a], outputs=[out], device=device)
after = dict(counts)
assert after == before, (before, after)
for k, out in enumerate(outs):
    assert np.allclose(out.numpy(), np.arange(8, dtype=np.float32) * (float(k) + 0.5)), out.numpy()

# control: a kernel that was not preloaded does compile at its launch (the counters see compiles)
lazy, lazy_out = make(9.5), wp.zeros(8, dtype=float, device=device)
wp.synchronize_device(device)
before = dict(counts)
wp.launch(lazy, dim=8, inputs=[a], outputs=[lazy_out], device=device)
assert counts["compile"] > before["compile"], counts
if device.is_metal:
    assert counts["metal_library"] > before["metal_library"] and counts["metal_pipeline"] > before["metal_pipeline"]
else:
    assert counts["cpu_load"] > before["cpu_load"] and counts["cpu_lookup"] > before["cpu_lookup"], counts
print("ISOLATED_OK", counts)
"""


def test_preload_records_and_loads(test, device):
    """Record six launches without running them, load them in parallel, then launch them with no compile or load.

    Runs in a fresh process with fresh kernels and a fresh kernel cache, so test order and the other tests' modules
    cannot matter. The counters cover every compile a launch could trigger on the device.
    """
    out = _run_isolated(test, _RECORDS_AND_LOADS_SCRIPT, str(device))
    test.assertIn("ISOLATED_OK", out)


def test_preload_generic_and_block_dim(test, device):
    src = wp.array(np.arange(4, dtype=np.float32), device=device)
    dst = wp.zeros(4, dtype=float, device=device)
    block_dim = 64 if not device.is_cpu else None
    with wp.preload_launches(device, max_workers=2) as preload:
        wp.launch(_generic_copy, dim=4, inputs=[src, dst], device=device, block_dim=block_dim)
    test.assertEqual(preload.errors, [])
    test.assertEqual(preload.modules, 1)
    wp.launch(_generic_copy, dim=4, inputs=[src, dst], device=device, block_dim=block_dim)
    np.testing.assert_array_equal(dst.numpy(), src.numpy())


def test_preload_other_device_and_record_cmd_pass_through(test, device):
    a = wp.zeros(4, dtype=float, device=device)
    with wp.preload_launches("cpu" if not device.is_cpu else None, max_workers=2):
        if not device.is_cpu:
            # launches on another device than the recorded one run normally
            wp.launch(_add_one, dim=4, inputs=[a], device=device)
        cmd = wp.launch(_add_one, dim=4, inputs=[a], device=device, record_cmd=True)
    test.assertIsNotNone(cmd)
    cmd.launch()
    expected = 2.0 if not device.is_cpu else 1.0
    np.testing.assert_array_equal(a.numpy(), np.full(4, expected, dtype=np.float32))


def test_preload_serial_matches_parallel_state(test, device):
    kernels = [_make_scale_kernel(float(k) + 10.25) for k in range(3)]
    a = wp.zeros(4, dtype=float, device=device)
    with wp.preload_launches(device, max_workers=0) as preload:
        for kernel in kernels:
            wp.launch(kernel, dim=4, inputs=[a], outputs=[a], device=device)
    test.assertEqual(preload.workers, 1)
    test.assertEqual(preload.errors, [])
    for kernel in kernels:
        test.assertTrue(_default_variant_loaded(kernel, device), kernel.key)


def test_preload_add_recorded_kernels(test, device):
    kernel = _make_scale_kernel(7.75)
    with wp.preload_launches(device, max_workers=2) as preload:
        preload.add(kernel)
    test.assertEqual(preload.errors, [])
    test.assertEqual(preload.modules, 1)
    test.assertTrue(_default_variant_loaded(kernel, device), kernel.key)
    with test.assertRaises(RuntimeError):
        with wp.preload_launches(device) as preload:
            preload.add(_generic_copy)


class TestPreloadLaunches(unittest.TestCase):
    def test_nested_raises(self):
        with wp.preload_launches("cpu"):
            with self.assertRaises(RuntimeError):
                with wp.preload_launches("cpu"):
                    pass

    def test_other_threads_not_intercepted(self):
        a = wp.zeros(4, dtype=float, device="cpu")
        with wp.preload_launches("cpu"):
            t = threading.Thread(target=lambda: wp.launch(_add_one, dim=4, inputs=[a], device="cpu"))
            t.start()
            t.join()
        np.testing.assert_array_equal(a.numpy(), np.ones(4, dtype=np.float32))

    def test_build_error_surfaces_at_launch(self):
        """Report a kernel that fails to build in ``errors`` and raise only at its own launch.

        Launches and preloads of other kernels afterwards are unaffected. Runs in its own process, so the broken
        kernel does not stay registered in the test process.
        """
        out = _run_isolated(self, _BUILD_ERROR_SCRIPT)
        self.assertIn("ISOLATED_OK", out)

    def test_cache_files_identical_to_lazy_loading(self):
        """Serial lazy loading and a parallel preload write byte-identical kernel cache entries."""
        script = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, sys.argv[3])  # the Warp under test, not another installed copy
            import warp as wp
            wp.config.kernel_cache_dir = sys.argv[1]
            wp.config.quiet = True
            preload = sys.argv[2] == "1"
            wp.init()

            def make(f):
                @wp.kernel(module="unique")
                def k(a: wp.array[float]):
                    i = wp.tid()
                    a[i] = wp.sin(a[i]) * f + float(i)
                return k

            ks = [make(float(j) + 0.5) for j in range(8)]
            a = wp.zeros(16, dtype=float, device="cpu")
            if preload:
                with wp.preload_launches("cpu", max_workers=8) as p:
                    for k in ks:
                        wp.launch(k, dim=16, inputs=[a], device="cpu")
                assert not p.errors, p.errors
            for k in ks:
                wp.launch(k, dim=16, inputs=[a], device="cpu")
            print(float(a.numpy().sum()))
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "preload_cache_check.py")
            with open(path, "w") as f:
                f.write(script)
            outputs = []
            trees = []
            for mode in ("0", "1"):
                cache = os.path.join(tmp, f"cache{mode}")
                out = subprocess.run(
                    [sys.executable, path, cache, mode, os.path.dirname(os.path.dirname(os.path.abspath(wp.__file__)))],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(out.returncode, 0, out.stderr)
                outputs.append(out.stdout.strip().splitlines()[-1])
                tree = {}
                for root, _, files in os.walk(cache):
                    for name in files:
                        if name.endswith((".cpp", ".meta", ".metal", ".metal.meta")):
                            full = os.path.join(root, name)
                            with open(full, "rb") as fh:
                                tree[os.path.relpath(full, cache)] = fh.read()
                trees.append(tree)
            self.assertEqual(outputs[0], outputs[1])
            self.assertTrue(trees[0])
            self.assertEqual(sorted(trees[0]), sorted(trees[1]))
            for name in trees[0]:
                self.assertEqual(trees[0][name], trees[1][name], name)


devices = get_test_devices()

add_function_test(
    TestPreloadLaunches, "test_preload_records_and_loads", test_preload_records_and_loads, devices=devices
)
add_function_test(
    TestPreloadLaunches, "test_preload_generic_and_block_dim", test_preload_generic_and_block_dim, devices=devices
)
add_function_test(
    TestPreloadLaunches,
    "test_preload_other_device_and_record_cmd_pass_through",
    test_preload_other_device_and_record_cmd_pass_through,
    devices=devices,
)
add_function_test(
    TestPreloadLaunches,
    "test_preload_serial_matches_parallel_state",
    test_preload_serial_matches_parallel_state,
    devices=devices,
)

add_function_test(
    TestPreloadLaunches, "test_preload_add_recorded_kernels", test_preload_add_recorded_kernels, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
