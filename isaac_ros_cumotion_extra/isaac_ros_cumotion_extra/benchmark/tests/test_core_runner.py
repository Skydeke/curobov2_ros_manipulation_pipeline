# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python tests for core_runner helpers that do not need torch/curobo.

``_reference_benchmark_module`` must locate the upstream
``benchmark/motion_plan_benchmark.py`` by file path (``curobo.benchmark`` is
not an importable package) and load it without ever touching curobo_core, so it
is exercised here with stub scripts on a synthetic sys.path.
"""

import builtins
import os
import sys
import types

import pytest

from isaac_ros_cumotion_extra.benchmark.core_runner import (
    _reference_benchmark_module,
)

_STUB = '''\
"""Fake upstream reference benchmark."""

CHECKED = "ok"


def check_problems(problems):
    return sum(problems)


def load_curobo(n_cubes, ik_seeds=None, trajopt_seeds=4, mpinets=False,
                collision_buffer=0.0, args=None):
    return {"n_cubes": n_cubes, "seeds": (ik_seeds, trajopt_seeds)}
'''


def _write_stub(root: str, in_package: bool) -> None:
    """Write the stub script; ``in_package`` -> <root>/curobo/benchmark/...,
    else the upstream layout <root>/benchmark/... (beside the package)."""
    base = os.path.join(root, "curobo", "benchmark") if in_package else os.path.join(root, "benchmark")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "motion_plan_benchmark.py"), "w") as f:
        f.write(_STUB)


def _block_curobo_import(monkeypatch) -> None:
    """Make ``import curobo`` raise so discovery goes through sys.path only —
    hermetic both in the local sandbox (no curobo) and in the docker image
    (curobo installed, and the real script would otherwise be found via
    ``curobo.__file__`` instead of the stub)."""
    real_import = builtins.__import__

    def no_curobo(name, *args, **kwargs):
        if name == "curobo":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_curobo)


class TestReferenceBenchmarkModule:
    def test_upstream_project_root_layout(self, tmp_path, monkeypatch):
        """The real docker layout: script at <editable-root>/benchmark/,
        beside the ``curobo/`` package dir (upstream runs it as a file)."""
        root = str(tmp_path)
        _write_stub(root, in_package=False)
        monkeypatch.syspath_prepend(root)
        _block_curobo_import(monkeypatch)

        module = _reference_benchmark_module()

        assert module.CHECKED == "ok"
        assert module.check_problems([2, 3]) == 5
        assert module.load_curobo(12, 32, 4)["n_cubes"] == 12
        assert module.load_curobo(12, 32, 4)["seeds"] == (32, 4)

    def test_pep660_editable_install_layout(self, tmp_path, monkeypatch):
        """The docker image layout: the project root is *not* on sys.path
        (PEP 660 editable install); `curobo` resolves into <root>/curobo and
        the script sits at <root>/benchmark, one level above the package dir.
        The project root must be derived from ``curobo.__file__``."""
        root = str(tmp_path)
        _write_stub(root, in_package=False)  # <root>/benchmark/motion_plan_benchmark.py

        fake_curobo = types.ModuleType("curobo")
        fake_curobo.__file__ = os.path.join(root, "curobo", "__init__.py")
        monkeypatch.setitem(sys.modules, "curobo", fake_curobo)
        # NOTE: `root` is intentionally NOT prepended to sys.path here.

        module = _reference_benchmark_module()
        assert module.check_problems([1, 1]) == 2
        assert module.load_curobo(12, 32, 4)["seeds"] == (32, 4)

    def test_colcon_symlink_install_layout(self, tmp_path, monkeypatch):
        """colcon ``--symlink-install`` copies: site-packages holds a symlinked
        ``curobo`` dir whose ``__file__`` must be realpath'd back to the source
        tree before the project root (= where ``benchmark/`` lives) is derived."""
        source_root = str(tmp_path / "src")
        _write_stub(source_root, in_package=False)  # <src>/benchmark/...

        site = str(tmp_path / "site-packages")
        os.makedirs(site)
        os.symlink(
            os.path.join(source_root, "curobo"), os.path.join(site, "curobo")
        )

        fake_curobo = types.ModuleType("curobo")
        fake_curobo.__file__ = os.path.join(site, "curobo", "__init__.py")
        monkeypatch.setitem(sys.modules, "curobo", fake_curobo)

        module = _reference_benchmark_module()
        assert module.check_problems([1, 1]) == 2
        assert module.load_curobo(12, 32, 4)["seeds"] == (32, 4)

    def test_finds_script_inside_package_via_curobo_file(self, tmp_path, monkeypatch):
        """Vendored/forks with benchmark/ inside the package: resolved from
        ``curobo.__file__`` rather than sys.path search."""
        root = str(tmp_path)
        _write_stub(root, in_package=True)
        monkeypatch.syspath_prepend(root)

        fake_curobo = types.ModuleType("curobo")
        fake_curobo.__file__ = os.path.join(root, "curobo", "__init__.py")
        monkeypatch.setitem(sys.modules, "curobo", fake_curobo)

        module = _reference_benchmark_module()
        assert module.check_problems([1, 1]) == 2
        assert module.load_curobo(12, 32, 4)["seeds"] == (32, 4)

    def test_raises_when_not_found(self, tmp_path, monkeypatch):
        _block_curobo_import(monkeypatch)
        monkeypatch.setattr(sys, "path", [str(tmp_path / "empty")])

        with pytest.raises(ImportError, match="motion_plan_benchmark.py"):
            _reference_benchmark_module()