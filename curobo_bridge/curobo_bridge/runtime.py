# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

_REGISTRY: Dict[int, dict] = {}
_NEXT_HANDLE = 1
_LANES: Dict[str, object] = {}
_GLOBAL_FLAGS = {"cuda_graph_reset": False, "debug_cuda_graphs": False}
def init(cfg: Optional[dict] = None) -> int:
    """Build the shared cuRobo objects and return their handle.

    M1 records the config; the cuRobo construction lands at M2.
    """
    global _NEXT_HANDLE
    handle = _NEXT_HANDLE
    _NEXT_HANDLE += 1
    _REGISTRY[handle] = {"cfg": dict(cfg or {}), "built": False}
    return handle
def shutdown(handle: int) -> None:
    """Release the registry entry for ``handle``."""
    _REGISTRY.pop(int(handle), None)
def bind_lane(name: str) -> None:
    """Create the calling lane's CUDA stream and store it thread-locally.

    No-op without torch/CUDA (unit tests, CI without a GPU).
    """
    try:
        import torch  # noqa: F401
        if torch.cuda.is_available():
            _LANES[name] = torch.cuda.Stream(device=torch.device("cuda", torch.cuda.current_device()))
            return
    except Exception:
        pass
    _LANES[name] = None
def set_global_flags(cuda_graph_reset: bool, debug_cuda_graphs: bool) -> None:
    """Process-global cuRobo flags (PORT: config_wrapper_motion import effects)."""
    import curobo._src.runtime as runtime_mod

    runtime_mod.cuda_graph_reset = bool(cuda_graph_reset)
    runtime_mod.debug_cuda_graphs = bool(debug_cuda_graphs)
    # `curobo/runtime.py` re-exports these with `from curobo._src.runtime import`,
    # so it holds its OWN binding. Consumers that do `from curobo import runtime`
    # (is_cuda_graph_reset_available, cuda_graph_util) read THAT copy - setting
    # only `curobo._src.runtime` here left the flag effectively off and made
    # reset_cuda_graph() raise. Update both bindings.
    try:
        import curobo.runtime as runtime_pkg

        runtime_pkg.cuda_graph_reset = bool(cuda_graph_reset)
        runtime_pkg.debug_cuda_graphs = bool(debug_cuda_graphs)
    except Exception:
        pass
    _GLOBAL_FLAGS["cuda_graph_reset"] = bool(cuda_graph_reset)
    _GLOBAL_FLAGS["debug_cuda_graphs"] = bool(debug_cuda_graphs)
def registered_handles() -> list:
    """Introspection for tests: the live handle ids."""
    return sorted(_REGISTRY)
def lane_names() -> list:
    """Introspection for tests: the lanes bound so far."""
    return sorted(_LANES)
