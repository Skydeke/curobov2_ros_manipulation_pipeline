#!/usr/bin/env python3
"""Create or edit a cuRobo v2 robot config, optionally with bubblify.

Create a new config from a URDF::

    ros2 run curobov2_ros_extra build_curobo_config \
        --urdf /tmp/kortex.urdf \
        --asset-path ~/ros2_ws \
        --output /tmp/kortex_curobo.yml \
        --tool-frame grasping_frame

Edit an existing config (loads spheres + collision matrix, like a
``RobotBuilder.from_config`` session)::

    ros2 run curobov2_ros_extra build_curobo_config \
        --config /tmp/adir.curobo.yml \
        --output /tmp/adir.curobo.yml \
        --with-bubblify

The tool uses cuRobo's :class:`RobotBuilder` to fit collision spheres and
compute the self-collision ignore matrix. With ``--with-bubblify`` it then
launches the interactive bubblify sphere editor
(https://github.com/bheijden/bubblify) preloaded with the current spheres.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

def collision_spheres_to_bubblify_data(
    collision_spheres: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Convert cuRobo collision spheres to a bubblify spherization dict."""
    links = sorted(collision_spheres.keys())
    total = sum(len(v) for v in collision_spheres.values())
    return {
        "collision_spheres": collision_spheres,
        "metadata": {"total_spheres": total, "links": links},
    }
def write_bubblify_spherization(
    collision_spheres: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Write spheres to a temp bubblify spherization yml, return its path."""
    import yaml

    data = collision_spheres_to_bubblify_data(collision_spheres)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, prefix="bubblify_spheres_"
    )
    tmp.write(yaml.safe_dump(data, default_flow_style=False, sort_keys=False))
    tmp.close()
    return tmp.name
def load_bubblify_spheres(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load a bubblify-exported spheres yml as cuRobo-style sphere dicts."""
    import yaml

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    raw = data.get("collision_spheres", {})
    spheres: Dict[str, List[Dict[str, Any]]] = {}
    for link, entries in raw.items():
        spheres[link] = [
            {
                "center": [float(c) for c in e["center"]],
                "radius": float(e["radius"]),
            }
            for e in entries
        ]
    return spheres
def launch_bubblify(
    builder: Any,
    urdf_path: str,
    collision_spheres: Dict[str, List[Dict[str, Any]]],
    port: int,
    show_collision: bool,
    fit_defaults: Dict[str, Any],
    save_fn: Any,
    output: str,
) -> bool:
    """Launch interactive bubblify with MorphIt refits and a Save button.

    Preloads the current spheres; morphit refits and manual edits made in
    the browser are pulled back into the builder, and saving writes only
    the cuRobo yml.

    Returns:
        True when the builder spheres changed during the session.
    """
    try:
        from curobov2_ros_extra.morphit_bubblify import launch_morphit_bubblify
    except ImportError:  # script executed directly, not as an installed package
        from morphit_bubblify import launch_morphit_bubblify
    spherization_yml = write_bubblify_spherization(collision_spheres)
    print(f"Preloading {sum(len(v) for v in collision_spheres.values())} "
          f"spheres into bubblify: {spherization_yml}")
    print("Refit links with MorphIt (budget or density) in the browser; "
          "manual edits work too. Use the Save button or Ctrl+C; "
          f"only {output} is written.")
    return launch_morphit_bubblify(
        builder, urdf_path, spherization_yml, port, show_collision,
        fit_defaults, save_fn, output,
    )
