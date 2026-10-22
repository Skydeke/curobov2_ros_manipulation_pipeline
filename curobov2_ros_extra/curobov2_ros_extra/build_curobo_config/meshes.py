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

def resolve_package_meshes(urdf_path: str, asset_path: str) -> str:
    """Rewrite mesh URIs in the URDF to absolute filesystem paths.

    cuRobo's RobotBuilder resolves relative mesh paths against asset_path but
    cannot resolve ROS 2 'package://' URIs, and URDFs exported from a different
    machine often carry stale absolute paths. This rewrites every mesh
    reference (package://, file://, or bare relative path) to a path that
    actually exists, resolving packages via ament_index_python.
    """
    with open(urdf_path, "r") as f:
        text = f.read()

    meshes: Dict[str, str] = {}

    def resolve_mesh(uri: str) -> str:
        """Return a path that exists for the given mesh URI."""
        pkg = rel = None

        m = re.match(r"^package://([^/\s]+)/(\S+)", uri)
        if m:
            pkg, rel = m.group(1), m.group(2)
            # Prefer the plain asset tree (e.g. /tmp/assets/<pkg>/...) so this
            # also works without a sourced ROS workspace (bubblify preview).
            if asset_path:
                candidate = os.path.abspath(os.path.join(asset_path, pkg, rel))
                if os.path.exists(candidate):
                    return candidate
        else:
            p = re.sub(r"^file://", "", uri)
            # Detect the package from the path so that relative paths, bare
            # package-relative paths, and stale absolute paths (from another
            # machine / workspace) all get remapped into this workspace's
            # install tree. Handle both '<pkg>/...' and 'share/<pkg>/...'.
            for known in ("kortex_description", "robotiq_description"):
                idx = p.find("share/" + known + "/")
                if idx != -1:
                    pkg = known
                    rel = p[idx + len("share/" + known) + 1:]
                    break
            if pkg is None:
                for known in ("kortex_description", "robotiq_description"):
                    idx = p.find(known + "/")
                    if idx != -1:
                        pkg = known
                        rel = p[idx + len(known) + 1:]
                        break
            if pkg is None:
                # Package can't be detected; if it's an existing absolute path
                # keep it, otherwise leave as-is so curobo tries asset_path.
                if os.path.isabs(p) and os.path.exists(p):
                    return p
                return uri

        try:
            from ament_index_python.packages import get_package_share_directory
        except ImportError:
            print(
                "ERROR: ament_index_python not available; cannot resolve mesh paths.",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            share_dir = get_package_share_directory(pkg)
        except Exception:
            return uri
        candidate = os.path.abspath(os.path.join(share_dir, rel))
        if os.path.exists(candidate):
            return candidate
        return uri

    def replace(match: Any, group: Any) -> str:
        """Resolve a single regex match and remember the mapping."""
        uri = match.group(group)
        resolved = resolve_mesh(uri)
        meshes[uri] = resolved
        return resolved

    # package:// URIs
    text = re.sub(
        r'package://[^\s"\'<>]+',
        lambda m: replace(m, 0),
        text,
    )
    # file:// URIs and bare absolute/relative paths in filename="..." attributes
    text = re.sub(
        r'(filename\s*=\s*["\'])([^"\']+)(["\'])',
        lambda m: m.group(1) + replace(m, 2) + m.group(3),
        text,
    )

    if meshes:
        found = sum(1 for v in meshes.values() if os.path.isabs(v))
        print(f"Resolved {found}/{len(meshes)} mesh paths to existing files.")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", delete=False, prefix="curobo_resolved_"
    )
    tmp.write(text)
    tmp.close()
    return tmp.name
