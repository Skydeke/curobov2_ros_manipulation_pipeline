#!/usr/bin/env python3
"""
ROS 2 node / CLI tool to generate a cuRobo v2 YAML config from a URDF.

Usage (standalone):
    ros2 run isaac_ros_cumotion_extra build_curobo_config \
        --urdf /tmp/kortex.urdf \
        --asset-path /path/to/mesh/parent \
        --output /tmp/kortex.curobo.yml

The tool uses cuRobo's RobotBuilder to:
  1. Load the URDF and extract the kinematic tree
  2. Fit collision spheres to each link mesh
  3. Compute the self-collision ignore matrix
  4. Export a v2-format YAML config for use with isaac_ros_cumotion
"""

import argparse
import os
import re
import sys
import tempfile


def resolve_package_meshes(urdf_path, asset_path):
    """Rewrite mesh URIs in the URDF to absolute filesystem paths.

    cuRobo's RobotBuilder resolves relative mesh paths against asset_path but
    cannot resolve ROS 2 'package://' URIs, and URDFs exported from a different
    machine often carry stale absolute paths. This rewrites every mesh
    reference (package://, file://, or bare relative path) to a path that
    actually exists, resolving packages via ament_index_python.
    """
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        print(
            "ERROR: ament_index_python not available; cannot resolve mesh paths.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(urdf_path, "r") as f:
        text = f.read()

    meshes = {}

    def resolve_mesh(uri):
        """Return a path that exists for the given mesh URI."""
        pkg = rel = None

        m = re.match(r"^package://([^/\s]+)/(\S+)", uri)
        if m:
            pkg, rel = m.group(1), m.group(2)
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
            share_dir = get_package_share_directory(pkg)
        except Exception:
            return uri
        candidate = os.path.abspath(os.path.join(share_dir, rel))
        if os.path.exists(candidate):
            return candidate
        return uri

    def replace(match, group):
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


def main():
    parser = argparse.ArgumentParser(
        description="Build cuRobo v2 YAML config from a URDF"
    )
    parser.add_argument("--urdf", required=True, help="Path to the robot URDF file")
    parser.add_argument(
        "--asset-path",
        required=True,
        help="Parent directory containing mesh dirs "
        "(e.g., the parent of both 'kortex_description/' and 'robotiq_description/')",
    )
    parser.add_argument(
        "--output", default="curobo_config.yml", help="Output YAML path"
    )
    parser.add_argument(
        "--tool-frame", default="grasping_frame", help="Tool frame name"
    )
    parser.add_argument(
        "--base-link", default=None, help="Base link (auto-detected if omitted)"
    )
    parser.add_argument(
        "--sphere-density", type=float, default=1.0, help="Sphere density multiplier"
    )
    parser.add_argument(
        "--num-collision-samples",
        type=int,
        default=1000,
        help="Collision matrix samples",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Show sphere fit in Viser browser"
    )
    args = parser.parse_args()

    try:
        from curobo.robot_builder import RobotBuilder
    except ImportError:
        print(
            "ERROR: curobo (nvidia-curobo) is not installed. "
            "Install with: pip install nvidia-curobo",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading URDF: {args.urdf}")
    resolved_urdf = resolve_package_meshes(args.urdf, args.asset_path)
    print(f"Using resolved URDF: {resolved_urdf}")

    kwargs = dict(
        urdf_path=resolved_urdf,
        asset_path=args.asset_path,
        tool_frames=[args.tool_frame],
    )
    if args.base_link:
        kwargs["base_link"] = args.base_link

    builder = RobotBuilder(**kwargs)

    print("Fitting collision spheres ...")
    builder.fit_collision_spheres(
        sphere_density=args.sphere_density,
        compute_metrics=True,
    )

    print("Computing self-collision matrix ...")
    builder.compute_collision_matrix(
        prune_collisions=True,
        num_samples=args.num_collision_samples,
    )

    print("Building config ...")
    config = builder.build()

    print(f"Saving to {args.output}")
    builder.save(config, args.output)
    print("Done!")

    if args.visualize:
        print("Starting Viser visualization...")
        builder.visualize(config)


if __name__ == "__main__":
    main()
