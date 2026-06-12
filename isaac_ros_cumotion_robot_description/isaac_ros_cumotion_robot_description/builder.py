#!/usr/bin/env python3
import argparse
import os
import re
import sys
import tempfile

from ament_index_python.packages import get_package_share_directory
from curobo.robot_builder import RobotBuilder


def _resolve_package_uris_in_urdf(urdf_path: str) -> str:
    """Return a temp URDF copy with ``package://<pkg>/...`` resolved to absolute paths.

    Each ``package://`` URI is turned into an absolute filesystem path using
    ``ament_index_python`` so that cuRobo (or any other consumer) can load the
    meshes without relying on symlinks under a single ``asset_root``.
    """
    with open(urdf_path) as f:
        content = f.read()

    def _replace(match):
        pkg = match.group(1)
        rel = match.group(2)
        try:
            share = get_package_share_directory(pkg)
            return os.path.join(share, rel)
        except Exception:
            print(f"Warning: package '{pkg}' not found via ament_index, leaving URI as-is",
                  file=sys.stderr)
            return match.group(0)

    resolved = re.sub(r'package://([^/]+)/(.+)', _replace, content)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    tmp.write(resolved)
    tmp.close()
    return tmp.name


def build_new_robot(args):
    print(f"Building robot model from URDF: {args.urdf}")

    resolved_urdf = _resolve_package_uris_in_urdf(args.urdf)
    print(f"Asset path: {args.asset_path or '(meshes resolved via package://)'}")

    # Create builder
    builder = RobotBuilder(
        urdf_path=resolved_urdf,
        asset_path='',
        tool_frames=args.tool_frames,
    )

    print(f"Found {len(builder.tool_frames)} links in robot")

    clip_links = None
    if args.clip_link:
        clip_links = {link: (axis, float(offset)) for link, axis, offset in args.clip_link}

    # Fit collision spheres
    print("\nFitting collision spheres...")
    builder.fit_collision_spheres(
        sphere_density=args.sphere_density,
        coverage_weight=args.coverage_weight,
        protrusion_weight=args.protrusion_weight,
        compute_metrics=args.compute_metrics,
        clip_links=clip_links,
    )

    print(f"Fitted {builder.num_spheres} spheres across {len(builder.collision_link_names)} links")

    if args.compute_metrics and builder.link_metrics:
        header = (
            f"  {'link':<25s} {'n_sph':>5s} "
            f"{'cover%':>7s} {'protr%':>7s} {'prot_mm':>8s} "
            f"{'gap_mm':>7s} {'vol_ratio':>9s}"
        )
        print(f"\n{header}")
        print(f"  {'-' * (len(header) - 2)}")
        for link_name, m in builder.link_metrics.items():
            print(
                f"  {link_name:<25s} {m.num_spheres:5d} "
                f"{m.coverage * 100:6.1f}% {m.protrusion * 100:6.1f}% "
                f"{m.protrusion_dist_mean * 1000:7.2f}mm "
                f"{m.surface_gap_mean * 1000:6.2f}mm "
                f"{m.volume_ratio:9.3f}"
            )

    # Compute collision matrix
    print("\nComputing collision matrix...")
    builder.compute_collision_matrix(
        prune_collisions=not args.no_prune,
        num_samples=args.num_collision_samples,
    )

    print(f"Created collision ignore matrix with {len(builder.collision_matrix)} entries")

    # Build configuration
    print("\nBuilding configuration...")
    config = builder.build()

    # Save
    print(f"Saving to: {args.output}")

    if args.output.endswith('.xrdf'):
        builder.save_xrdf(config, args.output)
    else:
        builder.save(config, args.output)

    # Also save in alternate format if requested
    if args.export_xrdf and not args.output.endswith('.xrdf'):
        xrdf_path = args.output.replace('.yml', '.xrdf').replace('.yaml', '.xrdf')
        print(f"Also exporting to XRDF: {xrdf_path}")
        builder.save_xrdf(config, xrdf_path)

    print("\nRobot model created successfully!")

    # Visualize if requested
    if args.visualize:
        print(f"\nStarting visualization server at http://localhost:{args.viz_port}")
        print("Press Ctrl+C to stop")
        viser = builder.visualize(config, port=args.viz_port)
        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping visualization")

    # Clean up temp file
    os.unlink(resolved_urdf)


def edit_existing_robot(args):
    print(f"Loading robot configuration: {args.edit_config}")

    builder = RobotBuilder.from_config(args.edit_config)

    print(f"Loaded robot with {builder.num_spheres} spheres")

    if args.refit_link:
        print(f"\nRefitting spheres for link: {args.refit_link}")
        new_spheres = builder.refit_link_spheres(
            args.refit_link,
            sphere_density=args.sphere_density,
        )
        print(f"Fitted {len(new_spheres)} spheres to {args.refit_link}")

    if args.add_collision_ignore:
        link_name, ignore_links = args.add_collision_ignore
        ignore_list = ignore_links.split(",")
        print(f"\nAdding collision ignores: {link_name} -> {ignore_list}")
        builder.add_collision_ignore(link_name, ignore_list)

    if args.recompute_collisions:
        print("\nRecomputing collision matrix...")
        builder.compute_collision_matrix(
            prune_collisions=not args.no_prune,
            num_samples=args.num_collision_samples,
        )

    print("\nBuilding updated configuration...")
    config = builder.build()

    print(f"Saving to: {args.output}")

    if args.output.endswith('.xrdf'):
        builder.save_xrdf(config, args.output)
    else:
        builder.save(config, args.output)

    if args.export_xrdf and not args.output.endswith('.xrdf'):
        xrdf_path = args.output.replace('.yml', '.xrdf').replace('.yaml', '.xrdf')
        print(f"Also exporting to XRDF: {xrdf_path}")
        builder.save_xrdf(config, xrdf_path)

    print("\nRobot model updated successfully!")

    if args.visualize:
        print(f"\nStarting visualization server at http://localhost:{args.viz_port}")
        viser = builder.visualize(config, port=args.viz_port)
        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping visualization")


def main():
    parser = argparse.ArgumentParser(
        description="ROS-aware cuRobo robot configuration builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--urdf", type=str, help="Path to URDF file (for creating new robot model)")
    mode.add_argument("--edit-config", type=str, help="Path to existing .yml config (for editing)")

    parser.add_argument("--output", type=str, default=None, help="Output file path (.yml or .xrdf)")
    parser.add_argument("--export-xrdf", action="store_true", help="Also export to XRDF format")

    parser.add_argument("--asset-path", type=str, default="",
                        help="Path to mesh assets (not needed when URDF uses package:// URIs)")
    parser.add_argument("--tool-frames", nargs="+", type=str, default=[],
                        help="Tool frames (optional)")

    parser.add_argument("--sphere-density", type=float, default=1.0,
                        help="Sphere density multiplier")
    parser.add_argument("--coverage-weight", type=float, default=None,
                        help="MorphIt coverage loss weight")
    parser.add_argument("--protrusion-weight", type=float, default=None,
                        help="MorphIt protrusion loss weight")
    parser.add_argument("--compute-metrics", action="store_true",
                        help="Print per-link sphere fit quality metrics")
    parser.add_argument("--clip-link", nargs=3, action="append", metavar=("LINK", "AXIS", "OFFSET"),
                        help="Clip spheres on LINK so they don't extend past a plane")

    parser.add_argument("--num-collision-samples", type=int, default=1000,
                        help="Number of samples for collision pruning")
    parser.add_argument("--no-prune", action="store_true",
                        help="Skip collision pruning")

    parser.add_argument("--refit-link", type=str, help="Refit spheres for specific link")
    parser.add_argument("--add-collision-ignore", nargs=2, metavar=("LINK", "IGNORE_LINKS"),
                        help="Add collision ignore: LINK IGNORE_LINKS (comma-separated)")
    parser.add_argument("--recompute-collisions", action="store_true",
                        help="Recompute entire collision matrix")

    parser.add_argument("--visualize", action="store_true",
                        help="Start Viser visualization server after building")
    parser.add_argument("--viz-port", type=int, default=8080,
                        help="Visualization server port")

    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    if args.output is None:
        parser.error("--output is required when using --urdf or --edit-config")

    if args.seed is not None:
        import numpy as np
        import torch
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    try:
        if args.urdf:
            build_new_robot(args)
        else:
            edit_existing_robot(args)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
