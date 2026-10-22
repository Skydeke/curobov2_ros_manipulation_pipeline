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

from .meshes import resolve_package_meshes
from .bubblify import launch_bubblify, load_bubblify_spheres

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Create or edit a cuRobo v2 YAML robot config, "
        "optionally with the bubblify sphere editor."
    )
    # Source: exactly one of --urdf (create) or --config (edit) is required.
    parser.add_argument(
        "--urdf",
        default=None,
        help="Path to the robot URDF file (create mode).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to an existing cuRobo yml config (edit mode, "
        "loaded via RobotBuilder.from_config).",
    )
    parser.add_argument(
        "--asset-path",
        default=None,
        help="Parent directory containing mesh dirs "
        "(e.g., the parent of both 'kortex_description/' and 'robotiq_description/'). "
        "Required in create mode.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output YAML path (default: 'curobo_config.yml' in create mode, "
        "the input --config path in edit mode).",
    )
    parser.add_argument(
        "--tool-frame", default="grasping_frame", help="Tool frame name (create mode)."
    )
    parser.add_argument(
        "--sphere-density", type=float, default=1.0, help="Sphere density multiplier."
    )
    parser.add_argument(
        "--fit-type",
        choices=["morphit", "voxel", "surface"],
        default="morphit",
        help="Sphere fitting algorithm (default: morphit, the cuRobo builder path).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Optimization iterations for morphit fits.",
    )
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=None,
        help="MorphIt coverage loss weight (builder default when omitted).",
    )
    parser.add_argument(
        "--protrusion-weight",
        type=float,
        default=None,
        help="MorphIt protrusion loss weight (builder default when omitted).",
    )
    parser.add_argument(
        "--num-collision-samples",
        type=int,
        default=1000,
        help="Collision matrix samples",
    )
    parser.add_argument(
        "--matrix-batch-size",
        type=int,
        default=2000,
        help="Collision matrix batch size (lower on CUDA OOM).",
    )
    # Edit-mode sphere/matrix control.
    parser.add_argument(
        "--refit-all",
        action="store_true",
        help="Edit mode: refit spheres for all links with --sphere-density.",
    )
    parser.add_argument(
        "--refit-link",
        action="append",
        default=[],
        metavar="LINK",
        help="Edit mode: refit spheres for LINK with --sphere-density (repeatable).",
    )
    parser.add_argument(
        "--budget",
        action="append",
        default=[],
        metavar="LINK=N",
        help="Edit mode: refit LINK with an exact sphere budget of N spheres "
        "(repeatable, e.g. --budget j2n6s200_link_3=8).",
    )
    parser.add_argument(
        "--lock-joint",
        action="append",
        default=[],
        metavar="JOINT",
        help="Keep JOINT out of the planning cspace and lock it unpinned "
        "(value null, follows the live joint state). Repeatable; the saved "
        "config still reflects this on a later edit run.",
    )
    parser.add_argument(
        "--recompute-matrix",
        action="store_true",
        help="Edit mode: recompute the self-collision matrix. "
        "Implied by any refit/import unless --skip-matrix is given.",
    )
    parser.add_argument(
        "--skip-matrix",
        action="store_true",
        help="Edit mode: do not recompute the self-collision matrix, "
        "even if spheres changed.",
    )
    parser.add_argument(
        "--import-bubblify",
        default=None,
        metavar="SPHERES_YML",
        help="Edit mode: merge a bubblify-exported spheres yml into the "
        "config before building.",
    )
    # Interactive viewers.
    parser.add_argument(
        "--with-bubblify",
        action="store_true",
        help="Launch the interactive bubblify sphere editor preloaded with "
        "the current spheres (needs 'pip install bubblify').",
    )
    parser.add_argument(
        "--bubblify-port", type=int, default=8080, help="Bubblify viser port."
    )
    parser.add_argument(
        "--bubblify-show-collision",
        action="store_true",
        help="Show collision meshes in the bubblify viewer.",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Show sphere fit in Viser browser."
    )
    parser.add_argument(
        "--viser-port", type=int, default=8080, help="cuRobo Viser viewer port."
    )
    return parser
def _resolve_fit_type(name: str) -> Any:
    """Map a CLI fit-type name to :class:`SphereFitType`.

    Falls back to the plain string when cuRobo is unavailable (test stubs).
    """
    try:
        from curobo._src.geom.sphere_fit.types import SphereFitType
    except ImportError:
        return name
    return SphereFitType(name)
def _parse_budget(spec: str) -> Tuple[str, int]:
    """Parse a LINK=N budget spec into (link, sphere count)."""
    link, sep, count = spec.partition("=")
    if not sep or not link or not count.isdigit() or int(count) < 1:
        print(f"ERROR: invalid --budget {spec!r}, expected LINK=N with N >= 1.",
              file=sys.stderr)
        sys.exit(2)
    return link, int(count)
def _fit_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect the shared sphere-fit kwargs from CLI args."""
    return {
        "sphere_density": args.sphere_density,
        "fit_type": _resolve_fit_type(args.fit_type),
        "iterations": args.iterations,
        "coverage_weight": args.coverage_weight,
        "protrusion_weight": args.protrusion_weight,
        "compute_metrics": True,
    }
def recompute_matrix_safe(builder: Any, num_samples: int,
                          batch_size: int) -> bool:
    """Recompute the collision matrix, surviving CUDA OOM.

    Returns True on success. On CUDA OOM the spheres are untouched and False
    is returned so the caller can still save.
    """
    try:
        builder.compute_collision_matrix(
            prune_collisions=True,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        return True
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
        print("ERROR: CUDA OOM during collision matrix recompute; spheres "
              "are unchanged and will still be saved. Retry with a smaller "
              "--matrix-batch-size/--num-collision-samples or --skip-matrix.",
              file=sys.stderr)
        return False
def _make_save_fn(builder: Any, output: str,
                  original: Optional[Dict[str, Any]],
                  args: argparse.Namespace):
    """Create the save callback used by the bubblify Save button and exit."""
    def save_now(dest: Optional[str] = None) -> str:
        """Recompute the matrix (unless skipped) and save one file."""
        dest = (dest or "").strip() or output
        if args.skip_matrix:
            note = " (matrix NOT recomputed: --skip-matrix)"
        elif recompute_matrix_safe(builder, args.num_collision_samples,
                                    args.matrix_batch_size):
            note = " (matrix recomputed)"
        else:
            note = " (matrix recompute OOMed, kept previous matrix)"
        build_and_save(builder, dest, original,
                       getattr(args, "lock_joint", []) or [])
        return f"Saved {dest}{note}"

    return save_now
def run_create_mode(args: argparse.Namespace, RobotBuilder: Any) -> Tuple[Any, str, str, None]:
    """Build a new config from a URDF, return (builder, urdf, output, None)."""
    if not args.asset_path:
        print("ERROR: --asset-path is required in create mode (--urdf).",
              file=sys.stderr)
        sys.exit(2)

    print(f"Loading URDF: {args.urdf}")
    resolved_urdf = resolve_package_meshes(args.urdf, args.asset_path)
    print(f"Using resolved URDF: {resolved_urdf}")

    builder = RobotBuilder(
        urdf_path=resolved_urdf,
        asset_path=args.asset_path,
        tool_frames=[args.tool_frame],
    )

    print("Fitting collision spheres ...")
    builder.fit_collision_spheres(**_fit_kwargs(args))

    print("Computing self-collision matrix ...")
    recompute_matrix_safe(builder, args.num_collision_samples,
                          args.matrix_batch_size)

    return builder, resolved_urdf, args.output or "curobo_config.yml", None
def run_edit_mode(
    args: argparse.Namespace, RobotBuilder: Any
) -> Tuple[Any, str, str, Dict[str, Any]]:
    """Load an existing config, return (builder, urdf, output, original)."""
    print(f"Loading existing config: {args.config}")
    builder = RobotBuilder.from_config(args.config)
    original = load_original_kinematics(args.config)

    spheres_changed = False

    if args.import_bubblify:
        print(f"Merging bubblify spheres: {args.import_bubblify}")
        # No public setter on RobotBuilder; the CLI edits the loaded spheres
        # directly before build().
        builder._collision_spheres = load_bubblify_spheres(args.import_bubblify)
        spheres_changed = True

    if args.refit_all:
        print("Refitting spheres for all links ...")
        builder.fit_collision_spheres(**_fit_kwargs(args))
        spheres_changed = True
    else:
        for link in args.refit_link:
            print(f"Refitting spheres for link: {link}")
            try:
                builder.refit_link_spheres(link, **_fit_kwargs(args))
            except ValueError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
            spheres_changed = True
        for spec in args.budget:
            link, count = _parse_budget(spec)
            print(f"Refitting link {link} with a budget of {count} spheres ...")
            try:
                builder.refit_link_spheres(link, num_spheres=count,
                                           **_fit_kwargs(args))
            except ValueError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
            spheres_changed = True

    if args.recompute_matrix or (spheres_changed and not args.skip_matrix):
        print("Computing self-collision matrix ...")
        recompute_matrix_safe(builder, args.num_collision_samples,
                              args.matrix_batch_size)
    elif spheres_changed:
        print("WARNING: spheres changed but the collision matrix was not "
              "recomputed (pass --recompute-matrix).")

    # URDF hint for the bubblify preview: resolve package:// against the
    # config's asset path so the viewer finds the meshes without ROS.
    urdf_hint = getattr(builder, "urdf_path", "") or ""
    asset_hint = getattr(builder, "asset_path", "") or args.asset_path or ""
    if urdf_hint and os.path.exists(urdf_hint):
        try:
            urdf_hint = resolve_package_meshes(urdf_hint, asset_hint)
        except SystemExit:
            pass
    return builder, urdf_hint, args.output or args.config, original
def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the build_curobo_config CLI."""
    args = build_parser().parse_args(argv)

    if bool(args.urdf) == bool(args.config):
        print("ERROR: pass exactly one of --urdf (create) or --config (edit).",
              file=sys.stderr)
        sys.exit(2)

    try:
        from curobo.robot_builder import RobotBuilder
    except ImportError:
        print(
            "ERROR: curobo (nvidia-curobo) is not installed. "
            "Install with: pip install nvidia-curobo",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.urdf:
        builder, urdf_for_bubblify, output, original = run_create_mode(
            args, RobotBuilder)
    else:
        builder, urdf_for_bubblify, output, original = run_edit_mode(
            args, RobotBuilder)

    save_fn = _make_save_fn(builder, output, original, args)

    if args.with_bubblify:
        if not urdf_for_bubblify or not os.path.exists(urdf_for_bubblify):
            print("ERROR: URDF for bubblify not found "
                  f"({urdf_for_bubblify!r}); mount it into the container.",
                  file=sys.stderr)
            sys.exit(1)
        print("NOTE: tune in the browser, then use the Save button "
              f"(writes {output}) or Ctrl+C; nothing else is written.")
        spheres = builder.collision_spheres or {}
        launch_bubblify(
            builder, urdf_for_bubblify, spheres, args.bubblify_port,
            args.bubblify_show_collision, _fit_kwargs(args), save_fn, output,
        )
        print("Building config ...")
        config = builder.build()
    else:
        print("Building config ...")
        config = build_and_save(builder, output, original,
                                getattr(args, "lock_joint", []) or [])
    print(f"Done! Wrote {output} (custom fields preserved).")

    if args.visualize:
        print("Starting Viser visualization...")
        builder.visualize(config, port=args.viser_port)
try:
    from curobov2_ros_extra.curobo_config_io import (
        build_and_save,
        load_original_kinematics,
    )
except ImportError:  # script executed directly, not as installed package
    from curobo_config_io import (
        build_and_save,
        load_original_kinematics,
    )
if __name__ == "__main__":
    main()
