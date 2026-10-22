# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""curobo_bridge - the ONLY place cuRobo/torch are called (design section 6). ROS-free module with plain functions. C++ holds opaque integer handles into the
registry below; no torch tensor, no CUDA pointer and no cuRobo object crosses the
boundary."""

from __future__ import annotations

from typing import Dict, Optional

def _resolve_uri(path: str, base_dir: str) -> str:
    """PORT: robot_description._resolve_uri - resolve package:// and relative paths."""
    import os

    if not path:
        return path
    if path.startswith("package://"):
        from ament_index_python.packages import get_package_share_directory

        rest = path[len("package://"):]
        pkg, _, tail = rest.partition("/")
        return os.path.join(get_package_share_directory(pkg), tail)
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))
def _detect_asset_root() -> str:
    """PORT: config_manager.ConfigManager._detect_asset_root - the ROS install
    share/ dir (the parent of this package's share). URDF meshes are referenced
    package://<pkg>/... and cuRobo joins the remainder onto this root."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("curobov2_ros")
        return os.path.dirname(os.path.abspath(share))
    except Exception:
        return ""
def _translate_xrdf(xrdf_path: str, urdf_path: str, asset_root: str) -> str:
    """PORT: config_manager.ConfigManager._translate_xrdf.

    An xrdf config (``format: xrdf``) is the canonical robot definition; cuRobo's
    v2 factories load robot files with a raw ``yaml.load`` and expect
    ``robot_cfg.kinematics``, so the xrdf must go through cuRobo's own translator
    (``load_robot_yaml`` / ``convert_xrdf_to_curobo``), which needs the URDF (for
    the actuated/locked joints and base link) plus a mesh asset root. The
    translated ``robot_cfg`` is written to a resolved copy on disk.
    """
    import hashlib
    import os

    import yaml

    from curobo._src.types.content_path import ContentPath
    from curobo._src.robot.loader.util import load_robot_yaml

    if not urdf_path or not os.path.isfile(urdf_path):
        raise ValueError(
            "xrdf robot config "
            f"'{xrdf_path}' requires a valid 'urdf_path' (pointing at the "
            "generated URDF), because cuRobo's xrdf translator derives the "
            "actuated joints, locked joints and base link from it. Got: "
            f"{urdf_path!r}. Set the 'urdf_path' ROS parameter (and, if the "
            "meshes use package:// URIs, 'asset_root_path').")

    if not asset_root:
        asset_root = _detect_asset_root()
    if not asset_root or not os.path.isdir(asset_root):
        # Non-fatal: cuRobo warns and mesh resolution may fail.
        print(
            f"[curobo_bridge] asset_root_path {asset_root!r} does not exist; "
            "meshes may fail to resolve", flush=True)

    content = ContentPath(
        robot_xrdf_absolute_path=os.path.abspath(xrdf_path),
        robot_urdf_absolute_path=os.path.abspath(urdf_path),
        robot_asset_absolute_path=os.path.abspath(asset_root) if asset_root else None,
    )
    translated = load_robot_yaml(content)
    digest = hashlib.sha1(os.path.abspath(xrdf_path).encode()).hexdigest()[:10]
    resolved_dir = "/tmp/curobov2_ros_resolved"
    os.makedirs(resolved_dir, exist_ok=True)
    resolved_path = os.path.join(resolved_dir, f"xrdf-{digest}.yml")
    with open(resolved_path, "w") as fh:
        yaml.safe_dump(translated, fh, default_flow_style=None, sort_keys=False)
    return resolved_path
def _robot_kinematics_block(data: dict) -> dict:
    """The kinematics mapping of a robot YAML, structure-aware.

    cuRobo's loader accepts three shapes: ``robot_cfg.kinematics.*``, a
    root-level ``kinematics`` block (``kate.curobo.yaml``), or a flat config with
    the kinematics keys at the document root. Reading ``data["lock_joints"]``
    directly therefore found nothing for ``kate.curobo.yaml`` - which silently
    left the unpinned lock joints (``lock_joints: {joint: null}``) unreported, so
    nothing ever drove them from the joint state.
    """
    if not isinstance(data, dict):
        return {}
    root = data.get("kinematics")
    if isinstance(root, dict):
        return root
    robot_cfg = data.get("robot_cfg")
    if isinstance(robot_cfg, dict):
        block = robot_cfg.get("kinematics")
        return block if isinstance(block, dict) else robot_cfg
    return data
def robot_model_build(handle: int, cfg_json: str) -> None:
    """Build the shared Kinematics from the robot YAML (PORT: RobotModelManager.__init__)."""
    import hashlib
    import json
    import os

    import yaml

    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo._src.types.robot import RobotCfg
    from curobo._src.types.device_cfg import DeviceCfg

    cfg = json.loads(cfg_json)
    robot_config_file = cfg["robot_config_file"]
    if not robot_config_file or not os.path.isfile(robot_config_file):
        raise ValueError(
            "robot_model_build: 'robot_config_file' is not set to a readable "
            f"file (got {robot_config_file!r}). Pass -p robot_config_file:=<path> "
            "or set it via the launch params.")

    with open(robot_config_file) as fh:
        data = yaml.safe_load(fh) or {}

    # PORT: config_manager._load_robot_config / _translate_xrdf - an xrdf config
    # is translated to cuRobo's robot_cfg.kinematics form using the urdf_path /
    # asset_root_path parameters (which are NOT inside the xrdf file).
    if data.get("format") == "xrdf":
        resolved = _translate_xrdf(
            robot_config_file,
            cfg.get("urdf_path", ""),
            cfg.get("asset_root_path", ""))
        if not os.path.isfile(resolved):
            raise ValueError(
                f"xrdf translation of {robot_config_file} produced no resolved "
                f"config at {resolved}")
        kin_cfg = KinematicsCfg.from_robot_yaml_file(resolved)
    else:
        # PORT: robot_description.resolve_curobo_config - resolve the yml's
        # package:// / yml-relative urdf_path + asset_root_path WITHOUT
        # restructuring the document. cuRobo's loader accepts either
        # ``robot_cfg.kinematics.*`` or the kinematics keys at the document ROOT
        # (see curobo's load_robot_yaml fallback), and injecting an empty
        # ``robot_cfg.kinematics`` would hide the root keys from that fallback.
        base = os.path.dirname(os.path.abspath(robot_config_file))
        kinematics = _robot_kinematics_block(data)

        changed = False
        for key in ("urdf_path", "asset_root_path"):
            original = kinematics.get(key)
            if not original:
                continue
            if original.startswith("package://"):
                kinematics[key] = _resolve_uri(original, base)
                changed = True
            elif not os.path.isabs(original):
                candidate = os.path.normpath(os.path.join(base, original))
                if os.path.exists(candidate):
                    kinematics[key] = candidate
                    changed = True

        if changed:
            digest = hashlib.sha1(
                os.path.abspath(robot_config_file).encode()).hexdigest()[:10]
            out_dir = "/tmp/curobo_bridge_resolved"
            os.makedirs(out_dir, exist_ok=True)
            resolved = os.path.join(
                out_dir, f"{os.path.basename(robot_config_file)}-{digest}.yml")
            with open(resolved, "w") as fh:
                yaml.safe_dump(data, fh)
        else:
            # Nothing to resolve: hand cuRobo the ORIGINAL file (its own loader
            # normalizes robot_cfg/kinematics).
            resolved = robot_config_file

        kin_cfg = KinematicsCfg.from_robot_yaml_file(resolved)

    # PORT: RobotModelManager.set_torque_mode / _wrap_robot_cfg - the pinocchio
    # inverse-dynamics model is attached ONLY when the node runs torque-limited
    # (load_dynamics:=true; the launch default is false). This must be gated:
    # with a dynamics model present the rollout computes joint_torque (RNEA
    # forward + backward) inside the cspace cost on EVERY optimizer iteration
    # (see rollout/cost_manager/cost_manager_robot.py, `joint_torque=`), which
    # roughly doubles the cost of pure-kinematics modes such as the standalone
    # IK solver used by the reachability client. The pre-port Python
    # RobotModelManager built `dynamics=None` unless load_dynamics was set.
    load_dynamics = bool(cfg.get("load_dynamics", False))
    dynamics = None
    if load_dynamics:
        dynamics = RobotCfg._create_dynamics_config(
            kinematics_config=kin_cfg.kinematics_config, device_cfg=DeviceCfg())
    robot_cfg = RobotCfg(kinematics=kin_cfg, dynamics=dynamics)
    kin = Kinematics(robot_cfg.kinematics)

    entry = _REGISTRY[handle]
    entry["kin_cfg"] = kin_cfg
    entry["robot_cfg"] = robot_cfg
    entry["kin"] = kin
    entry["joint_names"] = list(kin.joint_names)
    entry["dof"] = int(kin.get_dof())
    # Robot-masking knobs (PORT: RobotSegmentation.__init__, which reads
    # robot_segmentation_distance_threshold / robot_segmentation_mask_margin).
    # robot_segment() looks both up on THIS registry entry; without them it
    # silently fell back to its own 0.05 / 0.0 defaults, so the launch
    # parameters were accepted but had no effect.
    if "seg_distance_threshold" in cfg:
        entry["seg_distance_threshold"] = float(cfg["seg_distance_threshold"])
    if "seg_mask_margin" in cfg:
        entry["seg_mask_margin"] = float(cfg["seg_mask_margin"])
    # The robot YAML's own lock_joints (PORT: RobotModelManager.config_lock_joints).
    entry["resolved_config"] = resolved
    # PORT: the lock_joints map, structure-aware (robot_cfg.kinematics.* or the
    # kinematics keys at the document root).
    _root_kin = _robot_kinematics_block(data)
    _lock_joints = _root_kin.get("lock_joints") or {}
    # A `null` value marks an *unpinned* lock: the joint stays out of the planning
    # cspace but is not pinned, its value coming from the joint state (see
    # robot_model_update_lock_joints). Keep it out of the pinned map so nothing
    # here has to float(None) it.
    entry["lock_joints"] = {
        str(k): float(v) for k, v in _lock_joints.items() if v is not None
    }
    entry["unpinned_lock_joints"] = [
        str(k) for k, v in _lock_joints.items() if v is None
    ]
    # The URDF link the payload is fixed to, used to publish the attached
    # spheres LINK-RELATIVE so a trajectory preview can ride the gripper.
    _extra_links = _root_kin.get("extra_links") or {}
    _attach = _extra_links.get(ATTACH_LINK) or {}
    entry["attach_parent_link"] = (
        _attach.get("parent_link_name") if isinstance(_attach, dict) else None)
    entry["built"] = True

from .runtime import _REGISTRY
from .scene import ATTACH_LINK
