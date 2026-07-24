# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os as _os
import re as _re
import tempfile as _tempfile
from typing import Dict, List, Optional

from ament_index_python.packages import get_package_share_directory
from curobo.types import ContentPath
from curobo.types import DeviceCfg
from curobo.content import get_robot_configs_path
from curobo.config_io import join_path, load_yaml
from curobo.kinematics import KinematicsCfg
from rclpy.impl.rcutils_logger import RcutilsLogger


def _resolve_package_paths_in_urdf(urdf_path: str, logger: RcutilsLogger) -> str:
    """Return a temp URDF copy with ``package://<pkg>/...`` resolved to absolute paths.

    Uses ``ament_index_python`` to look up each package's share directory so that
    cuRobo can load meshes without requiring symlinks under a single ``asset_root``.
    """
    with open(urdf_path) as f:
        content = f.read()

    def _replace(match):
        pkg = match.group(1)
        rel = match.group(2)
        try:
            share = get_package_share_directory(pkg)
            return _os.path.join(share, rel)
        except Exception:
            logger.warn(
                f"Package '{pkg}' not found via ament_index, "
                "leaving package:// URI as-is"
            )
            return match.group(0)

    resolved = _re.sub(r'package://([^/]+)/(.+)', _replace, content)

    tmp = _tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    tmp.write(resolved)
    tmp.close()
    return tmp.name


def get_robot_config(robot_file: str,
                     urdf_file_path: str,
                     logger: RcutilsLogger,
                     object_link_name: Optional[str] = None) -> Dict:

    if robot_file.lower().endswith('.yml'):
        logger.warn(
            'YML files will be deprecated soon. Switch to XRDF files for future compatibility.')

        robot_file_path = join_path(get_robot_configs_path(), robot_file)
        robot_config = load_yaml(robot_file_path)
        if urdf_file_path is not None:
            robot_config['robot_cfg']['kinematics']['urdf_path'] = urdf_file_path

    elif robot_file.lower().endswith('.xrdf'):

        if urdf_file_path is None:
            logger.fatal('urdf_path is required to load robot from XRDF file')
            raise SystemExit

        # Resolve package:// URIs in the URDF so meshes are found without
        # symlinks under a curobo_assets/ directory.
        resolved_urdf = _resolve_package_paths_in_urdf(urdf_file_path, logger)

        xrdf_dir_path = _os.path.join(
            get_package_share_directory('isaac_ros_cumotion_robot_description'), 'xrdf')
        xrdf_file_path = join_path(xrdf_dir_path, robot_file)
        content_path = ContentPath(robot_xrdf_absolute_path=xrdf_file_path,
                                   robot_urdf_absolute_path=resolved_urdf)

        kcfg = KinematicsCfg.from_content_path(
            content_path,
            device_cfg=DeviceCfg(),
            extra_collision_spheres={object_link_name: 100}
            if object_link_name else None,
            tool_frames=[object_link_name] if object_link_name else None,
        )

        # Pass KinematicsCfg object directly so RobotCfg.create() skips re-parse
        robot_config = {"robot_cfg": {"kinematics": kcfg}}

    else:
        logger.fatal('Invalid robot file; only XRDF or YML files accepted. Halting.')
        raise SystemExit

    return robot_config


def update_collision_sphere_buffer(
    robot_yaml: Dict,
    link_name: str,
    num_spheres: int,
) -> Dict:

    updt_sphere_dict = robot_yaml['robot_cfg']['kinematics'].get(
        'extra_collision_spheres'
    )

    if updt_sphere_dict is None:
        updt_sphere_dict = {link_name: num_spheres}
    else:
        updt_sphere_dict[link_name] = num_spheres

    robot_yaml['robot_cfg']['kinematics']['extra_collision_spheres'] = updt_sphere_dict
    return robot_yaml


def append_tool_frames(
    robot_yaml: Dict,
    tool_frames: List[str]
) -> Dict:

    robot_dict = robot_yaml['robot_cfg']['kinematics']
    link_names = robot_dict.get('link_names', [])
    link_names.extend(tool_frames)
    link_names = list(set(link_names))  # Ensure link names are unique
    robot_dict['link_names'] = link_names
    robot_yaml = {'robot_cfg': {'kinematics': robot_dict}}
    return robot_yaml



