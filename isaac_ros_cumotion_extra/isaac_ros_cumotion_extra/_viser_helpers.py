# Shared helpers for the per-example viser nodes.
#
# This module lives in isaac_ros_cumotion_extra and provides:
#   - _CuPoseCompat: lightweight Pose stand-in (no curobo import)
#   - _resolve_package_paths_in_urdf: resolve package:// URIs
#   - _build_robot_data_dict: build robot YAML-compatible dict from xrdf+urdf
#   - viser_serve_forever: spin a Node inside a threading executor while ViserVisualizer runs
#
# All curobo imports are LAZY (inside functions) to keep the package importable
# without a GPU/CUDA/curobo stack.

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import yaml
import yourdfpy
from ament_index_python.packages import get_package_share_directory


# ---------------------------------------------------------------------------
# _CuPoseCompat — pose stand-in (no curobo dependency)
# ---------------------------------------------------------------------------

@dataclass
class _CuPoseCompat:
    """Minimal API-compatible Pose so this file doesn't import curobo at module scope.

    Matches ``curobo.types.Pose.from_list()`` and the attribute access pattern
    used by ``ViserVisualizer.add_frame()``.
    """
    position: object  # torch.Tensor
    quaternion: object  # torch.Tensor

    @staticmethod
    def from_list(values: List[float]) -> "_CuPoseCompat":
        import torch
        return _CuPoseCompat(
            position=torch.tensor(values[:3], dtype=torch.float32),
            quaternion=torch.tensor(values[3:7], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# URDF helpers
# ---------------------------------------------------------------------------

def _resolve_package_paths_in_urdf(urdf_path: str) -> str:
    """Resolve ``package://pkg/rel`` URIs in a URDF and return a temp file path."""
    with open(urdf_path) as f:
        content = f.read()

    def _replace(match):
        pkg = match.group(1)
        rel = match.group(2)
        try:
            share = get_package_share_directory(pkg)
            return os.path.join(share, rel)
        except Exception:
            return match.group(0)

    resolved = re.sub(r"package://([^/]+)/(.+)", _replace, content)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    tmp.write(resolved)
    tmp.close()
    return tmp.name


def _build_robot_data_dict(config_path: str, urdf_path: str, asset_path: str) -> dict:
    """Build a robot-data dict compatible with ViserVisualizer's ``content_path``.

    Supports both XRDF and cuRobo v2 YAML config files.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    resolved_urdf = _resolve_package_paths_in_urdf(urdf_path)
    urdf = yourdfpy.URDF.load(resolved_urdf, build_scene_graph=True)
    actuated_joint_names = [j.name for j in urdf.actuated_joints]

    mesh_link_names = []
    for link in urdf.robot.links:
        for visual in link.visuals:
            if visual.geometry is not None and hasattr(visual.geometry, "filename"):
                mesh_link_names.append(link.name)
                break

    child_links = {j.child for j in urdf.robot.joints}
    try:
        base_link = next(
            link.name for link in urdf.robot.links if link.name not in child_links
        )
    except StopIteration:
        base_link = urdf.robot.links[0].name if urdf.robot.links else "base_link"

    is_yaml = "robot_cfg" in raw or raw.get("format") != "xrdf"

    if is_yaml:
        return _build_from_yaml(raw, resolved_urdf, actuated_joint_names,
                                mesh_link_names, base_link)
    else:
        return _build_from_xrdf(raw, resolved_urdf, actuated_joint_names,
                                mesh_link_names, base_link)


def _build_from_yaml(raw: dict, resolved_urdf: str,
                     actuated_joint_names: list, mesh_link_names: list,
                     default_base_link: str) -> dict:
    """Build robot-data dict from a cuRobo v2 YAML config."""
    robot_cfg = raw.get("robot_cfg", raw)
    kin = robot_cfg.get("kinematics", {})

    kin["urdf_path"] = resolved_urdf
    kin["asset_root_path"] = kin.get("asset_root_path", "")
    kin["mesh_link_names"] = mesh_link_names
    kin.setdefault("base_link", default_base_link)

    if "collision_spheres" not in kin:
        kin["collision_spheres"] = {}
    kin.setdefault("collision_link_names", list(kin["collision_spheres"].keys()))
    kin.setdefault("collision_sphere_buffer", 0.0)
    kin.setdefault("self_collision_ignore", {})
    kin.setdefault("self_collision_buffer", {})
    kin.setdefault("tool_frames", [])
    kin.setdefault("lock_joints", {})
    kin.setdefault("extra_links", {})

    if "cspace" not in kin:
        kin["cspace"] = {
            "joint_names": actuated_joint_names,
            "default_joint_position": [0.0] * len(actuated_joint_names),
            "null_space_weight": [1.0] * len(actuated_joint_names),
            "cspace_distance_weight": [1.0] * len(actuated_joint_names),
            "max_acceleration": [10.0] * len(actuated_joint_names),
            "max_jerk": [500.0] * len(actuated_joint_names),
        }

    _reconcile_gripper_joints(kin, actuated_joint_names)

    return {"robot_cfg": {"kinematics": kin}}


def _reconcile_gripper_joints(kin: dict, actuated_joint_names: list) -> None:
    """Bring URDF actuated joints missing from the cspace into the tree as locked joints.

    A cuRobo robot config ships with only its *active* joints in ``cspace``
    (e.g. a 7-DOF arm). But the URDF may expose extra actuated joints (e.g. a
    gripper ``finger_joint`` plus its mimic joints). ``ViserVisualizer`` reads
    *all* actuated joints from the URDF for its ``_viser_joint_names``, so a
    kinematics model built only from the cspace (7 joints) mismatches the
    viewer (8 joints) and ``set_joint_state()`` crashes on reindex.

    This extends the kinematic chain past each missing actuated joint by
    adding its child link to ``collision_link_names`` (with a sentinel empty
    sphere entry so it contributes no collision primitives) and locks the joint
    at a fixed position. The result: the visualizer kinematics now report the
    same total joint count as the URDF, while the active cspace is unchanged.

    Only non-mimic joints are locked directly; pure mimic joints follow their
    actuated parent automatically.
    """
    active_names = set(kin.get("cspace", {}).get("joint_names") or [])
    locked = kin.setdefault("lock_joints", {})
    coll_links = set(kin.get("collision_link_names") or [])
    spheres = kin.get("collision_spheres") or {}

    urdf = yourdfpy.URDF.load(kin["urdf_path"], build_scene_graph=False)
    joint_by_name = {j.name: j for j in urdf.robot.joints}

    for name in actuated_joint_names:
        if name in active_names or name in locked:
            continue
        j = joint_by_name.get(name)
        if j is None or j.mimic is not None:
            # non-actuated roles (e.g. pure mimic joints) are handled by the
            # actuated parent; skip them.
            continue
        locked[name] = 0.0
        if j.child not in coll_links:
            coll_links.add(j.child)
            spheres.setdefault(j.child, [])

    kin["collision_link_names"] = sorted(coll_links)
    kin["collision_spheres"] = spheres


def _build_from_xrdf(xrdf: dict, resolved_urdf: str,
                      actuated_joint_names: list, mesh_link_names: list,
                      default_base_link: str) -> dict:
    """Build robot-data dict from an XRDF config (legacy format)."""
    kin = {}
    coll_geom = xrdf.get("collision", {}).get("geometry", "collision_model")
    spheres = xrdf.get("geometry", {}).get(coll_geom, {}).get("spheres", {})
    kin["collision_spheres"] = spheres
    kin["collision_link_names"] = list(spheres.keys())
    kin["collision_sphere_buffer"] = xrdf.get("collision", {}).get(
        "buffer_distance", 0.0
    )

    sc = xrdf.get("self_collision", {})
    kin["self_collision_ignore"] = sc.get("ignore", {})
    kin["self_collision_buffer"] = sc.get("buffer_distance", {})
    kin["tool_frames"] = xrdf.get("tool_frames", [])

    csp = xrdf.get("cspace", {})
    active_joints = csp.get("joint_names", [])
    default_pos = xrdf.get("default_joint_positions", {})

    active_config = []
    locked_joints = {}
    for j in actuated_joint_names:
        if j in active_joints:
            active_config.append(default_pos.get(j, 0.0))
        else:
            locked_joints[j] = default_pos.get(j, 0.0)

    all_joints = active_joints + list(locked_joints.keys())
    acc_limits = csp.get("acceleration_limits", [10.0])
    jerk_limits = csp.get("jerk_limits", [500.0])
    max_acc = max(acc_limits) if acc_limits else 10.0
    max_jerk = max(jerk_limits) if jerk_limits else 500.0

    kin["cspace"] = {
        "joint_names": all_joints,
        "default_joint_position": active_config + list(locked_joints.values()),
        "null_space_weight": [1.0] * len(all_joints),
        "cspace_distance_weight": [1.0] * len(all_joints),
        "max_acceleration": acc_limits + [max_acc] * len(locked_joints),
        "max_jerk": jerk_limits + [max_jerk] * len(locked_joints),
    }
    kin["lock_joints"] = locked_joints

    base_link = default_base_link
    extra_links = {}
    for mod in xrdf.get("modifiers", []):
        if "set_base_frame" in mod:
            base_link = mod["set_base_frame"]
        elif "add_frame" in mod:
            fd = mod["add_frame"]
            extra_links[fd["frame_name"]] = {
                "parent_link_name": fd["parent_frame_name"],
                "link_name": fd["frame_name"],
                "joint_name": fd["joint_name"],
                "joint_type": fd["joint_type"],
                "fixed_transform": (
                    fd["fixed_transform"]["position"]
                    + [fd["fixed_transform"]["orientation"]["w"]]
                    + fd["fixed_transform"]["orientation"]["xyz"]
                ),
            }

    kin["extra_links"] = extra_links
    kin["base_link"] = base_link
    kin["urdf_path"] = resolved_urdf
    kin["asset_root_path"] = ""
    kin["mesh_link_names"] = mesh_link_names

    return {"robot_cfg": {"kinematics": kin}}


# ---------------------------------------------------------------------------
# ESDF slice helper (for volumetric/feature mapping visualization)
# ---------------------------------------------------------------------------

def _extract_esdf_slice(
    esdf_grid,
    origin,
    voxel_size: float,
    slice_pose,
    grid_size_m,
    slice_resolution: int = 128,
):
    """Extract an ESDF slice as an RGB image for viser display.

    ``esdf_grid``, ``origin``, ``slice_pose`` are all torch.Tensors on CUDA.
    Returns an (H, W, 3) uint8 numpy array.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    device = esdf_grid.device
    nx, ny, nz = esdf_grid.shape

    half_extent = torch.tensor(
        [(nx - 1) * voxel_size / 2.0,
         (ny - 1) * voxel_size / 2.0,
         (nz - 1) * voxel_size / 2.0],
        device=device,
    )

    half = max(grid_size_m[0], grid_size_m[1]) / 2.0
    u = torch.linspace(-half, half, slice_resolution, device=device)
    v = torch.linspace(-half, half, slice_resolution, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    local_points = torch.stack(
        [
            uu.flatten(),
            vv.flatten(),
            torch.zeros(slice_resolution * slice_resolution, device=device),
            torch.ones(slice_resolution * slice_resolution, device=device),
        ],
        dim=1,
    )

    pose_tensor = torch.tensor(slice_pose, dtype=torch.float32, device=device)
    world_points = (pose_tensor @ local_points.T).T[:, :3]

    local_pts = world_points - origin.to(device)
    normalized = local_pts / half_extent

    coords = normalized[:, [2, 1, 0]]
    coords = coords.view(1, 1, slice_resolution, slice_resolution, 3).float()

    esdf_5d = esdf_grid.float().unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(
        esdf_5d, coords, mode="bilinear", padding_mode="border", align_corners=True
    )
    values = sampled.squeeze().cpu().numpy()

    max_dist = max(np.max(values), 0.1)
    max_negative_dist = max(np.abs(np.min(values)), 0.05)
    normalized_pos = np.clip(values / max_dist, -1, 1)
    normalized_neg = np.clip(values / max_negative_dist, -1, 1)

    colors = np.zeros((slice_resolution, slice_resolution, 3), dtype=np.uint8)
    neg_mask = normalized_neg < 0
    colors[neg_mask, 0] = ((1 + normalized_neg[neg_mask]) * 255).astype(np.uint8)
    colors[neg_mask, 1] = ((1 + normalized_neg[neg_mask]) * 255).astype(np.uint8)
    colors[neg_mask, 2] = 255
    pos_mask = normalized_pos >= 0
    colors[pos_mask, 0] = 255
    colors[pos_mask, 1] = ((1 - normalized_pos[pos_mask]) * 255).astype(np.uint8)
    colors[pos_mask, 2] = ((1 - normalized_pos[pos_mask]) * 255).astype(np.uint8)

    colors[np.abs(values) < voxel_size * 0.5] = [0, 255, 0]
    return colors


# ---------------------------------------------------------------------------
# Non-blocking service polling + joint-name helpers
# ---------------------------------------------------------------------------

def start_service_poll(node, client, on_ready, period=0.5):
    """Non-blocking service watchdog.

    Polls ``client.service_is_ready()`` on a ROS timer (instead of the blocking
    ``wait_for_service()``, which can deadlock a node that has not yet started
    spinning). As soon as the service appears, the timer is cancelled and
    ``on_ready(client)`` is invoked exactly once.

    Returns nothing; the timer is owned by ``node``.
    """
    state = {"done": False, "timer": None}

    def _tick():
        if state["done"]:
            return
        if not client.service_is_ready():
            return
        state["done"] = True
        state["timer"].cancel()
        try:
            on_ready(client)
        except Exception as exc:  # noqa: BLE001
            node.get_logger().error(f"Service handler failed: {exc}")

    state["timer"] = node.create_timer(period, _tick)


def active_joint_names_from_content(content_path) -> list:
    """Extract the robot's active (non-locked) joint names from the content dict
    built by ``_build_robot_data_dict``.

    Falls back to a generic ``joint_1..N`` naming when the dict does not expose
    them, which keeps standalone demos working with any robot config.
    """
    if isinstance(content_path, dict):
        kin = content_path.get("robot_cfg", {}).get("kinematics", {})
        csp = kin.get("cspace", {})
        names = csp.get("joint_names")
        locked = kin.get("lock_joints", {})
        if names:
            active = [n for n in names if n not in locked]
            if active:
                return list(active)
            return list(names)
        default_pos = kin.get("default_joint_position") or []
        return [f"joint_{i + 1}" for i in range(len(default_pos))]
    return []


def _home_positions_from_content(content_path, active_names):
    """Home (default) joint positions for the given active joint names.

    Reads the robot dict built by ``_build_robot_data_dict`` and returns a
    list of positions aligned with ``active_names``. Falls back to zeros when
    the content is unavailable or a joint is missing. Used as the plan-start
    fallback when no live ``/joint_states`` is available yet.
    """
    positions = [0.0] * len(active_names)
    if not isinstance(content_path, dict):
        return positions
    kin = content_path.get("robot_cfg", {}).get("kinematics", {})
    csp = kin.get("cspace", {})
    default_pos = csp.get("default_joint_position") or []
    all_names = csp.get("joint_names") or []
    if not all_names:
        return positions
    name_to_default = dict(zip(all_names, default_pos))
    for i, name in enumerate(active_names):
        if name in name_to_default:
            positions[i] = float(name_to_default[name])
    return positions


def _start_positions(content_path, active_names, js):
    """Resolve plan-start joint positions.

    Prefers the live ``/joint_states`` message (matched by joint name, in
    ``active_names`` order); otherwise returns the robot's home positions.
    """
    if js is not None and js.name and len(js.position) == len(js.name):
        js_map = dict(zip(js.name, js.position))
        if any(n in js_map for n in active_names):
            return [float(js_map.get(n, 0.0)) for n in active_names]
    return _home_positions_from_content(content_path, active_names)


def _viz_set_positions(viz, positions, names):
    """Apply joint positions to the viser robot safely.

    curobo's ``set_joint_positions`` builds a ``JointState`` via
    ``from_position`` and later calls ``joint_state.clone()`` (a torch op), so
    positions must be a torch tensor — neither a python list nor a numpy array
    works. The visualizer's kinematics live on CUDA, so the tensor must be
    created on that device (a CPU tensor causes a ``torch.cat`` device mismatch
    inside ``get_full_js``). This converts to a float32 torch tensor on the
    solver's device and guards against empty/ragged names or positions so a
    stray ``/joint_states`` message (or an unsolved request) never crashes the
    node.
    """
    import torch

    if not names or not positions or len(positions) != len(names):
        return
    # A robot-less visualizer (e.g. launched without content_path/urdf_path so
    # no robot was loaded) has no joint set yet; calling set_joint_positions
    # then crashes on the missing internal '_viser_joint_names'. Skip unless
    # the visualizer actually loaded a robot.
    if not getattr(viz, "_viser_joint_names", None):
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(list(positions), dtype=torch.float32, device=device)
    viz.set_joint_positions(tensor, list(names))


# ---------------------------------------------------------------------------
# Spin helper — keep a ROS Node alive inside a ViserVisualizer event loop
# ---------------------------------------------------------------------------

def viser_serve_forever(node, viz, logger=None):
    """Spin the ROS Node on the main thread.

    ``ViserVisualizer`` runs its own GUI/networking on background threads once
    constructed, so the ROS node can spin synchronously here on the main thread.
    This is the same proven pattern used by the original esdf_viser_node
    (``rclpy.spin(node)``) — spinning in a daemon thread does NOT reliably
    process rclpy timers/service callbacks, which breaks the service poll/solve
    cycle.

    Call this at the end of main() after node construction.
    """
    import rclpy

    if logger:
        logger.info("Viser visualizer running — open the web GUI, Ctrl-C to quit.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
