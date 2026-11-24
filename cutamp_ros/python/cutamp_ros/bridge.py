# SPDX-License-Identifier: Apache-2.0
"""cuTAMP planning bridge for ``cutamp_ros::CuTampComponent``.

This module is imported by the C++ component from inside the cuRobo process's
embedded CPython interpreter. It must never create an interpreter, a CUDA
context or a second cuRobo model: it only builds a ``RobotContainer`` around
the robot data supplied by the ROS node and runs ``cutamp.algorithm.solve_cutamp``.

Two entry points are exposed:

``ensure_initialized(settings)``
    Build and cache the robot-side cuTAMP objects:
    ``KinematicsCfg`` -> ``Kinematics`` -> ``RobotKinematics`` -> ``build_ik``
    -> ``RobotContainer``. Idempotent; the C++ node calls it on the first plan
    (or at construction when ``warmup`` is true).

``plan(request)``
    Run cuTAMP for one pick/place task and return a plain dict::

        {
          "success": bool,
          "message": str,
          "goalsets": [[16 row-major 4x4 doubles], ...],
          "gripper_close": [int, ...],
          "action_index": [int, ...],
          "num_satisfying": int,
        }

    Each ``goalsets`` entry is one ordered segment with exactly one tool pose
    in the robot base frame. The C++ side wraps it in a
    ``curobov2_ros_interfaces/msg/Goalset`` with ``poses=[Pose]``.

No exception escapes ``plan``: failures are returned as ``success=False`` plus
the formatted traceback, so the service always answers.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Legacy cuTAMP motion_solver applied this offset in the end-effector frame.
# It is converted into the tool frame below so the emitted approach/retreat
# poses stay correct for robots whose tool frame is rotated against the EE.
APPROACH_OFFSET = 0.12
RETREAT_OFFSET = 0.03

_DEVICE_CFG = None
_CONTAINER = None
_SETTINGS: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _require_file(path: str, what: str) -> str:
    if not path:
        raise ValueError(f"{what} is empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} does not exist: {path}")
    return path


def ensure_initialized(settings: Dict[str, Any]) -> None:
    """Build the cached cuTAMP robot container from the node parameters."""
    global _DEVICE_CFG, _CONTAINER, _SETTINGS

    if _CONTAINER is not None:
        return

    # Imported lazily so importing this module is cheap and errors surface at
    # initialization time with a clear message.
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import DeviceCfg
    from cutamp.robots import RobotContainer
    from cutamp.robots.ik import build_ik
    from cutamp.robots.kinematics import RobotKinematics

    robot_config_path = _require_file(
        str(settings.get("robot_config_path", "")), "robot_config_path"
    )
    urdf_path = str(settings.get("urdf_path", "") or "")
    if urdf_path:
        _require_file(urdf_path, "urdf_path")
    tool_frame = str(settings.get("tool_frame", "") or "")
    if not tool_frame:
        raise ValueError("tool_frame is empty")

    device_index = int(settings.get("device", 0) or 0)
    _DEVICE_CFG = DeviceCfg(device=f"cuda:{device_index}")

    kinematics_cfg = KinematicsCfg.from_robot_yaml_file(
        robot_config_path,
        tool_frames=[tool_frame],
        urdf_path=urdf_path or None,
        device_cfg=_DEVICE_CFG,
    )
    kinematics = Kinematics(kinematics_cfg)
    kin_model = RobotKinematics(kinematics, tool_frame)

    ik_solver = build_ik(
        kinematics_cfg,
        tool_frame,
        batch_size=int(settings.get("ik_batch_size", 0) or 0) or 1,
        num_seeds=int(settings.get("ik_num_seeds", 0) or 0) or 1,
        self_collision_check=bool(settings.get("self_collision_check", False)),
        use_cuda_graph=bool(settings.get("use_cuda_graph", False)),
        device_cfg=_DEVICE_CFG,
    )

    spheres_path = _require_file(
        str(settings.get("gripper_spheres", "")), "gripper_spheres"
    )
    gripper_spheres = torch.as_tensor(
        np.load(spheres_path), dtype=torch.float32, device=_DEVICE_CFG.device
    )
    if gripper_spheres.ndim != 2 or gripper_spheres.shape[1] != 4:
        raise ValueError(
            "gripper_spheres must be an (n, 4) array, got "
            f"{tuple(gripper_spheres.shape)}"
        )

    tool_from_ee_values = [float(v) for v in settings.get("tool_from_ee", [])]
    if len(tool_from_ee_values) != 16:
        raise ValueError(
            "tool_from_ee must contain exactly 16 row-major values, got "
            f"{len(tool_from_ee_values)}"
        )
    tool_from_ee = torch.as_tensor(
        tool_from_ee_values, dtype=torch.float32, device=_DEVICE_CFG.device
    ).reshape(4, 4)

    joint_limits = kin_model.joint_limits
    if joint_limits.ndim != 2 or joint_limits.shape[0] != 2:
        raise ValueError(
            f"expected joint limits with shape (2, dof), got {tuple(joint_limits.shape)}"
        )

    _CONTAINER = RobotContainer(
        name=str(settings.get("robot_name", "") or os.path.splitext(
            os.path.basename(robot_config_path)
        )[0]),
        kin_model=kin_model,
        joint_limits=joint_limits,
        gripper_spheres=gripper_spheres,
        tool_from_ee=tool_from_ee,
        ik_solver=ik_solver,
    )
    _SETTINGS = dict(settings)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

_SCENEOBJECT_CUBOID = 0
_SCENEOBJECT_SPHERE = 1
_SCENEOBJECT_CAPSULE = 2
_SCENEOBJECT_CYLINDER = 3
_SCENEOBJECT_MESH = 4


def _pose_to_curobo(pose: Sequence[float]) -> List[float]:
    """[x, y, z, qx, qy, qz, qw] -> cuRobo's [x, y, z, qw, qx, qy, qz]."""
    if len(pose) != 7:
        raise ValueError(f"pose must have 7 values, got {len(pose)}")
    x, y, z, qx, qy, qz, qw = (float(v) for v in pose)
    return [x, y, z, qw, qx, qy, qz]


def _pose_to_mat4x4(pose: Sequence[float]) -> torch.Tensor:
    """[x, y, z, qx, qy, qz, qw] -> 4x4 homogeneous matrix (float32)."""
    from cutamp.utils.common import pose_list_to_mat4x4

    return pose_list_to_mat4x4(_pose_to_curobo(pose)).to(torch.float32)


def _make_object(spec: Dict[str, Any]):
    from curobo.scene import Capsule, Cuboid, Cylinder, Sphere

    name = str(spec.get("name", "") or "")
    if not name:
        raise ValueError("every SceneObject needs a non-empty name")
    obj_type = int(spec.get("type", 0))
    pose = _pose_to_curobo(spec.get("pose", []))
    dims = [float(v) for v in spec.get("dims", [])]
    if len(dims) != 3:
        raise ValueError(f"object '{name}': dims must have 3 values, got {len(dims)}")

    if obj_type == _SCENEOBJECT_CUBOID:
        if min(dims) <= 0.0:
            raise ValueError(f"object '{name}': cuboid dims must be positive")
        return Cuboid(name=name, pose=pose, dims=dims)
    if obj_type == _SCENEOBJECT_SPHERE:
        if dims[0] <= 0.0:
            raise ValueError(f"object '{name}': sphere radius must be positive")
        return Sphere(name=name, pose=pose, radius=dims[0])
    if obj_type == _SCENEOBJECT_CYLINDER:
        if dims[0] <= 0.0 or dims[1] <= 0.0:
            raise ValueError(f"object '{name}': cylinder radius/height must be positive")
        return Cylinder(name=name, pose=pose, radius=dims[0], height=dims[1])
    if obj_type == _SCENEOBJECT_CAPSULE:
        if dims[0] <= 0.0 or dims[1] <= 0.0:
            raise ValueError(f"object '{name}': capsule radius/height must be positive")
        half_height = dims[1] / 2.0
        return Capsule(
            name=name,
            pose=pose,
            radius=dims[0],
            base=[0.0, 0.0, -half_height],
            tip=[0.0, 0.0, half_height],
        )
    if obj_type == _SCENEOBJECT_MESH:
        raise NotImplementedError(
            f"object '{name}': MESH objects are not supported by cuTAMP"
        )
    raise ValueError(f"object '{name}': unknown SceneObject type {obj_type}")


def _parse_goal_atoms(atoms: Sequence[str]):
    """Ground goal atoms such as ``On(block0,table)`` / ``HandEmpty()``."""
    from cutamp.tamp_domain import all_tamp_fluents

    by_name = {fluent.name: fluent for fluent in all_tamp_fluents}
    grounded = set()
    for atom in atoms:
        text = str(atom).strip()
        if not text:
            continue
        open_idx = text.find("(")
        if open_idx < 0 or not text.endswith(")"):
            raise ValueError(f"goal atom {text!r} is not of the form Name(args)")
        name = text[:open_idx].strip()
        inner = text[open_idx + 1 : -1].strip()
        values = [part.strip() for part in inner.split(",")] if inner else []
        fluent = by_name.get(name)
        if fluent is None:
            raise ValueError(
                f"goal atom {text!r}: unknown predicate {name!r}; "
                f"known predicates: {sorted(by_name)}"
            )
        if len(values) != len(fluent.parameters):
            raise ValueError(
                f"goal atom {text!r}: predicate {name} expects "
                f"{len(fluent.parameters)} arguments, got {len(values)}"
            )
        grounded.add(fluent.ground(*values))
    return frozenset(grounded)


def _build_environment(request: Dict[str, Any]):
    from cutamp.envs.utils import TAMPEnvironment

    objects = [_make_object(spec) for spec in request.get("objects", [])]
    by_name = {obj.name: obj for obj in objects}
    if len(by_name) != len(objects):
        raise ValueError("duplicate object names in the request")

    movable_names = [str(name) for name in request.get("movable_names", [])]
    missing = [name for name in movable_names if name not in by_name]
    if missing:
        raise ValueError(
            f"movable_names {missing} are not present in objects {sorted(by_name)}"
        )
    movable_set = set(movable_names)
    movables = [by_name[name] for name in movable_names]
    statics = [obj for obj in objects if obj.name not in movable_set]

    goal_state = _parse_goal_atoms(request.get("goal_atoms", []))
    if not goal_state:
        raise ValueError("goal_atoms is empty")

    # Pick/place only: everything that is not movable acts as a support
    # surface. Push/button domains are intentionally not inferred.
    type_to_objects = {"Movable": movables, "Surface": statics}
    return TAMPEnvironment(
        name=str(request.get("env_name", "") or "cutamp_scene"),
        movables=movables,
        statics=statics,
        type_to_objects=type_to_objects,
        goal_state=goal_state,
    )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _grasps_from_request(request: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """External tool-frame grasp poses -> cuTAMP object-frame 6-DOF grasps."""
    names = [str(name) for name in request.get("grasp_object_names", [])]
    poses = [list(pose) for pose in request.get("grasp_tool_poses", [])]
    scores = [float(score) for score in request.get("grasp_scores", [])]

    if not (len(names) == len(poses) == len(scores)):
        raise ValueError(
            "grasp_object_names, grasp_tool_poses and grasp_scores must have "
            f"equal lengths, got {len(names)}, {len(poses)}, {len(scores)}"
        )
    if not names:
        return {}

    object_poses: Dict[str, Sequence[float]] = {}
    for spec in request.get("objects", []):
        object_poses[str(spec.get("name", "") or "")] = spec.get("pose", [])

    grouped: Dict[str, Tuple[List[torch.Tensor], List[float]]] = {}
    for name, tool_pose, score in zip(names, poses, scores):
        if name not in object_poses:
            raise ValueError(
                f"grasp for unknown object '{name}'; request objects are "
                f"{sorted(object_poses)}"
            )
        # tool_from_ee is the identity for this bridge, so the tool pose is the
        # grasp pose used by cuTAMP.
        world_from_obj = _pose_to_mat4x4(object_poses[name]).double()
        obj_from_grasp = torch.linalg.inv(world_from_obj) @ _pose_to_mat4x4(tool_pose).double()
        grasps, confidences = grouped.setdefault(name, ([], []))
        grasps.append(obj_from_grasp)
        confidences.append(score)

    device = _DEVICE_CFG.device if _DEVICE_CFG is not None else torch.device("cpu")
    return {
        name: {
            "grasps_obj": torch.stack(grasps).to(device=device, dtype=torch.float32),
            "confidences_pt": torch.as_tensor(
                confidences, dtype=torch.float32, device=device
            ),
        }
        for name, (grasps, confidences) in grouped.items()
    }


def _build_config(request: Dict[str, Any], grasps: Optional[Dict[str, Any]] = None):
    from cutamp.config import TAMPConfiguration

    kwargs: Dict[str, Any] = {
        "robot_container": _CONTAINER,
        "curobo_plan": False,  # the cuRobo server plans the emitted goals
        "enable_visualizer": False,
        "enable_experiment_logging": False,
        "warmup_motion_gen": False,
        "initial_collision_tol": 2.0,
        "world_activation_distance": 0.0,
        "gripper_activation_distance": 0.0,
        "movable_activation_distance": 0.0,
        "cache_subgraphs": True,
    }

    if grasps:
        kwargs["m2t2_grasps"] = True
        kwargs["grasp_dof"] = 6
        kwargs["num_opt_steps"] = 1
        kwargs["lr"] = 1e-6
        kwargs["conf_lr"] = 1e-6

    num_particles = int(request.get("num_particles", 0) or 0)
    if num_particles > 0:
        kwargs["num_particles"] = num_particles
    num_opt_steps = int(request.get("num_opt_steps", 0) or 0)
    if num_opt_steps > 0 and not grasps:
        kwargs["num_opt_steps"] = num_opt_steps
    num_initial_plans = int(request.get("num_initial_plans", 0) or 0)
    kwargs["num_initial_plans"] = max(1, num_initial_plans or 10)
    max_loop_dur = float(request.get("max_loop_dur", 0.0) or 0.0)
    if max_loop_dur > 0.0:
        kwargs["max_loop_dur"] = max_loop_dur

    if _SETTINGS is not None:
        coll_n_spheres = int(_SETTINGS.get("coll_n_spheres", 0) or 0)
        if coll_n_spheres > 0:
            kwargs["coll_n_spheres"] = coll_n_spheres
        coll_sphere_radius = float(_SETTINGS.get("coll_sphere_radius", 0.0) or 0.0)
        if coll_sphere_radius > 0.0:
            kwargs["coll_sphere_radius"] = coll_sphere_radius

    return TAMPConfiguration(**kwargs)


def _resolve_q_init(request: Dict[str, Any]):
    from cutamp.robots import get_q_home

    values = [float(v) for v in request.get("q_init", [])]
    if values:
        return values
    try:
        return [float(v) for v in get_q_home(_CONTAINER.name)]
    except Exception as exc:  # custom robot without a built-in home config
        raise ValueError(
            "q_init is required: cuTAMP has no built-in home configuration for "
            f"robot '{_CONTAINER.name}'"
        ) from exc


def _approach_transform(tool_from_ee: torch.Tensor, distance: float) -> torch.Tensor:
    """Back-off by ``distance`` along the tool approach axis (+X).

    The legacy cuTAMP motion solver offset the *end-effector* pose along EE z.
    With ``world_from_ee = world_from_tool @ tool_from_ee`` the equivalent
    tool-frame transform is::

        tool_from_ee @ Trans(distance, 0, 0) @ inv(tool_from_ee)
    """
    offset = torch.eye(4, dtype=tool_from_ee.dtype, device=tool_from_ee.device)
    offset[0, 3] = -distance
    return offset


def _mat_to_row_major(matrix: torch.Tensor) -> List[float]:
    return [float(v) for v in matrix.detach().cpu().reshape(-1).tolist()]


_PICK = "Pick"
_PLACE = "Place"
_SKIPPED_OPERATORS = {"MoveFree", "MoveHolding"}


def _solution_to_response(solution, tool_from_ee: torch.Tensor) -> Dict[str, Any]:
    best_particle = solution.best_particle
    if best_particle is None:
        return {
            "success": False,
            "message": "cuTAMP found no satisfying particle",
            "goalsets": [],
            "gripper_close": [],
            "action_index": [],
            "num_satisfying": int(solution.num_satisfying or 0),
        }

    rollout_fn = solution.plan_info.get("rollout_fn")
    plan_skeleton = solution.plan_info.get("plan_skeleton")
    if rollout_fn is None or plan_skeleton is None:
        raise RuntimeError("cuTAMP result is missing rollout_fn/plan_skeleton")

    with torch.no_grad():
        rollout = rollout_fn({key: value[None] for key, value in best_particle.items()})

    # [1, T, 4, 4] -> [T, 4, 4]; one timestep per actionable operator.
    world_from_tool = rollout["world_from_tool_desired"][0]

    back_off = _approach_transform(tool_from_ee, APPROACH_OFFSET)
    pull_back = _approach_transform(tool_from_ee, RETREAT_OFFSET)

    goalsets: List[List[float]] = []
    gripper_close: List[int] = []
    action_index: List[int] = []
    timestep = 0
    action = 0

    for ground_op in plan_skeleton:
        name = ground_op.operator.name
        if name in _SKIPPED_OPERATORS:
            continue
        if name not in (_PICK, _PLACE):
            raise NotImplementedError(
                f"cuTAMP operator {name} is not supported by the pick/place bridge"
            )
        if timestep >= world_from_tool.shape[0]:
            raise RuntimeError(
                "cuTAMP rollout has fewer timesteps than actionable operators"
            )

        pose = world_from_tool[timestep]
        retreat = pose @ pull_back
        pre_grasp = pose @ back_off

        if name == _PICK:
            # Approach open at the pre-grasp, grasp closed, lift closed.
            segment_poses = [pre_grasp, pose, retreat]
            segment_close = [0, 1, 1]
        else:
            # Place still holding, open at the placement, retreat open.
            segment_poses = [pose, pose, retreat]
            segment_close = [1, 0, 0]

        for matrix, close in zip(segment_poses, segment_close):
            goalsets.append(_mat_to_row_major(matrix))
            gripper_close.append(int(close))
            action_index.append(action)

        timestep += 1
        action += 1

    if not goalsets:
        return {
            "success": False,
            "message": "cuTAMP plan contains no pick/place actions",
            "goalsets": [],
            "gripper_close": [],
            "action_index": [],
            "num_satisfying": int(solution.num_satisfying or 0),
        }

    skeleton = [ground_op.operator.name for ground_op in plan_skeleton]
    return {
        "success": True,
        "message": (
            f"cuTAMP plan {skeleton} with {action} pick/place action(s), "
            f"{int(solution.num_satisfying or 0)} satisfying particle(s)"
        ),
        "goalsets": goalsets,
        "gripper_close": gripper_close,
        "action_index": action_index,
        "num_satisfying": int(solution.num_satisfying or 0),
    }


def _run(request: Dict[str, Any]) -> Dict[str, Any]:
    import logging

    logging.getLogger("cutamp.particle_initialization").setLevel(logging.DEBUG)
    from cutamp.algorithm import solve_cutamp
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_reduction import CostReducer
    from cutamp.scripts.utils import (
        default_constraint_to_mult,
        default_constraint_to_tol,
    )

    env = _build_environment(request)
    grasps = _grasps_from_request(request)
    config = _build_config(request, grasps)
    q_init = _resolve_q_init(request)

    from cutamp.task_planning.constraints import (
        Collision,
        KinematicConstraint,
        StablePlacement,
    )

    tolerance = default_constraint_to_tol.copy()
    # The support surface and every other static object stay hard limits; only
    # the movable resting on its support gets a small allowance.
    tolerance[Collision.type] = {
        "default": 0.001,
        "robot_to_world": 0.001,
        "movable_to_world": 0.05,
        "robot_to_movables": 0.001,
    }
    tolerance[KinematicConstraint.type] = {"pos_err": 0.02, "rot_err": 0.2}
    stable = dict(tolerance.get(StablePlacement.type, {}))
    for spec in request.get("objects", []):
        name = str(spec.get("name", "") or "")
        if not name:
            continue
        stable[f"{name}_in_xy"] = 2e-2
        stable[f"{name}_support"] = 2e-2
    tolerance[StablePlacement.type] = stable

    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(tolerance)

    solution = solve_cutamp(
        env,
        config,
        cost_reducer,
        constraint_checker,
        q_init=q_init,
        experiment_id="cutamp_ros",
        grasps=grasps or None,
    )
    return _solution_to_response(solution, _CONTAINER.tool_from_ee)


def plan(request: Dict[str, Any]) -> Dict[str, Any]:
    """Run cuTAMP; never raises. Returns the dict documented at module level."""
    failure = {
        "success": False,
        "message": "",
        "goalsets": [],
        "gripper_close": [],
        "action_index": [],
        "num_satisfying": 0,
    }
    try:
        if _CONTAINER is None:
            raise RuntimeError(
                "cuTAMP bridge is not initialized; call ensure_initialized first"
            )
        return _run(request)
    except Exception:
        failure["message"] = traceback.format_exc()
        return failure
