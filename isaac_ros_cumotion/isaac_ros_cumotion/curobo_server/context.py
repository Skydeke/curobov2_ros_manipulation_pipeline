from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

from curobo.motion_planner import MotionPlanner
from curobo.perception import Mapper
from curobo.scene import Scene, VoxelGrid as CuVoxelGrid
from moveit_msgs.msg import CollisionObject

if TYPE_CHECKING:
    from rclpy.node import Node


@dataclass
class CuroboContext:
    """Single source of truth for all cuRobo GPU objects in the unified node.

    Built exactly once in ``CuroboServerNode.__init__`` and then passed (not
    the ROS node) to every handler class.  Handlers depend on this object,
    not on ``rclpy.Node``.
    """

    motion_planner: MotionPlanner
    mapper: Optional[Mapper] = None
    world_objects: Dict[str, CollisionObject] = field(default_factory=dict)
    attached_objects: Dict[str, CollisionObject] = field(default_factory=dict)
    esdf_scene: Optional[Scene] = None
    esdf_voxel_grid: Optional[CuVoxelGrid] = None
    device: str = "cuda:0"
    logger: Optional[Node] = None
    node: Optional[Node] = None
    _world_seq: int = 0
