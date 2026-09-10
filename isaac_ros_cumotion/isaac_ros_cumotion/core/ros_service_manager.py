from functools import partial
from std_srvs.srv import Trigger, SetBool
from isaac_ros_cumotion_interfaces.srv import AddObject, RemoveObject, GetVoxelGrid, GetCollisionDistance, SetCollisionCache, GetRobotStrategies, SetLinkCollision, SetMask
from isaac_ros_cumotion_interfaces.msg import SparseVoxelGrid
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from isaac_ros_cumotion.core.collision_distance import _attributed_collisions


class RosServiceManager:
    """
    Manages all ROS services, publishers, and timers.
    Responsible for:
    - Creating ROS services for obstacle management
    - Publishing visualization markers (collision spheres)
    - Managing timers for periodic publishing
    - Delegating service callbacks to appropriate managers
    """

    def __init__(self, node, obstacle_manager, robot_model_manager, config_manager, config_wrapper=None, robot_context=None):
        """
        Initialize ROS service manager.

        Args:
            node: ROS2 node instance
            obstacle_manager: ObstacleManager for obstacle operations
            robot_model_manager: RobotModelManager for robot collision spheres
            config_manager: ConfigManager for base_link access
            config_wrapper: ConfigWrapper parent for calling update_world_config()
        """
        self.node = node
        self.obstacle_manager = obstacle_manager
        self.robot_model_manager = robot_model_manager
        self.config_manager = config_manager
        self.config_wrapper = config_wrapper
        self.robot_context = robot_context

        # Callback groups assigned by UnifiedPlannerNode so the viz/marker
        # timers can run in parallel with the perception (camera) callbacks
        # under the MultiThreadedExecutor. Absent (standalone tests), the
        # timers fall back to the node's default group.
        self._viz_callback_group = getattr(node, '_viz_callback_group', None)
        self._marker_callback_group = getattr(node, '_marker_callback_group', None)

        # Services (initialized in init_services)
        self.add_object_srv = None
        self.remove_object_srv = None
        self.get_obstacles_srv = None
        self.remove_all_objects_srv = None
        self.get_voxel_map_srv = None
        self.get_collision_distance_srv = None
        self.set_collision_cache_srv = None
        # Robot-segmentation mask services (registered lazily via
        # register_robot_segmentation once the component exists).
        self.set_mask_srv = None
        self.remove_mask_srv = None
        self.segmentation = None

        # Publisher for collision spheres visualization. Enabled by default so the
        # collision spheres (and scene-obstacle markers) show up in RViz without a
        # service call; disable at launch via the publish_collision_spheres param
        # if a live CUDA-graph capture must run with no viz traffic.
        self.publish_collision_spheres_pub = None
        self.publish_collision_spheres_timer = None
        self.collision_spheres_enabled = True
        if node is not None and node.has_parameter('publish_collision_spheres'):
            self.collision_spheres_enabled = bool(
                node.get_parameter('publish_collision_spheres').value)

        # Sparse voxel grid topic publisher + timer (initialized in init_services)
        self.sparse_voxel_pub = None
        self.sparse_voxel_timer = None

        # Mapper workspace marker (wireframe TSDF/ESDF extent box for RViz).
        # On by default so the mapper's sensing/collision volume is visible;
        # disable at launch via the publish_workspace_visualisation param.
        # Published once (latched) at startup — the geometry is static.
        self.workspace_viz_pub = None
        self.workspace_viz_enabled = True
        if node is not None and node.has_parameter('publish_workspace_visualisation'):
            self.workspace_viz_enabled = bool(
                node.get_parameter('publish_workspace_visualisation').value)

    def init_services(self):
        """Create all ROS services, publishers, and timers"""
        # Create services
        self.add_object_srv = self.node.create_service(
            AddObject,
            self.node.get_name() + '/add_object',
            partial(self._callback_add_object, self.node)
        )

        self.remove_object_srv = self.node.create_service(
            RemoveObject,
            self.node.get_name() + '/remove_object',
            partial(self._callback_remove_object, self.node)
        )

        self.get_obstacles_srv = self.node.create_service(
            Trigger,
            self.node.get_name() + '/get_obstacles',
            partial(self._callback_get_obstacles, self.node)
        )

        # Readiness is exposed as the `node_is_available` ROS parameter
        # (declared in ConfigWrapper, flipped around warmup in
        # ConfigWrapperMotion). There is deliberately no /is_available service.

        self.remove_all_objects_srv = self.node.create_service(
            Trigger,
            self.node.get_name() + '/remove_all_objects',
            partial(self._callback_remove_all_objects, self.node)
        )

        self.get_voxel_map_srv = self.node.create_service(
            GetVoxelGrid,
            self.node.get_name() + '/get_voxel_grid',
            partial(self._callback_get_voxel_grid, self.node)
        )

        self.get_collision_distance_srv = self.node.create_service(
            GetCollisionDistance,
            self.node.get_name() + '/get_collision_distance',
            partial(self._callback_get_collision_distance, self.node)
        )

        self.set_collision_cache_srv = self.node.create_service(
            SetCollisionCache,
            self.node.get_name() + '/set_collision_cache',
            partial(self._callback_set_collision_cache, self.node)
        )

        # Service to get available robot strategies (for RViz plugin)
        if self.robot_context is not None:
            self.get_robot_strategies_srv = self.node.create_service(
                GetRobotStrategies,
                self.node.get_name() + '/get_robot_strategies',
                partial(self._callback_get_robot_strategies, self.node)
            )

        # Create publisher for collision spheres
        self.publish_collision_spheres_pub = self.node.create_publisher(
            MarkerArray,
            self.node.get_name() + '/collision_spheres',
            1
        )

        # Service to enable/disable individual link collision spheres
        self.set_link_collision_srv = self.node.create_service(
            SetLinkCollision,
            self.node.get_name() + '/set_link_collision',
            self.robot_model_manager.set_link_collision_callback
        )

        # Service to enable/disable collision sphere publishing (disabled by default)
        self.set_collision_spheres_srv = self.node.create_service(
            SetBool,
            self.node.get_name() + '/set_collision_spheres_enabled',
            self._callback_set_collision_spheres_enabled
        )

        # Create timer for periodic collision sphere publishing
        self.publish_collision_spheres_timer = self.node.create_timer(
            0.5,
            partial(self.publish_collision_spheres, self.node),
            callback_group=self._viz_callback_group
        )

        # Scene-obstacle markers (always on): shows every scene obstacle, with the
        # currently attached (→ disabled) one recoloured. Complements the sparse
        # voxel grid, which doesn't render primitives.
        self.scene_obstacle_pub = self.node.create_publisher(
            MarkerArray,
            self.node.get_name() + '/scene_obstacles',
            1
        )
        self.scene_obstacle_timer = self.node.create_timer(
            0.5,
            partial(self.publish_scene_obstacles, self.node),
            callback_group=self._marker_callback_group
        )

        # Mapper workspace marker: wireframe box of the mapper TSDF/ESDF extent
        # (mapper_extent_xyz centered at mapper_grid_center) in the base frame.
        # Transient-local so a late-joining RViz immediately sees the latched
        # frame. Published ONCE at startup — the workspace geometry is static and
        # never changes at runtime, so there is no periodic timer. Gated by the
        # publish_workspace_visualisation param.
        self.workspace_viz_pub = self.node.create_publisher(
            Marker,
            self.node.get_name() + '/mapper_workspace',
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
        )
        self.publish_mapper_workspace(self.node)

        # Sparse voxel grid topic publisher (occupied linear indices only).
        # Published periodically so the U-Net consumer gets a steady stream.
        # Transient-local so a late-joining consumer (e.g. the RViz
        # SparseVoxelGridDisplay) immediately receives the latest grid instead
        # of waiting for the next publish cycle.
        self.sparse_voxel_pub = self.node.create_publisher(
            SparseVoxelGrid,
            self.node.get_name() + '/voxel_grid_sparse',
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
        )
        sparse_rate = self.node.get_parameter(
            'sparse_voxel_publish_rate').get_parameter_value().double_value
        if sparse_rate > 0.0:
            self.sparse_voxel_timer = self.node.create_timer(
                1.0 / sparse_rate,
                partial(self._publish_sparse_voxel_grid, self.node),
                callback_group=self._viz_callback_group
            )

    def register_robot_segmentation(self, segmentation):
        """Register the depth-map robot segmentation's mask services.

        The RobotSegmentation component is composed into the node after this
        manager is built, so its set_mask / remove_mask services are registered
        here (node-namespaced, like every other service) rather than by the
        component itself. Delegates straight to the component's callbacks.

        Args:
            segmentation: The core.RobotSegmentation instance.
        """
        self.segmentation = segmentation
        self.set_mask_srv = self.node.create_service(
            SetMask,
            self.node.get_name() + '/set_mask',
            segmentation.set_mask_callback,
        )
        self.remove_mask_srv = self.node.create_service(
            RemoveObject,
            self.node.get_name() + '/remove_mask',
            segmentation.remove_mask_callback,
        )
        self.node.get_logger().info(
            f"Registered robot segmentation services on "
            f"{self.node.get_name()}/set_mask and "
            f"{self.node.get_name()}/remove_mask")

    def _publish_sparse_voxel_grid(self, node):
        """Timer callback: publish the current scene occupancy as SparseVoxelGrid.

        This queries the Mapper on the GPU (extract_occupied_voxels + primitive
        rasterization). It runs on a 7 Hz timer independent of planning, so
        without guarding it, it WILL eventually overlap a MotionGen/MPC CUDA graph
        capture and invalidate it — which poisons the whole process's CUDA context
        (cudaErrorStreamCaptureInvalidated), breaking every later plan until restart.

        Same non-blocking guard as the depth camera: if the planner is capturing a
        graph it holds gpu_lock; skip this publish cycle rather than race it. Never
        blocks the planner (non-blocking acquire) and costs nothing when idle.
        """
        gpu_lock = getattr(node, 'gpu_lock', None)
        if gpu_lock is not None and not gpu_lock.acquire(blocking=False):
            node.get_logger().warn(
                "gpu_lock busy (CUDA graph capture in progress) - skipping this "
                "voxel grid publish cycle",
                throttle_duration_sec=5.0)
            return  # planner is capturing a CUDA graph — skip this cycle
        try:
            self.obstacle_manager.publish_sparse_voxel_grid(node, self.sparse_voxel_pub)
        finally:
            if gpu_lock is not None:
                gpu_lock.release()

    def publish_scene_obstacles(self, node):
        """Publish scene obstacles as a MarkerArray, recolouring the currently
        attached (→ disabled by attach) obstacle so attach/detach is visible.

        Obstacle attrs follow curobo/add_object: cuboid has ``.dims`` + ``.pose``
        ([x,y,z,qw,qx,qy,qz]); sphere/cylinder have ``.radius`` (cylinder also
        ``.height``). The attached obstacle's name is exposed by AttachmentServices.
        """
        scene = self.obstacle_manager.get_scene()
        attach_svc = getattr(node, 'attachment_services', None)
        attached_name = attach_svc.attached_name if attach_svc is not None else None

        marker_array = MarkerArray()
        clear = Marker()
        # Empty namespace on purpose: rviz's DELETEALL is namespace-scoped
        # (ros2/rviz#685) — a namespaced DELETEALL only clears markers in that
        # namespace, so markers no longer re-published (e.g. an attached
        # object's spheres after detach) would freeze at their last pose.
        # With an unset ns it clears every marker in this display.
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        mid = 0
        buckets = (('cuboid', Marker.CUBE), ('sphere', Marker.SPHERE),
                   ('cylinder', Marker.CYLINDER), ('capsule', Marker.CYLINDER))
        for bucket, mtype in buckets:
            for obs in (getattr(scene, bucket, None) or []):
                try:
                    pose = list(obs.pose)
                except Exception:
                    continue
                m = Marker()
                m.header.frame_id = self.config_manager.base_link
                m.header.stamp = node.get_clock().now().to_msg()
                m.ns = 'scene_obstacles'
                m.id = mid
                mid += 1
                m.type = mtype
                m.action = Marker.ADD
                # float(): obs.pose may be a numpy array (world-file / round-tripped
                # obstacles), and rosidl asserts PyFloat_Check — numpy scalars abort.
                m.pose.position.x, m.pose.position.y, m.pose.position.z = \
                    (float(v) for v in pose[0:3])
                m.pose.orientation.w, m.pose.orientation.x, \
                    m.pose.orientation.y, m.pose.orientation.z = \
                    (float(v) for v in pose[3:7])
                if bucket == 'cuboid':
                    dims = list(obs.dims)
                    m.scale.x, m.scale.y, m.scale.z = \
                        float(dims[0]), float(dims[1]), float(dims[2])
                else:
                    d = 2.0 * float(getattr(obs, 'radius', 0.05))
                    m.scale.x = m.scale.y = d
                    m.scale.z = float(getattr(obs, 'height', d))
                if attached_name is not None and getattr(obs, 'name', None) == attached_name:
                    # attached → disabled world obstacle: grey, translucent
                    m.color.r, m.color.g, m.color.b, m.color.a = 0.5, 0.5, 0.5, 0.35
                else:
                    # active obstacle: orange
                    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.6
                marker_array.markers.append(m)

        self.scene_obstacle_pub.publish(marker_array)

    def publish_mapper_workspace(self, node):
        """Publish the Mapper workspace as a wireframe box marker for RViz.

        Renders the TSDF/ESDF voxel grid bounds: a box of `mapper_extent_xyz`
        centered at `mapper_grid_center`, expressed in the base frame. Purely
        diagnostic — it does not affect planning. Gated by the
        `publish_workspace_visualisation` param (see __init__). Reads the
        mapper params fresh each cycle so a live override is reflected.
        """
        if not self.workspace_viz_enabled:
            return
        try:
            extent = [float(v) for v in node.get_parameter('mapper_extent_xyz').value]
            center = [float(v) for v in node.get_parameter('mapper_grid_center').value]
        except Exception as e:
            node.get_logger().warn(
                f"Could not build mapper workspace marker from params: {e}",
                throttle_duration_sec=5.0)
            return

        half = [e / 2.0 for e in extent]
        corners = [
            [center[0] + sx * half[0], center[1] + sy * half[1], center[2] + sz * half[2]]
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ]
        # 12 box edges as corner-index pairs (enumeration below matches the
        # lexicographic corner order above).
        edges = [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        ]

        m = Marker()
        m.header.frame_id = self.config_manager.base_link
        m.header.stamp = node.get_clock().now().to_msg()
        m.ns = 'mapper_workspace'
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.01  # line width (m)
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 1.0, 1.0
        for i, j in edges:
            for c in (corners[i], corners[j]):
                p = Point()
                p.x, p.y, p.z = c[0], c[1], c[2]
                m.points.append(p)
        self.workspace_viz_pub.publish(m)

    def _callback_add_object(self, node, request: AddObject, response):
        """Delegate add_object to ObstacleManager.

        Propagation to the solvers happens through ObstacleManager's
        world-update observer (registered by ConfigWrapper), so it works for
        ROS service AND direct Python callers alike.
        """
        return self.obstacle_manager.add_object(node, request, response)

    def _callback_remove_object(self, node, request: RemoveObject, response):
        """Delegate remove_object to ObstacleManager (observer propagates)."""
        return self.obstacle_manager.remove_object(node, request, response)

    def _callback_get_obstacles(self, node, request: Trigger, response):
        """Delegate get_obstacles service to ObstacleManager"""
        return self.obstacle_manager.get_obstacles(node, request, response)

    def _callback_remove_all_objects(self, node, request: Trigger, response):
        """Delegate remove_all_objects to ObstacleManager (observer propagates)."""
        return self.obstacle_manager.remove_all_objects(node, request, response)

    def _callback_get_voxel_grid(self, node, request: GetVoxelGrid, response):
        """Delegate get_voxel_grid service to ObstacleManager"""
        return self.obstacle_manager.get_voxel_grid(node, request, response)

    def _callback_get_collision_distance(self, node, request: GetCollisionDistance, response):
        """Delegate get_collision_distance service to ConfigWrapper"""
        return self.config_wrapper.callback_get_collision_distance(node, request, response)

    def _callback_set_collision_cache(self, node, request: SetCollisionCache, response):
        """Delegate set_collision_cache service to ObstacleManager.

        Refused while an execution goal is active: a cache change forces a full
        solver rebuild (~20s, see rebuild_solvers_for_cache_change), which is not
        safe to run concurrently with an in-progress trajectory. Same guard/lock
        pattern as set_planner_callback.
        """
        goal_lock = getattr(node, '_goal_lock', None)
        if goal_lock is not None:
            with goal_lock:
                if getattr(node, '_goal_active', False):
                    response.success = False
                    response.message = (
                        "Cannot change collision cache while an execution goal is "
                        "active — cancel it first."
                    )
                    response.obb_cache = self.obstacle_manager.collision_cache["cuboid"]
                    response.mesh_cache = self.obstacle_manager.collision_cache["mesh"]
                    voxel = self.obstacle_manager.collision_cache["voxel"]
                    response.blox_cache = voxel["layers"] if isinstance(voxel, dict) else 0
                    self.node.get_logger().error(response.message)
                    return response
        response = self.obstacle_manager.set_collision_cache(node, request, response)
        if response.success:
            response.message += " - solvers rebuilt (blocking, ~20s)"
        return response

    def _callback_get_robot_strategies(self, node, request: GetRobotStrategies.Request, response: GetRobotStrategies.Response):
        """Delegate get_robot_strategies service to RobotContext"""
        return self.robot_context.get_robot_strategies_callback(node, request, response)

    def _callback_set_collision_spheres_enabled(self, request: SetBool.Request, response: SetBool.Response):
        """Enable or disable collision sphere publishing (e.g. disable during MPC graph capture)"""
        self.collision_spheres_enabled = request.data
        state = "enabled" if request.data else "disabled"
        self.node.get_logger().info(f"Collision sphere publishing {state}")
        response.success = True
        response.message = f"Collision spheres {state}"
        return response

    def publish_collision_spheres(self, node):
        """
        Publishes the robot's collision spheres as markers for visualization in RViz.
        Useful for debugging and ensuring proper masking of the robot in the point cloud.

        Spheres are coloured green when they stay clear of every obstacle and
        red once they come within the planner's ``collision_activation_distance``
        margin (the SAME activation distance the optimizer uses, so the RViz
        visual matches what the planner treats as a collision; with the default
        0.5 cm that means "would the planner refuse/reject this"). A collision
        also triggers a debug log naming the offending link + obstacle(s);
        except for spheres belonging to an attached object which are blue when
        not colliding and red when colliding.
        """
        if not self.collision_spheres_enabled:
            return

        # Source spheres from the MotionPlanner's kinematics when available — it
        # carries attaches (our robot_model_manager kin_model does not), so the
        # fitted attached-object spheres show and ride the arm. attached_mask
        # flags them for a distinct colour. Fall back to robot_model_manager.
        kin = self._attachment_kinematics()
        if kin is not None:
            try:
                robot_spheres, attached_mask = \
                    self.robot_model_manager.get_collision_spheres_with_attached(kin)
            except Exception as e:
                self.node.get_logger().debug(
                    f"attached-sphere viz fallback: {e}", throttle_duration_sec=5.0)
                robot_spheres = self.robot_model_manager.get_collision_spheres()
                attached_mask = [False] * len(robot_spheres)
        else:
            robot_spheres = self.robot_model_manager.get_collision_spheres()
            attached_mask = [False] * len(robot_spheres)

        # Determine per-sphere collision status so colliding spheres can be
        # coloured red. Purely a visualization nicety — any failure falls back
        # to "all green".
        colliding = self._spheres_in_collision(node, kin)
        red_indices = {r['index'] for r in colliding or []}

        # Prepend a DELETEALL so markers that are no longer re-published (e.g.
        # an attached object's spheres after detach) do not linger. It must
        # carry an EMPTY namespace: rviz's DELETEALL is namespace-scoped
        # (ros2/rviz#685), so a namespaced DELETEALL would only clear markers
        # in that namespace. The sphere markers all live under
        # ns='collision_spheres', so an unset-ns DELETEALL cannot collide with
        # any of them.
        marker_array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for i, sphere in enumerate(robot_spheres):
            if sphere[3] <= 0:  # skip disabled spheres (radius = -100)
                continue
            marker = Marker()
            marker.ns = 'collision_spheres'
            marker.header.frame_id = self.config_manager.base_link
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.pose.position.x = sphere[0]
            marker.pose.position.y = sphere[1]
            marker.pose.position.z = sphere[2]
            marker.scale.x = sphere[3] * 2  # Diameter
            marker.scale.y = sphere[3] * 2
            marker.scale.z = sphere[3] * 2
            marker.color.a = 0.5  # Transparency
            is_colliding = colliding is not None and i in red_indices
            if is_colliding:
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0  # colliding = red
            elif attached_mask[i]:  # attached object, not colliding = blue
                marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0  # collision-free = green
            marker_array.markers.append(marker)

        # Publish marker array
        self.publish_collision_spheres_pub.publish(marker_array)

    def _spheres_in_collision(self, node, kin):
        """
        Margin-aware sphere collisions with obstacle attribution for the RViz
        colouring: spheres go red when they come within the
        ``collision_activation_distance`` margin (the same activation distance
        the planner's optimizer uses — the display matches planning, so "red"
        means "the planner would treat this as in collision"; no clearance
        margin is added on top). A positive entry means the sphere surface is
        closer to an obstacle than the margin, and its value is the overlap of
        the inflated sphere in metres (depth beyond activation ⇒ real contact).

        When any sphere is red, logs what causes it: per-sphere link, obstacle
        name(s), position, radius and overlap (a fresh full dump on the
        red-set transition, then a throttled "still red" summary while the same
        collision persists, so the timer doesn't spam).

        Returns the list of attribution dicts (see
        collision_distance._attributed_collisions) or None if the solver /
        collision checker isn't ready (callers then fall back to "all green").
        Runs under the node's gpu_lock so the RViz timer thread can't race a
        CUDA-graph capture on the GPU.
        """
        wrapper = getattr(self, 'config_wrapper', None)
        if wrapper is None:
            return None

        # Reuse the non-blocking gpu_lock guard used elsewhere: the collision
        # checker atoms otherwise collide with the CUDA-graph capture.
        gpu_lock = getattr(node, 'gpu_lock', None)
        if gpu_lock is not None and not gpu_lock.acquire(blocking=False):
            self.node.get_logger().debug(
                "gpu_lock busy (CUDA graph capture) - skipping sphere "
                "collision colouring", throttle_duration_sec=5.0)
            return None

        try:
            reports = _attributed_collisions(wrapper, node, kin=kin)
            if not reports:
                self._collision_red_key = None
                return reports
            # Dedupe log spam: log a full attribution only when the SET of
            # (sphere, link, obstacles) actually changes; while it persists,
            # fall back to a throttled one-liner.
            red_key = tuple(sorted(
                (r['index'], r['link'], tuple(r['obstacles']), r['clobber_note'])
                for r in reports))
            if red_key != getattr(self, '_collision_red_key', None):
                act_mm = reports[0].get('activation')
                act_mm = act_mm * 1000.0 if act_mm is not None else None
                for r in sorted(reports, key=lambda r: -r['depth']):
                    what = ", ".join(r['obstacles']) if r['obstacles'] \
                        else r['clobber_note']
                    margin_txt = (f" within {act_mm:.1f} mm margin"
                                  if act_mm is not None else "")
                    self.node.get_logger().warn(
                        f"collision_spheres: sphere#{r['index']} on link "
                        f"{r['link']} (r={r['radius'] * 1000:.1f} mm at "
                        f"{', '.join(f'{v:.3f}' for v in r['position'])})"
                        f"{margin_txt} of {what} (overlap "
                        f"{r['depth'] * 1000:.1f} mm)")
                self._collision_red_key = red_key
            else:
                self.node.get_logger().debug(
                    f"collision_spheres: {len(reports)} sphere(s) still in "
                    "collision (unchanged)", throttle_duration_sec=5.0)
            return reports
        finally:
            if gpu_lock is not None:
                gpu_lock.release()

    def _attachment_kinematics(self):
        """The MotionPlanner's kinematics (carries attached-object spheres), or
        None if the planner/attachment_services isn't ready."""
        attach_svc = getattr(self.node, 'attachment_services', None)
        mp = getattr(self.node, 'motion_planner', None)
        if attach_svc is None or mp is None:
            return None
        return attach_svc.kinematics()
