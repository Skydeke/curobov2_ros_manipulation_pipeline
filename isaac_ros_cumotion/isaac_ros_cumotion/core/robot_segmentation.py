import os

import torch

from isaac_ros_cumotion.cameras.camera_context import CameraContext


class RobotSegmentation:
    """Removes the robot (and any user-defined mask shapes) from the raw depth
    streams before they reach the perception Mapper, so the arm and its mounting
    never become ESDF voxels.

    This is a composable component, NOT a standalone node: it is owned and
    constructed by the UnifiedPlannerNode and runs in that node's process. It
    reuses the node's RobotContext (joint state) and Kinematics (FK for the
    collision spheres) and runs ONE RobotSegmentationCameraStrategy per camera
    whose purpose is 'all' or 'segmentation' — registered through the shared
    CameraContext.add_camera just like the mapper's camera strategies, so a
    segmented camera IS a camera strategy. Each strategy subscribes to that
    camera's raw depth and camera-info topic and republishes the masked image
    on a topic DERIVED from the camera's own raw topic (its leaf segment is
    replaced by ``masked_depth``, e.g. ``/kortex_vision/depth/image`` ->
    ``/kortex_vision/depth/masked_depth``), so every segmented camera gets a
    unique, per-camera output with no separate configuration. The component is
    gated by the node's ``enable_robot_segmentation`` parameter; the
    set_mask / remove_mask services are registered by the RosServiceManager and
    act on a masks dict SHARED (by reference) across all streams.
    """

    def __init__(self, node, robot_context, kin_model, camera_cfgs=None,
                 base_frame=None, ops_dtype=None, device=None):
        self._node = node
        self.robot_context = robot_context
        self._kin_model = kin_model

        # Device/dtype come from the shared config (the node's `tensor_args`),
        # never hardcoded to 'cuda'/float32 here.
        node_tensor_args = getattr(node, 'tensor_args', None)
        self._ops_dtype = ops_dtype or getattr(node_tensor_args, 'dtype',
                                               torch.float32)
        if device is not None:
            self._device = device
        elif getattr(node_tensor_args, 'device', None) is not None:
            self._device = torch.device(node_tensor_args.device)
        else:
            self._device = (torch.device('cuda') if torch.cuda.is_available()
                            else torch.device('cpu'))

        # Segmenter-owned tunables (the masked-output topic is DERIVED per
        # camera from the shared camera config, not a segmenter param).
        # Declared at node level too, so launch overrides apply at node
        # construction; the has_parameter guard also covers nodes that already
        # declared them (e.g. via plugin).
        # Inflation added to every mask shape's half-extents / radius.
        if not node.has_parameter('robot_segmentation_mask_margin'):
            node.declare_parameter('robot_segmentation_mask_margin', 0.0)
        # Minimum distance (m) to a robot collision sphere for a depth point
        # to be kept (i.e. considered NOT part of the robot).
        if not node.has_parameter('robot_segmentation_distance_threshold'):
            node.declare_parameter('robot_segmentation_distance_threshold', 0.05)

        self._robot_base_frame = base_frame or self._fallback_base_frame()
        mask_margin = self._get_double('robot_segmentation_mask_margin')
        distance_threshold = self._get_double(
            'robot_segmentation_distance_threshold')

        # User-defined extra masks (e.g. a grasped object), keyed by name. Each
        # rides a TF frame so it follows the arm (see set_mask_callback). A mask
        # applies to EVERY stream: its world position doesn't depend on which
        # camera sees it. Shared (by reference) with every strategy.
        self._masks = {}

        self.camera_context = CameraContext(node)
        for cfg in camera_cfgs or []:
            self.camera_context.add_camera(
                camera_name=cfg.name,
                camera_type='robot_segmentation',
                topic=cfg.depth_topic,
                camera_info=cfg.camera_info_topic,
                frame_id=cfg.frame_id,
                intrinsics=cfg.intrinsics,
                extrinsics=cfg.extrinsics,
                robot_context=self.robot_context,
                kin_model=self._kin_model,
                base_frame=self._robot_base_frame,
                ops_dtype=self._ops_dtype,
                device=self._device,
                distance_threshold=distance_threshold,
                mask_margin=mask_margin,
                masks=self._masks,
            )
        for cfg in camera_cfgs or []:
            strategy = self.camera_context.cameras.get(cfg.name)
            if strategy is None:
                continue
            self._node.get_logger().info(
                f"RobotSegmentation stream for camera '{cfg.name}' "
                f"(index {cfg.camera_index}): "
                f"{cfg.depth_topic} -> {strategy.output_topic or cfg.depth_topic}")

        # The set_mask / remove_mask services are registered by the
        # RosServiceManager via register_robot_segmentation().

    @property
    def streams(self):
        """The per-camera masking strategies."""
        return list(self.camera_context.cameras.values())

    def _get_double(self, name):
        return self._node.get_parameter(name).get_parameter_value().double_value

    def _fallback_base_frame(self):
        """Base frame for the robot-distance test when none is injected.

        Prefers the node's shared 'base_link' parameter (ConfigManager's
        configurable base frame); falls back to 'base_link' only if the node
        has no such parameter.
        """
        if self._node.has_parameter('base_link'):
            try:
                return self._node.get_parameter(
                    'base_link').get_parameter_value().string_value
            except Exception:
                pass
        return 'base_link'

    def destroy(self):
        for strategy in self.streams:
            if hasattr(strategy, 'destroy'):
                strategy.destroy()

    # ---- Mask services ----

    def set_mask_callback(self, request, response):
        """Add or update a named mask shape (SetMask.srv).

        Dimension conventions mirror obstacle_manager.add_object:
        CUBOID [dx,dy,dz], SPHERE [r,_,_], CAPSULE [r,h,_] (segment
        [0,0,0]->[0,0,h]), CYLINDER [r,h,_] (centered, axis z), MESH [sx,sy,sz].
        """
        d = request.dimensions
        if request.type == 4:
            if not os.path.exists(request.mesh_file_path):
                response.success = False
                response.message = f'Mesh file not found: {request.mesh_file_path}'
                return response
        elif d.x <= 0 or d.y <= 0 or d.z <= 0:
            response.success = False
            response.message = 'Mask dimensions must be positive'
            return response

        p = request.pose.position
        o = request.pose.orientation
        self._masks[request.name] = {
            'type': request.type,
            'frame_id': request.frame_id,
            'mesh_file_path': request.mesh_file_path,
            'pos': torch.tensor([p.x, p.y, p.z],
                                dtype=self._ops_dtype, device=self._device),
            'quat': [o.x, o.y, o.z, o.w],  # scipy order
            'dims': [d.x, d.y, d.z],
        }
        frame = request.frame_id or self._robot_base_frame
        response.success = True
        response.message = (f"Mask '{request.name}' set "
                            f"(type={request.type}, frame='{frame}')")
        self._node.get_logger().info(response.message)
        return response

    def remove_mask_callback(self, request, response):
        """Remove a named mask (RemoveObject.srv). Idempotent."""
        if request.name in self._masks:
            del self._masks[request.name]
            response.success = True
            response.message = f"Mask '{request.name}' removed"
        else:
            response.success = True
            response.message = f"Mask '{request.name}' not present (nothing to remove)"
        self._node.get_logger().info(response.message)
        return response