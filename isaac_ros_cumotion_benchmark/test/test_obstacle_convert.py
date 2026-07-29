"""Round-trip CollisionObject through cuRobo scene objects and back."""

from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

from isaac_ros_cumotion_benchmark.obstacle_convert import (
    _cuboid_dict_to_collision_object,
    _cylinder_dict_to_collision_object,
    _sphere_dict_to_collision_object,
    obstacles_dict_to_collision_objects,
    scene_objects_to_collision_objects,
)


class TestCuboidConversion:
    def test_basic_cuboid(self):
        data = {'pose': [0.5, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0], 'dims': [0.4, 0.3, 0.2]}
        co = _cuboid_dict_to_collision_object('test_cube', data)
        assert co.id == 'test_cube'
        assert co.header.frame_id == 'world'
        assert co.operation == CollisionObject.ADD
        assert len(co.primitives) == 1
        assert co.primitives[0].type == SolidPrimitive.BOX
        assert list(co.primitives[0].dimensions) == [0.4, 0.3, 0.2]
        assert co.pose.position.x == 0.5
        assert co.pose.position.y == 0.0
        assert co.pose.position.z == 0.25
        assert co.pose.orientation.w == 1.0

    def test_primitive_pose_is_identity(self):
        data = {'pose': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'dims': [1, 1, 1]}
        co = _cuboid_dict_to_collision_object('id', data)
        pp = co.primitive_poses[0]
        assert pp.position.x == 0.0
        assert pp.position.y == 0.0
        assert pp.position.z == 0.0
        assert pp.orientation.w == 1.0

    def test_custom_frame_id(self):
        data = {'pose': [0, 0, 0, 1, 0, 0, 0], 'dims': [1, 1, 1]}
        co = _cuboid_dict_to_collision_object('f', data, frame_id='panda_hand')
        assert co.header.frame_id == 'panda_hand'


class TestCylinderConversion:
    def test_basic_cylinder(self):
        data = {'pose': [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], 'height': 0.5, 'radius': 0.2}
        co = _cylinder_dict_to_collision_object('test_cyl', data)
        assert co.id == 'test_cyl'
        assert len(co.primitives) == 1
        assert co.primitives[0].type == SolidPrimitive.CYLINDER
        assert list(co.primitives[0].dimensions) == [0.5, 0.2]

    def test_cylinder_pose(self):
        data = {'pose': [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], 'height': 1.0, 'radius': 0.5}
        co = _cylinder_dict_to_collision_object('cyl', data)
        assert co.pose.position.x == 1.0
        assert co.pose.position.y == 2.0
        assert co.pose.position.z == 3.0


class TestSphereConversion:
    def test_basic_sphere(self):
        data = {'pose': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], 'radius': 0.3}
        co = _sphere_dict_to_collision_object('test_sph', data)
        assert co.id == 'test_sph'
        assert len(co.primitives) == 1
        assert co.primitives[0].type == SolidPrimitive.SPHERE
        assert list(co.primitives[0].dimensions) == [0.3]

    def test_sphere_pose(self):
        data = {'pose': [0.5, 0.5, 0.5, 0.707, 0.0, 0.707, 0.0], 'radius': 0.1}
        co = _sphere_dict_to_collision_object('sph', data)
        assert co.pose.orientation.x == 0.0
        assert co.pose.orientation.y == 0.707


class TestObstaclesDictConversion:
    def test_empty_dict(self):
        result = obstacles_dict_to_collision_objects({})
        assert result == []

    def test_dict_with_no_known_keys(self):
        result = obstacles_dict_to_collision_objects({'unknown_type': {}})
        assert result == []

    def test_mixed_obstacles(self):
        obstacles = {
            'cuboid': {
                'box1': {'pose': [0, 0, 0, 1, 0, 0, 0], 'dims': [1, 1, 1]},
            },
            'cylinder': {
                'cyl1': {'pose': [1, 0, 0, 1, 0, 0, 0], 'height': 2, 'radius': 0.5},
            },
            'sphere': {
                'sph1': {'pose': [0, 1, 0, 1, 0, 0, 0], 'radius': 0.3},
            },
        }
        result = obstacles_dict_to_collision_objects(obstacles)
        assert len(result) == 3
        ids = {co.id for co in result}
        assert ids == {'box1', 'cyl1', 'sph1'}
        types = {co.primitives[0].type for co in result}
        assert types == {SolidPrimitive.BOX, SolidPrimitive.CYLINDER, SolidPrimitive.SPHERE}

    def test_custom_frame_id(self):
        obstacles = {
            'cuboid': {'b': {'pose': [0, 0, 0, 1, 0, 0, 0], 'dims': [1, 1, 1]}},
        }
        result = obstacles_dict_to_collision_objects(obstacles, frame_id='odom')
        assert result[0].header.frame_id == 'odom'


class TestRoundTrip:
    """Round-trip CollisionObject through cuRobo scene objects and back."""

    def _check_conversions_importable(self):
        try:
            from isaac_ros_cumotion.curobo_server.conversions import (
                collision_object_to_scene_objects,
            )
            return collision_object_to_scene_objects is not None
        except ModuleNotFoundError:
            return False

    def _roundtrip_cuboid(self, name, pose, dims):
        from isaac_ros_cumotion.curobo_server.conversions import (
            collision_object_to_scene_objects,
        )

        data = {'pose': pose, 'dims': dims}
        co = _cuboid_dict_to_collision_object(name, data)
        scene_objs, ok = collision_object_to_scene_objects(co)
        assert ok, 'conversion reported not-ok'
        assert len(scene_objs) == 1
        result = scene_objects_to_collision_objects(scene_objs)
        assert len(result) == 1
        return result[0]

    def test_cuboid_roundtrip(self):
        if not self._check_conversions_importable():
            return
        co2 = self._roundtrip_cuboid(
            'rt_box', [0.5, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0], [0.4, 0.3, 0.2]
        )
        assert co2.primitives[0].type == SolidPrimitive.BOX
        for d_in, d_out in zip([0.4, 0.3, 0.2], list(co2.primitives[0].dimensions)):
            assert abs(d_in - d_out) < 1e-6

    def test_cylinder_roundtrip(self):
        if not self._check_conversions_importable():
            return
        from isaac_ros_cumotion.curobo_server.conversions import (
            collision_object_to_scene_objects,
        )

        data = {'pose': [0, 0, 1, 1, 0, 0, 0], 'height': 0.8, 'radius': 0.15}
        co = _cylinder_dict_to_collision_object('rt_cyl', data)
        scene_objs, ok = collision_object_to_scene_objects(co)
        assert ok
        result = scene_objects_to_collision_objects(scene_objs)
        assert len(result) == 1
        assert result[0].primitives[0].type == SolidPrimitive.CYLINDER
        assert abs(result[0].primitives[0].dimensions[0] - 0.8) < 1e-6
        assert abs(result[0].primitives[0].dimensions[1] - 0.15) < 1e-6

    def test_sphere_roundtrip(self):
        if not self._check_conversions_importable():
            return
        from isaac_ros_cumotion.curobo_server.conversions import (
            collision_object_to_scene_objects,
        )

        data = {'pose': [0, 0, 0.5, 1, 0, 0, 0], 'radius': 0.25}
        co = _sphere_dict_to_collision_object('rt_sph', data)
        scene_objs, ok = collision_object_to_scene_objects(co)
        assert ok
        result = scene_objects_to_collision_objects(scene_objs)
        assert len(result) == 1
        assert result[0].primitives[0].type == SolidPrimitive.SPHERE
        assert abs(result[0].primitives[0].dimensions[0] - 0.25) < 1e-6
