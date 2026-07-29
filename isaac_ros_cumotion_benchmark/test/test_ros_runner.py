from isaac_ros_cumotion_benchmark.ros_runner import FRANKA_JOINT_NAMES


class TestFrankaJointNames:
    def test_has_seven_joints(self):
        assert len(FRANKA_JOINT_NAMES) == 7

    def test_all_start_with_panda_joint(self):
        for name in FRANKA_JOINT_NAMES:
            assert name.startswith('panda_joint')

    def test_numbered_one_to_seven(self):
        expected = [f'panda_joint{i}' for i in range(1, 8)]
        assert FRANKA_JOINT_NAMES == expected
