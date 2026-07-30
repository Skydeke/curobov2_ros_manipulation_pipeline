"""
Benchmark parity test: run core and ROS on demo set and compare.

Tests all cuRobo capabilities exposed through the ROS wrapper:
1. Motion planning (plan_pose) -- the primary benchmark
2. Inverse kinematics (compute_ik)
3. Forward kinematics (compute_fk)
4. Collision checking (check_collision)

Each test compares direct-cuRobo results against ROS-service results.
"""

import os
import subprocess
import sys
import time

import pytest

_BENCHMARK_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..',
                 'curobo_core', 'curobo', 'benchmark')
)
if os.path.isdir(_BENCHMARK_DIR):
    sys.path.insert(0, _BENCHMARK_DIR)


HAS_CUDA = True
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False


def _ros_server_running():
    try:
        result = subprocess.run(
            ['ros2', 'action', 'list'],
            capture_output=True, text=True, timeout=5,
        )
        return 'cumotion/plan_motion' in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


NEEDS_SERVER = pytest.mark.skipif(
    not _ros_server_running(),
    reason='curobo_server_node not running',
)


@pytest.mark.skipif(not HAS_CUDA, reason='No CUDA-capable GPU available')
class TestBenchmarkParity:
    """Integration test: runs core and ROS on demo set, compares results."""

    @pytest.fixture(scope='class')
    def core_results(self):
        from isaac_ros_cumotion_benchmark.core_runner import run_core
        return run_core('demo')

    @pytest.fixture(scope='class')
    def ros_results(self):
        from isaac_ros_cumotion_benchmark.ros_runner import run_ros
        return run_ros('demo')

    def test_core_produces_results(self, core_results):
        assert len(core_results) > 0, 'core_runner returned no results'

    def test_ros_produces_results(self, ros_results):
        assert len(ros_results) > 0, 'ros_runner returned no results'

    def test_same_problem_count(self, core_results, ros_results):
        core_names = {r['problem_name'] for r in core_results}
        ros_names = {r['problem_name'] for r in ros_results}
        assert core_names == ros_names, (
            f'Problem sets differ. '
            f'Core only: {core_names - ros_names}. '
            f'ROS only: {ros_names - core_names}.'
        )

    def test_no_success_mismatches(self, core_results, ros_results):
        from isaac_ros_cumotion_benchmark.compare import compare
        report = compare(core_results, ros_results)
        assert report['success_mismatches'] == 0, (
            f'Found {report["success_mismatches"]} problems where '
            f'core.success != ros.success.\n'
            f'Mismatches: {[m for m in report["details"]["mismatches"]]}'
        )


@pytest.mark.skipif(not HAS_CUDA, reason='No CUDA-capable GPU available')
class TestCoreRunnerStandalone:
    """Core runner smoke tests -- no ROS needed."""

    def test_core_planning_on_demo(self):
        from isaac_ros_cumotion_benchmark.core_runner import run_core
        results = run_core('demo')
        assert len(results) > 0
        assert any(r['success'] for r in results), (
            'All demo planning problems failed -- is cuRobo working?'
        )

    def test_core_ik_on_demo(self):
        from isaac_ros_cumotion_benchmark.core_runner import run_core_ik
        results = run_core_ik('demo')
        assert len(results) > 0
        assert any(r['success'] for r in results), (
            'All demo IK problems failed -- is IK solver working?'
        )

    def test_core_fk_on_demo(self):
        from isaac_ros_cumotion_benchmark.core_runner import run_core_fk
        results = run_core_fk('demo')
        assert len(results) > 0
        assert all(r['success'] for r in results), (
            'FK should always succeed given valid joint configs'
        )

    def test_core_collision_on_demo(self):
        from isaac_ros_cumotion_benchmark.core_runner import run_core_collision
        results = run_core_collision('demo')
        assert len(results) > 0
        for r in results:
            assert 'in_collision' in r
            assert 'world_distance' in r
            assert 'self_distance' in r

    def test_core_ik_batch_faster_than_single(self):
        """Batched IK (SolveMode.BATCH) must be faster than solving each problem separately.
        
        N problems solved one-at-a-time (single) should take longer than
        solving all N in a single batched solve_pose call, because the GPU
        processes all N in parallel.
        """
        from isaac_ros_cumotion_benchmark.core_runner import (
            run_core_ik_single, run_core_ik_batched,
        )
        _, single_time = run_core_ik_single('demo', max_problems=4)
        _, batch_time = run_core_ik_batched('demo', max_problems=4)

        if single_time <= 0:
            pytest.skip('IK zero time — GPU not warmed up?')

        ratio = single_time / batch_time
        print(f'\n  Single total: {single_time:.4f}s  Batch total: {batch_time:.4f}s  '
              f'Speedup: {ratio:.2f}x')

        assert batch_time < single_time * 0.9, (
            f'Batched IK not faster than single: '
            f'single={single_time:.4f}s batch={batch_time:.4f}s '
            f'(batch should be ≤90% of single)'
        )


@pytest.mark.skipif(not HAS_CUDA, reason='No CUDA-capable GPU available')
@NEEDS_SERVER
class TestRosCapabilities:
    """Test all ROS-exposed cuRobo capabilities via services/actions."""

    def test_ros_ik_service(self):
        from isaac_ros_cumotion_benchmark.ros_runner import run_ros_ik
        results = run_ros_ik('demo')
        assert len(results) > 0, 'ROS IK returned no results'
        for r in results:
            assert 'success' in r
            assert 'position_error' in r
            assert 'rotation_error' in r

    def test_ros_fk_service(self):
        from isaac_ros_cumotion_benchmark.ros_runner import run_ros_fk
        results = run_ros_fk('demo')
        assert len(results) > 0, 'ROS FK returned no results'
        for r in results:
            assert r['success'], f'{r["problem_name"]} FK failed'

    def test_ros_collision_service(self):
        from isaac_ros_cumotion_benchmark.ros_runner import run_ros_collision
        results = run_ros_collision('demo')
        assert len(results) > 0, 'ROS collision returned no results'
        for r in results:
            assert 'in_collision' in r
            assert 'world_distance' in r

    def test_ros_ik_matches_core_success_rate(self):
        from isaac_ros_cumotion_benchmark.core_runner import run_core_ik
        from isaac_ros_cumotion_benchmark.ros_runner import run_ros_ik
        core = run_core_ik('demo')
        ros = run_ros_ik('demo')
        core_success = sum(1 for r in core if r['success'])
        ros_success = sum(1 for r in ros if r['success'])
        core_rate = core_success / len(core) if core else 0.0
        ros_rate = ros_success / len(ros) if ros else 0.0
        assert abs(core_rate - ros_rate) <= 0.2, (
            f'IK success rate mismatch: core={core_success}/{len(core)} '
            f'({core_rate:.0%}) vs ros={ros_success}/{len(ros)} '
            f'({ros_rate:.0%})'
        )

    def test_ros_fk_matches_core(self):
        from isaac_ros_cumotion_benchmark.core_runner import run_core_fk
        from isaac_ros_cumotion_benchmark.ros_runner import run_ros_fk
        core = run_core_fk('demo')
        ros = run_ros_fk('demo')
        assert len(core) == len(ros), (
            f'FK problem count mismatch: core={len(core)} ros={len(ros)}'
        )
        core_ok = sum(1 for r in core if r['success'])
        ros_ok = sum(1 for r in ros if r['success'])
        assert core_ok == ros_ok, (
            f'FK success mismatch: core={core_ok}/{len(core)} '
            f'ros={ros_ok}/{len(ros)}'
        )

    def test_ros_collision_matches_core(self):
        from isaac_ros_cumotion_benchmark.core_runner import run_core_collision
        from isaac_ros_cumotion_benchmark.ros_runner import run_ros_collision
        core = run_core_collision('demo')
        ros = run_ros_collision('demo')
        assert len(core) == len(ros), (
            f'Collision problem count mismatch: core={len(core)} ros={len(ros)}'
        )


@NEEDS_SERVER
class TestRosRunnerConnectivity:
    """Smoke test: ROS client connects to a running curobo_server_node."""

    def test_action_server_available(self):
        import rclpy
        rclpy.init()
        try:
            from isaac_ros_cumotion_benchmark.ros_runner import RosBenchmarkRunner
            runner = RosBenchmarkRunner(time_dilation_factor=1.0)
            assert runner._action_client.server_is_ready()
            runner.destroy_node()
        finally:
            rclpy.shutdown()


@NEEDS_SERVER
class TestPlanningSceneObstacles:
    """Verify obstacles published to /planning_scene affect collision checking.

    Publishes a large blocking obstacle via ``PlanningScene``, asserts that
    ``PlanMotion`` fails, then removes the obstacle and asserts it succeeds.
    """

    @pytest.fixture(scope='class')
    def runner(self):
        import rclpy
        rclpy.init()
        from isaac_ros_cumotion_benchmark.ros_runner import RosBenchmarkRunner
        r = RosBenchmarkRunner(time_dilation_factor=1.0)
        yield r
        r.destroy_node()
        rclpy.shutdown()

    @pytest.fixture(scope='class')
    def problem(self):
        """First demo problem that is valid (collision_buffer_ik >= 0)."""
        from isaac_ros_cumotion_benchmark.problems import load_problems
        scene_problems = load_problems('demo')
        for problems in scene_problems.values():
            for p in problems:
                if p.get('collision_buffer_ik', 0.0) >= 0.0:
                    return p
        pytest.skip('No valid demo problem found')

    def _publish_planning_scene(self, runner, collision_objects, is_diff=True):
        from moveit_msgs.msg import PlanningScene, CollisionObject
        pub = runner.create_publisher(PlanningScene, '/planning_scene', 1)
        msg = PlanningScene()
        msg.is_diff = is_diff
        msg.world.collision_objects = collision_objects
        pub.publish(msg)
        runner.get_logger().info(
            f'Published {len(collision_objects)} object(s) to /planning_scene'
        )

    def _make_blocking_box(self, runner, position, size=1.0):
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose as RosPose
        obj = CollisionObject()
        obj.id = 'test_blocking_box'
        obj.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [size, size, size]
        pose = RosPose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.w = 1.0
        obj.primitives = [prim]
        obj.primitive_poses = [pose]
        return obj

    def test_motion_planning_blocked_by_obstacle(self, runner, problem):
        t0 = time.time()
        start_q = problem['start']
        gp = problem['goal_pose']
        pose_flat = gp['position_xyz'] + gp['quaternion_wxyz']

        # 1. Plan without obstacles — should succeed
        result_clean = runner._plan_motion(start_q, pose_flat)
        runner.get_logger().info(
            f'Clean plan: success={result_clean["success"]} '
            f'time={result_clean["time_s"]:.3f}s'
        )

        # 2. Publish a large obstacle at the goal via /planning_scene
        blocking_pos = gp['position_xyz']
        box = self._make_blocking_box(runner, blocking_pos, size=0.4)
        self._publish_planning_scene(runner, [box])
        time.sleep(0.5)

        # 3. Same plan with obstacle — should fail
        result_blocked = runner._plan_motion(start_q, pose_flat)
        runner.get_logger().info(
            f'Blocked plan: success={result_blocked["success"]} '
            f'time={result_blocked["time_s"]:.3f}s'
        )

        # 4. Remove the obstacle via /planning_scene
        from moveit_msgs.msg import CollisionObject
        remove_msg = box
        remove_msg.operation = CollisionObject.REMOVE
        self._publish_planning_scene(runner, [remove_msg])
        time.sleep(0.5)

        # 5. Plan again — should succeed again
        result_restored = runner._plan_motion(start_q, pose_flat)
        runner.get_logger().info(
            f'Restored plan: success={result_restored["success"]} '
            f'time={result_restored["time_s"]:.3f}s'
        )
        dt = time.time() - t0
        runner.get_logger().info(f'Test completed in {dt:.1f}s')

        # Clean up any leftover obstacle
        self._publish_planning_scene(runner, [remove_msg])

        assert result_clean['success'], (
            'Clean plan (no obstacles) should succeed'
        )
        assert not result_blocked['success'], (
            f'Plan blocked by obstacle at {blocking_pos} should fail '
            f'but got success={result_blocked["success"]}'
        )
        assert result_restored['success'], (
            'Plan after removing obstacle should succeed again'
        )
