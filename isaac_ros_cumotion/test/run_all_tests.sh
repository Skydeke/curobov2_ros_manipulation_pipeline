#!/bin/bash
# Auto-generated test runner for launch_test
# DO NOT EDIT - Changes will be overwritten

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}"

echo "========================================="
echo "Running all launch_test tests"
echo "========================================="
echo ""

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
FAILED_TEST_LIST=()


echo "Running test_test_collision_franka..."
if timeout 690 launch_test "${TEST_DIR}/test_test_collision_franka.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_collision_franka PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_collision_franka FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_collision_franka")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_collision_ur10e..."
if timeout 690 launch_test "${TEST_DIR}/test_test_collision_ur10e.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_collision_ur10e PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_collision_ur10e FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_collision_ur10e")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_kinematics_franka..."
if timeout 350 launch_test "${TEST_DIR}/test_test_kinematics_franka.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_kinematics_franka PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_kinematics_franka FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_kinematics_franka")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_kinematics_ur10e..."
if timeout 350 launch_test "${TEST_DIR}/test_test_kinematics_ur10e.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_kinematics_ur10e PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_kinematics_ur10e FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_kinematics_ur10e")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_mesh_obstacle_franka..."
if timeout 600 launch_test "${TEST_DIR}/test_test_mesh_obstacle_franka.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_mesh_obstacle_franka PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_mesh_obstacle_franka FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_mesh_obstacle_franka")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_mesh_obstacle_ur10e..."
if timeout 600 launch_test "${TEST_DIR}/test_test_mesh_obstacle_ur10e.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_mesh_obstacle_ur10e PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_mesh_obstacle_ur10e FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_mesh_obstacle_ur10e")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_object_franka..."
if timeout 540 launch_test "${TEST_DIR}/test_test_object_franka.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_object_franka PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_object_franka FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_object_franka")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_object_ur10e..."
if timeout 540 launch_test "${TEST_DIR}/test_test_object_ur10e.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_object_ur10e PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_object_ur10e FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_object_ur10e")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_planners_franka..."
if timeout 680 launch_test "${TEST_DIR}/test_test_planners_franka.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_planners_franka PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_planners_franka FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_planners_franka")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_planners_ur10e..."
if timeout 680 launch_test "${TEST_DIR}/test_test_planners_ur10e.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_planners_ur10e PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_planners_ur10e FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_planners_ur10e")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_robot_strategy_franka..."
if timeout 330 launch_test "${TEST_DIR}/test_test_robot_strategy_franka.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_robot_strategy_franka PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_robot_strategy_franka FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_robot_strategy_franka")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_robot_strategy_ur10e..."
if timeout 330 launch_test "${TEST_DIR}/test_test_robot_strategy_ur10e.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_robot_strategy_ur10e PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_robot_strategy_ur10e FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_robot_strategy_ur10e")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_trajectory_franka..."
if timeout 350 launch_test "${TEST_DIR}/test_test_trajectory_franka.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_trajectory_franka PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_trajectory_franka FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_trajectory_franka")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "Running test_test_trajectory_ur10e..."
if timeout 350 launch_test "${TEST_DIR}/test_test_trajectory_ur10e.py" > /dev/null 2>&1; then
    echo "  ✓ test_test_trajectory_ur10e PASSED"
    ((PASSED_TESTS++)) || true
else
    echo "  ✗ test_test_trajectory_ur10e FAILED"
    ((FAILED_TESTS++)) || true
    FAILED_TEST_LIST+=("test_test_trajectory_ur10e")
fi
((TOTAL_TESTS++)) || true
echo ""

echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total tests:  $TOTAL_TESTS"
echo "Passed:       $PASSED_TESTS"
echo "Failed:       $FAILED_TESTS"

if [ $FAILED_TESTS -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for test in "${FAILED_TEST_LIST[@]}"; do
        echo "  - $test"
    done
    exit 1
fi

echo ""
echo "All tests passed! ✓"
exit 0
