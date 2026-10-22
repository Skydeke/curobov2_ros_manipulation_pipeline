"""Unit tests for the build_curobo_config CLI routing (no GPU/ROS needed)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from test_curobo_config_fakes import build_parser, collision_spheres_to_bubblify_data

class TestParser(unittest.TestCase):
    """Parser defaults."""

    def test_create_defaults(self):
        """Create-mode flags keep their historical defaults."""
        args = build_parser().parse_args(["--urdf", "r.urdf",
                                          "--asset-path", "/a"])
        self.assertEqual(args.tool_frame, "grasping_frame")
        self.assertEqual(args.sphere_density, 1.0)
        self.assertEqual(args.num_collision_samples, 1000)
        self.assertEqual(args.refit_link, [])
        self.assertFalse(args.with_bubblify)

    def test_bubblify_data_conversion(self):
        """Sphere conversion keeps links and counts in metadata."""
        spheres = {"b": [{"center": [0.0, 0.0, 0.1], "radius": 0.08}],
                   "a": [{"center": [0.0, 0.0, 0.0], "radius": 0.05},
                         {"center": [0.1, 0.0, 0.0], "radius": 0.05}]}
        data = collision_spheres_to_bubblify_data(spheres)
        self.assertEqual(data["collision_spheres"], spheres)
        self.assertEqual(data["metadata"]["total_spheres"], 3)
        self.assertEqual(data["metadata"]["links"], ["a", "b"])
