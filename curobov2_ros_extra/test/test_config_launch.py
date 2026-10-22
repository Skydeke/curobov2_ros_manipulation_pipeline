"""Unit tests for the build_curobo_config CLI routing (no GPU/ROS needed)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from test_curobo_config_fakes import FakeBuilder, FakeDropdown, FakeGui, FakeGuiHandle, FakeStore, install_fake_curobo

class TestLaunchWiring(unittest.TestCase):
    """Save button and exit-save with a stubbed bubblify app."""

    def setUp(self):
        """Stub bubblify.gui and the curobo fit-type import."""
        install_fake_curobo()
        for name in ("curobo._src", "curobo._src.geom",
                     "curobo._src.geom.sphere_fit",
                     "curobo._src.geom.sphere_fit.types"):
            sys.modules[name] = ModuleType(name)
        sys.modules["curobo._src.geom.sphere_fit.types"].SphereFitType = str

        import yaml

        class FakeBubblifyApp:
            """Stub app preloading the spherization yml."""

            instances = []
            drop_first_on_run = False
            press_save_on_run = False
            save_dest = None

            def __init__(self, **kwargs):
                """Load spheres, build stub gui."""
                from curobov2_ros_extra.morphit_bubblify import (
                    SPHERE_EDITOR_FOLDER,
                )
                self.sphere_store = FakeStore()
                with open(kwargs["spherization_yml"], "r") as f:
                    data = yaml.safe_load(f)
                for link, entries in data.get(
                        "collision_spheres", {}).items():
                    for e in entries:
                        self.sphere_store.add(
                            link, xyz=tuple(e["center"]),
                            radius=float(e["radius"]))
                self.server = SimpleNamespace(gui=FakeGui())
                folder = FakeGuiHandle(label=SPHERE_EDITOR_FOLDER)
                self.server.gui._container_handle_from_uuid = {
                    "folder1": folder}
                self._folder = folder
                self._link_dropdown = FakeDropdown()
                self._link_dropdown.value = "link_a"
                self._current_link_dropdown = FakeDropdown()
                self._current_link_dropdown.value = "link_a"
                self._sphere_dropdown = FakeDropdown()
                self.current_link = "link_a"
                self.current_sphere_id = None
                self.visualized = []
                self.updated = []
                FakeBubblifyApp.instances.append(self)

            def _create_sphere_visualization(self, sphere):
                """Record visuals."""
                sphere.node = "node"
                self.visualized.append(sphere.id)

            def _sync_link_selection(self):
                """Record sync."""
                self.updated.append("sync_link")

            def _update_transform_control(self):
                """Record UI refresh."""
                self.updated.append("transform")

            def _update_sphere_opacities(self):
                """Record UI refresh."""
                self.updated.append("opacities")

            def _update_sphere_properties_ui(self):
                """Record UI refresh."""
                self.updated.append("props")

            def run(self):
                """Simulate an edit session, then exit."""
                if FakeBubblifyApp.drop_first_on_run:
                    first = self.sphere_store.get_spheres_for_link("link_a")
                    if first:
                        self.sphere_store.remove(first[0].id)
                if FakeBubblifyApp.save_dest:
                    for label, handle in self.server.gui.texts.items():
                        if label.startswith("Save cuRobo"):
                            handle.value = FakeBubblifyApp.save_dest
                if FakeBubblifyApp.press_save_on_run:
                    for btn in self.server.gui.buttons.values():
                        if btn.label.startswith("Save"):
                            for cb in btn.callbacks:
                                cb(None)

            @classmethod
            def reset(cls):
                """Clear session flags."""
                cls.instances = []
                cls.drop_first_on_run = False
                cls.press_save_on_run = False
                cls.save_dest = None

        gui_mod = ModuleType("bubblify.gui")
        gui_mod.BubblifyApp = FakeBubblifyApp
        pkg_mod = ModuleType("bubblify")
        pkg_mod.gui = gui_mod
        sys.modules["bubblify"] = pkg_mod
        sys.modules["bubblify.gui"] = gui_mod
        FakeBubblifyApp.reset()
        self.FakeBubblifyApp = FakeBubblifyApp

    def tearDown(self):
        """Remove all stubs."""
        for name in ("bubblify.gui", "bubblify", "curobo.robot_builder",
                     "curobo._src.geom.sphere_fit.types",
                     "curobo._src.geom.sphere_fit", "curobo._src.geom",
                     "curobo._src", "curobo"):
            sys.modules.pop(name, None)

    def run_launch(self, builder, output):
        """Run the real launch function against the stub app."""
        import curobov2_ros_extra.build_curobo_config as mod
        spheres = builder.collision_spheres or {}
        yml = mod.write_bubblify_spherization(spheres)
        try:
            from curobov2_ros_extra.morphit_bubblify import (
                launch_morphit_bubblify,
            )
            save_fn = mod._make_save_fn(
                builder, output, {"lock_joints": {"j": 1.0}},
                SimpleNamespace(skip_matrix=False, num_collision_samples=10,
                                matrix_batch_size=100))
            return launch_morphit_bubblify(
                builder, "/tmp/fake.urdf", yml, 8080, False,
                {"fit_type": "morphit", "iterations": 200,
                 "coverage_weight": None, "protrusion_weight": None,
                 "sphere_density": 1.0}, save_fn, output)
        finally:
            os.unlink(yml)

    def test_exit_saves_edits(self):
        """Unclean exit still writes the edited spheres to the file."""
        FakeBuilder.last_instance = None
        builder = FakeBuilder()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml",
                                         delete=False) as f:
            output = f.name
        try:
            self.FakeBubblifyApp.drop_first_on_run = True
            changed = self.run_launch(builder, output)
        finally:
            os.unlink(output)
        self.assertTrue(changed)
        self.assertEqual(len(builder.saved), 1)
        config, path = builder.saved[0]
        self.assertEqual(path, output)
        self.assertEqual(config.lock_joints, {"j": 1.0})
        self.assertEqual(config.collision_spheres, {"link_a": []})
        kinds = [c[0] for c in builder.calls]
        self.assertIn("matrix", kinds)

    def test_no_changes_no_save(self):
        """A view-only session writes nothing when output exists."""
        builder = FakeBuilder()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml",
                                         delete=False) as f:
            output = f.name
        try:
            changed = self.run_launch(builder, output)
        finally:
            os.unlink(output)
        self.assertFalse(changed)
        self.assertEqual(builder.saved, [])

    def test_save_button_no_changes(self):
        """Save with no edits reports it and writes nothing."""
        builder = FakeBuilder()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml",
                                         delete=False) as f:
            output = f.name
        try:
            self.FakeBubblifyApp.press_save_on_run = True
            changed = self.run_launch(builder, output)
        finally:
            os.unlink(output)
        self.assertFalse(changed)
        self.assertEqual(builder.saved, [])
        app = self.FakeBubblifyApp.instances[-1]
        self.assertIn("No changes since last save",
                      app.server.gui.texts["MorphIt status"].value)

    def test_save_button_with_changes(self):
        """Save writes once; exit does not duplicate it."""
        builder = FakeBuilder()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml",
                                         delete=False) as f:
            output = f.name
        try:
            self.FakeBubblifyApp.drop_first_on_run = True
            self.FakeBubblifyApp.press_save_on_run = True
            changed = self.run_launch(builder, output)
        finally:
            os.unlink(output)
        self.assertTrue(changed)
        self.assertEqual(len(builder.saved), 1)
        app = self.FakeBubblifyApp.instances[-1]
        self.assertTrue(
            app.server.gui.texts["MorphIt status"].value.startswith("Saved"))

    def test_save_button_uses_edited_dest(self):
        """An edited path field redirects the save."""
        builder = FakeBuilder()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml",
                                         delete=False) as f:
            output = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml",
                                         delete=False) as f:
            new_dest = f.name
        try:
            self.FakeBubblifyApp.drop_first_on_run = True
            self.FakeBubblifyApp.press_save_on_run = True
            self.FakeBubblifyApp.save_dest = new_dest
            changed = self.run_launch(builder, output)
        finally:
            os.unlink(output)
            os.unlink(new_dest)
        self.assertTrue(changed)
        self.assertEqual(len(builder.saved), 1)
        config, path = builder.saved[0]
        self.assertEqual(path, new_dest)
        self.assertEqual(config.collision_spheres, {"link_a": []})
