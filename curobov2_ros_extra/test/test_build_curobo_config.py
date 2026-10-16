"""Unit tests for the build_curobo_config CLI routing (no GPU/ROS needed)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from curobov2_ros_extra.build_curobo_config import (  # noqa: E402
    build_parser,
    collision_spheres_to_bubblify_data,
)


class FakeBuilder:
    """Minimal RobotBuilder stand-in recording calls."""

    last_instance = None

    def __init__(self, *args, **kwargs):
        """Record constructor args and default sphere state."""
        self.init_args = args
        self.init_kwargs = kwargs
        self.urdf_path = kwargs.get("urdf_path", "/tmp/fake.urdf")
        self.asset_path = kwargs.get("asset_path", "/tmp/assets")
        self.collision_spheres = {"link_a": [{"center": [0.0, 0.0, 0.0],
                                              "radius": 0.05}]}
        self.calls = []
        self.saved = []
        FakeBuilder.last_instance = self

    @classmethod
    def from_config(cls, config_path, device_cfg=None):
        """Record from_config and return an instance."""
        inst = cls()
        inst.calls.append(("from_config", config_path))
        return inst

    def fit_collision_spheres(self, **kwargs):
        """Record a full refit."""
        self.calls.append(("fit_all", kwargs))

    def refit_link_spheres(self, link_name, **kwargs):
        """Record a single-link refit."""
        self.calls.append(("refit_link", link_name, kwargs))
        return self.collision_spheres.get(link_name, [])

    def compute_collision_matrix(self, **kwargs):
        """Record a matrix recompute."""
        self.calls.append(("matrix", kwargs))
        return {}

    def build(self):
        """Record build and return a dummy config."""
        self.calls.append(("build",))
        ns = SimpleNamespace()
        ns.collision_spheres = getattr(self, "_collision_spheres",
                                       self.collision_spheres)
        return ns

    def save(self, config, output_path):
        """Record save."""
        self.calls.append(("save", output_path))
        self.saved.append((config, output_path))

    def visualize(self, config, port=8080):
        """Record visualize."""
        self.calls.append(("visualize", port))


def install_fake_curobo():
    """Stub curobo.robot_builder so main() needs no GPU install."""
    curobo = ModuleType("curobo")
    sub = ModuleType("curobo.robot_builder")
    sub.RobotBuilder = FakeBuilder
    sys.modules["curobo"] = curobo
    sys.modules["curobo.robot_builder"] = sub


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


class TestEditMode(unittest.TestCase):
    """Edit-mode routing with a stubbed RobotBuilder."""

    def setUp(self):
        """Install the curobo stub and an original-yml fixture."""
        install_fake_curobo()
        FakeBuilder.last_instance = None
        import curobov2_ros_extra.build_curobo_config as mod
        self._orig_patch = patch.object(
            mod, "load_original_kinematics",
            return_value={"base_link": "base_link",
                          "lock_joints": {"wheel_joint": 0.0},
                          "collision_spheres": {"stale": []}})
        self._orig_patch.start()

    def tearDown(self):
        """Remove the curobo stub."""
        self._orig_patch.stop()
        sys.modules.pop("curobo.robot_builder", None)
        sys.modules.pop("curobo", None)

    def run_main(self, argv):
        """Import main lazily (after stubbing) and run it."""
        from curobov2_ros_extra.build_curobo_config import main
        main(argv)

    def test_needs_exactly_one_source(self):
        """Passing both or neither source exits with code 2."""
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(["--urdf", "a", "--config", "b"])
        self.assertEqual(ctx.exception.code, 2)
        with self.assertRaises(SystemExit) as ctx:
            self.run_main([])
        self.assertEqual(ctx.exception.code, 2)

    def test_edit_load_save_defaults(self):
        """Plain edit loads via from_config and saves back to input."""
        self.run_main(["--config", "in.yml"])
        inst = FakeBuilder.last_instance
        kinds = [c[0] for c in inst.calls]
        self.assertIn("from_config", kinds)
        self.assertNotIn("fit_all", kinds)
        self.assertNotIn("matrix", kinds)
        self.assertIn(("save", "in.yml"), inst.calls)

    def test_edit_refit_link_recomputes_matrix(self):
        """Refitting a link implies a matrix recompute and honors --output."""
        self.run_main(["--config", "in.yml", "--refit-link", "link_a",
                       "--output", "out.yml"])
        inst = FakeBuilder.last_instance
        kinds = [c[0] for c in inst.calls]
        self.assertIn("refit_link", kinds)
        self.assertIn("matrix", kinds)
        self.assertIn(("save", "out.yml"), inst.calls)

    def test_edit_skip_matrix(self):
        """--skip-matrix keeps a stale matrix after a refit."""
        self.run_main(["--config", "in.yml", "--refit-link", "link_a",
                       "--skip-matrix"])
        inst = FakeBuilder.last_instance
        kinds = [c[0] for c in inst.calls]
        self.assertIn("refit_link", kinds)
        self.assertNotIn("matrix", kinds)

    def test_edit_explicit_recompute(self):
        """--recompute-matrix recomputes without any refit."""
        self.run_main(["--config", "in.yml", "--recompute-matrix"])
        inst = FakeBuilder.last_instance
        kinds = [c[0] for c in inst.calls]
        self.assertIn("matrix", kinds)

    def test_edit_preserves_custom_fields(self):
        """Custom yml fields survive the rebuild; spheres do not."""
        self.run_main(["--config", "in.yml"])
        inst = FakeBuilder.last_instance
        config, path = inst.saved[0]
        self.assertEqual(path, "in.yml")
        self.assertEqual(config.lock_joints, {"wheel_joint": 0.0})
        self.assertEqual(config.base_link, "base_link")
        # Rebuilt keys come from the builder, not the stale file snapshot.
        self.assertEqual(config.collision_spheres,
                         {"link_a": [{"center": [0.0, 0.0, 0.0],
                                      "radius": 0.05}]})

    def test_load_original_kinematics_shapes(self):
        """Both top-level and robot_cfg-wrapped yml shapes load."""
        try:
            import yaml
        except ImportError:
            self.skipTest("pyyaml not installed")
        self._orig_patch.stop()
        try:
            from curobov2_ros_extra.build_curobo_config import (
                load_original_kinematics,
            )
            body = {"kinematics": {"base_link": "b",
                                   "lock_joints": {"j": 1.0}}}
            for data in (body, {"robot_cfg": body}):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".yml",
                                                 delete=False) as f:
                    yaml.safe_dump(data, f)
                    path = f.name
                try:
                    loaded = load_original_kinematics(path)
                finally:
                    os.unlink(path)
                self.assertEqual(loaded["lock_joints"], {"j": 1.0})
        finally:
            self._orig_patch.start()

    def test_edit_budget_sets_num_spheres(self):
        """--budget LINK=N refits with an exact count and recomputes."""
        self.run_main(["--config", "in.yml", "--budget", "link_a=8"])
        inst = FakeBuilder.last_instance
        refits = [c for c in inst.calls if c[0] == "refit_link"]
        self.assertEqual(len(refits), 1)
        self.assertEqual(refits[0][1], "link_a")
        self.assertEqual(refits[0][2]["num_spheres"], 8)
        kinds = [c[0] for c in inst.calls]
        self.assertIn("matrix", kinds)

    def test_edit_bad_budget(self):
        """Malformed --budget specs exit with code 2."""
        for bad in ["link_a", "link_a=0", "link_a=x", "=5"]:
            with self.assertRaises(SystemExit) as ctx:
                self.run_main(["--config", "in.yml", "--budget", bad])
            self.assertEqual(ctx.exception.code, 2)

    def test_edit_refit_all_passes_morphit_opts(self):
        """--refit-all forwards fit-type and morphit tuning."""
        self.run_main(["--config", "in.yml", "--refit-all",
                       "--fit-type", "morphit", "--iterations", "50",
                       "--coverage-weight", "2000", "--protrusion-weight", "5"])
        inst = FakeBuilder.last_instance
        fits = [c for c in inst.calls if c[0] == "fit_all"]
        self.assertEqual(len(fits), 1)
        self.assertEqual(fits[0][1]["fit_type"], "morphit")
        self.assertEqual(fits[0][1]["iterations"], 50)
        self.assertEqual(fits[0][1]["coverage_weight"], 2000.0)
        self.assertEqual(fits[0][1]["protrusion_weight"], 5.0)

    def test_create_mode_defaults(self):
        """Create mode fits with morphit opts and saves the default name."""
        import curobov2_ros_extra.build_curobo_config as mod
        with patch.object(mod, "resolve_package_meshes",
                          return_value="/tmp/resolved.urdf"):
            self.run_main(["--urdf", "r.urdf", "--asset-path", "/a",
                           "--iterations", "50"])
        inst = FakeBuilder.last_instance
        self.assertEqual(inst.init_kwargs["urdf_path"], "/tmp/resolved.urdf")
        fits = [c for c in inst.calls if c[0] == "fit_all"]
        self.assertEqual(len(fits), 1)
        self.assertEqual(fits[0][1]["iterations"], 50)
        kinds = [c[0] for c in inst.calls]
        self.assertIn("matrix", kinds)
        self.assertIn(("save", "curobo_config.yml"), inst.calls)
    def test_edit_import_bubblify(self):
        """A bubblify export merges into the builder before build."""
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("pyyaml not installed")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml",
                                         delete=False) as f:
            f.write("collision_spheres:\n"
                    "  link_a:\n"
                    "  - center: [0.1, 0.2, 0.3]\n"
                    "    radius: 0.07\n")
            path = f.name
        try:
            self.run_main(["--config", "in.yml",
                           "--import-bubblify", path])
        finally:
            os.unlink(path)
        inst = FakeBuilder.last_instance
        kinds = [c[0] for c in inst.calls]
        self.assertIn("matrix", kinds)
        self.assertEqual(inst._collision_spheres["link_a"][0]["radius"], 0.07)

    def test_bubblify_missing_install(self):
        """--with-bubblify without the package exits non-zero."""
        if "bubblify" in sys.modules or "bubblify.gui" in sys.modules:
            self.skipTest("bubblify installed; cannot test missing path")
        import curobov2_ros_extra.build_curobo_config as mod
        with patch.object(mod, "resolve_package_meshes",
                          side_effect=lambda u, a: u), \
                patch("os.path.exists", return_value=True):
            with self.assertRaises(SystemExit) as ctx:
                self.run_main(["--config", "in.yml", "--with-bubblify"])
        self.assertNotEqual(ctx.exception.code, 0)


class FakeStore:
    """Minimal bubblify SphereStore stand-in."""

    def __init__(self):
        """Start empty."""
        self._next_id = 0
        self.by_id = {}
        self.ids_by_link = {}

    def add(self, link, xyz=(0.0, 0.0, 0.0), radius=0.05):
        """Add a sphere."""
        sid = self._next_id
        self._next_id += 1
        ns = type("S", (), {})()
        ns.id, ns.link, ns.local_xyz, ns.radius, ns.node = (
            sid, link, xyz, radius, None)
        self.by_id[sid] = ns
        self.ids_by_link.setdefault(link, []).append(sid)
        return ns

    def remove(self, sphere_id):
        """Remove a sphere."""
        sphere = self.by_id.pop(sphere_id)
        self.ids_by_link[sphere.link].remove(sphere_id)
        return sphere

    def get_spheres_for_link(self, link):
        """List spheres for a link."""
        return [self.by_id[sid] for sid in self.ids_by_link.get(link, [])]


class FakeGuiHandle:
    """Viser container/button/slider stand-in."""

    def __init__(self, label="", value=None):
        """Hold label and value."""
        self.label = label
        self.value = value
        self.options = []
        self.callbacks = []
        self._impl = SimpleNamespace(
            props=SimpleNamespace(label=label))

    def __enter__(self):
        """Enter."""
        return self

    def __exit__(self, *args):
        """Exit."""
        return False

    def on_click(self, fn):
        """Record button callback."""
        self.callbacks.append(fn)
        return fn

    def on_update(self, fn):
        """Record update callback."""
        self.callbacks.append(fn)
        return fn


class FakeGui:
    """Viser gui stand-in recording created controls."""

    def __init__(self):
        """Start empty."""
        self.folders = []
        self.markdowns = []
        self.buttons = {}
        self.texts = {}
        self.sliders = {}
        self.dropdowns = {}
        self._container_handle_from_uuid = {}

    def add_folder(self, label, **kwargs):
        """Record folder."""
        self.folders.append(label)
        return FakeGuiHandle(label=label)

    def add_markdown(self, content):
        """Record markdown."""
        self.markdowns.append(content)
        return FakeGuiHandle()

    def add_button(self, label):
        """Record button."""
        handle = FakeGuiHandle(label=label)
        self.buttons[label] = handle
        return handle

    def add_text(self, label, initial_value=""):
        """Record text."""
        handle = FakeGuiHandle(label=label, value=initial_value)
        self.texts[label] = handle
        return handle

    def add_slider(self, label, min=0, max=1, step=1, initial_value=0):
        """Record slider."""
        handle = FakeGuiHandle(label=label, value=initial_value)
        self.sliders[label] = handle
        return handle

    def add_dropdown(self, label, options=None, initial_value=None):
        """Record dropdown."""
        handle = FakeGuiHandle(label=label, value=initial_value)
        handle.options = options or []
        self.dropdowns[label] = handle
        return handle


class FakeDropdown:
    """Viser dropdown stand-in."""

    def __init__(self):
        """Start empty."""
        self.options = []
        self.value = "None"
        self.callbacks = []

    def on_update(self, fn):
        """Record update callback."""
        self.callbacks.append(fn)
        return fn


class FakeApp:
    """Minimal BubblifyApp stand-in."""

    def __init__(self):
        """Start with an empty store."""
        self.sphere_store = FakeStore()
        self._sphere_dropdown = FakeDropdown()
        self._link_dropdown = FakeDropdown()
        self.current_sphere_id = None
        self.current_link = ""
        self.visualized = []
        self.updated = []

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


class TestMorphitSync(unittest.TestCase):
    """Store sync/pull helpers with stub apps (no viser needed)."""

    def test_sync_replaces_link_spheres(self):
        """Sync drops old visuals and shows the fitted spheres."""
        from curobov2_ros_extra.morphit_bubblify import sync_link_spheres
        app = FakeApp()
        old = app.sphere_store.add("link_a", xyz=(9.0, 9.0, 9.0), radius=0.9)
        spheres = {"link_a": [{"center": [0.1, 0.0, 0.0], "radius": 0.05},
                              {"center": [0.2, 0.0, 0.0], "radius": 0.06}]}
        n = sync_link_spheres(app, spheres, "link_a")
        self.assertEqual(n, 2)
        self.assertNotIn(old.id, app.sphere_store.by_id)
        current = app.sphere_store.get_spheres_for_link("link_a")
        self.assertEqual(len(current), 2)
        self.assertEqual(app.visualized, [s.id for s in current])
        self.assertEqual(app._sphere_dropdown.value,
                         f"Sphere {current[0].id}")
        for step in ("transform", "opacities", "props"):
            self.assertIn(step, app.updated)

    def test_pull_reads_store(self):
        """Pull converts store spheres to cuRobo dicts."""
        from curobov2_ros_extra.morphit_bubblify import pull_store_spheres
        app = FakeApp()
        app.sphere_store.add("link_a", xyz=(0.1, 0.2, 0.3), radius=0.07)
        pulled = pull_store_spheres(app.sphere_store)
        self.assertEqual(pulled, {"link_a": [{"center": [0.1, 0.2, 0.3],
                                              "radius": 0.07}]})

    def test_selected_link_prefers_dropdown(self):
        """Selection follows bubblify's link dropdown."""
        from curobov2_ros_extra.morphit_bubblify import selected_link
        app = FakeApp()
        app._link_dropdown.value = "link_a"
        app.current_link = "link_b"
        self.assertEqual(selected_link(app), "link_a")

    def test_selected_link_falls_back(self):
        """Without a dropdown value the app link is used."""
        from curobov2_ros_extra.morphit_bubblify import selected_link
        app = FakeApp()
        app._link_dropdown.value = ""
        app.current_link = "link_b"
        self.assertEqual(selected_link(app), "link_b")

    def test_refresh_budget_sets_current_count(self):
        """Budget slider shows the spheres on the selected link."""
        from curobov2_ros_extra.morphit_bubblify import refresh_budget_slider
        app = FakeApp()
        app._link_dropdown.value = "link_a"
        app.sphere_store.add("link_a", radius=0.05)
        app.sphere_store.add("link_a", radius=0.06)
        app.sphere_store.add("link_b", radius=0.07)

        class Slider:
            """Viser slider stand-in."""

            def __init__(self):
                """Start at zero."""
                self.value = 0

        slider = Slider()
        self.assertEqual(refresh_budget_slider(app, slider), 2)
        self.assertEqual(slider.value, 2)

    def test_refresh_budget_clamps(self):
        """Budget slider clamps at the max budget."""
        from curobov2_ros_extra.morphit_bubblify import (
            MAX_BUDGET,
            refresh_budget_slider,
        )
        app = FakeApp()
        app._link_dropdown.value = "link_a"
        for _ in range(MAX_BUDGET + 10):
            app.sphere_store.add("link_a", radius=0.01)

        class Slider:
            """Viser slider stand-in."""

            def __init__(self):
                """Start at zero."""
                self.value = 0

        slider = Slider()
        self.assertEqual(refresh_budget_slider(app, slider), MAX_BUDGET)


class TestMatrixSafe(unittest.TestCase):
    """recompute_matrix_safe never loses spheres to OOM."""

    def test_success_returns_true(self):
        """A clean recompute returns True and forwards sizes."""
        from curobov2_ros_extra.build_curobo_config import (
            recompute_matrix_safe,
        )

        class Builder:
            """Minimal compute stand-in."""

            def __init__(self):
                """Start empty."""
                self.kwargs = None

            def compute_collision_matrix(self, **kwargs):
                """Record."""
                self.kwargs = kwargs

        builder = Builder()
        self.assertTrue(recompute_matrix_safe(builder, 500, 2000))
        self.assertEqual(builder.kwargs["num_samples"], 500)
        self.assertEqual(builder.kwargs["batch_size"], 2000)

    def test_oom_returns_false(self):
        """CUDA OOM returns False instead of raising."""
        from curobov2_ros_extra.build_curobo_config import (
            recompute_matrix_safe,
        )

        class Builder:
            """OOM stand-in."""

            def compute_collision_matrix(self, **kwargs):
                """Simulate OOM."""
                raise RuntimeError("CUDA out of memory. Tried to allocate")

        self.assertFalse(recompute_matrix_safe(Builder(), 500, 2000))

    def test_other_errors_raise(self):
        """Non-OOM errors still propagate."""
        from curobov2_ros_extra.build_curobo_config import (
            recompute_matrix_safe,
        )

        class Builder:
            """Failing stand-in."""

            def compute_collision_matrix(self, **kwargs):
                """Simulate bug."""
                raise ValueError("bad matrix")

        with self.assertRaises(ValueError):
            recompute_matrix_safe(Builder(), 500, 2000)


class TestNoExportApp(unittest.TestCase):
    """The bubblify subclass must not create Export buttons."""

    def test_export_section_replaced(self):
        """Upstream export controls are replaced by a save note."""
        from curobov2_ros_extra.morphit_bubblify import _bubblify_app_class

        class FakeFolder:
            """Viser folder context stand-in."""

            def __enter__(self):
                """Enter."""
                return self

            def __exit__(self, *args):
                """Exit."""
                return False

        class FakeGui:
            """Viser gui stand-in recording created controls."""

            def __init__(self):
                """Start empty."""
                self.folders = []
                self.markdowns = []
                self.buttons = []

            def add_folder(self, label, **kwargs):
                """Record folder."""
                self.folders.append(label)
                return FakeFolder()

            def add_markdown(self, content):
                """Record markdown."""
                self.markdowns.append(content)
                return object()

            def add_button(self, label):
                """Record button."""
                self.buttons.append(label)
                return object()

        class FakeServer:
            """Viser server stand-in."""

            def __init__(self, gui):
                """Hold the gui."""
                self.gui = gui

        base_calls = []

        class FakeBase:
            """Upstream BubblifyApp stand-in."""

            def __init__(self, **kwargs):
                """Build export controls like upstream."""
                self.server = FakeServer(FakeGui())
                self._setup_export_controls()

            def _setup_export_controls(self):
                """Upstream behavior: export buttons."""
                base_calls.append("export")
                with self.server.gui.add_folder("Export"):
                    self.server.gui.add_button("Export Spheres (YAML)")

        gui_mod = ModuleType("bubblify.gui")
        gui_mod.BubblifyApp = FakeBase
        pkg_mod = ModuleType("bubblify")
        pkg_mod.gui = gui_mod
        sys.modules["bubblify"] = pkg_mod
        sys.modules["bubblify.gui"] = gui_mod
        try:
            app_cls = _bubblify_app_class()
            app = app_cls()
        finally:
            sys.modules.pop("bubblify.gui", None)
            sys.modules.pop("bubblify", None)
        self.assertEqual(base_calls, [])
        self.assertEqual(app.server.gui.folders, [])
        self.assertEqual(app.server.gui.markdowns, [])
        self.assertEqual(app.server.gui.buttons, [])


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


if __name__ == "__main__":
    unittest.main()
