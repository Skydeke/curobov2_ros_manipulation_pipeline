"""Unit tests for the build_curobo_config CLI routing (no GPU/ROS needed)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

_HERE = Path(__file__).resolve()
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
sys.path.insert(0, str(_HERE.parent.parent))
from curobov2_ros_extra.build_curobo_config import (  # noqa: E402
    build_parser,
    collision_spheres_to_bubblify_data,
)
if __name__ == "__main__":
    unittest.main()
