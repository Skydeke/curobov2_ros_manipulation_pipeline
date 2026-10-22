"""Interactive bubblify session with cuRobo MorphIt refits in one section.

Wraps :class:`bubblify.gui.BubblifyApp` (https://github.com/bheijden/bubblify):
MorphIt controls (per-link sphere budget, fit options, refit buttons) are
appended into bubblify's own Sphere Editor folder, follow bubblify's link
selection, and the budget defaults to the spheres on the current link.

All GUI/viser imports are lazy so this module imports without bubblify.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

#: Folder label used by bubblify's sphere editor (must match bubblify.gui).
SPHERE_EDITOR_FOLDER = "⚪ Sphere Editor"

#: Upper bound of the budget slider.
MAX_BUDGET = 128


def pull_store_spheres(store: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Read bubblify store spheres as cuRobo-style sphere dicts."""
    spheres: Dict[str, List[Dict[str, Any]]] = {}
    for link, ids in dict(store.ids_by_link).items():
        entries = []
        for sid in ids:
            s = store.by_id[sid]
            center = s.local_xyz
            if hasattr(center, "tolist"):
                center = center.tolist()
            entries.append({"center": [float(v) for v in center],
                            "radius": float(s.radius)})
        spheres[link] = entries
    return spheres


def sync_link_spheres(app: Any, spheres: Dict[str, List[Dict[str, Any]]],
                      link: str) -> int:
    """Replace the app's spheres for link with the given fitted spheres.

    Returns the number of spheres synced.
    """
    store = app.sphere_store
    for s in list(store.get_spheres_for_link(link)):
        store.remove(s.id)
    count = 0
    for entry in spheres.get(link, []):
        s = store.add(link, xyz=tuple(entry["center"]),
                      radius=float(entry["radius"]))
        app._create_sphere_visualization(s)
        count += 1
    current = store.get_spheres_for_link(link)
    app.current_link = link
    sync = getattr(app, "_sync_link_selection", None)
    if callable(sync):
        sync()
    if current:
        app.current_sphere_id = current[0].id
    dropdown = getattr(app, "_sphere_dropdown", None)
    if dropdown is not None:
        dropdown.options = ([f"Sphere {s.id}" for s in current] or ["None"])
        if current:
            dropdown.value = f"Sphere {current[0].id}"
    for name in ("_update_transform_control", "_update_sphere_opacities",
                 "_update_sphere_properties_ui"):
        fn = getattr(app, name, None)
        if callable(fn):
            fn()
    return count


def selected_link(app: Any) -> str:
    """Return the link currently selected in bubblify's sphere editor."""
    dropdown = getattr(app, "_link_dropdown", None)
    if dropdown is not None and getattr(dropdown, "value", None):
        return dropdown.value
    return getattr(app, "current_link", "")


def refresh_budget_slider(app: Any, budget_slider: Any,
                          max_budget: int = MAX_BUDGET) -> int:
    """Set the budget slider to the spheres on the current link.

    Returns the value set.
    """
    count = len(app.sphere_store.get_spheres_for_link(selected_link(app)))
    budget_slider.value = min(count, max_budget)
    return budget_slider.value


def _parse_budget_value(value: Any) -> int:
    """Coerce a viser number/slider value to a budget int (>= 0)."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _find_folder(gui: Any, label: str) -> Any:
    """Find an existing viser folder container by label, if any."""
    registry = getattr(gui, "_container_handle_from_uuid", {})
    for handle in registry.values():
        state = getattr(handle, "_impl", handle)
        props = getattr(state, "props", None)
        if getattr(props, "label", None) == label:
            return handle
    return None


def _bubblify_app_class():
    """Return BubblifyApp with its Export section replaced by a save note.

    The CLI owns saving (Ctrl+C writes only the cuRobo yml), so bubblify's
    own Export buttons — which would write stray files next to the temp
    URDF — are never created.
    """
    try:
        from bubblify.gui import BubblifyApp
    except ImportError:
        print("ERROR: bubblify is not installed. "
              "Install with: pip install bubblify", file=sys.stderr)
        sys.exit(1)

    class _NoExportBubblifyApp(BubblifyApp):
        """Bubblify without file export; the CLI owns saving."""

        def _setup_export_controls(self):
            """Drop bubblify's Export section entirely."""

    return _NoExportBubblifyApp


def launch_morphit_bubblify(builder: Any, urdf_path: str,
                            spherization_yml: str,
                            port: int, show_collision: bool,
                            fit_defaults: Dict[str, Any],
                            save_fn: Any, output: str) -> bool:
    """Run bubblify with MorphIt refits and a Save button in its editor. Args:
        builder: cuRobo :class:`RobotBuilder` (already loaded). Refit buttons
            call its ``refit_link_spheres``/``fit_collision_spheres`` on the
            link selected in bubblify, and results sync into the viewer."""
    try:
        from curobov2_ros_extra.curobo_config_io import (
            pull_store_spheres_from_builder,
        )
    except ImportError:  # script executed directly, not as installed package
        from curobo_config_io import pull_store_spheres_from_builder

    before = pull_store_spheres_from_builder(builder)

    app = _bubblify_app_class()(
        robot_name="custom",
        urdf_path=Path(urdf_path),
        show_collision=show_collision,
        port=port,
        spherization_yml=Path(spherization_yml),
    )

    fit_type_default = str(fit_defaults.get("fit_type", "morphit"))
    if "SphereFitType" in type(fit_defaults.get("fit_type")).__name__:
        fit_type_default = str(fit_defaults["fit_type"].value)
    state: Dict[str, Any] = {"did_refit": False, "last_saved": before,
                               "last_dest": output}

    folder = _find_folder(app.server.gui, SPHERE_EDITOR_FOLDER)
    if folder is None:  # viser API drift fallback: separate section
        folder = app.server.gui.add_folder("MorphIt Refit")

    with folder:
        budget_slider = app.server.gui.add_slider(
            "MorphIt budget (0 = density)", min=0, max=MAX_BUDGET, step=1,
            initial_value=0)
        density_slider = app.server.gui.add_slider(
            "MorphIt density (budget 0)", min=0.1, max=10.0, step=0.1,
            initial_value=float(fit_defaults.get("sphere_density", 1.0)))
        iterations_slider = app.server.gui.add_slider(
            "MorphIt iterations", min=10, max=1000, step=10,
            initial_value=int(fit_defaults.get("iterations", 200)))
        fit_type_dropdown = app.server.gui.add_dropdown(
            "MorphIt fit type", options=["morphit", "voxel", "surface"],
            initial_value=fit_type_default)
        refit_btn = app.server.gui.add_button("MorphIt refit this link")
        refit_all_btn = app.server.gui.add_button("MorphIt refit ALL links")
        output_field = app.server.gui.add_text("Save cuRobo to",
                                               initial_value=output)
        save_btn = app.server.gui.add_button("Save cuRobo config")
        status = app.server.gui.add_text("MorphIt status",
                                         initial_value="Ready")

        def current_dest() -> str:
            """Output path from the text field, falling back to default."""
            return (output_field.value or "").strip() or output

        def current_opts() -> Dict[str, Any]:
            """Collect panel fit options, resolving the fit-type enum."""
            from curobo._src.geom.sphere_fit.types import SphereFitType
            return {
                "sphere_density": float(density_slider.value),
                "fit_type": SphereFitType(fit_type_dropdown.value),
                "iterations": int(iterations_slider.value),
                "coverage_weight": fit_defaults.get("coverage_weight"),
                "protrusion_weight": fit_defaults.get("protrusion_weight"),
                "compute_metrics": True,
            }

        def refit_one(link: str, budget: int) -> None:
            """Refit a single link with a budget (or density) and sync."""
            status.value = f"Fitting {link} ..."
            try:
                if budget > 0:
                    builder.refit_link_spheres(link, num_spheres=budget,
                                               **current_opts())
                else:
                    builder.refit_link_spheres(link, **current_opts())
            except ValueError as e:
                status.value = f"Failed: {e}"
                return
            n = sync_link_spheres(app, builder.collision_spheres or {}, link)
            state["did_refit"] = True
            refresh_budget_slider(app, budget_slider)
            status.value = f"{link}: {n} spheres"

        @refit_btn.on_click
        def _(_):
            link = selected_link(app)
            if not link:
                status.value = "Select a link first"
                return
            refit_one(link, _parse_budget_value(budget_slider.value))

        @refit_all_btn.on_click
        def _(_):
            status.value = "Fitting all links ..."
            old_links = set(app.sphere_store.ids_by_link.keys())
            builder.fit_collision_spheres(**current_opts())
            new_spheres = builder.collision_spheres or {}
            for link in old_links - set(new_spheres.keys()):
                sync_link_spheres(app, {}, link)
            for link in new_spheres:
                sync_link_spheres(app, new_spheres, link)
            state["did_refit"] = True
            refresh_budget_slider(app, budget_slider)
            total = sum(len(v) for v in new_spheres.values())
            status.value = f"All links: {total} spheres"

        @save_btn.on_click
        def _(_):
            dest = current_dest()
            pulled_now = pull_store_spheres(app.sphere_store)
            if pulled_now == state["last_saved"] and dest == state["last_dest"]:
                status.value = "No changes since last save"
                return
            # No public setter on RobotBuilder; the CLI owns the spheres.
            builder._collision_spheres = pulled_now
            try:
                status.value = save_fn(dest)
            except Exception as e:  # never lose the viewer session to save
                status.value = f"Save failed: {e}"
                return
            state["last_saved"] = pull_store_spheres(app.sphere_store)
            state["last_dest"] = dest

    # Follow bubblify's link selection: a new link sets the budget to the
    # spheres on that link. Both dropdowns drive the same selection.
    def _on_link_change(_):
        refresh_budget_slider(app, budget_slider)

    link_dropdown = getattr(app, "_link_dropdown", None)
    if link_dropdown is not None:
        link_dropdown.on_update(_on_link_change)
    current_dropdown = getattr(app, "_current_link_dropdown", None)
    if current_dropdown is not None and current_dropdown is not link_dropdown:
        current_dropdown.on_update(_on_link_change)
    refresh_budget_slider(app, budget_slider)

    print("MorphIt controls are in the Sphere Editor section: pick a link, "
          "the budget shows its sphere count, refit or edit, then Save "
          f"(writes {output}) or Ctrl+C.")
    app.run()

    dest = current_dest()
    pulled = pull_store_spheres(app.sphere_store)
    if (pulled != state["last_saved"] or dest != state["last_dest"]
            or not os.path.exists(dest)):
        builder._collision_spheres = pulled
        print(save_fn(dest))
        state["last_saved"] = pulled
        state["last_dest"] = dest
    else:
        print("No sphere changes, output already saved.")
    return pulled != before
