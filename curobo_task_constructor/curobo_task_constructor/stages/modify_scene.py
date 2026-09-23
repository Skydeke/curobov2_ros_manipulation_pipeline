"""ModifyScene — a zero-cost scene mutation stage (MTC ``ModifyPlanningScene``).

The joint state passes through unchanged; only the carried ``SceneDiff`` is
mutated (attach/detach/add/remove/allow-or-forbid collisions). The stage
records the concrete ``scene_ops`` delta it introduced so the executor can
materialize each op on the curobo server in chain order during execution.
"""

from __future__ import annotations

from curobo_task_constructor.core.registry import register_stage
from curobo_task_constructor.core.stage import PropagatingStage
from curobo_task_constructor.core.state import InterfaceState


@register_stage("modify_scene")
class ModifyScene(PropagatingStage):
    def compute_forward(self, state: InterfaceState) -> None:
        scene = state.scene or self._base_scene
        if self.params.get("attach"):
            name = self.params["attach"]
            new_scene = scene.with_attached(name)
            ops = [("attach", name)]
        elif self.params.get("detach"):
            name = self.params["detach"]
            new_scene = scene.with_detached(name)
            ops = [("detach", name)]
        elif self.params.get("add"):
            spec = self._object_spec(self.params["add"])
            new_scene = scene.with_object_added(spec)
            ops = [("add", spec)]
        elif self.params.get("remove"):
            name = self.params["remove"]
            new_scene = scene.with_object_removed(name)
            ops = [("remove", name)]
        elif self.params.get("allow_collisions") is not None:
            cfg = self.params["allow_collisions"]
            new_scene = scene.with_collisions(cfg["object"], cfg.get("links", []),
                                              cfg.get("enabled", True))
            ops = []  # collision disabling rides inside the goalsets
        else:
            self._fail(state, None, "modify_scene requires one of: attach, "
                                    "detach, add, remove, allow_collisions")
            return
        end = state.clone(scene=new_scene)
        # Zero-cost mutation: joint configuration unchanged, no trajectory.
        self.send_forward(state, end, trajectory=None, cost=0.0,
                          comment=self._comment(), scene_ops=ops)

    # ------------------------------------------------------------------
    def _object_spec(self, cfg: dict):
        from curobo_task_constructor.core.state import ObjectSpec
        dims = cfg.get("dimensions") or cfg.get("size")
        return ObjectSpec(
            name=cfg["name"],
            shape=cfg.get("shape", "cuboid"),
            pose=cfg.get("pose"),
            dimensions=([float(d) for d in dims] if dims else None),
            mesh_path=cfg.get("mesh_path"),
            vertices=cfg.get("vertices"),
            triangles=cfg.get("triangles"),
        )

    def _comment(self) -> str:
        parts = [f"{k}={v}" for k, v in self.params.items()
                 if k in ("attach", "detach", "remove")]
        if self.params.get("allow_collisions") is not None:
            ac = self.params["allow_collisions"]
            parts.append(f"collisions({ac.get('object')},{ac.get('links')},"
                         f"{ac.get('enabled', True)})")
        return " ".join(parts)

    def init(self, base_scene, robot) -> None:
        super().init(base_scene, robot)
        self._base_scene = base_scene