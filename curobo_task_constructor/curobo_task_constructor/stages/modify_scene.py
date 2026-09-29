"""ModifyScene — a zero-cost scene mutation stage (MTC ``ModifyPlanningScene``).

The joint state passes through unchanged; only the carried ``SceneDiff`` is
mutated (attach/detach/detach_all/add/remove/remove_all/allow-or-forbid
collisions). The stage records the concrete ``scene_ops`` delta it introduced
so the executor can materialize each op on the curobo server in chain order
during execution.
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
        elif self.params.get("detach_all"):
            # Unconditional release of whatever is in the hand — the named
            # detach's cousin of remove_all. The op clears the SERVER's
            # attach state (attached_name/excluded_obstacle_names) and the
            # executor's mirror whether or not THIS task's own stages ever
            # attached anything, so a scene-setup task cannot build on a
            # stale attach held over from a previous cycle: remove_all clears
            # obstacles but NOT the attach, and a leftover attached name
            # re-disables the re-added object in every solver checker
            # (reapply_attached_disables after each world push).
            new_scene = scene
            if scene.attached_object is not None:
                new_scene = new_scene.with_detached(scene.attached_object)
            ops = [("detach_all", None)]
        elif self.params.get("add"):
            spec = self._object_spec(self.params["add"])
            new_scene = scene.with_object_added(spec)
            ops = [("add", spec)]
        elif self.params.get("remove"):
            name = self.params["remove"]
            new_scene = scene.with_object_removed(name)
            ops = [("remove", name)]
        elif self.params.get("remove_all"):
            # Clear the WHOLE server world before a fresh set of adds — the
            # direct analog of the old grasp orchestrator's
            # remove_all_objects service call, expressed as a scene op so it
            # rides the executor's chain-order materialization on the server.
            # Adds reject duplicate names, so a new pick cycle needs the
            # previous cycle's cuboids gone before the new set lands — and a
            # full clear, not a tracked list, is the only honest answer to
            # "what is in the server right now" after a restart.
            #
            # The op is recorded once ("remove_all" -> the robot's
            # remove_all_objects) while the carried diff also drops every
            # base-scene name, so downstream stages read a scene that is
            # actually empty rather than one that still lists the old world.
            names = sorted(self._base_scene.objects_added)
            new_scene = scene
            for name in names:
                new_scene = new_scene.with_object_removed(name)
            ops = [("remove_all", None)]
        elif self.params.get("allow_collisions") is not None:
            cfg = self.params["allow_collisions"]
            new_scene = scene.with_collisions(cfg["object"], cfg.get("links", []),
                                              cfg.get("enabled", True))
            ops = []  # collision disabling rides inside the goalsets
        else:
            self._fail(state, None, "modify_scene requires one of: attach, "
                                    "detach, detach_all, add, remove, "
                                    "remove_all, allow_collisions")
            return
        end = state.clone(scene=new_scene)
        # Zero-cost mutation: joint configuration unchanged, no trajectory.
        self.send_forward(state, end, trajectory=None, cost=0.0,
                          comment=self._comment(), scene_ops=ops)

    # ------------------------------------------------------------------
    def _object_spec(self, cfg: dict):
        from curobo_task_constructor.core.state import ObjectSpec
        from curobo_task_constructor.stages._util import pose_from_params
        dims = cfg.get("dimensions") or cfg.get("size")
        pose = cfg.get("pose")
        if isinstance(pose, dict):
            # Declarative add stages ride as YAML, so the pose is a flat
            # {x,y,z,qx,qy,qz,qw} dict rather than a message — build the
            # Pose-like from it the same way the motion stages do.
            pose = pose_from_params(pose, self.robot)
        return ObjectSpec(
            name=cfg["name"],
            shape=cfg.get("shape", "cuboid"),
            pose=pose,
            dimensions=([float(d) for d in dims] if dims else None),
            mesh_path=cfg.get("mesh_path"),
            vertices=cfg.get("vertices"),
            triangles=cfg.get("triangles"),
        )

    def _comment(self) -> str:
        parts = [f"{k}={v}" for k, v in self.params.items()
                 if k in ("attach", "detach", "remove")]
        if self.params.get("detach_all") is not None:
            parts.append("detach_all")
        if self.params.get("remove_all") is not None:
            parts.append("remove_all")
        if self.params.get("allow_collisions") is not None:
            ac = self.params["allow_collisions"]
            parts.append(f"collisions({ac.get('object')},{ac.get('links')},"
                         f"{ac.get('enabled', True)})")
        return " ".join(parts)

    def init(self, base_scene, robot) -> None:
        super().init(base_scene, robot)
        self._base_scene = base_scene