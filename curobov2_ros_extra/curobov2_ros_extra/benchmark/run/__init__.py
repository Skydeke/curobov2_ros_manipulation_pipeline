"""Package facade for benchmark/run: re-exports the original API."""

from .common import _dump, _fmt_dur, _print_run_summary, _print_wall_time, _resolve_tol, _seed_knobs, _sidecar
from .motion import cmd_all, cmd_compare, cmd_core, cmd_reference, cmd_ros, cmd_webpage
from .ik_cost import _print_cost_core_summary, _print_ik_core_summary, cmd_cost, cmd_cost_core, cmd_cost_ros, cmd_ik, cmd_ik_core, cmd_ik_ros
from .cli import _add_compare_opts, _add_dataset, _add_ik_cost_opts, _add_max_attempts_opt, _add_motion_lock_opt, _add_pose_compare_opts, _add_reset_seed_opt, _add_ros_service_opts, _add_scene, _add_solver_opts, build_parser
from .app import main

__all__ = [
    '_add_compare_opts',
    '_add_dataset',
    '_add_ik_cost_opts',
    '_add_max_attempts_opt',
    '_add_motion_lock_opt',
    '_add_pose_compare_opts',
    '_add_reset_seed_opt',
    '_add_ros_service_opts',
    '_add_scene',
    '_add_solver_opts',
    '_dump',
    '_fmt_dur',
    '_print_cost_core_summary',
    '_print_ik_core_summary',
    '_print_run_summary',
    '_print_wall_time',
    '_resolve_tol',
    '_seed_knobs',
    '_sidecar',
    'build_parser',
    'cmd_all',
    'cmd_compare',
    'cmd_core',
    'cmd_cost',
    'cmd_cost_core',
    'cmd_cost_ros',
    'cmd_ik',
    'cmd_ik_core',
    'cmd_ik_ros',
    'cmd_reference',
    'cmd_ros',
    'cmd_webpage',
    'main',
]
