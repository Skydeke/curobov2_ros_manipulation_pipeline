import argparse
import json
import sys

CAPABILITIES = ('planning', 'ik', 'fk', 'collision')


def _run_core_capabilities(capability, dataset):
    from isaac_ros_cumotion_benchmark import core_runner
    if capability == 'planning':
        return core_runner.run_core(dataset)
    if capability == 'ik':
        return core_runner.run_core_ik(dataset)
    if capability == 'fk':
        return core_runner.run_core_fk(dataset)
    if capability == 'collision':
        return core_runner.run_core_collision(dataset)
    if capability == 'all':
        results = []
        for cap in CAPABILITIES:
            results.extend(_run_core_capabilities(cap, dataset))
        return results
    raise ValueError(f'Unknown capability: {capability}')


def _run_ros_capabilities(capability, dataset, time_dilation_factor):
    from isaac_ros_cumotion_benchmark import ros_runner
    if capability == 'planning':
        return ros_runner.run_ros(dataset, time_dilation_factor)
    if capability == 'ik':
        return ros_runner.run_ros_ik(dataset)
    if capability == 'fk':
        return ros_runner.run_ros_fk(dataset)
    if capability == 'collision':
        return ros_runner.run_ros_collision(dataset)
    if capability == 'all':
        results = []
        for cap in CAPABILITIES:
            results.extend(_run_ros_capabilities(cap, dataset, time_dilation_factor))
        return results
    raise ValueError(f'Unknown capability: {capability}')


def cmd_core(args):
    results = _run_core_capabilities(args.capability, args.dataset)
    _dump_results(results, args.output)
    _print_summary(results, 'core')


def cmd_ros(args):
    results = _run_ros_capabilities(
        args.capability, args.dataset, args.time_dilation_factor
    )
    _dump_results(results, args.output)
    _print_summary(results, 'ros')


def cmd_compare(args):
    from isaac_ros_cumotion_benchmark.compare import compare, print_report, save_report
    with open(args.core_results) as f:
        core = json.load(f)
    with open(args.ros_results) as f:
        ros = json.load(f)
    report = compare(core, ros)
    print_report(report, show_all=args.show_all)
    if args.output:
        save_report(report, args.output)
        print(f'Report saved to {args.output}')


def cmd_all(args):
    from isaac_ros_cumotion_benchmark.compare import compare, print_report, save_report

    core_results = _run_core_capabilities(args.capability, args.dataset)
    ros_results = _run_ros_capabilities(
        args.capability, args.dataset, args.time_dilation_factor
    )

    report = compare(core_results, ros_results)
    print_report(report, show_all=args.show_all)
    if args.output:
        save_report(report, args.output)
        if args.save_all:
            _dump_results(core_results, args.output.replace('.json', '_core.json'))
            _dump_results(ros_results, args.output.replace('.json', '_ros.json'))
        print(f'Report saved to {args.output}')


def _dump_results(results, path):
    if path:
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {path}')


def _print_summary(results, label):
    successes = sum(1 for r in results if r['success'])
    total = len(results)
    total_time = sum(r['time_s'] for r in results)
    print(f'[{label}] {successes}/{total} successes, {total_time:.3f}s total')


def main():
    parser = argparse.ArgumentParser(
        description='cuRobo benchmark: direct vs through-ROS comparison'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_core = subparsers.add_parser('core', help='Run benchmark directly against cuRobo')
    p_core.add_argument('--dataset', default='demo',
                        choices=['demo', 'motion_benchmaker', 'mpinets'])
    p_core.add_argument('--capability', default='planning',
                        choices=['planning', 'ik', 'fk', 'collision', 'all'],
                        help='Which capability to benchmark')
    p_core.add_argument('--output', '-o', help='Save results to JSON file')
    p_core.set_defaults(func=cmd_core)

    p_ros = subparsers.add_parser('ros', help='Run benchmark through curobo_server_node')
    p_ros.add_argument('--dataset', default='demo',
                       choices=['demo', 'motion_benchmaker', 'mpinets'])
    p_ros.add_argument('--capability', default='planning',
                       choices=['planning', 'ik', 'fk', 'collision', 'all'],
                       help='Which capability to benchmark')
    p_ros.add_argument('--time_dilation_factor', type=float, default=1.0)
    p_ros.add_argument('--output', '-o', help='Save results to JSON file')
    p_ros.set_defaults(func=cmd_ros)

    p_compare = subparsers.add_parser('compare', help='Compare core and ROS results')
    p_compare.add_argument('core_results', help='Path to core results JSON')
    p_compare.add_argument('ros_results', help='Path to ROS results JSON')
    p_compare.add_argument('--output', '-o', help='Save comparison report to JSON')
    p_compare.add_argument('--show-all', action='store_true',
                           help='Show matching entries too')
    p_compare.set_defaults(func=cmd_compare)

    p_all = subparsers.add_parser('all', help='Run core then ros then compare')
    p_all.add_argument('--dataset', default='demo',
                       choices=['demo', 'motion_benchmaker', 'mpinets'])
    p_all.add_argument('--capability', default='all',
                       choices=['planning', 'ik', 'fk', 'collision', 'all'],
                       help='Which capability to benchmark (default: all)')
    p_all.add_argument('--time_dilation_factor', type=float, default=1.0)
    p_all.add_argument('--output', '-o', help='Save comparison report to JSON')
    p_all.add_argument('--save-all', action='store_true',
                       help='Save individual core/ros results too')
    p_all.add_argument('--show-all', action='store_true',
                       help='Show matching entries in comparison report')
    p_all.set_defaults(func=cmd_all)

    parsed = parser.parse_args()
    try:
        parsed.func(parsed)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()