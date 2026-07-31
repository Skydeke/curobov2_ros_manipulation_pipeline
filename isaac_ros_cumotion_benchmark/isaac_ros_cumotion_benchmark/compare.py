import json


def _make_lookup(results):
    return {r['problem_name']: r for r in results}


CAPABILITY_FIELDS = (
    'n_waypoints',
    'position_error',
    'rotation_error',
    'in_collision',
    'world_distance',
    'self_distance',
    'num_frames',
    'frame_names',
)


def compare(core_results, ros_results, tolerance=1e-6):
    core_lookup = _make_lookup(core_results)
    ros_lookup = _make_lookup(ros_results)

    all_names = sorted(set(core_lookup.keys()) | set(ros_lookup.keys()))

    mismatches = []
    matches = []

    for name in all_names:
        c = core_lookup.get(name)
        r = ros_lookup.get(name)

        entry = {
            'problem_name': name,
            'capability': (c or r or {}).get('capability', 'planning'),
            'core_success': c['success'] if c else None,
            'ros_success': r['success'] if r else None,
            'core_time_s': c['time_s'] if c else None,
            'ros_time_s': r['time_s'] if r else None,
        }

        for field in CAPABILITY_FIELDS:
            if (c and field in c) or (r and field in r):
                entry[f'core_{field}'] = c.get(field) if c else None
                entry[f'ros_{field}'] = r.get(field) if r else None

        if c is None or r is None:
            entry['diff_type'] = 'missing'
            mismatches.append(entry)
        elif c['success'] != r['success']:
            entry['diff_type'] = 'success_mismatch'
            mismatches.append(entry)
        elif abs(c['time_s'] - r['time_s']) > tolerance:
            entry['diff_type'] = 'time_mismatch'
            entry['time_diff'] = c['time_s'] - r['time_s']
            mismatches.append(entry)
        else:
            entry['diff_type'] = 'match'
            matches.append(entry)

    return {
        'total': len(all_names),
        'matches': len(matches),
        'mismatches': len(mismatches),
        'core_only': sum(1 for n in all_names if n not in ros_lookup),
        'ros_only': sum(1 for n in all_names if n not in core_lookup),
        'success_mismatches': sum(
            1 for m in mismatches if m['diff_type'] == 'success_mismatch'
        ),
        'details': {'matches': matches, 'mismatches': mismatches},
    }


def print_report(report, show_all=False):
    print('=' * 72)
    print(f'  Total problems: {report["total"]}')
    print(f'  Matches:        {report["matches"]}')
    print(f'  Mismatches:     {report["mismatches"]}')
    print(f'  Core only:      {report["core_only"]}')
    print(f'  ROS only:       {report["ros_only"]}')
    print(f'  Success diffs:  {report["success_mismatches"]}')
    print('=' * 72)

    if report['mismatches']:
        print('\n  MISMATCHES:')
        for m in report['details']['mismatches']:
            dt = m.get('diff_type', '?')
            name = m['problem_name']
            cap = m.get('capability', 'planning')
            c_s = m['core_success']
            r_s = m['ros_success']
            c_t = m['core_time_s']
            r_t = m['ros_time_s']
            if dt == 'missing':
                print(f'    {name} [{cap}]: only in {"core" if c_s is not None else "ros"}')
            elif dt == 'success_mismatch':
                print(f'    {name} [{cap}]: core.success={c_s} != ros.success={r_s}'
                      f'  (core={c_t:.3f}s, ros={r_t:.3f}s)')
            elif dt == 'time_mismatch':
                print(f'    {name} [{cap}]: time diff {m.get("time_diff", 0):.3f}s'
                      f'  (core={c_t:.3f}s, ros={r_t:.3f}s)')

    if show_all and report['details']['matches']:
        print('\n  ALL MATCHES:')
        for m in report['details']['matches']:
            name = m['problem_name']
            c_t = m['core_time_s']
            r_t = m['ros_time_s']
            print(f'    {name}: both success'
                  f'  (core={c_t:.3f}s, ros={r_t:.3f}s)')


def save_report(report, path):
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('core_results', help='Path to JSON file with core results')
    parser.add_argument('ros_results', help='Path to JSON file with ROS results')
    parser.add_argument('--output', '-o', help='Save report to JSON file')
    parser.add_argument('--show-all', action='store_true', help='Show matching entries too')
    args = parser.parse_args()

    with open(args.core_results) as f:
        core = json.load(f)
    with open(args.ros_results) as f:
        ros = json.load(f)

    report = compare(core, ros)
    print_report(report, show_all=args.show_all)
    if args.output:
        save_report(report, args.output)
        print(f'\nReport saved to {args.output}')


if __name__ == '__main__':
    main()