import argparse
from pathlib import Path

def _parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', required=True, type=Path, help='directory where plans are saved')

    args = parser.parse_args()
    return args

# parses a policy's plans which are given as .policy files
def read_plans(path: Path):
    results = dict()

    for filename in path.glob(f'*.policy'):
        with filename.open('r') as fd:
            plan = []
            reading_plan = False
            for line in fd.readlines():
                line = line.strip('\n')
                if line.find('valid plan') > 0:
                    reading_plan = True
                    continue
                if reading_plan:
                    fields = line.split(' ')
                    if fields[0][-1] == ':' and fields[0][:-1].isdigit():
                        plan.append(' '.join(fields[2:]))
            if reading_plan:
                problem = filename.stem
                results[problem] = dict(length=len(plan), plan=plan)
    return results

def _main(args):
    best_solved = 0
    best_plan_quality = float('inf')
    best_version_dir = None
    for version_dir in args.logdir.glob('version_*'):
        plans = read_plans(version_dir)
        solved = len(plans.keys())
        plan_lengths = [plans[problem]["length"] for problem in plans.keys()]
        try:
            plan_quality = sum(plan_lengths) / solved
        except ZeroDivisionError:
            plan_quality = float('inf')

        print(f"Policy {version_dir} solved {solved} instances with plan quality {plan_quality}")

        # either the current run solved more instances or it achieved a better plan quality
        if (solved > best_solved) or (solved == best_solved and plan_quality < best_plan_quality):
            best_solved = solved
            best_plan_quality = plan_quality
            best_version_dir = version_dir

    print("\n")
    print(f"Best policy {best_version_dir} solved {best_solved} instances with plan quality {best_plan_quality}")


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)
