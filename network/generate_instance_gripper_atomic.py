import os
import argparse
import random
from pathlib import Path
from src.generators.generate_trajectories import load_pddl_problem


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--min_size", required=True, type=int, help="Minimum size of the generated instances")
    parser.add_argument("--max_size", required=True, type=int, help="Maximum size of the generated instances")
    parser.add_argument("--instances_per_size", required=True, type=int,
                        help="Number of instances to generate for each size")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output directory for the generated instances")
    parser.add_argument("--domain", type=Path, help="Path to domain file", default=Path("/Users/nicola_mueller/Desktop/eval_general_policies/data/gripper/domain.pddl"))

    return parser.parse_args()

def get_size(num_balls):
    return num_balls

def generate_instance(num_balls):
    instance = ""
    instance += f"(define (problem gripper-{num_balls})"
    instance += f"\n(:domain gripper-strips)"
    instance += f"\n(:objects "
    instance += f"rooma roomb left right "
    for i in range(num_balls):
        instance += f"ball{i+1} "
    instance += ")"
    instance += f"\n(:init"
    instance += f"\n(room rooma)"
    instance += f"\n(room roomb)"
    instance += f"\n(gripper left)"
    instance += f"\n(gripper right)"
    for i in range(num_balls):
        instance += f"\n(ball ball{i+1})"
    instance += f"\n(free left)"
    instance += f"\n(free right)"
    # place a random subset of balls in rooma
    balls_in_rooma = random.sample(range(num_balls-1), random.randint(0, num_balls-1))
    for i in range(num_balls):
        if i in balls_in_rooma:
            instance += f"\n(at ball{i+1} rooma)"
    for i in range(num_balls):
        if i not in balls_in_rooma:
            instance += f"\n(at ball{i+1} roomb)"
    instance += f"\n(at-robby rooma)"
    instance += f"\n)"
    instance += f"\n(:goal"
    instance += f"\n(and"
    for i in range(num_balls):
        if i in balls_in_rooma:
            instance += f"\n(at ball{i+1} roomb)"
    instance += f"\n)"
    instance += f"\n)"
    instance += f"\n)"
    return instance


if __name__ == "__main__":
    args = parse_arguments()

    for num_balls in range(args.min_size, args.max_size + 1):
        configs = []
        # generate random configurations for the instances of the current size
        for i in range(args.instances_per_size):
            configs.append((i, num_balls))

        # create the directory for the instances of the current size
        instance_size = get_size(num_balls)
        directory_name = "size_{}".format(instance_size)
        size_directory = args.output_dir / directory_name
        size_directory.mkdir(parents=True, exist_ok=True)

        # generate the instances
        problem_cache = []
        for config in configs:
            i, num_balls = config
            instance_name = f"instance_{num_balls}_{i}"
            # create directory for the instance
            instance_directory = size_directory / instance_name
            instance_directory.mkdir(parents=True, exist_ok=True)
            # generate the instance
            instance_file = instance_directory / f"{instance_name}.pddl"
            print(f"Generating instance {instance_name}")
            # os.system("/Users/nicola_mueller/Desktop/eval_general_policies/domains/gripper/gripper -n {} > {}".format(num_balls, instance_file))
            with open(instance_file, "w") as f:
                f.write(generate_instance(num_balls))

            # load the problem and ensure that no equivalent instance has been generated before
            generated_problem = load_pddl_problem(args.domain, instance_file, grounding=False)
            duplicate = False
            for problem in problem_cache:
                if generated_problem['initial'] == problem['initial'] and generated_problem['goal'] == problem['goal']:
                    # delete directory
                    instance_file.unlink()
                    instance_directory.rmdir()
                    duplicate = True
                    print(f"Duplicate instance {instance_name}!")
                    break
            if not duplicate:
                problem_cache.append(generated_problem)