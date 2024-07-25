import os
import random

def generate_instance_visitall(width, instances_dir):
    goal_ratio = random.choice([0.5, 1.0])
    seed = random.randint(0, 10000)
    instance_name = f"instance_{width}_{goal_ratio}_{seed}"
    # generate the instance
    instance_file = instances_dir / f"{instance_name}.pddl"
    os.system(f"network/verification_generators/visitall/grid -n {width} -r {goal_ratio} -s {seed} > {instance_file}")

    return instance_file
