import os
import random

def generate_instance_blocks(num_blocks, instances_dir):
    seed = random.randint(0, 10000)
    instance_name = f"instance_{num_blocks}_{seed}"
    # generate the instance
    instance_file = instances_dir / f"{instance_name}.pddl"
    os.system(f"network/verification_generators/blocks/blocksworld {4} {num_blocks} > {instance_file}")

    return instance_file
