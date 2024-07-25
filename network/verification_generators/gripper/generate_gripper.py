def generate_instance_gripper(num_balls, instances_dir):
    instance = ""
    instance += f"(define (problem gripper-{num_balls})"
    instance += f"\n(:domain gripper-strips)"
    instance += f"\n(:objects "
    instance += f"rooma roomb "
    for i in reversed(range(num_balls)):
        instance += f"ball{i+1} "
    instance += " left right)"
    instance += f"\n(:init"
    instance += f"\n(room rooma)"
    instance += f"\n(room roomb)"
    for i in reversed(range(num_balls)):
        instance += f"\n(ball ball{i+1})"
    instance += f"\n(at-robby rooma)"
    instance += f"\n(free left)"
    instance += f"\n(free right)"
    # place all balls in room a
    for i in reversed(range(num_balls)):
        instance += f"\n(at ball{i+1} rooma)"
    instance += f"\n(gripper left)"
    instance += f"\n(gripper right)"
    instance += f"\n)"
    instance += f"\n(:goal"
    instance += f"\n(and"
    # all balls must be in room b
    for i in reversed(range(num_balls)):
        instance += f"\n(at ball{i+1} roomb)"
    instance += f"\n)"
    instance += f"\n)"
    instance += f"\n)"

    instance_file = instances_dir / f"instance_size_{num_balls}.pddl"
    with open(instance_file, 'w') as f:
        f.write(instance)

    return instance_file