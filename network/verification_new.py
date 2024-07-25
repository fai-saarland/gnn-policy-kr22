import argparse
from termcolor import colored
import torch
from timeit import default_timer as timer
import plan
import pandas as pd
from pathlib import Path
from generators import load_pddl_problem_with_augmented_states, compute_traces_with_augmented_states
from verification_generators.gripper.generate_gripper import generate_instance_gripper
from verification_generators.visitall.generate_visitall import generate_instance_visitall
from verification_generators.blocks.generate_blocks import generate_instance_blocks


def _parse_arguments():
    parser = argparse.ArgumentParser()

    # default values for arguments
    default_gpus = 0  # No GPU

    # required arguments
    parser.add_argument('--policy', required=True, type=Path, help='path to policy (.ckpt)')
    parser.add_argument('--logdir', required=True, type=Path, help='directory where policies are saved')
    parser.add_argument('--min_size', required=True, type=int, help='minimum size of the generated instances')
    parser.add_argument('--max_size', required=True, type=int, help='maximum size of the generated instances')

    # default values for arguments
    default_gpus = 0  # No GPU
    default_aggregation = 'max'

    # arguments with meaningful default values
    parser.add_argument('--runs', type=int, default=1, help='number of planning runs per instance')
    parser.add_argument('--gpus', default=default_gpus, type=int, help=f'number of GPUs to use (default={default_gpus})')
    parser.add_argument('--aggregation', default=default_aggregation, nargs='?',
                        choices=['add', 'max', 'addmax', 'attention', 'planformer'],
                        help=f'readout aggregation function (default={default_aggregation})')
    parser.add_argument('--readout', action='store_true', help=f'use global readout at each iteration')

    # stuff that is required for loading a model
    default_size = 64
    default_iterations = 30
    default_batch_size = 64
    default_loss_constants = None
    default_learning_rate = 0.0002
    default_suboptimal_factor = 2.0
    default_l1 = 0.0
    default_weight_decay = 0.0
    default_gradient_accumulation = 1
    default_patience = 50
    default_gradient_clip = 0.1
    default_loss = "selfsupervised_suboptimal"
    default_max_samples_per_file = 1000
    default_max_samples = None

    parser.add_argument('--loss', default=default_loss, nargs='?',
                        choices=['supervised_optimal', 'selfsupervised_optimal', 'selfsupervised_suboptimal',
                                 'selfsupervised_suboptimal2', 'unsupervised_optimal', 'unsupervised_suboptimal',
                                 'online_optimal'])
    parser.add_argument('--size', default=default_size, type=int,
                        help=f'number of features per object (default={default_size})')
    parser.add_argument('--iterations', default=default_iterations, type=int,
                        help=f'number of convolutions (default={default_iterations})')
    parser.add_argument('--batch_size', default=default_batch_size, type=int,
                        help=f'maximum size of batches (default={default_batch_size})')
    parser.add_argument('--loss_constants', default=default_loss_constants, type=str,
                        help=f'constants (multipliers) in loss function (default={default_loss_constants})')
    parser.add_argument('--learning_rate', default=default_learning_rate, type=float,
                        help=f'learning rate of training session (default={default_learning_rate})')
    parser.add_argument('--suboptimal_factor', default=default_suboptimal_factor, type=float,
                        help=f'approximation factor of suboptimal learning (default={default_suboptimal_factor})')
    parser.add_argument('--l1', default=default_l1, type=float,
                        help=f'strength of L1 regularization (default={default_l1})')
    parser.add_argument('--weight_decay', default=default_weight_decay, type=float,
                        help=f'strength of weight decay regularization (default={default_weight_decay})')
    parser.add_argument('--gradient_accumulation', default=default_gradient_accumulation, type=int,
                        help=f'number of gradients to accumulate before step (default={default_gradient_accumulation})')
    parser.add_argument('--patience', default=default_patience, type=int,
                        help=f'patience for early stopping (default={default_patience})')
    parser.add_argument('--gradient_clip', default=default_gradient_clip, type=float,
                        help=f'gradient clip value (default={default_gradient_clip})')
    parser.add_argument('--max_samples_per_file', default=default_max_samples_per_file, type=int,
                        help=f'maximum number of states per dataset (default={default_max_samples_per_file})')
    parser.add_argument('--max_samples', default=default_max_samples, type=int,
                        help=f'maximum number of states in total (default={default_max_samples})')

    default_debug_level = 0
    default_cycles = 'avoid'
    default_logfile = 'log_plan.txt'
    default_max_length = 500
    default_registry_filename = '../derived_predicates/registry_rules.json'

    parser.add_argument('--domain', required=True, type=str, help='domain name')

    # optional arguments
    parser.add_argument('--augment', action='store_true', help='augment states with derived predicates')
    parser.add_argument('--cpu', action='store_true', help='use CPU', default=True)
    parser.add_argument('--cycles', type=str, default=default_cycles, choices=['avoid', 'detect'],
                        help=f'how planner handles cycles (default={default_cycles})')
    parser.add_argument('--debug_level', dest='debug_level', type=int, default=default_debug_level,
                        help=f'set debug level (default={default_debug_level})')
    parser.add_argument('--ignore_unsolvable', action='store_true',
                        help='ignore unsolvable states in policy controller', default=True)
    parser.add_argument('--logfile', type=Path, default=default_logfile, help=f'log file (default={default_logfile})')
    parser.add_argument('--max_length', type=int, default=default_max_length,
                        help=f'max trace length (default={default_max_length})')
    parser.add_argument('--print_trace', action='store_true', help='print trace', default=True)
    parser.add_argument('--registry_filename', type=Path, default=default_registry_filename,
                        help=f'registry filename (default={default_registry_filename})')
    parser.add_argument('--registry_key', type=str, default=None,
                        help=f'key into registry (if missing, calculated from domain path)')
    parser.add_argument('--spanner', action='store_true', help='special handling for Spanner problems')

    args = parser.parse_args()
    return args

def planning(args, policy, domain_file, problem_file, device):
    start_time = timer()
    result_string = ""

    # load model
    Model = plan._load_model(args)
    try:
        model = Model.load_from_checkpoint(checkpoint_path=str(policy), strict=False).to(device)
    except:
        try:
            model = Model.load_from_checkpoint(checkpoint_path=str(policy), strict=False,
                                               map_location=torch.device('cuda')).to(device)
        except:
            model = Model.load_from_checkpoint(checkpoint_path=str(policy), strict=False,
                                               map_location=torch.device('cpu')).to(device)
    elapsed_time = timer() - start_time

    result_string = result_string + f"Model '{policy}' loaded in {elapsed_time:.3f} second(s)"
    result_string = result_string + "\n"
    result_string = result_string + f"Loading PDDL files: domain='{domain_file}', problem='{problem_file}'"
    result_string = result_string + "\n"

    registry_filename = args.registry_filename if args.augment else None
    pddl_problem = load_pddl_problem_with_augmented_states(domain_file, problem_file, registry_filename,
                                                           args.registry_key, None)
    del pddl_problem['predicates']  # Why?

    result_string = result_string + f'Executing policy (max_length={args.max_length})'
    result_string = result_string + "\n"
    start_time = timer()
    is_spanner = args.spanner and 'spanner' in str(domain_file)
    unsolvable_weight = 0.0 if args.ignore_unsolvable else 100000.0
    action_trace, state_trace, value_trace, is_solution, num_evaluations = compute_traces_with_augmented_states(
        model=model, cycles=args.cycles, max_trace_length=args.max_length, unsolvable_weight=unsolvable_weight,
        logger=None, is_spanner=is_spanner, **pddl_problem)
    elapsed_time = timer() - start_time
    result_string = result_string + f'{len(action_trace)} executed action(s) and {num_evaluations} state evaluations(s) in {elapsed_time:.3f} second(s)'
    result_string = result_string + "\n"

    if is_solution:
        result_string = result_string + f'Found valid plan with {len(action_trace)} action(s) for {problem_file}'
        result_string = result_string + "\n"
    else:
        result_string = result_string + f'Failed to find a plan for {problem_file}'
        result_string = result_string + "\n"

    if args.print_trace:
        for index, action in enumerate(action_trace):
            value_from = value_trace[index]
            value_to = value_trace[index + 1]
            result_string = result_string + '{}: {} (value change: {:.2f} -> {:.2f} {})'.format(index + 1, action.name, float(value_from), float(value_to), 'D' if float(value_from) > float(value_to) else 'I')
            result_string = result_string + "\n"

    return result_string, action_trace, is_solution


def _main(args):
    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")

    domain_file = Path('data/pddl/' + args.domain + '/test/domain.pddl')

    results = {
        "size": [],
        "avg_coverage": [],
        "avg_plan_length": [],
        "num_instances": []
    }
    for size in range(args.min_size, args.max_size + 1):
        results["size"].append(size)
        coverages = []
        plan_lengths = []
        # create directory for storing generated instance files
        instances_dir = args.logdir / f"generated_instances/size_{size}"
        instances_dir.mkdir(parents=True, exist_ok=True)

        num_instances_per_size = 50
        instances_counter = 0
        for i in range(num_instances_per_size):
            if args.domain == "gripper":
                instance_file = generate_instance_gripper(size, instances_dir)
            elif args.domain == "visitall":
                instance_file = generate_instance_visitall(size, instances_dir)
            elif args.domain == "blocks":
                instance_file = generate_instance_blocks(size, instances_dir)
            instances_counter += 1

            result_string, action_trace, is_solution = planning(args, args.policy, domain_file, instance_file, device)
            if is_solution:
                coverages.append(1)
                plan_lengths.append(len(action_trace))
                print(colored(f"Instance {instance_file.stem} solved after {len(action_trace)} steps!", 'green', attrs=['bold']))
            else:
                coverages.append(0)
                print(colored(f"Instance {instance_file.stem} not solved after {len(action_trace)} steps!", 'red', attrs=['bold']))
            #print(action_trace)

        avg_coverage = sum(coverages) / len(coverages)
        if len(plan_lengths) == 0:
            avg_plan_length = 0
        else:
            avg_plan_length = sum(plan_lengths) / len(plan_lengths)
        results["avg_coverage"].append(avg_coverage)
        results["avg_plan_length"].append(avg_plan_length)
        results["num_instances"].append(instances_counter)

        results_data_frame = pd.DataFrame(results)
        results_data_frame.to_csv(args.logdir / "results.csv")


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)
