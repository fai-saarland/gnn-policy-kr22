import argparse
from termcolor import colored
import torch
import pandas as pd
import glob
from pathlib import Path
from training_old import load_dataset, planning
from training_new import model_classes

def load_predicates(args):
    print(colored('Loading datasets...', 'green', attrs = [ 'bold' ]))

    if args.domain == "gripper":
        dataset = Path('data_old/supervised/optimal/validation/gripper-atomic/gripper-atomic/')
    elif args.domain == "blocks-clear":
        dataset = Path('data_old/supervised/optimal/validation/blocks-clear/blocks-clear/')
    elif args.domain == "visitall":
        dataset = Path('data_old/supervised/optimal/validation/visitall-atomic/visitall-atomic/')
    elif args.domain == "parking-behind":
        dataset = Path('data_old/supervised/optimal/validation/parking-behind/parking-behind/')
    elif args.domain == "satellite":
        dataset = Path('data_old/supervised/optimal/validation/satellite/satellite/')

    decoded_predicates = load_dataset(dataset, 1)[2]

    i = 0
    decoded_predicate_dict = {}
    decoded_predicate_ids = {}
    max_arity = 0
    for predicate, arity in decoded_predicates:
        decoded_predicate_dict[predicate] = arity
        if arity > max_arity:
            max_arity = arity
        decoded_predicate_ids[predicate] = i
        i += 1

    return decoded_predicate_dict, decoded_predicate_ids, max_arity

def parse_optimal_plan_lengths(args):
    logs_path = ""
    if args.domain == "blocks-clear":
        logs_path = "data_old/logs/log_blocks-clear.txt"
    elif args.domain == "blocks-on":
        logs_path = "data_old/logs/log_blocks-on.txt"
    elif args.domain == "gripper":
        logs_path = "data_old/logs/log_gripper.txt"
    elif args.domain == "visitall":
        logs_path = "data_old/logs/log_visitall.txt"
    elif args.domain == "parking-behind":
        logs_path = "data_old/logs/log_parking-behind.txt"
    elif args.domain == "satellite":
        logs_path = "data_old/logs/log_satellite.txt"

    optimal_plan_lengths = {}
    with open(logs_path, "r") as f:
        instance_name = None
        for line in f.readlines():
            line = line.strip('\n')
            if line.find('Input:') > 0:
                instance_name = line.split(' ')[-1].removesuffix('.pddl')
                continue
            if instance_name is not None:
                if line.find('solves') > 0:
                    fields = line.split(' ')
                    for i in range(len(fields)):
                        if fields[i] == 'cost:':
                            plan_length = int(fields[i+1].removesuffix(')'))
                            optimal_plan_lengths[instance_name] = plan_length
                            instance_name = None

    return optimal_plan_lengths

def _parse_arguments():
    parser = argparse.ArgumentParser()

    # default values for arguments
    default_gpus = 0  # No GPU

    # required arguments
    parser.add_argument('--policy', required=True, type=Path, help='path to policy (.ckpt)')
    parser.add_argument('--type', required=True, type=str, help='type of network')
    parser.add_argument('--logdir', required=True, type=Path, help='directory where policies are saved')

    # arguments with meaningful default values
    parser.add_argument('--runs', type=int, default=1, help='number of policy runs per instance')
    parser.add_argument('--gpus', default=default_gpus, type=int, help=f'number of GPUs to use (default={default_gpus})')

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

# TODO: NO CYCLE AVOIDANCE!!!!
def _main(args):
    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")

    # load optimal plan lengths
    optimal_plan_lengths = parse_optimal_plan_lengths(args)


    # load model
    decoded_predicate_dict, decoded_predicate_ids, max_arity = load_predicates(args)
    # load model
    Model = model_classes[(args.type, "ADD", "MSE")]
    try:
        model = Model.load_from_checkpoint(checkpoint_path=str(args.policy), strict=False).to(device)
    except:
        try:
            model = Model.load_from_checkpoint(checkpoint_path=str(args.policy), strict=False,
                                               map_location=torch.device('cuda')).to(device)
        except:
            model = Model.load_from_checkpoint(checkpoint_path=str(args.policy), strict=False,
                                               map_location=torch.device('cpu')).to(device)
    model = model.to(device)
    # deactivate dropout!
    model.training = False
    model.eval()

    domain_file = Path('data_old/pddl/' + args.domain + '/test/domain.pddl')
    problem_files = glob.glob(str('data_old/pddl/' + args.domain + '/test/' + '*.pddl'))

    problem_dict = {}
    for problem_file in problem_files:
        problem_name = str(Path(problem_file).stem)
        if problem_name == 'domain':
            continue
        else:
            problem_dict[problem_name] = []

    # create directory for current run
    version_path = args.logdir / f"version_{0}"
    version_path.mkdir(parents=True, exist_ok=True)

    # initialize metrics for current run
    plan_lengths = []
    is_solutions = []
    for problem_file in problem_files:
        problem_name = str(Path(problem_file).stem)
        if problem_name == 'domain':
            continue
        problem_dict[problem_name].append(optimal_plan_lengths[problem_name])

        if args.cycles == 'detect':
            logfile_name = problem_name + ".markovian"
        else:
            logfile_name = problem_name + ".policy"
        log_file = version_path / logfile_name

        # run planning
        result_string, action_trace, is_solution = planning(decoded_predicate_dict, decoded_predicate_ids,
                                                            max_arity, args, args.policy, model, domain_file,
                                                            problem_file, device)

        # store results
        with open(log_file, "w") as f:
            f.write(result_string)

        if is_solution:
            problem_dict[problem_name].append(len(action_trace))
            problem_dict[problem_name].append(len(action_trace)/optimal_plan_lengths[problem_name])
            print(f"Solved problem {problem_name} with plan length: {len(action_trace)}")
        else:
            problem_dict[problem_name].append(-1)
            problem_dict[problem_name].append(-1)
            print(f"Failed to solve problem {problem_name}")

    results = {
        "instance": [],
        "num_instances": [],
        "coverage": [],
        "coverage_optimal": [],
        "coverage_suboptimal": [],
        "optimal_plan_length": [],
        "policy_plan_length": [],
        "optimal_plan_length_avg": [],
        "policy_plan_length_avg": [],
        "length_factor": [],
    }
    solved_optimal_plan_lengths = []
    solved_policy_plan_lengths = []
    for problem_name in problem_dict.keys():
        results["instance"].append(problem_name)
        results["num_instances"].append(1)
        results["coverage"].append(problem_dict[problem_name][1] != -1)
        results["coverage_optimal"].append(problem_dict[problem_name][0] == problem_dict[problem_name][1])
        results["coverage_suboptimal"].append(problem_dict[problem_name][1] != -1 and problem_dict[problem_name][0] != problem_dict[problem_name][1])
        results["optimal_plan_length"].append(problem_dict[problem_name][0])
        results["optimal_plan_length_avg"].append(problem_dict[problem_name][0])
        results["policy_plan_length"].append(problem_dict[problem_name][1])
        results["policy_plan_length_avg"].append(problem_dict[problem_name][1])
        results["length_factor"].append(0 if problem_dict[problem_name][1] != -1 else problem_dict[problem_name][2])

        if problem_dict[problem_name][1] != -1:
            solved_optimal_plan_lengths.append(problem_dict[problem_name][0])
            solved_policy_plan_lengths.append(problem_dict[problem_name][1])


    total_num_instances = sum(results["num_instances"])
    total_coverage = sum(results["coverage"])
    total_coverage_optimal = sum(results["coverage_optimal"])
    total_coverage_suboptimal = sum(results["coverage_suboptimal"])
    total_optimal_plan_length = sum(solved_optimal_plan_lengths)
    total_optimal_plan_length_avg = total_optimal_plan_length / total_coverage
    total_policy_plan_length = sum(solved_policy_plan_lengths)
    total_policy_plan_length_avg = total_policy_plan_length / total_coverage
    total_length_factor = total_policy_plan_length / total_optimal_plan_length
    results["instance"] = ["total"] + results["instance"]
    results["num_instances"] = [total_num_instances] + results["num_instances"]
    results["coverage"] = [total_coverage] + results["coverage"]
    results["coverage_optimal"] = [total_coverage_optimal] + results["coverage_optimal"]
    results["coverage_suboptimal"] = [total_coverage_suboptimal] + results["coverage_suboptimal"]
    results["optimal_plan_length"] = [total_optimal_plan_length] + results["optimal_plan_length"]
    results["optimal_plan_length_avg"] = [total_optimal_plan_length_avg] + results["optimal_plan_length_avg"]
    results["policy_plan_length"] = [total_policy_plan_length] + results["policy_plan_length"]
    results["policy_plan_length_avg"] = [total_policy_plan_length_avg] + results["policy_plan_length_avg"]
    results["length_factor"] = [total_length_factor] + results["length_factor"]

    print(results)

    print(colored('Storing results', 'red', attrs=['bold']))
    results = pd.DataFrame(results)
    results.to_csv(args.logdir / "results.csv")

if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)
