import argparse
from termcolor import colored
import torch
from timeit import default_timer as timer
import plan
import glob
import pandas as pd
from architecture import set_suboptimal_factor, set_loss_constants
from pathlib import Path
from generators import compute_traces_with_augmented_states
from bugfile_parser import parse_bug_file
from architecture import selfsupervised_suboptimal_loss_no_solvable_labels
from retraining import load_model
from datasets import g_dataset_methods

def load_predicates(args):
    print(colored('Loading datasets...', 'green', attrs = [ 'bold' ]))
    try:
        load_dataset, collate = g_dataset_methods[args.loss]
    except KeyError:
        raise NotImplementedError(f"Loss function '{args.loss}'")

    if args.domain == "reward":
        train = Path('data/states/train/reward/reward/')
    elif args.domain == "delivery":
        train = Path('data/states/train/delivery/delivery/')
    #elif args.domain == ""

    predicates = load_dataset(train, {}, 1, 1, False)[1]

    return predicates, collate

# loads all bug states from a given path
def load_bugs(path):
    bug_files = glob.glob(str(path) + "/*.bugfile")
    translation_pddl = []
    collated_bugs = []
    for bug_file in bug_files:
        bugs, sas = parse_bug_file(bug_file)
        bug_file_name = bug_file.split("/")[-1].split(".")[0]
        sas_file = Path(str(path) + "/" + bug_file_name + ".sas")
        with open(sas_file, "w") as f:
            f.write(sas)
        pddl_directory = "/" + str(path).split("/")[-1] + "/"
        domain_file = Path('data/pddl/' + str(args.domain) + pddl_directory + '/domain.pddl')
        print(domain_file)
        problem_file = Path("data/pddl/" + str(args.domain) + pddl_directory + bug_file_name + ".pddl")
        print(problem_file)

        setup_args = f"--domain {domain_file} --problem {problem_file} --model {None} --sas {sas_file}"
        _, pddl_problem = plan.setup_translation(setup_args)
        translated_bugs = []
        for bug in bugs:
            encoded = plan.translate_pddl(bug.state_vals)
            translated_bugs.append(encoded)

            collated, encoded = plan.translate(bug.state_vals)
            label = torch.tensor([bug.cost_bound])
            solvable_label = torch.tensor([True] * len(encoded))  # we will not use these
            if len(encoded) == 1:
                continue
            collated_bugs.append((label, encoded, solvable_label))

        translation_pddl.append((pddl_problem, domain_file, problem_file, translated_bugs))

    return translation_pddl, collated_bugs

# map a state to a string such that we can check whether we have seen this state before
def state_to_string(state):
    state_string = ""
    for predicate in state[1].keys():  # only need to look at the first state since the successors are fixed
        state_string += f'{predicate}: {state[1][predicate]} '
    return state_string

def _parse_arguments():
    parser = argparse.ArgumentParser()

    # default values for arguments
    default_gpus = 0  # No GPU
    default_max_epochs = None
    default_aggregation = 'max'

    # required arguments
    parser.add_argument('--bugs', required=True, type=Path, help='path to bug dataset')
    parser.add_argument('--logdir', required=True, type=Path, help='directory where policies are saved')
    parser.add_argument('--policy', default=None, type=Path, help='path to policy (.ckpt)')

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

def _process_args(args):
    if (not hasattr(args, 'readout')) or (args.readout is None): args.readout = False
    if (not hasattr(args, 'verbose')) or (args.verbose is None): args.verbose = False
    if not torch.cuda.is_available(): args.gpus = 0  # Ignore GPUs if there is no CUDA capable device.
    if args.max_samples_per_file <= 0: args.max_samples_per_file = None
    set_suboptimal_factor(args.suboptimal_factor)

    # check whether loss constants are properly defined
    if args.loss_constants:
        loss_constants = [ float(c) for c in args.loss_constants.split(',') ]
        if len(loss_constants) != 4 or min(loss_constants) < 0:
            print(colored(f'WARNING: Invalid constants {loss_constants} for loss function, using default values', 'magenta', attrs = [ 'bold' ]))
        else:
            set_loss_constants(loss_constants)

def planning(args, pddl_problem, bug, policy, domain_file, problem_file, device):
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
    #pddl_problem = load_pddl_problem_with_augmented_states(domain_file, problem_file, registry_filename,
    #                                                       args.registry_key, None)
    #del pddl_problem['predicates']  # Why?
    del pddl_problem['initial']
    pddl_problem['initial'] = bug

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

# writes results of a planning run ato a csv file
def save_results(results, policy_path, planning_results):
    results["policy_path"].append(policy_path)
    results["instances"].append(planning_results["instances"])
    results["max_coverage"].append(planning_results["max_coverage"])
    results["min_coverage"].append(planning_results["min_coverage"])
    results["avg_coverage"].append(planning_results["avg_coverage"])
    results["max_quality"].append(planning_results["max_quality"])
    results["min_quality"].append(planning_results["min_quality"])
    results["avg_quality"].append(planning_results["avg_quality"])
    results["max_bug_loss"].append(planning_results["max_bug_loss"])
    results["min_bug_loss"].append(planning_results["min_bug_loss"])
    results["avg_bug_loss"].append(planning_results["avg_bug_loss"])
    results["plans_directory"].append(planning_results["plans_directory"])
    results.update(vars(args))


def _main(args):
    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")

    # load bugs
    translated_bugs, collated_bugs = load_bugs(args.bugs)
    num_bugs = [len(translation[3]) for translation in translated_bugs]
    num_bugs = sum(num_bugs)

    # load model
    predicates, collate = load_predicates(args)
    gnn_model = load_model(args, predicates, path=args.policy, retrain=False)

    print(colored('Running policies on test instances', 'red', attrs=['bold']))

    results = {
        "policy_path": [],
        "instances": [],
        "max_coverage": [],
        "min_coverage": [],
        "avg_coverage": [],
        "max_quality": [],
        "min_quality": [],
        "avg_quality": [],
        "max_bug_loss": [],
        "min_bug_loss": [],
        "avg_bug_loss": [],
        "plans_directory": [],
    }

    # initialize metrics
    best_planning_run = None
    coverages = []
    best_coverage = 0
    qualities = []
    bugs_losses = []
    for i in range(args.runs):
        # TODO: EVALUATION
        with torch.no_grad():
            losses = []
            for bug in collated_bugs:
                try:
                    labels, collated_states_with_object_counts, solvable_labels, state_counts = collate([bug])

                    output = gnn_model(collated_states_with_object_counts)
                    loss = selfsupervised_suboptimal_loss_no_solvable_labels(output, labels, state_counts, device)
                    losses.append(loss.item())
                except:
                    print(f"Error processing bug")
                    print(bug)
                    continue
            bugs_losses.append(sum(losses) / len(losses))

        # create directory for current run
        version_path = args.logdir / f"version_{i}"
        version_path.mkdir(parents=True, exist_ok=True)
        # initialize metrics for current run
        plan_lengths = []
        is_solutions = []
        for translation in translated_bugs:
            pddl_problem = translation[0]
            domain_file = translation[1]
            problem_file = translation[2]
            bugs = translation[3]

            problem_name = str(Path(problem_file).stem)

            # run planning
            bug_id = -1
            for bug in bugs:
                bug_id += 1
                if args.cycles == 'detect':
                    logfile_name = problem_name + f"_{bug_id}" + ".markovian"
                else:
                    logfile_name = problem_name + f"_{bug_id}" + ".policy"
                log_file = version_path / logfile_name
                result_string, action_trace, is_solution = planning(args, pddl_problem, bug, args.policy, domain_file, problem_file, device)

                # store results
                with open(log_file, "w") as f:
                    f.write(result_string)

                is_solutions.append(is_solution)
                if is_solution:
                    print(f"Solved problem {problem_name} {bug_id} with plan length: {len(action_trace)}")
                else:
                    print(f"Failed to solve problem {problem_name} {bug_id}")
                plan_lengths.append(len(action_trace))

                # print(result_string)

        # compute coverage of this run and check whether it is the best one yet
        coverage = sum(is_solutions)
        coverages.append(coverage)
        try:
            plan_quality = sum(plan_lengths) / coverage
        except:
            continue
        qualities.append(plan_quality)
        if coverage > best_coverage:
            best_coverage = coverage
            best_planning_run = str(version_path)

        print(coverages)
        planning_results = dict(instances=num_bugs, max_coverage=max(coverages),
                                min_coverage=min(coverages), avg_coverage=sum(coverages) / len(coverages),
                                max_quality=max(qualities), min_quality=min(qualities),
                                avg_quality=sum(qualities) / len(qualities), max_bug_loss=max(bugs_losses),
                                min_bug_loss=min(bugs_losses), avg_bug_loss=sum(bugs_losses) / len(bugs_losses),
                                plans_directory=best_planning_run)
        print(planning_results)

        # save results of the best run
        save_results(results, args.policy, planning_results)

    print(colored('Storing results', 'red', attrs=['bold']))
    print(results)
    results = pd.DataFrame(results)
    results.to_csv(args.logdir / "results.csv")
    print(results)


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)
