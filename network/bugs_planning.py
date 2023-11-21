import argparse
import logging
from termcolor import colored
import pytorch_lightning as pl
import torch
import os
import re
from timeit import default_timer as timer
from sys import argv
import plan
import glob
import pandas as pd
from architecture import set_suboptimal_factor, set_loss_constants
from pathlib import Path
from generators import load_pddl_problem_with_augmented_states, compute_traces_with_augmented_states
from bugfile_parser import parse_bug_file

# loads all bug states from a given path
def load_bugs(path):
    bug_files = glob.glob(str(path) + "/*.bugfile")
    translation = []
    for bug_file in bug_files[0:1]:
        bugs, sas = parse_bug_file(bug_file)
        bug_file_name = bug_file.split("/")[-1].split(".")[0]
        sas_file = Path(str(path) + "/" + bug_file_name + ".sas")
        with open(sas_file, "w") as f:
            f.write(sas)
        domain_file = Path('data/pddl/' + str(args.domain) + '/test/domain.pddl')
        problem_file = Path("data/pddl/" + str(args.domain) + "/train/" + bug_file_name + ".pddl")

        setup_args = f"--domain {domain_file} --problem {problem_file} --model {None} --sas {sas_file}"
        plan.setup_translation(setup_args)
        translated_bugs = []
        for bug in bugs:
            encoded = plan.translate_pddl(bug.state_vals)
            translated_bugs.append(encoded)

        translation.append((domain_file, problem_file, translated_bugs))

    return translation

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

def planning(args, bug, policy, domain_file, problem_file, device):
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
    results["best_plan_quality"].append(planning_results["best_plan_quality"])
    results["plans_directory"].append(planning_results["plans_directory"])
    results.update(vars(args))


def _main(args):
    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")

    # TODO: STEP 9: PLANNING
    print(colored('Running policies on test instances', 'red', attrs=['bold']))


    results = {
        "policy_path": [],
        "instances": [],
        "max_coverage": [],
        "min_coverage": [],
        "avg_coverage": [],
        "best_plan_quality": [],
        "plans_directory": [],
    }

    # load files for planning
    translated_bugs = load_bugs(args.bugs)
    num_bugs = [len(translation[2]) for translation in translated_bugs]
    num_bugs = sum(num_bugs)

    # initialize metrics
    best_coverage = 0
    best_planning_run = None
    coverages = []
    for i in range(args.runs):
        # create directory for current run
        version_path = args.logdir / f"version_{i}"
        version_path.mkdir(parents=True, exist_ok=True)
        # initialize metrics for current run
        plan_lengths = []
        is_solutions = []
        for translation in translated_bugs:
            domain_file = translation[0]
            problem_file = translation[1]
            bugs = translation[2]

            # run planning
            bug_id = -1
            for bug in bugs:
                bug_id += 1
                problem_name = str(Path(problem_file).stem)
                if args.cycles == 'detect':
                    logfile_name = problem_name + f"_{bug_id}" + ".markovian"
                else:
                    logfile_name = problem_name + f"_{bug_id}" + ".policy"
                log_file = version_path / logfile_name
                result_string, action_trace, is_solution = planning(args, bug, args.policy, domain_file, problem_file, device)

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
        if coverage > best_coverage:
            best_coverage = coverage
            best_plan_quality = plan_quality
            best_planning_run = str(version_path)

        print(coverages)
        planning_results = dict(instances=num_bugs, max_coverage=max(coverages),
                                             min_coverage=min(coverages), avg_coverage=sum(coverages) / len(coverages),
                                             best_plan_quality=best_plan_quality, plans_directory=best_planning_run)
        print(planning_results)

        # save results of the best run
        save_results(results, args.policy, planning_results)

    print(colored('Storing results', 'red', attrs=['bold']))
    results = pd.DataFrame(results)
    results.to_csv(args.logdir / "results.csv")
    print(results)


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)
