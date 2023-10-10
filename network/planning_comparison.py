import glob
from sys import argv, stdout
from pathlib import Path
from termcolor import colored
from timeit import default_timer as timer
import argparse, logging
import torch

from generators import compute_traces_with_augmented_states, load_pddl_problem_with_augmented_states
from architecture import g_model_classes

import plan

def _parse_arguments(exec_path : Path):
    default_aggregation = 'max'
    default_debug_level = 0
    default_cycles = 'avoid'
    default_logfile = 'log_plan.txt'
    default_max_length = 500
    default_registry_filename = '../derived_predicates/registry_rules.json'

    # required arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--domain', required=True, type=str, help='domain names')
    parser.add_argument('--original', required=False, type=Path, help='original policy')
    parser.add_argument('--retrained', required=False, type=Path, help='re-trained policy')
    parser.add_argument('--continued', required=False, type=Path, help='policy for which the training was continued')

    #parser.add_argument('--domain', required=True, type=Path, help='domain file')
    #parser.add_argument('--model', required=True, type=Path, help='model file')
    #parser.add_argument('--problem', required=True, type=Path, help='problem file')

    # optional arguments
    parser.add_argument('--aggregation', default=default_aggregation, nargs='?', choices=['add', 'max', 'addmax', 'attention'], help=f'aggregation function for readout (default={default_aggregation})')
    parser.add_argument('--augment', action='store_true', help='augment states with derived predicates')
    parser.add_argument('--cpu', action='store_true', help='use CPU', default=True)
    parser.add_argument('--cycles', type=str, default=default_cycles, choices=['avoid', 'detect'], help=f'how planner handles cycles (default={default_cycles})')
    parser.add_argument('--debug_level', dest='debug_level', type=int, default=default_debug_level, help=f'set debug level (default={default_debug_level})')
    parser.add_argument('--ignore_unsolvable', action='store_true', help='ignore unsolvable states in policy controller', default=True)
    parser.add_argument('--logfile', type=Path, default=default_logfile, help=f'log file (default={default_logfile})')
    parser.add_argument('--max_length', type=int, default=default_max_length, help=f'max trace length (default={default_max_length})')
    parser.add_argument('--print_trace', action='store_true', help='print trace', default=True)
    parser.add_argument('--readout', action='store_true', help='use global readout')
    parser.add_argument('--registry_filename', type=Path, default=default_registry_filename, help=f'registry filename (default={default_registry_filename})')
    parser.add_argument('--registry_key', type=str, default=None, help=f'key into registry (if missing, calculated from domain path)')
    parser.add_argument('--spanner', action='store_true', help='special handling for Spanner problems')
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    return args

def _main(args, model_path, domain_file, problem_file, logger):
    start_time = timer()

    # load model
    use_cpu = args.cpu #hasattr(args, 'cpu') and args.cpu
    use_gpu = not use_cpu and torch.cuda.is_available()
    device = torch.cuda.current_device() if use_gpu else None
    Model = plan._load_model(args)  # TODO: WHAT FUNCTION IS THIS???
    try:
        model = Model.load_from_checkpoint(checkpoint_path=str(model_path), strict=False).to(device)
    except:
        try:
            model = Model.load_from_checkpoint(checkpoint_path=str(model_path), strict=False,
                                               map_location=torch.device('cuda')).to(device)
        except:
            model = Model.load_from_checkpoint(checkpoint_path=str(model_path), strict=False,
                                               map_location=torch.device('cpu')).to(device)
    elapsed_time = timer() - start_time
    logger.info(f"Model '{model_path}' loaded in {elapsed_time:.3f} second(s)")

    logger.info(f"Loading PDDL files: domain='{domain_file}', problem='{problem_file}'")
    registry_filename = args.registry_filename if args.augment else None
    pddl_problem = load_pddl_problem_with_augmented_states(domain_file, problem_file, registry_filename, args.registry_key, logger)
    del pddl_problem['predicates'] #  Why?

    logger.info(f'Executing policy (max_length={args.max_length})')
    start_time = timer()
    is_spanner = args.spanner and 'spanner' in str(domain_file)
    unsolvable_weight = 0.0 if args.ignore_unsolvable else 100000.0
    action_trace, state_trace, value_trace, is_solution, num_evaluations = compute_traces_with_augmented_states(model=model, cycles=args.cycles, max_trace_length=args.max_length, unsolvable_weight=unsolvable_weight, logger=logger, is_spanner=is_spanner, **pddl_problem)
    elapsed_time = timer() - start_time
    logger.info(f'{len(action_trace)} executed action(s) and {num_evaluations} state evaluations(s) in {elapsed_time:.3f} second(s)')

    if is_solution:
        logger.info(colored(f'Found valid plan with {len(action_trace)} action(s) for {problem_file}', 'green', attrs=[ 'bold' ]))
    else:
        logger.info(colored(f'Failed to find a plan for {problem_file}', 'red', attrs=[ 'bold' ]))

    if args.print_trace:
        for index, action in enumerate(action_trace):
            value_from = value_trace[index]
            value_to = value_trace[index + 1]
            logger.info('{}: {} (value change: {:.2f} -> {:.2f} {})'.format(index + 1, action.name, float(value_from), float(value_to), 'D' if float(value_from) > float(value_to) else 'I'))


if __name__ == "__main__":
    # setup timer and exec name
    entry_time = timer()
    exec_path = Path(argv[0]).parent
    exec_name = Path(argv[0]).stem

    # parse arguments
    args = _parse_arguments(exec_path)

    paths_and_directories = [(args.original, "plans_original"), (args.retrained, "plans_retrained"), (args.continued, "plans_continued")]

    for model_path, directory in paths_and_directories:
        domain_file = Path('../data/pddl/' + args.domain + '/test/domain.pddl')
        problem_files = glob.glob(str('../data/pddl/' + args.domain + '/test/' + '*.pddl'))
        print(model_path)
        print(domain_file)
        print("\n")
        print(problem_files)
        print("\n")
        for problem_file in problem_files:
            problem_name = str(Path(problem_file).stem)
            if problem_name == 'domain':
                continue
            print(problem_name)
            # setup logger
            if args.cycles == 'detect':  # TODO: IS THIS CORRECT?
                logfile_name = problem_name + ".markovian"
            elif args.cycles == "avoid":
                logfile_name = problem_name + ".policy"

            log_file = "../plans/" + directory + "/" + args.domain + "/" + logfile_name

            log_level = logging.INFO if args.debug_level == 0 else logging.DEBUG
            logger = plan._get_logger(exec_name + problem_name, log_file, log_level)
            logger.info(f'Call: {" ".join(argv)}')

            # do jobs
            _main(args, model_path, domain_file, problem_file, logger)

            # final stats
            elapsed_time = timer() - entry_time
            logger.info(f'All tasks completed in {elapsed_time:.3f} second(s)')

            del logger
