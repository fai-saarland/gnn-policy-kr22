from sys import stdin, stdout, argv
from timeit import default_timer as timer
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

from sys import path as sys_path
sys_path.append('../RelationalNeuralNetwork')
from datasets.protobuf import State, LabeledProblem, Predicate, LabeledState, State, Atom
from datasets.dataset import load_problem
from augmentation import load_registry, get_registry_record, update_registry_record, construct_augmentation_function, augment_ext_state

def _get_logger(name: str, log_file: Path, level = logging.INFO):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    # add stdout handler
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s')
    console = logging.StreamHandler(stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # add file handler
    if log_file != '':
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s')
        file_handler = logging.FileHandler(str(log_file), 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def _parse_arguments(exec_path: Path):
    default_debug_level = 0
    default_rules_filename = exec_path / 'registry_rules.json'

    parser = argparse.ArgumentParser()
    parser.add_argument('states', type=Path, help='states file, or path for folder containing .states files')
    parser.add_argument('domain', nargs='?', type=str, default='', help='domain name (key into rules file)')
    parser.add_argument('--debug_level', type=int, default=default_debug_level, help=f'set debug level (default={default_debug_level})')
    parser.add_argument('--force', action='store_true', help=f'force augmentation even if .augmented file exists')
    parser.add_argument('--recursive', action='store_true', help=f'recursively process .states files in given path')
    parser.add_argument('--rules', dest='rules', type=str, default=default_rules_filename, help=f'rules file (default={default_rules_filename})')
    return parser.parse_args()

def _process_single_file(states_filename: Path, output_filename, domain: str, registry_record: dict) -> int:
    global logger
    start_time = timer()

    # load problem elements
    problem = load_problem(states_filename)
    predicates = problem.Predicates
    objects = problem.Objects
    facts = problem.Facts
    goals = problem.Goals
    labeled_states = problem.LabeledStates

    # report some stats
    logger.info(f'Problem: #predicates={len(predicates)}, #objects={len(objects)}, #facts={len(facts)}, #states={len(labeled_states)}')
    logger.info(f'Problem: predicates={{{",".join([ pred.Name + "/" + str(pred.Arity) for pred in predicates ])}}}')
    logger.info(f'Problem: objects={{{",".join([ obj.Name for obj in objects ])}}}')

    # get sets of static and dynamic predicates
    all_predicates = set([ predicate.Name for predicate in predicates ])
    static_predicates = set([ predicates[atom.PredicateId].Name for atom in facts ])
    dynamic_predicates = all_predicates - static_predicates

    # get augmentation function
    update_registry_record(registry_record, static_predicates, dynamic_predicates)
    augment_fn = construct_augmentation_function(registry_record, predicates)

    # get facts from goals, and compute goal predicates (these are not added to predicates)
    logger.debug(f'**** AUGMENTING GOALS')
    proxy = State(Atoms=goals)
    _, ext_goals = augment_fn(proxy, 'static', logger=logger)
    ext_additional = dict([ (key + '@', ext_goals[key]) for key in ext_goals ])
    logger.debug(f'ext_goals={ext_goals}, ext_additional={ext_additional}')

    # augment facts
    logger.debug(f'**** AUGMENTING FACTS')
    proxy = State(Atoms=facts)
    state_facts, ext_facts = augment_fn(proxy, 'static', additional=ext_additional, logger=logger)
    augmented_facts = state_facts.Atoms
    augment_ext_state(ext_additional, ext_facts)
    logger.debug(f'ext_additonal={ext_additional}')

    # augment states
    logger.debug(f'**** AUGMENTING STATES')
    augmented_labeled_states = []
    for labeled_state in tqdm(labeled_states, desc=f'Augmenting states in {states_filename.name}', file=stdout):
        label = labeled_state.Label
        augmented_state, ext_state = augment_fn(labeled_state.State, 'dynamic', additional=ext_additional)
        if logger: logger.debug(f'ext_state={ext_state}')
        augmented_successors = [ augment_fn(succ, 'dynamic', additional=ext_additional)[0] for succ in labeled_state.SuccessorStates ]
        augmented_labeled_state = LabeledState(Label=label, State=augmented_state, SuccessorStates=augmented_successors)
        augmented_labeled_states.append(augmented_labeled_state)
    assert len(augmented_labeled_states) == len(labeled_states)

    # create augmented problem and serialize it to disk
    augmented_problem = LabeledProblem(Objects=objects, Predicates=predicates, Facts=augmented_facts, Goals=goals, LabeledStates=augmented_labeled_states)
    with output_filename.open('wb') as fd:
        fd.write(augmented_problem.SerializeToString())

    elapsed_time = timer() - start_time
    logger.info(f"Wrote '{output_filename}' [processing {len(augmented_labeled_states)} state(s) took {elapsed_time:.3f} second(s)]")

    return len(augmented_labeled_states)


if __name__ == "__main__":
    # setup timer and exec name
    entry_time = timer()
    exec_path = Path(argv[0]).parent
    exec_name = Path(argv[0]).stem

    # parse arguments
    args = _parse_arguments(exec_path)

    # setup logger
    log_path = exec_path
    log_file = log_path / 'log.txt'
    log_level = logging.INFO if args.debug_level == 0 else logging.DEBUG
    logger = _get_logger(exec_name, log_file, log_level)
    logger.info(f'Call: {" ".join(argv)}')

    # load registry file
    rules_filename = Path(args.rules)
    rules_registry = load_registry(rules_filename)

    # process file(s)
    if args.states.is_file():
        files = [ args.states ]
    elif args.recursive:
        files = sorted(list(args.states.rglob('*.states')))
    else:
        files = sorted(list(args.states.glob('*.states')))

    num_states = 0
    num_files = len(files)
    num_processed_files = 0

    logger.info(f'Got {num_files} file(s) in {args.states}')
    for i, states_filename in enumerate(files):
        output_filename = Path(f'{states_filename}.augmented')
        domain = args.domain if args.domain != '' else states_filename.parent.name
        if not output_filename.exists() or args.force:
            registry_record = get_registry_record(domain, rules_registry, rules_filename)
            if registry_record is not None:
                logger.info(f"Processing {str(states_filename)} with rules for '{domain}' ({1+i}/{num_files})")
                num_states += _process_single_file(states_filename, output_filename, args.domain, registry_record)
                num_processed_files += 1
            else:
                logger.info(f"Nothing to do since got 'null' record for key '{domain}' in registry")

    # final stats
    elapsed_time = timer() - entry_time
    logger.info(f'{num_states} state(s) processed in {num_processed_files} file(s)')
    logger.info(f'All tasks completed in {elapsed_time:.3f} second(s)')

