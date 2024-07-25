import argparse
from termcolor import colored
import torch
import random
import pandas as pd
from pathlib import Path
from timeit import default_timer as timer
#from generators import load_pddl_problem_with_augmented_states, compute_traces_with_augmented_states
#from architecture import OldMaxModel
#from architecture.max_base_old import OldMaxModelBase as OldMaxModel

import torch
import torch.nn as nn
import pytorch_lightning as pl

# Imports related to type annotations
from typing import List, Dict, Tuple
from torch.nn.functional import Tensor

from generators import load_pddl_problem_with_augmented_states


class RelationMessagePassing(nn.Module):
    def __init__(self, relations: List[Tuple[int, int]], hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.relation_modules = nn.ModuleList()
        for relation, arity in relations:
            assert relation == len(self.relation_modules)
            input_size = arity * hidden_size
            output_size = arity * hidden_size
            if (input_size > 0) and (output_size > 0):
                mlp = nn.Sequential(nn.Linear(input_size, input_size, True), nn.ReLU(), nn.Linear(input_size, output_size, True))
            else:
                mlp = None
            self.relation_modules.append(mlp)
        self.update = nn.Sequential(nn.Linear(2 * hidden_size, 2 * hidden_size, True), nn.ReLU(), nn.Linear(2 * hidden_size, hidden_size, True))
        self.dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self.dummy.device

    def forward(self, node_states: Tensor, relations: Dict[int, Tensor]) -> Tensor:
        #print(f"node_states: \n {node_states}")
        #print(f"relations: \n {relations}")
        # Compute an aggregated message for each recipient
        max_outputs = []
        outputs = []
        for relation, module in enumerate(self.relation_modules):
            #print(f"module {module}")
            #print(f"relation {relation}")
            #print(f"relations {relations}")
            if (module is not None) and (relation in relations):
                values = relations[relation]
                input = torch.index_select(node_states, 0, values).view(-1, module[0].in_features)
                output = module(input).view(-1, self.hidden_size)
                max_outputs.append(torch.max(output))
                node_indices = values.view(-1, 1).repeat(1, self.hidden_size)
                outputs.append((output, node_indices))

        max_offset = torch.max(torch.stack(max_outputs))
        exps_sum = torch.full_like(node_states, 1E-16, device=self.get_device())
        for output, node_indices in outputs:
            exps = torch.exp(8.0 * (output - max_offset))
            exps_sum = torch.scatter_add(exps_sum, 0, node_indices, exps)

        # Update states with aggregated messages
        max_msg = ((1.0 / 8.0) * torch.log(exps_sum)) + max_offset
        next_node_states = self.update(torch.cat([max_msg, node_states], dim=1))
        return next_node_states


class Readout(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(input_size, input_size, bias), nn.ReLU(), nn.Linear(input_size, input_size, bias))
        self.post = nn.Sequential(nn.Linear(input_size, input_size, bias), nn.ReLU(), nn.Linear(input_size, output_size, bias))

    def forward(self, batch_num_objects: List[int], node_states: Tensor) -> Tensor:
        results: List[Tensor] = []
        offset: int = 0
        nodes: Tensor = self.pre(node_states)
        for num_objects in batch_num_objects:
            results.append(self.post(torch.sum(nodes[offset:(offset + num_objects)], dim=0)))
            offset += num_objects
        return torch.stack(results)

    def feature_vectors(self, batch_num_objects: List[int], node_states: Tensor) -> Tensor:
        results: List[Tensor] = []
        offset: int = 0
        nodes: Tensor = self.pre(node_states)
        for num_objects in batch_num_objects:
            intermediate = []
            intermediate.append(torch.sum(nodes[offset:(offset + num_objects)], dim=0))
            for layer in self.post:
                intermediate.append(layer(intermediate[-1]))
            results.append(torch.cat(intermediate))
            offset += num_objects
        return torch.stack(results)


class RelationMessagePassingModel(nn.Module):
    def __init__(self, relations: list, hidden_size: int, iterations: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.iterations = iterations
        self.relation_network = RelationMessagePassing(relations, hidden_size)
        self.dummy = nn.Parameter(torch.empty(0))

    def get_device(self):
        return self.dummy.device

    def forward(self, states: Tuple[Dict[int, Tensor], List[int]]) -> Tensor:
        node_states = self._initialize_nodes(sum(states[1]))
        node_states = self._pass_messages(node_states, states[0])
        return node_states

    def _pass_messages(self, node_states: Tensor, relations: Dict[int, Tensor]) -> Tensor:
        for _ in range(self.iterations):
             node_states = self.relation_network(node_states, relations)
        return node_states

    def _initialize_nodes(self, num_objects: int) -> Tensor:
        init_zeroes = torch.zeros((num_objects, (self.hidden_size // 2) + (self.hidden_size % 2)), dtype=torch.float, device=self.get_device())
        init_random = torch.randn((num_objects, self.hidden_size // 2), device=self.get_device())
        init_nodes = torch.cat([init_zeroes, init_random], dim=1)
        return init_nodes


class MaxModelBase(pl.LightningModule):
    def __init__(self, predicates: list, hidden_size: int, iterations: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = RelationMessagePassingModel(predicates, hidden_size, iterations)
        self.readout = Readout(hidden_size, 1)

    def forward(self, states: Tuple[Dict[int, Tensor], List[int]]) -> Tensor:
        node_states = self.model(states)
        return self.readout(states[1], node_states)

    def feature_vectors(self, states: Tuple[Dict[int, Tensor], List[int]]) -> Tensor:
        node_states = self.model(states)
        return self.readout.feature_vectors(states[1], node_states)








def planning(args, model, domain_file, problem_file):
    result_string = ""

    result_string = result_string + f"Loading PDDL files: domain='{domain_file}', problem='{problem_file}'"
    result_string = result_string + "\n"

    registry_filename = args.registry_filename if args.augment else None
    pddl_problem = load_pddl_problem_with_augmented_states(domain_file, problem_file, registry_filename,
                                                           args.registry_key, None)
    #print(f"predicates: {pddl_problem['predicates']}")
    #print(f"predicate: {pddl_problem['predicates'][0], type(pddl_problem['predicates'][0])}")
    #print(f"predicate: {str(pddl_problem['predicates'][0].name)}")
    predicates = [str(predicate.name) for predicate in pddl_problem['predicates']]
    pred_ids = dict(zip(predicates, range(0, len(predicates))))
    del pddl_problem['predicates']  # Why?

    result_string = result_string + f'Executing policy (max_length={args.max_length})'
    result_string = result_string + "\n"
    start_time = timer()
    is_spanner = args.spanner and 'spanner' in str(domain_file)
    unsolvable_weight = 0.0 if args.ignore_unsolvable else 100000.0
    action_trace, state_trace, value_trace, is_solution, num_evaluations = compute_traces_with_augmented_states(
        pred_ids, model=model, cycles=args.cycles, max_trace_length=args.max_length, unsolvable_weight=unsolvable_weight,
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

    return result_string, action_trace, is_solution

from generators.plan import create_object_encoding
def compute_traces_with_augmented_states(pred_ids, actions, initial, goal, language, model: pl.LightningModule, augment_fn = None, cycles: str = 'avoid', max_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None, is_spanner = False):
    objects = language.constants()
    obj_encoding = create_object_encoding(objects)
    if logger: logger.info(f'{len(objects)} object(s), obj_encoding={obj_encoding}')

    with torch.no_grad():
        return policy_search_with_augmented_states(pred_ids, actions, initial, goal, obj_encoding, language, model, augment_fn=augment_fn, cycles=cycles, max_state_trace_length=max_trace_length, unsolvable_weight=unsolvable_weight, logger=logger, is_spanner=is_spanner)

from generators.plan import _get_goal_denotation, _to_input, _get_successor_states, _get_applicable_actions, _spanner_unsolvable, _spanner_solved
def policy_search_with_augmented_states(pred_ids, actions, initial, goals, obj_encoding: Dict[str, int], language, model: pl.LightningModule, augment_fn = None, cycles: str = 'avoid', max_state_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None, is_spanner = False):
    device = model.device
    closed_states = set()
    action_trace = []

    # calculate denotation of goal atoms that is equal for every state
    if logger: logger.info(f'goals={goals}')
    goal_denotation = _get_goal_denotation(goals, obj_encoding)

    # add goal denotation to pred_ids
    goal_names = goal_denotation.keys()
    goal_ids = dict(zip(goal_names, range(len(pred_ids), len(pred_ids) + len(goal_names))))
    pred_ids.update(goal_ids)

    # set initial state and value trace
    current_state = initial
    collated_input, encoded_states = _to_input([ current_state ], goal_denotation, obj_encoding, augment_fn, language, device, logger)
    state_trace = [ encoded_states[0] ]
    #print(f"Collated input:\n {collated_input}")
    #initial_values = model(collated_input)
    if logger: logger.debug(f'initial_state={current_state}')

    # calculate greedy trace
    step, num_evaluations = 1, 1
    while (not current_state[goals]) and (len(state_trace) < max_state_trace_length):
        if cycles == 'detect' and current_state in closed_states:
            if logger:
                logger.info(colored(f"Cycle detected after last action '{action_trace[-1]}'", 'magenta'))
            break
        closed_states.add(current_state)
        if logger: logger.debug(f'**** STEP {step+1}')
        step += 1

        # SPANNER: special case to avoid very time-consuming execution
        if is_spanner and _spanner_unsolvable(current_state, logger):
            if logger: logger.info(colored(f'SPANNER TASK FAILURE', 'red', attrs=[ 'bold' ]))
            break
        elif is_spanner and _spanner_solved(current_state, logger):
            if logger: logger.info(colored(f'SPANNER TASK SOLVED', 'green', attrs=[ 'bold' ]))
            if logger: logger.info(f'current_state={current_state}')
            break

        # explore current state (avoid loops by removing already visited successors)
        successor_candidates = [ transition for transition in _get_successor_states(current_state, actions) ]
        if cycles == 'avoid':
            successor_candidates = [ transition for transition in successor_candidates if transition[1] not in closed_states ]

        if len(successor_candidates) == 0:
            if logger: logger.info(f'No applicable action that yields unvisited state for current_state={current_state}')
            if logger: logger.info(f'Applicable actions = {_get_applicable_actions(current_state, actions)}')
            print(f'No applicable action that yields unvisited state for current_state')
            print(f'Applicable actions = {_get_applicable_actions(current_state, actions)}')
            break

        successor_actions = [ candidate[0] for candidate in successor_candidates ]
        successor_states = [ candidate[1] for candidate in successor_candidates ]
        if logger: logger.debug(f'#actions={len(successor_actions)}, actions={successor_actions}')

        # calculate values for successors and best successor
        collated_input, encoded_states = _to_input(successor_states, goal_denotation, obj_encoding, augment_fn, language, device, logger)
        print(f"Collated input: \n {collated_input[0].keys()}")
        encoded_states = (dict([(pred_ids[name], values) for name, values in collated_input[0].items()]), collated_input[1])
        output_values = model(encoded_states)
        best_successor_index = torch.argmin(output_values)
        num_evaluations += len(successor_actions)
        if logger:
            logger.debug(f'     values=[' + ", ".join([ f'{x[0]:.3f}' for x in output_values ]) + ']')
            logger.debug(f'best_action={successor_actions[best_successor_index]} (index={best_successor_index})\n')

        # extend traces and set next current state
        state_trace.append(encoded_states[best_successor_index])
        action_trace.append(successor_actions[best_successor_index])
        current_state = successor_states[best_successor_index]

        if logger:
            logger.debug(f'current_state={current_state}')
            logger.debug('')

    reached_goal = current_state[goals]
    if logger: logger.debug(f'status={1 if reached_goal else 0}')
    return action_trace, state_trace, [], reached_goal, num_evaluations

def _parse_arguments():
    parser = argparse.ArgumentParser()

    # default values for arguments
    default_gpus = 0  # No GPU

    # required arguments
    parser.add_argument('--policy', required=True, type=Path, help='path to policy (.ckpt)')
    parser.add_argument('--logdir', required=True, type=Path, help='directory where policies are saved')
    parser.add_argument('--min_size', required=True, type=int, help='minimum size of the generated instances')
    parser.add_argument('--max_size', required=True, type=int, help='maximum size of the generated instances')

    # arguments with meaningful default values
    parser.add_argument('--gpus', default=default_gpus, type=int, help=f'number of GPUs to use (default={default_gpus})')

    default_debug_level = 0
    default_cycles = 'avoid'
    default_logfile = 'log_plan.txt'
    default_max_length = 500  # TODO: CHOOSE DIFFERENT MAX LENGTH OR USE A TIME LIMIT?
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

# activate the dropout
def enable_dropout(mod: torch.nn.Module):
    if isinstance(mod, torch.nn.Dropout):
        mod.train()

def generate_instance_gripper_atomic(num_balls):
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
    # randomly sample one of the balls from room a and place it in room b
    ball_to_move = random.choice(range(num_balls))
    #ball_to_move = list(range(num_balls))[-1]
    instance += f"\n(at ball{ball_to_move+1} roomb)"
    instance += f"\n)"
    instance += f"\n)"
    instance += f"\n)"
    return instance

def _main(args):
    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")

    # load model
    try:
        model = MaxModelBase.load_from_checkpoint(checkpoint_path=str(args.policy), strict=False).to(device)
    except:
        try:
            model = MaxModelBase.load_from_checkpoint(checkpoint_path=str(args.policy), strict=False,
                                                      map_location=torch.device('cuda'))
        except:
            model = MaxModelBase.load_from_checkpoint(checkpoint_path=str(args.policy), strict=False,
                                                      map_location=torch.device('cpu'))
    model.eval()
    print(len(model.model.relation_network.relation_modules))

    domain_file = Path('data_old/pddl/' + args.domain + '/test/domain.pddl')

    # TODO: TRACK AVERAGE PLAN LENGTH?
    results = {
        "size": [],
        "avg_coverage": [],
        "avg_plan_length": []
    }
    for size in range(args.min_size, args.max_size + 1):
        results["size"].append(size)
        coverages = []
        plan_lengths = []
        # create directory for storing generated instance files
        instance_directory = args.logdir / f"generated_instances/size_{size}"
        instance_directory.mkdir(parents=True, exist_ok=True)

        num_instances_per_size = 1
        num_runs_per_instance = 1
        for i in range(num_instances_per_size):
            instance_string = generate_instance_gripper_atomic(size)
            instance_file = instance_directory / f"instance_size_{size}_num_{i}.pddl"
            with open(instance_file, 'w') as f:
                f.write(instance_string)

            for _ in range(num_runs_per_instance):
                result_string, action_trace, is_solution = planning(args, model, domain_file, instance_file)
                if is_solution:
                    coverages.append(1)
                    plan_lengths.append(len(action_trace))
                    print(colored(f"Instance size {size} num {i} solved after {len(action_trace)} steps!", 'green', attrs=['bold']))
                else:
                    coverages.append(0)
                    print(colored(f"Instance size {size} num {i} not solved!", 'red', attrs=['bold']))

        avg_coverage = sum(coverages) / len(coverages)
        if len(plan_lengths) == 0:
            avg_plan_length = 0
        else:
            avg_plan_length = sum(plan_lengths) / len(plan_lengths)
        results[f"avg_coverage"].append(avg_coverage)
        results[f"avg_plan_length"].append(avg_plan_length)

        results_data_frame = pd.DataFrame(results)
        results_data_frame.to_csv(args.logdir / "results.csv")


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)
