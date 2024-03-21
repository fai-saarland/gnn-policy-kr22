import argparse
from termcolor import colored
import os
import re
import glob
import pandas as pd
import torch
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from torch_geometric.loader import DataLoader as GraphDataLoader
from training_new import load_model, load_trainer, planning
from torch_geometric.data import Data, Batch
from timeit import default_timer as timer
from generators import load_pddl_problem_with_augmented_states

def _parse_arguments():
    parser = argparse.ArgumentParser()

    # default values for arguments
    default_batch_size = 64  # 64
    default_gpus = 0  # No GPU
    default_num_workers = 0
    default_learning_rate = 0.001
    default_weight_decay = 0.0
    default_gradient_accumulation = 1
    default_max_samples_per_file = 1000  # TODO: INCREASE THIS?
    default_max_samples = None
    default_patience = 50
    default_gradient_clip = 0.1
    default_profiler = None
    default_validation_frequency = 1
    default_save_top_k = 5
    default_max_epochs = None
    default_train_indices = None
    default_val_indices = None

    # TODO: COMPUTE PATHS AUTOMATICALLY FROM DOMAIN NAME?
    # arguments for training
    parser.add_argument('--train', required=True, type=Path, help='path to training dataset')
    parser.add_argument('--validation', required=True, type=Path, help='path to validation dataset')
    parser.add_argument('--rounds', required=True, type=int, help='how often training is repeated with a newly sampled training set')
    parser.add_argument('--seeds', required=True, type=int, help='number of random seeds used for training')
    parser.add_argument('--logdir', required=True, type=Path, help='directory where policies are saved')

    # arguments for the architecture
    parser.add_argument('--aggregation', required=True, choices=['GCN', 'GCNV2', 'GAT', 'GATV2', 'GIN', 'Transformer', 'GCNGPS'], help=f'aggregation function')
    parser.add_argument('--readout', required=True, choices=['ADD', 'MAX'], help=f'readout function')
    parser.add_argument('--loss', required=True, choices=['MSE', 'MAE'], help=f'loss function')

    parser.add_argument('--num_layers', required=True, type=int, help='number of GNN layers')
    parser.add_argument('--hidden_size', required=True, type=int, help='hidden size of GNN layers')
    parser.add_argument('--dropout', required=True, type=float, help='percentage of randomly deactivated neurons in each layer')
    parser.add_argument('--heads', default=1, type=int, help='number of attention heads')

    # when using an existing trained policy
    #parser.add_argument('--policy', default=None, type=Path, help='path to policy (.ckpt) for re-training')

    # specifying which states should be selected for training and validation sets
    parser.add_argument('--train_indices', default=default_train_indices, type=str, help=f'indices of states to use for training (default={default_train_indices})')
    parser.add_argument('--val_indices', default=default_val_indices, type=str, help=f'indices of states to use for validation (default={default_val_indices})')

    # arguments with meaningful default values
    parser.add_argument('--runs', type=int, default=1, help='number of planning runs per instance')
    parser.add_argument('--max_epochs', default=default_max_epochs, type=int, help=f'maximum number of epochs (default={default_max_epochs})')
    # parser.add_argument('--max_bugs_per_iteration', default=default_max_bugs_per_iteration, type=int, help=f'maximum number of bugs per iteration (default={default_max_bugs_per_iteration})')
    parser.add_argument('--batch_size', default=default_batch_size, type=int, help=f'maximum size of batches (default={default_batch_size})')
    parser.add_argument('--gpus', default=default_gpus, type=int, help=f'number of GPUs to use (default={default_gpus})')
    parser.add_argument('--num_workers', default=default_num_workers, type=int, help=f'number of workers for the data loader (use 0 on Windows) (default={default_num_workers})')
    parser.add_argument('--learning_rate', default=default_learning_rate, type=float, help=f'learning rate of training session (default={default_learning_rate})')
    parser.add_argument('--weight_decay', default=default_weight_decay, type=float, help=f'strength of weight decay regularization (default={default_weight_decay})')
    parser.add_argument('--gradient_accumulation', default=default_gradient_accumulation, type=int, help=f'number of gradients to accumulate before step (default={default_gradient_accumulation})')
    parser.add_argument('--max_samples_per_file', default=default_max_samples_per_file, type=int, help=f'maximum number of states per dataset (default={default_max_samples_per_file})')
    parser.add_argument('--max_samples', default=default_max_samples, type=int, help=f'maximum number of states in total (default={default_max_samples})')
    parser.add_argument('--patience', default=default_patience, type=int, help=f'patience for early stopping (default={default_patience})')
    parser.add_argument('--gradient_clip', default=default_gradient_clip, type=float, help=f'gradient clip value (default={default_gradient_clip})')
    parser.add_argument('--profiler', default=default_profiler, type=str, help=f'"simple", "advanced" or "pytorch" (default={default_profiler})')
    parser.add_argument('--validation_frequency', default=default_validation_frequency, type=int, help=f'evaluate on validation set after this many epochs (default={default_validation_frequency})')
    parser.add_argument('--verbose', action='store_true', help='print additional information during training')
    parser.add_argument('--verify_datasets', action='store_true', help='verify state labels are as expected')

    # logging and saving models
    parser.add_argument('--logname', default=None, type=str, help='if provided, versions are stored in folder with this name inside logdir')
    parser.add_argument('--save_top_k', default=default_save_top_k, type=int, help=f'number of top-k models to save (default={default_save_top_k})')

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

# writes results of a planning run ato a csv file
def save_results(results, policy_type, policy_path, val_loss, bug_loss, planning_results):
    results["type"].append(policy_type)
    results["policy_path"].append(policy_path)
    results["val_loss"].append(val_loss)
    results["bug_loss"].append(bug_loss)
    results["instances"].append(planning_results["instances"])
    results["max_coverage"].append(planning_results["max_coverage"])
    results["min_coverage"].append(planning_results["min_coverage"])
    results["avg_coverage"].append(planning_results["avg_coverage"])
    results["best_plan_quality"].append(planning_results["best_plan_quality"])
    results["plans_directory"].append(planning_results["plans_directory"])
    results.update(vars(args))

def _split(tokens: list) -> list:
    return [token.split(' ') for token in tokens]


def _read_between(index: int, start_line: str, end_line: str, file: list) -> list:
    index += 1
    lines = []
    if file[index] != start_line: raise Exception(start_line)
    while True:
        index += 1
        if file[index] == end_line: break
        else: lines.append(file[index])
    return index, _split(lines)


def _read_state(index: int, file: list) -> list:
    index += 1
    lines = []
    line = file[index]
    if line != "BEGIN_STATE": raise Exception("BEGIN_STATE")
    while True:
        index += 1
        line = file[index]
        if line == "END_STATE": break
        else: lines.append(line)
    return index, _split(lines)


def _read_labeled_state(index: int, file: list) -> list:
    index += 1
    lines = []
    line = file[index]
    if line != "BEGIN_LABELED_STATE":
        raise Exception("BEGIN_LABELED_STATE")
    index += 1
    lines.append(file[index])
    index, state = _read_state(index, file)
    lines.append(state)
    index += 1
    line = file[index]
    if line != "END_LABELED_STATE":
        Exception("END_LABELED_STATE")
    return (index, lines)


def _read_labeled_states(index: int, file: list) -> list:
    index += 1
    transitions = []
    line = file[index]
    if line != "BEGIN_STATE_LIST":
        raise Exception("BEGIN_STATE_LIST")
    while file[index + 1] == "BEGIN_LABELED_STATE":
        index, transition = _read_labeled_state(index, file)
        transitions.append(transition)
    index += 1
    line = file[index]
    if line != "END_STATE_LIST":
        raise Exception("END_STATE_LIST")
    return index, transitions


def _decode_predicate(objs_map: dict, preds_map: dict, encoded_predicate: list) -> tuple:
    predicate = preds_map[encoded_predicate[0]]
    arguments = tuple([objs_map[index] for index in encoded_predicate[1:]])
    return (predicate, arguments)


def _decode_predicates(objs_map: dict, preds_map: dict, encoded_predicates: list) -> list:
    return [_decode_predicate(objs_map, preds_map, encoded_predicate) for encoded_predicate in encoded_predicates]


def _intify_predicate(encoded_predicate: list) -> tuple:
    predicate = int(encoded_predicate[0])
    arguments = [int(index) for index in encoded_predicate[1:]]
    return (predicate, arguments)


def _intify_predicates(encoded_predicates: list) -> list:
    return [_intify_predicate(encoded_predicate) for encoded_predicate in encoded_predicates]


def _load_file(file: Path, decode: bool):
    with file.open('r') as fs: lines = [line.strip() for line in fs.readlines()]
    index = -1
    index, objs_map = _read_between(index, "BEGIN_OBJECTS", "END_OBJECTS", lines)
    index, preds_map = _read_between(index, "BEGIN_PREDICATES", "END_PREDICATES", lines)
    index, facts_encoded = _read_between(index, "BEGIN_FACT_LIST", "END_FACT_LIST", lines)
    index, goals_encoded = _read_between(index, "BEGIN_GOAL_LIST", "END_GOAL_LIST", lines)
    index, states_encoded = _read_labeled_states(index, lines)
    objs_map = dict(objs_map)
    preds_map = dict(preds_map)
    if decode:
        objs = list(objs_map.values())
        preds = list(preds_map.values())
        facts = _decode_predicates(objs_map, preds_map, facts_encoded)
        goals = _decode_predicates(objs_map, preds_map, goals_encoded)
        states = [(c, _decode_predicates(objs_map, preds_map, state)) for c, state in states_encoded]
    else:
        objs = [int(o) for o in objs_map.keys()]
        preds = [int(p) for p in preds_map.keys()]
        facts = _intify_predicates(facts_encoded)
        goals = _intify_predicates(goals_encoded)
        states = [(c, _intify_predicates(state)) for c, state in states_encoded]
    return {
        'objs': objs,
        'preds': preds,
        'facts': facts,
        'goals': goals,
        'states': states
    }


def _arity_of(predicate, facts, goals, states):
    def find_arity(preds):
        for (other_predicate, arguments) in preds:
            if predicate == other_predicate:
                return len(arguments)
    arity = find_arity(facts)
    if arity != None: return arity
    arity = find_arity(goals)
    if arity != None: return arity
    for (_, state) in states:
        arity = find_arity(state)
        if arity != None: return arity
    return 0


def _pack_by_predicate(predicates, to_tensor: bool):
    packed = {}
    for predicate, arguments in predicates:
        if predicate not in packed: packed[predicate] = []
        packed[predicate].append(arguments)
    if to_tensor:
        for predicate in packed.keys():
            packed[predicate] = torch.tensor(packed[predicate])
    return packed


class ValueDataset(Dataset):
    """State value dataset."""

    def __init__(self, file: Path, min_cost: float = None, max_cost: float = None, decode: bool = False):
        """
        directory (Path): Path to directory of *.txt files with state transitions.
        """
        self._decoded = decode
        data = _load_file(file, decode)

        initial_preds = [(predicate, _arity_of(predicate, data['facts'], data['goals'], data['states'])) for predicate in data['preds']]
        goal_predicate_offset = '_goal' if decode else len(initial_preds)
        goal_preds = [(predicate + goal_predicate_offset, arity) for predicate, arity in initial_preds]
        preds = initial_preds + goal_preds

        self.file = file
        self.objects = data['objs']
        self.facts = data['facts']
        self.goals = [(predicate + goal_predicate_offset, arguments) for predicate, arguments in data['goals']]
        if min_cost is not None and max_cost is not None:
            self.states = [state for state in data['states'] if (float(state[0]) >= min_cost) and (float(state[0]) <= max_cost)]
        elif min_cost is not None:
            self.states = [state for state in data['states'] if float(state[0]) >= min_cost]
        elif max_cost is not None:
            self.states = [state for state in data['states'] if float(state[0]) <= max_cost]
        else:
            self.states = data['states']
        self.predicates = preds
        #self.predicates.sort()

        decoded_data = _load_file(file, True)
        initial_preds = [(predicate, _arity_of(predicate, decoded_data['facts'], decoded_data['goals'], decoded_data['states'])) for predicate
                         in decoded_data['preds']]
        goal_predicate_offset = '_goal'
        goal_preds = [(predicate + goal_predicate_offset, arity) for predicate, arity in initial_preds]
        decoded_predicates = initial_preds + goal_preds
        self.decoded_predicates = decoded_predicates
        #self.decoded_predicates.sort()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        (cost, state) = self.states[idx]
        if self._decoded:
            input = _pack_by_predicate(self.facts + self.goals + state, False)
            target = float(cost)
        else:
            input = _pack_by_predicate(self.facts + self.goals + state, True)
            target = torch.tensor([float(cost)])
        return (input, target)


class LimitedDataset(Dataset):
    def __init__(self, dataset, max_samples_per_value) -> None:
        super().__init__()
        samples_by_value = {}
        for input, target in dataset:
            key = int(target)
            if key not in samples_by_value:
                samples_by_value[key] = []
            value_samples = samples_by_value[key]
            if len(value_samples) < max_samples_per_value:
                value_samples.append((input, target))
        self.samples = [sample for samples in samples_by_value.values() for sample in samples]
        self.predicates = dataset.predicates
        self.decoded_predicates = dataset.decoded_predicates

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ExtendedDataset(Dataset):
    def __init__(self, datasets, repeat = 1):
        self._datasets = datasets
        self._repeat = repeat

    def __getitem__(self, index):
        for dataset in self._datasets:
            if index < len(dataset) * self._repeat:
                return dataset[index % len(dataset)]
            else:
                index -= len(dataset) * self._repeat
        raise IndexError()

    def __len__(self):
        return sum(len(d) for d in self._datasets) * self._repeat


def load_dataset(path: Path, max_samples_per_value: int):
    datasets = [LimitedDataset(ValueDataset(d, max_cost=None, decode=False), max_samples_per_value) for d in path.glob('*states.txt')]
    predicates = datasets[0].predicates
    decoded_predicates = datasets[0].decoded_predicates
    return (ExtendedDataset(datasets, 1), predicates, decoded_predicates)

def states_to_graphs2(states, predicate_dict, predicate_ids, max_arity):
    # first we decode the given states such that we have easy access to the label, the relations and the objects
    decoded_states = []
    for (state, label) in states:
        atoms = []
        max_id = 0

        for pred, arg in state.items():
            for a in arg:
                arguments = [x.item() for x in list(a)]
                atoms.append((pred, arguments))

                for argument in arguments:
                    if argument > max_id:
                        max_id = argument

        decoded_states.append((label, atoms, list(range(max_id + 1))))

    graph_states = []
    for (label, atoms, objects) in decoded_states:
        nodes_x = []
        edge_index = [[], []]
        # create nodes for objects
        for i in range(len(objects)):
            obj = objects[i]
            # create tensor for object node's feature
            object_node = torch.ones(3 + max_arity) * -1
            # first feature is the id of the node
            object_node[0] = i
            # second feature indicates that this is an object node
            object_node[1] = 0
            # third feature is the id of the object
            object_node[2] = obj

            nodes_x.append(object_node)

        # create nodes for atoms and add edges between objects and atoms
        for i in range(len(atoms)):
            predicate, arguments = atoms[i]
            # create tensor for relation node's feature
            atom_node = torch.ones(3 + max_arity) * -1
            # first feature is the id of the node
            atom_node[0] = i + len(objects)
            # second feature indicates that this is an atom node
            atom_node[1] = 1
            # third feature is the id of the predicate
            atom_node[2] = predicate_ids[predicate]
            # next features are the object ids of the arguments
            for x, argument in enumerate(arguments):
                atom_node[x + 3] = argument

            nodes_x.append(atom_node)

            # if the atom takes no arguments we connect the atom node to all object nodes
            if len(arguments) == 0:
                for x in range(len(objects)):
                    edge_index[0].append(i + len(objects))
                    edge_index[1].append(x)
                    edge_index[0].append(x)
                    edge_index[1].append(i + len(objects))
            else:
                # connect atom node to corresponding object nodes
                for x, argument in enumerate(arguments):
                    edge_index[0].append(i + len(objects))
                    edge_index[1].append(argument)
                    edge_index[0].append(argument)
                    edge_index[1].append(i + len(objects))

        nodes_x = torch.stack(nodes_x).float()
        edge_index = torch.tensor(edge_index).long()
        label = label.float()
        graph_state = Data(x=nodes_x, edge_index=edge_index, y=label, num_nodes=len(objects) + len(atoms))
        graph_state.validate(raise_on_error=True)
        graph_states.append(graph_state)

    return graph_states

def state_to_graph2(state, predicate_dict, predicate_ids, max_arity):
    atoms = []
    max_id = 0
    # the states only have one entry for each predicate, so to get the individual atoms we need to split according
    # to the arity of the predicate
    for pred, arg in state.items():
        for a in arg:
            arguments = [x.item() for x in list(a)]
            atoms.append((pred, arguments))

            for argument in arguments:
                if argument > max_id:
                    max_id = argument

    objects = list(range(max_id + 1))


    nodes_x = []
    edge_index = [[], []]
    # create nodes for objects
    for i in range(len(objects)):
        obj = objects[i]
        # create tensor for object node's feature
        object_node = torch.ones(3 + max_arity) * -1
        # first feature is the id of the node
        object_node[0] = i
        # second feature indicates that this is an object node
        object_node[1] = 0
        # third feature is the id of the object
        object_node[2] = obj

        nodes_x.append(object_node)

    # create nodes for atoms and add edges between objects and atoms
    for i in range(len(atoms)):
        predicate, arguments = atoms[i]
        # create tensor for relation node's feature
        atom_node = torch.ones(3 + max_arity) * -1
        # first feature is the id of the node
        atom_node[0] = i + len(objects)
        # second feature indicates that this is an atom node
        atom_node[1] = 1
        # third feature is the id of the predicate
        atom_node[2] = predicate_ids[predicate]
        # next features are the object ids of the arguments
        for x, argument in enumerate(arguments):
            atom_node[x + 3] = argument

        nodes_x.append(atom_node)

        # if the atom takes no arguments we connect the atom node to all object nodes
        if len(arguments) == 0:
            for x in range(len(objects)):
                edge_index[0].append(i + len(objects))
                edge_index[1].append(x)
                edge_index[0].append(x)
                edge_index[1].append(i + len(objects))
        else:
            # connect atom node to corresponding object nodes
            for x, argument in enumerate(arguments):
                edge_index[0].append(i + len(objects))
                edge_index[1].append(argument)
                edge_index[0].append(argument)
                edge_index[1].append(i + len(objects))

    nodes_x = torch.stack(nodes_x).float()
    edge_index = torch.tensor(edge_index).long()
    graph_state = Data(x=nodes_x, edge_index=edge_index, num_nodes=len(objects) + len(atoms))
    graph_state.validate(raise_on_error=True)

    return graph_state

def planning2(predicate_dict, predicate_ids, max_arity, args, policy, model, domain_file, problem_file, device):
    start_time = timer()
    result_string = ""
    # deactivate dropout!
    model.eval()
    model.training = False

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
    unsolvable_weight = 0.0 if args.ignore_unsolvable else 100000.0
    action_trace, state_trace, value_trace, is_solution, num_evaluations = compute_traces_with_augmented_states(
        predicate_dict=predicate_dict, predicate_ids=predicate_ids, max_arity=max_arity,
        model=model, cycles=args.cycles, max_trace_length=args.max_length,
        unsolvable_weight=unsolvable_weight, logger=None, **pddl_problem)
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

from generators.plan import create_object_encoding
from generators.plan import _get_goal_denotation, _to_input, _get_successor_states, _get_applicable_actions, _spanner_unsolvable, _spanner_solved
def compute_traces_with_augmented_states2(predicate_dict, predicate_ids, max_arity, actions, initial, goal, language, model: pl.LightningModule, augment_fn = None, cycles: str = 'avoid', max_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None):
    objects = language.constants()
    obj_encoding = create_object_encoding(objects)
    if logger: logger.info(f'{len(objects)} object(s), obj_encoding={obj_encoding}')

    with torch.no_grad():
        device = model.device
        closed_states = set()
        action_trace = []

        # calculate denotation of goal atoms that is equal for every state
        if logger: logger.info(f'goals={goal}')
        goal_denotation = _get_goal_denotation(goal, obj_encoding)

        # set initial state and value trace
        current_state = initial
        collated_input, encoded_states = _to_input([current_state], goal_denotation, obj_encoding, augment_fn, language,
                                                   device, logger)
        state_trace = [encoded_states[0]]
        state_graph = state_to_graph2(encoded_states[0], predicate_dict, predicate_ids, max_arity)
        initial_values = model(state_graph)
        value_trace = [initial_values[0]]
        if logger: logger.debug(f'initial_state={current_state}')

        # calculate greedy trace
        step, num_evaluations = 1, 1
        while (not current_state[goal]) and (len(state_trace) < max_trace_length):
            if cycles == 'detect' and current_state in closed_states:
                if logger:
                    logger.info(colored(f"Cycle detected after last action '{action_trace[-1]}'", 'magenta'))
                break
            closed_states.add(current_state)
            if logger: logger.debug(f'**** STEP {step + 1}')
            step += 1

            # explore current state (avoid loops by removing already visited successors)
            successor_candidates = [transition for transition in _get_successor_states(current_state, actions)]
            if cycles == 'avoid':
                successor_candidates = [transition for transition in successor_candidates if
                                        transition[1] not in closed_states]

            if len(successor_candidates) == 0:
                if logger: logger.info(
                    f'No applicable action that yields unvisited state for current_state={current_state}')
                if logger: logger.info(f'Applicable actions = {_get_applicable_actions(current_state, actions)}')
                print(f'No applicable action that yields unvisited state for current_state')
                print(f'Applicable actions = {_get_applicable_actions(current_state, actions)}')
                break

            successor_actions = [candidate[0] for candidate in successor_candidates]
            successor_states = [candidate[1] for candidate in successor_candidates]
            if logger: logger.debug(f'#actions={len(successor_actions)}, actions={successor_actions}')

            # calculate values for successors and best successor
            collated_input, encoded_states = _to_input(successor_states, goal_denotation, obj_encoding, augment_fn,
                                                       language, device, logger)
            state_graphs = [state_to_graph2(encoded_state, predicate_dict, predicate_ids, max_arity) for encoded_state in encoded_states]
            state_graphs_batch = Batch.from_data_list(state_graphs)  # TODO: DEVICE????

            output_values = model(state_graphs_batch)
            best_successor_index = torch.argmin(output_values)
            num_evaluations += len(successor_actions)
            if logger:
                logger.debug(f'     values=[' + ", ".join([f'{x[0]:.3f}' for x in output_values]) + ']')
                logger.debug(f'best_action={successor_actions[best_successor_index]} (index={best_successor_index})\n')

            # extend traces and set next current state
            value_trace.append(output_values[best_successor_index])
            state_trace.append(encoded_states[best_successor_index])
            action_trace.append(successor_actions[best_successor_index])
            current_state = successor_states[best_successor_index]

            if logger:
                logger.debug(f'current_state={current_state}')
                logger.debug('')

        reached_goal = current_state[goal]
        if logger: logger.debug(f'status={1 if reached_goal else 0}')
        return action_trace, state_trace, value_trace, reached_goal, num_evaluations


def _main(args):
    # TODO: STEP 1: INITIALIZE
    print(colored('Initializing datasets and loaders', 'red', attrs=['bold']))
    args.logdir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available(): args.gpus = 0
    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")

    train_logdir = args.logdir / "trained"
    train_logdir.mkdir(parents=True, exist_ok=True)

    for round in range(args.rounds):
        round_dir = train_logdir / f"round_{round}"
        round_dir.mkdir(parents=True, exist_ok=True)

        train_dataset, predicates, decoded_predicates = load_dataset(args.train, args.max_samples_per_file)
        print(predicates)
        print(decoded_predicates)

        # assert arities are same, otherwise the orders could be different
        for i in range(len(predicates)):
            assert predicates[i][1] == decoded_predicates[i][1]

        validation_dataset, _, _ = load_dataset(args.validation, args.max_samples_per_file)

        #assert True == False


        # store the arities, ids and maximum arity of the predicates for creating the graphs later
        predicate_dict = {}
        predicate_ids = {}
        max_arity = 0
        i = 0
        for predicate, arity in predicates:
            predicate_dict[predicate] = arity
            if arity > max_arity:
                max_arity = arity
            predicate_ids[predicate] = i
            i += 1

        i = 0
        decoded_predicate_dict = {}
        decoded_predicate_ids = {}
        for predicate, arity in decoded_predicates:
            decoded_predicate_dict[predicate] = arity
            decoded_predicate_ids[predicate] = i
            i += 1

        train_samples = [train_dataset[i] for i in range(len(train_dataset))]
        validation_samples = [validation_dataset[i] for i in range(len(validation_dataset))]

        train_graphs = states_to_graphs2(train_samples, predicate_dict, predicate_ids, max_arity)
        validation_graphs = states_to_graphs2(validation_samples, predicate_dict, predicate_ids, max_arity)

        #for graph in validation_graphs:
        #    print(graph)
        #assert True == False

        train_loader = GraphDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=True)
        validation_loader = GraphDataLoader(validation_graphs, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)


        # TODO: STEP 2: TRAIN
        print(colored('Training policies from scratch', 'red', attrs=['bold']))
        for _ in range(args.seeds):
            model = load_model(args, max_arity=max_arity)
            trainer = load_trainer(args, logdir=round_dir)
            model.set_checkpoint_path(f"{round_dir}/version_{trainer.logger.version}/")
            print(colored('Training model...', 'green', attrs = [ 'bold' ]))
            print(type(model).__name__)
            trainer.fit(model, train_loader, validation_loader)

    # TODO: STEP 3: FIND BEST TRAINED MODEL
    print(colored('Determining best trained policy', 'red', attrs=['bold']))
    best_trained_val_loss = float('inf')
    best_trained_policy = None

    for round_dir in train_logdir.glob('round_*'):
        for version_dir in round_dir.glob('version_*'):
            checkpoint_dir = version_dir / 'checkpoints'
            for checkpoint in checkpoint_dir.glob('*.ckpt'):
                try:
                    # validation losses are stored in the name of the stored policy
                    val_loss = float(re.search("validation_loss=(.*?).ckpt", str(checkpoint)).group(1))
                except:
                    val_loss = float('inf')
                    print(f"Checkpoint encoding error: {checkpoint}")
                if val_loss < best_trained_val_loss:
                    best_trained_val_loss = val_loss
                    best_trained_policy = checkpoint

    print(f"The best trained policy achieved a validation loss of {best_trained_val_loss}")
    best_trained_bug_loss = None

    best_trained_policy_dir = train_logdir / 'best'
    best_trained_policy_dir.mkdir(parents=True, exist_ok=True)

    # copy the best policy to the new directory
    best_trained_policy_name = os.path.basename(best_trained_policy)
    best_trained_policy_path = os.path.join(best_trained_policy_dir, best_trained_policy_name)
    os.system("cp " + str(best_trained_policy) + " " + str(best_trained_policy_path))

    # TODO: STEP 3: PLANNING
    print(colored('Running policies on test instances', 'red', attrs=['bold']))
    policies_and_directories = []

    plans_trained_path = args.logdir / "plans_trained"
    plans_trained_path.mkdir(parents=True, exist_ok=True)
    policies_and_directories.append(("trained", best_trained_policy_path, plans_trained_path))

    results = {
        "type": [],
        "policy_path": [],
        "val_loss": [],
        "bug_loss": [],
        "instances": [],
        "max_coverage": [],
        "min_coverage": [],
        "avg_coverage": [],
        "best_plan_quality": [],
        "plans_directory": [],
    }
    for policy_type, policy, directory in policies_and_directories:
        # load files for planning
        domain_file = Path('data_old/pddl/' + args.domain + '/test/domain.pddl')
        problem_files = glob.glob(str('data_old/pddl/' + args.domain + '/test/' + '*.pddl'))

        # load model
        Model = load_model(args, max_arity=max_arity, path=policy)
        try:
            model = Model.load_from_checkpoint(checkpoint_path=str(policy), strict=False).to(device)
        except:
            try:
                model = Model.load_from_checkpoint(checkpoint_path=str(policy), strict=False,
                                                   map_location=torch.device('cuda')).to(device)
            except:
                model = Model.load_from_checkpoint(checkpoint_path=str(policy), strict=False,
                                                   map_location=torch.device('cpu')).to(device)
        # deactivate dropout!
        model.training = False
        model = model.to(device)

        # initialize metrics
        best_coverage = 0
        best_plan_quality = float('inf')
        best_planning_run = None
        coverages = []
        for i in range(args.runs):
            # create directory for current run
            version_path = directory / f"version_{i}"
            version_path.mkdir(parents=True, exist_ok=True)
            # initialize metrics for current run
            plan_lengths = []
            is_solutions = []
            for problem_file in problem_files:
                problem_name = str(Path(problem_file).stem)
                if problem_name == 'domain':
                    continue

                if args.cycles == 'detect':
                    logfile_name = problem_name + ".markovian"
                else:
                    logfile_name = problem_name + ".policy"
                log_file = version_path / logfile_name
                # logger.info(f'Call: {" ".join(argv)}')  # TODO: KEEP THIS?

                # run planning
                result_string, action_trace, is_solution = planning(decoded_predicate_dict, decoded_predicate_ids, max_arity, args, policy, model, domain_file, problem_file, device)

                # store results
                with open(log_file, "w") as f:
                    f.write(result_string)

                is_solutions.append(is_solution)
                if is_solution:
                    plan_lengths.append(len(action_trace))
                    print(f"Solved problem {problem_name} with plan length: {len(action_trace)}")
                else:
                    print(f"Failed to solve problem {problem_name}")

                print(result_string)

            # compute coverage of this run and check whether it is the best one yet
            coverage = sum(is_solutions)
            coverages.append(coverage)
            try:
                plan_quality = sum(plan_lengths) / coverage
            except:
                continue
            if coverage > best_coverage or (coverage == best_coverage and plan_quality < best_plan_quality):
                best_coverage = coverage
                best_plan_quality = plan_quality
                best_planning_run = str(version_path)

        print(coverages)
        planning_results = dict(instances=len(problem_files)-1, max_coverage=max(coverages),
                                             min_coverage=min(coverages), avg_coverage=sum(coverages) / len(coverages),
                                             best_plan_quality=best_plan_quality, plans_directory=best_planning_run)
        print(planning_results)

        # save results of the best run
        save_results(results, policy_type, best_trained_policy_path, best_trained_val_loss,
                                  best_trained_bug_loss, planning_results)



    print(colored('Storing results', 'red', attrs=['bold']))
    results = pd.DataFrame(results)
    results.to_csv(args.logdir / "results.csv")
    print(results)


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)