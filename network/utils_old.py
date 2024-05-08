from torch.utils.data.dataset import Dataset
import random
import pytorch_lightning as pl
import torch
from pathlib import Path
from torch_geometric.data import Data, Batch


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


def states_to_graphs(states, predicate_dict, predicate_ids, max_arity):
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

        # TODO: COMPUTING IT LIKE THIS IS REDUNDANT I GUESS
        object_ids = random.sample(range(0, 100000), len(objects))
        object_to_id = {}
        for i in range(len(objects)):
            object_to_id[objects[i]] = object_ids[i]

        # create nodes for objects
        for i in range(len(objects)):
            obj = objects[i]
            # create tensor for object node's feature
            object_node = torch.ones(3 + max_arity) * -1
            # first feature indicates that this is an object node
            object_node[0] = 0
            # second feature is the id of the object
            object_node[1] = object_to_id[obj]  # TODO: RANDOMIZE THIS!!!!

            nodes_x.append(object_node)

        # create nodes for atoms and add edges between objects and atoms
        for i in range(len(atoms)):
            predicate, arguments = atoms[i]
            # create tensor for relation node's feature
            atom_node = torch.ones(3 + max_arity) * -1
            # first feature indicates that this is an atom node
            atom_node[0] = 1
            # second feature is the id of the predicate
            atom_node[1] = predicate_ids[predicate]
            # next features are the object ids of the arguments
            for x, argument in enumerate(arguments):
                atom_node[x + 2] = object_to_id[argument]

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


def state_to_graph(state, predicate_dict, predicate_ids, max_arity):
    atoms = []
    max_id = 0
    # the states only have one entry for each predicate, so to get the individual atoms we need to split according
    # to the arity of the predicate
    for predicate in state.keys():
        arity = predicate_dict[predicate]
        # some atoms may have no arguments!
        if arity == 0:
            atoms.append((predicate, []))
            continue
        arguments = state[predicate]
        # keep track of the object with the highest id such that we now how many objects there are
        for arg in arguments:
            if arg > max_id:
                max_id = arg
        split_arguments = [arguments[i:i + arity] for i in range(0, len(arguments), arity)]
        for argument in split_arguments:
            atoms.append((predicate, argument))

    objects = list(range(max_id + 1))

    # TODO: COMPUTING IT LIKE THIS IS REDUNDANT I GUESS
    object_ids = random.sample(range(0, 100000), len(objects))
    object_to_id = {}
    for i in range(len(objects)):
        object_to_id[objects[i]] = object_ids[i]


    nodes_x = []
    edge_index = [[], []]
    # create nodes for objects
    for i in range(len(objects)):
        obj = objects[i]
        # create tensor for object node's feature
        object_node = torch.ones(3 + max_arity) * -1
        # first feature indicates that this is an object node
        object_node[0] = 0
        # third feature is the id of the object
        object_node[1] = object_to_id[obj]

        nodes_x.append(object_node)

    # create nodes for atoms and add edges between objects and atoms
    for i in range(len(atoms)):
        predicate, arguments = atoms[i]
        # create tensor for relation node's feature
        atom_node = torch.ones(3 + max_arity) * -1
        # first feature indicates that this is an atom node
        atom_node[0] = 1
        # third feature is the id of the predicate
        atom_node[1] = predicate_ids[predicate]
        # next features are the object ids of the arguments
        for x, argument in enumerate(arguments):
            atom_node[x + 2] = object_to_id[argument.item()]

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
                edge_index[1].append(argument.item())
                edge_index[0].append(argument.item())
                edge_index[1].append(i + len(objects))

    nodes_x = torch.stack(nodes_x).float()
    edge_index = torch.tensor(edge_index).long()
    graph_state = Data(x=nodes_x, edge_index=edge_index, num_nodes=len(objects) + len(atoms))
    graph_state.validate(raise_on_error=True)

    return graph_state

from timeit import default_timer as timer
from termcolor import colored
from generators import load_pddl_problem_with_augmented_states
def planning(predicate_dict, predicate_ids, max_arity, args, policy, model, domain_file, problem_file, device):
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
def compute_traces_with_augmented_states(predicate_dict, predicate_ids, max_arity, actions, initial, goal, language, model: pl.LightningModule, augment_fn = None, cycles: str = 'avoid', max_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None):
    max_test_graph_size = 0
    min_test_graph_size = 1000000000
    logger = False

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
        state_graph = state_to_graph(encoded_states[0], predicate_dict, predicate_ids, max_arity)
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
                # print(f'No applicable action that yields unvisited state for current_state')
                # print(f'Applicable actions = {_get_applicable_actions(current_state, actions)}')
                break

            successor_actions = [candidate[0] for candidate in successor_candidates]
            successor_states = [candidate[1] for candidate in successor_candidates]
            if logger: logger.debug(f'#actions={len(successor_actions)}, actions={successor_actions}')

            # calculate values for successors and best successor
            collated_input, encoded_states = _to_input(successor_states, goal_denotation, obj_encoding, augment_fn,
                                                       language, device, logger)
            state_graphs = [state_to_graph(encoded_state, predicate_dict, predicate_ids, max_arity) for encoded_state in encoded_states]
            state_graphs_batch = Batch.from_data_list(state_graphs)  # TODO: DEVICE????

            if state_graphs[0].num_nodes > max_test_graph_size:
                max_test_graph_size = state_graphs[0].num_nodes
            if state_graphs[0].num_nodes < min_test_graph_size:
                min_test_graph_size = state_graphs[0].num_nodes

            assert model.training == False

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

        # print(f'Max test graph size: {max_test_graph_size}')
        # print(f'Min test graph size: {min_test_graph_size}')

        return action_trace, state_trace, value_trace, reached_goal, num_evaluations


from gnns import create_GNN
from gnns import GraphConvolutionNetwork, GraphConvolutionNetworkV2, GraphAttentionNetwork, GraphAttentionNetworkV2, GraphIsomorphismNetwork
from gnns import Performer, Transformer, GCNGPS
from gnns import mse_loss, mae_loss
from torch_geometric.nn import global_add_pool, global_max_pool
model_classes = {
    ("GCN", "ADD", "MSE"): create_GNN(GraphConvolutionNetwork, global_add_pool, mse_loss),
    ("GCNV2", "ADD", "MSE"): create_GNN(GraphConvolutionNetworkV2, global_add_pool, mse_loss),
    ("GAT", "ADD", "MSE"): create_GNN(GraphAttentionNetwork, global_add_pool, mse_loss),
    ("GATV2", "ADD", "MSE"): create_GNN(GraphAttentionNetworkV2, global_add_pool, mse_loss),
    ("GIN", "ADD", "MSE"): create_GNN(GraphIsomorphismNetwork, global_add_pool, mse_loss),

    ("GCN", "MAX", "MSE"): create_GNN(GraphConvolutionNetwork, global_max_pool, mse_loss),
    ("GCNV2", "MAX", "MSE"): create_GNN(GraphConvolutionNetworkV2, global_max_pool, mse_loss),
    ("GAT", "MAX", "MSE"): create_GNN(GraphAttentionNetwork, global_max_pool, mse_loss),
    ("GATV2", "MAX", "MSE"): create_GNN(GraphAttentionNetworkV2, global_max_pool, mse_loss),
    ("GIN", "MAX", "MSE"): create_GNN(GraphIsomorphismNetwork, global_max_pool, mse_loss),

    ("GCN", "MAX", "MAE"): create_GNN(GraphConvolutionNetwork, global_max_pool, mae_loss),
    ("GCNV2", "MAX", "MAE"): create_GNN(GraphConvolutionNetworkV2, global_max_pool, mae_loss),
    ("GAT", "MAX", "MAE"): create_GNN(GraphAttentionNetwork, global_max_pool, mae_loss),
    ("GATV2", "MAX", "MAE"): create_GNN(GraphAttentionNetworkV2, global_max_pool, mae_loss),
    ("GIN", "MAX", "MAE"): create_GNN(GraphIsomorphismNetwork, global_max_pool, mae_loss),

    ("Performer", "ADD", "MSE"): create_GNN(Performer, global_add_pool, mse_loss),
    ("GCNGPS", "ADD", "MSE"): create_GNN(GCNGPS, global_add_pool, mse_loss),
    ("GCNGPS", "ADD", "MAE"): create_GNN(GCNGPS, global_add_pool, mae_loss),

    ("GCNGPS", "MAX", "MSE"): create_GNN(GCNGPS, global_max_pool, mse_loss),
    ("Performer", "MAX", "MSE"): create_GNN(Performer, global_max_pool, mse_loss),
    ("Transformer", "MAX", "MSE"): create_GNN(Transformer, global_max_pool, mse_loss)
}
