from termcolor import colored
import random
import torch
import json
from datasets import g_dataset_methods
from torch_geometric.data import Data

def load_datasets(args):
    print(colored('Loading datasets...', 'green', attrs = [ 'bold' ]))
    try:
        load_dataset, collate = g_dataset_methods["selfsupervised_suboptimal"]
    except KeyError:
        raise NotImplementedError(f"Loss function '{args.loss}'")

    # load indices of states to use for training and validation sets
    if args.train_indices is not None:
        with open(args.train_indices, 'r') as f:
            train_indices = json.load(f)
    else:
        train_indices = {}
    if args.val_indices is not None:
        with open(args.val_indices, 'r') as f:
            val_indices = json.load(f)
    else:
        val_indices = {}

    (train_dataset, predicates, train_indices_selected_states) = load_dataset(args.train, train_indices, args.max_samples_per_file, args.max_samples, args.verify_datasets)
    (validation_dataset, _, validation_indices_selected_states) = load_dataset(args.validation, val_indices, args.max_samples_per_file, args.max_samples, args.verify_datasets)

    print(f'{len(predicates)} predicate(s) in dataset; predicates=[ {", ".join([ f"{name}/{arity}" for name, arity in predicates ])} ]')
    return predicates, collate, train_dataset, validation_dataset, train_indices_selected_states, validation_indices_selected_states


def states_to_graphs(states, predicate_dict, predicate_ids, max_arity):
    # first we decode the given states such that we have easy access to the label, the relations and the objects
    decoded_states = []
    for (label, state, successor_states) in states:
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
        label = label.float()
        graph_state = Data(x=nodes_x, edge_index=edge_index, y=label, num_nodes=len(objects) + len(atoms))
        graph_state.validate(raise_on_error=True)
        graph_states.append(graph_state)

    return graph_states


