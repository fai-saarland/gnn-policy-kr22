import sys
from copy import deepcopy as deepcopy
from pathlib import Path
from termcolor import colored
import pytorch_lightning as pl
from torch.functional import Tensor
from typing import Dict, List, Tuple
import torch

from tarski.fstrips.fstrips import AddEffect, DelEffect
from tarski.model import Model as PDDLState
from tarski.syntax.formulas import Atom
from tarski.model import create as make_tarski_state


def _collate(batch: List[Dict[str, Tensor]], device):
    """
    Input: [state]
    Output: (states, sizes)
    """
    input = {}
    sizes = []
    offset = 0
    for state in batch:
        max_size = 0
        for predicate, values in state.items():
            if values.nelement() > 0:
                max_size = max(max_size, int(torch.max(values)) + 1)
            if predicate not in input: input[predicate] = []
            input[predicate].append(values + offset)
        sizes.append(max_size)
        offset += max_size
    for predicate in input.keys():
        input[predicate] = torch.cat(input[predicate]).view(-1).to(device=device, non_blocking=True)
    return (input, sizes)

def load_pddl_problem_with_augmented_states(domain: Path, problem: Path, registry_filename: Path = None, registry_key: str = None, logger = None):
    from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
    from tarski.errors import UndefinedPredicate
    from tarski.io import PDDLReader

    parser = PDDLReader(raise_on_error=True)
    parser.parse_domain(str(domain))
    problem = parser.parse_instance(str(problem))
    actions = ground_problem_schemas_into_plain_operators(problem)
    predicates = [ predicate for predicate in problem.language.predicates if '=' not in str(predicate.name) ]
    language = problem.language

    # calculate sets of static and dynamic predicates
    all_predicates = set([ predicate.name for predicate in predicates ])
    dynamic_predicates = set()
    for action_name in problem.actions:
        action = problem.actions[action_name]
        for effect in action.effects:
            dynamic_predicates.add(str(effect.atom.predicate.name))
    static_predicates = all_predicates - dynamic_predicates
    if logger: logger.info(f'Predicates: static={static_predicates}, dynamic={dynamic_predicates}')

    # set augmentation function if registry file and proper key found
    augment_fn = None
    if registry_filename is not None:
        # calculate registry_key from domain path if needed
        if not registry_key:
            known_domains = set([ dname.name for dname in Path('../Data/pddl').glob('*') if dname.is_dir() ])
            domain_parts = set(domain.parts)
            intersection = known_domains & domain_parts
            registry_key = intersection.pop() if len(intersection) == 1 else None

        # get augmentation function
        if registry_key:
            from sys import path as sys_path
            import os
            sys_path.append(os.path.dirname(os.path.abspath(registry_filename)))
            from augmentation import load_registry, get_registry_record, update_registry_record, construct_augmentation_function_simple

            if logger: logger.info(f"Using '{registry_key}' as registry key into '{registry_filename}'")
            registry = load_registry(registry_filename)
            registry_record = get_registry_record(registry_key, registry, registry_filename)
            update_registry_record(registry_record, static_predicates, dynamic_predicates)
            augment_fn = construct_augmentation_function_simple(registry_record)

            # extend language with derived and goal predicates
            if augment_fn is not None:
                # setup 'arities' for all predicates in language
                for predicate in language.predicates:
                    if '=' not in str(predicate.name):
                        registry_record['arities'][str(predicate.name)] = predicate.arity

                # get object 'object'
                try:
                    obj = language.get_sort('object')
                except:
                    if logger: logger.fatal(f"domain doesn't define 'object' type")
                    exit(-1)

                # extend language
                for predicate_name in registry_record['static'] | registry_record['dynamic']:
                    arity = registry_record['arities'][predicate_name]
                    args = [ obj for i in range(arity) ]
                    try:
                        language.get_predicate(predicate_name)
                        if predicate_name in registry_record['dynamic']:
                            predicate = language.predicate(predicate_name + '@', *args)
                            if logger: logger.info(f"Extending domain language with predicate '{predicate}'")
                    except UndefinedPredicate:
                        predicate = language.predicate(predicate_name, *args)
                        if logger: logger.info(f"Extending domain language with predicate '{predicate}'")
                    predicates = [ predicate for predicate in problem.language.predicates if '=' not in str(predicate.name) ]
        else:
            if logger: logger.warning(f'Unable to calculate registry key; bypassing registry ...')

    return {
        'actions': actions,
        'initial': problem.init,
        'goal': problem.goal,
        'predicates': predicates,
        'language': language,
        'augment_fn': augment_fn
    }

def load_pddl_problem(domain: Path, problem: Path, logger = None):
    from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
    from tarski.io import PDDLReader
    parser = PDDLReader(raise_on_error=True)
    parser.parse_domain(str(domain))
    problem = parser.parse_instance(str(problem))
    actions = ground_problem_schemas_into_plain_operators(problem)
    predicates = [ predicate for predicate in problem.language.predicates if '=' not in str(predicate.name) ]
    language = problem.language

    return {
        'actions': actions,
        'initial': problem.init,
        'goal': problem.goal,
        'predicates': predicates,
        'language': language,
        'augment_fn' : None
    }

def create_object_encoding(objects):
    object_ids = dict([ (obj.name, i) for i, obj in enumerate(objects) ])
    object_ids_inv = dict([ (i, obj.name) for i, obj in enumerate(objects) ])
    object_ids.update(object_ids_inv)
    return object_ids

def _get_goal_denotation(goals, obj_encoding: Dict[str, int]) -> Dict[str, List[int]]:
    atoms = [ goals ] if isinstance(goals, Atom) else goals.subformulas
    predicate_names = [ atom.predicate.name + '_goal' for atom in atoms ]
    argument_ids = [ [ obj_encoding[obj.name] for obj in atom.subterms ] for atom in atoms]
    denotation = dict()
    for predicate_name, object_ids in zip(predicate_names, argument_ids):
        if predicate_name not in denotation:
            denotation[predicate_name] = []
        denotation[predicate_name].extend(object_ids)
    return denotation

def _encode_atoms(state: PDDLState, object_ids: Dict[str, int], logger = None) -> dict:
    encoded_atoms = {}
    for atom in state.as_atoms():
        if hasattr(atom, 'predicate'):
            predicate_name = atom.predicate.name
            encoded_objects = [ object_ids[obj.name] for obj in atom.subterms ]
            if predicate_name not in encoded_atoms: encoded_atoms[predicate_name] = []
            encoded_atoms[predicate_name].extend(encoded_objects)
    #if logger: logger.debug(f'encoded_atoms={encoded_atoms}')
    return encoded_atoms

def _encode_state(state: PDDLState, goal_denotation, object_ids: Dict[str, int], logger = None) -> dict:
    encoded_state = _encode_atoms(state, object_ids, logger)
    for predicate_name in goal_denotation:
        encoded_state[predicate_name] = goal_denotation[predicate_name]
    #if logger: logger.debug(f'encoded_state={encoded_state}')
    return encoded_state

def _to_tensors(encoded_states: List[Dict[str, List[int]]]):
    for encoded_state in encoded_states:
        for pred_id, obj_ids in encoded_state.items():
            encoded_state[pred_id] = torch.tensor(obj_ids)

def _to_input(states: List[PDDLState], goal_denotation, obj_encoding, augment_fn, language, device, logger = None):
    encoded_states = [ _encode_state(_apply_derived_predicates(state, goal_denotation, obj_encoding, augment_fn, language), goal_denotation, obj_encoding, logger) for state in states ]
    _to_tensors(encoded_states)
    return _collate(encoded_states, device), encoded_states

def _get_applicable_actions(state, actions):
    return [ operator for operator in actions if state[operator.precondition] ]

def _apply_action(state, action):
    state = deepcopy(state)
    delete_atoms = [ effect.atom for effect in action.effects if isinstance(effect, DelEffect) ]
    add_atoms = [ effect.atom for effect in action.effects if isinstance(effect, AddEffect) ]
    for delete_atom in delete_atoms: state.discard(delete_atom.predicate, *delete_atom.subterms)
    for add_atom in add_atoms: state.add(add_atom.predicate, *add_atom.subterms)
    return state

def _get_successor_states(state, actions):
    applicable_actions = _get_applicable_actions(state, actions)
    return [ (action, _apply_action(state, action)) for action in applicable_actions ]

def _map_tarski_state_into_plain_state(tarski_state: PDDLState):
    plain_state = dict(_internal=dict())
    for key in tarski_state.predicate_extensions:
        #plain_state[key[0]] = tarski_state.predicate_extensions[key]
        plain_state[key[0]] = set([ tuple([ str(wref.expr) for wref in d ]) for d in tarski_state.predicate_extensions[key] ])
        plain_state['_internal'][key[0]] = key
    return plain_state

def _augment_tarski_state(tarski_state: PDDLState, augmentation, language):
    for key in augmentation:
        predicate = language.get_predicate(key)
        for d in augmentation[key]:
            #denotation = tuple([ wref.expr for wref in d ])
            #tarski_state.add(predicate, *denotation)
            tarski_state.add(predicate, *d)
    return tarski_state

def _apply_derived_predicates(tarski_state: PDDLState, goal_denotation, obj_encoding, augment_fn, language):
    assert augment_fn is None or language is not None, f'language={language}, augment_fn={augment_fn}'
    if augment_fn is not None:
        # get plain state from PDDL (tarski) state
        plain_state = _map_tarski_state_into_plain_state(tarski_state)

        # augment plain state with goal predicates of type <pred>@
        added_goal_denotations = []
        for predicate_name in goal_denotation:
            assert predicate_name[-5:] == '_goal'
            goal_predicate = language.get_predicate(predicate_name[:-5] + '@')
            denotation = goal_denotation[predicate_name]
            arity = goal_predicate.arity
            plain_state[str(goal_predicate.name)] = set()
            for i in range(0, len(denotation), arity):
                plain_state[str(goal_predicate.name)].add(tuple(map(lambda i: obj_encoding[i], denotation[i:i+arity])))
            assert goal_predicate.name not in plain_state['_internal']
            plain_state['_internal'][goal_predicate.name] = goal_predicate.signature
            added_goal_denotations.append(goal_predicate.name)

        # augment plain state with derived predicates
        augmented_state = augment_fn(plain_state, 'static')
        augmented_state = augment_fn(augmented_state, 'dynamic')

        # calculate augmentation
        keys = [ key for key in augmented_state if key not in plain_state ]
        augmentation = dict([ (key, augmented_state[key]) for key in keys ])
        return _augment_tarski_state(deepcopy(tarski_state), augmentation, language)
    else:
        return tarski_state

def _spanner_solved(state, logger=None):
    # assuming that unsolvable check was previously done, task is solvable if bob is at gate

    # extract information from state
    extensions = state.list_all_extensions()
    at = dict([ (obj1.name, obj2.name) for (obj1, obj2) in extensions[('at', 'object', 'object')] ])

    # get bob location
    assert len(extensions[('man', 'object')]) == 1
    bob = extensions[('man', 'object')][0][0].name
    bob_loc = at[bob]
    return bob_loc == 'gate'

def _spanner_unsolvable(state, logger=None):
    # if bob is at gate, check he has enough useable spanners

    # extract information from state
    extensions = state.list_all_extensions()
    at = dict([ (obj1.name, obj2.name) for (obj1, obj2) in extensions[('at', 'object', 'object')] ])

    # get bob location
    assert len(extensions[('man', 'object')]) == 1
    bob = extensions[('man', 'object')][0][0].name
    bob_loc = at[bob]
    if bob_loc != 'gate': return False

    # get spanner and nut subsets
    carrying = set([ obj2.name for (obj1, obj2) in extensions[('carrying', 'object', 'object')] if obj1.name == bob ])
    useable = set([ obj.name for (obj,) in extensions[('useable', 'object')] ])
    spanner = set([ obj.name for (obj,) in extensions[('spanner', 'object')] ])
    loose_nuts = set([ obj.name for (obj,) in extensions[('loose', 'object')] ])
    carrying_and_useable_spanners = carrying & useable & spanner

    # at this point, task is solvable iff bob is carrying enough useable spanners
    if logger and len(carrying_and_useable_spanners) < len(loose_nuts):
        logger.info(f'unsolvable_state={state}')
        logger.info(f'carrying={carrying}, useable={useable}, spanner={spanner}, loose_nuts={loose_nuts}')
        logger.info(f'carrying_and_useable_spanners={carrying_and_useable_spanners}')
    return len(carrying_and_useable_spanners) < len(loose_nuts)

def policy_search(actions, initial, goals, obj_encoding: Dict[str, int], model: pl.LightningModule, cycles: str = 'avoid', max_state_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None):
    language = None
    augment_fn = None

    device = model.device
    closed_states = set()
    action_trace = []

    # calculate denotation of goal atoms that is equal for every state
    if logger: logger.info(f'goals={goals}')
    goal_denotation = _get_goal_denotation(goals, obj_encoding)

    # set initial state and value trace
    current_state = initial
    collated_input, encoded_states = _to_input([ current_state ], goal_denotation, obj_encoding, augment_fn, language, device, logger)
    state_trace = [ encoded_states[0] ]
    initial_values, initial_solvables = model(collated_input)
    value_trace = [ initial_values[0] + (1.0 - torch.round(torch.sigmoid(initial_solvables))[0]) * unsolvable_weight ]  # A large value indicates an unsolvable state
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

        # explore current state (avoid loops by removing already visited successors)
        successor_candidates = [ transition for transition in _get_successor_states(current_state, actions) ]
        if cycles == 'avoid':
            successor_candidates = [ transition for transition in successor_candidates if transition[1] not in closed_states ]

        if len(successor_candidates) == 0:
            logger.info(f'No applicable action that yields unvisited state for current_state={current_state}')
            logger.info(f'Applicable actions = {_get_applicable_actions(current_state, actions)}')
            break

        successor_actions = [ candidate[0] for candidate in successor_candidates ]
        successor_states = [ candidate[1] for candidate in successor_candidates ]
        if logger: logger.debug(f'#actions={len(successor_actions)}, actions={successor_actions}')

        # calculate values for successors and best successor
        collated_input, encoded_states = _to_input(successor_states, goal_denotation, obj_encoding, augment_fn, language, device, logger)
        output_values, output_solvables = model(collated_input)
        output_values += (1.0 - torch.round(torch.sigmoid(output_solvables))) * unsolvable_weight
        best_successor_index = torch.argmin(output_values)
        num_evaluations += len(successor_actions)
        if logger:
            logger.debug(f'     values=[' + ", ".join([ f'{x[0]:.3f}' for x in output_values ]) + ']')
            logger.debug(f'  solvables=[' + ", ".join([ f'{x[0]:.3f}' for x in output_solvables ]) + ']')
            logger.debug(f'best_action={successor_actions[best_successor_index]} (index={best_successor_index})\n')

        # extend traces and set next current state
        action_trace.append(successor_actions[best_successor_index])
        state_trace.append(encoded_states[best_successor_index])
        value_trace.append(output_values[best_successor_index])
        current_state = successor_states[best_successor_index]
        if logger:
            logger.debug(f'current_state={current_state}')
            logger.debug('')

    reached_goal = current_state[goals]
    if logger: logger.debug(f'status={1 if reached_goal else 0}')
    return action_trace, state_trace, value_trace, reached_goal, num_evaluations

def compute_traces(actions, initial, goal, language, model: pl.LightningModule, cycles: str = 'avoid', max_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None):
    objects = language.constants()
    obj_encoding = create_object_encoding(objects)
    if logger: logger.info(f'{len(objects)} object(s), obj_encoding={obj_encoding}')

    with torch.no_grad():
        return policy_search(actions, initial, goal, obj_encoding, model, cycles=cycles, max_trace_length=max_trace_length, unsolvable_weight=unsolvable_weight, logger=logger)

def policy_search_with_augmented_states(actions, initial, goals, obj_encoding: Dict[str, int], language, model: pl.LightningModule, augment_fn = None, cycles: str = 'avoid', max_state_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None, is_spanner = False):
    device = model.device
    closed_states = set()
    action_trace = []

    # calculate denotation of goal atoms that is equal for every state
    if logger: logger.info(f'goals={goals}')
    goal_denotation = _get_goal_denotation(goals, obj_encoding)

    # set initial state and value trace
    current_state = initial
    collated_input, encoded_states = _to_input([ current_state ], goal_denotation, obj_encoding, augment_fn, language, device, logger)
    state_trace = [ encoded_states[0] ]
    initial_values, initial_solvables = model(collated_input)
    value_trace = [ initial_values[0] + (1.0 - torch.round(torch.sigmoid(initial_solvables))[0]) * unsolvable_weight ]  # A large value indicates an unsolvable state
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
            logger.info(colored(f'SPANNER TASK FAILURE', 'red', attrs=[ 'bold' ]))
            break
        elif is_spanner and _spanner_solved(current_state, logger):
            logger.info(colored(f'SPANNER TASK SOLVED', 'green', attrs=[ 'bold' ]))
            logger.info(f'current_state={current_state}')
            break

        # explore current state (avoid loops by removing already visited successors)
        successor_candidates = [ transition for transition in _get_successor_states(current_state, actions) ]
        if cycles == 'avoid':
            successor_candidates = [ transition for transition in successor_candidates if transition[1] not in closed_states ]

        if len(successor_candidates) == 0:
            logger.info(f'No applicable action that yields unvisited state for current_state={current_state}')
            logger.info(f'Applicable actions = {_get_applicable_actions(current_state, actions)}')
            break

        successor_actions = [ candidate[0] for candidate in successor_candidates ]
        successor_states = [ candidate[1] for candidate in successor_candidates ]
        if logger: logger.debug(f'#actions={len(successor_actions)}, actions={successor_actions}')

        # calculate values for successors and best successor
        collated_input, encoded_states = _to_input(successor_states, goal_denotation, obj_encoding, augment_fn, language, device, logger)
        output_values, output_solvables = model(collated_input)
        output_solvables = torch.round(torch.sigmoid(output_solvables))
        #output_values += (1.0 - torch.round(torch.sigmoid(output_solvables))) * unsolvable_weight
        output_values += (1.0 - output_solvables) * unsolvable_weight
        best_successor_index = torch.argmin(output_values)
        num_evaluations += len(successor_actions)
        if logger:
            logger.debug(f'     values=[' + ", ".join([ f'{x[0]:.3f}' for x in output_values ]) + ']')
            logger.debug(f'  solvables=[' + ", ".join([ f'{x[0]:.3f}' for x in output_solvables ]) + ']')
            logger.debug(f'best_action={successor_actions[best_successor_index]} (index={best_successor_index})\n')

        # extend traces and set next current state
        value_trace.append(output_values[best_successor_index])
        state_trace.append(encoded_states[best_successor_index])
        action_trace.append(successor_actions[best_successor_index])
        current_state = successor_states[best_successor_index]

        if logger:
            logger.debug(f'current_state={current_state}')
            logger.debug('')

    reached_goal = current_state[goals]
    if logger: logger.debug(f'status={1 if reached_goal else 0}')
    return action_trace, state_trace, value_trace, reached_goal, num_evaluations

def compute_traces_with_augmented_states(actions, initial, goal, language, model: pl.LightningModule, augment_fn = None, cycles: str = 'avoid', max_trace_length: int = 500, unsolvable_weight: float = 100000.0, logger = None, is_spanner = False):
    objects = language.constants()
    obj_encoding = create_object_encoding(objects)
    if logger: logger.info(f'{len(objects)} object(s), obj_encoding={obj_encoding}')

    with torch.no_grad():
        return policy_search_with_augmented_states(actions, initial, goal, obj_encoding, language, model, augment_fn=augment_fn, cycles=cycles, max_state_trace_length=max_trace_length, unsolvable_weight=unsolvable_weight, logger=logger, is_spanner=is_spanner)


def _variableMapping(language):
    try:
        line = next(sys.stdin)
        num_vars = int(line.strip())
        var_map = []
        facts = []
        for i in range(num_vars):
            line = next(sys.stdin)
            num_vals = int(line.strip())
            vals = []
            for j in range(num_vals):
                line = next(sys.stdin)
                line = line.strip()
                if not line.startswith('Atom'):
                    vals += [None]
                    continue
                atom_name = line.split(' ', 1)[1]
                atom_name = atom_name.replace('(', ' ')
                atom_name = atom_name.replace(', ', ' ')
                atom_name = atom_name.rstrip(')')
                els = atom_name.split()
                predicate = language.get_predicate(els[0])
                args = [language.get_constant(x) for x in els[1:]]
                atom = predicate(*args)
                vals += [atom]
                facts += [atom]
            var_map += [vals]
        return var_map, set(facts)

    except Exception as e:
        sys.stdout.write('ERR\n')
        sys.stdout.flush()
        raise e

def _actionMapping(actions):
    try:
        name_to_action = { a.ident() : a for a in actions }
        line = next(sys.stdin)
        num_actions = int(line.strip())
        action_map = {}
        for i in range(num_actions):
            line = next(sys.stdin)
            line = line.strip()
            els = line.split()
            name = els[0] + '(' + ', '.join(els[1:]) + ')'
            if name in action_map:
                raise Exception('Duplicate action names!')
            action_map[name_to_action[name]] = i
        return action_map, set(action_map.keys())

    except Exception as e:
        sys.stdout.write('ERR\n')
        sys.stdout.flush()
        raise e

def _collectStaticFacts(initial, available_facts, actions, language, logger):
    static_facts = deepcopy(initial)
    for fact in available_facts:
        static_facts.discard(fact.predicate, *fact.subterms)

    for a in actions:
        for e in a.effects:
            if isinstance(e, AddEffect):
                if e.atom not in available_facts:
                    if logger: logger.info(f'Add effect {e.atom} not in input -- considering it static')
            elif isinstance(e, DelEffect):
                if e.atom not in available_facts:
                    if logger: logger.info(f'Del effect {e.atom} not in input -- considering it static')
    return static_facts

def _serve_policy(actions, initial, goals, obj_encoding, language,
                  static_facts, var_map, action_map, available_actions,
                  model: pl.LightningModule,
                  augment_fn = None, unsolvable_weight: float = 100000.0,
                  logger = None, is_spanner = False):
    device = model.device
    # calculate denotation of goal atoms that is equal for every state
    if logger: logger.info(f'goals={goals}')
    goal_denotation = _get_goal_denotation(goals, obj_encoding)

    for line in sys.stdin:
        line = line.strip()
        fdr_state = [int(x) for x in line.split()]
        if logger: logger.info(f'STATE {fdr_state}')
        current_state = deepcopy(static_facts)
        for i in range(len(var_map)):
            atom = var_map[i][fdr_state[i]]
            current_state.add(atom.predicate, *atom.subterms)
        if logger: logger.info(f'Strips state {current_state}')
        if logger: logger.info(f'Init state {initial}')

        successor_candidates = [ transition for transition in _get_successor_states(current_state, actions) ]
        successor_candidates = sorted(successor_candidates, key = lambda x: x[0].ident())
        successor_actions = []
        successor_states = []
        for candidate in successor_candidates:
            if candidate[0] in available_actions:
                successor_actions += [candidate[0]]
                successor_states += [candidate[1]]
            else:
                if logger: logger.info(f'Skipping action {candidate[0]}')

        if logger: logger.info(f'#actions={len(successor_actions)}, actions={successor_actions}')
        collated_input, encoded_states = _to_input(successor_states, goal_denotation, obj_encoding, augment_fn, language, device, logger)
        output_values, output_solvables = model(collated_input)
        output_solvables = torch.round(torch.sigmoid(output_solvables))
        #output_values += (1.0 - torch.round(torch.sigmoid(output_solvables))) * unsolvable_weight
        output_values += (1.0 - output_solvables) * unsolvable_weight
        if logger: logger.info(f'Output values: {output_values}')
        out = ['{0} {1:.4f}'.format(action_map[successor_actions[idx]], output_values[idx][0])
                for idx in torch.argsort(output_values.flatten())]
        out = ' '.join(out)
        sys.stdout.write(f'{out}\n')
        sys.stdout.flush()

        best_successor_index = torch.argmin(output_values)
        best_action = successor_actions[best_successor_index]
        if logger: logger.info(f'Best action: {best_action}')
        #best_action_id = action_map[best_action]
        #sys.stdout.write(f'{best_action_id}\n')
        #sys.stdout.flush()

    return 0

def serve_policy(actions, initial, goal, language, model: pl.LightningModule,
                 augment_fn = None, unsolvable_weight: float = 100000.0,
                 logger = None, is_spanner = False):
    objects = language.constants()
    obj_encoding = create_object_encoding(objects)
    if logger: logger.info(f'{len(objects)} object(s), obj_encoding={obj_encoding}')

    # read mapping from variables to facts
    var_map, available_facts = _variableMapping(language)
    # read mapping from operator id to operator name
    action_map, available_actions = _actionMapping(actions)
    sys.stdout.write('OK\n')
    sys.stdout.flush()

    static_facts = _collectStaticFacts(initial, available_facts, actions, language, logger)
    if logger: logger.info(f'Static facts collected: {static_facts}')

    with torch.no_grad():
        return _serve_policy(actions, initial, goal, obj_encoding, language,
                             static_facts, var_map, action_map, available_actions,
                             model, augment_fn, unsolvable_weight, logger, is_spanner)


########################################################################################################################

class StaticServerData:
    static_facts = None
    actions = None
    initial = None
    goal = None
    language = None
    model: pl.LightningModule = None
    augment_fn = None
    unsolvable_weight: float = 100000.0
    var_map = None
    available_facts = None
    action_map = None
    available_actions = None
    obj_encoding = None
    device = None
    goal_denotation = None
    torch_context_handler = None
    num_vars = None


def expect_line(f, content, alternative_content=None):
    line = f.readline().rstrip()
    if alternative_content:
        assert line == content or line == alternative_content
    else:
        assert line == content


def parse_header(f):
    expect_line(f, "begin_version")
    expect_line(f, "3")
    expect_line(f, "end_version")
    expect_line(f, "begin_metric")
    expect_line(f, "0", "1")
    expect_line(f, "end_metric")


def parse_variables(f, language):
    # read mapping from variables to facts
    num_vars = int(f.readline().rstrip())
    var_map = []
    facts = []
    for i in range(num_vars):
        expect_line(f, "begin_variable")
        f.readline()  # skip variable name
        f.readline()  # skip axiom layer
        num_vals = int(f.readline().rstrip())
        vals = []
        for j in range(num_vals):
            line = f.readline().rstrip()
            if not line.startswith('Atom'):
                vals += [None]
                continue
            atom_name = line.split(' ', 1)[1]
            atom_name = atom_name.replace('(', ' ')
            atom_name = atom_name.replace(', ', ' ')
            atom_name = atom_name.rstrip(')')
            els = atom_name.split()
            predicate = language.get_predicate(els[0])
            args = [language.get_constant(x) for x in els[1:]]
            atom = predicate(*args)
            vals += [atom]
            facts += [atom]
        var_map += [vals]
        expect_line(f, "end_variable")
    return var_map, set(facts), num_vars


def parse_mutex_groups(f):
    num_mutex_groups = int(f.readline().rstrip())
    for _ in range(num_mutex_groups):
        expect_line(f, "begin_mutex_group")
        while line := f.readline().rstrip():
            if line == "end_mutex_group":
                break


def parse_initial_state(f):
    expect_line(f, "begin_state")
    while line := f.readline().rstrip():
        if line == "end_state":
            break


def parse_goals(f):
    expect_line(f, "begin_goal")
    num_goals = int(f.readline().rstrip())
    for _ in range(num_goals):
        f.readline()
    expect_line(f, "end_goal")


def parse_operators(f, actions):
    # read mapping from operator id to operator name
    name_to_action = {a.ident(): a for a in actions}
    num_actions = int(f.readline().rstrip())
    action_map = {}
    for i in range(num_actions):
        expect_line(f, "begin_operator")
        line = f.readline().strip()
        els = line.split()
        name = els[0] + '(' + ', '.join(els[1:]) + ')'
        if name in action_map:
            raise Exception('Duplicate action names!')
        action_map[name_to_action[name]] = i
        # skip over operator details
        while line := f.readline().rstrip():
            if line == "end_operator":
                break
    return action_map, set(action_map.keys())


def parse_sas_file(sas_file, language, actions):
    with open(sas_file) as f:
        try:
            parse_header(f)
            StaticServerData.var_map, StaticServerData.available_facts, StaticServerData.num_vars = parse_variables(f, language)
            parse_mutex_groups(f)
            parse_initial_state(f)
            parse_goals(f)
            StaticServerData.action_map, StaticServerData.available_actions = parse_operators(f, actions)
        except Exception as e:
            print('ERROR: Cannot determine mappings\n')
            raise e

def state_and_successors(fdr_state):
    current_state = deepcopy(StaticServerData.static_facts)
    for i in range(len(StaticServerData.var_map)):
        atom = StaticServerData.var_map[i][fdr_state[i]]
        # TODO check this
        if atom:
            current_state.add(atom.predicate, *atom.subterms)

    successor_candidates = [transition for transition in _get_successor_states(current_state, StaticServerData.actions)]
    successor_candidates = sorted(successor_candidates, key=lambda x: x[0].ident())
    successor_actions = []
    successor_states = []
    for candidate in successor_candidates:
        if candidate[0] in StaticServerData.available_actions:
            successor_actions += [candidate[0]]
            successor_states += [candidate[1]]
    return current_state, successor_actions, successor_states

def apply_policy_to_state(fdr_state):
    current_state, successor_actions, successor_states = state_and_successors(fdr_state)
    collated_input, encoded_states = _to_input(successor_states, StaticServerData.goal_denotation, StaticServerData.obj_encoding, StaticServerData.augment_fn, StaticServerData.language, StaticServerData.device, None)
    output_values, output_solvables = StaticServerData.model(collated_input)
    output_solvables = torch.round(torch.sigmoid(output_solvables))
    output_values += (1.0 - output_solvables) * StaticServerData.unsolvable_weight
    # out = ['{0} {1:.4f}'.format(StaticServerData.action_map[successor_actions[idx]], output_values[idx][0])
    #        for idx in torch.argsort(output_values.flatten())]
    # out = ' '.join(out)
    best_successor_index = torch.argmin(output_values)
    best_action = successor_actions[best_successor_index]
    best_action_id = StaticServerData.action_map[best_action]
    return best_action_id

def apply_policy_to_state_prob_dist(fdr_state):
    current_state, successor_actions, successor_states = state_and_successors(fdr_state)
    collated_input, encoded_states = _to_input(successor_states, StaticServerData.goal_denotation, StaticServerData.obj_encoding, StaticServerData.augment_fn, StaticServerData.language, StaticServerData.device, None)
    output_values, output_solvables = StaticServerData.model(collated_input)
    output_solvables = torch.round(torch.sigmoid(output_solvables))
    output_values += (1.0 - output_solvables) * StaticServerData.unsolvable_weight

    sum_output_values = sum(output_values)
    if sum_output_values > 0.:
        output_values /= sum_output_values

    output_values = [x[0] for x in output_values.tolist()]
    actions = [StaticServerData.action_map[x] for x in successor_actions]
    out = list(zip(actions, output_values))
    return out


def setup_policy_server(actions, initial, goal, language, model: pl.LightningModule, sas_file,
                        augment_fn=None, unsolvable_weight: float = 100000.0):
    StaticServerData.actions = actions
    StaticServerData.initial = initial
    StaticServerData.goal = goal
    StaticServerData.language = language
    StaticServerData.model = model
    StaticServerData.augment_fn = augment_fn
    StaticServerData.unsolvable_weight = unsolvable_weight
    print("Setting up GNN policy server")
    objects = language.constants()
    StaticServerData.obj_encoding = create_object_encoding(objects)
    parse_sas_file(sas_file, StaticServerData.language, StaticServerData.actions)
    print("Read variable and action mappings")
    StaticServerData.static_facts = _collectStaticFacts(StaticServerData.initial, StaticServerData.available_facts, StaticServerData.actions, StaticServerData.language, None)
    StaticServerData.torch_context_handler = torch.no_grad()
    StaticServerData.torch_context_handler.__enter__()
    # TODO shut this down again
    StaticServerData.device = StaticServerData.model.device
    # calculate denotation of goal atoms that is equal for every state
    StaticServerData.goal_denotation = _get_goal_denotation(StaticServerData.goal, StaticServerData.obj_encoding)


def get_num_vars():
    return StaticServerData.num_vars

