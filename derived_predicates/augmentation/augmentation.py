from sys import stdin, stdout, argv
from pathlib import Path
from itertools import product

from sys import path as sys_path
sys_path.append('../RelationalNeuralNetwork')
from datasets.protobuf import LabeledProblem, Predicate, LabeledState, State, Atom

def _apply_rule(ext_state: dict, pred: str, arity: int, head: str, body: list):
    global logger

    assert pred == head[:head.index('(')], f"Rule head doesn't match predicate '{pred}' in registry"
    body_keys = [ atom[:atom.index('(')] for atom in body ]
    body_vars = [ atom[1+atom.index('('):atom.index(')')].split(',') for atom in body ]
    values = [ ext_state[key] if key in ext_state else set() for key in body_keys ]
    assert '@' not in head
    head_vars = head[1+head.index('('):-1].split(',')
    #logger.debug(f'_apply_rule: head={head}, vars={head_vars}, body={body}, keys={body_keys}, at-goal={body_at_goal}, vars={body_vars}, values={values}')
    assert len(head_vars) == arity, f"Number of variables in rule head '{head}' doesn't match declared arity '{pred}/{arity}'"

    something_added = False
    for tup in product(*values): # WARNING: full joint of denotations of atoms in body, exponential in body size!
        trigger = True
        args = tuple(tup)
        assignment = dict()
        for i, arg in enumerate(args):
            key = body_keys[i]
            if key not in ext_state or arg not in ext_state[key]:
                trigger = False
                break
            else:
                inconsistent = False
                for j, var in enumerate(body_vars[i]):
                    if var not in assignment: assignment[var] = set()
                    assignment[var].add(arg[j])
                    if len(assignment[var]) > 1:
                        inconsistent = True
                        break
                if inconsistent:
                    trigger = False
                    break
                #logger.debug(f'match: key={key}, arg={arg}, vars={body_vars[i]}, assignment={assignment}')

        # if trigger, add atom in head
        if trigger:
            atom_args = tuple([ next(iter(assignment[var])) if var.isupper() else var for var in head_vars ])
            if atom_args not in ext_state[pred]:
                something_added = True
                ext_state[pred].add(atom_args)
                #logger.debug(f'trigger: pred={pred}/{arity}, head={head}, body={body}, args={args}, assignment={assignment}, atom_args={atom_args}')

    return something_added

def _apply_rules_for_pred(ext_state: dict, pred: str, arity: int, rules: list) -> bool:
    if pred not in ext_state: ext_state[pred] = set()
    something_added = False
    some_change = True
    while some_change:
        some_change = False
        for head, body in rules:
            if _apply_rule(ext_state, pred, arity, head, body):
                something_added = True
                some_change = True
    return something_added

def _apply_rules(ext_state: dict, registry_record: dict, ptype: str):
    rules = registry_record['rules']
    something_added = True
    while something_added:
        something_added = False
        for key in sorted(rules.keys()):
            assert len(key) > 2 and key[-2] == '/', f"Unexpected predicate name '{key}' in registry (format is <pred_name>/<arity>)"
            name = key[:-2]
            arity = int(key[-1:])
            if name in registry_record[ptype]:
                if _apply_rules_for_pred(ext_state, name, arity, rules[key]):
                    something_added = True

def _get_ext_state(state: State, predicates) -> dict:
    ext_state = dict()
    for atom in state.Atoms:
        pred_id = atom.PredicateId
        pred_name = predicates[pred_id].Name
        args = atom.ObjectIds
        if pred_name not in ext_state: ext_state[pred_name] = set()
        ext_state[pred_name].add(tuple(args))
    return ext_state

# Load registry file
def load_registry(registry_filename: Path):
    if registry_filename and registry_filename.exists() and registry_filename.is_file():
        import json
        return json.load(registry_filename.open('r'))
    else:
        return None

# Get record from registry associated with input_key while processing referrals
def get_registry_record(input_key: str, registry: dict, registry_filename: str, debug: bool = False) -> dict:
    if not registry or input_key not in registry:
        if debug: logger.debug(f"Nothing to do since '{input_key}' is not registered in '{registry_filename}'")
        return None

    deferrals = []
    key = input_key
    record = registry[key]
    while 'defer-to' in record and record['defer-to'] not in deferrals:
        next_key = record['defer-to']
        deferrals.append(next_key)
        if debug: logger.debug(f"Deferral of '{key}' to '{next_key}' in '{registry_filename}'")
        record = registry[next_key]
        key = next_key

    if 'defer-to' in record:
        if debug: logger.error(f"Circular deferrals in '{registry_filename}'; deferrals={deferrals}")
        return None
    elif 'rules' not in record:
        if debug: logger.debug(f"Nothing to do since no rules in record for '{input_key}'")
        return None
    else:
        return record

# Update registry records with fields 'arities', 'static', and 'dynamic', later needed by
# the augmentation function
def update_registry_record(record: dict, static_predicates: set, dynamic_predicates: set):
    # if no rules in record, nothing to do
    if not record or 'rules' not in record: return

    # store arities for derived predicates
    record['arities'] = dict()
    for key in record['rules']:
        name = key[:key.index('/')]
        arity = int(key[1+key.index('/'):])
        assert name not in record['arities'] or record['arities'][name] == arity
        record['arities'][name] = arity

    record['static'] = set(static_predicates)
    record['dynamic'] = set(dynamic_predicates)

    # identify derived predicates as static or dynamic depending on the
    # type of the atoms in the rules (fixpoint calculation)
    some_change = True
    while some_change:
        some_change = False
        for key in record['rules']:
            name = key[:key.index('/')]

            is_dynamic = False
            for rule in record['rules'][key]:
                for atom_in_body in rule[1]:
                    atom_name = atom_in_body[:atom_in_body.index('(')]
                    if atom_name in record['dynamic']:
                        is_dynamic = True
                        break
                if is_dynamic: break

            if is_dynamic:
                record['static'].discard(name)
                if name not in record['dynamic']:
                    some_change = True
                    record['dynamic'].add(name)
            else:
                assert name not in record['dynamic']
                if name not in record['static']:
                    some_change = True
                    record['static'].add(name)

def augment_ext_state(ext_state: dict, augmentation: dict):
    new_predicates = set()
    for key in augmentation.keys():
        if key not in ext_state:
            ext_state[key] = set()
            new_predicates.add(key)
        ext_state[key] = ext_state[key] | augmentation[key]
    return new_predicates

def contract_ext_state(ext_state: dict, augmentation: dict, new_predicates: set()):
    for key in augmentation.keys():
        if key in new_predicates:
            ext_state.pop(key, None)
        else:
            assert key in ext_state
            ext_state[key] = ext_state[key] - augmentation[key]

# Returns function to augment states with derived predicates. Input is augmented record from
# registry (that tells which derived predicates are static or dynamic), and predicates.
# The last two directly read from .states file.
def construct_augmentation_function(registry_record: dict, predicates):
    inv_map_predicates = dict([ (pred.Name, pred.Id) for pred in predicates ])
    def augment_fn(state: State, ptype: str, additional=dict(), logger=None) -> State:
        if not registry_record or not registry_record[ptype]:
            return state, dict()

        # add additional facts that can be used by rules
        ext_state = _get_ext_state(state, predicates)
        if logger: logger.debug(f'ext_state={ext_state}')
        new_predicates = augment_ext_state(ext_state, additional)
        if logger: logger.debug(f'ext_state.augmented={ext_state}, new_predicates={new_predicates}')

        # apply rules until fixpoint is reached
        _apply_rules(ext_state, registry_record, ptype)
        if logger: logger.debug(f'ext_state.rules={ext_state}')

        # remove facts from additional
        contract_ext_state(ext_state, additional, new_predicates)
        if logger: logger.debug(f'ext_state.contract={ext_state}')

        # construct state from ext_state
        new_state = State()
        for name in ext_state:
            if name not in inv_map_predicates:
                predicate = predicates.add()
                predicate.Id = len(predicates) - 1
                assert name in registry_record['arities'], f'Predicate {name} not found in registry record {registry_record["arities"]}'
                predicate.Arity = registry_record['arities'][name]
                predicate.Name = name
                inv_map_predicates[name] = predicate.Id
            pred_id = inv_map_predicates[name]

            for tup in ext_state[name]:
                atom = new_state.Atoms.add()
                atom.PredicateId = pred_id
                for i in tup: obj = atom.ObjectIds.append(i)
        return new_state, ext_state

    return augment_fn

# Same as before but where predicates are not updated and state is a dict
def construct_augmentation_function_simple(registry_record: dict):
    def augment_fn(ext_state: dict, ptype: str, additional=dict(), logger=None) -> dict:
        if not registry_record or not registry_record[ptype]:
            return ext_state

        # add additional facts that can be used by rules
        new_state = dict(ext_state)
        if logger: logger.debug(f'new_state={new_state}')
        new_predicates = augment_ext_state(new_state, additional)
        if logger: logger.debug(f'new_state.augmented={new_state}, new_predicates={new_predicates}')

        # apply rules until fixpoint is reached
        _apply_rules(new_state, registry_record, ptype)
        if logger: logger.debug(f'new_state.rules={new_state}')

        # remove facts from additional
        contract_ext_state(new_state, additional, new_predicates)
        if logger: logger.debug(f'new_state.contract={new_state}')

        return new_state

    return augment_fn

