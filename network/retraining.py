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
from helpers import ValidationLossLogging
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from datasets     import g_dataset_methods
from architecture import g_model_classes
from architecture import g_retrain_model_classes
from architecture import selfsupervised_suboptimal_loss_no_solvable_labels, selfsupervised_suboptimal_loss
from generators import load_pddl_problem_with_augmented_states, compute_traces_with_augmented_states
from bugfile_parser import parse_bug_file


class Oracle:
    def __init__(self, collate, logdir, max_bugs_per_iteration, bugs=None):
        self.collate = collate
        self.logdir = logdir
        self.max_bugs_per_iteration = max_bugs_per_iteration
        self.bugs = bugs

    def get_bug_states(self):
        if self.bugs is None:
            raise NotImplementedError
        else:
            return self.translate_bug_states(self.bugs)

    def translate_bug_states(self, path):
        # store bug and sas files in new directory
        bug_dir = Path(self.logdir + "/" + "bugfiles")
        bug_dir.mkdir(parents=True, exist_ok=True)

        bug_files = glob.glob(str(path) + "/*.bugfile")
        translated_bugs = []
        for bug_file in bug_files:
            bug_file_name = bug_file.split("/")[-1].split(".")[0]
            # copy bugfile to directory of the currently trained policy
            os.system(f"cp {bug_file} {bug_dir}/{bug_file_name + '.bugfile'}")

            bugs, sas = parse_bug_file(bug_file)
            sas_file = Path(str(bug_dir) + "/" + bug_file_name + ".sas")
            with open(sas_file, "w") as f:
                f.write(sas)
            domain_file = Path('data/pddl/' + str(args.domain) + '/test/domain.pddl')
            problem_file = Path("data/pddl/" + str(args.domain) + "/train/" + bug_file_name + ".pddl")

            setup_args = f"--domain {domain_file} --problem {problem_file} --model {None} --sas {sas_file}"
            plan.setup_translation(setup_args)
            for bug in bugs:
                collated, encoded = plan.translate(bug.state_vals)
                label = torch.tensor([bug.cost_bound])  # TODO: TO DEVICE?
                solvable_label = torch.tensor([True] * len(encoded))  # we will not use these
                if len(encoded) == 1:
                    print("BUG HAS NO SUCCESSORS")
                    print(bug_file_name)
                    print(bug)
                    continue
                translated_bugs.append(((label, encoded, solvable_label), bug.bug_value if bug.bug_value != -1 else float('inf')))

        # sort the bugs according to their bug value
        translated_bugs.sort(key=lambda x: x[1], reverse=True)
        # only keep the first max_bugs_per_iteration bugs and remove bug value
        n = min(len(translated_bugs), self.max_bugs_per_iteration)
        translated_bugs = [x[0] for x in translated_bugs[:n]]

        return translated_bugs

def load_bugs(path):
    bug_files = glob.glob(str(path) + "/*.bugfile")
    translated_bugs = []
    for bug_file in bug_files:
        bugs, sas = parse_bug_file(bug_file)
        bug_file_name = bug_file.split("/")[-1].split(".")[0]
        sas_file = Path(str(path) + "/" + bug_file_name + ".sas")
        domain_file = Path('data/pddl/' + str(args.domain) + '/test/domain.pddl')
        problem_file = Path("data/pddl/" + str(args.domain) + "/train/" + bug_file_name + ".pddl")

        setup_args = f"--domain {domain_file} --problem {problem_file} --model {None} --sas {sas_file}"
        plan.setup_translation(setup_args)
        for bug in bugs:
            collated, encoded = plan.translate(bug.state_vals)
            label = torch.tensor([bug.cost_bound])
            solvable_label = torch.tensor([True] * len(encoded))  # we will not use these
            translated_bugs.append((label, encoded, solvable_label))

    return translated_bugs

def _parse_arguments():
    parser = argparse.ArgumentParser()

    # default values for arguments
    default_aggregation = 'max'
    default_size = 64
    default_iterations = 30
    default_batch_size = 64  # 64
    default_gpus = 0  # No GPU
    default_num_workers = 0  # TODO: increase this?
    default_loss_constants = None
    default_learning_rate = 0.0002
    default_suboptimal_factor = 2.0
    default_l1 = 0.0
    default_weight_decay = 0.0
    default_gradient_accumulation = 1
    default_max_samples_per_file = 1000  # TODO: INCREASE THIS?
    default_max_samples = None
    default_patience = 50
    default_gradient_clip = 0.1
    default_profiler = None
    default_validation_frequency = 1
    default_save_top_k = 5
    default_update_interval = -1
    default_loss = "selfsupervised_suboptimal"
    default_max_bugs_per_iteration = 7000
    default_max_epochs = None

    # TODO: COMPUTE PATHS AUTOMATICALLY FROM DOMAIN NAME?
    # required arguments
    parser.add_argument('--train', required=True, type=Path, help='path to training dataset')
    parser.add_argument('--validation', required=True, type=Path, help='path to validation dataset')
    parser.add_argument('--bugs', required=True, type=Path, help='path to bug dataset')
    parser.add_argument('--repeat', required=True, type=int, help='how often each training step is repeated')
    parser.add_argument('--logdir', required=True, type=Path, help='directory where policies are saved')

    # when using an existing trained policy
    parser.add_argument('--policy', default=None, type=Path, help='path to policy (.ckpt) for re-training')

    # turn off components of the retraining algorithm
    parser.add_argument('--no_bug_loss_weight', action='store_true', help='turn off the bug loss weight')
    parser.add_argument('--no_bug_counts', action='store_true', help='turn off the bugs counter')

    # turn off steps of the retraining pipeline
    parser.add_argument('--no_retrain', action='store_true', help='turn off the re-training of the policy')
    parser.add_argument('--no_continue', action='store_true', help='turn off the continuation of the policy\'s training')

    # arguments with meaningful default values
    parser.add_argument('--max_epochs', default=default_max_epochs, type=int, help=f'maximum number of epochs (default={default_max_epochs})')
    parser.add_argument('--max_bugs_per_iteration', default=default_max_bugs_per_iteration, type=int, help=f'maximum number of bugs per iteration (default={default_max_bugs_per_iteration})')
    parser.add_argument('--loss', default=default_loss, nargs='?',
                        choices=['supervised_optimal', 'selfsupervised_optimal', 'selfsupervised_suboptimal',
                                 'selfsupervised_suboptimal2', 'unsupervised_optimal', 'unsupervised_suboptimal',
                                 'online_optimal'])
    parser.add_argument('--update_interval', default=default_update_interval, type=int,
                        help=f'frequency at which new bugs are collected (default={default_update_interval})')
    parser.add_argument('--aggregation', default=default_aggregation, nargs='?', choices=['add', 'max', 'addmax', 'attention'], help=f'readout aggregation function (default={default_aggregation})')
    parser.add_argument('--size', default=default_size, type=int, help=f'number of features per object (default={default_size})')
    parser.add_argument('--iterations', default=default_iterations, type=int, help=f'number of convolutions (default={default_iterations})')
    parser.add_argument('--readout', action='store_true', help=f'use global readout at each iteration')
    parser.add_argument('--batch_size', default=default_batch_size, type=int, help=f'maximum size of batches (default={default_batch_size})')
    parser.add_argument('--gpus', default=default_gpus, type=int, help=f'number of GPUs to use (default={default_gpus})')
    parser.add_argument('--num_workers', default=default_num_workers, type=int, help=f'number of workers for the data loader (use 0 on Windows) (default={default_num_workers})')
    parser.add_argument('--loss_constants', default=default_loss_constants, type=str, help=f'constants (multipliers) in loss function (default={default_loss_constants})')
    parser.add_argument('--learning_rate', default=default_learning_rate, type=float, help=f'learning rate of training session (default={default_learning_rate})')
    parser.add_argument('--suboptimal_factor', default=default_suboptimal_factor, type=float, help=f'approximation factor of suboptimal learning (default={default_suboptimal_factor})')
    parser.add_argument('--l1', default=default_l1, type=float, help=f'strength of L1 regularization (default={default_l1})')
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

    # arguments for continuing the training of the original policy after re-training
    parser.add_argument('--resume', default=None, type=Path, help='path to model (.ckpt) for resuming training')
    parser.add_argument('--retrained', default=None, type=Path, help='path to the re-trained policy')

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

def load_datasets(args):
    print(colored('Loading datasets...', 'green', attrs = [ 'bold' ]))
    try:
        load_dataset, collate = g_dataset_methods[args.loss]
    except KeyError:
        raise NotImplementedError(f"Loss function '{args.loss}'")
    (train_dataset, predicates) = load_dataset(args.train, args.max_samples_per_file, args.max_samples, args.verify_datasets)
    (validation_dataset, _) = load_dataset(args.validation, args.max_samples_per_file, args.max_samples, args.verify_datasets)

    print(f'{len(predicates)} predicate(s) in dataset; predicates=[ {", ".join([ f"{name}/{arity}" for name, arity in predicates ])} ]')
    return predicates, collate, train_dataset, validation_dataset

def load_model(args, predicates, path=None, retrain=False):
    print(colored('Loading model', 'green', attrs = [ 'bold' ]))
    model_params = {
        "predicates": predicates,
        "hidden_size": args.size,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "l1_factor": args.l1,
        "gradient_accumulation": args.gradient_accumulation,
        "loss": args.loss,
        "loss_constants": args.loss_constants,
        "suboptimal_factor": args.suboptimal_factor,
        "aggregation": args.aggregation,
        "batch_size": args.batch_size,
        "max_samples_per_file": args.max_samples_per_file,
        "max_samples": args.max_samples,
        "patience": args.patience,
        "gradient_clip": args.gradient_clip,
    }

    try:
        if not retrain:
            Model = g_model_classes[(args.aggregation, args.readout, args.loss)]
        else:
            Model = g_retrain_model_classes[(args.aggregation, args.readout, args.loss)]
    except KeyError:
        raise NotImplementedError(f"No model found for {(args.aggregation, args.readout, 'base')} combination")

    if path is None:
        model = Model(**model_params)
    else:
        print(f"Loading policy {path}")
        try:
            model = Model.load_from_checkpoint(checkpoint_path=str(path), strict=False)
        except:
            try:
                model = Model.load_from_checkpoint(checkpoint_path=str(path), strict=False,
                                                   map_location=torch.device('cuda'))
            except:
                model = Model.load_from_checkpoint(checkpoint_path=str(path), strict=False,
                                                   map_location=torch.device('cpu'))

    return model


def load_trainer(args, logdir, path=None):
    print(colored('Initializing trainer', 'green', attrs = [ 'bold' ]))

    max_epochs = args.max_epochs
    patience = args.patience
    # the training of the original policy will be continued for at most the number of epochs the re-trained policy was trained for
    if path is not None:
        # use regex to extract epoch from checkpoint path, may have one, two, or three digits
        max_epochs = int(re.search(r'epoch=(\d{1,3})', str(path)).group(1)) + 1
        patience = max_epochs

        print(f"Continuing training for {max_epochs} epochs")

    callbacks = []
    if not args.verbose: callbacks.append(ValidationLossLogging())
    callbacks.append(EarlyStopping(monitor='validation_loss', patience=patience))
    callbacks.append(ModelCheckpoint(save_top_k=args.save_top_k, monitor='validation_loss',
                                     filename='{epoch}-{step}-{validation_loss}'))

    trainer_params = {
        "num_sanity_val_steps": 0,
        "callbacks": callbacks,
        "profiler": args.profiler,
        "accumulate_grad_batches": args.gradient_accumulation,
        "gradient_clip_val": args.gradient_clip,
        "check_val_every_n_epoch": args.validation_frequency,
        "max_epochs": max_epochs,
    }
    if args.gpus == 0:
        trainer_params["accelerator"] = "cpu"
    else:
        trainer_params["accelerator"] = "gpu"

    trainer_params['logger'] = TensorBoardLogger(logdir, name="")
    trainer = pl.Trainer(**trainer_params)
    return trainer

def planning(args, policy, domain_file, problem_file, logger, device):
    start_time = timer()

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
    logger.info(f"Model '{policy}' loaded in {elapsed_time:.3f} second(s)")

    logger.info(f"Loading PDDL files: domain='{domain_file}', problem='{problem_file}'")
    registry_filename = args.registry_filename if args.augment else None
    pddl_problem = load_pddl_problem_with_augmented_states(domain_file, problem_file, registry_filename,
                                                           args.registry_key, logger)
    del pddl_problem['predicates']  # Why?

    logger.info(f'Executing policy (max_length={args.max_length})')
    start_time = timer()
    is_spanner = args.spanner and 'spanner' in str(domain_file)
    unsolvable_weight = 0.0 if args.ignore_unsolvable else 100000.0
    action_trace, state_trace, value_trace, is_solution, num_evaluations = compute_traces_with_augmented_states(
        model=model, cycles=args.cycles, max_trace_length=args.max_length, unsolvable_weight=unsolvable_weight,
        logger=logger, is_spanner=is_spanner, **pddl_problem)
    elapsed_time = timer() - start_time
    logger.info(
        f'{len(action_trace)} executed action(s) and {num_evaluations} state evaluations(s) in {elapsed_time:.3f} second(s)')

    if is_solution:
        logger.info(colored(f'Found valid plan with {len(action_trace)} action(s) for {problem_file}', 'green',
                            attrs=['bold']))
    else:
        logger.info(colored(f'Failed to find a plan for {problem_file}', 'red', attrs=['bold']))

    if args.print_trace:
        for index, action in enumerate(action_trace):
            value_from = value_trace[index]
            value_to = value_trace[index + 1]
            logger.info(
                '{}: {} (value change: {:.2f} -> {:.2f} {})'.format(index + 1, action.name, float(value_from),
                                                                    float(value_to),
                                                                    'D' if float(value_from) > float(
                                                                        value_to) else 'I'))

def read_plans(path: Path):
    problems = dict()
    problems[path] = dict()

    for filename in path.glob('*.plan'):
        problem = filename.stem
        if problem not in problems[path]: problems[path][problem] = dict()
        with filename.open('r') as fd:
            plan = [ line.strip('\n') for line in fd.readlines() if line[0] == '(' ]
        problems[path][problem]['planner'] = dict(length=len(plan), plan=plan)

    for suffix in ['policy']:
        for filename in path.glob(f'*.{suffix}'):
            problem = filename.stem
            if problem not in problems[path]: problems[path][problem] = dict()
            with filename.open('r') as fd:
                plan = []
                reading_plan = False
                for line in fd.readlines():
                    line = line.strip('\n')
                    if line.find('Found valid plan') > 0:
                        reading_plan = True
                        continue
                    if reading_plan:
                        fields = line.split(' ')
                        if len(fields) >= 5 and fields[4][-1] == ':' and fields[4][:-1].isdigit():
                            plan.append(' '.join(fields[5:]))
                if reading_plan:
                    problems[path][problem][suffix] = dict(length=len(plan), plan=plan)
    return problems

def fill_table(table, problems, suffix):
    for dpath in problems:
        size = len(problems[dpath].keys())
        table['size'] = size

        solved = [ problem for problem in problems[dpath] if problem in problems[dpath] and suffix in problems[dpath][problem] ]
        opt_solved = [ problem for problem in problems[dpath] if problem in problems[dpath] and 'planner' in problems[dpath][problem] ]
        assert len(solved) <= size and len(opt_solved) <= size

        lengths = [ problems[dpath][problem][suffix]['length'] for problem in solved ]
        L = sum(lengths)

        solved_and_opt = set(solved) & set(opt_solved)
        lengths = [ problems[dpath][problem][suffix]['length'] for problem in solved_and_opt ]
        opt_lengths = [ problems[dpath][problem]['planner']['length'] for problem in solved_and_opt ]
        PL, OL, N = sum(lengths), sum(opt_lengths), len(solved_and_opt)

        table[suffix] = dict(solved=len(solved), L=L, PL=PL, OL=OL, N=N)

def save_planning_results(results, policy_type, policy_path, val_loss, bug_loss, table):
    results["type"].append(policy_type)
    results["policy_path"].append(policy_path)
    results["val_loss"].append(val_loss)
    results["bug_loss"].append(bug_loss)
    results["instances"].append(table["size"])
    results["solved"].append(table["policy"]["solved"])
    results["L"].append(table["policy"]["L"])
    PL = table["policy"]["PL"]
    results["PL"].append(PL)
    OL = table["policy"]["OL"]
    results["OL"].append(OL)
    if OL > 0:
        results["PQ"].append(PL / OL)
    else:
        results["PQ"].append(0)
    results["N"].append(table["policy"]["N"])
    results.update(vars(args))

def _main(args):
    # TODO: STEP 1: INITIALIZE
    print(colored('Initializing datasets and loaders', 'red', attrs=['bold']))
    _process_args(args)
    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")
    predicates, collate, train_dataset, validation_dataset = load_datasets(args)

    loader_params = {
        "batch_size": args.batch_size,
        "drop_last": False,
        "collate_fn": collate,
        "pin_memory": True,
        "num_workers": args.num_workers,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_params)


    # TODO: STEP 2: TRAIN
    train_logdir = args.logdir / "trained"
    train_logdir.mkdir(parents=True, exist_ok=True)

    if args.policy is None:
        print(colored('Training policies from scratch', 'red', attrs=['bold']))
        for _ in range(args.repeat):
            model = load_model(args, predicates)
            trainer = load_trainer(args, logdir=train_logdir)
            print(colored('Training model...', 'green', attrs = [ 'bold' ]))
            print(type(model).__name__)
            trainer.fit(model, train_loader, validation_loader)

        # TODO: STEP 3: FIND BEST TRAINED MODEL
        print(colored('Determining best trained policy', 'red', attrs=['bold']))
        best_trained_val_loss = float('inf')
        best_trained_policy = None
        for version_dir in train_logdir.glob('version_*'):
            checkpoint_dir = version_dir / 'checkpoints'
            for checkpoint in checkpoint_dir.glob('*.ckpt'):
                try:
                    val_loss = float(re.search("validation_loss=(.*?).ckpt", str(checkpoint)).group(1))
                except:
                    val_loss = float('inf')
                    print(f"Checkpoint encoding error: {checkpoint}")
                if val_loss < best_trained_val_loss:
                    best_trained_val_loss = val_loss
                    best_trained_policy = checkpoint

    else:
        print(colored('Using existing trained policy', 'red', attrs=['bold']))
        best_trained_policy = args.policy

        # load the policy and eval on our validation set, since another validation set might have been used during training
        model = load_model(args, predicates, path=best_trained_policy, retrain=False)
        trainer = load_trainer(args, logdir=train_logdir)
        best_trained_val_loss = trainer.validate(model, validation_loader)[0]['validation_loss']
        best_trained_policy = args.policy
        #best_trained_val_loss = float(re.search("validation_loss=(.*?).ckpt", str(best_trained_policy)).group(1))

    print(f"The best trained policy achieved a validation loss of {best_trained_val_loss}")
    best_trained_bug_loss = None

    best_trained_policy_dir = train_logdir / 'best'
    best_trained_policy_dir.mkdir(parents=True, exist_ok=True)

    # copy the best policy to the new directory
    best_trained_policy_name = os.path.basename(best_trained_policy)
    best_trained_policy_path = os.path.join(best_trained_policy_dir, best_trained_policy_name)
    os.system("cp " + str(best_trained_policy) + " " + str(best_trained_policy_path))

    # TODO: STEP 4: RE-TRAINING
    if not args.no_retrain:
        print(colored('Re-training policy', 'red', attrs=['bold']))
        retrain_logdir = args.logdir / "retrained"
        retrain_logdir.mkdir(parents=True, exist_ok=True)

        for _ in range(args.repeat):
            model = load_model(args, predicates, path=best_trained_policy_path, retrain=True)
            trainer = load_trainer(args, logdir=retrain_logdir)
            checkpoint_path = f"{retrain_logdir}/version_{trainer.logger.version}/"
            oracle = Oracle(bugs=args.bugs, collate=collate, logdir=checkpoint_path, max_bugs_per_iteration=args.max_bugs_per_iteration)
            model.initialize(oracle, checkpoint_path, args.update_interval, args.no_bug_loss_weight, args.no_bug_counts)

            print('Re-training model...')
            print(type(model).__name__)

            trainer.fit(model, train_loader, validation_loader)

        # TODO: STEP 5: EVALUATE RE-TRAINING
        print(colored('Determining best re-trained policy', 'red', attrs=['bold']))
        best_retrained_val_loss = float('inf')
        best_retrained_policy = None
        successful_retrained_policies = []
        for version_dir in retrain_logdir.glob('version_*'):
            checkpoint_dir = version_dir / 'checkpoints'
            for checkpoint in checkpoint_dir.glob('*.ckpt'):
                retrained_val_loss = float(re.search("validation_loss=(.*?).ckpt", str(checkpoint)).group(1))
                if retrained_val_loss < best_retrained_val_loss:
                    best_retrained_val_loss = retrained_val_loss
                    best_retrained_policy = checkpoint

                if retrained_val_loss <= (best_trained_val_loss + 0.1 * best_trained_val_loss):
                    successful_retrained_policies.append(checkpoint)

        #if len(successful_retrained_policies) == 0:
        #    policy_paths = [best_retrained_policy]
        #    print("None of the re-trained policies achieved validation losses comparable to the trained policy!")
        #else:
        #    policy_paths = successful_retrained_policies
        #    print(f"{len(successful_retrained_policies)} re-trained policies achieved validation losses comparable to the trained policy!")

        policy_paths = [best_retrained_policy]  # for now, we only care about the retrained policy with the lowest validation loss

        # evaluate how much the best re-trained policies improved performance on bug states
        trained_model = load_model(args, predicates, path=best_trained_policy_path)
        device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")
        max_delta = float('-inf')
        best_retrained_policy = None
        for path in policy_paths:
            retrained_model = load_model(args, predicates, path=path, retrain=True)
            bug_path = path.parent.parent / "bugfiles"
            bugs = load_bugs(bug_path)
            retrained_bug_losses = []
            trained_bug_losses = []
            with torch.no_grad():
                for bug in bugs:
                    try:
                        labels, collated_states_with_object_counts, solvable_labels, state_counts = collate([bug])

                        retrained_output = retrained_model(collated_states_with_object_counts)
                        retrained_loss = selfsupervised_suboptimal_loss_no_solvable_labels(retrained_output, labels,
                                                                                           state_counts, device)
                        retrained_bug_losses.append(retrained_loss.item())

                        trained_output = trained_model(collated_states_with_object_counts)
                        trained_loss = selfsupervised_suboptimal_loss_no_solvable_labels(trained_output, labels, state_counts,
                                                                                         device)
                        trained_bug_losses.append(trained_loss.item())
                    except:
                        print(f"Error processing bug {bug}!")
                        continue

                avg_retrained_bug_loss = sum(retrained_bug_losses) / len(retrained_bug_losses)
                avg_trained_bug_loss = sum(trained_bug_losses) / len(trained_bug_losses)
                delta = avg_trained_bug_loss - avg_retrained_bug_loss
                if delta > max_delta:
                    max_delta = delta
                    best_retrained_policy = path
                    best_retrained_bug_loss = avg_retrained_bug_loss
                    best_trained_bug_loss = avg_trained_bug_loss

        best_retrained_val_loss = float(re.search("validation_loss=(.*?).ckpt", str(best_retrained_policy)).group(1))

        print(f"The best re-trained policy achieved a validation loss of {best_retrained_val_loss}, and a bug loss of {best_retrained_bug_loss} corresponding to an improvement over the trained model of {max_delta}")
        if max_delta <= 0:
            print("Re-training did not improve performance on bugs!!!")
        else:
            print("Re-training was successful")

        # create a new directory for the best policy
        best_retrained_policy_dir = Path(os.path.join(retrain_logdir, "best"))
        best_retrained_policy_dir.mkdir(parents=True, exist_ok=True)

        # copy the best policy to the new directory
        best_retrained_policy_name = os.path.basename(best_retrained_policy)
        best_retrained_policy_path = os.path.join(best_retrained_policy_dir, best_retrained_policy_name)
        os.system("cp " + str(best_retrained_policy) + " " + str(best_retrained_policy_path))
        os.system("cp -r " + str(best_retrained_policy.parent.parent / "bugfiles") + " " + str(best_retrained_policy_dir))

    # TODO: STEP 6: CONTINUE TRAINING
    if (not args.no_continue) and (not args.no_retrain):
        print(colored('Continuing training of trained policy for same number of epochs as best re-trained policy', 'red', attrs=['bold']))
        continue_logdir = args.logdir / "continued"
        continue_logdir.mkdir(parents=True, exist_ok=True)

        for _ in range(args.repeat):
            model = load_model(args, predicates, path=best_trained_policy_path, retrain=False)
            trainer = load_trainer(args, logdir=continue_logdir, path=best_retrained_policy_path)
            print(colored('Continuing training of model...', 'green', attrs=['bold']))
            print(type(model).__name__)
            trainer.fit(model, train_loader, validation_loader)

        # TODO: STEP 7: FIND BEST CONTINUED POLICY
        print(colored('Determining best continued policy', 'red', attrs=['bold']))
        best_continued_val_loss = float('inf')
        best_continued_policy = None
        for version_dir in continue_logdir.glob('version_*'):
            checkpoint_dir = version_dir / 'checkpoints'
            for checkpoint in checkpoint_dir.glob('*.ckpt'):
                try:
                    val_loss = float(re.search("validation_loss=(.*?).ckpt", str(checkpoint)).group(1))
                except:
                    print(f"Checkpoint encoding error: {checkpoint}")
                    val_loss = float('inf')
                if val_loss < best_continued_val_loss:
                    best_continued_val_loss = val_loss
                    best_continued_policy = checkpoint

        print(f"The best continued policy achieved a validation loss of {best_continued_val_loss}")

        best_continued_policy_dir = continue_logdir / 'best'
        best_continued_policy_dir.mkdir(parents=True, exist_ok=True)

        # copy the best policy to the new directory
        best_continued_policy_name = os.path.basename(best_continued_policy)
        best_continued_policy_path = os.path.join(best_continued_policy_dir, best_continued_policy_name)
        os.system("cp " + str(best_continued_policy) + " " + str(best_continued_policy_path))


        # TODO: STEP 8: EVALUATE CONTINUED MODEL ON BUGS
        print(colored('Evaluating performance of continued policy bug dataset', 'red', attrs=['bold']))
        continued_model = load_model(args, predicates, path=best_continued_policy_path, retrain=False)

        bug_path = Path(best_retrained_policy_path).parent / "bugfiles"
        bugs = load_bugs(bug_path)
        with torch.no_grad():
            continued_bug_losses = []
            for bug in bugs:
                try:
                    labels, collated_states_with_object_counts, solvable_labels, state_counts = collate([bug])

                    continued_output = continued_model(collated_states_with_object_counts)
                    continued_loss = selfsupervised_suboptimal_loss_no_solvable_labels(continued_output, labels, state_counts,
                                                                    device)
                    continued_bug_losses.append(continued_loss.item())
                except:
                    print(f"Error processing bug")
                    print(bug)
                    continue

            continued_bug_loss = sum(continued_bug_losses) / len(continued_bug_losses)
            print(f"Continued model's loss on bug states: {continued_bug_loss}")
            print("\n")

    # TODO: STEP 9: PLANNING
    print(colored('Running policies on test instances', 'red', attrs=['bold']))
    # setup timer and exec name
    entry_time = timer()
    exec_path = Path(argv[0]).parent  # TODO: WHAT IS THIS?
    exec_name = Path(argv[0]).stem

    policies_and_directories = []

    plans_trained_path = args.logdir / "plans_trained"
    plans_trained_path.mkdir(parents=True, exist_ok=True)
    policies_and_directories.append(("trained", best_trained_policy_path, plans_trained_path))

    if not args.no_retrain:
        plans_retrained_path = args.logdir / "plans_retrained"
        plans_retrained_path.mkdir(parents=True, exist_ok=True)
        policies_and_directories.append(("retrained", best_retrained_policy_path, plans_retrained_path))

    if (not args.no_continue) and (not args.no_retrain):
        plans_continued_path = args.logdir / "plans_continued"
        plans_continued_path.mkdir(parents=True, exist_ok=True)
        policies_and_directories.append(("continued", best_continued_policy_path, plans_continued_path))

    results = {
        "type": [],
        "policy_path": [],
        "val_loss": [],
        "bug_loss": [],
        "instances": [],
        "solved": [],
        "L": [],
        "PQ": [],
        "PL": [],
        "OL": [],
        "N": []
    }
    for policy_type, policy, directory in policies_and_directories:
        domain_file = Path('data/pddl/' + args.domain + '/test/domain.pddl')
        problem_files = glob.glob(str('data/pddl/' + args.domain + '/test/' + '*.pddl'))
        for i in range(args.repeat):
            version_path = directory / f"version_{i}"
            version_path.mkdir(parents=True, exist_ok=True)
            os.system("cp " + "optimal_plans/" + str(args.domain) + "/*.plan " + str(version_path))
            for problem_file in problem_files:
                problem_name = str(Path(problem_file).stem)
                if problem_name == 'domain':
                    continue

                # setup logger
                if args.cycles == 'detect':  # TODO: IS THIS CORRECT?
                    logfile_name = problem_name + ".markovian"
                elif args.cycles == "avoid":
                    logfile_name = problem_name + ".policy"

                log_file = version_path / logfile_name

                log_level = logging.INFO if args.debug_level == 0 else logging.DEBUG
                logger = plan._get_logger(exec_name + problem_name, log_file, log_level)
                logger.info(f'Call: {" ".join(argv)}')

                # do jobs
                planning(args, policy, domain_file, problem_file, logger, device)

                # final stats
                elapsed_time = timer() - entry_time
                logger.info(f'All tasks completed in {elapsed_time:.3f} second(s)')

                del logger

        best_solved = 0
        best_plan_quality = float('inf')
        best_table = None
        for version_dir in directory.glob('version_*'):
            plans = read_plans(version_dir)
            table = dict()
            fill_table(table, plans, 'policy')
            solved = table['policy']['solved']
            try:
                plan_quality = table['policy']['PL'] / table['policy']['OL']
            except:
                plan_quality = 0

            if (solved > best_solved) or (solved == best_solved and plan_quality < best_plan_quality):
                best_solved = solved
                best_plan_quality = plan_quality
                best_table = table
                best_version_dir = version_dir

        best_plans_path = directory / "best"
        best_plans_path.mkdir(parents=True, exist_ok=True)
        os.system("cp " + str(best_version_dir) + "/*.plan " + str(best_plans_path))
        os.system("cp " + str(best_version_dir) + "/*.policy " + str(best_plans_path))

        if policy_type == "trained":
            save_planning_results(results, "trained", best_trained_policy_path, best_trained_val_loss,
                                  best_trained_bug_loss, best_table)
        elif policy_type == "retrained":
            save_planning_results(results, "retrained", best_retrained_policy_path, best_retrained_val_loss,
                                  best_retrained_bug_loss, best_table)
        elif policy_type == "continued":
            save_planning_results(results, "continued", best_continued_policy_path, best_continued_val_loss,
                                  continued_bug_loss, best_table)

    print(colored('Storing results', 'red', attrs=['bold']))

    results = pd.DataFrame(results)
    results.to_csv(args.logdir / "results.csv")
    print(results)


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)
