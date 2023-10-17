import argparse
from termcolor import colored
import pytorch_lightning as pl
import torch
import os
import re
import tqdm

from architecture import set_suboptimal_factor, set_loss_constants
from helpers import ValidationLossLogging
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from datasets     import g_dataset_methods
from architecture import g_model_classes
from network.architecture import selfsupervised_suboptimal_loss_no_solvable_labels


class Oracle:
    def __init__(self, bugs, collate):
        self.bugs = bugs
        self.collate = collate
        self.counter = 0
        self.batch_size = len(self.bugs) // 1
        # split the bugs list into batches
        self.batches = [self.bugs[i:i + self.batch_size] for i in range(0, len(self.bugs), self.batch_size)]

    def get_bug_states(self):
        if self.counter == len(self.batches):
            self.counter = 0

        batch = self.batches[self.counter]
        self.counter += 1
        return batch

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
    default_save_top_k = 1
    default_loss = "selfsupervised_suboptimal"

    # TODO: COMPUTE PATHS AUTOMATICALLY FROM DOMAIN NAME
    # required arguments
    parser.add_argument('--train', required=True, type=Path, help='path to training dataset')
    parser.add_argument('--validation', required=True, type=Path, help='path to validation dataset')
    parser.add_argument('--bugs', required=True, type=Path, help='path to bug dataset')
    #parser.add_argument('--repeat', required=True, type=int, help='how often each training step is repeated')
    #parser.add_argument('--logdir', required=True, type=Path, help='directory where policies are saved')

    # arguments with meaningful default values
    parser.add_argument('--loss', default=default_loss, nargs='?',
                        choices=['supervised_optimal', 'selfsupervised_optimal', 'selfsupervised_suboptimal',
                                 'selfsupervised_suboptimal2', 'unsupervised_optimal', 'unsupervised_suboptimal',
                                 'online_optimal'])
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
    parser.add_argument('--logdir', default=None, type=str, help='folder name where logs are stored')
    parser.add_argument('--logname', default=None, type=str, help='if provided, versions are stored in folder with this name inside logdir')
    parser.add_argument('--save_top_k', default=default_save_top_k, type=int, help=f'number of top-k models to save (default={default_save_top_k})')

    # arguments for continuing the training of the original policy after re-training
    parser.add_argument('--resume', default=None, type=Path, help='path to model (.ckpt) for resuming training')
    parser.add_argument('--retrained', default=None, type=Path, help='path to the re-trained policy')

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
            #print(colored(f'Using constants {loss_constants} for loss function', 'green'), attrs = [ 'bold' ]))
            set_loss_constants(loss_constants)

def _load_datasets(args):
    print(colored('Loading datasets...', 'green', attrs = [ 'bold' ]))
    try:
        load_dataset, collate = g_dataset_methods[args.loss]
    except KeyError:
        raise NotImplementedError(f"Loss function '{args.loss}'")
    (train_dataset, predicates) = load_dataset(args.train, args.max_samples_per_file, args.max_samples, args.verify_datasets)
    (validation_dataset, _) = load_dataset(args.validation, args.max_samples_per_file, args.max_samples, args.verify_datasets)
    (bugs_dataset, _) = load_dataset(args.bugs, args.max_samples_per_file, args.max_samples, args.verify_datasets)
    loader_params = {
        "batch_size": args.batch_size,
        "drop_last": False,
        "collate_fn": collate,
        "pin_memory": True,
        "num_workers": args.num_workers,  # Use 0 on Windows, doesn't work with > 0 for some reason.
    }
    print(f'{len(predicates)} predicate(s) in dataset; predicates=[ {", ".join([ f"{name}/{arity}" for name, arity in predicates ])} ]')
    return predicates, collate, train_dataset, validation_dataset, bugs_dataset

def _load_model(args, predicates):
    print(colored('Loading model...', 'green', attrs = [ 'bold' ]))
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
        Model = g_model_classes[(args.aggregation, args.readout, args.loss)]
    except KeyError:
        raise NotImplementedError(f"No model found for {(args.aggregation, args.readout, 'base')} combination")

    print(Model)
    if args.resume is None:
        model = Model(**model_params)
    else:
        print(f"Resuming training of policy {args.resume}")
        try:
            model = Model.load_from_checkpoint(checkpoint_path=str(args.resume), strict=False)
        except:
            try:
                model = Model.load_from_checkpoint(checkpoint_path=str(args.resume), strict=False,
                                                   map_location=torch.device('cuda'))
            except:
                model = Model.load_from_checkpoint(checkpoint_path=str(args.resume), strict=False,
                                                   map_location=torch.device('cpu'))
    return model

def _load_trainer(args):
    print(colored('Initializing trainer...', 'green', attrs = [ 'bold' ]))
    callbacks = []
    if not args.verbose: callbacks.append(ValidationLossLogging())
    callbacks.append(EarlyStopping(monitor='validation_loss', patience=args.patience))
    callbacks.append(ModelCheckpoint(save_top_k=args.save_top_k, monitor='validation_loss', filename='{epoch}-{step}-{validation_loss}'))

    max_epochs = None
    # the training of the original policy will be continued for at most the number of epochs the re-trained policy was trained for
    if args.retrained is not None:
        # use regex to extract epoch from checkpoint path
        max_epochs = int(re.search("epoch=(\d)", str(args.retrained)).group(1))
        print(f"Continuing training for {max_epochs} epochs")

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

    if args.logdir or args.logname:
        logdir = args.logdir if args.logdir else 'lightning_logs'
        trainer_params['logger'] = TensorBoardLogger(logdir, name=args.logname)
    trainer = pl.Trainer(**trainer_params)
    return trainer

def _main(args):
    # TODO: STEP 1: INITIALIZE
    _process_args(args)
    predicates, collate, train_dataset, validation_dataset, bug_dataset = _load_datasets(args)

    loader_params = {
        "batch_size": args.batch_size,
        "drop_last": False,
        "collate_fn": collate,
        "pin_memory": True,
        "num_workers": args.num_workers,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_params)
    bug_loader = DataLoader(bug_dataset, shuffle=True, **loader_params)  # should shuffle be false for evaluation?


    # TODO: STEP 2: TRAIN
    #train_logdir = Path(args.logdir + "/trained/")
    train_logdir = Path(args.logdir)
    """
    for _ in range(args.repeat):
        model = _load_model(args, predicates)
        trainer = _load_trainer(logdir, args)  # TODO: LOG DIR?
        print(colored('Training model...', 'green', attrs = [ 'bold' ]))
        print(type(model).__name__)
        trainer.fit(model, train_loader, validation_loader)
    """

    # TODO: STEP 3: FIND BEST TRAINED MODEL
    best_trained_val_loss = float('inf')
    best_trained_policy = None
    for version_dir in train_logdir.glob('version_*'):
        checkpoint_dir = version_dir / 'checkpoints'
        for checkpoint in checkpoint_dir.glob('*.ckpt'):
            try:
                val_loss = float(re.search("validation_loss=(.*?).ckpt", str(checkpoint)).group(1))
            except:
                print(f"Checkpoint encoding error: {checkpoint}")
            if val_loss < best_trained_val_loss:
                best_trained_val_loss = val_loss
                best_trained_policy = checkpoint

    print(best_trained_policy)

    best_trained_policy_dir = train_logdir / 'best'
    best_trained_policy_dir.mkdir(parents=True, exist_ok=True)

    # copy the best policy to the new directory
    best_trained_policy_name = os.path.basename(best_trained_policy)
    best_trained_policy_path = os.path.join(best_trained_policy_dir, best_trained_policy_name)
    os.system("cp " + str(best_trained_policy) + " " + str(best_trained_policy_path))  # TODO: ALSO COPY LOSSES!

    # TODO: STEP 4: RE-TRAINING
    bug_states = [bug for bug in bug_dataset]
    oracle = Oracle(bug_states, collate)
    retrain_logdir = Path(args.logdir + "/retrained/")

    for _ in range(args.repeat):
        model = _load_model(args, predicates)  # TODO: LOAD DIFFERENT MODEL! define new function
        trainer = _load_trainer(args)
        checkpoint_path = f"{retrain_logdir}/version_{trainer.logger.version}/"
        model.initialize(oracle, checkpoint_path, args.update_interval, args.no_bug_loss_weight, args.no_bug_counts)

        # compute validation loss before re-training
        with torch.no_grad():
            original_validation_loss = trainer.validate(model, validation_loader)[0]['validation_loss']
            print(f'Validation loss before re-training: {original_validation_loss}')

        model.set_original_validation_loss(original_validation_loss)

        print('Training model...')
        print(type(model).__name__)

        trainer.fit(model, train_loader, validation_loader)

    # TODO: STEP 5: EVALUATE RE-TRAINING
    # iterate through retrained_dir
    checkpoint_paths = []
    for version_dir in retrain_logdir.glob('version_*'):
        checkpoint_dir = version_dir / 'checkpoints'
        for checkpoint in checkpoint_dir.glob('*.ckpt'):
            try:
                retrain_validation_loss = re.search("validation_loss=(.*?).ckpt", str(checkpoint)).group(1)
            except:
                retrain_validation_loss = "inf"
                print(f"Checkpoint encoding error: {checkpoint}")
            if retrain_validation_loss != "inf":
                checkpoint_paths.append(retrain_validation_loss)


    # get paths in checkpoint directory and discard those with validation loss = inf
    #checkpoint_paths = []
    #checkpoints_dir = Path(os.path.join(args.retrained, "checkpoints"))
    #for path in checkpoints_dir.iterdir():
    #    retrain_validation_loss = re.search("validation_loss=(.*?).ckpt", str(path)).group(1)
    #    if retrain_validation_loss != "inf":
    #        checkpoint_paths.append(path)

    print(checkpoint_paths)

    bug_losses = []
    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")  # TODO: ACCOUNT FOR MULTIPLE VERSION DIRS
    for path in checkpoint_paths:
        retrained_model = _load_model(args, predicates, path)  # TODO: USE OTHER FUNCTION?
        retrained_model.initialize(oracle, "", args.update_interval, args.no_bug_loss_weight, args.no_bug_counts)
        retrained_model.set_original_validation_loss(0)

        with torch.no_grad():
            losses = []
            for bug_batch in tqdm(bug_loader):
                labels, collated_states_with_object_counts, solvable_labels, state_counts = bug_batch

                retrained_output = retrained_model(collated_states_with_object_counts)
                retrained_loss = selfsupervised_suboptimal_loss_no_solvable_labels(retrained_output, labels,
                                                                                   state_counts, device)
                losses.append(retrained_loss.item())
            bug_losses.append(sum(losses) / len(losses))

    best_retrained_policy_index = bug_losses.index(min(bug_losses))
    best_retrained_policy = checkpoint_paths[best_retrained_policy_index]

    print(f"The best re-trained policy achieved a bug loss of {min(bug_losses)}")

    original_model = _load_model(args, predicates)  # TODO: USE SAME FUNCTION AS FOR TRAINING
    #original_model.initialize(oracle, "", args.update_interval, args.no_bug_loss_weight, args.no_bug_counts)
    #original_model.set_original_validation_loss(0)

    with torch.no_grad():
        losses = []
        for bug_batch in tqdm(bug_loader):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = bug_batch

            original_output = original_model(collated_states_with_object_counts)
            original_loss = selfsupervised_suboptimal_loss_no_solvable_labels(original_output, labels, state_counts,
                                                                              device)
            losses.append(original_loss.item())
        original_loss = sum(losses) / len(losses)

    print(f"Original model's loss on bug states: {original_loss}")

    if original_loss < bug_losses[best_retrained_policy_index]:
        print("Re-training did not improve performance on bugs!!!")
        return
    else:
        print("Re-training successful, storing best policy")
        # create a new directory for the best policy
        best_retrained_policy_dir = Path(os.path.join(retrain_logdir, "best"))
        best_retrained_policy_dir.mkdir(parents=True, exist_ok=True)

        # copy the best policy to the new directory
        best_retrained_policy_name = os.path.basename(best_retrained_policy)
        best_retrained_policy_path = os.path.join(best_retrained_policy_dir, best_retrained_policy_name)
        os.system("cp " + str(best_retrained_policy) + " " + str(best_retrained_policy_path))  # TODO: ALSO COPY LOSSES!

    # TODO: STEP 6: CONTINUE TRAINING
    max_epochs = int(re.search("epoch=(\d)", str(best_trained_policy_path)).group(1))
    print(f"Continuing training for {max_epochs} epochs")
    continue_logdir = Path(args.logdir + "/continued/")

    for _ in range(args.repeat):
        model = _load_model(args, predicates)  # TODO: USE SEPARATE FUNCTION?
        trainer = _load_trainer(continue_logdir, args)
        print(colored('Training model...', 'green', attrs=['bold']))
        print(type(model).__name__)
        trainer.fit(model, train_loader, validation_loader)

    # TODO: STEP 7: FIND BEST CONTINUED MODEL
    best_continued_val_loss = float('inf')
    best_continued_policy = None
    for version_dir in continue_logdir.glob('version_*'):
        checkpoint_dir = version_dir / 'checkpoints'
        for checkpoint in checkpoint_dir.glob('*.ckpt'):
            try:
                val_loss = float(re.search("validation_loss=(.*?).ckpt", str(checkpoint)).group(1))
            except:
                print(f"Checkpoint encoding error: {checkpoint}")
            if val_loss < best_continued_val_loss:
                best_continued_val_loss = val_loss
                best_continued_policy = checkpoint

    print(best_continued_policy)

    best_continued_policy_dir = continue_logdir / 'best'
    best_continued_policy_dir.mkdir(parents=True, exist_ok=True)

    # copy the best policy to the new directory
    best_continued_policy_name = os.path.basename(best_continued_policy)
    best_continued_policy_path = os.path.join(best_continued_policy_dir, best_continued_policy_name)
    os.system("cp " + str(best_continued_policy) + " " + str(best_continued_policy_path))  # TODO: ALSO COPY LOSSES!


    # TODO: STEP 8: EVALUATE ALL, write to pandas dataframe, here or later?
    original_model = _load_model(args, predicates, args.original)
    original_model.initialize(oracle, "", args.update_interval, args.no_bug_loss_weight, args.no_bug_counts)
    continued_model = _load_model(args, predicates, args.continued)
    continued_model.initialize(oracle, "", args.update_interval, args.no_bug_loss_weight, args.no_bug_counts)
    retrained_model = _load_model(args, predicates, args.retrained)
    retrained_model.initialize(oracle, "", args.update_interval, args.no_bug_loss_weight, args.no_bug_counts)

    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")
    with torch.no_grad():
        original_val_losses = []
        continued_val_losses = []
        retrained_val_losses = []
        for val_batch in tqdm(validation_loader):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = val_batch

            original_output = original_model(collated_states_with_object_counts)
            original_loss = selfsupervised_suboptimal_loss(original_output, labels, solvable_labels, state_counts,
                                                           device)
            original_val_losses.append(original_loss)

            continued_output = continued_model(collated_states_with_object_counts)
            continued_loss = selfsupervised_suboptimal_loss(continued_output, labels, solvable_labels, state_counts,
                                                            device)
            continued_val_losses.append(continued_loss)

            retrained_output = retrained_model(collated_states_with_object_counts)
            retrained_loss = selfsupervised_suboptimal_loss(retrained_output, labels, solvable_labels, state_counts,
                                                            device)
            retrained_val_losses.append(retrained_loss)

        print(f"Original Validation Loss: {sum(original_val_losses) / len(original_val_losses)}")
        print(f"Continued Validation Loss: {sum(continued_val_losses) / len(continued_val_losses)}")
        print(f"Retrained Validation Loss: {sum(retrained_val_losses) / len(retrained_val_losses)}")
        print("\n")

        original_bug_losses = []
        continued_bug_losses = []
        retrained_bug_losses = []
        for bug_batch in tqdm(bug_loader):
            labels, collated_states_with_object_counts, solvable_labels, state_counts = bug_batch

            original_output = original_model(collated_states_with_object_counts)
            original_loss = selfsupervised_suboptimal_loss(original_output, labels, solvable_labels, state_counts,
                                                           device)
            original_bug_losses.append(original_loss)

            continued_output = continued_model(collated_states_with_object_counts)
            continued_loss = selfsupervised_suboptimal_loss(continued_output, labels, solvable_labels, state_counts,
                                                            device)
            continued_bug_losses.append(continued_loss)

            retrained_output = retrained_model(collated_states_with_object_counts)
            retrained_loss = selfsupervised_suboptimal_loss(retrained_output, labels, solvable_labels, state_counts,
                                                            device)
            retrained_bug_losses.append(retrained_loss)

        print(f"Original Bug Loss: {sum(original_bug_losses) / len(original_bug_losses)}")
        print(f"Continued Bug Loss: {sum(continued_bug_losses) / len(continued_bug_losses)}")
        print(f"Retrained Bug Loss: {sum(retrained_bug_losses) / len(retrained_bug_losses)}")
        print("\n")

    # TODO: STEP 9: PLANNING
    def _main(args, model_path, domain_file, problem_file, logger):
        start_time = timer()

        # load model
        use_cpu = args.cpu  # hasattr(args, 'cpu') and args.cpu
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

    if __name__ == "__main__":
        # setup timer and exec name
        entry_time = timer()
        exec_path = Path(argv[0]).parent
        exec_name = Path(argv[0]).stem

        # parse arguments
        args = _parse_arguments(exec_path)

        paths_and_directories = [(args.original, "plans_original"), (args.retrained, "plans_retrained"),
                                 (args.continued, "plans_continued")]

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

    # TODO: STEP 9: WRITE EVERYTHING TO A PANDAS DATAFRAME



if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)
