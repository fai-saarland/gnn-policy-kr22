import argparse
import pytorch_lightning as pl
import torch

from architecture import set_suboptimal_factor, set_loss_constants
from helpers import ValidationLossLogging
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from datasets     import g_dataset_methods
from architecture import g_retrain_model_classes

from retrain import Oracle

from architecture.loss import selfsupervised_suboptimal_loss_no_solvable_labels, selfsupervised_suboptimal_loss

import torch
import os
from numpy import inf
from tqdm import tqdm

import re

def _parse_arguments():
    parser = argparse.ArgumentParser()

    # default values for arguments
    default_aggregation = 'retrain_max'
    default_size = 64
    default_iterations = 30
    default_batch_size = 64
    default_gpus = 0  # No GPU
    default_num_workers = 0
    default_loss_constants = None
    default_learning_rate = 0.0002
    default_suboptimal_factor = 2.0
    default_l1 = 0.0
    default_weight_decay = 0.0
    default_gradient_accumulation = 1
    default_max_samples_per_file = 1000
    default_max_samples = None
    default_patience = 50
    default_gradient_clip = 0.1
    default_profiler = None
    default_validation_frequency = 1
    default_save_top_k = 1
    default_update_interval = -1
    default_loss = "selfsupervised_suboptimal"

    # required arguments
    parser.add_argument('--original', required=True, type=Path,
                        help='path to the original policy that was used for re-training')
    parser.add_argument('--retrained', required=True, type=Path, help='path to directory containing re-trained policies')
    parser.add_argument('--bugs', required=True, type=Path, help='path to bugs dataset')

    # turn off components of the retraining algorithm
    parser.add_argument('--no_bug_loss_weight', action='store_false', help='turn off the bug loss weight')
    parser.add_argument('--no_bug_counts', action='store_false', help='turn off the bugs counter')

    # arguments with meaningful default values
    parser.add_argument('--loss', default=default_loss, nargs='?',
                        choices=['supervised_optimal', 'selfsupervised_optimal', 'selfsupervised_suboptimal',
                                 'selfsupervised_suboptimal2', 'unsupervised_optimal', 'unsupervised_suboptimal',
                                 'online_optimal'])
    parser.add_argument('--update_interval', default=default_update_interval, type=int, help=f'frequency at which new bugs are collected (default={default_update_interval})')
    parser.add_argument('--aggregation', default=default_aggregation, nargs='?', choices=['retrain_add', 'retrain_max', 'retrain_addmax', 'retrain_attention'], help=f'readout aggregation function (default={default_aggregation})')
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
            print(f'WARNING: Invalid constants {loss_constants} for loss function, using default values')
        else:
            set_loss_constants(loss_constants)

def _load_datasets(args):
    print('Loading datasets...')
    try:
        load_dataset, collate = g_dataset_methods[args.loss]
    except KeyError:
        raise NotImplementedError(f"Loss function '{args.loss}'")
    (bugs_dataset, predicates) = load_dataset(args.bugs, args.max_samples_per_file, args.max_samples, args.verify_datasets)
    return predicates, collate, bugs_dataset

def _load_model(args, predicates, path):
    print('Loading model...')
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
        Model = g_retrain_model_classes[(args.aggregation, args.readout, args.loss)]
    except KeyError:
        raise NotImplementedError(f"No model found for {(args.aggregation, args.readout, 'base')} combination")

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

def _load_trainer(args):
    print('Initializing trainer...')
    callbacks = []
    if not args.verbose: callbacks.append(ValidationLossLogging())

    trainer_params = {
        "num_sanity_val_steps": 0,
        "callbacks": callbacks,
        "profiler": args.profiler,
        "accumulate_grad_batches": args.gradient_accumulation,
        "gradient_clip_val": args.gradient_clip,
        "check_val_every_n_epoch": args.validation_frequency,
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
    _process_args(args)
    predicates, collate, bugs_dataset = _load_datasets(args)

    bug_states = [bug for bug in bugs_dataset]
    oracle = Oracle(bug_states, collate)

    loader_params = {
        "batch_size": args.batch_size,
        "drop_last": False,
        "collate_fn": collate,
        "pin_memory": True,
        "num_workers": args.num_workers,
    }
    bug_loader = DataLoader(bugs_dataset, shuffle=False, **loader_params)

    # get paths in checkpoint directory and discard those with validation loss = inf
    checkpoint_paths = []
    checkpoints_dir = Path(os.path.join(args.retrained, "checkpoints"))
    for path in checkpoints_dir.iterdir():
        retrain_validation_loss = re.search("validation_loss=(.*?).ckpt", str(path)).group(1)
        if retrain_validation_loss != "inf":
            checkpoint_paths.append(path)

    print(checkpoint_paths)

    bug_losses = []
    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")
    for path in checkpoint_paths:
        retrained_model = _load_model(args, predicates, path)
        retrained_model.initialize(oracle, "", args.update_interval, args.no_bug_loss_weight, args.no_bug_counts)
        retrained_model.set_original_validation_loss(0)

        with torch.no_grad():
            losses = []
            for bug_batch in tqdm(bug_loader):
                labels, collated_states_with_object_counts, solvable_labels, state_counts = bug_batch

                retrained_output = retrained_model(collated_states_with_object_counts)
                retrained_loss = selfsupervised_suboptimal_loss_no_solvable_labels(retrained_output, labels, state_counts, device)
                losses.append(retrained_loss.item())
            bug_losses.append(sum(losses) / len(losses))

    best_policy_index = bug_losses.index(min(bug_losses))
    best_policy = checkpoint_paths[best_policy_index]

    print(f"The best re-trained policy achieved a bug loss of {min(bug_losses)}")

    original_model = _load_model(args, predicates, args.original)
    original_model.initialize(oracle, "", args.update_interval, args.no_bug_loss_weight, args.no_bug_counts)
    original_model.set_original_validation_loss(0)

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

    if original_loss < bug_losses[best_policy_index]:
        print("Re-training did not improve performance on bugs!!!")
        return
    else:
        print("Re-training successful, storing best policy")
        # create a new directory for the best policy
        best_policy_dir = Path(os.path.join(args.retrained, "best"))
        best_policy_dir.mkdir(parents=True, exist_ok=True)

        # copy the best policy to the new directory
        best_policy_name = os.path.basename(best_policy)
        best_policy_path = os.path.join(best_policy_dir, best_policy_name)
        os.system("cp " + str(best_policy) + " " + str(best_policy_path))


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)


