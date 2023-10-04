import argparse
import time

#from termcolor import colored
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
from architecture import g_model_classes

from retrain import Oracle

from architecture.loss import selfsupervised_suboptimal_loss_no_solvable_labels, selfsupervised_suboptimal_loss

import torch
import os
from numpy import inf
from tqdm import tqdm

def _parse_arguments():
    parser = argparse.ArgumentParser()

    # default values for arguments
    default_aggregation = 'max'
    default_size = 64  # 64, 128
    default_iterations = 30  # 30, 4
    default_batch_size = 1 # 64  # TODO: HALF THIS FOR RETRAINING? MAYBE DONT
    default_gpus = 0 # No GPU
    default_num_workers = 0
    default_loss_constants = None
    default_learning_rate = 0.0002  # 0.0002
    default_suboptimal_factor = 2.0
    default_l1 = 0.0
    default_weight_decay = 0.0  # 0.0
    default_gradient_accumulation = 1
    default_max_samples_per_file = 1000  # TODO: INCREASE THIS?
    default_max_samples = None
    default_patience = 50  # TODO: REDUCE THIS From 50
    default_gradient_clip = 0.1
    default_profiler = None
    default_validation_frequency = 1
    default_save_top_k = 1
    default_loss = "selfsupervised_suboptimal"

    # required arguments or --resume that requires a path
    parser.add_argument('--trained', required=True, type=Path, help='trained policy')
    parser.add_argument('--train', required=True, type=Path, help='path to training dataset')
    parser.add_argument('--loss', nargs='?', choices=['supervised_optimal', 'selfsupervised_optimal', 'selfsupervised_suboptimal', 'selfsupervised_suboptimal2', 'unsupervised_optimal', 'unsupervised_suboptimal', 'online_optimal'], default=default_loss)

    # arguments with meaningful default values
    parser.add_argument('--aggregation', default=default_aggregation, nargs='?', choices=['retrain_add', 'retrain_max', 'retrain_addmax', 'retrain_attention', 'retrain_planformer'], help=f'readout aggregation function (default={default_aggregation})')
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
            # print(colored(f'WARNING: Invalid constants {loss_constants} for loss function, using default values', 'magenta', attrs = [ 'bold' ]))
            print(f'WARNING: Invalid constants {loss_constants} for loss function, using default values')
        else:
            #print(colored(f'Using constants {loss_constants} for loss function', 'green'), attrs = [ 'bold' ]))
            set_loss_constants(loss_constants)

def _load_datasets(args):
    # print(colored('Loading datasets...', 'green', attrs = [ 'bold' ]))
    print('Loading datasets...')
    try:
        load_dataset, collate = g_dataset_methods[args.loss]
    except KeyError:
        raise NotImplementedError(f"Loss function '{args.loss}'")
    (train_dataset, predicates) = load_dataset(args.train, args.max_samples_per_file, args.max_samples, args.verify_datasets)
    return predicates, collate,  train_dataset

def _load_model(args, predicates, path):
    # print(colored('Loading model...', 'green', attrs = [ 'bold' ]))
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
        Model = g_model_classes[(args.aggregation, args.readout, args.loss)]
    except KeyError:
        raise NotImplementedError(f"No model found for {(args.aggregation, args.readout, 'base')} combination")

    print(Model)
    model = Model.load_from_checkpoint(checkpoint_path=str(path), strict=False, map_location=torch.device('cpu'))
    return model

def _load_trainer(args):
    print('Initializing trainer...')
    callbacks = []
    callbacks.append(ValidationLossLogging())
    trainer_params = {
        "accelerator": "cpu",  # added this
        "num_sanity_val_steps": 0,
        # "progress_bar_refresh_rate": 30 if args.verbose else 0,
        "callbacks": callbacks,
        # "weights_summary": None,
        # "auto_lr_find": True,   # TODO: THIS MIGHT BE IMPORTANT
        "profiler": args.profiler,
        "accumulate_grad_batches": args.gradient_accumulation,
        "gradient_clip_val": args.gradient_clip,
        "check_val_every_n_epoch": args.validation_frequency,
    }
    trainer = pl.Trainer(**trainer_params)
    return trainer

def _main(args):
    _process_args(args)
    predicates, collate, train_dataset = _load_datasets(args)


    loader_params = {
        "batch_size": args.batch_size,
        "drop_last": False,
        "collate_fn": collate,
        "pin_memory": True,
        "num_workers": args.num_workers,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)

    trained_model = _load_model(args, predicates, args.trained)

    with torch.no_grad():
        trained_inference_speeds = []
        for train_batch in tqdm(train_loader):
            start = time.process_time()

            labels, collated_states_with_object_counts, solvable_labels, state_counts = train_batch

            trained_output = trained_model(collated_states_with_object_counts)

            trained_inference_speeds.append((time.process_time() - start))
            print(trained_inference_speeds[-1])

    # print average inference time
    print(f"Average inference time: {sum(trained_inference_speeds) / len(trained_inference_speeds)}")


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)