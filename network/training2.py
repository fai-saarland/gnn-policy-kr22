import argparse
from termcolor import colored
import torch
import os
import re
import glob
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader as GraphDataLoader
from utils_old import planning
from utils_old import load_dataset as load_dataset_old
from utils_old import states_to_graphs as states_to_graphs_old
from utils_new import load_datasets as load_dataset_new
from utils_new import states_to_graphs as states_to_graphs_new
from tuning2 import load_model, load_trainer

CONFIGS = {}

BLOCKS_CLEAR_CONFIGS = {}
BLOCKS_CLEAR_CONFIGS['GCN'] = [(2, 64, 0.1, 1)]
BLOCKS_CLEAR_CONFIGS['GIN'] = [(2, 64, 0.1, 1)]
CONFIGS['blocks-clear'] = BLOCKS_CLEAR_CONFIGS


def _parse_arguments():
    parser = argparse.ArgumentParser()

    # default values for arguments
    default_batch_size = 64  # 64
    default_gpus = 0  # No GPU
    default_num_workers = 0
    default_learning_rate = 0.001
    default_weight_decay = 0.01
    default_max_samples_per_value = 100  # TODO: INCREASE THIS?
    default_max_samples_per_file = 2000
    default_max_samples = None
    default_patience = 50
    default_gradient_clip = 1
    default_profiler = None
    default_validation_frequency = 1
    default_save_top_k = 5
    default_max_epochs = None
    default_train_indices = None
    default_val_indices = None
    default_runs = 1
    default_readout = 'MAX'
    default_loss = 'MSE'

    # TODO: COMPUTE PATHS AUTOMATICALLY FROM DOMAIN NAME?
    # arguments for training
    parser.add_argument('--domain', required=True, type=str, help='domain name')
    parser.add_argument('--train', required=True, type=Path, help='path to training dataset')
    parser.add_argument('--validation', required=True, type=Path, help='path to validation dataset')
    parser.add_argument('--seeds', required=True, type=int, help='number of random seeds used for training')
    parser.add_argument('--logdir', required=True, type=Path, help='directory where policies are saved')
    parser.add_argument('--architectures', required=True, nargs='+', type=str, help='Architectures to train')

    parser.add_argument('--new_data', action='store_true', help='uses the datasets from the newer Stahlberg paper')

    # arguments for the architecture
    parser.add_argument('--aggregation', choices=['GCN', 'GCNV2', 'GAT', 'GATV2', 'GIN', 'Performer', 'Transformer', 'GCNGPS'], help=f'aggregation function')
    parser.add_argument('--readout', default=default_readout, choices=['ADD', 'MAX'], help=f'readout function')
    parser.add_argument('--loss', default=default_loss, choices=['MSE', 'MAE'], help=f'loss function')

    parser.add_argument('--num_layers', default=2, type=int, help='number of GNN layers')
    parser.add_argument('--hidden_size', default=256, type=int, help='hidden size of GNN layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='percentage of randomly deactivated neurons in each layer')
    parser.add_argument('--heads', default=1, type=int, help='number of attention heads')

    # specifying which states should be selected for training and validation sets, only needed for new data
    parser.add_argument('--train_indices', default=default_train_indices, type=str, help=f'indices of states to use for training (default={default_train_indices})')
    parser.add_argument('--val_indices', default=default_val_indices, type=str, help=f'indices of states to use for validation (default={default_val_indices})')

    # arguments with meaningful default values
    parser.add_argument('--runs', type=int, default=default_runs, help='number of planning runs per test instance')
    parser.add_argument('--max_epochs', default=default_max_epochs, type=int, help=f'maximum number of epochs (default={default_max_epochs})')
    parser.add_argument('--batch_size', default=default_batch_size, type=int, help=f'maximum size of batches (default={default_batch_size})')
    parser.add_argument('--gpus', default=default_gpus, type=int, help=f'number of GPUs to use (default={default_gpus})')
    parser.add_argument('--num_workers', default=default_num_workers, type=int, help=f'number of workers for the data loader (use 0 on Windows) (default={default_num_workers})')
    parser.add_argument('--learning_rate', default=default_learning_rate, type=float, help=f'learning rate of training session (default={default_learning_rate})')
    parser.add_argument('--weight_decay', default=default_weight_decay, type=float, help=f'strength of weight decay regularization (default={default_weight_decay})')
    parser.add_argument('--max_samples_per_value', default=default_max_samples_per_value, type=int, help=f'maximum number of states per dataset (default={default_max_samples_per_value})')
    parser.add_argument('--max_samples_per_file', default=default_max_samples_per_file, type=int, help=f'maximum number of states per instance file (default={default_max_samples_per_file})')
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

    # needed for planning during coverage validation
    default_debug_level = 0
    default_cycles = 'avoid'
    default_logfile = 'log_plan.txt'
    default_max_length = 500
    default_registry_filename = '../derived_predicates/registry_rules.json'

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
def save_results(results, architecture, policy_type, policy_path, val_loss, val_coverage, val_avg_plan_length, planning_results, num_layers, hidden_size, dropout, heads):
    results["architecture"].append(architecture)
    results["type"].append(policy_type)
    results["policy_path"].append(policy_path)
    results["instances"].append(planning_results["instances"])
    results["val_loss"].append(val_loss)
    results["val_coverage"].append(val_coverage)
    results["val_avg_plan_length"].append(val_avg_plan_length)
    results["max_coverage"].append(planning_results["max_coverage"])
    results["min_coverage"].append(planning_results["min_coverage"])
    results["avg_coverage"].append(planning_results["avg_coverage"])
    results["n_layers"].append(num_layers)
    results["h_size"].append(hidden_size)
    results["drop"].append(dropout)
    results["head"].append(heads)
    results["best_plan_quality"].append(planning_results["best_plan_quality"])
    results["plans_directory"].append(planning_results["plans_directory"])
    results.update(vars(args))
    results["architectures"] = "".join([x + "," for x in args.architectures])

def _main(args):
    # get hyperparameter configurations
    domain_configs = CONFIGS[args.domain]
    configs = []
    for architecture in args.architectures:
        config = domain_configs[architecture]
        config = [architecture] + config
        configs.append(config)

    # initialize results
    results = {
        "architecture": [],
        "type": [],
        "policy_path": [],
        "val_loss": [],
        "val_coverage": [],
        "val_avg_plan_length": [],
        "instances": [],
        "max_coverage": [],
        "min_coverage": [],
        "avg_coverage": [],
        "n_layers": [],
        "h_size": [],
        "drop": [],
        "head": [],
        "best_plan_quality": [],
        "plans_directory": [],
    }

    # load dataset
    if args.new_data:
        predicates, collate, train_dataset, validation_dataset, train_indices_selected_states, validation_indices_selected_states = load_dataset_new(args)

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

        train_graphs = states_to_graphs_new(train_dataset.get_states(), predicate_dict, predicate_ids, max_arity)
        validation_graphs = states_to_graphs_new(validation_dataset.get_states(), predicate_dict, predicate_ids, max_arity)

        problem_files = glob.glob(str('data/pddl/' + args.domain + '/validation/' + '*.pddl'))
        domain_file = Path('data/pddl/' + args.domain + '/validation/domain.pddl')
    else:
        train_dataset, predicates, decoded_predicates = load_dataset_old(args.train, args.max_samples_per_value)
        validation_dataset, _, _ = load_dataset_old(args.validation, args.max_samples_per_value)

        # assert arities are same, otherwise the orders could be different
        for i in range(len(predicates)):
            assert predicates[i][1] == decoded_predicates[i][1]

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
        train_graphs = states_to_graphs_old(train_samples, predicate_dict, predicate_ids, max_arity)
        validation_samples = [validation_dataset[i] for i in range(len(validation_dataset))]
        validation_graphs = states_to_graphs_old(validation_samples, predicate_dict, predicate_ids, max_arity)

        problem_files = glob.glob(str('data_old/pddl/' + args.domain + '/validation/' + '*.pddl'))
        domain_file = Path('data_old/pddl/' + args.domain + '/validation/domain.pddl')

    validation_instances = [instance for instance in problem_files if str(Path(instance).stem) != 'domain']

    train_loader = GraphDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                   num_workers=args.num_workers, pin_memory=True)
    validation_loader = GraphDataLoader(validation_graphs, batch_size=args.batch_size, shuffle=False,
                                        drop_last=False, num_workers=args.num_workers, pin_memory=True)

    # begin training
    args.logdir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        # set hyperparameters
        args.aggregation = config[0]
        args.num_layers = config[1][0]
        args.hidden_size = config[1][1]
        args.dropout = config[1][2]
        args.heads = config[1][3]

        config_dir = args.logdir / f"config_{args.aggregation}"
        config_dir.mkdir(parents=True, exist_ok=True)
        print("\n")
        print("CONFIG: ", config)
        print("\n")


        # TODO: STEP 1: INITIALIZE
        print(colored('Initializing datasets and loaders', 'red', attrs=['bold']))
        if not torch.cuda.is_available(): args.gpus = 0
        device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")

        train_logdir = config_dir / f"trained"
        train_logdir.mkdir(parents=True, exist_ok=True)


        # TODO: STEP 2: TRAIN
        print(colored('Training policies from scratch', 'red', attrs=['bold']))
        for _ in range(args.seeds):
            model = load_model(args, max_arity=max_arity)
            if args.new_data:
                model.enable_coverage_validation(validation_instances=validation_instances,
                                                 decoded_predicate_dict=predicate_dict,
                                                 decoded_predicate_ids=predicate_ids, max_arity=max_arity,
                                                 args=args,
                                                 domain_file=domain_file)
            else:
                model.enable_coverage_validation(validation_instances=validation_instances,
                                                 decoded_predicate_dict=decoded_predicate_dict,
                                                 decoded_predicate_ids=decoded_predicate_ids, max_arity=max_arity,
                                                 args=args,
                                                 domain_file=domain_file)
            trainer = load_trainer(args, logdir=train_logdir)
            model.set_checkpoint_path(f"{train_logdir}/version_{trainer.logger.version}/")
            print(colored('Training model...', 'green', attrs = [ 'bold' ]))
            trainer.fit(model, train_loader, validation_loader)

        # TODO: STEP 3: FIND BEST TRAINED MODEL
        print(colored('Determining best trained policy', 'red', attrs=['bold']))
        loss_validation_best_val_loss = float('inf')
        loss_validation_best_policy = None
        loss_validation_best_val_coverage = -1
        loss_validation_best_avg_plan_length = float('inf')

        coverage_validation_best_val_loss = float('inf')
        coverage_validation_best_policy = None
        coverage_validation_best_val_coverage = -1
        coverage_validation_best_avg_plan_length = float('inf')

        for version_dir in train_logdir.glob('version_*'):
            checkpoint_dir = version_dir / 'checkpoints'
            for checkpoint in checkpoint_dir.glob('*.ckpt'):
                # checkpoint of coverage validation
                if re.search("validation_loss=(.*?)-coverage=(.*?)-avg_plan_length=(.*?).ckpt", str(checkpoint)) is None:
                    val_coverage, val_avg_plan_length, val_loss = re.search("coverage=(.*?)-avg_plan_length=(.*?)-validation_loss=(.*?).ckpt", str(checkpoint)).groups()
                    val_coverage = float(val_coverage)
                    val_avg_plan_length = float(val_avg_plan_length)
                    val_loss = float(val_loss)

                    if val_coverage > coverage_validation_best_val_coverage:
                        coverage_validation_best_val_coverage = val_coverage
                        coverage_validation_best_avg_plan_length = val_avg_plan_length
                        coverage_validation_best_val_loss = val_loss
                        coverage_validation_best_policy = checkpoint
                    elif val_coverage == coverage_validation_best_val_coverage and val_avg_plan_length < coverage_validation_best_avg_plan_length:
                        coverage_validation_best_avg_plan_length = val_avg_plan_length
                        coverage_validation_best_val_loss = val_loss
                        coverage_validation_best_policy = checkpoint

                # checkpoint of loss validation
                else:
                    val_loss, val_coverage, val_avg_plan_length = re.search(
                        "validation_loss=(.*?)-coverage=(.*?)-avg_plan_length=(.*?).ckpt", str(checkpoint)).groups()
                    val_loss = float(val_loss)
                    val_coverage = float(val_coverage)
                    val_avg_plan_length = float(val_avg_plan_length)

                    if val_loss < loss_validation_best_val_loss:
                        loss_validation_best_val_loss = val_loss
                        loss_validation_best_val_coverage = val_coverage
                        loss_validation_best_avg_plan_length = val_avg_plan_length
                        loss_validation_best_policy = checkpoint

        loss_validation_best_policy_dir = train_logdir / 'best_loss_validation'
        loss_validation_best_policy_dir.mkdir(parents=True, exist_ok=True)

        # copy the best policy to the new directory
        loss_validation_best_policy_name = os.path.basename(loss_validation_best_policy)
        loss_validation_best_policy_path = os.path.join(loss_validation_best_policy_dir, loss_validation_best_policy_name)
        loss_validation_best_policy_parent_dir = loss_validation_best_policy.parent.parent
        os.system("cp " + str(loss_validation_best_policy) + " " + str(loss_validation_best_policy_path))

        # copy the losses to the new directory for later visualisation
        train_losses_path = loss_validation_best_policy_parent_dir / "losses.train"
        val_losses_path = loss_validation_best_policy_parent_dir / "losses.val"
        coverage_losses_path = loss_validation_best_policy_parent_dir / "losses.coverage"
        avg_plan_lengths_path = loss_validation_best_policy_parent_dir / "losses.avg_plan_length"
        os.system("cp " + str(train_losses_path) + " " + str(loss_validation_best_policy_dir / "losses.train"))
        os.system("cp " + str(val_losses_path) + " " + str(loss_validation_best_policy_dir / "losses.val"))
        os.system("cp " + str(coverage_losses_path) + " " + str(loss_validation_best_policy_dir / "losses.coverage"))
        os.system("cp " + str(avg_plan_lengths_path) + " " + str(loss_validation_best_policy_dir / "losses.avg_plan_length"))

        coverage_validation_best_policy_dir = train_logdir / 'best_coverage_validation'
        coverage_validation_best_policy_dir.mkdir(parents=True, exist_ok=True)

        # copy the best policy to the new directory
        coverage_validation_best_policy_name = os.path.basename(coverage_validation_best_policy)
        coverage_validation_best_policy_path = os.path.join(coverage_validation_best_policy_dir, coverage_validation_best_policy_name)
        coverage_validation_best_policy_parent_dir = coverage_validation_best_policy.parent.parent
        os.system("cp " + str(coverage_validation_best_policy) + " " + str(coverage_validation_best_policy_path))

        # copy the losses to the new directory for later visualisation
        train_losses_path = coverage_validation_best_policy_parent_dir / "losses.train"
        val_losses_path = coverage_validation_best_policy_parent_dir / "losses.val"
        coverage_losses_path = coverage_validation_best_policy_parent_dir / "losses.coverage"
        avg_plan_lengths_path = coverage_validation_best_policy_parent_dir / "losses.avg_plan_length"
        os.system("cp " + str(train_losses_path) + " " + str(coverage_validation_best_policy_dir / "losses.train"))
        os.system("cp " + str(val_losses_path) + " " + str(coverage_validation_best_policy_dir / "losses.val"))
        os.system("cp " + str(coverage_losses_path) + " " + str(coverage_validation_best_policy_dir / "losses.coverage"))
        os.system("cp " + str(avg_plan_lengths_path) + " " + str(coverage_validation_best_policy_dir / "losses.avg_plan_length"))

        # TODO: STEP 4: RUN POLICY ON TEST INSTANCES
        print(colored('Running policies on test instances', 'red', attrs=['bold']))
        policies_and_directories = []

        plans_loss_validation_path = config_dir / "plans_loss_validation"
        plans_loss_validation_path.mkdir(parents=True, exist_ok=True)
        policies_and_directories.append(("loss_validation", loss_validation_best_policy_path, plans_loss_validation_path))
        plans_coverage_validation_path = config_dir / "plans_coverage_validation"
        plans_coverage_validation_path.mkdir(parents=True, exist_ok=True)
        policies_and_directories.append(("coverage_validation", coverage_validation_best_policy_path, plans_coverage_validation_path))

        for policy_type, policy, directory in policies_and_directories:
            # load files for planning
            if args.new_data:
                domain_file = Path('data/pddl/' + args.domain + '/test/domain.pddl')
                problem_files = glob.glob(str('data/pddl/' + args.domain + '/test/' + '*.pddl'))
            else:
                domain_file = Path('data_old/pddl/' + args.domain + '/test/domain.pddl')
                problem_files = glob.glob(str('data_old/pddl/' + args.domain + '/test/' + '*.pddl'))

            # load model
            Model = load_model(args, max_arity=max_arity,  path=policy)
            try:
                model = Model.load_from_checkpoint(checkpoint_path=str(policy), strict=False).to(device)
            except:
                try:
                    model = Model.load_from_checkpoint(checkpoint_path=str(policy), strict=False,
                                                       map_location=torch.device('cuda')).to(device)
                except:
                    model = Model.load_from_checkpoint(checkpoint_path=str(policy), strict=False,
                                                       map_location=torch.device('cpu')).to(device)
            # ensure dropout is deactivated!
            model.training = False
            model.eval()
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

                    # run planning
                    if args.new_data:
                        result_string, action_trace, is_solution = planning(predicate_dict,
                                                                            predicate_ids, max_arity, args,
                                                                            policy, model, domain_file, problem_file,
                                                                            device)
                    else:
                        result_string, action_trace, is_solution = planning(decoded_predicate_dict,
                                                                            decoded_predicate_ids, max_arity, args,
                                                                            policy, model, domain_file, problem_file,
                                                                            device)

                    # store results
                    with open(log_file, "w") as f:
                        f.write(result_string)

                    is_solutions.append(is_solution)
                    if is_solution:
                        plan_lengths.append(len(action_trace))
                        print(f"Solved problem {problem_name} with plan length: {len(action_trace)}")
                    else:
                        print(f"Failed to solve problem {problem_name}")

                    # print(result_string)

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

            planning_results = dict(instances=len(problem_files)-1, max_coverage=max(coverages),
                                                 min_coverage=min(coverages), avg_coverage=sum(coverages) / len(coverages),
                                                 best_plan_quality=best_plan_quality, plans_directory=best_planning_run)

            # save results of the best run
            if policy_type == "loss_validation":
                save_results(results, architecture=args.aggregation, policy_type=policy_type, policy_path=policy, val_loss=loss_validation_best_val_loss,
                             val_coverage=loss_validation_best_val_coverage, val_avg_plan_length=loss_validation_best_avg_plan_length,
                             planning_results=planning_results, num_layers=args.num_layers, hidden_size=args.hidden_size, dropout=args.dropout, heads=args.heads)
            elif policy_type == "coverage_validation":
                save_results(results, architecture=args.aggregation, policy_type=policy_type, policy_path=policy, val_loss=coverage_validation_best_val_loss,
                             val_coverage=coverage_validation_best_val_coverage, val_avg_plan_length=coverage_validation_best_avg_plan_length,
                             planning_results=planning_results, num_layers=args.num_layers, hidden_size=args.hidden_size, dropout=args.dropout, heads=args.heads)

            print(colored('Storing results', 'red', attrs=['bold']))
            print(results)
            results_df = pd.DataFrame(results)
            results_df.to_csv(args.logdir / "results.csv")


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)
