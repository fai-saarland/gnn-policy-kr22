import argparse
from termcolor import colored
import torch
import glob
import pandas as pd
from pathlib import Path
from training_new import load_model, planning, load_datasets

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

    # when using an existing trained policy
    parser.add_argument('--policy', default=None, type=Path, help='path to policy (.ckpt) for testing')
    parser.add_argument('--logdir', required=True, type=Path, help='directory where plans are saved')
    parser.add_argument('--train', required=True, type=Path, help='path to training dataset')
    parser.add_argument('--validation', required=True, type=Path, help='path to validation dataset')

    # arguments for the architecture
    parser.add_argument('--aggregation', required=True, choices=['GCN', 'GCNV2', 'GAT', 'GATV2', 'GIN'],
                        help=f'aggregation function')
    parser.add_argument('--readout', required=True, choices=['ADD', 'MAX'], help=f'readout function')
    parser.add_argument('--loss', required=True, choices=['MSE'], help=f'loss function')

    parser.add_argument('--num_layers', default=2, type=int, help='number of GNN layers')
    parser.add_argument('--hidden_size', default=256, type=int, help='hidden size of GNN layers')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='percentage of randomly deactivated neurons in each layer')
    parser.add_argument('--heads', default=1, type=int, help='number of attention heads')

    parser.add_argument('--train_indices', default=None, type=str, help=f'indices of states to use for training')
    parser.add_argument('--val_indices', default=None, type=str, help=f'indices of states to use for validation')

    # arguments with meaningful default values
    parser.add_argument('--runs', type=int, default=1, help='number of planning runs per instance')
    parser.add_argument('--max_epochs', default=default_max_epochs, type=int,
                        help=f'maximum number of epochs (default={default_max_epochs})')
    # parser.add_argument('--max_bugs_per_iteration', default=default_max_bugs_per_iteration, type=int, help=f'maximum number of bugs per iteration (default={default_max_bugs_per_iteration})')
    parser.add_argument('--batch_size', default=default_batch_size, type=int,
                        help=f'maximum size of batches (default={default_batch_size})')
    parser.add_argument('--gpus', default=default_gpus, type=int,
                        help=f'number of GPUs to use (default={default_gpus})')
    parser.add_argument('--num_workers', default=default_num_workers, type=int,
                        help=f'number of workers for the data loader (use 0 on Windows) (default={default_num_workers})')
    parser.add_argument('--learning_rate', default=default_learning_rate, type=float,
                        help=f'learning rate of training session (default={default_learning_rate})')
    parser.add_argument('--weight_decay', default=default_weight_decay, type=float,
                        help=f'strength of weight decay regularization (default={default_weight_decay})')
    parser.add_argument('--gradient_accumulation', default=default_gradient_accumulation, type=int,
                        help=f'number of gradients to accumulate before step (default={default_gradient_accumulation})')
    parser.add_argument('--max_samples_per_file', default=default_max_samples_per_file, type=int,
                        help=f'maximum number of states per dataset (default={default_max_samples_per_file})')
    parser.add_argument('--max_samples', default=default_max_samples, type=int,
                        help=f'maximum number of states in total (default={default_max_samples})')
    parser.add_argument('--patience', default=default_patience, type=int,
                        help=f'patience for early stopping (default={default_patience})')
    parser.add_argument('--gradient_clip', default=default_gradient_clip, type=float,
                        help=f'gradient clip value (default={default_gradient_clip})')
    parser.add_argument('--profiler', default=default_profiler, type=str,
                        help=f'"simple", "advanced" or "pytorch" (default={default_profiler})')
    parser.add_argument('--validation_frequency', default=default_validation_frequency, type=int,
                        help=f'evaluate on validation set after this many epochs (default={default_validation_frequency})')
    parser.add_argument('--verbose', action='store_true', help='print additional information during training')
    parser.add_argument('--verify_datasets', action='store_true', help='verify state labels are as expected')

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
def save_results(results, policy_type, policy_path, planning_results):
    results["type"].append(policy_type)
    results["policy_path"].append(policy_path)
    results["instances"].append(planning_results["instances"])
    results["max_coverage"].append(planning_results["max_coverage"])
    results["min_coverage"].append(planning_results["min_coverage"])
    results["avg_coverage"].append(planning_results["avg_coverage"])
    results["best_plan_quality"].append(planning_results["best_plan_quality"])
    results["plans_directory"].append(planning_results["plans_directory"])
    results.update(vars(args))


def _main(args):
    # TODO: STEP 1: INITIALIZE
    print(colored('Initializing datasets and loaders', 'red', attrs=['bold']))
    args.logdir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available(): args.gpus = 0
    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")

    predicates, collate, train_dataset, validation_dataset, train_indices_selected_states, validation_indices_selected_states = load_datasets(
        args)

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

    # TODO: STEP 3: PLANNING
    print(colored('Running policies on test instances', 'red', attrs=['bold']))
    policies_and_directories = []

    plans_trained_path = args.logdir / "plans"
    plans_trained_path.mkdir(parents=True, exist_ok=True)
    policies_and_directories.append(("trained", args.policy, plans_trained_path))

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
        domain_file = Path('data/pddl/' + args.domain + '/test/domain.pddl')
        problem_files = glob.glob(str('data/pddl/' + args.domain + '/test/' + '*.pddl'))

        # load model
        Model = load_model(args, policy)
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
                result_string, action_trace, is_solution = planning(predicate_dict, predicate_ids, max_arity, args, policy, model, domain_file, problem_file, device)

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
        save_results(results, policy_type, args.policy, planning_results)



    print(colored('Storing results', 'red', attrs=['bold']))
    results = pd.DataFrame(results)
    results.to_csv(args.logdir / "results.csv")
    print(results)


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)
