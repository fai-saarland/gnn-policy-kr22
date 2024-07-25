import argparse
from termcolor import colored
import torch
import random
import pandas as pd
from pathlib import Path
from utils_old import planning_old
from utils_old import load_dataset

from gnns import create_GNN
from gnns import GraphConvolutionNetwork, GraphConvolutionNetworkV2, GraphAttentionNetwork, GraphAttentionNetworkV2, GraphIsomorphismNetwork
from gnns import Performer, GCNGPS
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

    ("Transformer", "ADD", "MSE"): create_GNN(Performer, global_add_pool, mse_loss),
    ("GCNGPS", "ADD", "MSE"): create_GNN(GCNGPS, global_add_pool, mse_loss),
    ("GCNGPS", "ADD", "MAE"): create_GNN(GCNGPS, global_add_pool, mae_loss),

    ("GCNGPS", "MAX", "MSE"): create_GNN(GCNGPS, global_add_pool, mse_loss),
}

def load_predicates(args):
    print(colored('Loading datasets...', 'green', attrs = [ 'bold' ]))

    if args.domain == "gripper":
        dataset = Path('data_old/supervised/optimal/validation/gripper-atomic/gripper-atomic/')
    elif args.domain == "blocks-clear":
        dataset = Path('data_old/supervised/optimal/validation/blocks-clear/blocks-clear/')
    elif args.domain == "visitall":
        dataset = Path('data_old/supervised/optimal/validation/visitall-atomic/visitall-atomic/')
    elif args.domain == "parking-behind":
        dataset = Path('data_old/supervised/optimal/validation/parking-behind/parking-behind/')
    elif args.domain == "satellite":
        dataset = Path('data_old/supervised/optimal/validation/satellite/satellite/')

    decoded_predicates = load_dataset(dataset, 1)[2]

    i = 0
    decoded_predicate_dict = {}
    decoded_predicate_ids = {}
    max_arity = 0
    for predicate, arity in decoded_predicates:
        decoded_predicate_dict[predicate] = arity
        if arity > max_arity:
            max_arity = arity
        decoded_predicate_ids[predicate] = i
        i += 1

    return decoded_predicate_dict, decoded_predicate_ids, max_arity

def _parse_arguments():
    parser = argparse.ArgumentParser()

    # default values for arguments
    default_gpus = 0  # No GPU

    # required arguments
    parser.add_argument('--policy', required=True, type=Path, help='path to policy (.ckpt)')
    parser.add_argument('--type', required=True, type=str, help='type of network')
    parser.add_argument('--logdir', required=True, type=Path, help='directory where policies are saved')
    parser.add_argument('--min_size', required=True, type=int, help='minimum size of the generated instances')
    parser.add_argument('--max_size', required=True, type=int, help='maximum size of the generated instances')

    # arguments with meaningful default values
    parser.add_argument('--gpus', default=default_gpus, type=int, help=f'number of GPUs to use (default={default_gpus})')

    default_debug_level = 0
    default_cycles = 'avoid'
    default_logfile = 'log_plan.txt'
    default_max_length = 500  # TODO: CHOOSE DIFFERENT MAX LENGTH OR USE A TIME LIMIT?
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

# activate the dropout
def enable_dropout(mod: torch.nn.Module):
    if isinstance(mod, torch.nn.Dropout):
        mod.train()

def generate_instance_gripper_atomic(num_balls):
    instance = ""
    instance += f"(define (problem gripper-{num_balls})"
    instance += f"\n(:domain gripper-strips)"
    instance += f"\n(:objects "
    instance += f"rooma roomb "
    for i in reversed(range(num_balls)):
        instance += f"ball{i+1} "
    instance += " left right)"
    instance += f"\n(:init"
    instance += f"\n(room rooma)"
    instance += f"\n(room roomb)"
    for i in reversed(range(num_balls)):
        instance += f"\n(ball ball{i+1})"
    instance += f"\n(at-robby rooma)"
    instance += f"\n(free left)"
    instance += f"\n(free right)"
    # place all balls in room a
    for i in reversed(range(num_balls)):
        instance += f"\n(at ball{i+1} rooma)"
    instance += f"\n(gripper left)"
    instance += f"\n(gripper right)"
    instance += f"\n)"
    instance += f"\n(:goal"
    instance += f"\n(and"
    # randomly sample one of the balls from room a and place it in room b
    ball_to_move = random.choice(range(num_balls))
    #ball_to_move = list(range(num_balls))[-1]
    instance += f"\n(at ball{ball_to_move+1} roomb)"
    instance += f"\n)"
    instance += f"\n)"
    instance += f"\n)"
    return instance

def _main(args):
    device = torch.device("cuda") if args.gpus > 0 else torch.device("cpu")


    # load model
    decoded_predicate_dict, decoded_predicate_ids, max_arity = load_predicates(args)
    # load model
    Model = model_classes[(args.type, "ADD", "MSE")]  # TODO: USE MAX AGGREGATION?
    try:
        model = Model.load_from_checkpoint(checkpoint_path=str(args.policy), strict=False).to(device)
    except:
        try:
            model = Model.load_from_checkpoint(checkpoint_path=str(args.policy), strict=False,
                                               map_location=torch.device('cuda')).to(device)
        except:
            model = Model.load_from_checkpoint(checkpoint_path=str(args.policy), strict=False,
                                               map_location=torch.device('cpu')).to(device)
    model = model.to(device)
    # TODO: KEEP DROPOUT ACTIVATED FOR PROBABILISTIC POLICIES?
    model.training = False
    model.eval()
    #model.apply(enable_dropout)

    domain_file = Path('data_old/pddl/' + args.domain + '/test/domain.pddl')

    # TODO: TRACK AVERAGE PLAN LENGTH?
    results = {
        "size": [],
        "avg_coverage": [],
        "avg_plan_length": []
    }
    for size in range(args.min_size, args.max_size + 1):
        results["size"].append(size)
        coverages = []
        plan_lengths = []
        # create directory for storing generated instance files
        instance_directory = args.logdir / f"generated_instances/size_{size}"
        instance_directory.mkdir(parents=True, exist_ok=True)

        num_instances_per_size = 1
        num_runs_per_instance = 1
        for i in range(num_instances_per_size):
            instance_string = generate_instance_gripper_atomic(size)
            instance_file = instance_directory / f"instance_size_{size}_num_{i}.pddl"
            with open(instance_file, 'w') as f:
                f.write(instance_string)

            for _ in range(num_runs_per_instance):
                result_string, action_trace, is_solution = planning_old(decoded_predicate_dict, decoded_predicate_ids,
                                                                        max_arity, args, args.policy, model, domain_file,
                                                                        instance_file, device)
                if is_solution:
                    coverages.append(1)
                    plan_lengths.append(len(action_trace))
                    print(colored(f"Instance size {size} num {i} solved after {len(action_trace)} steps!", 'green', attrs=['bold']))
                else:
                    coverages.append(0)
                    print(colored(f"Instance size {size} num {i} not solved!", 'red', attrs=['bold']))

        avg_coverage = sum(coverages) / len(coverages)
        if len(plan_lengths) == 0:
            avg_plan_length = 0
        else:
            avg_plan_length = sum(plan_lengths) / len(plan_lengths)
        results[f"avg_coverage"].append(avg_coverage)
        results[f"avg_plan_length"].append(avg_plan_length)

        results_data_frame = pd.DataFrame(results)
        results_data_frame.to_csv(args.logdir / "results.csv")


if __name__ == "__main__":
    args = _parse_arguments()
    _main(args)
