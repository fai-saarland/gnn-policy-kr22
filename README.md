# Documentation

## bugfiles / bugfiles2
Contains the bugs of the policy for the reward domain (our_models/reward_old or our_models/reward).

## data
Contains the data sets of all domains and the corresponding pddl files.

## derived_predicates
Contains code for augmenting states such that the GNN becomes more expressive. Not relevant for our purposes.

## models
Contains the original policies trained by Stahlberg.

## new_docker
Contains a Dockerfile that Stahlberg apparently used.

## optimal_plans
Contains the optimal plans for all domains.

## our_models
Contains the policies we trained.

## results
Contains the plans that Stahlberg used to generate the table in his paper.

## server_scripts 
Contains some of the scripts that I used to run experiments on the GPU cluster. The .sub files describe the required 
resources of the job, and the .sh files describe which scripts are executed in the job.

## spanner-bidirectional
Contains some code that Stahlberg used to modify the Spanner domain.

## network
This is the directory that contains all the important code. The most important scripts are:
- `training.py`: Trains policies. For example, to train policies for the reward domain by sampling 3 data sets (rounds),
repeating the optimization 2 times for each (seeds), and then evaluating the policy with the best validation set by running it on
each test instance 10 times (runs), run the following command below. All policies, the indices of their data sets, and the planning results
will be stored in the log directory.
``` 
python3 network/training.py --train data/states/train/reward/reward --validation data/states/validation/reward/reward --logdir /Users/nicola_mueller/desktop/gnn-policy-kr22/train_reward --domain reward --runs 10 --seeds 2 --rounds 3 
```
- `retraining.py`: Retrains policies. It is a large script that implements the complete retraining pipeline consisting of the following steps:
  1. Loading the trained policy given by the --policy argument (if omitted then a policy is trained from scratch but this is not done by sampling multiple data sets)
  2. Retraining the policy using the bugfiles given by the --bugs argument. If you provide the "--no_retrain" flag, then this and the following steps are skipped and the script just runs the given policy 
     on the test instances.
  3. Determining the best retrained policy according the validation loss (there are alternatives for doing this in the code).
  4. Continuing the training of the original policy for the same number of epochs as the best retrained policy was trained for.
     This is done to test whether the retrained policy is better than the original policy due to being trained on bugs or just because it was trained longer. 
     This part can be skipped by providing the "--no_continue" flag.
  5. Finding the best continued policy according to the validation loss.
  6. Evaluating the best continued policy's loss on bugs. This tests whether training for more epochs can improve performance on bugs without needing to specifically train on them.
  7. All policies are run on the test instances multiple times (--runs) and the results are stored in a "results.csv" file inside the log directory. The plans of each policy are stored there too.

For example, to retrain the policy trained on the reward domain with the corresponding bugfiles, using the same training and validation set as the original policy (--train_indices and --val_indices),
repeating the optimization with 10 random seeds (--seeds), and running the resulting best policies on each test instance 10 times (--runs), run the following command:
```
python3 network/retraining.py --policy /Users/nicola_mueller/Desktop/gnn-policy-kr22/our_models/reward/epoch=137-step=19320-validation_loss=0.0020315158180892467.ckpt --train data/states/train/reward/reward --validation data/states/validation/reward/reward --train_indices test/train_indices_selected_states.json --val_indices test/validation_indices_selected_states.json --bugs /Users/nicola_mueller/Desktop/gnn-policy-kr22/bugfiles2/train --logdir /Users/nicola_mueller/desktop/gnn-policy-kr22/retrain_reward --domain reward --seeds 10 --runs 30 
```
The retraining script has many more optinal arguments, and you can find them in the `_parse_arguments` function in line 130.
- `bugs_planning.py`: Runs policies on bug states. For a given policy (--policy) and a set of bugs (--bugs), this script computes the policy's loss and planning performance on each bug multiple times
   (--runs) and stores the results in a "results.csv" file inside the log directory. The plans of each policy are stored there too. For example, run the following command:
```
python3 network/bugs_planning.py --policy /Users/nicola_mueller/Desktop/gnn-policy-kr22/our_models/reward/epoch=137-step=19320-validation_loss=0.0020315158180892467.ckpt --bugs /Users/nicola_mueller/Desktop/gnn-policy-kr22/bugfiles2/train --logdir /Users/nicola_mueller/desktop/gnn-policy-kr22/bugs_reward --domain reward --runs 3
```
- `read_plans.py`: Reads plans and prints coverage and plan quality. This simple utility script was used to debug the planning code. For example, run the following command:
``` 
python3 network/read_plans.py --logdir /Users/nicola_mueller/Desktop/gnn-policy-kr22/test_reward/plans_trained
```
## network/architecture
This directory contains the code for the GNN architectures and loading the models. In `model.py` the `_create_unsupervised_retrain_model_class` implements the retraining algorithm.