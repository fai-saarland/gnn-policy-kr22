#!/bin/sh

cp -r /home/s8namuel/gnn/gnn-policy-kr22/ .
cd gnn-policy-kr22

python3.10 network/tuning_old2.py --train data_old/supervised/optimal/train/blocks-clear/blocks-clear --validation data_old/supervised/optimal/validation/blocks-clear/blocks-clear --logdir /home/s8namuel/gnn/tune_new_gcngps/tune_new_gcngps_blocks_clear --domain blocks-clear --aggregation GCNGPS --readout MAX --loss MSE --runs 1 --seeds 3 --rounds 1 --num_layers_range 2 4 --hidden_size_range 32 64 128 --dropout_range 0.1 0.5 --heads_range 1 2 --gpus 1 --coverage_validation

python3.10 network/tuning_old2.py --train data_old/supervised/optimal/train/gripper-atomic/gripper-atomic --validation data_old/supervised/optimal/validation/gripper-atomic/gripper-atomic --logdir /home/s8namuel/gnn/tune_new_gcngps/tune_new_gcngps_gripper_atomic --domain gripper --aggregation GCNGPS --readout MAX --loss MSE --runs 1 --seeds 3 --rounds 1 --num_layers_range 2 4 --hidden_size_range 32 64 128 --dropout_range 0.1 0.5 --heads_range 1 2 --gpus 1 --coverage_validation

python3.10 network/tuning_old2.py --train data_old/supervised/optimal/train/visitall-atomic/visitall-atomic --validation data_old/supervised/optimal/validation/visitall-atomic/visitall-atomic --logdir /home/s8namuel/gnn/tune_new_gcngps/tune_new_gcngps_visitall_atomic --domain visitall --aggregation GCNGPS --readout MAX --loss MSE --runs 1 --seeds 3 --rounds 1 --num_layers_range 2 4 --hidden_size_range 32 64 128 --dropout_range 0.1 0.5 --heads_range 1 2 --gpus 1 --coverage_validation

python3.10 network/tuning_old2.py --train data_old/supervised/optimal/train/parking-behind/parking-behind --validation data_old/supervised/optimal/validation/parking-behind/parking-behind --logdir /home/s8namuel/gnn/tune_new_gcngps/tune_new_gcngps_parking_behind --domain parking-behind --aggregation GCNGPS --readout MAX --loss MSE --runs 1 --seeds 3 --rounds 1 --num_layers_range 2 4 --hidden_size_range 32 64 128 --dropout_range 0.1 0.5 --heads_range 1 2 --gpus 1 --coverage_validation

python3.10 network/tuning_old2.py --train data_old/supervised/optimal/train/satellite-atomic/satellite-atomic --validation data_old/supervised/optimal/validation/satellite-atomic/satellite-atomic --logdir /home/s8namuel/gnn/tune_new_gcngps/tune_new_gcngps_satellite_atomic --domain satellite  --aggregation GCNGPS --readout MAX --loss MSE --runs 1 --seeds 3 --rounds 1 --num_layers_range 2 4 --hidden_size_range 32 64 128 --dropout_range 0.1 0.5 --heads_range 1 2 --gpus 1 --coverage_validation
