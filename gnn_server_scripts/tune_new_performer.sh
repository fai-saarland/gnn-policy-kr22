#!/bin/sh

cp -r /home/s8namuel/gnn/gnn-policy-kr22/ .
cd gnn-policy-kr22

python3.10 network/tuning2.py --train data_old/supervised/optimal/train/blocks-clear/blocks-clear --validation data_old/supervised/optimal/validation/blocks-clear/blocks-clear --logdir /home/s8namuel/gnn/tune_new_performer2/tune_new_performer_blocks_clear --domain blocks-clear --aggregation Performer --seeds 4 --num_layers_range 2 4 --hidden_size_range 32 64 128 --dropout_range 0.1 0.5 --heads_range 1 2 --gpus 1 

python3.10 network/tuning2.py --train data_old/supervised/optimal/train/visitall-atomic/visitall-atomic --validation data_old/supervised/optimal/validation/visitall-atomic/visitall-atomic --logdir /home/s8namuel/gnn/tune_new_performer2/tune_new_performer_visitall_atomic --domain visitall --aggregation Performer --seeds 4 --num_layers_range 2 4 --hidden_size_range 32 64 128 --dropout_range 0.1 0.5 --heads_range 1 2 --gpus 1 

python3.10 network/tuning2.py --train data_old/supervised/optimal/train/parking-behind/parking-behind --validation data_old/supervised/optimal/validation/parking-behind/parking-behind --logdir /home/s8namuel/gnn/tune_new_performer2/tune_new_performer_parking_behind --domain parking-behind --aggregation Performer --seeds 4 --num_layers_range 2 4 --hidden_size_range 32 64 128 --dropout_range 0.1 0.5 --heads_range 1 2 --gpus 1 

python3.10 network/tuning2.py --train data_old/supervised/optimal/train/satellite-atomic/satellite-atomic --validation data_old/supervised/optimal/validation/satellite-atomic/satellite-atomic --logdir /home/s8namuel/gnn/tune_new_performer2/tune_new_performer_satellite_atomic --domain satellite  --aggregation Performer --seeds 4 --rounds 1 --num_layers_range 2 4 --hidden_size_range 32 64 128 --dropout_range 0.1 0.5 --heads_range 1 2 --gpus 1 
