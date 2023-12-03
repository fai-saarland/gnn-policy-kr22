#!/bin/sh

cp -r /home/s8namuel/gnn/gnn-policy-kr22/ .
cd gnn-policy-kr22
python3.10 network/training.py --train data/states/train/reward2/reward --validation data/states/validation/reward2/reward --logdir /home/s8namuel/gnn/trans_reward3_30_small --iterations 30 --domain reward --seeds 2 --rounds 3 --runs 30 --gpus 1 --aggregation planformer --d_model 64 --d_ff 128 --n_heads 4 --n_layers 2
