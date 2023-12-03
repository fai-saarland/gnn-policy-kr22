#!/bin/sh

cp -r /home/s8namuel/gnn/gnn-policy-kr22/ .
cd gnn-policy-kr22
python3.10 network/training.py --train data/states/train/reward2/reward/ --validation data/states/validation/reward2/reward/ --logdir /home/s8namuel/gnn/reward2_training  --domain reward --seeds 1 --rounds 3 --runs 10 --gpus 1 
