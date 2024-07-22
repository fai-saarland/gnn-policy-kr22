#!/bin/sh

cp -r /home/s8namuel/gnn/gnn-policy-kr22/ .
cd gnn-policy-kr22
python3.10 network/training.py --train data/states/train/reward2/reward/ --validation data/states/validation/reward2/reward/ --logdir /home/s8namuel/gnn/ablation_reward/L1  --domain reward --seeds 2 --rounds 3 --runs 10 --gpus 1 --loss L1

python3.10 network/training.py --train data/states/train/reward2/reward/ --validation data/states/validation/reward2/reward/ --logdir /home/s8namuel/gnn/ablation_reward/MSE  --domain reward --seeds 2 --rounds 3 --runs 10 --gpus 1 --loss mean_squared_error

python3.10 network/training.py --train data/states/train/reward2/reward/ --validation data/states/validation/reward2/reward/ --logdir /home/s8namuel/gnn/ablation_reward/L1_MSE  --domain reward --seeds 2 --rounds 3 --runs 10 --gpus 1 --loss L1_MSE
