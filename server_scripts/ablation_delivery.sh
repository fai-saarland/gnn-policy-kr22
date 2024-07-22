#!/bin/sh

cp -r /home/s8namuel/gnn/gnn-policy-kr22/ .
cd gnn-policy-kr22
python3.10 network/training.py --train data/states/train/delivery/delivery/ --validation data/states/validation/delivery/delivery/ --logdir /home/s8namuel/gnn/ablation_delivery/L1  --domain blocks --seeds 2 --rounds 3 --runs 10 --gpus 1 --loss L1

python3.10 network/training.py --train data/states/train/delivery/delivery/ --validation data/states/validation/delivery/delivery/ --logdir /home/s8namuel/gnn/ablation_delivery/MSE  --domain blocks --seeds 2 --rounds 3 --runs 10 --gpus 1 --loss mean_squared_error

python3.10 network/training.py --train data/states/train/delivery/delivery/ --validation data/states/validation/delivery/delivery/ --logdir /home/s8namuel/gnn/ablation_delivery/L1_MSE  --domain blocks --seeds 2 --rounds 3 --runs 10 --gpus 1 --loss L1_MSE
