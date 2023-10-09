#!/bin/sh

cp -r /home/s8namuel/general_policies/ .
cd general_policies
python3.10 network/train.py --train data/states/train/reward/reward/ --validation data/states/validation/reward/reward/  --loss selfsupervised_suboptimal  --logdir /home/s8namuel/general_policies/lightning_logs/reward/ --aggregation max --gpus 1
