#!/bin/sh

cp -r /home/s8namuel/general_policies/ .
cd general_policies
python3.10 network/retrain.py --train data/states/train/reward_retrain/reward_retrain/ --validation data/states/validation/reward/reward/ --bugs data/states/train/reward_bugs/reward_bugs/ --logdir /home/s8namuel/general_policies/lightning_logs/reward_retrain --aggregation retrain_max  --resume /home/s8namuel/general_policies/lightning_logs/reward_retrain/version_0/checkpoints/epoch=260-step=31842-validation_loss=0.0015830622287467122.ckpt --gpus 1
