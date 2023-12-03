#!/bin/sh

cp -r /home/s8namuel/gnn/gnn-policy-kr22/ .
cd gnn-policy-kr22
python3.10 network/retraining.py --train data/states/train/reward/reward/ --validation data/states/validation/reward/reward/ --bugs bugfiles2/train --logdir /home/s8namuel/gnn/reward_new_loss --policy /home/s8namuel/gnn/gnn-policy-kr22/our_models/reward/epoch=137-step=19320-validation_loss=0.0020315158180892467.ckpt --train_indices /home/s8namuel/gnn/gnn-policy-kr22/our_models/reward/train_indices_selected_states.json  --val_indices /home/s8namuel/gnn/gnn-policy-kr22/our_models/reward/validation_indices_selected_states.json  --domain reward --seeds 10 --runs 30  --gpus 1
