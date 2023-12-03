#!/bin/sh

cp -r /home/s8namuel/gnn/gnn-policy-kr22/ .
cd gnn-policy-kr22
python3.10 network/bugs_planning.py --bugs bugfiles2/train  --logdir /home/s8namuel/gnn/plan_bugs_train --policy /home/s8namuel/gnn/gnn-policy-kr22/our_models/reward/epoch=137-step=19320-validation_loss=0.0020315158180892467.ckpt  --domain reward --runs 10 --gpus 1 
