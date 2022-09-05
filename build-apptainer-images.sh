#!/bin/bash

apptainer build gnn-plan.img Apptainer.plan
apptainer build gnn-train.img Apptainer.train
apptainer build gnn-policy-server.img Apptainer.policy-server
