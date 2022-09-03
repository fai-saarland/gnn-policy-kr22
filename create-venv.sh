#!/bin/bash

python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install termcolor
pip install torch
pip install pytorch_lightning
pip install tarski
