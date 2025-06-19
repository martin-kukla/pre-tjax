#! /bin/bash
# Assumes you are using the following nvidia image: jax24.04-py3 (https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/rel-24-04.html)
pip install datasets
pip install wandb
wandb login
pip install evaluate