#! /bin/bash
# Assumes you are using (Pytorch) image with cuda12.4
pip install jupyter
pip install datasets
pip install torch
pip install tensorboard
pip install line_profiler
pip install matplotlib

# TODO XXX: Needed for memory profiling, but revist if it has side effects:
pip install tensorflow
pip install -U tensorboard-plugin-profile

pip install evaluate # For BLEU computation

/usr/local/bin/jupyter-notebook --port=3000 --ip=0.0.0.0 --no-browser --allow-root --ServerApp.token= --notebook-dir=/efs/notebooks/mkukla > /root/nb.out 2> /root/nb.err &
tensorboard --port=6006 --host=0.0.0.0 --path_prefix=/tb --logdir /lego/storage/output > /root/tb.out 2> /root/tb.err &

git config --global user.email "martin.kukla@cantab.net"
git config --global user.name "Martin Kukla"
git config --global credential.helper store # This is not secure...
git config --global --add safe.directory /efs/notebooks/mkukla/pre-tjax

pip install wandb
wandb login


# Corect packages of torch/triton for running torchfunc_jit/triton version
pip install torch==2.7.1
pip install triton==3.3.1
pip install optree==0.14.0

# Pin down numpy to the verion priox to 2.x
pip install numpy==1.26.4