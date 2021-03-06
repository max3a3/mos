#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
source activate deep
python savematrix.py --savedir PTB5MOC-20180501-164549 --type MoC --n_experts 5
python savematrix.py --savedir PTB10MOC-20180501-113137 --type MoC --n_experts 10
python savematrix.py --savedir PTB1MOC-20180501-160447 --type MoC --n_experts 1
python savematrix.py --savedir PTB3MOC-20180501-113056 --type MoC --n_experts 3
python savematrix.py --savedir PTB15MOC-20180503-011712 --type MoC --n_experts 15
python savematrix.py --savedir PTB20MOC-20180502-123458 --type MoC --n_experts 20