#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
source activate deep
python savematrix.py --savedir PTB20-20180428-112448 --n_experts 20
python savematrix.py --savedir PTB10-20180428-112310 --n_experts 10
python savematrix.py --savedir PTB1-20180428-110808 --n_experts 1
python savematrix.py --savedir PTB3-20180428-111733 --n_experts 3
python savematrix.py --savedir PTB15-20180503-010443 --n_experts 15
python savematrix.py --savedir PTB5-20180428-112240 --n_experts 5