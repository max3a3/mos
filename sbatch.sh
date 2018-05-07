#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
source activate deep
python main.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 20.0 --epoch 1000 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 1 --save PTB1 --single_gpu --underflow
