import argparse

import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import data
import model

from utils import get_batch, repackage_hidden


parser = argparse.ArgumentParser(description='Save numpy matrix from model saved (trained)')
parser.add_argument('--savedir', type=str, default='PTB1-20180428-110808',
                    help='where the model is saved')
parser.add_argument('--n_experts', type=int, default=1,
                    help='number of experts')
parser.add_argument('--type', type=str, default='MoS',
                    help='MoS or MoC')
args = parser.parse_args()

model = torch.load(args.savedir +'/model.pt')
f = open(args.savedir + '/log.txt', 'rb')

corpus = data.Corpus('data/penn')


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    data = data.cuda()
    return data


eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, 12)
val_data = batchify(corpus.valid, 10)
test_data = batchify(corpus.test, 1)

test = test_data


def get_batch(source, i, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else 70, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len])
    return data, target


parallel_model = nn.DataParallel(model, dim=1).cuda()


def evaluate(test, batch_size=1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    for i in range(0, test.size(0) - 1, 70):
        data, targets = get_batch(test, i, evaluation=True)
        targets = targets.view(-1)

        log_prob, hidden = parallel_model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)
    return math.exp(total_loss[0] / len(test))


perplexity = evaluate(test)


def get_learning_curve(file):
    val_per = []
    for line in file.readlines():
        str_line = str(line)
        if 'valid' in str_line:
            val_per.append(float(str_line.split()[-1][:-3]))
    return val_per


val_per = get_learning_curve(f)

pytorch_total_params = sum(p.numel() for p in model.parameters())


pickle.dump({'param': pytorch_total_params,
             'test_per': perplexity,
             'val_per': val_per},
            open('results_{}_{}.pkl'.format(args.type, args.n_experts), 'wb'))