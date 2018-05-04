import argparse

import math
import numpy as np
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
parser.add_argument('--tosave_dir', type=str, default='/data/milatmp1/lahlosal',
                    help='where to save the matrix')
parser.add_argument('--type', type=str, default='MoS',
                    help='MoS or MoC')
args = parser.parse_args()

model = torch.load(args.savedir +'/model.pt')

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
    log_probs = []
    for i in range(0, test.size(0) - 1, 70):
        data, targets = get_batch(test, i, evaluation=True)
        targets = targets.view(-1)

        log_prob, hidden = parallel_model(data, hidden)
        log_probs.append(log_prob)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)
    return log_probs, math.exp(total_loss[0] / len(test))

log_probs, perplexity = evaluate(test)

print('Perplexity on test set with {} experts {}: {}'.format(perplexity, args.n_experts, args.type))

mat = torch.cat([mat.cpu().view(-1, 10000) for mat in log_probs])
mat = mat.data.numpy()

np.save(args.tosave_dir + '/{}_{}.npy'.format(args.type, args.n_experts), mat)