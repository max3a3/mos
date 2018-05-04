import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
directory = 'results/'
colors = 'rbgcyk'
i = 0
for filename in os.listdir(directory):
    if not filename.endswith('pkl') or not 'MoS' in filename:
        continue
    type = filename.split('_')[1]
    n_experts = filename.split('_')[2].split('.')[0]
    with open(directory + filename, 'rb') as f:
        dic = pickle.load(f)
        plt.plot(range(50, len(dic['val_per'])), dic['val_per'][50:], colors[i], label='{}_{}'.format(type, n_experts))
        #plt.title("Validation perplexity for {} with {} experts, starting from the 50th epoch".format(type, n_experts))
        plt.xlabel("epoch")
        plt.plot([np.argmin(dic['val_per'])] * 10, np.linspace(55, min(dic['val_per']), 10), colors[i] + '-.')
        #plt.savefig(directory + 'val_per_{}_{}.png'.format(type, n_experts))
        #plt.show()
        #plt.clf()
    i += 1
plt.title("Validation perplexity for different MoS models, starting from epoch 50")
plt.legend()
plt.savefig(directory + 'val_per.png')
plt.show()
