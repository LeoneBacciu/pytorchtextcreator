# Imports

import torch
from model import WordLSTM
from predict import Generator

f_name = 'lstm_rap'

# Define and print the net
n_hidden = 512
n_layers = 4

net = WordLSTM(95, n_hidden, n_layers, drop_prob=0.3)
print(net)

net.load_state_dict(torch.load(f_name + '_2.pt', map_location='cpu'))

gen = Generator(net)

# Generating new text
for n in range(5):
    print('Song ' + str(n))
    print(gen.predict(net, 1000, prime='nascondino ', top_k=2))
    print()
