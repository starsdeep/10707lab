# As usual, a bit of setup
from __future__ import print_function
import time
import numpy as np
#import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from data_loader import load_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
from cs231n.classifiers.rbm import *

# model = RBM()
# solver = None

# #data = get_CIFAR10_data()
# data = load_data()
# for k, v in list(data.items()):
#   print(('%s: ' % k, v.shape))

# solver = Solver(model, data, update_rule='sgd', optim_config={'learning_rate': 1e-3}, print_every=10, lr_decay=0.9, num_epochs=10, batch_size=100)
# solver.train_unsupervise()


import pickle as pickle

data = load_data()
checkpoint = pickle.load(open('RBM_epoch_10.pkl', 'rb'))
rbm_params = checkpoint['model']

model = TwoLayerNet(input_dim=28*28)
solver = Solver(model, data, update_rule='sgd', optim_config={'learning_rate': 1e-3}, print_every=100, lr_decay=0.9, num_epochs=10, batch_size=200)
solver.train()
solver.check_accuracy(data['X_test'], data['y_test'])
plot_solver(solver)


model = TwoLayerNet(input_dim=28*28)
model.params = rbm_params
solver = Solver(model, data, update_rule='sgd', optim_config={'learning_rate': 1e-3}, print_every=100, lr_decay=0.9, num_epochs=10, batch_size=200)
solver.train()
solver.check_accuracy(data['X_test'], data['y_test'])
plot_solver(solver)
