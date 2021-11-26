"""
Script to run pruning experiments for CNN on CIFAR-10 data.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import utils as uu
from utils.loading import load_data
from utils.pp_modules import CNN


model_seed = 0
torch.manual_seed(model_seed)

# Training Configuration
dataset = 'CIFAR10'
alphas = [1.0, 2.0, 3.0, 4.0, 5.0]
report_interval = 2500

# Training Hyperparameters
train_batch_size = 60
num_train_steps = 100000
learning_rate = 0.0025
momentum = 0.9

# load data
train_x, train_y = load_data(dataset, train=True)
test_x, test_y = load_data(dataset, train=False)

dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=train_batch_size, shuffle=True)

model_list = []
model_types = []
for alpha in alphas:
    model = CNN(alpha=alpha)
    model.apply(uu.init_weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    CE_loss = torch.nn.CrossEntropyLoss()

    uu.train(model, dataloader, optimizer, CE_loss, training_steps=num_train_steps)

    test_loss, test_acc = uu.evaluate(model, test_x, test_y, CE_loss)
    print(f'Test Set [alpha={alpha}]:\tLoss={test_loss:.4f}\tAcc={test_acc:.4f}')

    if alpha > 1.0:
        model_types.append('Powerprop. ($\\alpha={}$)'.format(alpha))
    else:
        model_types.append('Baseline')

    model_list.append(model)

CE_loss = torch.nn.CrossEntropyLoss()
acc_at_sparsity, eval_at_sparsity_level = uu.evaluate_pruning(model_list, test_x, test_y, alphas, CE_loss)
uu.plot_sparsity_performance(acc_at_sparsity, eval_at_sparsity_level, model_types, dataset_desc=dataset)
