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

epochs = int(num_train_steps * train_batch_size / train_x.shape[0])
dataset = TensorDataset(train_x, train_y)
dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

model_list = []
model_types = []
init_weights_list = []

for alpha in alphas:
    model = CNN(alpha=alpha)
    model.apply(uu.init_weights)

    #Capturing initial weights to be used later for plots
    init_weights = uu.get_init_weights(model).flatten()
    init_weights_list.append(init_weights)

    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    CE_loss = torch.nn.CrossEntropyLoss()

    uu.train(model, dataloader, optimizer, CE_loss, epochs=epochs)

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

masks_test = uu.get_mask_by_perc(.1, model_list[0])
pruned_weights = model_list[0].forward(test_x, masks_test)
uu.plot_pruned_vs_remaining_weights(init_weights_list[0], pruned_weights, chart_name="Baseline", dataset_desc="CIFAR")

masks_test = uu.get_mask_by_perc(.1, model_list[0])
pruned_weights = model_list[4].forward(test_x, masks_test)
uu.plot_pruned_vs_remaining_weights(init_weights_list[0], pruned_weights, chart_name="High_Alpha", dataset_desc="CIFAR")