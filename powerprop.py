import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import utils as uu
from utils.pp_modules import MLP
from utils.loading import load_data


model_seed = 0
torch.manual_seed(model_seed)

# Training Configuration
dataset = 'MNIST'
alphas = [1.0, 2.0, 3.0, 4.0, 5.0]
validation_size = 5000
report_interval = 2500

# Training Hyperparameters
train_batch_size = 60
num_train_steps = 50000
learning_rate = 0.0025
momentum = 0.9

# load data
train_x, train_y = load_data(dataset, train=True)
test_x, test_y = load_data(dataset, train=False)

epochs = int(num_train_steps * train_batch_size / train_x.shape[0])
dataset = TensorDataset(train_x, train_y)
dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

model_list = []
model_types = []
for alpha in alphas:
    model = MLP(alpha=alpha)
    model.apply(uu.init_weights)
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
uu.plot_sparsity_performance(acc_at_sparsity, eval_at_sparsity_level, model_types, dataset_desc="MNIST")
