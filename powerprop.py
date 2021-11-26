import torch

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR100

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset

from utils.pp_modules import MLP
from utils.utils import preprocess, train_val_split, init_weights, train, evaluate, evaluate_pruning, \
    plot_sparsity_performance

# Training Configuration
model_seed = 0  # @param

alphas = [1.0, 2.0, 3.0, 4.0, 5.0]  # @param
init_distribution = 'truncated_normal'
init_mode = 'fan_in'
init_scale = 1.0

# Fixed values taken from the Lottery Ticket Hypothesis paper
train_batch_size = 60
validation_size = 5000
num_train_steps = 50000
learning_rate = 0.1
report_interval = 2500

# Training setup
train_data = MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=ToTensor()
)

# reshape and normalize data
train_data.data = preprocess(train_data.data)
test_data.data = preprocess(test_data.data)

train_x, train_y, valid_x, valid_y = train_val_split(train_data, validation_size)

test_x = test_data.data
test_y = test_data.targets

dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=train_batch_size, shuffle=True)
model_list = []
model_types = []
for alpha in alphas:
    model = MLP(alpha=alpha)
    model.apply(init_weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    CE_loss = torch.nn.CrossEntropyLoss()

    train(model, dataloader, optimizer, CE_loss)

    evaluate(model, test_x, test_y, CE_loss)

    if alpha > 1.0:
        model_types.append('Powerprop. ($\\alpha={}$)'.format(alpha))
    else:
        model_types.append('Baseline')

    model_list.append(model)

acc_at_sparsity, eval_at_sparsity_level = evaluate_pruning(model_list, test_x, test_y, alphas)
plot_sparsity_performance(acc_at_sparsity, eval_at_sparsity_level, model_types, dataset_desc="MNIST")
