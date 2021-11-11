import torch

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset

from utils.pp_modules import MLP
from utils.utils import init_weights, train, cat_loss, accuracy

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
train_data.data = train_data.data.flatten(start_dim=1).float() / 255.
test_data.data = test_data.data.flatten(start_dim=1).float() / 255.

train_x = train_data.data[:-validation_size]
train_y = train_data.targets[:-validation_size]

# Reserve some data for a validation set
valid_x = train_data.data[-validation_size:]
valid_y = train_data.targets[-validation_size:]

test_x = test_data.data
test_y = test_data.targets

dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=train_batch_size, shuffle=True)

model = MLP(alpha=alphas[0])
model.apply(init_weights)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
CE_loss = torch.nn.CrossEntropyLoss()

train(model, dataloader, optimizer, CE_loss, accuracy)
