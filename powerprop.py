import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import random_split

from torchvision.transforms import ToTensor

#Training Configuration
model_seed = 0  #@param

alphas = [1.0, 2.0, 3.0, 4.0, 5.0]  #@param
init_distribution = 'truncated_normal'
init_mode = 'fan_in'
init_scale = 1.0

# Fixed values taken from the Lottery Ticket Hypothesis paper
train_batch_size = 60
num_train_steps = 50000
learning_rate = 0.1
report_interval = 2500

#Training setup
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True
)
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor()
)

# calculate size of train and validation sets. 80% train and 20% validation
train_size = int(0.8 * len(train_data.train_data))
valid_size = len(train_data.train_data) - train_size

#Randomly split the train and validation sets
partial_train_data, validation_data = random_split(train_data, [train_size, valid_size])

train_x = partial_train_data.dataset.train_data
train_y = partial_train_data.dataset.train_labels

val_x = validation_data.dataset.train_data
val_y = validation_data.dataset.train_labels

test_x = test_data.test_data
test_y = test_data.test_labels


