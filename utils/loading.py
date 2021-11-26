import torch
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor


VALID_DATASETS = ['MNIST', 'CIFAR10']


def preprocess(data: torch.Tensor, dataset: str) -> torch.Tensor:
    """
    Preprocess MNIST or CIFAR-10 Data.
    Normalize pixel values between [0, 1].

    :param data: input dataset
    :type data: torch.Tensor
    :param dataset: MNIST or CIFAR10
    :type dataset: str
    :return: normalized input dataset
    :rtype: torch.Tensor
    """
    data = torch.as_tensor(data)
    data = data.float() / 255.
    if dataset == 'MNIST':
        data = data.flatten(start_dim=1)
    else:
        data = data.permute((0, 3, 1, 2))

    return data


def train_val_split(data, val_size: int):
    """
    Split dataset into training and validation sets.

    Validation set is taken as last 'val_size' records in data.

    :param data:
    :type data:
    :param val_size:
    :type val_size:
    :return:
    :rtype: tuple
    """
    train_x = data.data[:-val_size]
    train_y = data.targets[:-val_size]

    valid_x = data.data[-val_size:]
    valid_y = data.targets[-val_size:]

    return train_x, train_y, valid_x, valid_y


def load_data(dataset, train=True):
    """
    Load the specified dataset.

    :param dataset: One of MNIST or CIFAR10
    :type dataset: str
    :param train: whether to get train or test data
    :type train: bool
    :return: tuple of input, output pairs of data
    :rtype: Tuple[Torch.Tensor, Torch.Tensor]
    """
    if dataset not in VALID_DATASETS:
        raise ValueError(f'Dataset must be one of {", ".join(VALID_DATASETS)}.')

    if dataset == 'MNIST':
        train_data = MNIST(
            root=r'../data/',
            train=train,
            download=True,
            transform=ToTensor(),
        )
    else:
        train_data = CIFAR10(
            root=r'../data/',
            train=train,
            download=True,
            transform=ToTensor(),
        )

    x = preprocess(train_data.data, dataset)
    y = torch.as_tensor(train_data.targets)

    return x, y
