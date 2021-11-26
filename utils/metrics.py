import torch


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ Accuracy """
    acc = (torch.argmax(outputs, dim=1) == targets).float()
    return torch.mean(acc)


class MetricsContainer:
    """Container for holding mini-batch metrics and epoch metrics"""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.batch_loss = []
        self.batch_acc = []
        self.avg_loss = 0.
        self.avg_acc = 0.
        self.total_loss = 0.
        self.total_acc = 0.
        self.count = 0

    def reset(self):
        self.batch_loss = []
        self.batch_acc = []
        self.avg_loss = 0.
        self.avg_acc = 0.
        self.total_loss = 0.
        self.total_acc = 0.
        self.count = 0

    def update(self, loss: float, acc: float):
        self.batch_loss.append(loss)
        self.batch_acc.append(acc)

        self.count += self.batch_size
        self.total_loss += loss * self.batch_size
        self.total_acc += acc * self.batch_size

        self.avg_loss = self.total_loss / self.count
        self.avg_acc = self.total_acc / self.count