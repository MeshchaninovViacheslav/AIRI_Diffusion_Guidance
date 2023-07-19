import torch


@torch.no_grad()
def accuracy(output, target):
    """Computes the accuracy"""
    assert output.shape == target.shape
    return torch.sum(output == target) / target.size(0)
