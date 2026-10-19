"""Research optimizer helper: remove conflicting components of two era gradients."""
import torch


def merge_era_gradients(first, second, project=False):
    dot = torch.dot(first, second)
    if project and dot < 0:
        a = first - dot / second.square().sum().clamp_min(1e-20) * second
        b = second - dot / first.square().sum().clamp_min(1e-20) * first
        return (a + b) / 2
    return (first + second) / 2
