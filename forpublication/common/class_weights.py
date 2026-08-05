import torch

def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * len(counts)
    return weights
