import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


def get_loss_coefs(alpha: float, beta: float = None) -> ndarray:
    """
    Returns loss weights. Sum of the weights is 1.
    Args:
        alpha: logit for second loss.
        beta:  logit for third loss.
    Returns:
        Array with weights.
    """
    if beta is None:
        return torch.softmax([1, alpha]).numpy()
    return torch.softmax([1, alpha, beta]).numpy()
