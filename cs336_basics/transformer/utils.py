from typing import Iterable
from jaxtyping import Float, Int

import torch
import torch.nn as nn
from torch import Tensor


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
):
    total = 0
    for param in parameters:
        if param.grad is not None:
            total += torch.sum(param.grad.data**2).item()

    total = total**0.5
    if total > max_l2_norm:
        scale = max_l2_norm / total
        for param in parameters:
            if param.grad is not None:
                param.grad *= scale

    return parameters


def cross_entropy_loss(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
):
    
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    logp = inputs.gather(1, targets.unsqueeze(1)).squeeze(1)

    logp_1 = torch.log(torch.sum(torch.exp(inputs), dim=-1))

    return (-logp + logp_1).mean()
