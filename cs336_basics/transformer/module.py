from typing import Optional

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: Optional[str] = None,
        device: Optional[str] = None,
    )-> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device=device,
                dtype=dtype
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._weight_init()

    def _weight_init(self):
        a = (6 / (self.in_features + self.out_features)) ** 0.5
        self.weight.data.uniform_(-a, a)
        if self.bias:
            self.bias.data.fill_(0)

    def forward(self, x):
        output = torch.einsum("...i,ji->...j", x, self.weight)
        if self.bias:
            output = output + self.bias
        return output
