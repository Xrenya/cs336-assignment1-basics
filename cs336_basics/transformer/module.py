from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from torch import Tensor
from einops import rearrange


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

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        output = torch.einsum("...i,ji->...j", x, self.weight)
        if self.bias:
            output = output + self.bias
        return output


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embed = nn.Parameter(
            torch.empty(
                (vocab_size, embedding_dim),
            )
        )
        self._weight_init()

    def _weight_init(self):
        self.embed.data.uniform_(-0.1, 0.1)

    def forward(self, x: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return torch.stack(
            [
                torch.index_select(
                self.embed, dim=0, index=idx
                ) for idx in x
            ]
        )


class FFNSwiGLU(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        bias: bool = False,
        dtype: Optional[None] = None,
        device: Optional[None] = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(
            embed_dim,
            hidden_dim,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.w2 = Linear(
            hidden_dim,
            embed_dim,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.w3 = Linear(
            embed_dim,
            hidden_dim,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.swish = Swish(beta=1)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, " ... d_model"]:
        swish = self.swish(self.w1(x))
        gate = self.w3(x)
        swiglu = swish * gate
        return self.w2(swiglu)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def _positive_forward(self, x):
        return 1. / (1. + torch.exp(-x))

    def _negative_forward(self, x):
        exp = torch.exp(x)
        return exp / (1. + exp)

    def forward(self, x):
        positive_mask = x >= 0
        output = torch.zeros_like(x)
        output[positive_mask] = self._positive_forward(x[positive_mask])
        output[~positive_mask] = self._negative_forward(x[~positive_mask])
        return output
    

class Swish(nn.Module):
    def __init__(self, beta: int = 1) -> None:
        super().__init__()
        self.beta = beta
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x * self.sigmoid(self.beta * x)
        return x


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.swish = Swish(beta=1)

    def forward(self, x):
        return self.swish(x)


class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        values, indices = torch.max(x, dim=self.dim, keepdim=True)
        x = x - values
        e_x = torch.exp(x)
        return e_x / torch.sum(e_x, dim=self.dim, keepdim=True)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax(dim=-1)

    def forward(
        self,
        q: Float[Tensor, " ... queries d_k"],
        k: Float[Tensor, " ... keys d_k"],
        v: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
    ):
        *_, head_dim = q.shape
        q = q * head_dim**-0.5
        weight = q @ k.transpose(-2, -1)
        if mask is not None:
            weight.masked_fill_(mask.logical_not(), float("-inf"))
            # weight[mask.logical_not()] = -torch.inf
            # weight[~mask] = -torch.inf
        score = self.softmax(weight)
        attn = score @ v
        return attn
    

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: Optional[None] = None,
        dtype: Optional[None] = None,
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = Linear(
            in_features=d_model,
            out_features=3 * d_model,
            device=device,
            dtype=dtype,
        )
        self.projection = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.attention = Attention()

    def forward(
        self, in_features: Float[Tensor, " ... sequence_length d_in"]
    ):  
        seq_len = in_features.shape[-2]
        qkv = self.qkv(in_features)
        q, k, v = rearrange(
            qkv,
            "... seq_len (split head d_k) -> split ... head seq_len d_k",
            split=3,
            head=self.num_heads
        )
        mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool)
        ).to(self.device)
        attn = self.attention(q, k, v, mask=mask)
        attn = rearrange(
            attn,
            "... head seq_len d_k -> ... seq_len (head d_k)")
        proj = self.projection(attn)
        return proj


