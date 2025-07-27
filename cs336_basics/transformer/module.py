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
        dtype: Optional[None] = None,
        device: Optional[str] = None,
    )-> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device=device,
                dtype=dtype,
            ) # type: ignore
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
        dtype: Optional[None] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embed = nn.Parameter(
            torch.empty(
                (vocab_size, embedding_dim),
                device=device,
                dtype=dtype,
            ) # type: ignore
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
        """
        Multi-Head Attention
        
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        """
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
        """
        Forward method
        
        Args:
            in_features (Float[Tensor, "... sequence_length d_in"]): Input tensor
        """
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


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        theta: int,
        d_k: int,
        max_seq_len: int,
        scale_theta: int = 1,
        device: Optional[None] = None,
    ):
        """
        RoPE embeddings

        Args:
            theta: Frequency base (typically 10,000)
            d_k: Dimenstio of query/key vectors (must be even)
            max_seq_len: Maximum sequence length to precompute
            device: Target device (optinal)
        """
        super().__init__()
        self.theta = theta * scale_theta**(d_k / (d_k - 2))
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self.register_buffer("rope", self._precompute(), persistent=False)

    def _precompute(self) -> Float[Tensor, "... sequence_length d_k // 2"]:
        freqs = 1.0 / (
            self.theta ** (
                torch.arange(
                    0, self.d_k, 2, device=self.device, dtype=torch.float32
                ) / self.d_k
            )
        )  # (self.d_k // 2,)

        seq_indices = torch.arange(
            self.max_seq_len, device=self.device, dtype=torch.float32
        )  # [0, 1, 2, ... , max_seq_len - 1]

        freqs = torch.einsum("i,j -> ij", seq_indices, freqs)
        # position x frequency = (max_seq_len, d_k // 2)
        
        # Convert to complex numbers: e^(i * m * theta_i)
        # (max_seq_len, d_k // 2)
        return torch.polar(torch.ones_like(freqs), freqs)
    
    def forward(
        self,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ):
        """
        Apply rotary position embeddings to input tensor (query or key)
        
        Args:
            in_query_or_key: Input tensor of shape Float[Tensor, " ... sequence_length d_k"],
            token_positions: Position indices of shape (*, seq_len)
        """
        if in_query_or_key.size(-1) != self.d_k:
            raise ValueError(
                f"Input dimension ({in_query_or_key.size(-1)}) != d_k ({self.d_k})"
            )
        
        if token_positions.max() >= self.max_seq_len:
            raise IndexError(
                f"Position index {token_positions.max()} exceed max_seq_len ({self.max_seq_len})"
            )
        
        # Reshape to handle dimension pairs: (*, seq_len, d_k) -> (*, seq_len, d_k//2, 2)
        x_complex = rearrange(in_query_or_key, "... seq (d two) -> ... seq d two", two=2).float()

        x_complex = torch.view_as_complex(x_complex)
        
        rope_pos = self.rope[token_positions]  # (*, seq_len, d_k // 2)

        x_rotated = x_complex * rope_pos  # (*, seq_len, d_k // 2)

        x_real = torch.view_as_real(x_rotated)

        x_real = rearrange(x_real, "... seq d two -> ... seq (d two)")

        return x_real.to(in_query_or_key.dtype)
