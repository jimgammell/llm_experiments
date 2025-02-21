# Adapted from https://github.com/jzhang38/TinyLlama

from typing import *
import torch
from torch import nn
from xformers.ops import SwiGLU

class Attention(nn.Module):
    def __init__(self, input_dims: int, head_count, n_query_groups, head_size, bias):
        super().__init__()
        self.input_dims = input_dims
        self.head_count = head_count
        self.n_query_groups = n_query_groups
        self.head_size = head_size
        self.bias = bias
    
    def construct(self):
        shape = self.head_size*(self.head_count + 2*self.n_query_groups)
        self.to_qkv = nn.Linear(self.input_dims, shape, bias=self.bias)
        

class FeedForwardNet(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        
    def construct(self):
        self.swiglu = SwiGLU(self.input_dims, self.hidden_dims, bias=False, _pack_weights=False)
    
    def forward(self, x):
        return self.swiglu(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(in_dims={self.input_dims}, hidden_dims={self.hidden_dims})'

def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    theta = 1./(base**(torch.arange(0, n_elem, 2, device=device) / n_elem))
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio
    idx_theta = torch.outer(seq_idx, theta)
    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    elif dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    else:
        return cos, sin

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., :head_size//2]
    x2 = x[..., head_size//2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    roped = (x*cos) + (rotated*sin)
    return roped.type_as(x)