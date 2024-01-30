from dataclasses import dataclass
import torch as t

@dataclass
class Config:
    d_model: int = 768
#     debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    device: str = t.device("cuda" if t.cuda.is_available() else "cpu")
