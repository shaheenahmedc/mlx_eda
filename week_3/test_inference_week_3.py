from model import TransformerSampler

# import sys
# import einops
# from dataclasses import dataclass
from transformer_lens import HookedTransformer
from model import Config
from model import DemoTransformer

# from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
# from torch import Tensor
# import torch.nn as nn
# import numpy as np
# import math
# from tqdm import tqdm
# from typing import Tuple, List, Optional, Dict
# from jaxtyping import Float, Int
# from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
# from collections import defaultdict
# from rich.table import Table
# from rich import print as rprint
# import datasets
# from torch.utils.data import DataLoader
# import wandb
# from pathlib import Path
# import webbrowser

reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False
)
tokenizer = reference_gpt2.tokenizer

model_cfg = Config(
    d_model=256,
    n_heads=4,
    d_head=64,
    d_mlp=1024,
    n_layers=12,
    n_ctx=256,
    d_vocab= 50257
)


sampling_model = DemoTransformer(model_cfg).to(model_cfg.device)
sampling_model.load_state_dict(t.load("gpt2_style_model_weights_12_layers_3_epochs.pth"))

sampler = TransformerSampler(sampling_model, tokenizer, model_cfg)

prompt = 'What do you think about apples?'
print (sampler.sample(prompt = prompt))
