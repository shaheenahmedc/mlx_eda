#!/usr/bin/env python
# coding: utf-8

# # Overview

# This notebook runs through the week 3 task from the MLX apprenticeship, namely re-implementing GPT-2 from scratch.
# It follows the tutorial [here](https://colab.research.google.com/drive/1Zl3zSdli_epSfaoQ_HeBCuE6dkGWTowd).

# # Initial Imports

# In[1]:
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import webbrowser
# # Initialise Config

# In[2]:


from model import Config
cfg = Config()


# # Initialise Demo Transformer

# In[3]:


from model import DemoTransformer
# demo_transformer = DemoTransformer(Config).to(cfg.device)
print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")


# # Some Sanity Checking

# In[4]:


# reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
# demo_transformer.load_state_dict(reference_gpt2.state_dict(), strict=False)
# reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
# tokens = reference_gpt2.to_tokens(reference_text).to(cfg.device)
# demo_logits = demo_transformer(tokens)

# def get_log_probs(
#     logits: Float[Tensor, "batch posn d_vocab"],
#     tokens: Int[Tensor, "batch posn"]
# ) -> Float[Tensor, "batch posn-1"]:

#     log_probs = logits.log_softmax(dim=-1)
#     # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
#     log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

#     return log_probs_for_tokens


# pred_log_probs = get_log_probs(demo_logits, tokens)
# print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
# print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_transformer.cfg.d_vocab):4f}")
# print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

# test_string = '''The Total Perspective Vortex derives its picture of the whole Universe on the principle of'''
# for i in tqdm(range(100)):
#     test_tokens = reference_gpt2.to_tokens(test_string).to(cfg.device)
#     demo_logits = demo_transformer(test_tokens)
#     test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

# print(test_string)
print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")


# # Training Loop

# ## Create smaller model

# In[5]:


model_cfg = Config(
    d_model=512,
    n_heads=4,
    d_head=64,
    d_mlp=1024,
    n_layers=12,
    n_ctx=256,
    d_vocab= 50257
)
model = DemoTransformer(model_cfg)
print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")


# ## Initialise Training Args

# In[6]:


from train import TransformerTrainingArgs
args = TransformerTrainingArgs()


# ## Prep Dataset and Sanity Check

# In[7]:


from datasets import load_dataset
tiny_stories = load_dataset('roneneldan/TinyStories',split='train')
print (tiny_stories)
print (tiny_stories[0])

print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")


# In[8]:


# reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
# print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
# print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")


# # In[9]:


# tokenized_dataset = tokenize_and_concatenate(tiny_stories,
#                                             reference_gpt2.tokenizer,
#                                             streaming=False,
#                                             max_length=model.cfg.n_ctx,
#                                             column_name="text",
#                                             add_bos_token=True,
#                                             num_proc=10)

# print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
# print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")


# # In[11]:


# # tokenized_dataset.save_to_disk('tokenized_tinystories')


# # In[10]:


# dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
# train_loader = DataLoader(
#     dataset_dict["train"],
#     batch_size=args.batch_size,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=False)
# print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
# print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")


# # In[11]:


# test_loader = DataLoader(
#     dataset_dict["test"],
#     batch_size=args.batch_size,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=False)
# print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
# print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")


# # In[12]:


# first_batch = train_loader.dataset[:args.batch_size]
# print(first_batch.keys())
# print(first_batch['tokens'].shape)
# print (first_batch)
# print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
# print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")


# # ## Loss fn

# # In[13]:


# def get_log_probs(
#     logits: Float[Tensor, "batch posn d_vocab"],
#     tokens: Int[Tensor, "batch posn"]
# ) -> Float[Tensor, "batch posn-1"]:

#     log_probs = logits.log_softmax(dim=-1)
#     # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
#     log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

#     return log_probs_for_tokens
# print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
# print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")


# # ## Initialise Training Loop Function and Train

# # In[23]:


# import importlib
# import train
# importlib.reload(train)
# from train import TransformerTrainer
# model = DemoTransformer(model_cfg).to(cfg.device)
# args = TransformerTrainingArgs()
# print (args.lr)
# args.max_steps_per_epoch = 10^9
# args.batch_size = 32
# args.epochs = 3
# print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
# print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")
# args.wandb_project = "demo_gpt2_may_24"
# print (args.max_steps_per_epoch)
# trainer = TransformerTrainer(args,
#                              model,
#                              dataset_dict,
#                              cfg,
#                              get_log_probs)
# trainer.train()
# print(f"Memory Allocated: {t.cuda.memory_allocated() / 1024**2:.2f} MB")
# print(f"Memory Reserved: {t.cuda.memory_reserved() / 1024**2:.2f} MB")



# # # Save Result

# # In[ ]:


# # t.save(model.state_dict(), 'gpt2_style_model_weights.pth')
# t.save(model.state_dict(), f'gpt2_style_model_weights_{model_cfg.n_layers}_layers_{args.epochs}_epochs_{model_cfg.d_model}_d_model.pth')


# # # Test Output Sampling

# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:
