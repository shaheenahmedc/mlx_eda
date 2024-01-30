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
from tqdm.notebook import tqdm
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
from datasets import load_dataset
import sentencepiece as spm
from datasets import concatenate_datasets

class EncoderDecoderDataset(t.utils.data.Dataset):
    def __init__(self, input_data, sp_tokenizer_model, config):
        self.column_names = ['prompt', 'response']
        self.data = input_data
        self.tokenizer = sp_tokenizer_model
        self.cfg = config
        
    def __len__(self):
        return (len(self.data))
    
    def __getitem__(self, idx):
        row = self.data[idx]
        # model_input = (str(self.tokenizer.bos_id()) 
        #                + ', ' + self.tokenizer.encode_as_ids(row['prompt'])
        #                + ', ' + str(self.tokenizer.eos_id()))
    
        # ground_truth = (str(self.tokenizer.bos_id()) 
        #                 + ', ' + self.tokenizer.encode_as_ids(row['response'])
        #                 + ', ' + str(self.tokenizer.eos_id()))
        
        # Encode the prompt and response into lists of token ids
        model_input_ids = [self.tokenizer.bos_id()] + self.tokenizer.encode_as_ids(row['prompt']) + [self.tokenizer.eos_id()]
        ground_truth_ids = [self.tokenizer.bos_id()] + self.tokenizer.encode_as_ids(row['response']) + [self.tokenizer.eos_id()]

        # Limit input sequences to a maximum length of n_ctx
        max_seq_length = self.cfg.n_ctx
        if len(model_input_ids) > max_seq_length:
            model_input_ids = model_input_ids[:max_seq_length]
        if len(ground_truth_ids) > max_seq_length:
            ground_truth_ids = ground_truth_ids[:max_seq_length]
        
        # Convert the lists of token ids to tensors
        tensor_model_input = t.tensor(model_input_ids, dtype=t.long)
        tensor_ground_truth = t.tensor(ground_truth_ids, dtype=t.long)

        return {
            'model_input': model_input_ids,
            'ground_truth': ground_truth_ids,
            'tensor_model_input': tensor_model_input,
            'tensor_ground_truth': tensor_ground_truth
        }
        
    def collate_fn(self, batch):
        # This should really be using [PAD], but easier to directly add 0s, as 0 is PAD ID from tokenizer
        input_pad = t.nn.utils.rnn.pad_sequence([item['tensor_model_input'] for item in batch], batch_first=True, padding_value=0)
        label_pad = t.nn.utils.rnn.pad_sequence([item['tensor_ground_truth'] for item in batch], batch_first=True, padding_value=0)
        
        return {
            'model_inputs': [item['model_input'] for item in batch],
            'ground_truths': [item['ground_truth'] for item in batch],
            'tensor_model_input': input_pad,
            'tensor_ground_truth': label_pad
        }