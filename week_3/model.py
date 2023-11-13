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




device = t.device("cuda" if t.cuda.is_available() else "cpu")


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty(cfg.n_ctx, cfg.d_model))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(
            self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch
        )


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_K = nn.Parameter(t.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_V = nn.Parameter(t.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_O = nn.Parameter(t.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        Keys = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head",
            )
            + self.b_K
        )

        Queries = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head",
            )
            + self.b_Q
        )
        Values = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head",
            )
            + self.b_V
        )
        Attention_Scores = einops.einsum(
            Queries,
            Keys,
            "batch seq_len_Q n_heads d_head, batch seq_len_K n_heads d_head -> batch n_heads seq_len_Q seq_len_K",
        )
        Attention_Scores_Masked_Scaled = self.apply_causal_mask(
            Attention_Scores / self.cfg.d_head**0.5
        )
        Attention_Scores_Masked_Scaled_Softmaxed = (
            Attention_Scores_Masked_Scaled.softmax(-1)
        )

        #         Z = einops.einsum(Attention_Scores_Masked_Scaled_Softmaxed, self.W_V, "batch seq_len_Q seq_len_K , batch seq_len_K n_heads d_head -> batch seq_len_Q n_heads d_head")
        Z = einops.einsum(
            Values,
            Attention_Scores_Masked_Scaled_Softmaxed,
            "batch seq_len_K n_heads d_head, batch n_heads seq_len_Q seq_len_K -> batch seq_len_Q n_heads d_head",
        )

        Attention_Out = (
            einops.einsum(
                Z,
                self.W_O,
                "batch seq_len_Q n_heads d_head, n_heads d_head d_model -> batch seq_len_Q d_model",
            )
            + self.b_O
        )

        return Attention_Out

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        key_by_query_ones = t.ones(
            attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device
        )
        mask = t.triu(key_by_query_ones, diagonal=1).bool()
        attn_scores.masked_fill(mask, self.IGNORE)
        return attn_scores


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty(cfg.d_model, cfg.d_mlp))
        self.W_out = nn.Parameter(t.empty(cfg.d_mlp, cfg.d_model))
        self.b_in = nn.Parameter(t.zeros(cfg.d_mlp))
        self.b_out = nn.Parameter(t.zeros(cfg.d_model))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        post_W_in = (
            einops.einsum(
                normalized_resid_mid,
                self.W_in,
                "batch seq_len d_model, d_model d_mlp -> batch seq_len d_mlp",
            )
            + self.b_in
        )

        post_activation = gelu_new(post_W_in)

        post_W_out = (
            einops.einsum(
                post_activation,
                self.W_out,
                "batch seq_len d_mlp, d_mlp d_model -> batch seq_len d_model",
            )
            + self.b_out
        )
        return post_W_out


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (
            residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps
        ).sqrt()

        residual = (residual - residual_mean) / residual_std
        return residual * self.w + self.b


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post


class Unembed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty(cfg.d_model, cfg.d_vocab))
        self.b_U = nn.Parameter(t.zeros(cfg.d_vocab), requires_grad=False)
        nn.init.normal_(self.W_U, std=self.cfg.init_range)

    def forward(
        self, resid_stream: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_vocab"]:
        Unembedding = (
            einops.einsum(
                resid_stream,
                self.W_U,
                "batch seq_len d_model, d_model d_vocab -> batch seq_len d_vocab",
            )
            + self.b_U
        )
        return Unembedding


class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Float[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits


class TransformerSampler:
    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=1000, verbose=False, **kwargs):
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how
        new tokens are chosen.
        """
        # SOLUTION
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)[0]

        for i in range(max_tokens_generated):
            # Get new logits (make sure we don't pass in more tokens than the model's context length)
            logits = self.model(input_ids[None, -self.cfg.n_ctx :])
            # We only take logits for the last token, because this is what we're sampling
            logits = logits[0, -1]
            # Get next token (as a tensor of size (1, 1) so we can concat it to input_ids)
            next_token = t.tensor(
                [TransformerSampler.sample_next_token(input_ids, logits, **kwargs)],
                device=device,
            )
            # Create new input ids string, with shape (1, old_seq_len + 1)
            input_ids = t.cat([input_ids, next_token], dim=-1)
            # Print out results, if required
            if verbose:
                print(self.tokenizer.decode(input_ids), end="\r")
            # If our new token was the end-of-text token, stop
            if next_token == getattr(self.tokenizer, "eos_token_id", None):
                break

        return self.tokenizer.decode(input_ids)

    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int = 0,
        verbose=False,
    ) -> List[Tuple[float, t.Tensor]]:
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how
        new tokens are chosen.
        """
        pass

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "seq_len d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None,
    ):
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (
            top_p != 0 and top_k != 0
        ), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(
                input_ids, logits, frequency_penalty
            )
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        out = logits.argmax().item()
        return out

    @staticmethod
    def apply_temperature(
        logits: Float[Tensor, "d_vocab"], temperature: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        # SOLUTION
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        freq_penalty: float,
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        # SOLUTION
        d_vocab = logits.size(0)
        id_freqs = t.bincount(input_ids, minlength=d_vocab)
        return logits - freq_penalty * id_freqs

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        # SOLUTION
        sampled_token = t.distributions.categorical.Categorical(logits=logits).sample()
        return sampled_token.item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        # SOLUTION
        top_k_logits, top_k_token_ids = logits.topk(k)
        # Get sampled token (which is an index corresponding to the list of top-k tokens)
        sampled_token_idx = t.distributions.categorical.Categorical(
            logits=top_k_logits
        ).sample()
        # Get the actual token id, as an int
        return top_k_token_ids[sampled_token_idx].item()

    @staticmethod
    def sample_top_p(
        logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1
    ) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
        # SOLUTION
        # Sort logits, and get cumulative probabilities
        logits_sorted, indices = logits.sort(descending=True, stable=True)
        cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
        # Choose which tokens to keep, in the set we sample from
        n_keep = t.searchsorted(cumul_probs, top_p, side="left").item() + 1
        n_keep = max(n_keep, min_tokens_to_keep)
        keep_idx = indices[:n_keep]
        keep_logits = logits[keep_idx]
        # Perform the sampling
        sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
        return keep_idx[sample].item()
