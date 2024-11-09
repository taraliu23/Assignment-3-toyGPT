"""
Full definition of a GPT Language Model.
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

import time

from src.args import Config

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the GELU activation of the input x.
        """
        # --- TODO: start of your code ---
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        # --- TODO: end of your code ---
        raise NotImplementedError


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config: Config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.input_projection = nn.Linear(config.d_model, 3 * config.d_model)
        # output projection
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.gpt_seq_len, config.gpt_seq_len)).view(
                1, 1, config.gpt_seq_len, config.gpt_seq_len
            ),
        )
        self.n_head = config.n_head
        self.d_model = config.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality (d_model)
        batch_size, seq_len, d_model = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.input_projection(x).split(self.d_model, dim=2)
        k = k.view(batch_size, seq_len, self.n_head, d_model //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(batch_size, seq_len, self.n_head, d_model //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, seq_len, self.n_head, d_model //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(
            self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.res_dropout(self.output_projection(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.d_model, 4 * config.d_model),
                c_proj=nn.Linear(4 * config.d_model, config.d_model),
                act=GELU(),
                dropout=nn.Dropout(config.dropout),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(
            m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model"""

    def __init__(self, config: Config):
        super().__init__()
        assert config.n_digits is not None
        assert config.gpt_seq_len is not None
        self.seq_length = config.gpt_seq_len

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.n_digits, config.d_model),
                wpe=nn.Embedding(config.gpt_seq_len, config.d_model),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config)
                                for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.d_model),
            )
        )
        self.output_head = nn.Linear(
            config.d_model, config.n_digits, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        logger.info("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, config):
        """
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(decay))], "weight_decay": 0.1},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=config.lr, betas=(0.9, 0.95))
        return optimizer

    def forward(self, idx, targets=None):
        """
        The forward function of the GPT model processes a sequence of token indices `idx` through the GPT architecture, 
        generating token predictions and computing optional loss if target labels are provided.

        Parameters
        ----------
        idx: torch.Tensor
            Tensor of token indices of shape (batch_size, seq_len), where each token corresponds to a word/sub-word in 
            the vocabulary.
        targets: torch.Tensor, optional
            Tensor of target indices of the same shape as `idx`, used for computing the cross-entropy loss 
            if provided. If not provided, no loss is calculated.

        Returns
        -------
        logits: torch.Tensor
            Logits of shape (batch_size, seq_len, vocab_size) representing the unnormalized probabilities of each 
            token at each position in the sequence.
        loss: torch.Tensor, optional
            Cross-entropy loss if `targets` is provided, otherwise None.

        Hint: First, you can generate token embeddings using the input token indices and the model's token embedding layer. Similarly, generate position embeddings using the positions (ranging from 0 to the length of the sequence) and the position embedding layer.
        """
        # --- #TODO: start of your code ---
        total_start_time = time.time()  
        device = idx.device

        b, t = idx.size()

        # positional token
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) 

        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        logger.info(f"Total Forward pass time: {time.time() - total_start_time:.4f}s")  # Log the total time for the forward pass
        return logits, loss
    
        # --- TODO: end of your code ---

        # raise NotImplementedError

    @torch.no_grad()
    def inference(self, ids, max_new_tokens):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.

        Parameters
        ----------
        ids: torch.Tensor
            shape (batch_size, seq_len) giving the initial sequence to complete
        max_new_tokens: int
            number of tokens to generate on top of the input indices

        Returns
        -------
        ids: torch.Tensor
            shape (batch_size, seq_len + max_new_tokens) giving the completed sequence
        """
        self.eval()

        for _ in range(max_new_tokens):
            # --- TODO: start of your code ---
            logits, _ = self.forward(ids)

            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits, dim=-1, keepdim=True)  

            ids = torch.cat((ids, next_token), dim=1)  
            # --- TODO: end of your code ---
            pass

        return ids
