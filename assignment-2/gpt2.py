""" A simple GPT-2 style transformer model in PyTorch.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


GPT2_PRESETS = {
    "gpt2-tiny":  dict(num_layers=6,  num_heads=6,  model_dim=384,  feed_forward_hidden_dim=4*384),
    "gpt2-small": dict(num_layers=12, num_heads=12, model_dim=768,  feed_forward_hidden_dim=4*768),
    "gpt2-medium":dict(num_layers=24, num_heads=16, model_dim=1024, feed_forward_hidden_dim=4*1024),
    "gpt2-large": dict(num_layers=36, num_heads=20, model_dim=1280, feed_forward_hidden_dim=4*1280),
    "gpt2-xl":    dict(num_layers=48, num_heads=25, model_dim=1600, feed_forward_hidden_dim=4*1600),
    "gpt2-2.7B":  dict(num_layers=60, num_heads=30, model_dim=1920, feed_forward_hidden_dim=4*1920),
    "gpt2-7.3B":  dict(num_layers=72, num_heads=30, model_dim=2880, feed_forward_hidden_dim=4*2880),
    "gpt2-12.2B": dict(num_layers=84, num_heads=36, model_dim=3456, feed_forward_hidden_dim=4*3456),
}


class GPT2Config:
    """ GPT-2 model configuration.
    """
    def __init__(
        self,
        vocab_size=50257,
        max_seq_len=1024,
        num_layers=6,
        num_heads=8,
        model_dim=512,
        feed_forward_hidden_dim=2048,
        dropout=0.1
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.feed_forward_hidden_dim = feed_forward_hidden_dim or 4 * model_dim
        self.dropout = dropout


def register_module_grad_nvtx_hooks(module, name):
    def pre_hook(module, grad_output):
        torch.cuda.nvtx.range_push(f"{name} grad")
    
    def post_hook(module, grad_input, grad_output):
        torch.cuda.nvtx.range_pop()

    module.register_full_backward_pre_hook(pre_hook)
    module.register_full_backward_hook(post_hook)


def get_module_num_grad_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class GPT2Block(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.model_dim = config.model_dim
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads

        self.ln_1 = nn.LayerNorm(config.model_dim)
        self.ln_2 = nn.LayerNorm(config.model_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=config.model_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(config.model_dim, config.feed_forward_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feed_forward_hidden_dim, config.model_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, attn_mask=None):
        x_norm = self.ln_1(x)
        attn_output, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            need_weights=False,
            is_causal=True
        )
        x = x + attn_output

        x_norm = self.ln_2(x)
        feed_forward_output = self.feed_forward(x_norm)
        x = x + feed_forward_output
        return x


class GPT2Model(nn.Module):

    def __init__(self, config: GPT2Config, profile_grads: bool = False):
        super().__init__()
        assert config.model_dim % config.num_heads == 0, \
            "model_dim must be divisible by num_heads"

        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.model_dim)

        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        # create hooks
        if profile_grads:
            register_module_grad_nvtx_hooks(self.token_embed, "token_embed")
            register_module_grad_nvtx_hooks(self.pos_embed, "pos_embed")
            register_module_grad_nvtx_hooks(self.lm_head, "lm_head")
            register_module_grad_nvtx_hooks(self.ln_f, "ln_f")
            for idx, block in enumerate(self.blocks):
                register_module_grad_nvtx_hooks(block, f"block_{idx}")

    def get_num_params_per_layer(self):
        from pandas import DataFrame
        values = []

        values.append({"Name": "Token embedding", "No. Params": get_module_num_grad_params(self.token_embed)})
        values.append({"Name": "Position embedding", "No. Params": get_module_num_grad_params(self.pos_embed)})
        
        for idx, block in enumerate(self.blocks):
            values.append({"Name": f"Block {idx}", "No. Params": get_module_num_grad_params(block)})

        values.append({"Name": "LayerNorm final", "No. Params": get_module_num_grad_params(self.ln_f)})
        values.append({"Name": "LM head", "No. Params": get_module_num_grad_params(self.lm_head)})

        return DataFrame(values)


    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        token_embeddings = self.token_embed(input_ids)

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeddings = self.pos_embed(positions)

        x = token_embeddings + pos_embeddings
        x = self.dropout(x)

        attn_mask = torch.ones((seq_len, seq_len), device=device)
        attn_mask = torch.tril(attn_mask)
        # MHA wants True for "do not attend"
        attn_mask = attn_mask == 0

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=0, top_p=0.0, tokenizer=None):
        """
        Simple text generation loop, purely for demonstration.
        pruning code based on: https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
        """
        for _ in range(max_new_tokens):

            # pop first tokens if we're too long
            if input_ids.shape[1] >= self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            # generate next token from model
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature

            # prune logits based on top_k and top_p
            if top_k > 0:
                assert 0 < top_k <= self.config.vocab_size, "top_k should be in [1, vocab_size]"
                indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            
            assert 0.0 <= top_p <= 1.0, "top_p must be in [0, 1]"
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cdf = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cdf > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            sorted_logits[sorted_indices_to_remove] = float('-inf')
            logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

            # sample from remaining logits
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config = GPT2Config(
        vocab_size=50257,
        max_seq_len=128,
        num_layers=4,
        num_heads=4,
        model_dim=256,
        feed_forward_hidden_dim=1024,
        dropout=0.1
    )
    model = GPT2Model(config).to(device)
    x = torch.randint(0, config.vocab_size, (2, 16), device=device)
    logits = model(x)
    print("Logits shape:", logits.shape)

    start_ids = torch.randint(0, config.vocab_size, (1, 5), device=device)
    generated = model.generate(start_ids, max_new_tokens=5)
    print("Generated sequence:", generated)

    free, total = torch.cuda.mem_get_info(device)
    mem_used_gb = (total - free) / 1024**3
    print(f"GPU memory used: {mem_used_gb:.2f} GB")
