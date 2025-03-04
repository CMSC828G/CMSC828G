""" Dataset class for Wikitext-103. See here: https://huggingface.co/datasets/Salesforce/wikitext
    Uses the GPT-2 tokenizer from tiktoken for tokenizing.
"""
from typing import Callable, Literal
import os
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class WikiTextDataset(Dataset):
    def __init__(self, data_root: os.PathLike, tokenizer: Callable, seq_len: int = 1024, cache: bool = True, split: Literal["train", "valid", "test"] = "train"):
        super().__init__()
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"Dataset directory not found: {data_root}")

        # if wiki.train.tensors exists, load it. Otherwise read in wiki.train.tokens and tokenize it.
        self.raw_data_path = os.path.join(data_root, f"wiki.{split}.tokens")
        self.tensor_data_path = os.path.join(data_root, f"wiki.{split}.tensors")

        if cache and os.path.exists(self.tensor_data_path):
            self.tokens = torch.load(self.tensor_data_path)
        else:
            self.tokens = self._load_and_tokenize(self.raw_data_path, tokenizer)
            self.tokens = torch.tensor(self.tokens, dtype=torch.long)
            if cache:
                torch.save(self.tokens, self.tensor_data_path)
        
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.n_vocab
        self.seq_len = seq_len
        self.window_size = self.seq_len + 1
        self.num_sequences = len(self.tokens) // self.window_size

    def _load_and_tokenize(self, file_path: str, tokenizer: Callable):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return tokenizer.encode(text)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.window_size
        end = start + self.window_size
        chunk = self.tokens[start:end]

        x = chunk[:-1].clone().detach().to(torch.long)
        y = chunk[1:].clone().detach().to(torch.long)
        return x, y


def get_wikitext103_collate_fn(batch):
    x, y = zip(*batch)
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return x, y


def get_wikitext103_dataloader(
    file_path: str,
    tokenizer: Callable,
    cache: bool = True,
    split: Literal["train", "valid", "test"] = "train",
    seq_len: int = 1024,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    distributed: bool = False,
):
    dataset = WikiTextDataset(file_path, tokenizer, seq_len=seq_len, cache=cache, split=split)
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and (sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=get_wikitext103_collate_fn
    )
    return loader
