from datetime import datetime
from collections import OrderedDict
import gc
import math
# from torch.utils.data import Dataset
import os
from pathlib import Path
import random
import glob
from functools import partial
from utils_mp import load_chunk, load_chunk_safe
from multiprocessing import Pool
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
from collator import Collator
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datasets import concatenate_datasets
from transformers import AutoTokenizer, get_scheduler, DataCollatorForWholeWordMask
import argparse
import json
from typing import Union
from typing import List, Iterator
from transformers import DataCollatorForLanguageModeling
import sys
from transformers import (
    ElectraConfig,
    ElectraForPreTraining,
    ElectraTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# helper to compute total real tokens in a Dataset
def compute_total_tokens(ds: Union[Dataset, object]) -> int:
    """
    If `ds` has an index_map of (path, seq_idx, start, end) tuples,
    sum up (end - start). Otherwise assume __getitem__ returns a dict
    with 'input_ids' and sum their lengths.
    """
    if hasattr(ds, "index_map"):
        return sum(end - start for (_, _, start, end) in ds.index_map)
    # fallback for map‐style datasets returning dicts
    return sum(len(ex) for ex in ds)

# helper to build index entries for one chunk file in parallel
def _index_for_path(args):
    path, block_size = args
    seqs = torch.load(path, map_location="cpu")
    entries = []
    for seq_i, seq in enumerate(seqs):
        length = len(seq)
        # We iterate up to length - 1. This elegantly ensures that any
        # created block will have at least one token and a subsequent
        # token to predict. It naturally skips sequences with length <= 1.
        for i in range(0, length - 1, block_size):
            start = i
            # The end of the slice is the minimum of the full sequence length
            # or the start of the next block.
            end = min(length, i + block_size)
            entries.append((path, seq_i, start, end))
    return entries

class ChunkedDataset(Dataset):
    # FIX 1: Change the default dtype to torch.long for compatibility with embedding layers.
    def __init__(self, chunk_paths, block_size, dtype=torch.long, pad_token_id=None, cache_size=30):
        # shuffle once
        self.chunk_paths = list(chunk_paths)
        # random.shuffle(self.chunk_paths)
        self.block_size   = block_size
        self.dtype        = dtype
        self.pad_token_id = pad_token_id or 0
        self.cache_size   = cache_size
        # path -> loaded list of sequences
        self._chunk_cache = OrderedDict()
        # build a flat index of every block
        self._build_index()

    def _build_index(self):
        """Create self.index_map = [ (path, seq_idx, start, end), ... ]"""
        # build flat index in parallel by chunk file using multiprocessing.Pool
        args = [(path, self.block_size) for path in self.chunk_paths]
        max_workers = min(mp.cpu_count(), len(args))
        self.index_map = []
        with mp.Pool(processes=max_workers) as pool:
            # imap_unordered streams back partial results as soon as they're done
            for entries in pool.imap_unordered(_index_for_path, args, chunksize=10):
                self.index_map.extend(entries)
        # Precompute the true length of each block (before padding)
        self.block_lengths = [end - start for (_, _, start, end) in self.index_map]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        path, seq_i, start, end = self.index_map[idx]
        # simple LRU cache
        if path not in self._chunk_cache:
            data = torch.load(path, map_location="cpu")
            self._chunk_cache[path] = data
            # evict oldest if over capacity
            if len(self._chunk_cache) > self.cache_size:
                self._chunk_cache.popitem(last=False)
        seq = self._chunk_cache[path][seq_i]
        sub = seq[start:end] if isinstance(seq, torch.Tensor) else seq[start:end]
        lst = sub.tolist() if isinstance(sub, torch.Tensor) else sub
        if len(lst) < self.block_size:
            lst = lst + [self.pad_token_id] * (self.block_size - len(lst))
        return torch.tensor(lst, dtype=self.dtype)


def create_and_cache_splits(config):
    """Create train/val/test splits once and cache them."""
    
    cache_path = config["cache_path"]
    # Create splits directory
    splits_dir = Path(cache_path) / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    # Check if splits already exist
    splits_file = splits_dir / "dataset_splits.json"
    if splits_file.exists():
        print("Dataset splits already exist, loading cached splits...")
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        return splits['train_paths'], splits['val_paths'], splits['test_paths']
    
    # Create new splits
    print("Creating new dataset splits...")
    chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
    
    # Shuffle once and save the order
    random.shuffle(chunk_paths)
    
    train_frac = config.get("train_frac", 0.85)
    val_frac   = config.get("val_frac", 0.05)
    
    N = len(chunk_paths)
    idx1 = int(train_frac * N)
    idx2 = int((train_frac + val_frac) * N)

    train_paths = chunk_paths[:idx1]
    val_paths   = chunk_paths[idx1:idx2]
    test_paths  = chunk_paths[idx2:]
    
    # Cache the splits
    splits = {
        'train_paths': train_paths,
        'val_paths': val_paths,
        'test_paths': test_paths,
        'created_at': str(datetime.now()),
        'config': {
            'train_frac': train_frac,
            'val_frac': val_frac,
            'total_chunks': N
        }
    }
    
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Cached dataset splits to {splits_file}")
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    return train_paths, val_paths, test_paths

class TokenBudgetBatchSampler(BatchSampler):
    """
    A BatchSampler that creates batches of indices where the total number of
    tokens (based on sample lengths) does not exceed a specified budget.

    To minimize padding and wasted computation, it sorts samples by length
    before creating batches.
    """
    def __init__(self, lengths: List[int], max_tokens: int, shuffle: bool = True):
        """
        Args:
            lengths: A list of integers representing the length of each sample
                     in the dataset.
            max_tokens: The maximum number of tokens allowed in a single batch.
            shuffle: If True, the order of batches is shuffled at the beginning
                     of each epoch.
        """
        if not isinstance(lengths, list) or not all(isinstance(l, int) for l in lengths):
            raise TypeError("lengths must be a list of integers.")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.batches = self._create_batches()

    def _create_batches(self) -> List[List[int]]:
        # Create a list of (index, length) tuples and sort by length
        indices_with_lengths = sorted(enumerate(self.lengths), key=lambda x: x[1])

        batches = []
        current_batch = []
        current_token_count = 0

        for index, length in indices_with_lengths:
            if length > self.max_tokens:
                print(f"Warning: Sample {index} with length {length} is larger than max_tokens "
                      f"{self.max_tokens} and will be skipped.")
                continue

            if current_token_count + length > self.max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_token_count = 0

            current_batch.append(index)
            current_token_count += length

        if current_batch:
            batches.append(current_batch)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            random.shuffle(self.batches)
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)

# Update the data_loader function to use PreloadedDataset
def data_loader(config, tokenizer, cache_path):
    block_size = config["block_size"]
    batch_size = config["batch_size"]

    # Load or create cached splits
    train_paths, val_paths, test_paths = create_and_cache_splits(config)

    pad_id = tokenizer.pad_token_id
    print("Creating ChunkedDataset instances...")
    train_ds = ChunkedDataset(train_paths, block_size=block_size, pad_token_id=pad_id)
    val_ds   = ChunkedDataset(val_paths,   block_size=block_size, pad_token_id=pad_id)
    test_ds  = ChunkedDataset(test_paths,  block_size=block_size, pad_token_id=pad_id)

    # Compute total tokens for each split
    total_tokens_train = sum(train_ds.block_lengths)
    total_tokens_val   = sum(val_ds.block_lengths)
    total_tokens_test  = sum(test_ds.block_lengths)
    print(f"Total tokens → train: {total_tokens_train:,}, val: {total_tokens_val:,}, test: {total_tokens_test:,}")

    # Print length of datasets
    print(f"Train dataset length: {len(train_ds)}")
    print(f"Val dataset length: {len(val_ds)}")
    print(f"Test dataset length: {len(test_ds)}")

    # build the base MLM collator
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # wrap it so we also add an attention_mask
    def collate_fn_with_mask(examples):
        batch = base_collator(examples)
        batch["attention_mask"] = (batch["input_ids"] != tokenizer.pad_token_id).long()
        return batch

    # build dynamic, token‐based training batches
    max_tokens = config.get("max_tokens", config["block_size"] * config["batch_size"])
    print(f"Creating DataLoader with dynamic token batching (max_tokens={max_tokens})...")
    lengths = train_ds.block_lengths

    train_batch_sampler = TokenBudgetBatchSampler(
        lengths=lengths, 
        max_tokens=max_tokens, 
        shuffle=True
    )

    
    print(f"  → {len(train_batch_sampler)} batches, up to {max_tokens} tokens each")
    train_loader = DataLoader(
        train_ds,
        # batch_sampler=train_batch_sampler,
        batch_size=config["batch_size"],
        num_workers=12,
        pin_memory=True,
        collate_fn=collate_fn_with_mask,
        prefetch_factor=6,
    )

    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True,
                               collate_fn=collate_fn_with_mask, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True,
                               collate_fn=collate_fn_with_mask, drop_last=True)

    print(f"Data preparation complete. "
          f"Train files: {len(train_paths)}, "
          f"Val files: {len(val_paths)}, "
          f"Test files: {len(test_paths)}")

    return train_loader, val_loader, test_loader, collate_fn_with_mask, total_tokens_train
    # return val_loader, val_loader, test_loader, collate_fn, total_tokens_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iter Data Loader Script")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    block_size = config["block_size"]
    batch_size = config["batch_size"]

    # Load or create cached splits
    train_paths, val_paths, test_paths = create_and_cache_splits(config)