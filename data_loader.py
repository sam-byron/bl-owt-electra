from datetime import datetime
from collections import OrderedDict
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
    return sum(len(ex["input_ids"]) for ex in ds)

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
        # build flat index in parallel by chunk file
        args = [(path, self.block_size) for path in self.chunk_paths]
        max_workers = min(mp.cpu_count(), len(args))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            lists_of_entries = executor.map(_index_for_path, args)
        # flatten into a single index_map
        self.index_map = list(chain.from_iterable(lists_of_entries))
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
        # if len(lst) < self.block_size:
        #     lst = lst + [self.pad_token_id] * (self.block_size - len(lst))
        return torch.tensor(lst, dtype=self.dtype)


def create_and_cache_splits(config):
    """Create multiple small train‐subsets plus one val/test, cache to disk."""
    cache_path   = config["cache_path"]
    splits_dir   = Path(cache_path) / "splits"
    splits_dir.mkdir(exist_ok=True, parents=True)
    splits_file  = splits_dir / "dataset_splits.json"

    # If already cached, just load and return
    if splits_file.exists():
        print("Dataset splits already exist, loading cached splits...")
        with open(splits_file, "r") as f:
            splits = json.load(f)
        return splits["train_subsets"], splits["val_paths"], splits["test_paths"]

    print("Creating new dataset splits…")
    chunk_paths       = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
    random.shuffle(chunk_paths)

    N                 = len(chunk_paths)
    train_frac        = config.get("train_frac", 0.9)
    val_frac          = config.get("val_frac",   0.05)
    subset_frac       = config.get("train_subset_frac", 0.035)

    # how many chunks total for train/val/test
    n_train_total     = int(train_frac * N)
    n_val             = int(val_frac   * N)
    # remainder is test
    n_test            = N - n_train_total - n_val

    train_all         = chunk_paths[:n_train_total]
    val_paths         = chunk_paths[n_train_total:n_train_total + n_val]
    test_paths        = chunk_paths[n_train_total + n_val:]

    # split train_all into small subsets of size subset_frac * N
    subset_size       = max(1, int(subset_frac * n_train_total))
    train_subsets     = [
        train_all[i: i + subset_size]
        for i in range(0, len(train_all), subset_size)
    ]

    # cache
    out = {
        "train_subsets": train_subsets,
        "val_paths":     val_paths,
        "test_paths":    test_paths,
        "created_at":    str(datetime.now()),
        "config": {
            "train_frac":        train_frac,
            "val_frac":          val_frac,
            "train_subset_frac": subset_frac,
            "total_chunks":      N,
            "n_train_subsets":   len(train_subsets),
            "subset_size":       subset_size,
        },
    }
    with open(splits_file, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Cached splits → {len(train_subsets)} train subsets, "
          f"val: {len(val_paths)}, test: {len(test_paths)}")
    return train_subsets, val_paths, test_paths


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


def data_loader(config, tokenizer, cache_path, train_subset_index=None):
    block_size = config["block_size"]
    batch_size = config["batch_size"]

    # Load or create cached splits
    subset_train_paths, val_paths, test_paths = create_and_cache_splits(config)
    if train_subset_index is None:
        # — skip any subsets we've already trained on (per run_mapping.json) —
        runs_base    = Path(cache_path) / "runs"
        mapping_file = runs_base / "run_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r") as mf:
                completed_map = json.load(mf)
            completed_indices = {int(k) for k in completed_map.keys()}
        else:
            completed_indices = set()

        # find indices not yet processed
        all_indices     = list(range(len(subset_train_paths)))
        available       = [i for i in all_indices if i not in completed_indices]
        if not available:
            print("All train subsets have been processed. Exiting.")
            sys.exit(100)

        # pick one remaining subset at random
        train_subset_index = random.choice(available)
    else:
        train_subset_index = int(train_subset_index)
    train_paths        = subset_train_paths[train_subset_index]
    # shuffle the train paths
    random.shuffle(train_paths)
    print(f"Using train subset {train_subset_index}/{len(subset_train_paths)}: {len(train_paths)} files")

    pad_id = tokenizer.pad_token_id
    print("Creating ChunkedDataset instances (with padding)...")
    train_ds = ChunkedDataset(train_paths, block_size=block_size, pad_token_id=pad_id)
    val_ds   = ChunkedDataset(val_paths,   block_size=block_size, pad_token_id=pad_id)
    test_ds  = ChunkedDataset(test_paths,  block_size=block_size, pad_token_id=pad_id)

    # Compute total tokens for each split
    total_tokens_train = compute_total_tokens(train_ds)
    total_tokens_val   = compute_total_tokens(val_ds)
    total_tokens_test  = compute_total_tokens(test_ds)
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
        batch_sampler=train_batch_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=collate_fn_with_mask,
        prefetch_factor=6,
    )

    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True,
                               collate_fn=collate_fn_with_mask, prefetch_factor=2, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True,
                               collate_fn=collate_fn_with_mask, prefetch_factor=2, drop_last=True)

    print(f"Data preparation complete. "
          f"Train files: {len(train_paths)}, "
          f"Val files: {len(val_paths)}, "
          f"Test files: {len(test_paths)}")

    return train_loader, val_loader, test_loader, collate_fn_with_mask, total_tokens_train, train_subset_index
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