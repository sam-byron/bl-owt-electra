import math
from torch.utils.data import Dataset
import os
import random
import glob
from datasets import load_dataset
from functools import partial
from utils import save_checkpoint, load_checkpoint, batch_generator_sequential,  batch_generator, tokenize_sample, load_chunk, process_and_save_chunk
from multiprocessing import Pool
from itertools import chain
import torch
from torch.utils.data import DataLoader
from accelerate import  Accelerator
from collator import Collator
from concurrent.futures import ThreadPoolExecutor
import time
from datasets import concatenate_datasets

class ChatDataset(Dataset):
    def __init__(self, tokenized_texts, block_size=256):
        """
        Initialize the dataset with tokenized texts and block size.
        Precompute all chunks for index-based access.
        """
        self.tokenized_texts = tokenized_texts
        self.block_size = block_size
        self.chunks = self._create_chunks()

    def _create_chunks(self):
        """
        Precompute all chunks from the tokenized texts.
        Each chunk is of size `block_size`.
        """
        chunks = []
        for ids in self.tokenized_texts:
            for i in range(0, len(ids), self.block_size):
                chunk = ids[i : i + self.block_size]
                if len(chunk) == self.block_size:
                    chunks.append(chunk)
        return chunks

    def __len__(self):
        """
        Return the total number of chunks.
        """
        return len(self.chunks)

    def __getitem__(self, idx):
        """
        Return the chunk at the specified index.
        """
        return self.chunks[idx]

def prepare_open_web_text_data(config, tokenizer, cache_path, process_rank=None):

    print("Preparing data...")
    block_size = config["block_size"]
    batch_size = config["batch_size"]

    accelerator = Accelerator()

    # Ensure the cache folder exists
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print(f"Created cache folder: {cache_path}")

    # Collect cached chunk files that include the current process's rank.
    # Here we expect filenames like "chunk_..._rank_{process_rank}.pt"
    chunk_paths = glob.glob(os.path.join(cache_path, f"chunk_*_rank_{process_rank}.pt"))
    print(f"Found {len(chunk_paths)} cached chunks for rank {process_rank}.")

    if len(chunk_paths) == 0:
        print("No cached chunks found. Tokenizing and caching the dataset...")
        files = glob.glob("/home/sam-byron/.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/f3808c30e817981b845ec549c43e82bb467d8144/subsets/*")
        print(f"Found {len(files)} data files: {files}")
        hf_ds = None
        if accelerator.is_main_process:
            print("Loaded openwebtext dataset from HuggingFace...")
            hf_ds = load_dataset("openwebtext", streaming=False, trust_remote_code=True, num_proc=8, files=chunk_paths[0])
        accelerator.wait_for_everyone()
        num_ranks = accelerator.num_processes
        num_samples = config["num_samples"]
        print(f"Total number of samples: {num_samples}, Number of ranks: {num_ranks}")
        chunk_size = config["chunk_size"]

        tokenize_with_tokenizer = partial(tokenize_sample, tokenizer=tokenizer)

        print("Processing dataset in chunks...")
        # chunk_args = [
        #     (sample_chunk, i, cache_path, tokenize_with_tokenizer)
        #     for i, sample_chunk in enumerate(batch_generator_sequential(hf_ds, chunk_size, num_samples))
        # ]
        # Split the dataset based on process rank
        total_samples = num_samples
        samples_per_rank = total_samples / num_ranks
        start_idx = int(process_rank * samples_per_rank)
        end_idx = int(start_idx + samples_per_rank) if process_rank < num_ranks - 1 else total_samples
        print(f"Process {process_rank} will process samples from index {start_idx} to {end_idx}.")

        hf_ds_split = hf_ds.select(range(start_idx, end_idx))

        world_size = accelerator.num_processes  # total GPUs
        target_rank = accelerator.process_index
        chunk_args = [
            (sample_chunk, i, cache_path, tokenize_with_tokenizer, target_rank, world_size)
            for i, sample_chunk in enumerate(batch_generator_sequential(hf_ds_split, chunk_size, samples_per_rank))
        ]

        with Pool(58) as pool:
            chunk_paths = pool.map(process_and_save_chunk, chunk_args)

        print(f"Processed and saved {len(chunk_paths)} chunks.")

    chunk_paths = glob.glob(os.path.join(cache_path, "chunk_*.pt"))
    print(f"Found {len(chunk_paths)} cached chunks.")

    # If a process_rank is provided, only load chunks for that process.
    # if process_rank is not None:
    #     filtered_chunk_paths = [p for p in chunk_paths if f"rank_{process_rank}" in os.path.basename(p)]
    #     print(f"Process {process_rank} will load {len(filtered_chunk_paths)} chunks.")
    # else:
    #     filtered_chunk_paths = chunk_paths

    # with Pool(32) as pool:  # Reduce the number of processes to 32 for better performance
    #     tokenized_texts_chunks = pool.map(load_chunk, filtered_chunk_paths)
    # print(f"Loaded {len(tokenized_texts_chunks)} chunks.")
        # --- load all the .pt files serially to avoid nested multiprocessing hangs ---
    tokenized_texts_chunks = []
    chunk_paths = sorted(chunk_paths)  # Sort paths to ensure consistent order
    for p in chunk_paths:
        print(f"Loading chunk from {p}...")
        tokenized_texts_chunks.append(load_chunk(p))
    print(f"Loaded {len(tokenized_texts_chunks)} chunks.")

    # Flatten chunks and convert to tensors in batches for efficiency
    tokenized_texts = list(chain.from_iterable(tokenized_texts_chunks))
    print(f"Loaded {len(tokenized_texts)} samples from cache.")

    print("Shuffling tokenized texts...")
    random.shuffle(tokenized_texts)
    split_index = int(0.85 * len(tokenized_texts))
    train_texts = tokenized_texts[:split_index]
    test_texts = tokenized_texts[split_index:]

    print(f"Train texts: {len(train_texts)}, Test texts: {len(test_texts)}")

    train_dataset = ChatDataset(train_texts, block_size=block_size)
    test_dataset = ChatDataset(test_texts, block_size=block_size)
    print(f"Train dataset created of size: {len(train_dataset)}, Test dataset created of size: {len(test_dataset)}")

    def collate_fn(batch):
        pad_id = tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [
                seq.detach().clone().long() if isinstance(seq, torch.Tensor)
                else torch.tensor(seq, dtype=torch.long)
                for seq in batch
            ],
            batch_first=True,
            padding_value=pad_id,
        )
        attention_mask = (input_ids != pad_id).long()
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader, test_texts

def prepare_data(config, tokenizer, cache_path, accelerator):
    nw = min(os.cpu_count(), 8)

    # if accelerator.is_main_process:
    # Only rank 0 loads & tokenizes the raw data
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    num_samples = config["num_samples"]
    chunk_size = config["chunk_size"]
    num_chunks = math.ceil(num_samples / chunk_size)
        
        # nw = 60
        # how many workers we’ll use for loading chunks later
        
    # Ensure the cache folder exists
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print(f"Created cache folder: {cache_path}")

    # Check if chunk files exist
    file_path = "/home/sam-byron/.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/f3808c30e817981b845ec549c43e82bb467d8144/subsets/*"
    #Count the number of files in file_path
    num_files = len(sorted(glob.glob(file_path)))
    chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk_*.pt")))

    # while len(chunk_paths) < num_chunks:
    while len(chunk_paths) < 5:
        print("Tokenizing and caching the dataset...")
        # c4_subset = load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz")
        data_files = sorted(glob.glob(file_path))
        print(f"Found {len(data_files)} data files: {data_files}")
        # load each file into its 'train' split and concatenate
        
        
        max_samples = 0
        for file in data_files[num_files-1:]:
            print(f"Loading dataset from file: {file}")
            ds_list = []
            dd = load_dataset(
                "Skylion007/openwebtext",
                data_files=file,
                trust_remote_code=True,
                # num_proc=64,
                streaming=True,
            )
            # dd is a DatasetDict, extract the single 'train' split
            ds_list.append(dd["train"])
            hf_ds_subset = concatenate_datasets(ds_list)

            tokenize_with_tokenizer = partial(tokenize_sample, tokenizer=tokenizer)


            # max_samples = len(hf_ds_subset)
            for i, sample_chunk in enumerate(batch_generator(hf_ds_subset, chunk_size)):
                chunk_args = (sample_chunk, i + len(chunk_paths), cache_path, tokenize_with_tokenizer)
                process_and_save_chunk(chunk_args)
            # rank       = accelerator.process_index
            # world_size = accelerator.num_processes
         
            if my_args:
                num_workers = min(50 - 4, max(len(s[0]["text"]) for s in my_args))
                # Use threads (not processes) so we don’t nest Pools
                max_threads = 2  # at most two texts in-flight at once
                tokenized_chunks = []
        chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk_*.pt")))
    # else: 
    #     time.sleep(60)  # wait for rank 0 to finish tokenizing
    # barrier: make sure rank0 is done writing .pt files
    # if 
    # rank 0 has written out chunk_*.pt; others just wait for them
    # if accelerator.is_main_process:
        # chunk_paths = glob.glob(os.path.join(cache_path, "chunk_*.pt"))
    accelerator.wait_for_everyone()
    chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk_*.pt")))
    
    print(f"Found {len(chunk_paths)} cached chunks.")
    if not chunk_paths:
        raise RuntimeError(f"No cached chunks found in {cache_path}")

    if len(chunk_paths) == 0:
        raise RuntimeError(f"No cached chunks found in {cache_path}. Please run the tokenization step first.")
    chunk_paths = sorted(chunk_paths)
    # with Pool(min(nw, len(chunk_paths))) as pool:
    tokenized_texts_chunks = []
    for chunk_path in chunk_paths:
        tokenized_texts_chunks.append(load_chunk(chunk_path))
    print(f"Loaded {len(tokenized_texts_chunks)} chunks.")
    tokenized_texts = list(chain.from_iterable(tokenized_texts_chunks))
    tokenized_texts = list(map(torch.tensor, tokenized_texts))
    print(f"Loaded {len(tokenized_texts)} samples from cache.")

    print("Shuffling tokenized texts...")
    random.shuffle(tokenized_texts)
    split_index = int(0.75 * len(tokenized_texts))
    train_texts = tokenized_texts[:split_index]
    test_texts = tokenized_texts[split_index:]

    print(f"Train texts: {len(train_texts)}, Test texts: {len(test_texts)}")

    train_dataset = ChatDataset(train_texts, block_size=block_size)
    test_dataset = ChatDataset(test_texts, block_size=block_size)

    pad_id = tokenizer.pad_token_id
    collate_fn = Collator(pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader, test_texts, collate_fn