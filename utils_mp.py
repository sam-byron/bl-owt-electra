import os
import random
import torch
from itertools import islice
from multiprocessing import Pool, current_process
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from datasets import concatenate_datasets

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path="checkpoint.pt"):
     # if model is wrapped (DDP, DistributedDataParallel, etc.) it has .module
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}", flush=True)

def load_checkpoint(checkpoint_path="checkpoint.pt"):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    else:
        print("No checkpoint found. Starting from scratch.")
        return None
    
def batch_generator_sequential(dataset, chunk_size, max_samples=2**31):
    """Generate batches of samples from a non-streaming dataset more efficiently."""
    for i in range(0, min(len(dataset), max_samples), chunk_size):
        yield dataset[i:i + chunk_size]

def batch_generator(dataset, chunk_size, max_samples):
    """Generate batches of samples from a streaming dataset more efficiently."""
    iterator = iter(dataset)  # Create an iterator from the dataset
    num_samples = 0

    while num_samples < max_samples:
        # Use islice to fetch a batch of size `batch_size`
        batch = list(islice(iterator, chunk_size))
        if not batch:
            break  # Stop if there are no more samples
        yield batch
        num_samples += len(batch)

def fetch_batch(args):
    dataset, start, end = args
    return list(islice(dataset, start, end))

def batch_generator_parallel(dataset, batch_size, max_samples, num_workers, start_index=0):
    """Generate batches in parallel using multiprocessing, starting from a specific index."""
    # Calculate the total number of batches, starting from the given start_index
    total_batches = (max_samples + batch_size - 1) // batch_size
    args = [
        (dataset, start_index + i * batch_size, start_index + (i + 1) * batch_size)
        for i in range(total_batches)
    ]

    with Pool(num_workers) as pool:
        for batch in pool.imap(fetch_batch, args):
            yield batch

def tokenize_sample(sample, tokenizer):
    """Tokenize a single sample."""

    if not isinstance(sample, list) or not all(isinstance(x, str) for x in sample):
        raise ValueError(f"Expected list[str], got {type(sample)}")
    return tokenizer(sample, add_special_tokens=False, truncation=True)["input_ids"]  # Tokenize the sample and return the input IDs

def tokenize_samples(samples, tokenizer):
    """Tokenize a batch of samples."""
    if isinstance(samples, list):
        # If the samples are a list of strings, tokenize each string
        texts = samples
    else:
        raise ValueError(f"Unexpected samples format: samples")
    #     print(f"Using batch_encode_plus for better performance with large batches")
    texts = [text + tokenizer.eos_token for sublist in texts for text in sublist]  # Add EOS token to each text
    
    return tokenizer.encode(
        texts,
        add_special_tokens=False,
        truncation=True,
        padding=False,  # No padding for now, we will handle it later
        # return_tensors='pt'  # Return as PyTorch tensors
        )
def load_chunk(chunk_path):
    """Helper function to load a single chunk."""
    print(f"Loading chunk from {chunk_path}...", flush=True)
    return torch.load(chunk_path, map_location="cpu")

def load_chunk_safe(path):
    try:
        return load_chunk(path)
    except (RuntimeError, EOFError) as e:
        print(f"[Warning] failed to load {path}: {e}")
        return []

from concurrent.futures import ThreadPoolExecutor

# global thread pool for parallel disk writes
_SAVE_POOL = ThreadPoolExecutor(max_workers=64)

def process_and_save_chunk(arg, tokenizer):
    """
    Tokenize and save a single chunk.
    Now each chunk filename includes the target_rank so that different GPUs won't overwrite each other.
    """
    cache_path = []  # Initialize chunk_paths
    # tokenized_chunks = []
    
    sample_chunk, chunk_index, cache_path, tokenize_with_tokenizer = arg
    # sample_chunk = [sample["text"] + tokenizer.eos_token for sample in sample_chunk]
    sample_chunk = [sample["text"] for sample in sample_chunk]
    if not isinstance(sample_chunk, list) or not all(isinstance(t, str) for t in sample_chunk):
        print(f"Expected list[str], got {type(sample_chunk)}")
        raise ValueError(f"Expected list[str], got {type(sample_chunk)}")
    try:
        tokenized_chunk = tokenize_with_tokenizer(sample_chunk)  # Ensure the tokenizer is called to avoid lazy evaluation issues
        if not tokenized_chunk:
            raise ValueError("Tokenizer did not produce any output. Check the input or tokenizer implementation.")
    except Exception as e:
        print(f"Error tokenizing chunk {chunk_index}: {e}", flush=True)
        return None
    cache_path = os.path.join(cache_path, f"chunk{chunk_index}.pt")
    # print(f"Saving chunk {chunk_index}", flush=True)
    # torch.save(tokenized_chunk, cache_path)
    fut = _SAVE_POOL.submit(torch.save, tokenized_chunk, cache_path)
    fut.add_done_callback(lambda f: print(f"Saved chunk {chunk_index} → {cache_path}", flush=True))
    # print(f"Saved chunk {chunk_index}", flush=True)
    print(f"Tokenization complete for {len(tokenized_chunk)} chunks", flush=True)

    return fut

