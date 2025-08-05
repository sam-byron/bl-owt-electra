import glob
import os
# Disable the tokenizers' parallelism to avoid deadlocks when using multiprocessing
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import gc
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from functools import partial
from utils_mp import tokenize_sample, load_chunk, process_and_save_chunk
# from utils import tokenize_sample, process_and_save_chunk
from multiprocessing import Pool
from itertools import chain
import torch
import multiprocessing as mp
from itertools import islice
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

# limit PyTorch to 1 thread per process as well
# torch.set_num_threads(1)

# Define the collate function at the top level to make it picklable by multiprocessing workers
def identity_collate_fn(examples):
    """An identity collate function that simply returns the batch as a list."""
    return examples
    
def chunked(iterable, size):
    it = iter(iterable)
    while True:
        try:
            # Attempt to get the next item from the iterator
            batch = list(islice(it, size))
            yield batch
        except StopIteration:
            # If StopIteration is raised, it means the iterator is exhausted
            break

def prepare_data(config, tokenizer, cache_path):

    # Ensure the cache folder exists
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print(f"Created cache folder: {cache_path}")

    # Check if chunk files exist
    # chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))

    chunk_size = config["chunk_size"]

    tokenize_with_tokenizer = partial(tokenize_sample, tokenizer=tokenizer)

    # build paths to your local dataset script and subset files
    script_path = os.path.join(os.path.dirname(__file__), "openwebtext.py")
    subset_files = glob.glob(os.path.join(cache_path, "subsets", "*"))

    ds = load_dataset(
        # script=script_path,              
        # name="Skylion007/openwebtext",                     # ← use your local dataset script
        # data_files={"train": subset_files},            # ← point at your already-downloaded subsets
        cache_dir=cache_path,
        trust_remote_code=True,
        streaming=False,
        split="train",
        num_proc=max(1, mp.cpu_count() - 4),
        path=script_path
    )

    # wrap the HuggingFace streaming IterableDataset in a PyTorch DataLoader
    # to parallelize I/O with num_workers > 1
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        ds,
        batch_size=chunk_size,
        num_workers=min(6, mp.cpu_count()),
        collate_fn=identity_collate_fn,  # identity: list of raw examples
        # shuffle=True,  # shuffle the dataset to ensure randomness
    )
    
    # chunk_idx = len(chunk_paths)
    chunk_idx = 0
    # chunks = chunked(stream, chunk_size)
    
    pool = Pool(processes=min(mp.cpu_count(), 100))
    # now iterate batches of size chunk_size in parallel
    for chunk_idx, chunk in enumerate(dataloader):
        print(f"Appending chunk {chunk_idx}, with {len(chunk)} examples")
        chunk_arg = (chunk, chunk_idx, cache_path, tokenize_with_tokenizer)
        # pool.apply(process_and_save_chunk, args=(chunk_arg, tokenizer))
        pool.apply_async(process_and_save_chunk,
                             args=(chunk_arg, tokenizer))
        if len(chunk) == 0:
            print(f"Empty chunk encountered at chunk index {chunk_idx}, stopping processing.")
            break
        del chunk  # free memory
        gc.collect()  # force garbage collection to free memory

    # Wait for all worker processes to finish
    pool.close()
    pool.join()
    return 1

def check_chunk_file(path):
    """Check if a single chunk file is valid. Returns (path, is_valid, error_msg)"""
    try:
        torch.load(path, map_location="cpu")
        return (path, True, None)
    except Exception as e:  # Catch all exceptions instead of specific ones
        return (path, False, str(e))

def sanitize_chunks_fast(config, max_workers=None):
    """Fast parallel sanitization using all available cores."""
    cache_path = config["cache_path"]
    chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
    print(f"Found {len(chunk_paths)} chunk files to check...")
    
    if not chunk_paths:
        print("No chunk files found!")
        return 0, 0
    
    # Use all cores if not specified
    if max_workers is None:
        max_workers = min(32, mp.cpu_count())
    
    print(f"Using {max_workers} parallel workers...")
    
    corrupted_files = []
    valid_files = []
    
    # Use ProcessPoolExecutor for true parallelism
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(check_chunk_file, path): path for path in chunk_paths}
        
        # Process results with progress bar
        for future in tqdm(as_completed(future_to_path), total=len(chunk_paths), desc="Checking files"):
            path, is_valid, error_msg = future.result()
            
            if is_valid:
                valid_files.append(path)
            else:
                print(f"\nCorrupted file: {os.path.basename(path)} - {error_msg}")
                corrupted_files.append(path)
    
    print(f"\nSanitization complete:")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\nRemoving {len(corrupted_files)} corrupted files...")
        
        # Sequential deletion is fast enough for small numbers
        for path in corrupted_files:
            try:
                os.remove(path)
                print(f"  Removed: {os.path.basename(path)}")
            except OSError as e:
                print(f"  Failed to remove {os.path.basename(path)}: {e}")
        print("Cleanup complete!")
    else:
        print("No corrupted files found!")
    
    return len(valid_files), len(corrupted_files)

def main():
    parser = argparse.ArgumentParser(description="Tokenize and prepare data for training")    
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    # sanitize flag
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Sanitize chunks in the cache directory"
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if args.sanitize:
        # Sanitize the chunks in the cache directory
        print(f"Sanitizing chunks...")
        valid_count, corrupted_count = sanitize_chunks_fast(config, 100)
        print(f"Valid chunks: {valid_count}, Corrupted chunks removed: {corrupted_count}")
        # tokenizer.save_pretrained(config["cache_path"])  # Save tokenizer to cache path
        return
    
    # tokenizer.save_pretrained(config["cache_path"])  # Save tokenizer to cache path
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer.pad_token = tokenizer.eos_token
    prepare_data(
        config, tokenizer, config["cache_path"]
    )

if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
     # Force the start method to 'spawn' to avoid deadlocks with transformers tokenizers
    # This is crucial for robust multiprocessing with complex libraries.
    mp.set_start_method("spawn", force=True)
    main()