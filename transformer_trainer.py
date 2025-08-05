import glob
import os
import torch
import json
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, get_scheduler
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.local_sgd import LocalSGD
# from iter_data_loader import iter_data_loader
from data_loader import data_loader
from evaluation import evaluate_perplexity, create_test_subset
import argparse
import time
import torch.distributed as dist
import traceback
from datetime import timedelta
# import torch.distributed as dist

class EmptyDataset(Dataset):
    def __len__(self): 
        return 0
    def __getitem__(self, idx): 
        raise IndexError

# @torch.compile
def build_model(config):
    model_config = BertConfig(
        vocab_size=config["vocab_size"],
        max_position_embeddings=config["n_positions"],
        hidden_size=config["n_embed"],
        num_hidden_layers=config["n_layer"],
        num_attention_heads=config["n_head"],
        intermediate_size=config["n_embed"] * 4,
    )
    model = BertForMaskedLM(model_config)

    return model

import time
import datetime

def train_loop(
    accelerator,
    model,
    tokenizer,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    config,
    checkpoint_path,
    start_epoch,
    total_tokens_train,
    train_subset_index,
    tqdm_offset=0
):
    num_epochs = config["num_epochs"]
    if start_epoch >= num_epochs:
        print("Training already completed. Exiting.")
        return

    # 1) Compute number of steps and total tokens for ETA
    steps_per_epoch = len(train_loader)
    total_tokens = total_tokens_train

    last_checkpoint_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        processed_tokens = 0
        epoch_start = time.time()

        # Only the main process draws the bars
        is_main = accelerator.is_main_process

        # Token‐based progress bar
        token_bar = tqdm(
            total=total_tokens,
            unit="tok",
            unit_scale=True,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=False,
            position=tqdm_offset,
            disable=not is_main,
        )
        # Loss bar (steps)
        loss_bar = tqdm(
            total=steps_per_epoch,
            desc="Loss",
            leave=False,
            position=tqdm_offset + 1,
            bar_format="{l_bar}{bar}| {postfix}",
            disable=not is_main,
        )
        with LocalSGD(accelerator=accelerator, model=model, local_sgd_steps=8, enabled=True) as local_sgd:
            for step, batch in enumerate(train_loader):
                # --- ensure every GPU sees the same sequence‐length before any collectives ---
                # pad each tensor in batch to the global max length across processes
                for key in ("input_ids", "attention_mask", "labels"):
                    batch[key] = accelerator.pad_across_processes(batch[key], dim=1)

                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        loss = outputs.loss

                        accelerator.backward(loss)
                         # Clip gradients to prevent them from exploding, a common cause of NaNs.
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        local_sgd.step()

                        # Gather and average loss across GPUs
                        loss = accelerator.gather(loss).mean()
                        if loss.isnan() or loss.isinf():
                            # Print a warning and skip this step
                            if is_main:
                                print(f"Warning: NaN or Inf loss encountered at step {step} in epoch {epoch+1}. Skipping this step.")
                            continue
                        else:
                            total_loss += loss.item()
                        avg_loss = total_loss / (step + 1)
                    # print("done accumulating gradients")
                    # Count only real tokens on this shard
                    # accelerator.wait_for_everyone()
                    real_tokens = accelerator.gather(batch["attention_mask"]).sum().item()
                    processed_tokens += real_tokens
                    # print("Done gathering real tokens")
                    if is_main:
                        # print("updating bars")
                        token_bar.update(real_tokens)
                        # Update loss bar
                        loss_bar.update(1)
                        loss_bar.set_postfix({
                            "loss": f"{loss:.4f}",
                            "avg_loss": f"{avg_loss:.4f}"
                        })
                        # Update token bar postfix with tok/s and ETA
                        elapsed = time.time() - epoch_start
                        tok_per_sec = processed_tokens / elapsed if elapsed > 0 else 0
                        remaining = max(total_tokens - processed_tokens, 0)
                        eta_sec = remaining / tok_per_sec if tok_per_sec > 0 else 0
                        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                        token_bar.set_postfix({
                            "tok/s": f"{tok_per_sec:.0f}",
                            "ETA": eta_str
                        })
                        # print("Done updating bars")

                    # ——— Periodic checkpointing every 30 min ———
                    current_time = time.time()
                    if current_time - last_checkpoint_time >= 10 * 60:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
                            unwrapped = accelerator.unwrap_model(model)
                            unwrapped.save_pretrained(checkpoint_path)
                            tokenizer.save_pretrained(checkpoint_path)  # Save tokenizer to cache path
                            print("[Checkpoint] Saved model state and optimizer")
                        accelerator.wait_for_everyone()
                        last_checkpoint_time = current_time

            token_bar.close()
            loss_bar.close()

            # End‑of‑epoch summary
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1} completed in {datetime.timedelta(seconds=int(epoch_time))} — "
                f"Final Avg Loss {avg_loss:.4f}"
            )

            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir=checkpoint_path)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)  # Save tokenizer to cache path
            print("[Checkpoint] Saved model state and optimizer")



def main():
    print(torch._dynamo.list_backends())
    
    # 2) pin this process to the correct CUDA device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # grab LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT from torchrun/accelerate launch
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    accelerator = Accelerator()
    accelerator.even_batches = False  # Disable "even batches" enforcement since we use a batch_sampler without fixed batch_size

    parser = argparse.ArgumentParser(description="Chatbot Training Script with Accelerate")
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Directory to save to or load from"
    )

    parser.add_argument(
        "--tqdm_offset", type=int, default=0,
        help="line offset for tqdm so multiple bars don't collide"
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    checkpoint_path = os.path.join(config["cache_path"], "checkpoint.pt")
    # decide checkpoint dir: resume existing or create new
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    # tokenizer.pad_token = tokenizer.eos_token

    # get loaders and which subset‐index we're on
    train_loader, val_loader, test_loader, collate_fn, total_tokens_train, train_subset_index = \
        data_loader(config, tokenizer, config["cache_path"])
    if args.checkpoint_path:
        if not os.path.isdir(args.checkpoint_path):
            raise ValueError(f"Invalid --checkpoint_path for resume: {args.checkpoint_path}")
        checkpoint_path = args.checkpoint_path
        print(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        # ensure run mapping directory exists and record subset-index → run_dir
        runs_dir = os.path.join(config["cache_path"], "runs")
        os.makedirs(runs_dir, exist_ok=True)
        mapping_file = os.path.join(runs_dir, "run_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, "r") as mf:
                mapping = json.load(mf)
        else:
            mapping = {}
        # determine and register this run's folder for subset
        run_dir = os.path.join(runs_dir, str(train_subset_index))
        os.makedirs(run_dir, exist_ok=True)
        mapping[str(train_subset_index)] = run_dir
        checkpoint_path = run_dir
        with open(mapping_file, "w") as mf:
            json.dump(mapping, mf, indent=2)

   

    # Build the model
    model = build_model(config)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
   
    model, optimizer, train_loader, test_loader, val_loader, batch_sampler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, val_loader, train_loader.batch_sampler
    )
    # model = torch.compile(model, backend="inductor")
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["num_epochs"] * len(train_loader),
    )

    # check if checkpoint path contains a file ending .bin


    # check if checkpoint path contains a file ending .bin
    bin_files = glob.glob(os.path.join(checkpoint_path, "*.bin"))
    if checkpoint_path and os.path.exists(checkpoint_path) and bin_files:
        print(f"Loading checkpoint from {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")  
    start_epoch = 0

    train_loop(
        accelerator,
        model,
        tokenizer,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config,
        checkpoint_path,
        start_epoch,
        total_tokens_train,
        train_subset_index,
        args.tqdm_offset,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise