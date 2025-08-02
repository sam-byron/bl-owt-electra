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
    total_tokens_train
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
            position=0,
            disable=not is_main,
        )
        # Loss bar (steps)
        loss_bar = tqdm(
            total=steps_per_epoch,
            desc="Loss",
            leave=False,
            position=1,
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


    # initialize process group with explicit device_id
    # dist.init_process_group(
    #     backend="nccl",
    #     init_method=f"tcp://{master_addr}:{master_port}",
    #     world_size=world_size,
    #     rank=local_rank,
    #     timeout=timedelta(minutes=10),
    #     # device_id=local_rank,
    #     device_id=torch.device(f"cuda:{local_rank}"),   # <-- pass a torch.device
    # )
    
    # # You need to tell Accelerate not to “dispatch” the same batch to every rank when using an IterableDataset (or else split each global batch into per‐rank pieces). Two ways to do this:
    # # Option A) Disable dispatching entirely by passing dispatch_batches=False
    # dataloader_config = DataLoaderConfiguration(
    # dispatch_batches=False,  # Each process fetches its own batch
    # split_batches=True,       # Split fetched batches across processes
    # )
    # accelerator = Accelerator(dataloader_config=dataloader_config, gradient_accumulation_steps=1)
    # accelerator = Accelerator(dynamo_backend="eager")  # use inductor mode for dynamo
    # dataloader_config = DataLoaderConfiguration(
    #     split_batches=False,  # Each process fetches its own batch
    #     dispatch_batches=False,  # Disable dispatching to avoid splitting batches
    # )
    # accelerator = Accelerator(dataloader_config)
    accelerator = Accelerator()
    accelerator.even_batches = False  # Disable "even batches" enforcement since we use a batch_sampler without fixed batch_size

    parser = argparse.ArgumentParser(description="Chatbot Training Script with Accelerate")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    checkpoint_path = os.path.join(config["cache_path"], "checkpoint.pt")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    # tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader, test_loader, collate_fn, total_tokens_train = data_loader(config, tokenizer, config["cache_path"])
    # Build the GPT-2 model from scratch based on our config
    model = build_model(config)
    # model = torch.compile(model, backend="inductor")
    # Compile with TorchDynamo + Inductor in dynamic‐shape mode:
    # - dynamic=True tells Inductor to emit a single polymorphic kernel
    #   that can handle varying sequence lengths without recompiling each
    #   time a new input shape is seen.
    # - This avoids unbounded graph‐caching and GPU memory growth
    #   when training with multi‐GPU and variable padding.
    # model = torch.compile(model, backend="inductor", dynamic=True)
    # Compile with TorchDynamo + Inductor for maximum training throughput.
    # We use a static (fullgraph) compile since all batch‐shapes are now fixed.
    # model = torch.compile(
    #     model,
    #     backend="inductor",
    #     fullgraph=True,
    #     dynamic=False,
    # )

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

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")  
    start_epoch = 0

    train_loop(accelerator, model, tokenizer, train_loader, val_loader, optimizer, scheduler, config, checkpoint_path, start_epoch, total_tokens_train)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise