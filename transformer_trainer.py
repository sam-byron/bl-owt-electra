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
from transformers import (
    ElectraConfig,
    ElectraForPreTraining,
    ElectraTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    ElectraForMaskedLM,
)


class EmptyDataset(Dataset):
    def __len__(self): 
        return 0
    def __getitem__(self, idx): 
        raise IndexError

# 4. Build an ELECTRA config
    #    - embedding_size: size of the generator’s embeddings
    #    - hidden_size etc. define the discriminator
def build_model(config, tokenizer):
    gen_config = ElectraConfig.from_pretrained("google/electra-base-generator")
    disc_config = ElectraConfig.from_pretrained("google/electra-base-discriminator")

    # 5. Instantiate ELECTRA for pretraining
    generator     = ElectraForMaskedLM(gen_config)
    discriminator = ElectraForPreTraining(disc_config)

    return generator, discriminator

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
    disc,
    tqdm_offset=0,
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
        disc.train()
        total_loss = 0.0
        processed_tokens = 0
        epoch_start = time.time()

        # Join checkpoint path with "generator" and "discriminator"
        gen_path = os.path.join(checkpoint_path, f"generator")
        disc_path = os.path.join(checkpoint_path, f"discriminator")
        tokenizer_path = os.path.join(checkpoint_path, "tokenizer")
        os.makedirs(gen_path, exist_ok=True)
        os.makedirs(disc_path, exist_ok=True)
        os.makedirs(tokenizer_path, exist_ok=True)

        # Only the main process draws the bars
        is_main = accelerator.is_main_process

        # Token‐based progress bar
        token_bar = tqdm(
            total=len(train_loader),
            unit="batch",
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
        # with LocalSGD(accelerator=accelerator, model=model, local_sgd_steps=8, enabled=True) as local_sgd:
        for step, batch in enumerate(train_loader):
            # --- ensure every GPU sees the same sequence‐length before any collectives ---
            # pad each tensor in batch to the global max length across processes
            # for key in ("input_ids", "attention_mask"):
            #     batch[key] = accelerator.pad_across_processes(batch[key], dim=1)

            with accelerator.accumulate(model, disc):
                with accelerator.autocast():
                    mlm_labels = batch["labels"]
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        # For ElectraForPreTraining, the 'labels' argument should contain the original
                        # input_ids. The model uses these as the ground truth for the generator's MLM task.
                        # It handles the creation of the discriminator's binary labels internally.
                        labels=mlm_labels,
                    )
                    gen_loss = outputs.loss
                    gen_logits = outputs.logits

                    # Build corrupted inputs for discriminator
                    with torch.no_grad():
                        preds = torch.argmax(gen_logits, dim=-1)
                        corrupted = batch["input_ids"].clone()
                        mask_pos = mlm_labels != -100
                        corrupted[mask_pos] = preds[mask_pos]

                    # Discriminator forward + loss (RTD)
                    # labels: 1 if original, 0 if replaced → mask_pos==True
                    disc_labels = (~mask_pos).long()
                    disc_out = disc(
                        input_ids=corrupted,
                        attention_mask=batch["attention_mask"],
                        labels=disc_labels,
                    )
                    disc_loss = disc_out.loss
                        # Combined loss (you can scale generator loss if desired)
                    loss = disc_loss + gen_loss

                    # --- DEBUG: Print loss components ---
                    # if is_main and (loss.isnan() or loss.isinf() or loss < 0):
                    #     print(f"Problematic Loss Detected: {loss.item()}")
                    #     print(f"  - Generator Loss: {outputs.generator_loss.item()}")
                    #     print(f"  - Discriminator Loss: {outputs.discriminator_loss.item()}")
                    # --- END DEBUG ---

                    accelerator.backward(loss)
                        # Clip gradients to prevent them from exploding, a common cause of NaNs.
                    if accelerator.sync_gradients:
                        # This is the recommended way to clip gradients with Accelerate
                        # It unscales the gradients, clips them, and then scales them back.
                        accelerator.clip_grad_norm_(
                        list(model.parameters()) + list(disc.parameters()),
                        config["max_grad_norm"]
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    # local_sgd.step()
                    
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
                processed_tokens += 1
                # print("Done gathering real tokens")
                # if is_main:
                # print("updating bars")
                token_bar.update(1)
                # Update loss bar
                loss_bar.update(1)
                loss_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "avg_loss": f"{avg_loss:.4f}"
                })
                # Update token bar postfix with tok/s and ETA
                elapsed = time.time() - epoch_start
                tok_per_sec = processed_tokens / elapsed if elapsed > 0 else 0
                remaining = max(len(train_loader) - processed_tokens, 0)
                eta_sec = remaining / tok_per_sec if tok_per_sec > 0 else 0
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                token_bar.set_postfix({
                    "tok/s": f"{tok_per_sec:.0f}",
                    "ETA": eta_str
                })
                    # print("Done updating bars")
                # accelerator.wait_for_everyone()
                # ——— Periodic checkpointing every 30 min ———
                current_time = time.time()
                if current_time - last_checkpoint_time >= 10 * 60:
                    # accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
                        unwrapped_gen = accelerator.unwrap_model(model)
                        unwrapped_disc = accelerator.unwrap_model(disc)
                        unwrapped_gen.save_pretrained(gen_path)
                        unwrapped_disc.save_pretrained(disc_path)
                        tokenizer.save_pretrained(tokenizer_path)  # Save tokenizer to cache path
                        print("[Checkpoint] Saved generator, discriminatorm and tokenizer")
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

        # accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
            unwrapped_gen = accelerator.unwrap_model(model)
            unwrapped_disc = accelerator.unwrap_model(disc)
            unwrapped_gen.save_pretrained(gen_path)
            unwrapped_disc.save_pretrained(disc_path)
            tokenizer.save_pretrained(tokenizer_path)  # Save tokenizer to cache path
            print("[Checkpoint] Saved generator, discriminatorm and tokenizer")



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

    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=2)
    # accelerator = Accelerator()
    accelerator.even_batches = False  # Disable "even batches" enforcement since we use a batch_sampler without fixed batch_size

    parser = argparse.ArgumentParser(description="Chatbot Training Script with Accelerate")
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="",
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
    
    # 2. Initialize an ELECTRA tokenizer (you can train your own vocab too)
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")
    tokenizer.model_max_length = 512

    
    train_loader, val_loader, test_loader, collate_fn, total_tokens_train = \
                data_loader(config, tokenizer, config["cache_path"])
    if args.checkpoint_path:
        if not os.path.isdir(args.checkpoint_path):
            raise ValueError(f"Invalid --checkpoint_path for resume: {args.checkpoint_path}")
        checkpoint_path = args.checkpoint_path
        print(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint path provided, starting from scratch.")
   

    # Build the model
    model, disc = build_model(config, tokenizer)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
   
    model, optimizer, train_loader, test_loader, val_loader, disc = accelerator.prepare(
        model, optimizer, train_loader, test_loader, val_loader, disc
    )

    if accelerator.is_main_process:
        # —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # Print model architecture, parameter counts, and full size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        total_mb = total_bytes / (1024 ** 2)
        print("===== Model Summary =====", flush=True)
        print(model, flush=True)
        print(f"Total parameters:     {total_params:,}", flush=True)
        print(f"Trainable parameters: {trainable_params:,}", flush=True)
        print(f"Approx. model size:   {total_mb:.2f} MB", flush=True)
        print("==========================", flush=True)
        # —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # Print discriminator architecture, parameter counts, and full size
        total_disc_params = sum(p.numel() for p in disc.parameters())
        trainable_disc_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
        total_disc_bytes = sum(p.numel() * p.element_size() for p in disc.parameters())
        total_disc_mb = total_disc_bytes / (1024 ** 2)
        print("===== Discriminator Summary =====", flush=True)
        print(disc, flush=True)
        print(f"Total parameters:     {total_disc_params:,}", flush=True)
        print(f"Trainable parameters: {trainable_disc_params:,}", flush=True)
        print(f"Approx. model size:   {total_disc_mb:.2f} MB", flush=True)
        print("==================================", flush=True) 
    
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
        disc,
        args.tqdm_offset,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise