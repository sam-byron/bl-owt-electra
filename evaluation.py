import random
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from dataset import ChatDataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from accelerate import Accelerator
from safetensors.torch import load_file
import argparse
import json
from data_loader import data_loader as iter_data_loader
import math
import torch.nn.functional as F
from datasets import load_dataset


def build_model(config):
    model_config = GPT2Config(
        vocab_size=config["vocab_size"],
        n_positions=config["n_positions"],
        n_embd=config["n_embed"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        loss_type="cross_entropy", # Original loss used to train GPT-2
    )
    model = GPT2LMHeadModel(model_config)

    return model

def eval_ppl(model, dataloader, device, val_steps_per_epoch):
    import math, torch
    from torch.nn.functional import log_softmax

    model.eval()
    total_logp, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(
                dataloader, 
                desc="Evaluating Val Set Loss and Perplexity",
                total=val_steps_per_epoch,
                leave=True,):
            input_ids = batch["input_ids"].to(device)
            attn       = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask=attn).logits[:, :-1, :]
            labels = input_ids[:, 1:]

            logp = log_softmax(logits, dim=-1)
            token_logp = logp.gather(2, labels.unsqueeze(2)).squeeze(2)

            mask = attn[:, 1:]            # ignore padding
            total_logp   += (token_logp * mask).sum().item()
            total_tokens += mask.sum().item()

    nll = - total_logp / total_tokens
    return math.exp(nll)


# def evaluate_perplexity(model, val_loader, accelerator):
#     """Return (avg_nll, perplexity) on a *clean* validation loader."""
#     model.eval()
#     total_nll = 0.0          # negative log-likelihood (sum over tokens)
#     total_tokens = 0         # number of *active* tokens

#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc="Val perplexity", leave=True):
#             # Move to correct device
#             batch = {k: v.to(accelerator.device) for k, v in batch.items()}
#             input_ids      = batch["input_ids"]
#             attention_mask = batch["attention_mask"]
#             labels         = batch["labels"]

#             # Forward pass : HuggingFace returns mean loss over active tokens
#             outputs = model(input_ids=input_ids,
#                             attention_mask=attention_mask,
#                             labels=labels)
#             mean_loss = outputs.loss          # scalar on this process

#             # How many tokens contributed to that mean?
#             token_count = (labels != -100).sum()

#             # Gather across all processes
#             mean_loss, token_count = accelerator.gather(
#                 (mean_loss, token_count)
#             )

#             # Convert back to python numbers
#             token_count = token_count.sum().item()
#             nll = mean_loss.sum().item() * token_count   # undo the mean

#             total_nll    += nll
#             total_tokens += token_count

#     avg_nll = total_nll / total_tokens
#     perplexity = math.exp(avg_nll)
#     return avg_nll, perplexity


# def evaluate_perplexity(model, val_loader, accelerator, total_val_batches):
#     """Return (avg_nll, perplexity) on a *clean* validation loader."""
#     model.eval()
#     total_loss = 0.0          # negative log-likelihood (sum over tokens)
#     # total_tokens = 0         # number of *active* tokens

#     with torch.no_grad():
#         for batch in tqdm(
#             val_loader, 
#             desc="Evaluating Val Set Loss and Perplexity",
#             total=total_val_batches,
#             leave=True,):

#             # Move to correct device
#             # batch = {k: v.to(accelerator.device) for k, v in batch.items()}
#             input_ids      = batch["input_ids"]
#             attention_mask = batch["attention_mask"]
#             labels         = batch["labels"]

#             # Forward pass : HuggingFace returns mean loss over active tokens
#             outputs = model(input_ids=input_ids,
#                             attention_mask=attention_mask,
#                             labels=labels)
#             loss = outputs.loss          # scalar on this process

#             # How many tokens contributed to that mean?
#             # token_count = (labels != -100).sum()

#             # Gather across all processes
#             # loss, token_count = accelerator.gather(
#             #     (mean_loss, token_count)
#             # )
#             loss = accelerator.gather(loss)

#             mean_loss = loss.mean()  # mean loss across all processes

#             # Convert back to python numbers
#             # token_count = token_count.sum().item()
#             # nll = mean_loss.sum().item() * token_count   # undo the mean

#             # total_nll    += nll
#             total_loss += mean_loss  # sum the loss across processes
#             # total_tokens += token_count
#     # get number of devices
#     # total_val_batches = accelerator.num_processes * len(val_loader)  # total number of batches across all processes
#     avg_nll = total_loss / total_val_batches  # average loss over all batches
#     perplexity = math.exp(avg_nll)
#     return avg_nll, perplexity


# def evaluate_perplexity(model, val_loader, accelerator, total_val_batches):
#     """
#     • Correctly weights every *token* (not batch) even if padding / ragged.
#     • Works in DDP with Accelerate.
#     Returns (avg_nll, ppl) where avg_nll is in nats/token.
#     """
#     model.eval()
#     total_nll, total_tok = 0.0, 0

#     with torch.no_grad():
#         for batch in tqdm(
#             val_loader,
#             desc="Val-ppl",
#             total=total_val_batches,
#             disable=not accelerator.is_main_process,
#         ):
#             # batch = {k: v.to(accelerator.device) for k, v in batch.items()}
#             logits = model(**batch, labels=batch["input_ids"]).logits[:, :-1]
#             input_ids      = batch["input_ids"]
#             labels = batch["input_ids"][:, 1:]
#             mask   = batch["attention_mask"][:, 1:]   # align with labels

#             log_probs = F.log_softmax(logits, dim=-1)
#             token_logp = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)

#             # sum negative log-likelihood over *active* tokens
#             nll = -(token_logp * mask).sum()
#             ntok = mask.sum()

#             # gather across GPUs
#             nll, ntok = accelerator.gather_for_metrics((nll, ntok))
#             total_nll += nll.item()
#             total_tok += ntok.item()

#     avg_nll = total_nll / total_tok
#     ppl = math.exp(avg_nll)
#     return avg_nll, ppl

# def evaluate_perplexity(model, val_loader, accelerator, total_val_batches):
#     """
#     Evaluate the model's perplexity on the validation set.
#     Returns (avg_nll, perplexity) where avg_nll is in nats/token
#     """

#     test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#     encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

#     max_length = model.config.n_positions
#     stride = 512
#     seq_len = encodings.input_ids.size(1)

#     nll_sum = 0.0
#     n_tokens = 0
#     prev_end_loc = 0
#     for begin_loc in tqdm(range(0, seq_len, stride)):
#         end_loc = min(begin_loc + max_length, seq_len)
#         trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
#         input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#         target_ids = input_ids.clone()
#         target_ids[:, :-trg_len] = -100

#         with torch.no_grad():
#             outputs = model(input_ids, labels=target_ids)

#             # loss is calculated using CrossEntropyLoss which averages over valid labels
#             # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
#             # to the left by 1.
#             neg_log_likelihood = outputs.loss

#         # Accumulate the total negative log-likelihood and the total number of tokens
#         num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
#         batch_size = target_ids.size(0)
#         num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
#         nll_sum += neg_log_likelihood * num_loss_tokens
#         n_tokens += num_loss_tokens

#         prev_end_loc = end_loc
#         if end_loc == seq_len:
#             break

#     avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
#     ppl = torch.exp(avg_nll)

#     return avg_nll.item(), ppl.item()
    

def evaluate_perplexity(model, tokenizer, val_loader, accelerator, total_val_batches):
    """
    Evaluate the model's perplexity on the validation set using the provided val_loader.
    Returns (avg_nll, perplexity) where avg_nll is in nats/token
    """

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(
            val_loader, 
            desc="Evaluating validation perplexity", 
            total=total_val_batches,
            disable=not accelerator.is_main_process
        ):
            # Move batch to correct device (handled by accelerator)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # outputs.loss is already the mean CrossEntropyLoss over valid tokens
            batch_loss = outputs.loss
            
            # Count valid tokens (exclude -100 labels and padding)
            valid_token_mask = (labels != -100) & (labels != tokenizer.pad_token_id)
            batch_token_count = valid_token_mask.sum()
            
            # Gather metrics across all processes
            batch_loss, batch_token_count = accelerator.gather_for_metrics(
                (batch_loss, batch_token_count)
            )
            
            # Convert gathered tensors to scalars and accumulate
            if accelerator.is_main_process:
                # batch_loss is mean loss per token, so multiply by token count to get total NLL
                batch_nll = batch_loss.sum() * batch_token_count.sum() / len(batch_loss)
                total_nll += batch_nll.item()
                total_tokens += batch_token_count.sum().item()
    
    # Only compute final metrics on main process
    if accelerator.is_main_process:
        avg_nll = total_nll / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_nll)).item()
        return avg_nll, perplexity
    else:
        return 0.0, 0.0


def create_test_subset(test_texts, num_samples, block_size, batch_size, collate_fn, world_size, rank):
    """Create a subset of the test set and return a DataLoader."""
    # random.seed(42)  # Set seed for reproducibility
    subset_indices = random.sample(range(len(test_texts)), num_samples)  # Randomly sample indices
    test_subset_texts = [test_texts[i] for i in subset_indices]  # Create a subset of test_texts
    test_subset = ChatDataset(test_subset_texts, block_size=block_size)  # Create a ChatDataset for the subset
    test_subset_sampler = DistributedSampler(test_subset, num_replicas=world_size, rank=rank, shuffle=True)
    print(f"Test subset created with {len(test_subset_texts)} samples")

    # Create a DataLoader for the test subset
    test_subset_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        sampler=test_subset_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return test_subset_loader

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer using the model name from the config
    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize the Accelerator
    accelerator = Accelerator()
    

    model = build_model(config)

    model = accelerator.prepare(model)
    # Load the model weights from the checkpoint
    # Load weights saved by Accelerate
    state_dict = load_file("model_open_web_full/checkpoint.pt/model.safetensors")

    # Step 3: Inspect keys if necessary
    if "lm_head.weight" not in state_dict:
        print("⚠️ WARNING: lm_head.weight missing from checkpoint!")
        print("Keys available:", list(state_dict.keys())[:10])

    model.load_state_dict(state_dict, strict=False)

    train_loader, val_loader, test_loader, collate_fn, total_train_batches, total_val_batches, total_test_batches = iter_data_loader(config, tokenizer, config["cache_path"])

    # perplexity = eval_ppl(model, val_loader, device, total_val_batches)
    avg_nll, perplexity = evaluate_perplexity(model, val_loader, accelerator, total_val_batches)
    print(f"Validation Perplexity: {perplexity:.4f}")