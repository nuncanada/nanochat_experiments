"""
Retrain a Dynamic Recursive GPT model to flexibilize dependency parameters.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import torch
import torch.distributed as dist

from nanochat.gpt import GPTConfig, Linear
from nanochat.recursive import DynamicRecursiveGPT as RecursiveGPT
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import compute_init, print0, print_banner, autodetect_device_type, COMPUTE_DTYPE
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Retrain dynamic recursive model")
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--checkpoint-dir", type=str, required=True)
parser.add_argument("--resume-from-step", type=int, required=True)
parser.add_argument("--n-steps", type=int, default=-1)
parser.add_argument("--top-fixed", type=int, default=0)
parser.add_argument("--bottom-fixed", type=int, default=0)
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--num-iterations", type=int, default=1000)
parser.add_argument("--dependency-lr", type=float, default=0.01)
parser.add_argument("--ponder-weight", type=float, default=0.01)
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=1024*1024)
parser.add_argument("--save-every", type=int, default=-1)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# -----------------------------------------------------------------------------
# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)

# -----------------------------------------------------------------------------
# Load the Model
model_data, optimizer_data, meta_data = load_checkpoint(args.checkpoint_dir, args.resume_from_step, device, load_optimizer=False, rank=ddp_rank)
config = GPTConfig(**meta_data["model_config"])
model = RecursiveGPT(config)
model.load_state_dict(model_data, strict=False)
model.to(device)

if args.n_steps > 0:
    model.set_n_steps(args.n_steps, top_fixed=args.top_fixed, bottom_fixed=args.bottom_fixed)
    model.to(device)

orig_model = model

# -----------------------------------------------------------------------------
# Optimizer
optimizer = model.setup_optimizer(matrix_lr=args.dependency_lr)

# -----------------------------------------------------------------------------
# DataLoaders
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, config.sequence_len, split="train", device=device)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, config.sequence_len, split="val", device=device)

# -----------------------------------------------------------------------------
# Training loop
step = args.resume_from_step
num_iterations = args.num_iterations

while step < num_iterations:
    t0 = time.time()
    
    if step % 10 == 0:
        orig_model.plot_dependency_matrix()
    
    # Eval
    if args.eval_every > 0 and step % args.eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * config.sequence_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step} | Val bpb: {val_bpb:.4f}")
        model.train()

    # Optimization step
    optimizer.zero_grad()
    x, y = next(train_loader)
    loss, stats = model(x, y, return_ponder_stats=True, ponder_weight=args.ponder_weight)
    loss.backward()
    
    # Apply gradient mask to dependency_matrix
    if hasattr(orig_model, 'dependency_mask') and orig_model.dependency_matrix.grad is not None:
        orig_model.dependency_matrix.grad.mul_(orig_model.dependency_mask)
        
    optimizer.step()
    
    t1 = time.time()
    dt = t1 - t0
    if step % 1 == 0:
        print0(f"Step {step} | loss: {stats['loss']:.4f} | ponder: {stats['ponder']:.4f} | avg_steps: {stats['avg_steps']:.2f} | time: {dt*1000:.2f}ms")
    
    step += 1

# Final save
if master_process:
    save_checkpoint(args.checkpoint_dir, step, orig_model.state_dict(), optimizer.state_dict(), {"model_config": asdict(config), "step": step}, rank=ddp_rank)
