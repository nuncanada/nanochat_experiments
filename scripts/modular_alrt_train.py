"""
Train Modular ALRT from scratch.
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

from nanochat.gpt import GPTConfig
from nanochat.recursive import BG_ALRT as RecursiveGPT
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import compute_init, print0, print_banner, autodetect_device_type, COMPUTE_DTYPE
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Train Modular ALRT from scratch")
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--device-type", type=str, default="")
# Model architecture
parser.add_argument("--depth", type=int, default=16)
parser.add_argument("--aspect-ratio", type=int, default=8)
parser.add_argument("--max-seq-len", type=int, default=64)
parser.add_argument("--n-groups", type=int, default=8)
# ALRT specific
parser.add_argument("--n-steps", type=int, default=8)
parser.add_argument("--top-fixed", type=int, default=2)
parser.add_argument("--bottom-fixed", type=int, default=2)
parser.add_argument("--ponder-weight", type=float, default=0.01)
# Optimization
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--num-iterations", type=int, default=500)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--dependency-lr", type=float, default=0.05)
parser.add_argument("--warmup-steps", type=int, default=40)
# Evaluation
parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--eval-tokens", type=int, default=4096)
parser.add_argument("--model-tag", type=str, default="modular_alrt_scratch")
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
# Model
n_embd = args.depth * args.aspect_ratio
# Ensure n_embd is divisible by n_groups
if n_embd % args.n_groups != 0:
    n_embd = ((n_embd // args.n_groups) + 1) * args.n_groups

config = GPTConfig(
    n_layer=args.depth,
    n_embd=n_embd,
    n_head=args.n_groups, 
    n_kv_head=args.n_groups,
    sequence_len=args.max_seq_len,
    vocab_size=tokenizer.get_vocab_size(),
)
model = RecursiveGPT(config, n_groups=args.n_groups)
model.set_n_steps(args.n_steps, top_fixed=args.top_fixed, bottom_fixed=args.bottom_fixed)
model.to(device)

# -----------------------------------------------------------------------------
# Optimizer
optimizer = model.setup_optimizer(matrix_lr=args.matrix_lr, adapter_lr=args.dependency_lr)

# -----------------------------------------------------------------------------
# DataLoaders
total_batch_size = args.device_batch_size * ddp_world_size
train_loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, config.sequence_len, split="train", device=device)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, config.sequence_len, split="val", device=device)

# -----------------------------------------------------------------------------
# Training loop
step = 0
num_iterations = args.num_iterations

while step < num_iterations:
    t0 = time.time()
    
    if step % 50 == 0:
        model.plot_dependency_matrix()
    
    # Eval
    if args.eval_every > 0 and step % args.eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (total_batch_size * config.sequence_len)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step} | Val bpb: {val_bpb:.4f}")
        model.train()

    # Optimization step
    optimizer.zero_grad()
    x, y = next(train_loader)
    
    curr_lr = args.matrix_lr * min(1.0, (step + 1) / args.warmup_steps)
    for group in optimizer.param_groups:
        group['lr'] = curr_lr

    loss, stats = model(x, y, return_ponder_stats=True, ponder_weight=args.ponder_weight, n_steps=args.n_steps)
    loss.backward()
    
    if hasattr(model, 'dependency_mask') and model.dependency_matrix.grad is not None:
        model.dependency_matrix.grad.mul_(model.dependency_mask)
        
    optimizer.step()
    
    t1 = time.time()
    dt = t1 - t0
    if step % 10 == 0:
        print0(f"Step {step} | loss: {stats['loss']:.4f} | ponder: {stats['ponder']:.4f} | avg_steps: {stats['avg_steps']:.2f} | time: {dt*1000:.2f}ms")
    
    step += 1

# Final save
if master_process:
    checkpoint_dir = os.path.join(os.environ.get("NANOCHAT_BASE_DIR", "."), "scratch_checkpoints", args.model_tag)
    save_checkpoint(checkpoint_dir, step, model.state_dict(), optimizer.state_dict(), {"model_config": asdict(config), "step": step}, rank=ddp_rank)
