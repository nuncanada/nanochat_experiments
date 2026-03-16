import os
import argparse
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.recursive import DynamicRecursiveGPT as RecursiveGPT
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import print0

def transform(checkpoint_path, output_path, step):
    device = "cpu"
    # 1. Load original checkpoint metadata to get config
    # We use a dummy rank 0
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_path, step, device, load_optimizer=False, rank=0)
    
    config_dict = meta_data["model_config"]
    config = GPTConfig(**config_dict)
    
    # 2. Create RecursiveGPT model
    model = RecursiveGPT(config)
    
    # 3. Initialize dependency matrix to Identity (isomorphic)
    with torch.no_grad():
        model.dependency_matrix.copy_(torch.eye(config.n_layer))
    
    # 4. Load weights from original model
    # RecursiveGPT inherits from GPT, so state_dict keys match except for dependency_matrix
    model.load_state_dict(model_data, strict=False)
    
    # 5. Save the new checkpoint
    os.makedirs(output_path, exist_ok=True)
    
    # Update metadata
    meta_data["model_type"] = "recursive"
    
    save_checkpoint(
        output_path,
        step,
        model.state_dict(),
        None, # No optimizer for now
        meta_data,
        rank=0
    )
    print(f"Successfully transformed model to recursive and saved to {output_path} at step {step}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    
    transform(args.checkpoint_dir, args.output_dir, args.step)
