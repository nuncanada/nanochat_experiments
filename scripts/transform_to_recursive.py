import os
import argparse
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.recursive import GraphRecursiveGPT as RecursiveGPT
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import print0

def transform(checkpoint_path, output_path, step):
    device = "cpu"
    # 1. Load original checkpoint metadata to get config
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_path, step, device, load_optimizer=False, rank=0)
    
    config_dict = meta_data["model_config"]
    config = GPTConfig(**config_dict)
    
    # 2. Create RecursiveGPT model
    model = RecursiveGPT(config)
    
    # 3. Initialize dependency matrix to sub-diagonal (Isomorphic to sequential)
    with torch.no_grad():
        D = torch.zeros((config.n_layer, config.n_layer))
        for i in range(1, config.n_layer):
            D[i, i-1] = 1.0
        model.dependency_matrix.copy_(D)
    
    # 4. Load weights from original model
    model.load_state_dict(model_data, strict=False)
    
    # 5. Save the new checkpoint
    os.makedirs(output_path, exist_ok=True)
    meta_data["model_type"] = "recursive"
    
    save_checkpoint(
        output_path,
        step,
        model.state_dict(),
        None,
        meta_data,
        rank=0
    )
    print(f"Successfully transformed model to GraphRecursive and saved to {output_path} at step {step}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    
    transform(args.checkpoint_dir, args.output_dir, args.step)
