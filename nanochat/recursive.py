import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.gpt import GPT, norm, apply_rotary_emb
from nanochat.common import COMPUTE_DTYPE

class DynamicRecursiveGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        # dependency_matrix[t, i] is the weight of block i at step t
        self.n_steps = config.n_layer
        self.dependency_matrix = nn.Parameter(torch.eye(config.n_layer))
        self.register_buffer('dependency_mask', torch.ones((self.n_steps, config.n_layer)))
        
        # Layer Adapters: Translation matrices between original layers and shared state
        # These help preserve the geometric transformations expected by each block.
        self.layer_adapters = nn.ModuleList([
            nn.Linear(config.n_embd, config.n_embd, bias=False) for _ in range(config.n_layer)
        ])
        for adapter in self.layer_adapters:
            nn.init.eye_(adapter.weight)
            
        # Router for Adaptive Computation Time (ACT)
        self.router = nn.Linear(config.n_embd, 1)
        nn.init.constant_(self.router.bias, -3.0) 

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_ponder_stats=False, ponder_weight=0.01):
        B, T = idx.size()

        # Grab the rotary embeddings
        assert T <= self.cos.size(1)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Forward the trunk
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        
        # ACT tracking
        p_continue = torch.ones((B, T), device=x.device, dtype=x.dtype)
        step_counts = torch.zeros((B, T), device=x.device, dtype=x.dtype)
        
        # Recursive loop
        for t in range(self.n_steps):
            step_update = torch.zeros_like(x)
            for i in range(self.config.n_layer):
                w = self.dependency_matrix[t, i]
                if w != 0:
                    # Apply the Layer Adapter (Translation Matrix) to align the latent space
                    # before passing the state into the block.
                    x_aligned = self.layer_adapters[i](x)
                    
                    # Resid rescalers (applied to the aligned state)
                    x_rescaled = self.resid_lambdas[i] * x_aligned + self.x0_lambdas[i] * x0
                    
                    ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                    block = self.transformer.h[i]
                    block_out = block(x_rescaled, ve, cos_sin, self.window_sizes[i], kv_cache)
                    
                    # The update is (output - rescaled input)
                    update = block_out - x_rescaled
                    step_update = step_update + w * update
            
            x = x + step_update * p_continue.unsqueeze(-1)
            step_counts = step_counts + p_continue
            
            # Router decision
            is_trainable_step = self.dependency_mask[t].sum() > 0
            if is_trainable_step:
                p_halt = torch.sigmoid(self.router(x)).squeeze(-1)
                if self.training:
                    p_continue = p_continue * (1.0 - p_halt)
                else:
                    p_continue = (p_halt < 0.5).float() * p_continue
            
        x = norm(x)

        # Logits and loss
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            
            # Ponder Loss (encourage less steps)
            ponder_penalty = step_counts.mean() * ponder_weight
            total_loss = loss + ponder_penalty
            
            if return_ponder_stats:
                return total_loss, {"loss": loss.item(), "ponder": ponder_penalty.item(), "avg_steps": step_counts.mean().item()}
            return total_loss
        else:
            return logits

    def setup_optimizer(self, matrix_lr=0.02, weight_decay=0.0, router_lr=0.01, adapter_lr=0.01):
        params = [
            {'params': [p for n, p in self.named_parameters() if 'router' not in n and 'layer_adapters' not in n], 'lr': matrix_lr},
            {'params': self.router.parameters(), 'lr': router_lr},
            {'params': self.layer_adapters.parameters(), 'lr': adapter_lr}
        ]
        return torch.optim.AdamW(params, weight_decay=weight_decay)

    def set_n_steps(self, n_steps, top_fixed=0, bottom_fixed=0):
        n_layer = self.config.n_layer
        self.n_steps = n_steps
        new_matrix = torch.zeros((n_steps, n_layer), device=self.dependency_matrix.device, dtype=self.dependency_matrix.dtype)
        new_mask = torch.ones((n_steps, n_layer), device=self.dependency_matrix.device)

        for i in range(min(top_fixed, n_steps)):
            new_matrix[i, i] = 1.0
            new_mask[i, :] = 0.0
        for i in range(min(bottom_fixed, n_steps - top_fixed)):
            s_idx, l_idx = n_steps - 1 - i, n_layer - 1 - i
            if s_idx >= 0 and l_idx >= 0:
                new_matrix[s_idx, l_idx] = 1.0
                new_mask[s_idx, :] = 0.0

        mid_steps, mid_layers = n_steps - top_fixed - bottom_fixed, n_layer - top_fixed - bottom_fixed
        if mid_steps > 0 and mid_layers > 0:
            for i in range(mid_layers):
                l_idx = top_fixed + i
                s_idx = top_fixed + (i * mid_steps) // mid_layers
                new_matrix[s_idx, l_idx] = 1.0
                
        self.dependency_matrix = nn.Parameter(new_matrix)
        self.register_buffer('dependency_mask', new_mask)

    def plot_dependency_matrix(self):
        with torch.no_grad():
            matrix = self.dependency_matrix.data.cpu()
            print(f"Dependency Matrix ({self.n_steps} Steps x {self.config.n_layer} Layers):")
            for t in range(self.n_steps):
                row = ["#" if v > 0.9 else "o" if v > 0.1 else "." for v in matrix[t]]
                trainable = "[T]" if self.dependency_mask[t].sum() > 0 else "[F]"
                print(f"t={t:02d}: {' '.join(row)} {trainable}")
