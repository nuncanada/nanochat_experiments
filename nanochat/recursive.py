import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.gpt import norm, apply_rotary_emb, Linear, GPTConfig
from nanochat.common import COMPUTE_DTYPE

class BG_ALRT(nn.Module):
    """
    Branch-Graph Adaptive Latent Recursive Transformer.
    Optimized for massively parallel execution of 128 independent branches.
    """
    def __init__(self, config, n_groups=8):
        super().__init__()
        self.config, self.n_layer, self.n_groups = config, config.n_layer, n_groups
        self.n_nodes = self.n_layer * n_groups
        self.group_dim = config.n_embd // n_groups
        self.head_dim = self.group_dim # 1 head per branch
        
        # Vectorized weights: [N_nodes, Out, In]
        self.adapters = nn.Parameter(torch.empty(self.n_nodes, self.group_dim, config.n_embd))
        self.qkv_w = nn.Parameter(torch.empty(self.n_nodes, 3 * self.group_dim, self.group_dim))
        self.attn_proj = nn.Parameter(torch.empty(self.n_nodes, self.group_dim, self.group_dim))
        self.mlp_fc = nn.Parameter(torch.empty(self.n_nodes, 4 * self.group_dim, self.group_dim))
        self.mlp_proj = nn.Parameter(torch.empty(self.n_nodes, self.group_dim, 4 * self.group_dim))
        
        self.dependency_matrix = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes))
        self.router = nn.Linear(config.n_embd, 1); nn.init.constant_(self.router.bias, -3.0)
        self.transformer = nn.ModuleDict({"wte": nn.Embedding(config.vocab_size, config.n_embd)})
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        freqs = torch.outer(torch.arange(config.sequence_len * 10).float(), inv_freq)
        self.register_buffer("cos", freqs.cos()[None, :, :]); self.register_buffer("sin", freqs.sin()[None, :, :])
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.adapters.zero_()
            for i in range(self.n_nodes):
                idx = (i % self.n_groups) * self.group_dim
                self.adapters[i, :, idx:idx+self.group_dim] = torch.eye(self.group_dim)
            for p in [self.qkv_w, self.attn_proj, self.mlp_fc, self.mlp_proj]: nn.init.normal_(p, std=0.02)
            nn.init.normal_(self.transformer.wte.weight, std=0.02); nn.init.normal_(self.lm_head.weight, std=0.02)
            D = torch.zeros(self.n_nodes, self.n_nodes)
            for l in range(1, self.n_layer):
                for gc in range(self.n_groups):
                    for gp in range(self.n_groups): D[l*self.n_groups + gc, (l-1)*self.n_groups + gp] = 1.0 / self.n_groups
            self.dependency_matrix.copy_(D)

    def forward(self, idx, targets=None, loss_reduction='mean', return_ponder_stats=False, ponder_weight=0.01, n_steps=8):
        B, T = idx.size(); S = n_steps
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        p_cont = torch.ones((B, T), device=x.device, dtype=x.dtype); steps = torch.zeros((B, T), device=x.device, dtype=x.dtype)
        with torch.no_grad():
            depths = torch.zeros(self.n_nodes, device=x.device); dp = torch.relu(self.dependency_matrix)
            for _ in range(self.n_layer): depths = torch.matmul(dp, depths + 1.0)
        
        for t in range(S):
            td = t * (self.n_layer / S); w_all = torch.exp(-torch.abs(depths - td))
            active = torch.where(w_all > 0.15)[0]
            if len(active) == 0: continue
            
            # 1. Parallel Subspace Projection: (B, T, N_act, C/G)
            xi = torch.einsum('btc,ngc->btng', x, self.adapters[active])
            # 2. Parallel Attention
            qkv = torch.einsum('btng,nog->btno', xi, self.qkv_w[active])
            q, k, v = qkv.view(B, T, len(active), 3, self.group_dim).unbind(3)
            # Rotary: apply to (B, T, N, D). Flatten N into B for 4D: (B*N, T, 1, D)
            q_rot = q.transpose(1, 2).reshape(-1, T, 1, self.group_dim)
            k_rot = k.transpose(1, 2).reshape(-1, T, 1, self.group_dim)
            c4, s4 = cos.view(1, T, 1, -1), sin.view(1, T, 1, -1)
            q_rot = apply_rotary_emb(q_rot, c4, s4); k_rot = apply_rotary_emb(k_rot, c4, s4)
            
            # Local Attention: SDPA expects (B*N, 1, T, D)
            q_att = norm(q_rot).transpose(1, 2)
            k_att = norm(k_rot).transpose(1, 2)
            v_att = v.transpose(1, 2).reshape(-1, 1, T, self.group_dim)
            att = F.scaled_dot_product_attention(q_att, k_att, v_att, is_causal=True)
            # Proj: (B, T, N_act, C/G)
            att = att.view(B, len(active), T, self.group_dim).transpose(1, 2)
            xi_mid = xi + torch.einsum('btng,ngk->btng', att, self.attn_proj[active])
            # 3. Parallel MLP
            mlp_fc = torch.einsum('btng,nog->btno', norm(xi_mid), self.mlp_fc[active])
            mlp_proj = torch.einsum('btno,ngk->btng', F.relu(mlp_fc).square(), self.mlp_proj[active])
            up_all = (xi_mid + mlp_proj) - xi
            # 4. Aggregated Resid Update
            full_up = torch.zeros_like(x)
            weighted_up = up_all * w_all[active].view(1, 1, -1, 1)
            # Fast scatter-add using slicing (since each node i updates a specific subspace slice i % G)
            for g in range(self.n_groups):
                # Find nodes in 'active' that belong to group g
                g_mask = (active % self.n_groups) == g
                if g_mask.any():
                    full_up[:, :, g*self.group_dim : (g+1)*self.group_dim] += weighted_up[:, :, g_mask].sum(dim=2)
            
            x = x + full_up * p_cont.unsqueeze(-1); steps += p_cont
            ph = torch.sigmoid(self.router(x)).squeeze(-1); p_cont = p_cont * (1.0 - ph) if self.training else (ph < 0.5).float() * p_cont
            
        x = norm(x); logits = self.lm_head(x); logits = 15 * torch.tanh(logits.float() / 15)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            total = loss + steps.mean() * ponder_weight
            if return_ponder_stats: return total, {"loss": loss.item(), "ponder": (steps.mean()*ponder_weight).item(), "avg_steps": steps.mean().item()}
            return total
        return logits

    def setup_optimizer(self, matrix_lr=0.02, weight_decay=0.0, adapter_lr=0.01):
        return torch.optim.AdamW(self.parameters(), lr=matrix_lr, weight_decay=weight_decay)
    def get_device(self): return self.transformer.wte.weight.device
    def set_n_steps(self, n_steps, top_fixed=0, bottom_fixed=0): pass
    def plot_dependency_matrix(self): pass
