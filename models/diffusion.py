"""
models/diffusion.py
Person B — Diffusion Model (core architecture)

A self-contained graph diffusion model used for anomaly detection:
  - forward noising process: q(x_t | x_0) adds Gaussian noise to node features
    over T steps on a fixed linear/cosine beta schedule
  - reverse process: a GNN (GCN or GAT) denoiser predicts the noise added at
    step t, conditioned on the (noisy) node features, the graph structure,
    and a sinusoidal embedding of t
  - anomaly score: nodes whose features the model reconstructs/denoises
    poorly (relative to the normal-node distribution it was trained on)
    score as more anomalous

Trained on Person A's `get_normal_nodes(data, step)` output only (licit
nodes), so the model learns the distribution of *normal* transaction
behavior. At eval time, `compute_anomaly_score` is called on
`get_labeled_eval_nodes(data, step)` (licit + illicit) — illicit nodes are
expected to sit further from the learned normal manifold and score higher.

Public API (Person C builds against this without touching internals):
    GraphDiffusionModel(nn.Module)
        .forward(x, edge_index, t)          -> predicted noise, same shape as x
        .train_step(batch)                  -> dict with 'loss' (scalar tensor)
        .compute_anomaly_score(x, edge_index) -> FloatTensor [num_nodes], higher = more anomalous
    save_checkpoint(model, path, **meta)
    load_checkpoint(path, in_dim, device=...) -> (model, meta)
"""

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


# --------------------------------------------------------------------------
# Noise schedule
# --------------------------------------------------------------------------

def _make_beta_schedule(num_steps: int, schedule: str = "linear",
                         beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_steps)
    elif schedule == "cosine":
        s = 0.008
        steps = torch.arange(num_steps + 1, dtype=torch.float64)
        f = torch.cos(((steps / num_steps + s) / (1 + s)) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999).float()
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: LongTensor [batch] -> FloatTensor [batch, dim]"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# --------------------------------------------------------------------------
# GNN denoiser
# --------------------------------------------------------------------------

class _GNNDenoiser(nn.Module):
    """Predicts the noise added to node features, conditioned on t."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, time_dim: int = 64,
                 num_layers: int = 3, gnn_type: str = "gcn", heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.convs.append(
                    GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
                )
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x_t: torch.Tensor, edge_index: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(_sinusoidal_time_embedding(t, self.time_dim))  # [N, hidden]
        h = self.in_proj(x_t) + t_emb
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.silu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual
        return self.out_proj(h)


# --------------------------------------------------------------------------
# Public model class
# --------------------------------------------------------------------------

class GraphDiffusionModel(nn.Module):
    """
    Static (single-time-step) graph diffusion model for anomaly detection.

    Args:
        in_dim: node feature dimension (165 for Elliptic, per DATA_SPEC.md)
        num_diffusion_steps: T, number of forward/reverse diffusion steps
        hidden_dim: GNN hidden width
        gnn_type: "gcn" or "gat"
        schedule: "linear" or "cosine" beta schedule
        lr: learning rate used internally by train_step's optimizer
    """

    def __init__(self, in_dim: int, num_diffusion_steps: int = 200,
                 hidden_dim: int = 128, num_layers: int = 3,
                 gnn_type: str = "gcn", schedule: str = "linear", lr: float = 1e-3):
        super().__init__()
        self.in_dim = in_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.gnn_type = gnn_type

        betas = _make_beta_schedule(num_diffusion_steps, schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.denoiser = _GNNDenoiser(
            in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, gnn_type=gnn_type
        )
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ---- core diffusion math -----------------------------------------

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward noising: x_t = sqrt(a_bar_t) * x0 + sqrt(1 - a_bar_t) * noise"""
        sqrt_ac = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_ac * x0 + sqrt_om * noise

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise component of x (assumed already noised to step t)."""
        return self.denoiser(x, edge_index, t)

    # ---- training ------------------------------------------------------

    def train_step(self, batch) -> dict:
        """
        batch: an object/namespace with `.x` [N, in_dim] and `.edge_index` [2, E]
               (e.g. the Data object returned by get_time_step_subgraph, indexed
               down to normal nodes only — see notebooks/model_dev.ipynb).

        Runs one optimizer step and returns {'loss': <python float>}.
        """
        self.train()
        x0, edge_index = batch.x, batch.edge_index
        device = x0.device
        n = x0.size(0)

        t = torch.randint(0, self.num_diffusion_steps, (n,), device=device)
        noise = torch.randn_like(x0)
        x_t = self._q_sample(x0, t, noise)

        pred_noise = self.forward(x_t, edge_index, t)
        loss = F.mse_loss(pred_noise, noise)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return {"loss": loss.item()}

    # ---- anomaly scoring -------------------------------------------------

    @torch.no_grad()
    def compute_anomaly_score(self, x: torch.Tensor, edge_index: torch.Tensor,
                               num_eval_steps: Optional[int] = None) -> torch.Tensor:
        """
        Anomaly score per node: expected denoising error averaged over a
        handful of sampled diffusion steps, run at inference (eval mode).
        Higher score = further from the learned "normal" distribution
        (i.e. more anomalous).

        Args:
            x: [num_nodes, in_dim] raw (un-noised) node features
            edge_index: [2, num_edges]
            num_eval_steps: how many t values to average over (default: 10,
                spread evenly across the schedule). More steps = smoother
                but slower score.

        Returns:
            FloatTensor [num_nodes]
        """
        self.eval()
        device = x.device
        n = x.size(0)
        num_eval_steps = num_eval_steps or min(10, self.num_diffusion_steps)
        t_values = torch.linspace(
            0, self.num_diffusion_steps - 1, num_eval_steps, device=device
        ).long()

        scores = torch.zeros(n, device=device)
        for t_scalar in t_values:
            t = t_scalar.expand(n)
            noise = torch.randn_like(x)
            x_t = self._q_sample(x, t, noise)
            pred_noise = self.forward(x_t, edge_index, t)
            # per-node MSE between predicted and true injected noise
            err = ((pred_noise - noise) ** 2).mean(dim=-1)
            scores += err
        return scores / num_eval_steps


# --------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------

def save_checkpoint(model: GraphDiffusionModel, path: str, **meta) -> None:
    """Save model weights + constructor args + any extra metadata (e.g. time step trained on)."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": model.in_dim,
            "num_diffusion_steps": model.num_diffusion_steps,
            "gnn_type": model.gnn_type,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "meta": meta,
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu"):
    """Returns (model, meta_dict). Reconstructs the model with the saved constructor args."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GraphDiffusionModel(
        in_dim=ckpt["in_dim"],
        num_diffusion_steps=ckpt["num_diffusion_steps"],
        gnn_type=ckpt["gnn_type"],
        hidden_dim=ckpt.get("hidden_dim", 128),
        num_layers=ckpt.get("num_layers", 3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, ckpt.get("meta", {})


# --------------------------------------------------------------------------
# Smoke test using Person A's loader directly (validates the integration point)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.loader import (
        load_elliptic_graph, get_time_step_subgraph,
        get_normal_nodes, get_labeled_eval_nodes,
    )

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../raw"
    g = load_elliptic_graph(data_dir)
    step = sorted(g.time_step.unique().tolist())[0]
    sub = get_time_step_subgraph(g, step)
    normal_idx = get_normal_nodes(g, step)
    eval_idx = get_labeled_eval_nodes(g, step)

    model = GraphDiffusionModel(in_dim=sub.x.size(1), num_diffusion_steps=100, hidden_dim=32, num_layers=2)

    class _Batch:
        pass

    normal_sub_x = sub.x[normal_idx]
    # rebuild an edge_index restricted to normal nodes only, for training
    node_mask = torch.zeros(sub.num_nodes, dtype=torch.bool)
    node_mask[normal_idx] = True
    g2l = torch.full((sub.num_nodes,), -1, dtype=torch.long)
    g2l[normal_idx] = torch.arange(normal_idx.numel())
    src, dst = sub.edge_index
    em = node_mask[src] & node_mask[dst]
    normal_edge_index = torch.stack([g2l[src[em]], g2l[dst[em]]])

    batch = _Batch()
    batch.x, batch.edge_index = normal_sub_x, normal_edge_index

    print(f"Training on step {step}: {normal_sub_x.size(0)} normal nodes, "
          f"{normal_edge_index.size(1)} edges")
    for epoch in range(20):
        out = model.train_step(batch)
        if epoch % 5 == 0:
            print(f"  epoch {epoch}: loss={out['loss']:.4f}")

    # score using the eval-node-induced subgraph for a fair signal:
    node_mask_e = torch.zeros(sub.num_nodes, dtype=torch.bool)
    node_mask_e[eval_idx] = True
    g2l_e = torch.full((sub.num_nodes,), -1, dtype=torch.long)
    g2l_e[eval_idx] = torch.arange(eval_idx.numel())
    em_e = node_mask_e[src] & node_mask_e[dst]
    eval_edge_index = torch.stack([g2l_e[src[em_e]], g2l_e[dst[em_e]]])
    scores = model.compute_anomaly_score(sub.x[eval_idx], eval_edge_index)

    labels = sub.y[eval_idx]
    print(f"Anomaly scores computed for {scores.numel()} eval nodes.")
    print(f"  mean score | licit  (y=0): {scores[labels == 0].mean().item():.4f}")
    print(f"  mean score | illicit(y=1): {scores[labels == 1].mean().item():.4f}")

    save_checkpoint(model, "static_checkpoint.pt", trained_on_step=step)
    loaded_model, meta = load_checkpoint("static_checkpoint.pt")
    print("Checkpoint saved & reloaded. meta:", meta)
