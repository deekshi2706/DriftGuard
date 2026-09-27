"""
Full A+B+C integration smoke test. Not part of the deliverable itself —
just proof the three contracts actually fit together before handing off.
"""
import torch
from data.loader import load_elliptic_graph, get_time_step_subgraph, get_normal_nodes, get_labeled_eval_nodes
from models.diffusion import GraphDiffusionModel
from training.continual import ContinualTrainer
from eval.metrics import evaluate_step, forgetting_measure


def build_normal_batch(g, step):
    sub = get_time_step_subgraph(g, step)
    normal_idx = get_normal_nodes(g, step)
    node_mask = torch.zeros(sub.num_nodes, dtype=torch.bool)
    node_mask[normal_idx] = True
    g2l = torch.full((sub.num_nodes,), -1, dtype=torch.long)
    g2l[normal_idx] = torch.arange(normal_idx.numel())
    src, dst = sub.edge_index
    em = node_mask[src] & node_mask[dst]
    normal_edge_index = torch.stack([g2l[src[em]], g2l[dst[em]]])

    class _B: pass
    b = _B(); b.x = sub.x[normal_idx]; b.edge_index = normal_edge_index
    return b, sub


def build_eval_batch(g, step):
    sub = get_time_step_subgraph(g, step)
    eval_idx = get_labeled_eval_nodes(g, step)
    node_mask = torch.zeros(sub.num_nodes, dtype=torch.bool)
    node_mask[eval_idx] = True
    g2l = torch.full((sub.num_nodes,), -1, dtype=torch.long)
    g2l[eval_idx] = torch.arange(eval_idx.numel())
    src, dst = sub.edge_index
    em = node_mask[src] & node_mask[dst]
    eval_edge_index = torch.stack([g2l[src[em]], g2l[dst[em]]])
    return sub.x[eval_idx], eval_edge_index, sub.y[eval_idx]


g = load_elliptic_graph("raw")
steps = sorted(g.time_step.unique().tolist())
print("Steps:", steps)

torch.manual_seed(0)
model = GraphDiffusionModel(in_dim=165, num_diffusion_steps=100, hidden_dim=32, num_layers=2)
trainer = ContinualTrainer(model, replay_buffer_size=3)

history = [[None] * len(steps) for _ in steps]

for i, step in enumerate(steps):
    batch, _ = build_normal_batch(g, step)
    if batch.x.size(0) == 0:
        print(f"step {step}: no normal nodes, skipping update")
        continue
    out = trainer.update(batch, epochs=5, step_id=step)
    print(f"trained through step {step}, losses={[round(l,4) for l in out['losses']]}")

    # after training through step i, evaluate on ALL steps seen so far (j <= i)
    for j in range(i + 1):
        ev_x, ev_ei, ev_y = build_eval_batch(g, steps[j])
        if len(set(ev_y.tolist())) < 2:
            continue  # skip if this slice happens to be single-class
        scores = model.compute_anomaly_score(ev_x, ev_ei)
        metrics = evaluate_step(scores, ev_y)
        history[i][j] = metrics["auroc"]
        print(f"  eval on step {steps[j]}: AUROC={metrics['auroc']:.3f} AUPRC={metrics['auprc']:.3f}")

fm = forgetting_measure(history)
print("\nForgetting measure:", fm)
print("\nALL THREE MODULES INTEGRATE CORRECTLY")
