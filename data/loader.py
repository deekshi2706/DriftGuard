"""
data/loader.py
Person A — Data & Graph Pipeline

Loads the Elliptic Bitcoin dataset from the raw Kaggle CSVs into a single
PyTorch Geometric `Data` object, and provides helper functions to slice
that graph by time step for training / evaluation.

Expected raw files (unmodified Kaggle download), all in `data_dir`:
    elliptic_txs_features.csv   (no header; 167 cols: txId, time_step, 165 features)
    elliptic_txs_classes.csv    (header: txId,class ; class in {"1","2","unknown"})
    elliptic_txs_edgelist.csv   (header: txId1,txId2)

Label convention used throughout this file and by Person B / Person C:
    1 = illicit
    0 = licit
   -1 = unknown / unlabeled   (~77% of nodes in the real dataset)

Public API (this is the contract Person B and Person C build against):
    load_elliptic_graph(data_dir) -> torch_geometric.data.Data
    get_time_step_subgraph(data, step) -> torch_geometric.data.Data
    get_normal_nodes(data, step) -> torch.LongTensor of node indices
    get_labeled_eval_nodes(data, step) -> torch.LongTensor of node indices

See DATA_SPEC.md for exact shapes and field semantics.
"""

from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch_geometric.data import Data


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _read_raw_csvs(data_dir: Path):
    feat_path = data_dir / "elliptic_txs_features.csv"
    class_path = data_dir / "elliptic_txs_classes.csv"
    edge_path = data_dir / "elliptic_txs_edgelist.csv"

    for p in (feat_path, class_path, edge_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected Elliptic raw file not found: {p}. "
                f"Make sure data_dir points at the folder containing the "
                f"three unmodified Kaggle CSVs."
            )

    # Features file has NO header.
    features_df = pd.read_csv(feat_path, header=None)
    classes_df = pd.read_csv(class_path)
    edges_df = pd.read_csv(edge_path)
    return features_df, classes_df, edges_df


def _build_label_map(classes_df: pd.DataFrame) -> dict:
    """Map txId -> {1: illicit, 0: licit, -1: unknown}."""
    mapping = {"1": 1, "2": 0, "unknown": -1}
    class_str = classes_df["class"].astype(str)
    labels = class_str.map(mapping)
    if labels.isna().any():
        bad = class_str[labels.isna()].unique()
        raise ValueError(f"Unexpected class values in classes CSV: {bad}")
    return dict(zip(classes_df["txId"], labels))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def load_elliptic_graph(data_dir: Union[str, Path]) -> Data:
    """
    Load the full Elliptic graph from the raw Kaggle CSVs.

    Returns
    -------
    torch_geometric.data.Data with fields:
        x            : FloatTensor [num_nodes, 165]   node features (no txId, no time step)
        edge_index   : LongTensor  [2, num_edges]      directed edges, 0-indexed, PyG format
        y            : LongTensor  [num_nodes]         1=illicit, 0=licit, -1=unknown
        time_step    : LongTensor  [num_nodes]         integer time step, 1..49 in real data
        tx_id        : LongTensor  [num_nodes]         original Elliptic txId, for traceability
        num_nodes    : int
    """
    data_dir = Path(data_dir)
    features_df, classes_df, edges_df = _read_raw_csvs(data_dir)

    # --- node ordering & id -> contiguous index map -----------------------
    tx_ids = features_df.iloc[:, 0].values
    tx_id_to_idx = {tx_id: i for i, tx_id in enumerate(tx_ids)}

    time_step = torch.tensor(features_df.iloc[:, 1].values, dtype=torch.long)
    x = torch.tensor(
        features_df.iloc[:, 2:].values, dtype=torch.float
    )  # [num_nodes, 165]

    # --- labels -------------------------------------------------------------
    label_map = _build_label_map(classes_df)
    y = torch.tensor(
        [label_map.get(tx_id, -1) for tx_id in tx_ids], dtype=torch.long
    )

    # --- edges: remap txId -> contiguous node index, drop dangling edges ----
    src = edges_df["txId1"].map(tx_id_to_idx)
    dst = edges_df["txId2"].map(tx_id_to_idx)
    valid = src.notna() & dst.notna()
    dropped = int((~valid).sum())
    if dropped:
        print(
            f"[loader] Warning: dropped {dropped} edges referencing txIds "
            f"absent from the features file."
        )
    import numpy as np

    edge_index = torch.from_numpy(
        np.stack([src[valid].astype(int).values, dst[valid].astype(int).values])
    ).long()

    data = Data(x=x, edge_index=edge_index, y=y)
    data.time_step = time_step
    data.tx_id = torch.tensor(tx_ids, dtype=torch.long)
    data.num_nodes = x.size(0)
    return data


def get_time_step_subgraph(data: Data, step: int) -> Data:
    """
    Return the induced subgraph containing only nodes whose time_step == step,
    with edges restricted to those that connect two such nodes, and all
    tensors reindexed to a fresh contiguous [0, num_nodes_in_step) range.

    Node indices in the returned Data are LOCAL to this subgraph (0-indexed
    within the time step) — they do NOT match indices into the full graph.
    `tx_id` is preserved so callers can map back to the original transaction.
    """
    node_mask = data.time_step == step
    node_idx = node_mask.nonzero(as_tuple=True)[0]  # global indices, sorted
    if node_idx.numel() == 0:
        raise ValueError(f"No nodes found with time_step == {step}")

    # global -> local index map
    global_to_local = torch.full((data.num_nodes,), -1, dtype=torch.long)
    global_to_local[node_idx] = torch.arange(node_idx.numel())

    src, dst = data.edge_index
    edge_mask = node_mask[src] & node_mask[dst]
    new_edge_index = torch.stack(
        [global_to_local[src[edge_mask]], global_to_local[dst[edge_mask]]]
    )

    sub = Data(
        x=data.x[node_idx],
        edge_index=new_edge_index,
        y=data.y[node_idx],
    )
    sub.time_step = data.time_step[node_idx]
    sub.tx_id = data.tx_id[node_idx]
    sub.num_nodes = node_idx.numel()
    return sub


def get_normal_nodes(data: Data, step: int) -> torch.Tensor:
    """
    Return LOCAL node indices (into the time-step subgraph returned by
    get_time_step_subgraph(data, step)) of licit (y == 0) nodes only.

    Intended use: Person B trains the diffusion model on "normal" behavior
    only, so anomalies show up as high reconstruction / denoising error.
    """
    sub = get_time_step_subgraph(data, step)
    return (sub.y == 0).nonzero(as_tuple=True)[0]


def get_labeled_eval_nodes(data: Data, step: int) -> torch.Tensor:
    """
    Return LOCAL node indices (into the time-step subgraph returned by
    get_time_step_subgraph(data, step)) of all labeled nodes — licit (y==0)
    AND illicit (y==1) — excluding unknown (y==-1) nodes.

    Intended use: Person C's evaluation (AUROC / AUPRC / forgetting) needs
    ground truth, so unknown nodes must be excluded here.
    """
    sub = get_time_step_subgraph(data, step)
    return (sub.y != -1).nonzero(as_tuple=True)[0]


# --------------------------------------------------------------------------
# Quick manual smoke test: `python -m data.loader /path/to/raw_csvs`
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "raw"
    g = load_elliptic_graph(data_dir)
    print("Full graph:", g)
    steps = sorted(g.time_step.unique().tolist())
    print("Time steps present:", steps)
    step = steps[0]
    sub = get_time_step_subgraph(g, step)
    print(f"Subgraph for step {step}:", sub)
    print("  normal (licit) nodes:", get_normal_nodes(g, step).shape)
    print("  labeled eval nodes:", get_labeled_eval_nodes(g, step).shape)
