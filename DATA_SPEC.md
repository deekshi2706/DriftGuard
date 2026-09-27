# DATA_SPEC.md — Elliptic Graph Loader Contract

Owner: Person A · `data/loader.py`
Consumers: Person B (`models/diffusion.py`), Person C (`training/continual.py`, `eval/metrics.py`)

This is the contract. If you're calling `load_elliptic_graph`,
`get_time_step_subgraph`, `get_normal_nodes`, or `get_labeled_eval_nodes`,
everything you need is here — you shouldn't need to open loader.py.

## Input

Raw Kaggle Elliptic files, unmodified, in one folder:
```
elliptic_txs_features.csv   # no header, 167 columns
elliptic_txs_classes.csv    # header: txId,class
elliptic_txs_edgelist.csv   # header: txId1,txId2
```

## Label convention (used everywhere downstream)

| meaning | value |
|---|---|
| illicit  | `1`  |
| licit    | `0`  |
| unknown  | `-1` |

(The raw CSV uses `"1"`=illicit, `"2"`=licit, `"unknown"` — the loader remaps this immediately.)

## `load_elliptic_graph(data_dir) -> Data`

Returns a single `torch_geometric.data.Data` object for the **entire** dataset (all 49 time steps in the real data).

| field | type | shape | notes |
|---|---|---|---|
| `x` | `FloatTensor` | `[num_nodes, 165]` | node features. **Does not include txId or time_step** — those are separate fields below. In the real dataset `num_nodes = 203,769`. |
| `edge_index` | `LongTensor` | `[2, num_edges]` | PyG format, 0-indexed into `x`/`y` (contiguous, NOT the original txId). Directed as given in the raw edgelist. |
| `y` | `LongTensor` | `[num_nodes]` | `1`/`0`/`-1` per the convention above. In the real dataset ~2% illicit, ~21% licit, ~77% unknown. |
| `time_step` | `LongTensor` | `[num_nodes]` | integer, `1..49` in the real data. |
| `tx_id` | `LongTensor` | `[num_nodes]` | original Elliptic transaction id — use this only for traceability/debugging, never as a model input. |
| `num_nodes` | `int` | scalar | convenience, equals `x.size(0)`. |

Feature dim breakdown (for context, not enforced by code): of the 165 columns, the first 93 are "local" transaction features, the remaining 72 are aggregated features from 1-hop neighbors, as defined by the original Elliptic paper. The loader does not split these — you get all 165 as one block.

## `get_time_step_subgraph(data, step) -> Data`

Induced subgraph: only nodes with `time_step == step`, plus edges where **both** endpoints are in that time step. Cross-time-step edges are dropped.

**Important: indices are re-based to `[0, num_nodes_in_step)` for this subgraph.** They do NOT line up with indices into the full graph. Use the subgraph's own `tx_id` field if you need to trace a node back to the original transaction.

Returns a `Data` object with the same field set as above (`x`, `edge_index`, `y`, `time_step`, `tx_id`, `num_nodes`), just restricted to one time step.

## `get_normal_nodes(data, step) -> LongTensor`

Shape: `[num_normal_nodes]`. **Local indices** (into the subgraph you'd get from `get_time_step_subgraph(data, step)`) of licit (`y == 0`) nodes only. Unknown and illicit nodes excluded.

Use this to train the diffusion model on normal-only behavior (Person B).

## `get_labeled_eval_nodes(data, step) -> LongTensor`

Shape: `[num_labeled_nodes]`. **Local indices** (same indexing space as above) of licit + illicit nodes (`y != -1`). Unknown nodes excluded.

Use this for AUROC / AUPRC / forgetting-measure evaluation (Person C) — you have ground truth for every node this function returns.

`get_normal_nodes(...)` result is always a strict subset of `get_labeled_eval_nodes(...)` result, for the same `step`.

## Gotchas

- **Index spaces don't mix.** Indices from `get_normal_nodes`/`get_labeled_eval_nodes` are local to that time step's subgraph, not the full graph. Don't use them to index into `data.x` directly — call `get_time_step_subgraph(data, step)` first and index into *that*.
- Some raw edges reference txIds not present in the features file; the loader drops these and prints a warning with the count. This is expected and matches known quirks of the public Elliptic release.
- `time_step` values are 1-indexed to match the paper/Kaggle convention (not 0-indexed).
- Real dataset class imbalance is severe (~2% illicit among labeled nodes). Don't be alarmed if a given time step has very few illicit examples — that's expected, and part of why continual learning + anomaly-style detection (rather than plain supervised classification) is the point of this project.

## Tested against

`data/loader.py` includes a `__main__` smoke test (`python -m data.loader <raw_dir>`) and has been verified against synthetic data with the exact real-data schema for: shape correctness, time-step partitioning correctness, label-filtering correctness, and normal-nodes-subset-of-eval-nodes invariant. Run it against the real CSVs once available and sanity-check the printed counts against the known dataset stats above (49 steps, ~2% illicit).
