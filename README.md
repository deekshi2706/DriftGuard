# Elliptic Graph Diffusion — Continual Anomaly Detection

Team split, contracts, and how the pieces fit together.

```
data/
  loader.py          Person A — load_elliptic_graph, get_time_step_subgraph,
                      get_normal_nodes, get_labeled_eval_nodes
  DATA_SPEC.md        Person A — exact shapes/semantics; read this before touching loader.py
models/
  diffusion.py        Person B — GraphDiffusionModel (forward, train_step, compute_anomaly_score)
training/
  continual.py        Person C — ReplayBuffer, ContinualTrainer (wraps Model B, no internals touched)
eval/
  metrics.py          Person C — compute_auroc, compute_auprc, forgetting_measure
```

## Status

All three contracts are implemented and tested end-to-end against synthetic
data matching the real Elliptic CSV schema (`raw/`, generated for testing —
**replace with your real Kaggle download before running for real**). See
`test_full_pipeline.py` for the integration test that trains Person B's
model continually across 5 synthetic time steps using Person C's trainer,
sourcing data entirely through Person A's loader, and computes AUROC/AUPRC/
forgetting.

## To run against the real dataset

1. Download the Elliptic Bitcoin dataset (Kaggle: "Elliptic Data Set") —
   you said you already have it. Put the three raw CSVs in one folder,
   unmodified:
   ```
   elliptic_txs_features.csv
   elliptic_txs_classes.csv
   elliptic_txs_edgelist.csv
   ```
2. `python -m data.loader /path/to/that/folder` — sanity check: should
   report 49 time steps and ~203,769 total nodes.
3. `python -m models.diffusion /path/to/that/folder` — trains a tiny model
   on time step 1's licit nodes and prints licit vs illicit anomaly scores.
   On the real data you should see a real (larger) gap between the two
   means — the synthetic test data is random noise, so its "signal" is
   coincidental.
4. Wire up `test_full_pipeline.py`-style code in `notebooks/experiments.ipynb`
   for the actual 3-way comparison (static / naive continual / continual+replay):
   - **static**: one `GraphDiffusionModel`, call `.train_step()` repeatedly
     on step 1's normal nodes only, never call `ContinualTrainer.update()`
     again after that.
   - **naive continual**: `ContinualTrainer(model, replay_buffer_size=0)`,
     call `.update()` once per new time step.
   - **continual + replay**: `ContinualTrainer(model, replay_buffer_size=N)`
     for some N (start with 3–5), same loop.
   - Evaluate all three the same way: after training through step *i*, call
     `compute_anomaly_score` on `get_labeled_eval_nodes(g, j)` for every
     `j <= i`, run `evaluate_step`, and feed the resulting AUROC grid into
     `forgetting_measure`.

## Division of remaining work

- **Person A**: loader + DATA_SPEC are done. Remaining: exploratory
  analysis notebook (class imbalance stats, feature distributions,
  time-step-by-time-step node/edge counts) for the paper's dataset section
  — this doesn't block B or C.
- **Person B**: static MVP architecture is implemented and runs. Remaining:
  actually tune it on the real data (GCN vs GAT, `num_diffusion_steps`,
  `hidden_dim`, noise schedule) and save a real checkpoint via
  `save_checkpoint(model, path, trained_on_step=1)` for Person C to
  benchmark against.
- **Person C**: continual trainer + metrics are implemented and integration-
  tested. Remaining: run the actual 3-way comparison on the real data once
  Person B's tuned static checkpoint exists, and produce the paper's result
  tables/figures.

## Known limitations to flag in the paper

- `ContinualTrainer.update()` currently does one gradient step (or `epochs`
  steps) per call on whatever batch it's given — for the real dataset with
  much bigger per-time-step node counts, consider mini-batching within a
  time step rather than passing the whole step in as one batch.
- `forgetting_measure` expects AUROC (or any "higher is better") values; if
  you swap in a "lower is better" metric, you'll need to flip the sign
  convention inside it.
- `ReplayBuffer` stores whole time-step subgraphs, not sampled individual
  nodes — with real data sizes you likely want to cap how many nodes per
  step get buffered (e.g. random subsample before `.add()`), not just cap
  the number of time steps retained.
