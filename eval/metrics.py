"""
eval/metrics.py
Person C — Evaluation

Metrics for the 3-way comparison (static vs naive continual vs
continual+replay): AUROC, AUPRC, and a forgetting measure across time
steps.

Ground truth for these metrics must come from Person A's
`get_labeled_eval_nodes(data, step)` (licit + illicit only — never pass
unknown-labeled nodes in here, scores/labels must line up 1:1 and every
label must be 0 or 1).

Public API:
    compute_auroc(scores, labels) -> float
    compute_auprc(scores, labels) -> float
    evaluate_step(scores, labels) -> dict with 'auroc', 'auprc'
    forgetting_measure(history) -> dict with 'forgetting' (float) and 'per_step_max_drop' (list)
"""

from typing import Dict, List, Sequence, Union

import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ArrayLike = Union[torch.Tensor, Sequence[float]]


def _to_numpy(a: ArrayLike):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return a


def compute_auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    """
    scores: anomaly scores, higher = more anomalous (e.g. from
        GraphDiffusionModel.compute_anomaly_score).
    labels: ground truth, 1=illicit, 0=licit. Must contain at least one of
        each class or AUROC is undefined.
    """
    s, y = _to_numpy(scores), _to_numpy(labels)
    if len(set(y.tolist())) < 2:
        raise ValueError(
            "AUROC undefined: labels contain only one class for this slice "
            "(check you filtered to get_labeled_eval_nodes and that this "
            "time step actually has both licit and illicit examples)."
        )
    return float(roc_auc_score(y, s))


def compute_auprc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Average precision (area under precision-recall curve), positive class = illicit (1)."""
    s, y = _to_numpy(scores), _to_numpy(labels)
    return float(average_precision_score(y, s))


def evaluate_step(scores: ArrayLike, labels: ArrayLike) -> Dict[str, float]:
    """Convenience wrapper: both metrics for one time step in one call."""
    return {
        "auroc": compute_auroc(scores, labels),
        "auprc": compute_auprc(scores, labels),
    }


def forgetting_measure(history: List[List[float]]) -> Dict[str, object]:
    """
    Standard continual-learning forgetting measure.

    Args:
        history: history[i][j] = performance metric (e.g. AUROC) on time
            step j, measured right after training/updating through time
            step i. Must be a "lower triangular"-ish structure: only
            j <= i entries are meaningful (you can't evaluate on a step
            not yet trained through) — pass 0.0 or None for j > i, they
            are ignored.

            Example for 3 time steps:
                history = [
                    [0.81, None, None],   # after training on step 0
                    [0.74, 0.83, None],   # after training on step 0,1
                    [0.69, 0.79, 0.85],   # after training on step 0,1,2
                ]

    Returns:
        {
          "forgetting": float,             # average forgetting across all steps
                                            # except the last (final) one
          "per_step_max_drop": [float,...] # for each step j, the largest
                                            # drop in performance on step j
                                            # observed at any later point vs
                                            # its best-ever performance
        }
    """
    n = len(history)
    per_step_max_drop = []
    for j in range(n - 1):  # last step has no "later" evaluations to forget against
        perf_j = [history[i][j] for i in range(j, n) if history[i][j] is not None]
        if len(perf_j) < 2:
            per_step_max_drop.append(0.0)
            continue
        best_so_far = perf_j[0]
        max_drop = 0.0
        for later_perf in perf_j[1:]:
            best_so_far = max(best_so_far, later_perf) if later_perf > best_so_far else best_so_far
            # forgetting = drop from the best performance seen so far on step j
            # to the current (later) performance on step j
            drop = best_so_far - later_perf
            max_drop = max(max_drop, drop)
            best_so_far = max(best_so_far, perf_j[0])
        per_step_max_drop.append(max_drop)

    avg_forgetting = sum(per_step_max_drop) / len(per_step_max_drop) if per_step_max_drop else 0.0
    return {"forgetting": avg_forgetting, "per_step_max_drop": per_step_max_drop}
