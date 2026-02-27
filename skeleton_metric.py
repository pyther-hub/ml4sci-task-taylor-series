"""
skeleton_metric.py
==================
Evaluation metrics for Taylor coefficient regression.

The model predicts the 5 integer numerators [a0, a1, a2, a3, a4] of a Taylor
series.  Denominators are the fixed factorials [1, 1, 2, 6, 24].

Metrics
-------
Per coefficient position AND overall:
  - MSE      : mean squared error (continuous predictions vs rounded int targets)
  - MAE      : mean absolute error
  - Accuracy : fraction of samples where the prediction rounds to the correct integer
                (i.e. |round(pred) - target| <= tolerance, default tolerance=0.5)

Reported as:
  coeff0_mse, coeff1_mse, ..., coeff4_mse, overall_mse
  coeff0_mae, coeff1_mae, ..., coeff4_mae, overall_mae
  coeff0_acc, coeff1_acc, ..., coeff4_acc, overall_acc, exact_match_acc

where "coeff0" = f(0), "coeff1" = f'(0), ..., "coeff4" = f''''(0).
"""

from typing import Dict, List, Optional

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_COEFFS   = 5
COEFF_LABELS = ["coeff0", "coeff1", "coeff2", "coeff3", "coeff4"]


# ---------------------------------------------------------------------------
# Per-coefficient MSE (with optional weighting)
# ---------------------------------------------------------------------------

def coeff_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Per-coefficient and overall weighted MSE.

    Parameters
    ----------
    pred    : (B, NUM_COEFFS) — model predictions in de-normalised space
    target  : (B, NUM_COEFFS) — ground-truth rounded integer values
    weights : (NUM_COEFFS,)   — per-coefficient weights (default all 1.0)
                                Hyperparameter — change to emphasise specific
                                coefficient positions.

    Returns
    -------
    dict
        Keys : 'coeff0_mse' … 'coeff4_mse' (per position)
               'overall_mse'               (weighted average)
    """
    if weights is None:
        weights = torch.ones(pred.shape[-1], dtype=torch.float32)

    sq_err    = (pred - target) ** 2          # (B, NUM_COEFFS)
    per_coeff = sq_err.mean(dim=0)            # (NUM_COEFFS,)
    overall   = (per_coeff * weights).sum() / weights.sum()

    result = {
        f"{COEFF_LABELS[k]}_mse": per_coeff[k].item()
        for k in range(NUM_COEFFS)
    }
    result["overall_mse"] = overall.item()
    return result


# ---------------------------------------------------------------------------
# Per-coefficient MAE
# ---------------------------------------------------------------------------

def coeff_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    Per-coefficient and overall MAE.

    Parameters
    ----------
    pred   : (B, NUM_COEFFS)
    target : (B, NUM_COEFFS)

    Returns
    -------
    dict
        Keys : 'coeff0_mae' … 'coeff4_mae' and 'overall_mae'
    """
    abs_err   = (pred - target).abs()         # (B, NUM_COEFFS)
    per_coeff = abs_err.mean(dim=0)           # (NUM_COEFFS,)

    result = {
        f"{COEFF_LABELS[k]}_mae": per_coeff[k].item()
        for k in range(NUM_COEFFS)
    }
    result["overall_mae"] = abs_err.mean().item()
    return result


# ---------------------------------------------------------------------------
# Per-coefficient accuracy
# ---------------------------------------------------------------------------

def coeff_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    tolerance: float = 0.5,
) -> Dict[str, float]:
    """
    Per-coefficient and overall integer-accuracy.

    A prediction at position k is "correct" if |pred_k - target_k| <= tolerance.
    With tolerance=0.5 this means pred must round to the same integer as target.

    Parameters
    ----------
    pred      : (B, NUM_COEFFS) — de-normalised float predictions
    target    : (B, NUM_COEFFS) — ground-truth rounded integers (as floats)
    tolerance : correctness threshold (default 0.5)

    Returns
    -------
    dict
        Keys : 'coeff0_acc' … 'coeff4_acc'  (per position)
               'overall_acc'                (average over all positions)
               'exact_match_acc'            (fraction where ALL 5 are correct)
    """
    correct     = (pred - target).abs() <= tolerance   # (B, NUM_COEFFS)
    per_coeff   = correct.float().mean(dim=0)          # (NUM_COEFFS,)
    exact_match = correct.all(dim=1).float().mean()    # scalar

    result = {
        f"{COEFF_LABELS[k]}_acc": per_coeff[k].item()
        for k in range(NUM_COEFFS)
    }
    result["overall_acc"]     = correct.float().mean().item()
    result["exact_match_acc"] = exact_match.item()
    return result


# ---------------------------------------------------------------------------
# Combined metric entry point
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    tolerance: float = 0.5,
) -> Dict[str, float]:
    """
    Compute MSE, MAE, and accuracy in a single call.

    Parameters
    ----------
    pred      : (B, NUM_COEFFS) — de-normalised float predictions
    target    : (B, NUM_COEFFS) — de-normalised ground truth (rounded integers)
    weights   : (NUM_COEFFS,)   — optional per-coefficient MSE weights
    tolerance : accuracy threshold (default 0.5 → round to correct integer)

    Returns
    -------
    dict — union of coeff_mse / coeff_mae / coeff_accuracy results
    """
    mse = coeff_mse(pred, target, weights)
    mae = coeff_mae(pred, target)
    acc = coeff_accuracy(pred, target, tolerance)
    return {**mse, **mae, **acc}


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_metrics(
    epoch: int,
    train_loss: float,
    val_metrics: Dict[str, float],
) -> None:
    """Print a nicely formatted per-epoch summary."""
    sep = "─" * 72
    print(sep)
    print(f"  Epoch {epoch:>4d}")
    print(sep)
    print(f"  {'Train loss (weighted MSE)':<34s}: {train_loss:.6f}")
    print(f"  {'Val loss  (weighted MSE)':<34s}: {val_metrics.get('val_loss', float('nan')):.6f}")
    print(f"  {'Val overall MSE':<34s}: {val_metrics.get('overall_mse', float('nan')):.6f}")
    print(f"  {'Val overall MAE':<34s}: {val_metrics.get('overall_mae', float('nan')):.6f}")
    print(f"  {'Val overall accuracy':<34s}: {val_metrics.get('overall_acc', float('nan')):.4f}")
    print(f"  {'Val exact-match accuracy':<34s}: {val_metrics.get('exact_match_acc', float('nan')):.4f}")
    print()
    print(f"  {'Position':<12s} {'MSE':>10s} {'MAE':>10s} {'Accuracy':>12s}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*12}")
    for k, label in enumerate(COEFF_LABELS):
        mse = val_metrics.get(f"{label}_mse", float("nan"))
        mae = val_metrics.get(f"{label}_mae", float("nan"))
        acc = val_metrics.get(f"{label}_acc", float("nan"))
        print(f"  {label:<12s} {mse:>10.4f} {mae:>10.4f} {acc:>12.4f}")
    print(sep)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B = 8

    # Simulate de-normalised predictions (floats near integers)
    target = torch.tensor(
        [[0, 1, 1, 1, 1],
         [0, 0, 2, 0, -12],
         [0, 7, 0, 0, 0],
         [1, -1, 2, -6, 24],
         [0, 0, 0, 6, 0],
         [2, 0, -2, 0, 8],
         [0, 1, 0, -6, 0],
         [1, 0, 2, 0, -24]],
        dtype=torch.float32,
    )
    # Add small noise to simulate imperfect predictions
    pred = target + torch.randn_like(target) * 0.3

    mse = coeff_mse(pred, target)
    mae = coeff_mae(pred, target)
    acc = coeff_accuracy(pred, target, tolerance=0.5)
    all_m = compute_all_metrics(pred, target)

    print("=== coeff_mse ===")
    for k, v in mse.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== coeff_mae ===")
    for k, v in mae.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== coeff_accuracy ===")
    for k, v in acc.items():
        print(f"  {k}: {v:.4f}")

    print()
    print_metrics(
        epoch=1,
        train_loss=0.42,
        val_metrics={"val_loss": 0.38, **all_m},
    )
