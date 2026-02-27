"""
skeleton_train_validate.py
==========================
Training and validation utilities for the Taylor coefficient regression model.

Exports
-------
- WeightedMSELoss                       — per-coefficient weighted MSE
- build_criterion(weights, ...)         -> WeightedMSELoss
- build_optimizer(model, lr)            -> Adam
- build_scheduler(optimizer, ...)       -> ReduceLROnPlateau
- train_epoch(model, loader, ...)       -> float   (avg weighted MSE)
- validate(model, val_loader, ...)      -> dict    (all metrics)

Loss weighting
--------------
COEFF_WEIGHTS is a length-5 list [w0, w1, w2, w3, w4].
All weights default to 1.0 (= standard MSE).  Increase a weight to penalise
errors on that coefficient position more heavily during training.
This is passed as a hyperparameter from skeleton_main.py.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from skeleton_metric import compute_all_metrics, NUM_COEFFS


# ---------------------------------------------------------------------------
# Weighted MSE loss
# ---------------------------------------------------------------------------

class WeightedMSELoss(nn.Module):
    """
    Per-coefficient weighted MSE.

        loss = mean_over_batch(
                   sum_k( weights_norm[k] * (pred[k] - target[k])^2 )
               )

    where weights_norm = weights / sum(weights)  (internally normalised).

    Default: all weights 1.0  →  identical to standard MSE.

    Parameters
    ----------
    weights    : (NUM_COEFFS,) raw weight tensor (will be normalised)
    num_coeffs : number of output coefficients (default 5)
    """

    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        num_coeffs: int = NUM_COEFFS,
    ):
        super().__init__()
        if weights is None:
            weights = torch.ones(num_coeffs)
        # Store normalised weights as a buffer (moved to device automatically)
        self.register_buffer("weights", weights / weights.sum())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (B, NUM_COEFFS) — model output (normalised space during training)
        target : (B, NUM_COEFFS) — normalised ground-truth coefficients

        Returns
        -------
        scalar loss tensor
        """
        sq_err   = (pred - target) ** 2                      # (B, NUM_COEFFS)
        weighted = (sq_err * self.weights).sum(dim=1)        # (B,)
        return weighted.mean()


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def build_criterion(
    weights: Optional[List[float]] = None,
    num_coeffs: int = NUM_COEFFS,
    device: str = "cpu",
) -> WeightedMSELoss:
    """
    Return a WeightedMSELoss.

    Parameters
    ----------
    weights    : list of NUM_COEFFS floats — per-coefficient loss weights
                 (default: all 1.0 = standard MSE)
                 Example: [1, 1, 2, 1, 1] penalises coeff2 errors twice as much.
    num_coeffs : number of output coefficients
    device     : torch device string

    Returns
    -------
    WeightedMSELoss
    """
    if weights is not None:
        w = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        w = torch.ones(num_coeffs, device=device)
    return WeightedMSELoss(w, num_coeffs)


def build_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Adam:
    """Return an Adam optimiser."""
    return torch.optim.Adam(model.parameters(), lr=lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float = 0.5,
    patience: int = 3,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Return a ReduceLROnPlateau scheduler tracking validation loss."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: WeightedMSELoss,
    device: torch.device,
    clip_grad: float = 1.0,
) -> float:
    """
    One full training pass (teacher-forced / direct regression).

    Parameters
    ----------
    model     : CoeffPredictorTransformer
    loader    : training DataLoader  — yields (src, coeffs, src_lens)
    optimizer : configured optimiser
    criterion : WeightedMSELoss (targets are in normalised space)
    device    : target device
    clip_grad : max gradient norm  (0 → disabled)

    Returns
    -------
    float — weighted-average MSE loss for the epoch
    """
    model.train()
    total_loss    = 0.0
    total_samples = 0

    for src, coeffs, _src_lens in tqdm(loader, desc="Train", leave=False):
        src    = src.to(device)
        coeffs = coeffs.to(device)       # (B, NUM_COEFFS) — normalised targets

        optimizer.zero_grad()

        pred = model(src)                # (B, NUM_COEFFS) — normalised predictions
        loss = criterion(pred, coeffs)

        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        B = src.shape[0]
        total_loss    += loss.item() * B
        total_samples += B

    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: WeightedMSELoss,
    device: torch.device,
    normaliser=None,                           # CoeffNormaliser | None
    coeff_weights: Optional[torch.Tensor] = None,
    tolerance: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate the model on the full validation set.

    Two kinds of metrics are computed:
    1. val_loss   — weighted MSE in normalised space (same scale as training loss)
    2. All other metrics (MSE, MAE, accuracy) — in de-normalised integer space

    Parameters
    ----------
    model          : CoeffPredictorTransformer
    val_loader     : validation DataLoader — yields (src, coeffs, src_lens)
    criterion      : WeightedMSELoss
    device         : evaluation device
    normaliser     : CoeffNormaliser to invert the hybrid normalisation before
                     metric computation.
                     coeff0-2 → inverse z-score  (x * std + mean)
                     coeff3-4 → inverse signed log  sign(z) * (exp(|z|) - 1)
                     Pass None if training without normalisation.
    coeff_weights  : (NUM_COEFFS,) tensor for MSE weighting in metrics
                     (independent of training weights — for reporting only)
    tolerance      : accuracy threshold; default 0.5 → must round to correct int

    Returns
    -------
    dict with keys:
        'val_loss'         — weighted MSE in hybrid-normalised space
        'overall_mse'      — de-normalised MSE in original integer space (weighted)
        'overall_mae'      — de-normalised MAE in original integer space
        'overall_acc'      — fraction of (sample, position) pairs correct
        'exact_match_acc'  — fraction of samples where ALL 5 coefficients correct
        'coeff{k}_mse'     — per-position MSE for k in 0..4
        'coeff{k}_mae'     — per-position MAE
        'coeff{k}_acc'     — per-position accuracy
    """
    model.eval()

    total_loss    = 0.0
    total_samples = 0
    all_preds:   List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for src, coeffs, _src_lens in val_loader:
            src    = src.to(device)
            coeffs = coeffs.to(device)

            pred = model(src)                   # (B, NUM_COEFFS) — normalised
            loss = criterion(pred, coeffs)

            B = src.shape[0]
            total_loss    += loss.item() * B
            total_samples += B

            # De-normalise before accumulating for metric computation
            if normaliser is not None:
                pred_dn   = normaliser.denormalise(pred.cpu())
                target_dn = normaliser.denormalise(coeffs.cpu())
            else:
                pred_dn   = pred.cpu()
                target_dn = coeffs.cpu()

            all_preds.append(pred_dn)
            all_targets.append(target_dn)

    val_loss   = total_loss / max(total_samples, 1)
    pred_all   = torch.cat(all_preds,   dim=0)   # (N, NUM_COEFFS)
    target_all = torch.cat(all_targets, dim=0)   # (N, NUM_COEFFS)

    metrics = compute_all_metrics(pred_all, target_all, coeff_weights, tolerance)
    return {"val_loss": val_loss, **metrics}


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from skeleton_model import CoeffPredictorTransformer

    VOCAB_SIZE  = 50
    B, S        = 16, 10
    NUM_BATCHES = 3

    model     = CoeffPredictorTransformer(vocab_size=VOCAB_SIZE, d_model=64, nhead=4,
                                          num_encoder_layers=2, dim_feedforward=128)
    criterion = build_criterion(weights=None)       # all weights = 1.0
    optimizer = build_optimizer(model, lr=1e-3)
    scheduler = build_scheduler(optimizer)

    # Fake DataLoader
    def fake_loader(n_batches):
        for _ in range(n_batches):
            src    = torch.randint(1, VOCAB_SIZE, (B, S))
            coeffs = torch.randn(B, NUM_COEFFS)    # normalised targets
            lens   = torch.full((B,), S, dtype=torch.long)
            yield src, coeffs, lens

    device = torch.device("cpu")

    train_loss = train_epoch(model, fake_loader(NUM_BATCHES), optimizer, criterion, device)
    print(f"train_loss : {train_loss:.4f}")

    val_metrics = validate(model, fake_loader(NUM_BATCHES), criterion, device)
    print(f"val_loss   : {val_metrics['val_loss']:.4f}")
    print(f"overall_mse: {val_metrics['overall_mse']:.4f}")
    print(f"exact_match: {val_metrics['exact_match_acc']:.4f}")

    from skeleton_metric import print_metrics
    print_metrics(1, train_loss, val_metrics)

    print("All checks passed.")
