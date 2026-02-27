"""
skeleton_main.py
================
End-to-end training pipeline for Taylor coefficient regression.

Task
----
Given a mathematical function in prefix notation (e.g. "sin x"), predict the
5 integer numerators [a0, a1, a2, a3, a4] of its Taylor expansion around 0:

    f(x) ≈ a0 + a1·x + (a2/2!)·x² + (a3/3!)·x³ + (a4/4!)·x⁴

Denominators [1, 1, 2, 6, 24] are fixed; we predict ONLY the numerators.

Usage
-----
Edit the CONFIG section below and run:
    python skeleton_main.py
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from dataset import WordLevelBPETokenizer
from skeleton_dataset import CoeffDataset, CoeffNormaliser
from skeleton_model import CoeffPredictorTransformer
from skeleton_metric import COEFF_LABELS, print_metrics
from skeleton_train_validate import (
    build_criterion,
    build_optimizer,
    build_scheduler,
    train_epoch,
    validate,
)


# ============================================================
# CONFIG  — edit everything here; do not change code below
# ============================================================

# --- Paths ---
DATASET_JSON_PATH    = "datasets/taylor_dataset_10k.json"
TOKENIZER_SAVE_PATH  = "checkpoints/coeff_tokenizer.pkl"
NORMALISER_SAVE_PATH = "checkpoints/coeff_normaliser.pkl"
CHECKPOINT_DIR       = "checkpoints"
BEST_MODEL_PATH      = os.path.join(CHECKPOINT_DIR, "best_coeff_model.pt")

# --- Data split ---
VAL_RATIO   = 0.1    # 15 % of data for validation
RANDOM_SEED = 42

# --- Tokenizer ---
NUM_MERGES = 200      # BPE merge operations

# --- Model architecture ---
D_MODEL            = 256    # embedding / model dimension (must be divisible by NHEAD)
NHEAD              = 8      # attention heads
NUM_ENCODER_LAYERS = 3      # Transformer encoder layers
DIM_FEEDFORWARD    = 512    # inner FFN dimension
DROPOUT            = 0.1
MLP_HIDDEN         = 256    # hidden dim of the regression head MLP

# --- Training ---
BATCH_SIZE     = 32
NUM_EPOCHS     = 50
LEARNING_RATE  = 1e-3
CLIP_GRAD      = 1.0        # max gradient norm (0 → disabled)
VALIDATE_EVERY = 1          # run validation every N epochs

# --- LR scheduler ---
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR   = 1e-6

# --- Cosine LR annealing (applied every epoch alongside the plateau scheduler) ---
LR_COSINE_DECAY = True

# --- Loss weights (one per coefficient, length 5) ---
# All 1.0 → standard MSE.  Increase a value to penalise that position more.
# Example: [1, 1, 2, 1, 1] weights coeff2 errors twice as heavily.
COEFF_WEIGHTS: Optional[List[float]] = None   # None → all 1.0

# --- Accuracy tolerance ---
# A prediction is "correct" at position k if |pred_k - target_k| <= TOLERANCE.
# TOLERANCE=0.5 means the prediction must round to the correct integer.
TOLERANCE: float = 0.5

# --- Best-model selection criterion ---
BEST_MODEL_METRIC   = "val_loss"   # "val_loss" or any metric key from validate()
BEST_MODEL_MINIMIZE = True         # True → lower is better

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_ratio: float, seed: int):
    """Return (train_indices, val_indices) with a reproducible random split."""
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    val_size = max(1, int(n * val_ratio))
    return indices[: n - val_size], indices[n - val_size :]


def is_better(new_val: float, best: float, minimize: bool) -> bool:
    return new_val < best if minimize else new_val > best


def initial_best(minimize: bool) -> float:
    return float("inf") if minimize else float("-inf")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(RANDOM_SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(DEVICE)

    print(f"\n{'='*65}")
    print(f"  Taylor Coefficient Regression")
    print(f"  Device   : {device}")
    print(f"  Dataset  : {DATASET_JSON_PATH}")
    print(f"  Epochs   : {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"  Weights  : {COEFF_WEIGHTS or '[1, 1, 1, 1, 1] (default)'}")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    # 1. Tokenizer (input side only — no decoder vocabulary needed)
    # ------------------------------------------------------------------
    tokenizer = WordLevelBPETokenizer(num_merges=NUM_MERGES)

    if os.path.exists(TOKENIZER_SAVE_PATH):
        print(f"[tokenizer] Loading from '{TOKENIZER_SAVE_PATH}' ...")
        tokenizer.load(TOKENIZER_SAVE_PATH)
    else:
        print(f"[tokenizer] Fitting on '{DATASET_JSON_PATH}' ...")
        raw_data = json.load(open(DATASET_JSON_PATH))
        tokenizer.fit([item["function_prefix"] for item in raw_data])
        tokenizer.save(TOKENIZER_SAVE_PATH)
        print(f"[tokenizer] Saved to '{TOKENIZER_SAVE_PATH}'")

    print(f"[tokenizer] Vocab size : {len(tokenizer)}\n")

    # ------------------------------------------------------------------
    # 2. Normaliser (fit ONLY on the training split — no data leakage)
    # ------------------------------------------------------------------
    normaliser = CoeffNormaliser()

    if os.path.exists(NORMALISER_SAVE_PATH):
        print(f"[normaliser] Loading from '{NORMALISER_SAVE_PATH}' ...")
        normaliser.load(NORMALISER_SAVE_PATH)
    else:
        # Load raw coefficients without normalisation to compute stats
        temp_ds    = CoeffDataset(DATASET_JSON_PATH, tokenizer, normaliser=None)
        train_idx, _ = split_indices(len(temp_ds), VAL_RATIO, RANDOM_SEED)
        train_coeffs = [temp_ds.coeffs[i] for i in train_idx]
        normaliser.fit(train_coeffs)
        normaliser.save(NORMALISER_SAVE_PATH)
        del temp_ds
        print(f"[normaliser] Fitted on {len(train_coeffs)} training samples")
        print(f"[normaliser] Mean : {normaliser.mean.tolist()}")
        print(f"[normaliser] Std  : {normaliser.std.tolist()}\n")

    # ------------------------------------------------------------------
    # 3. Dataset and dataloaders
    # ------------------------------------------------------------------
    full_ds = CoeffDataset(DATASET_JSON_PATH, tokenizer, normaliser=normaliser)
    N = len(full_ds)

    train_idx, val_idx = split_indices(N, VAL_RATIO, RANDOM_SEED)
    print(f"[data] Total : {N}  |  Train : {len(train_idx)}  |  Val : {len(val_idx)}\n")

    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=full_ds.collate_fn,
    )
    val_loader = DataLoader(
        Subset(full_ds, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=full_ds.collate_fn,
    )

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model = CoeffPredictorTransformer(
        vocab_size         = len(tokenizer),
        d_model            = D_MODEL,
        nhead              = NHEAD,
        num_encoder_layers = NUM_ENCODER_LAYERS,
        dim_feedforward    = DIM_FEEDFORWARD,
        dropout            = DROPOUT,
        src_pad_id         = tokenizer.pad_id,
        mlp_hidden         = MLP_HIDDEN,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Trainable parameters : {n_params:,}\n")

    # ------------------------------------------------------------------
    # 5. Training components
    # ------------------------------------------------------------------
    criterion = build_criterion(COEFF_WEIGHTS, device=str(device))
    optimizer = build_optimizer(model, lr=LEARNING_RATE)
    scheduler = build_scheduler(
        optimizer,
        factor   = SCHEDULER_FACTOR,
        patience = SCHEDULER_PATIENCE,
        min_lr   = SCHEDULER_MIN_LR,
    )
    cosine_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS, eta_min=SCHEDULER_MIN_LR
        )
        if LR_COSINE_DECAY else None
    )

    # Tensor version of weights for metric reporting (not training)
    coeff_weights_tensor = (
        torch.tensor(COEFF_WEIGHTS, dtype=torch.float32)
        if COEFF_WEIGHTS else None
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    best_val = initial_best(BEST_MODEL_MINIMIZE)
    history  = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, CLIP_GRAD
        )
        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:>4d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.5f}  "
            f"lr={current_lr:.2e}  "
            f"{elapsed:.1f}s",
            end="",
        )

        if epoch % VALIDATE_EVERY == 0:
            val_metrics = validate(
                model,
                val_loader,
                criterion,
                device,
                normaliser        = normaliser,
                coeff_weights     = coeff_weights_tensor,
                tolerance         = TOLERANCE,
            )

            print()   # newline after train line
            print_metrics(epoch, train_loss, val_metrics)

            # ReduceLROnPlateau step (based on validation loss)
            scheduler.step(val_metrics["val_loss"])

            # Save best model
            current = val_metrics[BEST_MODEL_METRIC]
            if is_better(current, best_val, BEST_MODEL_MINIMIZE):
                best_val = current
                torch.save(
                    {
                        "epoch":             epoch,
                        "model_state_dict":  model.state_dict(),
                        "optimizer_state":   optimizer.state_dict(),
                        "scheduler_state":   scheduler.state_dict(),
                        "cosine_state":      cosine_scheduler.state_dict() if cosine_scheduler else None,
                        "val_metrics":       val_metrics,
                        "train_loss":        train_loss,
                        "normaliser_mean":   normaliser.mean,
                        "normaliser_std":    normaliser.std,
                        "config": {
                            "vocab_size":          len(tokenizer),
                            "d_model":             D_MODEL,
                            "nhead":               NHEAD,
                            "num_encoder_layers":  NUM_ENCODER_LAYERS,
                            "dim_feedforward":     DIM_FEEDFORWARD,
                            "dropout":             DROPOUT,
                            "mlp_hidden":          MLP_HIDDEN,
                        },
                    },
                    BEST_MODEL_PATH,
                )
                dir_sign = "↓" if BEST_MODEL_MINIMIZE else "↑"
                print(
                    f"  ★ New best {BEST_MODEL_METRIC}: "
                    f"{current:.6f} {dir_sign}  → saved to '{BEST_MODEL_PATH}'"
                )

            history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        else:
            print()   # close the train-only line

        # Cosine annealing step (every epoch)
        if cosine_scheduler is not None:
            cosine_scheduler.step()

    # ------------------------------------------------------------------
    # 7. Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Training complete.")
    print(f"  Best {BEST_MODEL_METRIC} : {best_val:.6f}")
    print(f"  Checkpoint        : {BEST_MODEL_PATH}")
    print(f"{'='*65}\n")


# ============================================================
# INFERENCE HELPER — load best model and predict
# ============================================================

def load_best_model(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> CoeffPredictorTransformer:
    """
    Load the best saved model from a checkpoint.

    Example
    -------
    >>> model = load_best_model("checkpoints/best_coeff_model.pt")
    >>> model.eval()
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["config"]

    model = CoeffPredictorTransformer(
        vocab_size         = cfg["vocab_size"],
        d_model            = cfg["d_model"],
        nhead              = cfg["nhead"],
        num_encoder_layers = cfg["num_encoder_layers"],
        dim_feedforward    = cfg["dim_feedforward"],
        dropout            = cfg["dropout"],
        mlp_hidden         = cfg["mlp_hidden"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    # Reconstruct normaliser from checkpoint
    norm = CoeffNormaliser()
    norm.mean = ckpt.get("normaliser_mean")
    norm.std  = ckpt.get("normaliser_std")

    print(
        f"[load] Loaded epoch {ckpt['epoch']}  |  "
        f"val_loss={ckpt['val_metrics']['val_loss']:.5f}"
    )
    return model, norm


if __name__ == "__main__":
    main()
