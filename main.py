"""
main.py
=======
End-to-end training pipeline for the Taylor-expansion Seq2SeqLSTM model.

Usage
-----
Just edit the CONFIG section below and run:
    python main.py
"""

from __future__ import annotations

import os
import time
import json
import random
import tempfile
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from dataset import DualBPETokenizer, TaylorDataset
from model import Seq2SeqLSTM
from train_validate_model import (
    build_criterion,
    build_optimizer,
    build_scheduler,
    print_metrics,
    train_epoch,
    validate,
)

# ============================================================
# CONFIG  — edit everything here; do not change code below
# ============================================================

# --- Paths ---
DATASET_JSON_PATH   = "taylor_dataset_sample.json"   # single JSON file
TOKENIZER_SAVE_PATH = "dual_bpe_tokenizer.pkl"       # where to save/load tokenizer
CHECKPOINT_DIR      = "checkpoints"                   # directory for saved models
BEST_MODEL_PATH     = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# --- Data split ---
VAL_RATIO   = 0.15     # fraction of data used for validation (e.g. 0.15 = 15 %)
RANDOM_SEED = 42       # for reproducible splits

# --- Tokenizer ---
NUM_MERGES = 200       # BPE merge operations per side

# --- Model architecture ---
EMBEDDING_DIM = 256
HIDDEN_DIM    = 512
NUM_LAYERS    = 2
DROPOUT       = 0.3

# --- Training ---
BATCH_SIZE           = 32
NUM_EPOCHS           = 50
LEARNING_RATE        = 1e-3
CLIP_GRAD            = 1.0       # max gradient norm (0 → disabled)
VALIDATE_AFTER_EPOCH = 1         # run validation every N epochs

# --- LR scheduler (ReduceLROnPlateau) ---
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR   = 1e-6

# --- Validation / generation ---
BEAM_WIDTH            = 3
MAX_GEN_LEN           = 128
COMPUTE_FUNCTIONAL    = True     # set False to skip sympy / numerical metrics

# --- Best-model selection criterion ---
# Options: "val_loss" (lower is better) or any metric where HIGHER is better:
#   "token_acc", "sentence_acc", "strict_sentence_acc", "prefix_acc"
BEST_MODEL_METRIC   = "val_loss"
BEST_MODEL_MINIMIZE = True       # True → lower is better; False → higher is better

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
    """Return (train_indices, val_indices) with a fixed random split."""
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    val_size   = max(1, int(n * val_ratio))
    train_size = n - val_size
    return indices[:train_size], indices[train_size:]


def make_subset_loader(
    dataset: TaylorDataset,
    indices,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )


def is_better(new_val: float, best_val: float, minimize: bool) -> bool:
    if minimize:
        return new_val < best_val
    return new_val > best_val


def initial_best_value(minimize: bool) -> float:
    return float("inf") if minimize else float("-inf")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(RANDOM_SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(DEVICE)
    print(f"\n{'='*65}")
    print(f"  Device : {device}")
    print(f"  Config : {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    # 1. Fit or load tokenizer
    # ------------------------------------------------------------------
    tokenizer = DualBPETokenizer(num_merges=NUM_MERGES)

    if os.path.exists(TOKENIZER_SAVE_PATH):
        print(f"[tokenizer] Loading from '{TOKENIZER_SAVE_PATH}' …")
        tokenizer.load(TOKENIZER_SAVE_PATH)
    else:
        print(f"[tokenizer] Fitting on '{DATASET_JSON_PATH}' …")
        tokenizer.fit(DATASET_JSON_PATH)
        tokenizer.save(TOKENIZER_SAVE_PATH)
        print(f"[tokenizer] Saved to '{TOKENIZER_SAVE_PATH}'")

    print(f"[tokenizer] Input  vocab size : {tokenizer.input_vocab_size}")
    print(f"[tokenizer] Output vocab size : {tokenizer.output_vocab_size}\n")

    # ------------------------------------------------------------------
    # 2. Load full dataset, then split into train / val
    # ------------------------------------------------------------------
    full_dataset = TaylorDataset(DATASET_JSON_PATH, tokenizer)
    N = len(full_dataset)

    train_idx, val_idx = split_indices(N, VAL_RATIO, RANDOM_SEED)
    print(f"[data] Total samples  : {N}")
    print(f"[data] Train samples  : {len(train_idx)}")
    print(f"[data] Val   samples  : {len(val_idx)}\n")

    train_loader = make_subset_loader(full_dataset, train_idx, BATCH_SIZE, shuffle=True)
    val_loader   = make_subset_loader(full_dataset, val_idx,   BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    model = Seq2SeqLSTM(
        input_vocab_size  = tokenizer.input_vocab_size,
        output_vocab_size = tokenizer.output_vocab_size,
        embedding_dim     = EMBEDDING_DIM,
        hidden_dim        = HIDDEN_DIM,
        num_layers        = NUM_LAYERS,
        dropout           = DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Trainable parameters : {n_params:,}\n")

    # ------------------------------------------------------------------
    # 4. Build training components
    # ------------------------------------------------------------------
    pad_id    = tokenizer.output_tokenizer.pad_id
    criterion = build_criterion(pad_id)
    optimizer = build_optimizer(model, lr=LEARNING_RATE)
    scheduler = build_scheduler(
        optimizer,
        factor   = SCHEDULER_FACTOR,
        patience = SCHEDULER_PATIENCE,
        min_lr   = SCHEDULER_MIN_LR,
    )

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    best_metric_val = initial_best_value(BEST_MODEL_MINIMIZE)
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # --- Train one epoch ---
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, CLIP_GRAD
        )

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>4d}/{NUM_EPOCHS}  |  train_loss={train_loss:.5f}  |  {elapsed:.1f}s", end="")

        # --- Validate every VALIDATE_AFTER_EPOCH epochs ---
        if epoch % VALIDATE_AFTER_EPOCH == 0:
            val_metrics = validate(
                model,
                val_loader,
                tokenizer,
                device,
                criterion,
                beam_width         = BEAM_WIDTH,
                max_gen_len        = MAX_GEN_LEN,
                compute_functional = COMPUTE_FUNCTIONAL,
            )

            print()  # newline after train line
            print_metrics(epoch, train_loss, val_metrics)

            # --- LR scheduler step (on val_loss) ---
            scheduler.step(val_metrics["val_loss"])

            # --- Save best model ---
            current_metric = val_metrics[BEST_MODEL_METRIC]
            if is_better(current_metric, best_metric_val, BEST_MODEL_MINIMIZE):
                best_metric_val = current_metric
                torch.save(
                    {
                        "epoch":            epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state":  optimizer.state_dict(),
                        "scheduler_state":  scheduler.state_dict(),
                        "val_metrics":      val_metrics,
                        "train_loss":       train_loss,
                        "config": {
                            "input_vocab_size":  tokenizer.input_vocab_size,
                            "output_vocab_size": tokenizer.output_vocab_size,
                            "embedding_dim":     EMBEDDING_DIM,
                            "hidden_dim":        HIDDEN_DIM,
                            "num_layers":        NUM_LAYERS,
                            "dropout":           DROPOUT,
                        },
                    },
                    BEST_MODEL_PATH,
                )
                dir_sign = "↓" if BEST_MODEL_MINIMIZE else "↑"
                print(
                    f"  ★ New best {BEST_MODEL_METRIC}: "
                    f"{current_metric:.6f} {dir_sign}  → saved to '{BEST_MODEL_PATH}'"
                )

            history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        else:
            print()   # close the train-only line

    # ------------------------------------------------------------------
    # 6. Final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Training complete.")
    print(f"  Best {BEST_MODEL_METRIC} : {best_metric_val:.6f}")
    print(f"  Checkpoint      : {BEST_MODEL_PATH}")
    print(f"{'='*65}\n")


# ============================================================
# LOADING A CHECKPOINT  (convenience helper)
# ============================================================

def load_best_model(checkpoint_path: str, device: Optional[str] = None) -> Seq2SeqLSTM:
    """Load and return the best saved model from a checkpoint file.

    Example
    -------
    >>> model = load_best_model("checkpoints/best_model.pt")
    >>> model.eval()
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["config"]
    model = Seq2SeqLSTM(
        input_vocab_size  = cfg["input_vocab_size"],
        output_vocab_size = cfg["output_vocab_size"],
        embedding_dim     = cfg["embedding_dim"],
        hidden_dim        = cfg["hidden_dim"],
        num_layers        = cfg["num_layers"],
        dropout           = cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[load] Loaded epoch {ckpt['epoch']} | "
          f"val_loss={ckpt['val_metrics']['val_loss']:.5f}")
    return model


if __name__ == "__main__":
    main()