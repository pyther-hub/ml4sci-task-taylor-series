"""
train_validate_model.py
=======================
Training and validation utilities for the Seq2SeqLSTM Taylor-expansion model.

Exports
-------
- build_criterion(pad_id)            -> nn.CrossEntropyLoss
- build_optimizer(model, lr)         -> torch.optim.Adam
- build_scheduler(optimizer, ...)    -> ReduceLROnPlateau
- train_epoch(...)                   -> float          (avg train loss)
- validate(...)                      -> dict           (all metrics)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# metric.py imports
# ---------------------------------------------------------------------------
from metric import (
    batch_function_closeness,
    length_accuracy,
    prefix_accuracy,
    sentence_accuracy,
    strict_sentence_accuracy,
    token_accuracy,
)

# ---------------------------------------------------------------------------
# sympy expression parser  (handles infix strings with '^' as power)
# ---------------------------------------------------------------------------
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

_SYMPY_TRANSFORMS = standard_transformations + (convert_xor,)
_X = sp.Symbol("x")


def _try_parse_expr(text: str) -> Optional[sp.Expr]:
    """Attempt to parse a whitespace-separated infix string into a SymPy Expr.

    Returns ``None`` on any parse failure so the caller can skip the sample.
    """
    try:
        return parse_expr(text, transformations=_SYMPY_TRANSFORMS, local_dict={"x": _X})
    except Exception:
        return None


# ============================================================
# 1. BUILD TRAINING COMPONENTS
# ============================================================

def build_criterion(pad_id: int) -> nn.CrossEntropyLoss:
    """Return a CrossEntropyLoss that ignores PAD tokens.

    ``reduction='mean'`` averages the loss over all non-PAD positions,
    giving an unbiased per-token loss regardless of sequence length.
    """
    return nn.CrossEntropyLoss(ignore_index=pad_id, reduction="mean")


def build_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Adam:
    """Return an Adam optimiser with the given learning rate."""
    return torch.optim.Adam(model.parameters(), lr=lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float = 0.5,
    patience: int = 3,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Return a ReduceLROnPlateau scheduler that tracks validation loss.

    Parameters
    ----------
    factor   : multiplicative factor applied when the plateau is detected
    patience : number of epochs with no improvement before reducing LR
    min_lr   : lower bound on the learning rate
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


# ============================================================
# 2. TRAIN ONE EPOCH  (teacher-forced, batched)
# ============================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float = 1.0,
) -> float:
    """Run one full pass over *loader* with teacher forcing.

    Parameters
    ----------
    model      : Seq2SeqLSTM (or compatible)
    loader     : training DataLoader (yields src, tgt, src_lens, tgt_lens)
    optimizer  : configured optimiser
    criterion  : loss function (ignore_index should already be set)
    device     : target device
    clip_grad  : max gradient norm for gradient clipping (0 → disabled)

    Returns
    -------
    float
        Weighted-average loss across the epoch (weighted by batch size so
        the last partial batch is handled correctly).
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (src, tgt, src_lens, tgt_lens) in tqdm(enumerate(loader)):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # logits : (B, tgt_len-1, V_out)  — predicts tgt[:,1:]
        logits = model(src, tgt)

        # Ground truth : tgt shifted left by 1  (drop leading <SOS>)
        tgt_out = tgt[:, 1:]              # (B, tgt_len-1)

        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))

        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss    += loss.item() * B
        total_samples += B

    return total_loss / max(total_samples, 1)


# ============================================================
# 3. VALIDATE ENTIRE DATASET
# ============================================================

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    tokenizer,                         # DualBPETokenizer
    device: torch.device,
    criterion: nn.Module,
    beam_width: int = 3,
    max_gen_len: int = 128,
    compute_functional: bool = True,
    functional_mse_threshold: float = 100.0,
    use_beam: bool = False,
) -> Dict[str, float]:
    """Validate the model on the full validation set.

    Two passes are made:
      1. **Batched teacher-forced pass** — computes validation loss efficiently
         using the same forward path as training.
      2. **Generation pass** — produces predictions and computes sequence-level
         and symbolic metrics.  By default uses fast batched greedy decoding
         (``use_beam=False``).  Pass ``use_beam=True`` to use per-sample beam
         search instead (slower but higher quality).

    Parameters
    ----------
    model                    : trained / training Seq2SeqLSTM
    val_loader               : validation DataLoader
    tokenizer                : DualBPETokenizer (already fitted)
    device                   : evaluation device
    criterion                : loss function (same as training, has ignore_index)
    beam_width               : beam width used when ``use_beam=True`` (ignored otherwise)
    max_gen_len              : maximum tokens to generate per sample
    compute_functional       : whether to run symbolic / numerical metrics
                               (slower; disable for quick checks)
    functional_mse_threshold : passed through to batch_function_closeness
    use_beam                 : if True, use per-sample beam search (slow);
                               if False (default), use fast batched greedy decoding

    Returns
    -------
    dict with keys
        ``"val_loss"``           — teacher-forced cross-entropy loss
        ``"token_acc"``          — token-level accuracy (non-PAD positions)
        ``"sentence_acc"``       — exact sequence match (PAD-masked)
        ``"strict_sentence_acc"``— exact match with EOS alignment
        ``"prefix_acc"``         — mean longest-correct-prefix ratio
        ``"length_acc"``         — fraction of sequences with correct length
        ``"avg_l1"``             — mean L1 distance (functional, if enabled)
        ``"avg_mse"``            — mean MSE         (functional, if enabled)
        ``"avg_rmse"``           — mean RMSE        (functional, if enabled)
        ``"avg_relative_mse"``   — mean relative MSE (functional, if enabled)
        ``"n_functional_ok"``    — samples with valid functional evaluation
        ``"n_functional_failed"``— samples that could not be parsed / evaluated
    """
    model.eval()
    out_tok = tokenizer.output_tokenizer
    pad_id  = out_tok.pad_id
    eos_id  = out_tok.eos_id

    # ------------------------------------------------------------------
    # Pass 1 — batched loss (teacher forcing)
    # ------------------------------------------------------------------
    total_loss    = 0.0
    total_samples = 0

    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            logits  = model(src, tgt)          # (B, T-1, V_out)
            tgt_out = tgt[:, 1:]               # (B, T-1)

            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))

            total_loss    += loss.item() * B
            total_samples += B

    val_loss = total_loss / max(total_samples, 1)

    # ------------------------------------------------------------------
    # Pass 2 — generation + metric accumulation
    #
    # Fast path (default): batched greedy decoding via model.generate_batch()
    #   — one encoder call per batch instead of one per sample.
    # Slow path (use_beam=True): per-sample beam search via model.generate()
    #   — higher quality but ~beam_width× slower; run infrequently.
    # ------------------------------------------------------------------
    all_pred_ids: List[List[int]] = []
    all_tgt_ids:  List[List[int]] = []
    pred_strings: List[str] = []
    tgt_strings:  List[str] = []

    # Count total samples for beam search progress bar
    if use_beam:
        total_val_samples = sum(src.shape[0] for src, *_ in val_loader)
        print(f"[Beam Search] Starting beam decoding: {total_val_samples} samples, beam_width={beam_width}, max_gen_len={max_gen_len}")
        beam_pbar = tqdm(total=total_val_samples, desc=f"Beam decode (w={beam_width})", unit="sample", dynamic_ncols=True)

    with torch.no_grad():
        for src, tgt, _src_lens, _tgt_lens in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            B   = src.shape[0]

            if use_beam:
                # Per-sample beam search (slow; use every N epochs only)
                batch_preds: List[Tuple[List[int], str]] = []
                for i in range(B):
                    pred_ids, pred_str = model.generate(
                        src[i : i + 1],
                        tokenizer,
                        max_len=max_gen_len,
                        beam_width=beam_width,
                    )
                    batch_preds.append((pred_ids, pred_str))
                    beam_pbar.update(1)
            else:
                # Batched greedy decoding (fast; one encoder call per batch)
                batch_preds = model.generate_batch(src, tokenizer, max_len=max_gen_len)

            for i, (pred_ids, pred_str) in enumerate(batch_preds):
                # Ground truth: strip <SOS> and <EOS> from tgt[i]
                tgt_seq = tgt[i].tolist()
                if tgt_seq and tgt_seq[0] == out_tok.sos_id:
                    tgt_seq = tgt_seq[1:]
                tgt_ids_clean = []
                for tid in tgt_seq:
                    if tid == eos_id:
                        tgt_ids_clean.append(tid)   # include EOS for strict_sentence_accuracy
                        break
                    if tid != pad_id:
                        tgt_ids_clean.append(tid)

                all_pred_ids.append(pred_ids)
                all_tgt_ids.append(tgt_ids_clean)

                tgt_str = out_tok.decode([t for t in tgt_ids_clean if t != eos_id])
                pred_strings.append(pred_str)
                tgt_strings.append(tgt_str)

    if use_beam:
        beam_pbar.close()
        print(f"[Beam Search] Decoding complete. {total_val_samples} samples decoded.")

    # ------------------------------------------------------------------
    # Pad pred / tgt lists to tensors for metric functions
    # ------------------------------------------------------------------
    max_pred = max((len(p) for p in all_pred_ids), default=1)
    max_tgt  = max((len(t) for t in all_tgt_ids),  default=1)
    max_len  = max(max_pred, max_tgt)

    N = len(all_pred_ids)
    pred_tensor = torch.full((N, max_len), pad_id, dtype=torch.long)
    tgt_tensor  = torch.full((N, max_len), pad_id, dtype=torch.long)

    for i, (p, t) in enumerate(zip(all_pred_ids, all_tgt_ids)):
        if p:
            pred_tensor[i, : len(p)] = torch.tensor(p, dtype=torch.long)
        if t:
            tgt_tensor[i,  : len(t)] = torch.tensor(t, dtype=torch.long)

    tok_acc    = token_accuracy(pred_tensor, tgt_tensor, pad_id)
    sent_acc   = sentence_accuracy(pred_tensor, tgt_tensor, pad_id)
    strict_acc = strict_sentence_accuracy(pred_tensor, tgt_tensor, pad_id, eos_id)
    pref_acc   = prefix_accuracy(pred_tensor, tgt_tensor, pad_id)
    len_acc    = length_accuracy(pred_tensor, tgt_tensor, pad_id)

    # ------------------------------------------------------------------
    # Optional functional / symbolic evaluation
    # ------------------------------------------------------------------
    func_metrics = {
        "avg_l1":           float("nan"),
        "avg_mse":          float("nan"),
        "avg_rmse":         float("nan"),
        "avg_relative_mse": float("nan"),
        "n_functional_ok":  0,
        "n_functional_failed": 0,
    }

    if compute_functional:
        pred_exprs: List[sp.Expr] = []
        tgt_exprs:  List[sp.Expr] = []
        n_failed = 0

        for ps, ts in zip(pred_strings, tgt_strings):
            pe = _try_parse_expr(ps)
            te = _try_parse_expr(ts)
            if pe is not None and te is not None:
                pred_exprs.append(pe)
                tgt_exprs.append(te)
            else:
                n_failed += 1

        if pred_exprs:
            fc = batch_function_closeness(
                pred_exprs,
                tgt_exprs,
                _X,
                mse_threshold=functional_mse_threshold,
                verbose=False,
                verbose_warning=False,
            )
            func_metrics.update(
                {
                    "avg_l1":           fc["avg_l1"],
                    "avg_mse":          fc["avg_mse"],
                    "avg_rmse":         fc["avg_rmse"],
                    "avg_relative_mse": fc["avg_relative_mse"],
                    "n_functional_ok":  fc["n_ok"] + fc["n_restricted"],
                    "n_functional_failed": n_failed
                    + fc["n_scattered"]
                    + fc["n_high_mse"]
                    + fc["n_no_valid"],
                }
            )
        else:
            func_metrics["n_functional_failed"] = n_failed

    return {
        "val_loss":            val_loss,
        "token_acc":           tok_acc,
        "sentence_acc":        sent_acc,
        "strict_sentence_acc": strict_acc,
        "prefix_acc":          pref_acc,
        "length_acc":          len_acc,
        **func_metrics,
    }


# ============================================================
# 4. PRETTY PRINT METRICS
# ============================================================

def print_metrics(
    epoch: int,
    train_loss: float,
    val_metrics: Dict[str, float],
    use_beam: bool = False,
) -> None:
    """Print a nicely formatted summary for one epoch."""
    sep = "─" * 65
    decode_tag = "beam" if use_beam else "greedy"
    print(sep)
    print(f"  Epoch {epoch:>4d}  [{decode_tag} decode]")
    print(sep)
    print(f"  {'Train loss':<28s}: {train_loss:.6f}")
    print(f"  {'Val loss':<28s}: {val_metrics['val_loss']:.6f}")
    print()
    print(f"  {'Token accuracy':<28s}: {val_metrics['token_acc']:.4f}")
    print(f"  {'Sentence accuracy':<28s}: {val_metrics['sentence_acc']:.4f}")
    print(f"  {'Strict sentence accuracy':<28s}: {val_metrics['strict_sentence_acc']:.4f}")
    print(f"  {'Prefix accuracy':<28s}: {val_metrics['prefix_acc']:.4f}")
    print(f"  {'Length accuracy':<28s}: {val_metrics['length_acc']:.4f}")

    if not np.isnan(val_metrics.get("avg_mse", float("nan"))):
        print()
        print(f"  {'Avg L1':<28s}: {val_metrics['avg_l1']:.6f}")
        print(f"  {'Avg MSE':<28s}: {val_metrics['avg_mse']:.6f}")
        print(f"  {'Avg RMSE':<28s}: {val_metrics['avg_rmse']:.6f}")
        print(f"  {'Avg Relative MSE':<28s}: {val_metrics['avg_relative_mse']:.6f}")
        print(
            f"  Functional OK / Failed: "
            f"{val_metrics['n_functional_ok']} / {val_metrics['n_functional_failed']}"
        )
    print(sep)