"""
metric.py
=========
Evaluation metrics for a Taylor-expansion sequence learning task.

Sections
--------
1. PREFIX PREDICTION METRICS  — token-level and sequence-level accuracy
2. FUNCTION CLOSENESS METRICS — symbolic equivalence and numerical distance
3. VISUALIZATION              — side-by-side comparison of predicted vs target
"""

import torch
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# HYPERPARAMETERS  (edit here — do not hard-code in functions)
# ============================================================

X_RANGE: tuple[float, float] = (-0.5, 0.5)   # evaluation window for x
NUM_POINTS: int = 100                          # number of linspace sample points
RELATIVE_MSE_EPS: float = 1e-10               # denominator stabiliser for relative MSE
OVERFLOW_THRESHOLD: float = 1e15              # values above this are treated as overflow


# ============================================================
# SECTION 1 — PREFIX PREDICTION METRICS
# ============================================================


def token_accuracy(
    pred_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    pad_id: int,
) -> float:
    """Compute token-wise accuracy, ignoring PAD tokens.

    Parameters
    ----------
    pred_ids : torch.Tensor
        Predicted token ids of shape ``(batch_size, seq_len)``.
    tgt_ids : torch.Tensor
        Ground-truth token ids of shape ``(batch_size, seq_len)``.
    pad_id : int
        Token id used for padding; these positions are excluded from the
        accuracy computation.

    Returns
    -------
    float
        Fraction of non-PAD tokens that are predicted correctly,
        in the range [0, 1].
    """
    mask = tgt_ids != pad_id
    correct = (pred_ids == tgt_ids) & mask
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return correct.sum().item() / total


def sentence_accuracy(
    pred_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    pad_id: int,
) -> float:
    """Compute exact-sequence-match accuracy, ignoring trailing PAD tokens.

    A sequence is considered correct only if every non-PAD position matches
    exactly.

    Parameters
    ----------
    pred_ids : torch.Tensor
        Predicted token ids of shape ``(batch_size, seq_len)``.
    tgt_ids : torch.Tensor
        Ground-truth token ids of shape ``(batch_size, seq_len)``.
    pad_id : int
        Token id used for padding; these positions are masked out before
        comparison.

    Returns
    -------
    float
        Fraction of sequences that are entirely correct, in the range [0, 1].
    """
    mask = tgt_ids != pad_id
    pred_masked = pred_ids.masked_fill(~mask, pad_id)
    tgt_masked = tgt_ids.masked_fill(~mask, pad_id)
    match = (pred_masked == tgt_masked).all(dim=1)
    return match.float().mean().item()


def strict_sentence_accuracy(
    pred_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    pad_id: int,
    eos_id: int,
) -> float:
    """Compute exact-sequence-match accuracy with strict EOS alignment.

    A sequence is correct only if (1) the prediction contains an EOS token,
    (2) the first EOS appears at the same position as in the target, and
    (3) every token up to and including EOS matches exactly.  This is
    stricter than :func:`sentence_accuracy` because it penalises predictions
    that produce the right tokens but terminate at the wrong position.

    Parameters
    ----------
    pred_ids : torch.Tensor
        Predicted token ids of shape ``(batch_size, seq_len)``.
    tgt_ids : torch.Tensor
        Ground-truth token ids of shape ``(batch_size, seq_len)``.
    pad_id : int
        Token id used for padding (unused in the comparison but kept for
        API consistency with the other metrics).
    eos_id : int
        Token id for the end-of-sequence marker.

    Returns
    -------
    float
        Fraction of sequences that are entirely correct with matching EOS
        positions, in the range [0, 1].
    """
    B, L = tgt_ids.shape

    tgt_is_eos = (tgt_ids == eos_id)
    pred_is_eos = (pred_ids == eos_id)

    tgt_eos_pos = tgt_is_eos.float().argmax(dim=1)
    pred_eos_pos = pred_is_eos.float().argmax(dim=1)

    pred_has_eos = pred_is_eos.any(dim=1)
    eos_match = (tgt_eos_pos == pred_eos_pos)

    positions = torch.arange(L, device=tgt_ids.device).unsqueeze(0)
    compare_mask = positions <= tgt_eos_pos.unsqueeze(1)

    token_match = (pred_ids == tgt_ids) | ~compare_mask
    all_tokens_match = token_match.all(dim=1)

    correct = (pred_has_eos & eos_match & all_tokens_match).sum()
    return correct.item() / B


# ---------------------------------------------------------------------------
# Additional metrics
# ---------------------------------------------------------------------------

def prefix_accuracy(
    pred_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    pad_id: int,
) -> float:
    """Compute the average longest-correct-prefix ratio per sequence.

    For each sequence, finds the first position where the prediction
    diverges from the target, then reports that length as a fraction of
    the full (non-PAD) target length.  A value of 1.0 means the entire
    sequence was correct; 0.0 means the very first token was wrong.

    This is more informative than sentence accuracy during early training
    because it gives partial credit and shows whether the model is at
    least getting the beginning of each sequence right.

    Parameters
    ----------
    pred_ids : torch.Tensor
        Predicted token ids of shape ``(batch_size, seq_len)``.
    tgt_ids : torch.Tensor
        Ground-truth token ids of shape ``(batch_size, seq_len)``.
    pad_id : int
        Token id used for padding.

    Returns
    -------
    float
        Mean prefix-correct ratio across the batch, in the range [0, 1].
    """
    B, L = tgt_ids.shape
    mask = tgt_ids != pad_id                              # (B, L)
    seq_lens = mask.sum(dim=1).float()                    # (B,)

    matches = (pred_ids == tgt_ids) & mask                # (B, L)
    # Cumulative product along seq dim: stays 1 until first mismatch
    cum_correct = matches.float().cumprod(dim=1)          # (B, L)
    prefix_lens = cum_correct.sum(dim=1)                  # (B,)

    ratios = prefix_lens / seq_lens.clamp(min=1)
    return ratios.mean().item()


def length_accuracy(
    pred_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    pad_id: int,
) -> float:
    """Compute the fraction of sequences where predicted length matches target.

    Length is defined as the number of non-PAD tokens.  This metric isolates
    whether the model has learned when to stop generating, independent of
    whether the actual tokens are correct.

    Parameters
    ----------
    pred_ids : torch.Tensor
        Predicted token ids of shape ``(batch_size, seq_len)``.
    tgt_ids : torch.Tensor
        Ground-truth token ids of shape ``(batch_size, seq_len)``.
    pad_id : int
        Token id used for padding.

    Returns
    -------
    float
        Fraction of sequences with matching lengths, in the range [0, 1].
    """
    pred_lens = (pred_ids != pad_id).sum(dim=1)
    tgt_lens = (tgt_ids != pad_id).sum(dim=1)
    return (pred_lens == tgt_lens).float().mean().item()



def per_position_accuracy(
    pred_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Compute accuracy at each sequence position across the batch.

    Returns a 1-D tensor of length ``seq_len`` where entry *t* is the
    fraction of batch elements that are correct at position *t* (excluding
    positions that are PAD in the target).  Plotting this curve reveals
    whether accuracy degrades towards the end of sequences, which is a
    common failure mode in autoregressive models.

    Parameters
    ----------
    pred_ids : torch.Tensor
        Predicted token ids of shape ``(batch_size, seq_len)``.
    tgt_ids : torch.Tensor
        Ground-truth token ids of shape ``(batch_size, seq_len)``.
    pad_id : int
        Token id used for padding.

    Returns
    -------
    torch.Tensor
        Accuracy per position, shape ``(seq_len,)``, in the range [0, 1].
        Positions where no batch element has a real token are set to 0.
    """
    mask = tgt_ids != pad_id                                   # (B, L)
    correct = ((pred_ids == tgt_ids) & mask).float().sum(0)    # (L,)
    counts = mask.float().sum(0).clamp(min=1)                  # (L,)
    return correct / counts


# ============================================================
# SECTION 2 — FUNCTION CLOSENESS METRICS
# ============================================================

def symbolic_equivalence(pred_expr: sp.Expr, tgt_expr: sp.Expr) -> bool:
    """Check whether two SymPy expressions are symbolically equivalent.

    Uses ``sympy.simplify(pred - tgt) == 0`` as the criterion.

    Parameters
    ----------
    pred_expr : sympy.Expr
        Predicted symbolic expression.
    tgt_expr : sympy.Expr
        Target symbolic expression.

    Returns
    -------
    bool
        ``True`` if the expressions are symbolically identical, ``False``
        otherwise (including on any exception).
    """
    try:
        diff = sp.simplify(pred_expr - tgt_expr)
        return diff == 0
    except Exception as e:
        raise ValueError(f"Error during symbolic equivalence check: {e}") from e
    return False

def evaluate_function_closeness(
    pred_expr: sp.Expr,
    tgt_expr: sp.Expr,
    x_symbol: sp.Symbol,
    x_values: np.ndarray,
    *,
    mse_threshold: float = 100.0,
    verbose: bool = False,
    verbose_warning: bool = False,
) -> dict[str, float]:
    """Numerically compare a predicted expression against the target.

    Both expressions are lambdified with a NumPy backend and evaluated at
    every point in *x_values*.  If some points produce invalid values (NaN,
    overflow, or complex), evaluation is restricted to the contiguous valid
    domain ``[min_good_x, max_good_x]``.  If valid points are scattered
    (non-contiguous), a warning is printed (when *verbose_warning* is ``True``)
    and all metrics are returned as ``inf``.  If the computed MSE exceeds
    *mse_threshold*, the pair is also skipped and ``status`` is set to
    ``"high_mse"``.  If no valid points exist at all, a ``ValueError`` is
    raised.

    Parameters
    ----------
    pred_expr : sympy.Expr
        Predicted symbolic expression (function of *x_symbol*).
    tgt_expr : sympy.Expr
        Target symbolic expression (function of *x_symbol*).
    x_symbol : sympy.Symbol
        The free variable used in both expressions.
    x_values : numpy.ndarray
        1-D array of evaluation points.
    mse_threshold : float
        Pairs whose MSE exceeds this value are skipped (``status="high_mse"``).
        Defaults to ``100.0``.
    verbose : bool
        When ``True``, print an info message whenever the evaluation domain is
        narrowed.  Defaults to ``False``.
    verbose_warning : bool
        When ``True``, print a warning message for scattered domains and
        high-MSE pairs.  Defaults to ``False``.

    Returns
    -------
    dict with keys
        ``"l1"``          — mean absolute error
        ``"mse"``         — mean squared error
        ``"rmse"``        — root mean squared error
        ``"relative_mse"``— MSE normalised by mean squared magnitude of target
        ``"status"``      — one of ``"ok"``, ``"restricted"``, ``"scattered"``,
                            ``"high_mse"``, ``"no_valid"``
    """
    _inf = {"l1": np.inf, "mse": np.inf, "rmse": np.inf, "relative_mse": np.inf}

    def _valid_mask(arr: np.ndarray) -> np.ndarray:
        """Return a boolean mask of points that are real, finite, and within bounds."""
        if np.iscomplexobj(arr):
            return np.zeros(len(arr), dtype=bool)
        a = arr.astype(float)
        return ~np.isnan(a) & (np.abs(a) <= OVERFLOW_THRESHOLD)

    try:
        pred_fn = sp.lambdify(x_symbol, pred_expr, modules="numpy")
        tgt_fn = sp.lambdify(x_symbol, tgt_expr, modules="numpy")

        # Evaluate without forcing dtype=float so complex outputs are preserved.
        # Suppress numpy RuntimeWarnings (divide-by-zero, invalid domain, etc.)
        # — NaNs/infs are already caught by the validity mask below.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            pred_raw = np.asarray(pred_fn(x_values))
            tgt_raw = np.asarray(tgt_fn(x_values))

        # Broadcast scalars (e.g. constant expressions) to full length
        if pred_raw.ndim == 0:
            pred_raw = np.full(len(x_values), pred_raw.item())
        if tgt_raw.ndim == 0:
            tgt_raw = np.full(len(x_values), tgt_raw.item())

        valid = _valid_mask(pred_raw) & _valid_mask(tgt_raw)

        if not np.any(valid):
            raise ValueError(
                "No valid evaluation points found: all x values produce NaN, "
                "overflow, or complex output for the given expressions."
            )

        valid_indices = np.where(valid)[0]

        # Check whether valid points form a contiguous block
        is_contiguous = (
            int(valid_indices[-1]) - int(valid_indices[0]) + 1 == len(valid_indices)
        )

        if not is_contiguous:
            if verbose_warning:
                print(
                    f"Warning: valid domain is scattered (non-contiguous) for "
                    f"{pred_expr} vs {tgt_expr}. Skipping — returning inf."
                )
            return {**_inf, "status": "scattered"}

        # Restrict to the contiguous valid domain
        pred_vals = pred_raw[valid_indices].astype(float)
        tgt_vals = tgt_raw[valid_indices].astype(float)

        status = "ok"
        if len(valid_indices) < len(x_values):
            if verbose:
                x_lo = x_values[valid_indices[0]]
                x_hi = x_values[valid_indices[-1]]
                print(
                    f"Info: evaluation restricted to x ∈ [{x_lo:.4g}, {x_hi:.4g}] "
                    f"({len(valid_indices)}/{len(x_values)} points) due to invalid "
                    f"values outside this range."
                )
            status = "restricted"

        residuals = pred_vals - tgt_vals
        l1 = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))

        if mse > mse_threshold:
            if verbose_warning:
                print(
                    f"Warning: MSE={mse:.4e} exceeds threshold {mse_threshold} for "
                    f"{pred_expr} vs {tgt_expr}. Skipping — returning inf."
                )
            return {**_inf, "status": "high_mse"}

        tgt_sq_mean = float(np.mean(tgt_vals ** 2))
        relative_mse = mse / (tgt_sq_mean + RELATIVE_MSE_EPS)

        return {"l1": l1, "mse": mse, "rmse": rmse, "relative_mse": relative_mse, "status": status}

    except ValueError:
        raise
    except Exception:
        return {**_inf, "status": "no_valid"}


def batch_function_closeness(
    pred_expr_list: list[sp.Expr],
    tgt_expr_list: list[sp.Expr],
    x_symbol: sp.Symbol,
    x_range: tuple[float, float] = X_RANGE,
    num_points: int = NUM_POINTS,
    mse_threshold: float = 100.0,
    verbose: bool = False,
    verbose_warning: bool = False
) -> dict[str, float]:
    """Average function-closeness metrics over a batch of expression pairs.

    Parameters
    ----------
    pred_expr_list : list of sympy.Expr
        Predicted expressions, one per sample.
    tgt_expr_list : list of sympy.Expr
        Target expressions, one per sample (same length as *pred_expr_list*).
    x_symbol : sympy.Symbol
        The free variable used in all expressions.
    x_range : tuple[float, float]
        ``(x_min, x_max)`` evaluation window.  Defaults to ``X_RANGE``.
    num_points : int
        Number of evaluation points sampled uniformly in *x_range*.
        Defaults to ``NUM_POINTS``.
    mse_threshold : float
        Pairs whose MSE exceeds this value are excluded from the average
        (``status="high_mse"``).  Defaults to ``100.0``.
    verbose : bool
        When ``True``, print info messages about narrowed evaluation domains.
        Defaults to ``False``.
    verbose_warning : bool
        When ``True``, print warnings for scattered domains and high-MSE pairs.
        Defaults to ``False``.

    Returns
    -------
    dict with keys
        ``"avg_l1"``, ``"avg_mse"``, ``"avg_rmse"``, ``"avg_relative_mse"`` —
            averages computed only over pairs with a valid (possibly restricted)
            domain and MSE within *mse_threshold*.
        ``"n_total"``      — total number of pairs.
        ``"n_ok"``         — pairs evaluated over the full domain.
        ``"n_restricted"`` — pairs evaluated over a narrowed contiguous domain.
        ``"n_scattered"``  — pairs skipped because the valid domain was scattered.
        ``"n_high_mse"``   — pairs skipped because MSE exceeded *mse_threshold*.
        ``"n_no_valid"``   — pairs skipped because no valid points existed at all.
    """
    if type(pred_expr_list) != list or type(tgt_expr_list) != list:
        raise TypeError("pred_expr_list and tgt_expr_list must be lists.")

    n = len(pred_expr_list)
    if n != len(tgt_expr_list):
        raise ValueError("pred_expr_list and tgt_expr_list must be of the same length.")

    x_values = np.linspace(x_range[0], x_range[1], num_points)

    totals = {"l1": 0.0, "mse": 0.0, "rmse": 0.0, "relative_mse": 0.0}
    counts = {"ok": 0, "restricted": 0, "scattered": 0, "high_mse": 0, "no_valid": 0}

    for pred_expr, tgt_expr in zip(pred_expr_list, tgt_expr_list):
        try:
            metrics = evaluate_function_closeness(
                pred_expr, tgt_expr, x_symbol, x_values,
                mse_threshold=mse_threshold,
                verbose=verbose,
                verbose_warning=verbose_warning,
            )
        except ValueError:
            counts["no_valid"] += 1
            continue

        status = metrics.get("status", "ok")
        counts[status] = counts.get(status, 0) + 1

        # Only accumulate metrics for pairs that produced valid (finite) results
        if status in ("ok", "restricted"):
            for key in totals:
                totals[key] += metrics[key]

    n_counted = counts["ok"] + counts["restricted"]
    if n_counted == 0:
        avg = {"avg_l1": np.inf, "avg_mse": np.inf, "avg_rmse": np.inf, "avg_relative_mse": np.inf}
    else:
        avg = {
            "avg_l1":           totals["l1"]           / n_counted,
            "avg_mse":          totals["mse"]           / n_counted,
            "avg_rmse":         totals["rmse"]          / n_counted,
            "avg_relative_mse": totals["relative_mse"]  / n_counted,
        }

    return {
        **avg,
        "n_total":      n,
        "n_ok":         counts["ok"],
        "n_restricted": counts["restricted"],
        "n_scattered":  counts["scattered"],
        "n_high_mse":   counts["high_mse"],
        "n_no_valid":   counts["no_valid"],
    }

def coefficient_accuracy(
    pred_expr: sp.Expr,
    tgt_expr: sp.Expr,
    x_symbol: sp.Symbol,
    max_degree: int = 4,
) -> dict[str, float]:
    """Compare Taylor coefficients degree by degree.

    Parameters
    ----------
    pred_expr : sympy.Expr
        Predicted symbolic expression.
    tgt_expr : sympy.Expr
        Target symbolic expression.
    x_symbol : sympy.Symbol
        The free variable.
    max_degree : int
        Maximum polynomial degree (inclusive).

    Returns
    -------
    dict with keys
        ``"exact_coeff_match"`` — fraction of coefficients (degree 0..max_degree)
            that match exactly.
        ``"coeff_mse"`` — mean squared error across coefficients.
        ``"per_degree_correct"`` — list of bools, one per degree.
    """
    try:
        pred_poly = sp.Poly(sp.expand(pred_expr), x_symbol)
        tgt_poly = sp.Poly(sp.expand(tgt_expr), x_symbol)
    except (sp.GeneratorsNeeded, sp.PolificationFailed):
        return {
            "exact_coeff_match": 0.0,
            "coeff_mse": float("inf"),
            "per_degree_correct": [False] * (max_degree + 1),
        }

    per_degree = []
    sq_errors = []
    for d in range(max_degree + 1):
        pc = float(pred_poly.nth(d))
        tc = float(tgt_poly.nth(d))
        per_degree.append(sp.Rational(pred_poly.nth(d)) == sp.Rational(tgt_poly.nth(d)))
        sq_errors.append((pc - tc) ** 2)

    return {
        "exact_coeff_match": sum(per_degree) / len(per_degree),
        "coeff_mse": sum(sq_errors) / len(sq_errors),
        "per_degree_correct": per_degree,
    }

# ============================================================
# SECTION 3 — VISUALIZATION
# ============================================================

def plot_comparison(
    pred_expr: sp.Expr,
    tgt_expr: sp.Expr,
    x_symbol: sp.Symbol,
    x_range: tuple[float, float] = X_RANGE,
    num_points: int = NUM_POINTS,
    title: str = "Taylor Expansion — Predicted vs Target",
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot a two-panel comparison of predicted and target expressions.

    Left panel  — overlaid curves of ``pred_expr`` and ``tgt_expr``.
    Right panel — pointwise absolute error ``|pred(x) − tgt(x)|``.

    Parameters
    ----------
    pred_expr : sympy.Expr
        Predicted symbolic expression.
    tgt_expr : sympy.Expr
        Target symbolic expression.
    x_symbol : sympy.Symbol
        The free variable used in both expressions.
    x_range : tuple[float, float]
        Evaluation window ``(x_min, x_max)``.  Defaults to ``X_RANGE``.
    num_points : int
        Number of sample points.  Defaults to ``NUM_POINTS``.
    title : str
        Overall figure title.
    figsize : tuple[int, int]
        Matplotlib figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure (caller can call ``fig.savefig(...)`` or
        ``plt.show()`` as needed).
    """
    x_vals = np.linspace(x_range[0], x_range[1], num_points)

    try:
        pred_fn = sp.lambdify(x_symbol, pred_expr, modules="numpy")
        tgt_fn = sp.lambdify(x_symbol, tgt_expr, modules="numpy")

        pred_vals = np.asarray(pred_fn(x_vals), dtype=float)
        tgt_vals = np.asarray(tgt_fn(x_vals), dtype=float)

        if pred_vals.ndim == 0:
            pred_vals = np.full_like(x_vals, pred_vals.item())
        if tgt_vals.ndim == 0:
            tgt_vals = np.full_like(x_vals, tgt_vals.item())

        has_overflow = (
            np.any(np.isnan(pred_vals)) or np.any(np.isinf(pred_vals))
            or np.any(np.isnan(tgt_vals)) or np.any(np.isinf(tgt_vals))
            or np.any(np.abs(pred_vals) > OVERFLOW_THRESHOLD)
            or np.any(np.abs(tgt_vals) > OVERFLOW_THRESHOLD)
        )

    except Exception:
        has_overflow = True
        pred_vals = tgt_vals = None

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax_fn = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1])

    if has_overflow or pred_vals is None:
        for ax in (ax_fn, ax_err):
            ax.text(
                0.5, 0.5,
                "Numerical evaluation failed\n(NaN / overflow / exception)",
                ha="center", va="center", transform=ax.transAxes,
                color="red", fontsize=11,
            )
    else:
        error_vals = np.abs(pred_vals - tgt_vals)

        # — Left panel: function overlay —
        ax_fn.plot(x_vals, tgt_vals,  label="Target",    color="steelblue",  linewidth=2)
        ax_fn.plot(x_vals, pred_vals, label="Predicted", color="darkorange",
                   linewidth=2, linestyle="--")
        ax_fn.set_xlabel(str(x_symbol))
        ax_fn.set_ylabel("f(x)")
        ax_fn.set_title("Function Comparison")
        ax_fn.legend(framealpha=0.9)
        ax_fn.grid(True, linestyle=":", alpha=0.6)

        # — Right panel: pointwise error —
        ax_err.plot(x_vals, error_vals, color="crimson", linewidth=2)
        ax_err.fill_between(x_vals, error_vals, alpha=0.15, color="crimson")
        ax_err.set_xlabel(str(x_symbol))
        ax_err.set_ylabel("|pred(x) − tgt(x)|")
        ax_err.set_title("Pointwise Absolute Error")
        ax_err.grid(True, linestyle=":", alpha=0.6)

        # Annotate summary stats
        metrics = evaluate_function_closeness(
            pred_expr, tgt_expr, x_symbol, x_vals
        )
        stats_text = (
            f"L1   = {metrics['l1']:.4e}\n"
            f"MSE  = {metrics['mse']:.4e}\n"
            f"RMSE = {metrics['rmse']:.4e}"
        )
        ax_err.text(
            0.97, 0.97, stats_text,
            transform=ax_err.transAxes,
            ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85),
        )

    # Expression labels below the panels
    pred_str = sp.pretty(pred_expr, use_unicode=True)
    tgt_str = sp.pretty(tgt_expr, use_unicode=True)
    fig.text(
        0.5, -0.04,
        f"Predicted: {pred_str}     |     Target: {tgt_str}",
        ha="center", fontsize=9, color="dimgray",
    )

    fig.tight_layout()
    return fig
