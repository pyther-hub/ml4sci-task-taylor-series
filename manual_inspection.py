"""
manual_inspection.py
====================
Manual inspection of the LSTM Taylor-expansion model.

Loads lstm_best_model.pt, runs greedy + beam-search decoding on a small
slice of the dataset, and prints a side-by-side comparison with:
  - Raw input / greedy / beam / target strings
  - Token-level diff (which positions differ)
  - SymPy symbolic equivalence check
  - Summary statistics over all inspected samples
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
)

from dataset import DualBPETokenizer
from lstm_model import Seq2SeqLSTM

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "model/checkpoints/lstm_best_model.pt"
TOKENIZER_PATH  = "model/dual_bpe_tokenizer.pkl"
DATASET_PATH    = "datasets/taylor_dataset_sample.json"

N_SAMPLES   = 20        # how many samples to inspect
BEAM_WIDTH  = 3         # beam width for beam-search decoding
MAX_GEN_LEN = 128       # max tokens to generate
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ───────────────────────────────────────────────────────────────────────────────

_SYMPY_TRANSFORMS = standard_transformations + (convert_xor,)
_X = sp.Symbol("x")


# ─── LOADERS ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load Seq2SeqLSTM from a checkpoint. Returns (model, checkpoint_dict)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    model.eval()

    vm = ckpt["val_metrics"]
    print(f"[checkpoint] Epoch {ckpt['epoch']}  |  "
          f"val_loss={vm['val_loss']:.5f}  |  "
          f"token_acc={vm['token_acc']:.4f}  |  "
          f"train_loss={ckpt['train_loss']:.5f}")
    return model, ckpt


def load_tokenizer(tokenizer_path: str) -> DualBPETokenizer:
    """Load a fitted DualBPETokenizer from disk."""
    tok = DualBPETokenizer()
    tok.load(tokenizer_path)
    print(f"[tokenizer]  Input vocab: {tok.input_vocab_size}  |  "
          f"Output vocab: {tok.output_vocab_size}")
    return tok


def load_dataset(json_path: str) -> list:
    """Load the raw JSON dataset."""
    with open(json_path) as f:
        data = json.load(f)
    print(f"[dataset]    {len(data)} total samples  |  path: '{json_path}'")
    return data


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def try_parse(text: str):
    """Parse a whitespace-separated infix expression into a SymPy Expr, or None."""
    if not text or not text.strip():
        return None
    try:
        return parse_expr(text, transformations=_SYMPY_TRANSFORMS,
                          local_dict={"x": _X})
    except Exception:
        return None


def sympy_equiv(pred_str: str, tgt_expr) -> str:
    """
    Check symbolic equivalence of pred_str against a pre-parsed target.
    Returns one of: 'EXACT', 'DIFF=<expr>', 'PARSE_FAIL', 'EVAL_FAIL'.
    """
    if tgt_expr is None:
        return "TGT_PARSE_FAIL"
    pred_expr = try_parse(pred_str)
    if pred_expr is None:
        return "PARSE_FAIL"
    try:
        diff = sp.simplify(pred_expr - tgt_expr)
        return "EXACT" if diff == 0 else f"DIFF={diff}"
    except Exception:
        return "EVAL_FAIL"


def token_diff(pred_tokens: list, tgt_tokens: list) -> str:
    """
    Token-level alignment string.
      ✓<tok>         → correct token
      ✗[pred→tgt]   → wrong token
      +[pred]        → extra token in prediction (target ended)
      -[tgt]         → missing token (prediction ended early)
    """
    max_len = max(len(pred_tokens), len(tgt_tokens), 1)
    marks = []
    for i in range(max_len):
        p = pred_tokens[i] if i < len(pred_tokens) else None
        t = tgt_tokens[i]  if i < len(tgt_tokens)  else None
        if p == t:
            marks.append(f"✓{p}")
        elif p is None:
            marks.append(f"-[{t}]")
        elif t is None:
            marks.append(f"+[{p}]")
        else:
            marks.append(f"✗[{p}→{t}]")
    return "  ".join(marks)


# ─── CORE INSPECTION ───────────────────────────────────────────────────────────

def inspect_sample(
    idx: int,
    item: dict,
    model: Seq2SeqLSTM,
    tokenizer: DualBPETokenizer,
    device: str,
) -> dict:
    """
    Run greedy + beam decoding on one data sample.
    Returns a dict with all relevant strings and metrics.
    """
    inp_tok = tokenizer.input_tokenizer
    out_tok = tokenizer.output_tokenizer

    src_ids = torch.tensor(
        [inp_tok.encode(item["function_prefix"])],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        greedy_ids, greedy_str = model.generate(
            src_ids, tokenizer, max_len=MAX_GEN_LEN, beam_width=1
        )
        beam_ids, beam_str = model.generate(
            src_ids, tokenizer, max_len=MAX_GEN_LEN, beam_width=BEAM_WIDTH
        )

    # Target: decode through tokenizer so it is in the same token-space
    target_str = out_tok.decode(out_tok.encode(item["taylor_prefix"]))

    # SymPy reference expression (prefer taylor_sympy if available)
    tgt_expr = try_parse(item.get("taylor_sympy") or target_str)

    greedy_sym = sympy_equiv(greedy_str, tgt_expr)
    beam_sym   = sympy_equiv(beam_str,   tgt_expr)

    tgt_tokens    = target_str.split()
    greedy_tokens = greedy_str.split()
    beam_tokens   = beam_str.split()

    return {
        "idx":             idx,
        "function_infix":  item.get("function_infix", "?"),
        "function_prefix": item["function_prefix"],
        "taylor_infix":    item.get("taylor_series", "?"),
        "target":          target_str,
        "greedy":          greedy_str,
        "beam":            beam_str,
        "greedy_sym":      greedy_sym,
        "beam_sym":        beam_sym,
        "greedy_diff":     token_diff(greedy_tokens, tgt_tokens),
        "beam_diff":       token_diff(beam_tokens,   tgt_tokens),
        "tgt_len":         len(tgt_tokens),
        "greedy_len":      len(greedy_tokens),
        "beam_len":        len(beam_tokens),
        "exact_greedy":    target_str == greedy_str,
        "exact_beam":      target_str == beam_str,
    }


# ─── DISPLAY ───────────────────────────────────────────────────────────────────

def print_results(results: list) -> None:
    """Print per-sample results and an aggregate summary."""
    SEP  = "─" * 80
    SEP2 = "═" * 80

    for r in results:
        print(SEP)
        print(f"  Sample #{r['idx']:>3d}")
        print(f"  Input (infix)  : {r['function_infix']}")
        print(f"  Input (prefix) : {r['function_prefix']}")
        print(f"  Target         : {r['target']}  (len={r['tgt_len']})")
        print(f"  Target (infix) : {r['taylor_infix']}")
        print()
        print(f"  Greedy         : {r['greedy']}  (len={r['greedy_len']})  [{r['greedy_sym']}]")
        print(f"  Beam (w={BEAM_WIDTH})     : {r['beam']}  (len={r['beam_len']})  [{r['beam_sym']}]")
        print()
        print(f"  Token diff (greedy): {r['greedy_diff']}")
        print(f"  Token diff (beam)  : {r['beam_diff']}")

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(results)
    exact_greedy   = sum(r["exact_greedy"]              for r in results)
    exact_beam     = sum(r["exact_beam"]                for r in results)
    sym_greedy     = sum(r["greedy_sym"] == "EXACT"     for r in results)
    sym_beam       = sum(r["beam_sym"]   == "EXACT"     for r in results)
    pfail_greedy   = sum("PARSE_FAIL" in r["greedy_sym"] for r in results)
    pfail_beam     = sum("PARSE_FAIL" in r["beam_sym"]   for r in results)
    len_ok_greedy  = sum(r["tgt_len"] == r["greedy_len"] for r in results)
    len_ok_beam    = sum(r["tgt_len"] == r["beam_len"]   for r in results)

    print(f"\n{SEP2}")
    print(f"  SUMMARY  ({n} samples inspected   |   beam_width={BEAM_WIDTH})")
    print(SEP2)
    print(f"  {'Metric':<35s}  {'Greedy':>10s}  {'Beam':>10s}")
    print(f"  {'─'*35}  {'─'*10}  {'─'*10}")
    print(f"  {'Exact string match':<35s}  {exact_greedy:>5d}/{n:<4d}  {exact_beam:>5d}/{n}")
    print(f"  {'SymPy exact equivalence':<35s}  {sym_greedy:>5d}/{n:<4d}  {sym_beam:>5d}/{n}")
    print(f"  {'SymPy parse failures':<35s}  {pfail_greedy:>5d}/{n:<4d}  {pfail_beam:>5d}/{n}")
    print(f"  {'Correct output length':<35s}  {len_ok_greedy:>5d}/{n:<4d}  {len_ok_beam:>5d}/{n}")
    print(SEP2)

    # ── Qualitative observation ────────────────────────────────────────────────
    print("\n  Quick observations:")
    if exact_greedy == 0 and exact_beam == 0:
        print("  • No exact string matches — the model is still learning structure.")
    elif exact_beam > exact_greedy:
        print(f"  • Beam search gives {exact_beam - exact_greedy} more exact matches than greedy.")
    else:
        print("  • Greedy and beam-search perform similarly on exact matches.")

    if sym_greedy > exact_greedy or sym_beam > exact_beam:
        print("  • Some predictions are symbolically correct but differ in surface form "
              "(e.g. different ordering / simplification).")
    if pfail_greedy > n // 2:
        print("  • Many greedy predictions fail SymPy parsing — output is likely malformed.")
    if pfail_beam > n // 2:
        print("  • Many beam predictions fail SymPy parsing — output is likely malformed.")
    print()


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def inspect(n_samples: int = N_SAMPLES) -> list:
    """
    Entry point for manual inspection.

    Args:
        n_samples : number of samples to inspect (taken from the start of the dataset)

    Returns:
        List of per-sample result dicts (useful for further analysis in a REPL).
    """
    print(f"\n{'═'*80}")
    print(f"  LSTM Manual Inspection")
    print(f"  checkpoint : {CHECKPOINT_PATH}")
    print(f"  tokenizer  : {TOKENIZER_PATH}")
    print(f"  dataset    : {DATASET_PATH}")
    print(f"  device     : {DEVICE}   |   samples: {n_samples}   |   beam_width: {BEAM_WIDTH}")
    print(f"{'═'*80}\n")

    model, _ckpt = load_model(CHECKPOINT_PATH, DEVICE)
    tokenizer    = load_tokenizer(TOKENIZER_PATH)
    data         = load_dataset(DATASET_PATH)
    print()

    subset  = data[:n_samples]
    results = []

    for i, item in enumerate(subset):
        results.append(inspect_sample(i, item, model, tokenizer, DEVICE))

    print()
    print_results(results)
    return results


if __name__ == "__main__":
    inspect()
