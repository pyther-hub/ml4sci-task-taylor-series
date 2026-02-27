"""
skeleton_dataset.py
===================
Dataset for Taylor-coefficient regression.

The task: given function_prefix (tokenized), predict the 5 rounded integer
numerators [a0, a1, a2, a3, a4] of the Taylor series expansion around 0:

    f(x) = a0/0! + a1/1! * x + a2/2! * x^2 + a3/3! * x^3 + a4/4! * x^4
         = a0    + a1*x      + (a2/2)*x^2   + (a3/6)*x^3  + (a4/24)*x^4

The denominators are the fixed factorial values [1, 1, 2, 6, 24].
We predict ONLY the numerators a_k = raw_coeff_k * k!  (the k-th derivative at 0).

Normalisation: per-coefficient z-score fitted on the training split only,
               stored in a CoeffNormaliser so it can be saved/loaded.
"""

import json
import pickle
import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# Re-use the input-side tokenizer from the existing dataset module
from dataset import WordLevelBPETokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ORDER  = 4                      # predict x^0 … x^4
NUM_COEFFS = MAX_ORDER + 1          # = 5
FACTORIALS = [1, 1, 2, 6, 24]      # k! for k = 0..4


# ---------------------------------------------------------------------------
# taylor_to_coeffs  — converts Taylor series string → derivative values
# ---------------------------------------------------------------------------

def taylor_to_coeffs(taylor_str: str, max_order: int = MAX_ORDER) -> List[float]:
    """
    Convert a Taylor series string to derivative coefficients [a0 .. a_max_order].

    Output vector : [f(0), f'(0), f''(0), f'''(0), f''''(0)]
    Relationship  : a_k = raw_coeff_k * k!   (k-th derivative at 0)

    The denominator of each term is the fixed factorial k!; we predict only
    the numerator a_k here.

    Examples
    --------
    'x + (1/2)*x**2 + (1/6)*x**3 + (1/24)*x**4'  ->  [0, 1, 1, 1, 1]
    'x**2 + (-1/2)*x**4'                           ->  [0, 0, 2, 0, -12]
    '(7)*x'                                         ->  [0, 7, 0, 0, 0]
    """
    factorials = [1, 1, 2, 6, 24, 120]      # covers up to order 5
    coeffs = [Fraction(0)] * (max_order + 1)

    # Normalise the string
    s = (
        taylor_str
        .replace(" ", "")
        .replace("**", "^")
        .replace("(", "")
        .replace(")", "")
    )

    for term in re.split(r"(?<![e^*/])(?=[+\-])", s):
        if not term:
            continue

        m = re.match(r"^([+\-]?[\d/]*)\*?x(?:\^(\d+))?$", term)
        if m:
            raw_c = m.group(1)
            power = int(m.group(2) or 1)
            if power > max_order:
                continue
            if raw_c in ("", "+"):
                c = Fraction(1)
            elif raw_c == "-":
                c = Fraction(-1)
            else:
                c = Fraction(raw_c)
            coeffs[power] += c
        elif "x" not in term:
            try:
                coeffs[0] += Fraction(term)
            except ValueError:
                pass

    return [float(coeffs[k] * factorials[k]) for k in range(max_order + 1)]


# ---------------------------------------------------------------------------
# Signed log helpers  (parameter-free, used for coeff3 and coeff4)
# ---------------------------------------------------------------------------

def _signed_log(x: torch.Tensor) -> torch.Tensor:
    """Signed log transform: sign(x) * ln(1 + |x|).

    Works for all real values including negatives and zero.
    Compresses large magnitudes while preserving the sign.
    """
    return x.sign() * (x.abs() + 1.0).log()


def _signed_exp(z: torch.Tensor) -> torch.Tensor:
    """Inverse of _signed_log: sign(z) * (exp(|z|) - 1)."""
    return z.sign() * (z.abs().exp() - 1.0)


# ---------------------------------------------------------------------------
# CoeffNormaliser — hybrid: z-score for coeff0-2, signed log for coeff3-4
# ---------------------------------------------------------------------------

# Which coefficient indices get which treatment
_ZSCORE_SLICE = slice(0, 3)   # coeff0, coeff1, coeff2  → z-score
_LOG_SLICE    = slice(3, 5)   # coeff3, coeff4           → signed log transform


class CoeffNormaliser:
    """
    Hybrid normaliser for the 5 Taylor derivative coefficients.

    coeff0 .. coeff2  (indices 0-2) : per-coefficient z-score
        normalised = (x - mean_k) / std_k   for k in {0, 1, 2}

    coeff3, coeff4   (indices 3-4) : signed log transform
        normalised = sign(x) * ln(1 + |x|)
        inverse    = sign(z) * (exp(|z|) - 1)

    The signed log is used for coeff3/coeff4 because their standard
    deviations are enormous (~7 000 and ~105 000 respectively), making
    plain z-score unstable.  The log transform compresses the scale
    while still preserving the sign of the derivative.

    Attributes
    ----------
    mean : Tensor of shape (3,)  — means  for coeff0..coeff2
    std  : Tensor of shape (3,)  — stdevs for coeff0..coeff2, clamped ≥ 1e-6
    """

    def __init__(self):
        self.mean: Optional[torch.Tensor] = None   # (3,)
        self.std:  Optional[torch.Tensor] = None   # (3,)

    def fit(self, coeffs: List[List[int]]) -> None:
        """Compute z-score stats from training coefficient rows.

        Only coeff0..coeff2 require fitting; the log transform is
        parameter-free, so coeff3/coeff4 need no statistics.

        Parameters
        ----------
        coeffs : list of N rows, each row is [c0, c1, c2, c3, c4]
        """
        t         = torch.tensor(coeffs, dtype=torch.float32)   # (N, 5)
        zscore    = t[:, _ZSCORE_SLICE]                          # (N, 3)
        self.mean = zscore.mean(dim=0)                           # (3,)
        self.std  = zscore.std(dim=0).clamp(min=1e-6)           # (3,)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the hybrid forward transform.

        Parameters
        ----------
        x : Tensor of shape (5,) or (B, 5) — raw rounded integer coefficients

        Returns
        -------
        Tensor same shape as x — normalised coefficients
        """
        out = x.clone().float()
        # Z-score: coeff0, coeff1, coeff2
        out[..., _ZSCORE_SLICE] = (
            (x[..., _ZSCORE_SLICE] - self.mean.to(x.device))
            / self.std.to(x.device)
        )
        # Signed log: coeff3, coeff4
        out[..., _LOG_SLICE] = _signed_log(x[..., _LOG_SLICE].float())
        return out

    def denormalise(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the hybrid inverse transform.

        Parameters
        ----------
        z : Tensor of shape (5,) or (B, 5) — normalised coefficients

        Returns
        -------
        Tensor same shape as z — recovered original-scale coefficients
        """
        out = z.clone().float()
        # Inverse z-score: coeff0, coeff1, coeff2
        out[..., _ZSCORE_SLICE] = (
            z[..., _ZSCORE_SLICE] * self.std.to(z.device)
            + self.mean.to(z.device)
        )
        # Inverse signed log: coeff3, coeff4
        out[..., _LOG_SLICE] = _signed_exp(z[..., _LOG_SLICE])
        return out

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean, "std": self.std}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.mean = data["mean"]
        self.std  = data["std"]


# ---------------------------------------------------------------------------
# CoeffDataset
# ---------------------------------------------------------------------------

class CoeffDataset(Dataset):
    """
    PyTorch Dataset for Taylor coefficient regression.

    Each item returns:
        src_ids : List[int]        — tokenized function_prefix
        coeffs  : Tensor[NUM_COEFFS] — rounded integer derivative values,
                                       normalised if a CoeffNormaliser is given

    Parameters
    ----------
    json_path  : path to the JSON dataset file
    tokenizer  : a fitted WordLevelBPETokenizer
    normaliser : optional fitted CoeffNormaliser (applies z-score to targets)
    """

    def __init__(
        self,
        json_path: str,
        tokenizer: WordLevelBPETokenizer,
        normaliser: Optional[CoeffNormaliser] = None,
    ):
        with open(json_path) as f:
            raw = json.load(f)

        self.tokenizer  = tokenizer
        self.normaliser = normaliser

        # Encode inputs
        self.src_ids: List[List[int]] = [
            tokenizer.encode(item["function_prefix"]) for item in raw
        ]

        # Extract and round coefficients (these are integers in practice)
        self.coeffs: List[List[int]] = [
            [round(v) for v in taylor_to_coeffs(item["taylor_series"])]
            for item in raw
        ]

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Tuple[List[int], torch.Tensor]:
        coeff_tensor = torch.tensor(self.coeffs[idx], dtype=torch.float32)
        if self.normaliser is not None:
            coeff_tensor = self.normaliser.normalise(coeff_tensor)
        return self.src_ids[idx], coeff_tensor

    def collate_fn(
        self,
        batch: List[Tuple[List[int], torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad source sequences and stack coefficient tensors.

        Returns
        -------
        src_tensor   : (B, src_max)  — padded token ids
        coeff_tensor : (B, NUM_COEFFS)
        src_lengths  : (B,)
        """
        src_batch, coeff_batch = zip(*batch)
        pad_id = self.tokenizer.pad_id

        src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
        src_max     = int(src_lengths.max())
        B           = len(src_batch)

        src_tensor = torch.full((B, src_max), pad_id, dtype=torch.long)
        for i, s in enumerate(src_batch):
            src_tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long)

        coeff_tensor = torch.stack(coeff_batch)   # (B, NUM_COEFFS)
        return src_tensor, coeff_tensor, src_lengths


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def get_dataloader(
    json_path: str,
    tokenizer: WordLevelBPETokenizer,
    batch_size: int,
    normaliser: Optional[CoeffNormaliser] = None,
    shuffle: bool = True,
) -> Tuple[DataLoader, CoeffDataset]:
    dataset = CoeffDataset(json_path, tokenizer, normaliser)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    JSON_PATH = "datasets/taylor_dataset_10k.json"

    # 1. Tokenizer
    tokenizer = WordLevelBPETokenizer(num_merges=100)
    raw_data  = json.load(open(JSON_PATH))
    tokenizer.fit([item["function_prefix"] for item in raw_data])
    print(f"Vocab size : {len(tokenizer)}")

    # 2. Dataset without normaliser
    ds = CoeffDataset(JSON_PATH, tokenizer, normaliser=None)
    print(f"Dataset size : {len(ds)}")

    # 3. Normaliser fitted on full dataset (for demo; in training fit on train split only)
    norm = CoeffNormaliser()
    norm.fit(ds.coeffs)
    print(f"Coeff mean : {norm.mean.tolist()}")
    print(f"Coeff std  : {norm.std.tolist()}")

    # 4. Dataset with normaliser
    ds_norm = CoeffDataset(JSON_PATH, tokenizer, normaliser=norm)
    src, coeffs, lengths = ds_norm.collate_fn([ds_norm[i] for i in range(4)])
    print(f"src shape    : {src.shape}")
    print(f"coeffs shape : {coeffs.shape}")
    print(f"lengths      : {lengths.tolist()}")

    # 5. Random sample
    idx = random.randint(0, len(ds) - 1)
    print(f"\nSample [{idx}]")
    print(f"  function_prefix : {raw_data[idx]['function_prefix']}")
    print(f"  taylor_series   : {raw_data[idx]['taylor_series']}")
    raw_coeffs = taylor_to_coeffs(raw_data[idx]["taylor_series"])
    print(f"  raw coeffs      : {raw_coeffs}")
    print(f"  rounded coeffs  : {[round(v) for v in raw_coeffs]}")
