"""
skeleton_model.py
=================
Encoder-only Transformer with an MLP regression head for Taylor coefficient
prediction.

Pipeline
--------
1. Embed + positionally encode the tokenized function_prefix.
2. Pass through a Transformer encoder.
3. Mean-pool the encoder output (PAD positions excluded).
4. Pass the pooled vector through a 2-layer MLP.
5. Output NUM_COEFFS floats (in normalised space during training;
   de-normalised + rounded to integers at inference time).

The denominator of each Taylor term is the fixed factorial k! = [1,1,2,6,24].
This model predicts ONLY the numerators a_k = raw_coeff_k * k!.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn

NUM_COEFFS = 5   # a0 … a4  (Taylor order 0..4)


# ---------------------------------------------------------------------------
# Positional Encoding (identical to existing codebase)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# CoeffPredictorTransformer
# ---------------------------------------------------------------------------

class CoeffPredictorTransformer(nn.Module):
    """
    Transformer encoder + MLP head for Taylor coefficient regression.

    Parameters
    ----------
    vocab_size         : source vocabulary size
    d_model            : embedding / model dimension (must be divisible by nhead)
    nhead              : number of attention heads
    num_encoder_layers : number of Transformer encoder layers
    dim_feedforward    : inner FFN dimension inside each encoder layer
    dropout            : dropout probability applied throughout
    max_seq_len        : maximum input sequence length (positional encoding)
    src_pad_id         : PAD token id — masked out in attention and mean-pooling
    num_coeffs         : number of output coefficients (default 5)
    mlp_hidden         : hidden dimension of the MLP regression head

    Forward
    -------
    src : (B, S) — tokenized function_prefix
    returns (B, num_coeffs) — predicted (normalised) Taylor numerators
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int            = 256,
        nhead: int              = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int    = 512,
        dropout: float          = 0.1,
        max_seq_len: int        = 256,
        src_pad_id: int         = 0,
        num_coeffs: int         = NUM_COEFFS,
        mlp_hidden: int         = 256,
    ):
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        self.src_pad_id = src_pad_id
        self.d_model    = d_model
        self.num_coeffs = num_coeffs

        # --- Input side ---
        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, enable_nested_tensor=False
        )

        # --- MLP regression head: d_model → mlp_hidden → num_coeffs ---
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_coeffs),
        )

        self._init_weights()

    # -----------------------------------------------------------------------
    # Weight init
    # -----------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # -----------------------------------------------------------------------
    # Mean pooling (excluding PAD positions)
    # -----------------------------------------------------------------------

    def _mean_pool(
        self, enc_out: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean-pool encoder output over non-PAD positions.

        Args
        ----
        enc_out : (B, S, d_model)
        src     : (B, S) raw token ids

        Returns
        -------
        pooled  : (B, d_model)
        """
        mask   = (src != self.src_pad_id).float().unsqueeze(-1)   # (B, S, 1)
        pooled = (enc_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        src : (B, S) — tokenized function_prefix

        Returns
        -------
        (B, num_coeffs) — predicted (normalised) Taylor numerators
        """
        pad_mask = src == self.src_pad_id                          # (B, S)
        emb      = self.embedding(src) * math.sqrt(self.d_model)   # scale
        emb      = self.pos_encoder(emb)                           # (B, S, d_model)
        enc_out  = self.encoder(emb, src_key_padding_mask=pad_mask) # (B, S, d_model)
        pooled   = self._mean_pool(enc_out, src)                   # (B, d_model)
        return self.head(pooled)                                   # (B, num_coeffs)

    # -----------------------------------------------------------------------
    # Predict — de-normalise + round to integers
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        src: torch.Tensor,
        normaliser=None,          # CoeffNormaliser | None
    ) -> torch.Tensor:
        """
        Run forward pass, optionally de-normalise, and round to integers.

        Args
        ----
        src        : (B, S) tokenized inputs
        normaliser : CoeffNormaliser to invert the z-score (pass None if no norm)

        Returns
        -------
        (B, num_coeffs) int tensor of rounded predicted numerators
        """
        self.eval()
        out = self.forward(src)                    # (B, num_coeffs) normalised
        if normaliser is not None:
            out = normaliser.denormalise(out)      # back to original scale
        return out.round().long()


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dataset import WordLevelBPETokenizer

    VOCAB_SIZE = 50
    B, S       = 4, 12

    model = CoeffPredictorTransformer(
        vocab_size         = VOCAB_SIZE,
        d_model            = 64,
        nhead              = 4,
        num_encoder_layers = 2,
        dim_feedforward    = 128,
        dropout            = 0.1,
        src_pad_id         = 0,
        num_coeffs         = 5,
        mlp_hidden         = 64,
    )
    model.eval()

    src    = torch.randint(1, VOCAB_SIZE, (B, S))
    src[0, 8:] = 0   # simulate PAD on first sequence

    out = model(src)
    print(f"forward output shape : {out.shape}")   # (4, 5)
    assert out.shape == (B, NUM_COEFFS)

    rounded = out.round().long()
    print(f"rounded predictions  : {rounded.tolist()}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters : {n_params:,}")

    print("\nAll checks passed.")
