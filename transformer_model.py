import math
from typing import List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                                    # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))                           # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Seq2Seq Transformer (encoder-decoder)
# ---------------------------------------------------------------------------

class Seq2SeqTransformer(nn.Module):
    """Transformer encoder-decoder for sequence-to-sequence tasks.

    Presents the same public interface as Seq2SeqLSTM so it can be swapped
    into the training pipeline without any other changes:

        forward(src, tgt)                -> logits (B, tgt_len-1, V_out)
        generate(src, tokenizer, ...)    -> (token_ids, decoded_string)
        generate_batch(src, tokenizer, ...)  -> List[(token_ids, decoded_string)]

    Architecture
    ------------
    * Sinusoidal positional encoding shared between encoder and decoder sides.
    * nn.TransformerEncoder  (batch_first=True)
    * nn.TransformerDecoder  (batch_first=True)
    * Linear output projection to target vocabulary.

    Parameters
    ----------
    input_vocab_size   : source vocabulary size
    output_vocab_size  : target vocabulary size
    d_model            : model / embedding dimension (must be divisible by nhead)
    nhead              : number of attention heads
    num_encoder_layers : number of encoder layers
    num_decoder_layers : number of decoder layers
    dim_feedforward    : inner dimension of the position-wise feed-forward network
    dropout            : dropout probability applied throughout
    max_seq_len        : maximum sequence length supported by positional encoding
    src_pad_id         : padding token ID for source sequences (used in attention mask)
    tgt_pad_id         : padding token ID for target sequences (used in attention mask)
    """

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        src_pad_id: int = 0,
        tgt_pad_id: int = 0,
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model    = d_model
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

        # Embeddings
        self.src_embedding = nn.Embedding(input_vocab_size,  d_model)
        self.tgt_embedding = nn.Embedding(output_vocab_size, d_model)

        # Positional encoding (shared scale, separate instances not required)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, enable_nested_tensor=False)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output projection
        self.fc_out = nn.Linear(d_model, output_vocab_size)

        self._init_weights()

    # -----------------------------------------------------------------------
    # Weight initialisation
    # -----------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.src_embedding.weight)
        nn.init.xavier_uniform_(self.tgt_embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular bool mask that prevents attending to future positions."""
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    def _encode(
        self, src: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder and return (memory, src_key_padding_mask).

        Args:
            src : (B, src_len)

        Returns:
            memory              : (B, src_len, d_model)
            src_key_padding_mask: (B, src_len)  — True where PAD
        """
        src_pad_mask = src == self.src_pad_id                                 # (B, src_len)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)                                   # (B, src_len, d_model)
        memory  = self.transformer_encoder(
            src_emb, src_key_padding_mask=src_pad_mask
        )
        return memory, src_pad_mask

    # -----------------------------------------------------------------------
    # forward — teacher-forced training pass
    # -----------------------------------------------------------------------
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src : (B, src_len)   — source token IDs
            tgt : (B, tgt_len)   — target token IDs (including <SOS> prefix)

        Returns:
            logits : (B, tgt_len - 1, output_vocab_size)
                     Predictions for positions 1 … tgt_len-1
                     (i.e. we predict the next token at every step).
        """
        tgt_input = tgt[:, :-1]                                               # (B, tgt_len-1)
        tgt_len   = tgt_input.shape[1]

        # Encoder
        memory, src_pad_mask = self._encode(src)

        # Decoder masks
        tgt_causal_mask = self._causal_mask(tgt_len, src.device)              # (tgt_len, tgt_len)
        tgt_pad_mask    = tgt_input == self.tgt_pad_id                        # (B, tgt_len-1)

        # Decoder
        tgt_emb = self.tgt_embedding(tgt_input) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)                                   # (B, tgt_len-1, d_model)

        out = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )                                                                      # (B, tgt_len-1, d_model)

        return self.fc_out(out)                                                # (B, tgt_len-1, V_out)

    # -----------------------------------------------------------------------
    # generate_batch — batched greedy decoding (fast validation path)
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate_batch(
        self,
        src: torch.Tensor,
        tokenizer,
        max_len: int = 128,
    ) -> List[Tuple[List[int], str]]:
        """Greedy decode a full batch in one pass.

        Runs the encoder once for the entire batch and steps the decoder
        autoregressively in parallel (one pass per time step).

        Args:
            src       : (B, src_len) — batch of source sequences
            tokenizer : DualBPETokenizer (already fitted)
            max_len   : maximum tokens to generate per sequence

        Returns:
            List of (token_ids, decoded_string) — one entry per sample.
        """
        out_tok = tokenizer.output_tokenizer
        sos_id  = out_tok.vocab["<SOS>"]
        eos_id  = out_tok.vocab["<EOS>"]
        pad_id  = out_tok.pad_id

        B      = src.shape[0]
        device = src.device

        memory, src_pad_mask = self._encode(src)

        # Decoder input starts with SOS for every sequence in the batch
        generated = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        outputs   = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len):
            tgt_emb  = self.tgt_embedding(generated) * math.sqrt(self.d_model)
            tgt_emb  = self.pos_encoder(tgt_emb)
            tgt_mask = self._causal_mask(generated.shape[1], device)

            out = self.transformer_decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_pad_mask,
            )                                                                  # (B, t+1, d_model)

            logits   = self.fc_out(out[:, -1, :])                             # (B, V_out)
            next_ids = logits.argmax(dim=-1)                                  # (B,)
            next_ids = next_ids.masked_fill(finished, pad_id)
            outputs[:, t] = next_ids
            finished = finished | (next_ids == eos_id)
            generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)

            if finished.all():
                break

        results: List[Tuple[List[int], str]] = []
        for i in range(B):
            ids: List[int] = []
            for tid in outputs[i].tolist():
                if tid == eos_id:
                    ids.append(tid)   # include EOS for strict_sentence_accuracy
                    break
                if tid != pad_id:
                    ids.append(tid)
            results.append((ids, out_tok.decode([t for t in ids if t != eos_id])))
        return results

    # -----------------------------------------------------------------------
    # generate — autoregressive inference (greedy or beam search)
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        tokenizer,
        max_len: int = 128,
        beam_width: int = 1,
    ) -> Tuple[List[int], str]:
        """
        Args:
            src        : (1, src_len)  — single source sequence (batch size 1)
            tokenizer  : DualBPETokenizer whose output_tokenizer must have
                         '<SOS>' and '<EOS>' in its vocab.
            max_len    : maximum number of tokens to generate
            beam_width : 1 → greedy decoding, >1 → beam search

        Returns:
            (token_ids, decoded_string)
        """
        out_tok = tokenizer.output_tokenizer
        sos_id  = out_tok.vocab["<SOS>"]
        eos_id  = out_tok.vocab["<EOS>"]

        memory, src_pad_mask = self._encode(src)

        if beam_width <= 1:
            return self._greedy(memory, src_pad_mask, sos_id, eos_id, max_len, out_tok)
        else:
            return self._beam_search(
                memory, src_pad_mask, sos_id, eos_id, max_len, beam_width, out_tok
            )

    # -----------------------------------------------------------------------
    # Greedy decoding
    # -----------------------------------------------------------------------
    def _greedy(
        self,
        memory: torch.Tensor,
        src_pad_mask: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int,
        tokenizer,
    ) -> Tuple[List[int], str]:
        device    = memory.device
        generated = [sos_id]

        for _ in range(max_len):
            tgt_seq  = torch.tensor([generated], dtype=torch.long, device=device)  # (1, t)
            tgt_emb  = self.tgt_embedding(tgt_seq) * math.sqrt(self.d_model)
            tgt_emb  = self.pos_encoder(tgt_emb)
            tgt_mask = self._causal_mask(len(generated), device)

            out    = self.transformer_decoder(
                tgt_emb, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_pad_mask,
            )                                                                  # (1, t, d_model)
            logits  = self.fc_out(out[:, -1, :])                              # (1, V_out)
            next_id = logits.argmax(dim=-1).item()
            if next_id == eos_id:
                generated.append(next_id)   # include EOS for strict_sentence_accuracy
                break
            generated.append(next_id)

        ids     = generated[1:]   # strip leading <SOS>, keep EOS if present
        decoded = tokenizer.decode([t for t in ids if t != eos_id])
        return ids, decoded

    # -----------------------------------------------------------------------
    # Beam search decoding
    # -----------------------------------------------------------------------
    def _beam_search(
        self,
        memory: torch.Tensor,
        src_pad_mask: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int,
        beam_width: int,
        tokenizer,
    ) -> Tuple[List[int], str]:
        """
        Beam state: (neg_log_prob, token_ids)
        The transformer reconstructs decoder state from the full token sequence,
        so no recurrent state needs to be carried between steps.
        We use neg_log_prob so Python's sort gives best (lowest neg) first.
        """
        device = memory.device

        # Seed the beam: decode from [SOS] → first token predictions
        init_seq = torch.tensor([[sos_id]], dtype=torch.long, device=device)
        tgt_emb  = self.tgt_embedding(init_seq) * math.sqrt(self.d_model)
        tgt_emb  = self.pos_encoder(tgt_emb)
        out      = self.transformer_decoder(
            tgt_emb, memory, memory_key_padding_mask=src_pad_mask
        )                                                                      # (1, 1, d_model)
        log_probs       = torch.log_softmax(self.fc_out(out[:, -1, :]).squeeze(0), dim=-1)
        topk_lp, topk_ids = log_probs.topk(beam_width)

        # heap entry: (neg_score, token_ids_without_SOS)
        beam: List[Tuple[float, List[int]]] = [
            (-lp.item(), [tid.item()])
            for lp, tid in zip(topk_lp, topk_ids)
        ]

        completed: List[Tuple[float, List[int]]] = []

        for _ in range(max_len - 1):
            candidates: List[Tuple[float, List[int]]] = []

            for neg_score, ids in beam:
                if ids[-1] == eos_id:
                    completed.append((neg_score, ids))   # keep <EOS> for strict_sentence_accuracy
                    continue

                # Decode from [SOS] + ids so far
                seq     = [sos_id] + ids
                tgt_seq = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_emb = self.tgt_embedding(tgt_seq) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoder(tgt_emb)
                tgt_msk = self._causal_mask(len(seq), device)

                out = self.transformer_decoder(
                    tgt_emb, memory,
                    tgt_mask=tgt_msk,
                    memory_key_padding_mask=src_pad_mask,
                )                                                              # (1, t, d_model)
                log_probs       = torch.log_softmax(
                    self.fc_out(out[:, -1, :]).squeeze(0), dim=-1
                )
                topk_lp, topk_ids = log_probs.topk(beam_width)

                for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                    candidates.append((neg_score - lp, ids + [tid]))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0])
            beam = candidates[:beam_width]

        # Collect any unfinished hypotheses
        for neg_score, ids in beam:
            completed.append((neg_score, ids))

        if not completed:
            completed = list(beam)

        # Best hypothesis = lowest neg log prob (highest log prob)
        _, best_ids = min(completed, key=lambda x: x[0])
        decoded = tokenizer.decode([t for t in best_ids if t != eos_id])
        return best_ids, decoded


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dataset import DualBPETokenizer

    # --- Build a tiny dual tokenizer with special tokens ---
    dual_tok = DualBPETokenizer(num_merges=10)

    input_seqs  = ["sin x + cos x", "x ^ 2 + x + 1", "exp x - 1"]
    output_seqs = ["1 + x + x ^ 2 / 2", "x - x ^ 3 / 6", "x + x ^ 2 / 2"]

    dual_tok.input_tokenizer.fit(input_seqs)
    dual_tok.output_tokenizer.fit(output_seqs)

    for special in ["<PAD>", "<SOS>", "<EOS>"]:
        if special not in dual_tok.output_tokenizer.vocab:
            new_id = len(dual_tok.output_tokenizer.vocab)
            dual_tok.output_tokenizer.vocab[special] = new_id
            dual_tok.output_tokenizer.inv_vocab[new_id] = special

    INPUT_VOCAB  = len(dual_tok.input_tokenizer.vocab)
    OUTPUT_VOCAB = len(dual_tok.output_tokenizer.vocab)

    print(f"input  vocab size : {INPUT_VOCAB}")
    print(f"output vocab size : {OUTPUT_VOCAB}")

    model = Seq2SeqTransformer(
        input_vocab_size=INPUT_VOCAB,
        output_vocab_size=OUTPUT_VOCAB,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    model.eval()

    BATCH, SRC_LEN, TGT_LEN = 4, 10, 12

    src = torch.randint(0, INPUT_VOCAB,  (BATCH, SRC_LEN))
    tgt = torch.randint(0, OUTPUT_VOCAB, (BATCH, TGT_LEN))

    # --- Forward pass ---
    logits = model(src, tgt)
    print(f"forward  logits shape : {logits.shape}")
    assert logits.shape == (BATCH, TGT_LEN - 1, OUTPUT_VOCAB)

    # --- Greedy generate ---
    src_single = src[:1]
    ids_greedy, text_greedy = model.generate(src_single, dual_tok, max_len=20, beam_width=1)
    print(f"greedy   token ids    : {ids_greedy}")
    print(f"greedy   decoded      : {text_greedy!r}")

    # --- Beam search generate ---
    ids_beam, text_beam = model.generate(src_single, dual_tok, max_len=20, beam_width=3)
    print(f"beam(3)  token ids    : {ids_beam}")
    print(f"beam(3)  decoded      : {text_beam!r}")

    # --- Batch greedy generate ---
    batch_results = model.generate_batch(src, dual_tok, max_len=20)
    print(f"generate_batch count  : {len(batch_results)}")

    print("\nAll checks passed.")
