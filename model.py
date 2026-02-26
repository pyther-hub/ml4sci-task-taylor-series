from typing import List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # src: (B, src_len)
        embedded = self.dropout(self.embedding(src))          # (B, src_len, emb_dim)
        _, (hidden, cell) = self.lstm(embedded)               # hidden/cell: (num_layers, B, H)
        return hidden, cell


# ---------------------------------------------------------------------------
# Decoder (single step)
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(
        self,
        output_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # token: (B,)  →  unsqueeze to (B, 1) for batch_first LSTM
        token = token.unsqueeze(1)                            # (B, 1)
        embedded = self.dropout(self.embedding(token))        # (B, 1, emb_dim)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: (B, 1, H)
        logits = self.fc_out(output.squeeze(1))               # (B, output_vocab_size)
        return logits, hidden, cell


# ---------------------------------------------------------------------------
# Seq2Seq model
# ---------------------------------------------------------------------------

class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_vocab_size, embedding_dim, hidden_dim, num_layers, dropout
        )
        self.decoder = Decoder(
            output_vocab_size, embedding_dim, hidden_dim, num_layers, dropout
        )

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
        _, tgt_len = tgt.shape

        hidden, cell = self.encoder(src)

        # Teacher forcing: feed tgt[:, t] to predict tgt[:, t+1]
        all_logits = []
        for t in range(tgt_len - 1):
            logits, hidden, cell = self.decoder(tgt[:, t], hidden, cell)
            all_logits.append(logits.unsqueeze(1))             # (B, 1, V_out)

        return torch.cat(all_logits, dim=1)                    # (B, tgt_len-1, V_out)

    # -----------------------------------------------------------------------
    # generate — autoregressive inference (greedy or beam search)
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        tokenizer,                 # DualBPETokenizer
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
        sos_id = out_tok.vocab["<SOS>"]
        eos_id = out_tok.vocab["<EOS>"]

        hidden, cell = self.encoder(src)

        if beam_width <= 1:
            return self._greedy(hidden, cell, sos_id, eos_id, max_len, out_tok)
        else:
            return self._beam_search(
                hidden, cell, sos_id, eos_id, max_len, beam_width, out_tok
            )

    # -----------------------------------------------------------------------
    # Greedy decoding
    # -----------------------------------------------------------------------
    def _greedy(
        self,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int,
        tokenizer,
    ) -> Tuple[List[int], str]:
        device = hidden.device
        token = torch.tensor([sos_id], device=device)        # (1,)
        generated: List[int] = []

        for _ in range(max_len):
            logits, hidden, cell = self.decoder(token, hidden, cell)
            next_id = logits.argmax(dim=-1).item()
            if next_id == eos_id:
                break
            generated.append(next_id)
            token = torch.tensor([next_id], device=device)

        decoded = tokenizer.decode(generated)
        return generated, decoded

    # -----------------------------------------------------------------------
    # Beam search decoding
    # -----------------------------------------------------------------------
    def _beam_search(
        self,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int,
        beam_width: int,
        tokenizer,
    ) -> Tuple[List[int], str]:
        """
        Beam state: (neg_log_prob, token_ids, hidden, cell)
        We use neg_log_prob so Python's min-heap gives best (lowest neg) first.
        """
        device = hidden.device

        # Initial beam: single hypothesis starting with <SOS>
        init_token = torch.tensor([sos_id], device=device)
        logits, h0, c0 = self.decoder(init_token, hidden, cell)
        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # (V_out,)

        # Seed the beam with top-k tokens
        topk_log_probs, topk_ids = log_probs.topk(beam_width)
        # heap entry: (-score, token_list, hidden, cell)
        beam = []
        for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
            beam.append((-lp, [tid], h0, c0))

        completed: List[Tuple[float, List[int]]] = []

        for _ in range(max_len - 1):
            candidates = []
            for neg_score, ids, h, c in beam:
                last_id = ids[-1]
                if last_id == eos_id:
                    completed.append((neg_score, ids[:-1]))  # drop <EOS> from output
                    continue

                token = torch.tensor([last_id], device=device)
                logits, h_new, c_new = self.decoder(token, h, c)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                topk_log_probs, topk_ids = log_probs.topk(beam_width)
                for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    new_score = neg_score - lp          # accumulate neg log probs
                    candidates.append((new_score, ids + [tid], h_new, c_new))

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:beam_width]

        # Collect any unfinished hypotheses
        for neg_score, ids, _, _ in beam:
            last = ids[-1] if ids else eos_id
            if last == eos_id:
                completed.append((neg_score, ids[:-1]))
            else:
                completed.append((neg_score, ids))

        if not completed:
            completed = [(neg_score, ids) for neg_score, ids, _, _ in beam]

        # Best hypothesis = lowest neg log prob (highest log prob)
        _, best_ids = min(completed, key=lambda x: x[0])
        decoded = tokenizer.decode(best_ids)
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

    # Manually populate both sides (simulates fitting on real data)
    input_seqs  = ["sin x + cos x", "x ^ 2 + x + 1", "exp x - 1"]
    output_seqs = ["1 + x + x ^ 2 / 2", "x - x ^ 3 / 6", "x + x ^ 2 / 2"]

    dual_tok.input_tokenizer.fit(input_seqs)
    dual_tok.output_tokenizer.fit(output_seqs)

    # Inject special tokens into output tokenizer
    # (in real training, prepend them to sequences before calling fit)
    for special in ["<PAD>", "<SOS>", "<EOS>"]:
        if special not in dual_tok.output_tokenizer.vocab:
            new_id = len(dual_tok.output_tokenizer.vocab)
            dual_tok.output_tokenizer.vocab[special] = new_id
            dual_tok.output_tokenizer.inv_vocab[new_id] = special

    INPUT_VOCAB  = len(dual_tok.input_tokenizer.vocab)
    OUTPUT_VOCAB = len(dual_tok.output_tokenizer.vocab)

    print(f"input  vocab size : {INPUT_VOCAB}")
    print(f"output vocab size : {OUTPUT_VOCAB}")

    model = Seq2SeqLSTM(
        input_vocab_size=INPUT_VOCAB,
        output_vocab_size=OUTPUT_VOCAB,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=2,
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

    # --- Greedy generate (pass full DualBPETokenizer) ---
    src_single = src[:1]
    ids_greedy, text_greedy = model.generate(
        src_single, dual_tok, max_len=20, beam_width=1
    )
    print(f"greedy   token ids    : {ids_greedy}")
    print(f"greedy   decoded      : {text_greedy!r}")

    # --- Beam search generate ---
    ids_beam, text_beam = model.generate(
        src_single, dual_tok, max_len=20, beam_width=3
    )
    print(f"beam(3)  token ids    : {ids_beam}")
    print(f"beam(3)  decoded      : {text_beam!r}")

    print("\nAll checks passed.")
