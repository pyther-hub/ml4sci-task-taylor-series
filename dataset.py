import json
import os
import pickle
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]


# ---------------------------------------------------------------------------
# Word-Level BPE Tokenizer (special tokens baked into vocab at fixed IDs 0-3)
# ---------------------------------------------------------------------------

class WordLevelBPETokenizer:
    """
    BPE applied over whitespace-separated tokens.
    Special tokens (PAD, SOS, EOS, UNK) are always reserved at IDs 0-3.
    All BPE-derived vocab entries are assigned IDs starting from 4.
    """

    def __init__(self, num_merges: int = 200):
        self.num_merges = num_merges
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}

    # -- Special token ID properties --
    @property
    def pad_id(self) -> int:
        return self.vocab[PAD]

    @property
    def sos_id(self) -> int:
        return self.vocab[SOS]

    @property
    def eos_id(self) -> int:
        return self.vocab[EOS]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK]

    # -- BPE helpers --
    def _get_stats(self, corpus: List[List[str]]) -> Counter:
        pairs: Counter = Counter()
        for token_list in corpus:
            for i in range(len(token_list) - 1):
                pairs[(token_list[i], token_list[i + 1])] += 1
        return pairs

    def _merge_pair(
        self,
        pair: Tuple[str, str],
        corpus: List[List[str]],
    ) -> List[List[str]]:
        merged_corpus = []
        for token_list in corpus:
            i, new_tokens = 0, []
            while i < len(token_list):
                if (
                    i < len(token_list) - 1
                    and (token_list[i], token_list[i + 1]) == pair
                ):
                    new_tokens.append(pair[0] + " " + pair[1])
                    i += 2
                else:
                    new_tokens.append(token_list[i])
                    i += 1
            merged_corpus.append(new_tokens)
        return merged_corpus

    # -- Fit --
    def fit(self, sequences: List[str]):
        corpus = [s.split() for s in sequences if s.split()]

        for _ in range(self.num_merges):
            stats = self._get_stats(corpus)
            if not stats:
                break
            best_pair = stats.most_common(1)[0][0]
            self.merges.append(best_pair)
            corpus = self._merge_pair(best_pair, corpus)

        # Collect all BPE tokens plus base words
        vocab_set: set = set()
        for token_list in corpus:
            vocab_set.update(token_list)
        for seq in sequences:
            vocab_set.update(seq.split())

        # IDs 0-3 are reserved for special tokens; BPE vocab starts at 4
        self.vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        for tok in sorted(vocab_set):
            if tok not in self.vocab:          # don't overwrite special tokens
                self.vocab[tok] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # -- Encode --
    def encode(self, sequence: str) -> List[int]:
        token_list = sequence.split()
        for pair in self.merges:
            i, new_tokens = 0, []
            while i < len(token_list):
                if (
                    i < len(token_list) - 1
                    and (token_list[i], token_list[i + 1]) == pair
                ):
                    new_tokens.append(pair[0] + " " + pair[1])
                    i += 2
                else:
                    new_tokens.append(token_list[i])
                    i += 1
            token_list = new_tokens
        return [self.vocab.get(t, self.vocab[UNK]) for t in token_list]

    # -- Decode (strips PAD / SOS / EOS automatically) --
    def decode(self, ids: List[int]) -> str:
        skip = {self.pad_id, self.sos_id, self.eos_id}
        tokens = [self.inv_vocab.get(i, UNK) for i in ids if i not in skip]
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.vocab)

    # -- Save / Load --
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"merges": self.merges, "vocab": self.vocab}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.merges = data["merges"]
        self.vocab = data["vocab"]
        self.inv_vocab = {v: k for k, v in self.vocab.items()}


# ---------------------------------------------------------------------------
# Dual BPE Tokenizer — one tokenizer per side (input / output)
# ---------------------------------------------------------------------------

class DualBPETokenizer:
    """Wraps two independent WordLevelBPETokenizers, one per modality."""

    def __init__(self, num_merges: int = 200):
        self.input_tokenizer  = WordLevelBPETokenizer(num_merges)
        self.output_tokenizer = WordLevelBPETokenizer(num_merges)

    # -- Fit both sides from a JSON file --
    def fit(self, json_path: str):
        with open(json_path) as f:
            raw_data = json.load(f)
        input_seqs  = [item["function_prefix"] for item in raw_data]
        output_seqs = [item["taylor_prefix"]   for item in raw_data]
        self.input_tokenizer.fit(input_seqs)
        self.output_tokenizer.fit(output_seqs)

    # -- Encode / Decode convenience wrappers --
    def encode_input(self, seq: str) -> List[int]:
        return self.input_tokenizer.encode(seq)

    def encode_output(self, seq: str) -> List[int]:
        return self.output_tokenizer.encode(seq)

    def decode_input(self, ids: List[int]) -> str:
        return self.input_tokenizer.decode(ids)

    def decode_output(self, ids: List[int]) -> str:
        return self.output_tokenizer.decode(ids)

    # -- Vocab size helpers --
    @property
    def input_vocab_size(self) -> int:
        return len(self.input_tokenizer)

    @property
    def output_vocab_size(self) -> int:
        return len(self.output_tokenizer)

    # -- Save / Load --
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "input_merges":  self.input_tokenizer.merges,
                    "input_vocab":   self.input_tokenizer.vocab,
                    "output_merges": self.output_tokenizer.merges,
                    "output_vocab":  self.output_tokenizer.vocab,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.input_tokenizer.merges   = data["input_merges"]
        self.input_tokenizer.vocab    = data["input_vocab"]
        self.input_tokenizer.inv_vocab = {v: k for k, v in data["input_vocab"].items()}
        self.output_tokenizer.merges   = data["output_merges"]
        self.output_tokenizer.vocab    = data["output_vocab"]
        self.output_tokenizer.inv_vocab = {v: k for k, v in data["output_vocab"].items()}


# ---------------------------------------------------------------------------
# Dataset  (tokenizer is injected — no fitting happens here)
# ---------------------------------------------------------------------------

class TaylorDataset(Dataset):
    """
    Reads JSON, encodes sequences with a pre-fitted DualBPETokenizer.
    The tokenizer must already be fitted before passing it in.
    """

    def __init__(self, json_path: str, tokenizer: DualBPETokenizer):
        with open(json_path) as f:
            raw = json.load(f)

        self.tokenizer = tokenizer
        inp_tok = tokenizer.input_tokenizer
        out_tok = tokenizer.output_tokenizer

        # Source: plain encoding
        self.src_ids: List[List[int]] = [
            inp_tok.encode(item["function_prefix"]) for item in raw
        ]
        # Target: wrapped with <SOS> … <EOS>
        self.tgt_ids: List[List[int]] = [
            [out_tok.sos_id]
            + out_tok.encode(item["taylor_prefix"])
            + [out_tok.eos_id]
            for item in raw
        ]

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.src_ids[idx], self.tgt_ids[idx]

    def collate_fn(
        self,
        batch: List[Tuple[List[int], List[int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inp_tok = self.tokenizer.input_tokenizer
        out_tok = self.tokenizer.output_tokenizer

        src_batch, tgt_batch = zip(*batch)

        src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
        tgt_lengths = torch.tensor([len(t) for t in tgt_batch], dtype=torch.long)

        src_max = int(src_lengths.max())
        tgt_max = int(tgt_lengths.max())
        B = len(src_batch)

        src_tensor = torch.full((B, src_max), inp_tok.pad_id, dtype=torch.long)
        tgt_tensor = torch.full((B, tgt_max), out_tok.pad_id, dtype=torch.long)

        for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
            src_tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            tgt_tensor[i, : len(t)] = torch.tensor(t, dtype=torch.long)

        return src_tensor, tgt_tensor, src_lengths, tgt_lengths


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def get_dataloader(
    json_path: str,
    tokenizer: DualBPETokenizer,
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[DataLoader, TaylorDataset]:
    dataset = TaylorDataset(json_path, tokenizer)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    JSON_PATH  = "taylor_dataset_sample.json"
    TOK_PATH   = "dual_bpe_tokenizer.pkl"
    BATCH_SIZE = 8
    NUM_MERGES = 100

    # ------------------------------------------------------------------
    # 1. Fit tokenizer (or load from disk if already saved)
    # ------------------------------------------------------------------
    tokenizer = DualBPETokenizer(num_merges=NUM_MERGES)

    if os.path.exists(TOK_PATH):
        print(f"[tokenizer] Loading from '{TOK_PATH}' ...")
        tokenizer.load(TOK_PATH)
    else:
        print(f"[tokenizer] Fitting on '{JSON_PATH}' ...")
        tokenizer.fit(JSON_PATH)
        tokenizer.save(TOK_PATH)
        print(f"[tokenizer] Saved to '{TOK_PATH}'")

    print(f"Input  vocab size : {tokenizer.input_vocab_size}")
    print(f"Output vocab size : {tokenizer.output_vocab_size}")
    print()

    # ------------------------------------------------------------------
    # 2. Build dataset & dataloader using the fitted tokenizer
    # ------------------------------------------------------------------
    loader, dataset = get_dataloader(
        JSON_PATH, tokenizer, batch_size=BATCH_SIZE, shuffle=False
    )
    print(f"Dataset size      : {len(dataset)}")
    print()

    # ------------------------------------------------------------------
    # 3. Inspect first batch
    # ------------------------------------------------------------------
    src, tgt, src_lens, tgt_lens = next(iter(loader))

    print(f"src tensor shape  : {src.shape}")
    print(f"tgt tensor shape  : {tgt.shape}")
    print(f"src lengths       : {src_lens.tolist()}")
    print(f"tgt lengths       : {tgt_lens.tolist()}")
    print()

    # ------------------------------------------------------------------
    # 4. Decode first example (PAD / SOS / EOS stripped automatically)
    # ------------------------------------------------------------------
    src_decoded = tokenizer.decode_input(src[0].tolist())
    tgt_decoded = tokenizer.decode_output(tgt[0].tolist())
    print(f"decoded src[0]    : {src_decoded!r}")
    print(f"decoded tgt[0]    : {tgt_decoded!r}")






