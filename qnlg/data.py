"""Word-level vocabulary and batching for the corpora built by prepare_data.py."""

from __future__ import annotations

import json
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader

CORPORA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "corpora")
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
DATASETS = ["simple_sentences", "quantum_phrases", "short_stories", "haiku", "proverbs", "ptb"]


class Vocab:
    def __init__(self, train_sents, min_freq=2):
        c = Counter(w for s in train_sents for w in s)
        words = sorted(w for w, k in c.items() if k >= min_freq and w not in (PAD, UNK, BOS, EOS))
        self.itos = [PAD, UNK, BOS, EOS] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad, self.unk, self.bos, self.eos = 0, 1, 2, 3

    def __len__(self):
        return len(self.itos)

    def encode(self, sent, max_len):
        ids = [self.bos] + [self.stoi.get(w, self.unk) for w in sent] + [self.eos]
        return ids[:max_len]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.eos:
                break
            if i in (self.pad, self.bos):
                continue
            out.append(self.itos[i])
        return out


def load_corpus(name):
    d = os.path.join(CORPORA, name)
    sp = {s: [ln.split() for ln in open(os.path.join(d, f"{s}.txt")) if ln.strip()] for s in ("train", "valid", "test")}
    meta = json.load(open(os.path.join(d, "meta.json")))
    return sp, meta


def map_unk(sent, vocab):
    return [w if w in vocab.stoi else UNK for w in sent]


def make_loader(sents, vocab, max_len, batch_size, shuffle, seed=0):
    data = [vocab.encode(s, max_len) for s in sents]

    def collate(batch):
        T = max(len(x) for x in batch)
        arr = torch.full((len(batch), T), vocab.pad, dtype=torch.long)
        for i, x in enumerate(batch):
            arr[i, : len(x)] = torch.tensor(x)
        return arr[:, :-1], arr[:, 1:]

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=g)
