"""Build the six evaluation corpora with de-duplicated, disjoint train/valid/test splits.

    python -m qnlg.prepare_data            # writes corpora/<name>/{train,valid,test}.txt

Corpora
-------
simple_sentences  synthetic (this work): full enumeration of a subject-verb-PP grammar
quantum_phrases   synthetic (this work): 10 hand-written phrases + a term-action-object grammar
short_stories     TinyStories (Eldan & Li, 2023; CDLA-Sharing-1.0): first sentence of each story
haiku             statworx/haiku on Hugging Face (aggregated public haiku collections)
proverbs          Wiktionary "Category:English proverbs" (CC BY-SA 4.0), page titles
ptb               Penn Treebank, Mikolov et al. (2010) pre-processed version, standard splits

Every non-PTB corpus is de-duplicated *before* splitting, so no test sentence
appears verbatim in the training set (checked at the end of this script).
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sys
import urllib.parse
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "corpora")
RAW = os.path.join(ROOT, "_raw")
UA = {"User-Agent": "qnlg-data-prep/1.0 (academic research script)"}
SEED = 20251125

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:/]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("’", "'"))


def fetch(url: str, dest: str) -> str:
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, dest)
    if not os.path.exists(path):
        print(f"  downloading {url}")
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------

def simple_sentences() -> list[str]:
    subjects = ["the cat", "a dog", "the bird", "a child", "the man", "a woman",
                "the tree", "a flower", "the sun", "the moon"]
    verbs = ["runs", "walks", "flies", "jumps", "sits", "stands", "sleeps", "plays", "sings", "dances"]
    objects = ["in the park", "on the table", "under the tree", "near the house", "by the river",
               "in the garden", "on the hill", "in the sky", "through the forest", "across the field"]
    base = ["The cat sits on the mat.", "Birds fly in the sky.", "Water flows down the river.",
            "The sun shines bright today.", "Children play in the park.", "Books contain many stories.",
            "Music fills the quiet room.", "Flowers bloom in spring time.", "Dogs run fast through fields.",
            "Stars twinkle at night."]
    return base + [f"{s} {v} {o}." for s in subjects for v in verbs for o in objects]


def quantum_phrases() -> list[str]:
    base = ["Quantum entanglement connects distant particles.",
            "The wave function collapses upon measurement.",
            "Superposition allows multiple states simultaneously.",
            "Quantum computing uses qubits instead of bits.",
            "Uncertainty principle limits precise knowledge.",
            "Quantum tunneling enables barrier penetration.",
            "Coherence time determines quantum state stability.",
            "Bell inequalities test local realism.",
            "Quantum gates manipulate qubit states.",
            "Decoherence destroys quantum information."]
    terms = ["quantum", "qubit", "superposition", "entanglement", "coherence", "decoherence",
             "measurement", "wave function", "probability"]
    actions = ["creates", "destroys", "measures", "controls", "manipulates", "observes",
               "influences", "affects", "determines"]
    props = ["states", "systems", "particles", "information", "data", "circuits", "gates",
             "algorithms", "computations"]
    return base + [f"{t.capitalize()} {a} {p} in quantum systems." for t in terms for a in actions for p in props]


def short_stories() -> list[str]:
    import pandas as pd
    p = fetch("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/"
              "validation-00000-of-00001-869c898b519ad725.parquet", "tinystories_valid.parquet")
    out = []
    for story in pd.read_parquet(p)["text"]:
        story = story.strip().replace("\n", " ")
        m = re.match(r"(.+?[.!?])(\s|$)", story)
        if m:
            out.append(m.group(1))
    return out


def haiku() -> list[str]:
    import pandas as pd
    p = fetch("https://huggingface.co/datasets/statworx/haiku/resolve/main/train.parquet", "haiku.parquet")
    df = pd.read_parquet(p)
    df = df[df["source"] != "twaiku"]  # drop the Twitter-derived subset (conversational, noisy)
    texts = []
    for t in df["text"]:
        if not isinstance(t, str) or not t.isascii():
            continue
        # the source appends an artificial '.' to every line; strip it
        lines = [ln.strip().rstrip(".").strip() for ln in t.split("/") if ln.strip()]
        if len(lines) == 3:
            texts.append(" / ".join(lines))
    return texts


def proverbs() -> list[str]:
    path = os.path.join(RAW, "wiktionary_english_proverbs.json")
    if not os.path.exists(path):
        os.makedirs(RAW, exist_ok=True)
        titles, cont = [], {}
        while True:
            q = {"action": "query", "list": "categorymembers", "cmtitle": "Category:English proverbs",
                 "cmlimit": "500", "cmnamespace": "0", "format": "json", **cont}
            url = "https://en.wiktionary.org/w/api.php?" + urllib.parse.urlencode(q)
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                d = json.load(r)
            titles += [m["title"] for m in d["query"]["categorymembers"]]
            if "continue" not in d:
                break
            cont = {"cmcontinue": d["continue"]["cmcontinue"]}
        json.dump(titles, open(path, "w"), indent=0)
    return json.load(open(path))


def ptb() -> dict[str, list[list[str]]]:
    base = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
    out = {}
    for split in ("train", "valid", "test"):
        p = fetch(base + f"ptb.{split}.txt", f"ptb.{split}.txt")
        out[split] = [ln.split() for ln in open(p) if ln.strip()]
    return out


SOURCES = {
    "simple_sentences": dict(fn=simple_sentences, origin="synthetic grammar (this work)",
                             license="released with this paper", n=None, min_len=3, max_len=30),
    "quantum_phrases": dict(fn=quantum_phrases, origin="synthetic grammar + 10 hand-written phrases (this work)",
                            license="released with this paper", n=None, min_len=3, max_len=30),
    "short_stories": dict(fn=short_stories, origin="TinyStories validation split, first sentence of each story",
                          license="CDLA-Sharing-1.0", n=10000, min_len=4, max_len=30),
    "haiku": dict(fn=haiku, origin="statworx/haiku (Hugging Face) without the 'twaiku' Twitter subset; "
                                   "three-line haiku, lines joined by '/'",
                  license="unspecified on source; text not redistributed, rebuild with this script",
                  n=10000, min_len=5, max_len=30),
    "proverbs": dict(fn=proverbs, origin="Wiktionary Category:English proverbs (page titles)",
                     license="CC BY-SA 4.0", n=None, min_len=3, max_len=30),
}


def split_unique(token_lists, n, seed):
    seen, uniq = set(), []
    for toks in token_lists:
        key = " ".join(toks)
        if key not in seen:
            seen.add(key)
            uniq.append(toks)
    rng = random.Random(seed)
    rng.shuffle(uniq)
    if n is not None:
        uniq = uniq[:n]
    n_tot = len(uniq)
    n_va = n_te = max(1, round(0.1 * n_tot))
    return {"train": uniq[: n_tot - n_va - n_te], "valid": uniq[n_tot - n_va - n_te: n_tot - n_te],
            "test": uniq[n_tot - n_te:]}, len(seen)


def write(name, splits, meta):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    for s, rows in splits.items():
        with open(os.path.join(d, f"{s}.txt"), "w") as f:
            f.writelines(" ".join(t) + "\n" for t in rows)
    train_set = {" ".join(t) for t in splits["train"]}
    meta["test_in_train"] = sum(" ".join(t) in train_set for t in splits["test"])
    meta["sizes"] = {s: len(r) for s, r in splits.items()}
    meta["avg_len"] = round(sum(map(len, splits["train"])) / len(splits["train"]), 2)
    meta["max_len"] = max(len(t) for r in splits.values() for t in r)
    meta["train_types"] = len({w for t in splits["train"] for w in t})
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    print(f"{name:17s} {meta['sizes']} avg_len={meta['avg_len']} types={meta['train_types']} "
          f"test_in_train={meta['test_in_train']}")


def main(names=None):
    for name, cfg in SOURCES.items():
        if names and name not in names:
            continue
        toks = [tokenize(t) for t in cfg["fn"]()]
        toks = [t for t in toks if cfg["min_len"] <= len(t) <= cfg["max_len"]]
        splits, n_unique = split_unique(toks, cfg["n"], SEED)
        write(name, splits, {"origin": cfg["origin"], "license": cfg["license"], "unique_after_filter": n_unique,
                             "tokenisation": TOKEN_RE.pattern, "seed": SEED})
    if not names or "ptb" in names:
        sp = ptb()
        write("ptb", sp, {"origin": "Penn Treebank (Mikolov et al. 2010 pre-processing), standard splits",
                          "license": "LDC; not redistributed, rebuild with this script", "seed": None,
                          "tokenisation": "as distributed (whitespace)"})


if __name__ == "__main__":
    main(sys.argv[1:] or None)
