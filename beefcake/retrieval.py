"""Retrieval over the BeefCake docs.

The docs are split into chunks at each '## ' heading, and chunks are ranked with
BM25, a classic keyword-matching score. BM25 needs no API key and gives the same
results every time, which keeps the workshop exercises reproducible. It also
makes the kind of mistakes real retrieval makes: it matches words, not meaning.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

STOPWORDS = set(
    """a an and are as at be but by can do does for from how i if in is it its
    me my of on or so that the this to was what when where which why will with
    you your t s""".split()
)


@dataclass
class Chunk:
    id: str          # for example "pulse_manual#signal-drops-or-keeps-disconnecting"
    doc: str         # document title, for example "BeefCake Pulse manual"
    heading: str     # section heading
    text: str        # section body

    def as_context(self) -> str:
        return f"[{self.doc} > {self.heading}]\n{self.text}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        lines = path.read_text().splitlines()
        doc_title = lines[0].lstrip("# ").strip()
        heading, body = None, []
        for line in lines[1:] + ["## END"]:
            if line.startswith("## "):
                if heading and "".join(body).strip():
                    chunks.append(Chunk(f"{path.stem}#{_slug(heading)}", doc_title, heading, "\n".join(body).strip()))
                heading, body = line[3:].strip(), []
            else:
                body.append(line)
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.k1, self.b = k1, b
        self.docs = [tokenize(f"{c.doc} {c.heading} {c.text}") for c in self.chunks]
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        df = Counter(term for d in self.docs for term in set(d))
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def score(self, query: str) -> list[float]:
        terms = tokenize(query)
        scores = []
        for tf, doc in zip(self.tf, self.docs):
            s = 0.0
            for t in terms:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            scores.append(s)
        return scores

    def retrieve(self, query: str, k: int = 3) -> list[Chunk]:
        scores = self.score(query)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k]]


_default: BM25Retriever | None = None


def retrieve(query: str, k: int = 3) -> list[Chunk]:
    """Return the top-k chunks for a query using the default BM25 index."""
    global _default
    if _default is None:
        _default = BM25Retriever()
    return _default.retrieve(query, k)


def get_chunk(chunk_id: str) -> Chunk:
    for c in load_chunks():
        if c.id == chunk_id:
            return c
    raise KeyError(chunk_id)
