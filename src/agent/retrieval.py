"""Clause retrieval: BM25 over character bigrams + TF-IDF (char n-gram) cosine, fused with RRF.

No word segmenter is needed for Chinese: character bigrams handle terms like "免赔额" and
"既往症" well, and the corpus is small (a few hundred chunks).
"""

from __future__ import annotations

import math
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from agent.knowledge import Chunk


def _bigrams(text: str) -> list[str]:
    s = "".join(ch for ch in text if not ch.isspace())
    return [s[i : i + 2] for i in range(len(s) - 1)] or [s]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [_bigrams(d) for d in docs]
        self.tf = [Counter(t) for t in self.toks]
        self.lens = [len(t) for t in self.toks]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(term for t in self.tf for term in t)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = Counter(_bigrams(query))
        out = []
        for tf, ln in zip(self.tf, self.lens):
            s = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.bm25 = BM25(texts)
        self.tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)
        self.mat = self.tfidf.fit_transform(texts)

    def _rank(self, scores) -> list[int]:
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def search(self, query: str, k: int = 5, method: str = "hybrid") -> list[Chunk]:
        bm = self.bm25.scores(query)
        if method == "bm25":
            order = self._rank(bm)
        else:
            tf = linear_kernel(self.tfidf.transform([query]), self.mat).ravel()
            if method == "tfidf":
                order = self._rank(tf)
            else:  # reciprocal rank fusion
                fused = [0.0] * len(self.chunks)
                for ranking in (self._rank(bm), self._rank(tf)):
                    for r, i in enumerate(ranking):
                        fused[i] += 1.0 / (60 + r + 1)
                order = self._rank(fused)
        return [self.chunks[i] for i in order[:k]]
