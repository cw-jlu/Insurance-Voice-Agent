"""Embed the knowledge chunks once and cache them (experiments/cache/, git-ignored)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import cloud  # noqa: E402
from agent.knowledge import Chunk, clause_chunks  # noqa: E402
from agent.pipeline import rate_chunks  # noqa: E402

CACHE = ROOT / "experiments" / "cache"


def load(chunks: list[Chunk], name: str) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((cloud.EMBED_MODEL + "\n".join(c.text for c in chunks)).encode()).hexdigest()[:16]
    path = CACHE / f"{name}_{key}.npy"
    if path.exists():
        return np.load(path)
    vecs = np.array(cloud.embed([c.text[:2000] for c in chunks]), dtype=np.float32)
    np.save(path, vecs)
    return vecs


def corpus(with_rates: bool = False) -> list[Chunk]:
    return clause_chunks() + (rate_chunks() if with_rates else [])


if __name__ == "__main__":
    for wr in (False, True):
        c = corpus(wr)
        v = load(c, "with_rates" if wr else "clauses")
        print(len(c), v.shape)
    print(json.dumps({"model": cloud.EMBED_MODEL}))
