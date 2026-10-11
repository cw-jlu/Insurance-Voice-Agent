"""Load the cleaned competition data (rates, coverage table, clause text)."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from agent.source_locations import pdf_locations

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "clean"

PRODUCT = "尊享e生·百万医疗保险2026版"


@dataclass(frozen=True)
class RateTable:
    """One premium CSV: age bands × columns. ``None`` means the cell is '-' or blank."""

    name: str  # e.g. "年缴_必选"
    columns: list[str]
    bands: list[tuple[int, int]]
    cells: dict[tuple[int, int], dict[str, float | None]]

    def band_for(self, age: int) -> tuple[int, int] | None:
        for lo, hi in self.bands:
            if lo <= age <= hi:
                return (lo, hi)
        return None

    def price(self, age: int, column: str) -> tuple[float | None, tuple[int, int] | None]:
        band = self.band_for(age)
        if band is None:
            return None, None
        return self.cells[band][column], band


def _parse_cell(raw: str) -> float | None:
    raw = raw.strip()
    if raw in ("", "-"):
        return None
    return float(raw)


@lru_cache(maxsize=None)
def rate_table(name: str) -> RateTable:
    path = CLEAN / "premium" / f"{name}.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    columns = header[1:]
    bands: list[tuple[int, int]] = []
    cells: dict[tuple[int, int], dict[str, float | None]] = {}
    for row in body:
        lo, hi = (int(x) for x in row[0].strip("[]").split(","))
        bands.append((lo, hi))
        # Short rows mean trailing blanks (e.g. 药省保 71+ in the annual table).
        padded = row[1:] + [""] * (len(columns) - len(row[1:]))
        cells[(lo, hi)] = {c: _parse_cell(v) for c, v in zip(columns, padded)}
    return RateTable(name=name, columns=columns, bands=bands, cells=cells)


@lru_cache(maxsize=1)
def coverage_items() -> list[dict]:
    data = json.loads((CLEAN / "保险责任.json").read_text(encoding="utf-8"))
    return data["items"]


@dataclass
class Chunk:
    """A retrievable piece of text with its provenance."""

    chunk_id: str
    doc: str
    text: str
    source: str  # human-readable citation, e.g. "保障计划表 D8" or "投保须知 第2页"
    meta: dict = field(default_factory=dict)


_PAGE_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->")


def _text_chunks(filename: str, doc_label: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    page = 0
    locations = pdf_locations(filename)
    for md_line, line in enumerate((CLEAN / "text" / filename).read_text(encoding="utf-8").splitlines(), 1):
        m = _PAGE_RE.search(line)
        if m:
            page = int(m.group(1))
            continue
        line = line.strip()
        if not line or line.startswith("<!--"):
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc_label}#{len(chunks)}",
                doc=doc_label,
                text=line,
                source=(f"{doc_label} 第{page}页 提取文本行"
                        f"{locations[md_line]['page_text_line_start']}–{locations[md_line]['page_text_line_end']}"
                        if md_line in locations else f"{doc_label} 第{page}页") + f"（MD行{md_line}）",
                meta={"page": page, "md_line":md_line, "location":locations.get(md_line,{})},
            )
        )
    return chunks


@lru_cache(maxsize=1)
def clause_chunks() -> list[Chunk]:
    """Corpus for retrieval: coverage-table cells + 投保须知 + 责任免除 lines."""
    chunks: list[Chunk] = []
    for item in coverage_items():
        name = item["责任名称"]
        for cell in item["责任描述"]:
            chunks.append(
                Chunk(
                    chunk_id=f"{item['id']}:{cell['cell']}",
                    doc="保障计划表",
                    text=f"{name}：{cell['text']}",
                    source=f"保障计划表 {cell['cell']}",
                    meta={"item": item["id"], "cell": cell["cell"]},
                )
            )
    chunks += _text_chunks("尊享E生百万医疗2026投保须知.md", "投保须知")
    chunks += _text_chunks("尊享E生百万医疗2026责任免除.md", "责任免除")
    return chunks
