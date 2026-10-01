"""Sentence splitting for streaming TTS, and slot extraction from (ASR) user text."""

from __future__ import annotations

import re

from agent.guardrail import cn_to_int

_TERMINATORS = "。！？!?；;\n"
_SOFT_CUT = "，,、："
SOFT_FLUSH_CHARS = 60


class SentenceFeeder:
    """Feed LLM text deltas; yields speakable pieces as soon as a sentence (or a long clause) ends."""

    def __init__(self, soft_flush: int | None = SOFT_FLUSH_CHARS):
        self.buf = ""
        self.soft_flush = soft_flush

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while True:
            idx = next((i for i, ch in enumerate(self.buf) if ch in _TERMINATORS), -1)
            if idx >= 0:
                piece, self.buf = self.buf[: idx + 1], self.buf[idx + 1 :]
                if piece.strip():
                    out.append(piece.strip())
                continue
            if self.soft_flush and len(self.buf) > self.soft_flush:
                cut = max(self.buf.rfind(c) for c in _SOFT_CUT)
                if cut > 0:
                    piece, self.buf = self.buf[: cut + 1], self.buf[cut + 1 :]
                    out.append(piece.strip())
                    continue
            break
        return out

    def flush(self) -> list[str]:
        piece, self.buf = self.buf.strip(), ""
        return [piece] if piece else []


# --- slot extraction ----------------------------------------------------------------------------

_AGE_RE = re.compile(r"(\d{1,3}|[一二两三四五六七八九十]{1,4})\s*(周岁|岁)")
_SOCIAL_NO = re.compile(r"(没有|没|无|不交|没交|没买|没上|断了)(社保|医保)|(社保|医保)(没有|没交|断了|停了)")
_SOCIAL_YES = re.compile(r"(有|交了|在交|一直交|有交)(社保|医保)|(社保|医保)(有|交着|在交|一直交)")
_PAY_MONTH = re.compile(r"按月|月缴|每个?月(交|付)|月付")
_PAY_YEAR = re.compile(r"按年|年缴|一年交一次|一次性交|年付")
_RELATION = [
    ("父母", re.compile(r"(我|给)?(爸|妈|父亲|母亲|爸妈|父母|老人)")),
    ("子女", re.compile(r"(孩子|小孩|儿子|女儿|宝宝|闺女)")),
    ("配偶", re.compile(r"(老婆|老公|爱人|媳妇|丈夫|妻子|对象)")),
    ("本人", re.compile(r"(我自己|给自己|本人|我)")),
]
_GENDER = [("女", re.compile(r"女|妈|母亲|老婆|妻子|媳妇|女儿|闺女")), ("男", re.compile(r"男|爸|父亲|老公|丈夫|儿子"))]


def extract_slots(text: str) -> dict:
    slots: dict = {}
    m = _AGE_RE.search(text)
    if m:
        tok = m.group(1)
        slots["age"] = int(tok) if tok.isdigit() else cn_to_int(tok)
    if _SOCIAL_NO.search(text):
        slots["has_social"] = False
    elif _SOCIAL_YES.search(text):
        slots["has_social"] = True
    if _PAY_MONTH.search(text):
        slots["pay"] = "月缴"
    elif _PAY_YEAR.search(text):
        slots["pay"] = "年缴"
    for rel, pat in _RELATION:
        if pat.search(text):
            slots["relation"] = rel
            break
    for g, pat in _GENDER:
        if pat.search(text):
            slots["gender"] = g
            break
    return slots
