"""Bounded first response: a small model selects a pre-recorded, fact-free phrase."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from agent import cloud

FALLBACK = "好的，我帮您看一下。"
PHRASES = {
    "价格": "好的，我帮您查一下价格。",
    "保障": "好的，我查一下具体保障。",
    "病史": "这个要结合您的情况，我查一下依据。",
    "身份": "我是AI助理，您可以直接问我。",
    "人工": "明白，我来处理您的要求。",
    "其他": FALLBACK,
}
_PROMPT = """你只需为保险电话中用户刚说的一句话，选择一条简短接话语的类别。
只输出 JSON：{"category":"价格/保障/病史/身份/人工/其他"}。
价格：问多少钱、保费、月缴或年缴；保障：问报销、免赔、等待期、条款；
病史：说自己或家人有病史并问投保或理赔；身份：问你是真人、机器人还是AI；
人工：要求人工、投诉、拒绝继续通话；
其他：无法确定。不要生成保险事实或回答。"""
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="first-response")


def _classify(utterance: str, client=None) -> str:
    raw = cloud.chat([{"role": "system", "content": _PROMPT}, {"role": "user", "content": utterance}],
                     model=cloud.FAST_MODEL, client=client, temperature=0,
                     response_format={"type": "json_object"})
    try:
        return json.loads(raw).get("category", "其他")
    except json.JSONDecodeError:
        return "其他"


def choose(utterance: str, client=None, timeout_s: float = 0.7) -> tuple[str, str, float]:
    """Return (phrase, path, elapsed); never wait past timeout for the model."""
    t0 = time.perf_counter()
    future = _POOL.submit(_classify, utterance, client)
    try:
        category = future.result(timeout=timeout_s)
        if category == "身份" and not any(x in utterance.lower() for x in ("ai", "机器人", "真人", "录音", "人工智能", "机器")):
            category = "其他"
        if category in PHRASES:
            return PHRASES[category], "model", time.perf_counter() - t0
    except Exception:
        future.cancel()
    return FALLBACK, "timeout_or_error", time.perf_counter() - t0
