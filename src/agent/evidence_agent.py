"""Evidence agent: lets the model add tool calls on top of the fixed routing ("叠 step").

The routing already gathered some evidence. The agent sees what is there and may call tools to
add more — several tools per step are allowed (run as one step) — until it says it has enough,
hits ``max_steps``, or runs out of ``budget_s``. It never writes the answer; generation and the
guardrails stay in the pipeline.
"""

from __future__ import annotations

import json
import time

from agent import cloud
from agent.knowledge import Chunk, coverage_items

TOOLS = [
    {"type": "function", "function": {
        "name": "get_coverage", "description": "取某项保险责任在保障计划表里的完整内容（保额、比例、免赔额、等待期、条件）",
        "parameters": {"type": "object", "properties": {"item_id": {"type": "string", "description": "责任编号，如 R01"}},
                       "required": ["item_id"]}}},
    {"type": "function", "function": {
        "name": "get_topic", "description": "取关键主题的固定条款原文",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "search_clauses", "description": "在投保须知和责任免除中检索条款原文，用条款用语写查询",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

_SYS = """你负责为保险电销 AI 收集回答所需的资料，不负责回答。
已有资料列在下面。判断这些资料是否足够回答用户的问题；不够就调用工具补充，可以一次调用多个工具。
足够时直接回复"够了"，不要调用工具。不要收集与问题无关的资料。
责任目录：
%(catalog)s
关键主题：%(topics)s"""


def _summ(chunks: list[Chunk]) -> str:
    return "\n".join(f"- [{c.meta.get('item') or c.doc}] {c.text[:80]}" for c in chunks) or "（无）"


def gather(question: str, slots: dict, history: str, have: list[Chunk], client=None,
           max_steps: int = 3, budget_s: float = 3.0, retriever=None) -> tuple[list[Chunk], int]:
    from agent.pipeline import TOPIC_NAMES, item_chunks, topic_chunks

    client = client or cloud.llm_client()
    catalog = "\n".join(f"{i['id']} {i['责任名称']}（{i['责任&加油包']}）" for i in coverage_items())
    msgs = [{"role": "system", "content": _SYS % {"catalog": catalog, "topics": "、".join(TOPIC_NAMES)}},
            {"role": "user", "content": (f"用户条件：{json.dumps({k: v for k, v in slots.items() if not str(k).startswith('_')}, ensure_ascii=False)}\n"
                                         + (f"最近对话：\n{history}\n" if history else "")
                                         + f"用户问题：{question}\n已有资料：\n{_summ(have)}")}]
    t0 = time.perf_counter()
    got: list[Chunk] = []
    steps = 0
    while steps < max_steps and time.perf_counter() - t0 < budget_s:
        r = client.chat.completions.create(model=cloud.FAST_MODEL, messages=msgs, tools=TOOLS, temperature=0,
                                           extra_body={"enable_thinking": False},
                                           timeout=max(0.5, budget_s - (time.perf_counter() - t0)))
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            break
        steps += 1
        msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [c.model_dump() for c in calls]})
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            res: list[Chunk] = []
            if c.function.name == "get_coverage" and args.get("item_id") in {i["id"] for i in coverage_items()}:
                res = item_chunks(args["item_id"])
            elif c.function.name == "get_topic" and args.get("topic") in TOPIC_NAMES:
                res = topic_chunks(args["topic"])
            elif c.function.name == "search_clauses" and retriever is not None:
                res = retriever.search(args.get("query", question), k=3)
            got += res
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": _summ(res)})
    return got, steps
