"""Evidence builders shared by experiment 3 sets: same text format as retrieval chunks."""

from agent.guardrail import extract_numbers
from agent.knowledge import coverage_items
from agent.quote import QuoteResult

COV = {i["id"]: i for i in coverage_items()}


def cov(*ids: str) -> list[str]:
    out = []
    for i in ids:
        it = COV[i]
        name = it["责任名称"]
        out.append(f"{name}：保额{it['保额']}；免赔{it['免赔额/免赔天数']}；等待期{it['等待期']}；赔付{'，'.join(it['赔付比例'])}")
        out += [f"{name}：{c['text']}" for c in it["责任描述"]]
    return out


def split(evidence: list) -> tuple[list[str], set[float]]:
    """Text chunks go to context matching; QuoteResults and user-stated facts are structured."""
    texts, structured = [], set()
    for e in evidence:
        if isinstance(e, QuoteResult):
            structured |= e.numbers()
        elif isinstance(e, str) and e.startswith("用户："):
            structured |= {n.value for n in extract_numbers(e)}
        else:
            texts.append(e)
    return texts, structured
