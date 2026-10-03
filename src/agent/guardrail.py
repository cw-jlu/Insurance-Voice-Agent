"""Output guardrails applied to every answer before it is spoken.

1. Number guard: every number in the answer must appear in this turn's evidence, with a
   compatible unit, and in a sentence that talks about the same thing.
2. Gap guard: for questions the data cannot answer, block conclusive statements.
3. Compliance guard: forbidden sales phrases (negated forms such as "不能保证续保" are allowed).
4. AI-identity rule: detect "are you a human?" questions; the reply is fixed, not generated.

These are rule-based fast checks. Experiment 3 shows they miss unseen phrasings, so in the
full system they are paired with an LLM judge for gap/compliance checks (see 02 方案 3.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}


def cn_to_int(s: str) -> int | None:
    """'三十' -> 30, '三百一十八' -> 318, '十五' -> 15. Returns None if not a Chinese numeral."""
    if not s or any(ch not in _CN_DIGIT and ch not in _CN_UNIT for ch in s):
        return None
    total, num = 0, 0
    for ch in s:
        if ch in _CN_DIGIT:
            num = _CN_DIGIT[ch]
        else:
            total += (num or 1) * _CN_UNIT[ch]
            num = 0
    return total + num


_NUM_RE = re.compile(r"(\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千]+)\s*(万)?\s*(%|元|天|周岁|岁|日|人|种)?")
# List markers in source text ("1）", "a.", "ii.") are not facts and must not whitelist numbers.
_LIST_MARKER = re.compile(r"(?:(?<=^)|(?<=[\s；;。:：]))\s*(?:\d{1,2}[）)、.]|[a-z][.、]|[ivx]+[.、])")


@dataclass(frozen=True)
class Num:
    value: float
    unit: str  # "", "%", "元", "天", "岁", "人", "种"
    raw: str
    start: int


def extract_numbers(text: str) -> list[Num]:
    out: list[Num] = []
    for m in _NUM_RE.finditer(text):
        token, wan, unit = m.group(1), m.group(2), m.group(3) or ""
        if token[0].isdigit():
            value = float(token)
        else:
            iv = cn_to_int(token)
            # A lone Chinese "一" etc. without unit is almost always a word, not a number; a bare
            # "百万"/"千万" is a phrase ("百万医疗险"), not an amount.
            if iv is None or (len(token) == 1 and not (wan or unit)) or token in ("百", "千", "十"):
                continue
            value = float(iv)
        if wan:
            value *= 10_000
        unit = {"周岁": "岁", "日": "天"}.get(unit, unit)
        out.append(Num(value=value, unit=unit, raw=m.group(0), start=m.start()))
    return out


@dataclass
class GuardResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


_SENTENCE_END = re.compile(r"[。；;！？!?\n]")
_STOP_BIGRAMS = {"保险", "险金", "元为", "为限", "给付", "本项", "按照", "比例", "的赔", "赔付", "付比", "以上", "及以", "及二",
                 "二级", "级及", "保额", "万元", "限额", "累计", "年累", "计给", "付以"}
PRODUCT_NAME_RE = re.compile(r"(尊享\s*[eE]\s*生[·・.\s]*)?(百万医疗(保险|险)?)?\s*2026\s*版?")
CONTEXT_BEFORE = 24
CONTEXT_AFTER = 8

# Customer wording → clause wording, so "一般住院" in an answer can match "一般医疗" in a clause.
ALIASES = {
    "住院": "医疗", "重疾": "重大疾病", "大病": "重大疾病", "特药": "特定药品", "门诊": "门急诊", "癌症": "恶性肿瘤",
    "医保": "基本医疗保险", "社保": "基本医疗保险", "护工": "护工费用", "津贴": "住院津贴", "康复": "康复住院",
    "买药": "药品费用", "交通": "公共交通", "住宿": "住宿费用",
}


def _bigrams(s: str) -> set[str]:
    s = re.sub(r"[^一-鿿]", "", s)
    return {s[i : i + 2] for i in range(len(s) - 1)} - _STOP_BIGRAMS


def _context(text: str, start: int, end: int) -> str:
    """Words around a number, within the same sentence, with customer wording expanded."""
    s_start = max((m.end() for m in _SENTENCE_END.finditer(text, 0, start)), default=0)
    m_end = _SENTENCE_END.search(text, end)
    s_end = m_end.start() if m_end else len(text)
    # Chinese often states the topic first ("门急诊加油包……一年最多1万"): keep the sentence head too.
    head = text[s_start : min(s_start + 8, start)]
    ctx = head + " " + text[max(s_start, start - CONTEXT_BEFORE) : start] + " " + text[end : min(s_end, end + CONTEXT_AFTER)]
    return ctx + " " + " ".join(v for k, v in ALIASES.items() if k in ctx)


def _units_compatible(a: str, b: str) -> bool:
    return not a or not b or a == b


def number_guard(answer: str, evidence: list[str], structured: set[float] | None = None) -> GuardResult:
    """Every number in ``answer`` must be supported.

    ``structured``: values from deterministic lookups (quote engine, user-stated slots). They are
    trusted by value, because their provenance is a table cell, not prose.
    ``evidence``: retrieved text chunks (each should carry its item name, as Chunk.text does). A
    number is supported if some chunk has the same value with a compatible unit and, when the
    answer names what the number is about, that chunk talks about the same thing.
    """
    ev: list[tuple[Num, set[str]]] = []
    for e in evidence:
        clean = _LIST_MARKER.sub(" ", e)
        terms = _bigrams(clean)
        ev += [(n, terms) for n in extract_numbers(clean)]
    answer = PRODUCT_NAME_RE.sub("本产品", answer)  # "2026版" is a name, not a fact to verify
    bad = []
    for n in extract_numbers(answer):
        if re.fullmatch(r"0\d+", re.sub(r"[^\d]", "", n.raw)):
            bad.append(n.raw)  # "05", "003": never a valid spoken amount
            continue
        if structured and n.value in structured:
            continue
        same_value = [terms for e, terms in ev if e.value == n.value and _units_compatible(e.unit, n.unit)]
        if not same_value:
            bad.append(n.raw)
            continue
        ctx = _bigrams(_context(answer, n.start, n.start + len(n.raw)))
        if ctx and not any(ctx & terms for terms in same_value):
            bad.append(n.raw)
    return GuardResult(ok=not bad, violations=[f"无依据数字：{b}" for b in bad])


# --- hedge handling shared by gap and compliance guards --------------------------------------

_HEDGED = re.compile(
    r"能不能[^，。；]*|能否[^，。；]*|是否[^，。；]*|如果[^，。；]*|"
    r"(不能|不会|不|没法|无法|并不|不敢)(保证|承诺|确定|判断|替)[^，。；]*|"
    r"(以|要看|看)[^，。；]{0,12}(为准|结果|结论|约定)"
)


def _unhedged(text: str) -> str:
    return _HEDGED.sub("", text)


# --- gap guard ----------------------------------------------------------------------------------

# The customer states a condition of their own ("我有…", "得过…", "做过…手术", "一直吃…药").
_HEALTH = (r"((我|俺|他|她|我妈|我爸|老人|孩子)?(有|得过|患有|查出|确诊)(过)?.{0,6}(病|症|炎|瘤|癌|结节|囊肿|息肉|高|三高|乙肝)|"
           r"(血压|血糖|血脂)(有点)?(高|偏高)|高血压|糖尿病|痛风|乙肝|甲状腺|乳腺|心脏|支架|哮喘|抑郁|焦虑|"
           r"做过.{0,6}手术|吃.{0,4}药|住过院)")
_BUY_OR_CLAIM = r"(买|投|保|报|赔|承保|核保)"


@dataclass(frozen=True)
class Gap:
    name: str
    question: re.Pattern
    conclusive: re.Pattern
    fallback: str


GAPS = [
    Gap(
        "疾病核保",
        re.compile(_HEALTH + r".{0,20}" + _BUY_OR_CLAIM + r"|能不能(买|投保)"),
        re.compile(
            r"(可以|能|也能|都能|肯定能|一般能|应该能|就能|照样能)(买|投|保|承保|通过|过|报)|没问题|不影响|问题不大|没关系|"
            r"控制得好|核保(会|能|一般)?(通过|能过)|标准体承保|放心"
        ),
        "能不能投保要先做健康告知，以核保结果为准，我这边没法替核保下结论。",
    ),
    Gap(
        "续保",
        re.compile(r"续保|还能续|怎么续|能续|一直(续|买|保)|停售|明年还能|每年都能|到期.{0,4}(续|买)"),
        re.compile(r"(保证|承诺|肯定|一定|可以一直|能一直|每年都能|随时都能|也能|照样)(续|保)|续保(没问题|是保证的)|不影响续保|不会停售|没问题|放心"),
        "续保以保险条款约定为准，我可以帮您转专员确认。",
    ),
    Gap(
        "社保抵扣免赔额",
        re.compile(r"(医保|社保).{0,15}(免赔|一万|1万)|(免赔|一万|1万).{0,15}(医保|社保)|(医保|社保)报(完|了|过).{0,12}(还|再|能|怎么)"),
        re.compile(r"(可以|能|也能|都能|会|也)(抵|算进|算在|计入|冲抵)|算(在|进)(免赔|一万|1万)|可以抵扣|能抵扣|"
                   r"(医保|社保)报(完|了)?.{0,8}剩下的.{0,12}(超过|满|够).{0,4}(1万|一万)|剩下的.{0,6}(扣|超过)(掉)?.{0,3}(1万|一万)"),
        "医保报销的部分能不能抵扣免赔额，要以条款为准，我帮您转专员确认。",
    ),
]


def gap_of(question: str) -> Gap | None:
    for g in GAPS:
        if g.question.search(question):
            return g
    return None


def gap_guard(question: str, answer: str, names: list[str] | None = None) -> GuardResult:
    """``names``: gap categories already identified by the understanding model; combined with
    the rule-based detection so that either layer can trigger the check."""
    plain = _unhedged(answer)
    hits = []
    for g in GAPS:
        if g.question.search(question) or (names and g.name in names):
            m = g.conclusive.search(plain)
            if m:
                hits.append(f"缺口问题给了结论（{g.name}）：{m.group(0)}")
    return GuardResult(ok=not hits, violations=hits)


_SOCIAL_DEDUCTION = re.compile(
    r"(医保|社保)(报|报销)(完|过|了)[^。！？；]{0,16}(扣|超过|满|够|减)(掉|去)?[^。！？；]{0,4}(1万|一万|免赔)|"
    r"剩下的[^。！？；]{0,6}(扣|超过|满|够)(掉|去)?[^。！？；]{0,3}(1万|一万)|"
    r"(医保|社保)报(完|过|了)[^。！？；]{0,12}[，,]?[^。！？；]{0,4}(1万|一万)(元)?(的)?免赔(额)?(后|以后|之后|扣)|"
    r"(医保|社保)(报销)?(的|那)?(部分|钱)[^。！？；]{0,6}(能|可以|也)?(算进|算在|计入|抵扣|冲抵|抵)")


def social_deduction_conclusion(answer: str) -> GuardResult:
    """Always-on: whether social-insurance reimbursement counts towards the 1万 deductible is not in the data."""
    m = _SOCIAL_DEDUCTION.search(_unhedged(answer))
    return GuardResult(ok=not m, violations=[f"缺口问题给了结论（社保抵扣免赔额）：{m.group(0)}"] if m else [])


# --- compliance guard ----------------------------------------------------------------------------

FORBIDDEN = re.compile(
    r"什么(病|都)都?能(报|赔|保)|什么都(保|报|赔)|(大病小病|小病大病)都能(报|赔)|全都能报|全额报销|随便报|"
    r"保证续保|续保(是)?保证的|续保肯定没问题|肯定能赔|一定能赔|百分百(赔|报)|"
    r"高血压没关系|没有任何限制|不用健康告知|不用如实告知|不告知也没事"
)
_NEGATED_SPAN = re.compile(r"(不是|不能|不会|没法|无法|并不|不承诺|不保证)[^，。；]{0,10}")


def compliance_guard(answer: str) -> GuardResult:
    plain = _NEGATED_SPAN.sub("", answer)
    hits = [m.group(0) for m in FORBIDDEN.finditer(plain)]
    return GuardResult(ok=not hits, violations=[f"禁用表述：{h}" for h in hits])


# --- AI identity ---------------------------------------------------------------------------------

_ID_TERM = r"(真人|真的人|活人|假人|人工|机器人|机器|AI|ai|人工智能|智能语音|智能客服|AI客服|自动语音|语音助手|录音|程序|电脑合成|合成)"
_ID_TARGET = r"(你|您|你们|对面|这|刚才|说话|打电话|打的电话|聊天|声音|念稿)"
_ID_ASK = r"(吗|吧|么|嘛|还是|是不是|该不会|不会是|在跟|像|是人|人吗|啊)"
_IDENTITY = re.compile(_ID_TARGET + r".{0,12}" + _ID_TERM + r"|" + _ID_TERM + r".{0,8}" + _ID_TARGET)
_IDENTITY_ASK = re.compile(_ID_ASK)
_NOT_IDENTITY = re.compile(
    r"转人工|找人工|要人工|找真人|要真人|人工(服务|坐席|客服)(怎么|在哪|电话|什么时候)|真人客服什么时候|"
    r"(AI|ai)(能|可以)帮|(机器人|AI|ai)(手术|辅助|问诊)|真人能帮|"
    r"(是|叫)(真|对|没错)的吗|保险是真的"
)

IDENTITY_REPLY = "我是众安保险的AI助理。"


_IDENTITY_ALT = re.compile(
    r"是人还是|(说话|打电话|对面)的?是(人|谁)|(在|是)?放(的)?录音|有没有真人|听着像(机器|AI|ai|机器人)"
)


def is_identity_question(utterance: str) -> bool:
    if _NOT_IDENTITY.search(utterance):
        return False
    if _IDENTITY_ALT.search(utterance):
        return True
    return bool(_IDENTITY.search(utterance) and _IDENTITY_ASK.search(utterance))


def admits_ai(reply: str) -> bool:
    return bool(re.search(r"(我是|我是一个|我是一名).{0,8}(AI|ai|人工智能|智能助理|机器人)", reply)) and not re.search(
        r"我是真人|我不是(AI|机器人)", reply
    )
