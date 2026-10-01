"""Interruption handling: playback ledger, barge-in classification, echo filter, cancellable slow lane."""

from __future__ import annotations

import asyncio
import difflib
import re
import time
from dataclasses import dataclass, field
from enum import Enum


class SentState(str, Enum):
    QUEUED = "queued"
    PLAYING = "playing"
    PLAYED = "played"
    DROPPED = "dropped"


@dataclass
class Sentence:
    text: str
    state: SentState = SentState.QUEUED


@dataclass
class PlaybackLedger:
    """Tracks exactly what the user has heard, so history after a barge-in is truthful."""

    sentences: list[Sentence] = field(default_factory=list)

    def add(self, text: str) -> Sentence:
        s = Sentence(text)
        self.sentences.append(s)
        return s

    def interrupt(self) -> tuple[list[str], list[str]]:
        """Returns (heard, dropped). The sentence being played counts as heard: the user
        heard at least part of it, and treating it as unsaid would make the agent repeat it."""
        heard, dropped = [], []
        for s in self.sentences:
            if s.state in (SentState.PLAYED, SentState.PLAYING):
                s.state = SentState.PLAYED
                heard.append(s.text)
            elif s.state == SentState.QUEUED:
                s.state = SentState.DROPPED
                dropped.append(s.text)
        return heard, dropped


# --- barge-in classification ---------------------------------------------------------------------

_BC_TOKEN = r"(嗯+|哦+|噢+|啊+|对+的?|是+的?|好+的?|好吧|行|可以|明白|知道了|了解|收到|没错|ok|OK|这样啊)"
_BACKCHANNEL = re.compile(rf"^{_BC_TOKEN}([。，！!,.、\s]*{_BC_TOKEN})*[。，！!,.\s]*$")
_STOP = re.compile(r"等一下|等等|停一下|先别说|不对|不是|错了|说错了|我没有|我没|打断一下|慢点|稍等|你听我说")


def classify_barge_in(text: str) -> str:
    """'backchannel' (keep talking), 'interrupt' (stop and handle), or 'unclear'."""
    t = text.strip()
    if not t:
        return "backchannel"
    if _BACKCHANNEL.match(t):
        return "backchannel"
    if _STOP.search(t):
        return "interrupt"
    # Anything substantive (a question, new info) is an interrupt; very short fragments are unclear.
    core = re.sub(r"[^\w一-鿿]", "", t)
    return "interrupt" if len(core) >= 3 else "unclear"


def echo_similarity(asr_text: str, playing_text: str) -> float:
    a = re.sub(r"[^\w一-鿿]", "", asr_text)
    b = re.sub(r"[^\w一-鿿]", "", playing_text)
    if not a or not b:
        return 0.0
    # ASR of an echo is usually a fragment of what is playing: score the fragment against the
    # best-matching window of the playing sentence.
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(a)


# A correction repeats the agent's words plus a negation ("我没有社保" vs "有社保的话……").
# Such text must never be dropped as echo, however similar it is.
_CORRECTION = re.compile(r"没有|没|不是|不对|不要|错了|不用|别")


def is_echo(asr_text: str, playing_text: str, threshold: float = 0.6) -> bool:
    for m in _CORRECTION.finditer(asr_text):
        if m.group(0) not in playing_text:
            return False
    return echo_similarity(asr_text, playing_text) >= threshold


# --- cancellable slow lane -----------------------------------------------------------------------

@dataclass
class TurnLog:
    emitted: list[str] = field(default_factory=list)
    cancelled_at: float | None = None
    stale_after_cancel: int = 0


async def slow_lane(sentences: list[str], per_sentence_s: float, out: asyncio.Queue, log: TurnLog, turn_id: int,
                    current_turn: list[int]) -> None:
    """Simulated streaming generation: emits one sentence every ``per_sentence_s``."""
    for s in sentences:
        await asyncio.sleep(per_sentence_s)
        if current_turn[0] != turn_id:
            # Should never happen if cancellation works; counted as a leak.
            log.stale_after_cancel += 1
        await out.put((turn_id, s))
        log.emitted.append(s)


async def player(inp: asyncio.Queue, ledger: PlaybackLedger, play_s_per_char: float, current_turn: list[int],
                 stale: list[str]) -> None:
    while True:
        turn_id, text = await inp.get()
        if turn_id != current_turn[0]:
            stale.append(text)  # stale sentence from a superseded turn reached the player
            continue
        s = ledger.add(text)
        s.state = SentState.PLAYING
        try:
            await asyncio.sleep(len(text) * play_s_per_char)
        except asyncio.CancelledError:
            raise
        if s.state == SentState.PLAYING:
            s.state = SentState.PLAYED


def now() -> float:
    return time.perf_counter()
