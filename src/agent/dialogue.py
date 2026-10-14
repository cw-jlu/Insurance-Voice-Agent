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
    heard_prefix: str = ''


@dataclass
class PlaybackLedger:
    """Tracks exactly what the user has heard, so history after a barge-in is truthful."""

    sentences: list[Sentence] = field(default_factory=list)

    def add(self, text: str) -> Sentence:
        s = Sentence(text)
        self.sentences.append(s)
        return s

    def interrupt(self) -> tuple[list[str], list[str]]:
        """A playing sentence is NOT fully heard; keep only a confirmed aligned prefix."""
        heard, dropped = [], []
        for s in self.sentences:
            if s.state == SentState.PLAYED:
                heard.append(s.text)
            elif s.state == SentState.PLAYING:
                if s.heard_prefix:
                    if not s.text.startswith(s.heard_prefix):
                        raise ValueError('Heard text must be a prefix of the sentence')
                    heard.append(s.heard_prefix)
                if len(s.heard_prefix) < len(s.text):
                    dropped.append(s.text[len(s.heard_prefix):])
                s.state = SentState.DROPPED
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


# --- barge-in controller (adapted from LockedIn AudioPlayer + speakAloudBus) ---------------------
#
# LockedIn (a copilot) stops TTS on ANY user transcript, and against self-echo it either hard-mutes
# speech capture while TTS plays (L2) or drops transcripts that mostly repeat recently spoken text
# within a time window (L1). A full-duplex phone agent cannot mute capture (that would make barge-in
# impossible) and must not stop on backchannels, so the flow here is:
#   1. VAD speech onset while speaking   -> duck (lower volume), keep listening
#   2. first partial transcript          -> echo check (content overlap with what is playing / just
#                                           played, LockedIn L1 style) with correction/question exemptions
#   3. not echo                          -> classify: rule fast path, small model for the rest
#   4. interrupt -> stop playback, cancel the slow lane, ledger.interrupt(); backchannel -> restore volume

QUESTION_MARKERS = re.compile(r"吗|呢|多少|多久|几|什么|怎么|哪|能不能|是不是|可不可以|？|\?")


@dataclass
class BargeDecision:
    action: str  # "duck", "resume", "stop", "ignore_echo"
    reason: str
    t: float


class BargeInController:
    def __init__(self, classify, echo_threshold: float = 0.6, echo_window_s: float = 3.0):
        self.classify = classify  # text -> ("附和"|"打断", path)
        self.echo_threshold = echo_threshold
        self.echo_window_s = echo_window_s
        self.recent_spoken: list[tuple[float, str]] = []  # (t, sentence) — LockedIn's "recent" memory
        self.ducked = False

    def note_spoken(self, t: float, sentence: str) -> None:
        self.recent_spoken.append((t, sentence))

    def on_vad_onset(self, t: float, speaking: bool) -> BargeDecision | None:
        if speaking and not self.ducked:
            self.ducked = True
            return BargeDecision("duck", "检测到用户开口", t)
        return None

    def _is_echo(self, text: str, t: float) -> bool:
        if QUESTION_MARKERS.search(text):
            return False  # a question is never treated as echo (see 实验 5 第三轮)
        return any(is_echo(text, s, self.echo_threshold) for ts, s in self.recent_spoken if t - ts <= self.echo_window_s)

    def on_partial(self, t: float, text: str, speaking: bool) -> BargeDecision | None:
        if not speaking or not text.strip():
            return None
        if self._is_echo(text, t):
            return BargeDecision("ignore_echo", "与刚播报内容高度重合", t)
        label, path = self.classify(text)
        self.ducked = False
        return BargeDecision("stop" if label == "打断" else "resume", f"{label}（{path}）", t)
