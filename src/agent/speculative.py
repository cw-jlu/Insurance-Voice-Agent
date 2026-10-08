"""Opt-in speculative turns. Partial transcripts never mutate the live call or emit speech.

Only an exact confirmed transcript may release buffered sentences and commit state.
Cancellation is cooperative at sentence boundaries; in-flight classifiers may finish.
"""
from __future__ import annotations

import copy
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields


class CancelledPreparation(Exception):
    pass


def transcript_key(text):
    from agent.guardrail import cn_to_int
    # Only normalize formatting and unambiguous numerals with units. Preserve decimal dots,
    # negations and other words: 1.5 must never compare equal to 15.
    text = re.sub(r"[零〇一二两三四五六七八九十百千]+(?=周岁|岁|天|元|人|个月)",
                  lambda m: str(cn_to_int(m.group())), text)
    return re.sub(r"[\s，,。！？?!；;：:、]", "", text)


class PreparedTurn:
    def __init__(self, session, text, executor, streaming):
        self.text = text
        self.cancelled = threading.Event()
        self.sentences = queue.Queue()
        self.session = type(session)(**{
            f.name: session.pipe if f.name == "pipe" else copy.deepcopy(getattr(session, f.name))
            for f in fields(session)
        })
        self.future = executor.submit(self._run, streaming)

    def _emit(self, sentence):
        self._check_cancelled()
        self.sentences.put(sentence)

    def _check_cancelled(self):
        if self.cancelled.is_set():
            raise CancelledPreparation()

    def _run(self, streaming):
        if self.cancelled.is_set():
            raise CancelledPreparation()
        result = self.session.turn(self.text, on_sentence=self._emit if streaming else None,
                                   cancel_check=self._check_cancelled)
        if self.cancelled.is_set():
            raise CancelledPreparation()
        if not streaming:
            from agent.session import _sentences
            for sentence in _sentences(result.text):
                self._emit(sentence)
        return result

    def cancel(self):
        self.cancelled.set()
        self.future.cancel()


class SpeculativeTurn:
    """One input turn, with bounded speculative work and final-only speech release."""
    def __init__(self, session, streaming=True, min_chars=6, min_interval_s=0.6, max_preparations=3):
        self.session, self.streaming = session, streaming
        self.min_chars, self.min_interval_s = min_chars, min_interval_s
        self.max_preparations = max_preparations
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.final_executor = ThreadPoolExecutor(max_workers=1)  # stale work must not queue the final turn
        self.candidate = None
        self.last_start = float("-inf")
        self.stats = {"preparations": 0, "invalidations": 0, "reused": False, "failed_preparations": 0}

    def partial(self, text, now=None):
        now = time.monotonic() if now is None else now
        key = transcript_key(text)
        self.invalidate(text)
        if (not key or len(key) < self.min_chars or self.candidate
                or self.stats["preparations"] >= self.max_preparations
                or now - self.last_start < self.min_interval_s):
            return
        self.candidate = PreparedTurn(self.session, text, self.executor, self.streaming)
        self.stats["preparations"] += 1
        self.last_start = now

    def invalidate(self, text):
        key = transcript_key(text)
        if self.candidate and transcript_key(self.candidate.text) != key:
            self.candidate.cancel()
            self.candidate = None
            self.stats["invalidations"] += 1

    def confirm(self, text, on_sentence):
        candidate = self.candidate
        if candidate is not None and candidate.future.done() and not candidate.future.cancelled():
            if candidate.future.exception() is not None:
                candidate.cancel()
                candidate = None
                self.stats["failed_preparations"] += 1
        if (candidate is not None and not candidate.cancelled.is_set()
                and transcript_key(candidate.text) == transcript_key(text)):
            self.stats["reused"] = True
        else:
            if candidate:
                candidate.cancel()
                self.stats["invalidations"] += 1
            candidate = PreparedTurn(self.session, text, self.final_executor, self.streaming)
        try:
            while not candidate.future.done() or not candidate.sentences.empty():
                try:
                    on_sentence(candidate.sentences.get(timeout=0.02))
                except queue.Empty:
                    continue
            result = candidate.future.result()
            for f in fields(self.session):
                if f.name != "pipe":
                    setattr(self.session, f.name, getattr(candidate.session, f.name))
            return result
        except Exception:
            candidate.cancel()
            raise

    def close(self):
        if self.candidate:
            self.candidate.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.final_executor.shutdown(wait=True, cancel_futures=True)
