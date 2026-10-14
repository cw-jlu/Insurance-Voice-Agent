"""Fresh live ASR interruptions verify actual playback history and next DS context."""
import asyncio
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
from agent import cloud, small_model
from agent.dialogue import BargeInController
from agent.evidence_answer import Evidence, EvidenceAnswerer
from agent.evidence_pipeline import EvidencePipeline
from agent.fast_lane import CachedSpeech
from agent.playback import PlaybackReceipts, TrackedDacPlayer
from agent.session import CallSession, _sentences
from exp30_fewshot import save
from exp36_new_voice_acceptance import user_audio, wav
from exp5_audio import mix

OUT = ROOT / 'experiments/results'
REC = ROOT / 'experiments/recordings/exp41'
CASES = [('social', '等一下我没有社保我30岁首次年缴多少钱', 679),
         ('age', '不对我不是30岁我是40岁有社保首次年缴多少钱', 517)]


def main():
    REC.mkdir(exist_ok=True, parents=True)
    folder = OUT / 'exp41_history_records'
    folder.mkdir(exist_ok=True)
    evidence = [Evidence(**e) for e in json.loads((OUT / 'exp33_organized_sources.json').read_text(encoding='utf-8'))['evidence']]
    save(OUT / 'exp41_plan.json', {'cases': CASES, 'model': cloud.SLOW_MODEL,
        'method': 'Consultation-stage two fresh DS answers per case, actual TTS/WASAPI receipt markers and live Tencent ASR with synthetic correction and 3% echo. Not telephone or physical microphone. Completed sentences only: unaligned interrupted sentence conservatively omitted. Initial price question is text input, correction uses actual ASR. Next actual model input context is captured, no model grader or insurance rule repair.',
        'code_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
            ['src/agent/session.py', 'src/agent/playback.py', 'src/agent/dialogue.py']}})
    for name, question, expected in CASES:
        path = folder / (name + '.json')
        if path.exists():
            continue
        row = {'id': name, 'correction': question, 'expected_price_not_sent': expected}
        prompts = []
        engine = EvidenceAnswerer()
        original = engine.call
        def call(system, user, model):
            prompts.append(user.split('【全部本轮证据】')[0])
            return original(system, user, model)
        engine.call = call
        session = CallSession(EvidencePipeline(evidence, answerer=engine), playback_tracking=True,
            profile={'relation': '本人', 'age': 30, 'has_social': True})
        session.flags.update(quoted=True, limits=True, disclosure=True)
        tts = cloud.RealtimeTTS(voice='Cherry', speech_rate=1.2)
        pool = ThreadPoolExecutor(1)
        player = None
        try:
            speech = CachedSpeech(tts)
            speech.warm()
            base = session.turn('我30岁有社保第一次买基础版年缴多少钱')
            row['history_before_playback'] = list(session.history)
            receipts = PlaybackReceipts(session, base.playback_id)
            prepared = []
            for text in _sentences(base.text):
                token = receipts.enqueue(text)
                audio = speech.synth(text)[2]
                prepared.append((token, audio))
            full_audio = b''.join(audio for _, audio in prepared)
            wav(REC / (name + '_base.wav'), full_audio)
            pcm = user_audio(question, 'Ethan')
            mic, onset, _ = mix(cloud.resample_24k_to_16k(full_audio), pcm, 1.2, .03)
            player = TrackedDacPlayer(receipts)
            origin = [None]
            stopped = [None]
            decisions = []
            futures = []
            ctrl = BargeInController(lambda text: small_model.classify_barge_in(text, client=session.pipe.client))
            ctrl.note_spoken(0, base.text)
            def decide(ev):
                if stopped[0] is not None:
                    return
                now = time.perf_counter() - origin[0]
                decision = ctrl.on_partial(now, ev.text, now < len(full_audio) / 48000)
                if decision is not None:
                    decisions.append({'action': decision.action, 'asr': ev.text, 'time_s': now})
                    if decision.action == 'stop':
                        stopped[0] = player.abort()
            def event(ev):
                if ev.text:
                    futures.append(pool.submit(decide, ev))
            def frame(data, at):
                if origin[0] is None:
                    origin[0] = time.perf_counter() - (at - len(data) / 32000)
                    for token, audio in prepared:
                        player.feed(audio)
                        player.sentence_end(token)
            events, _ = asyncio.run(cloud.tencent_asr_stream(mic, vad_silence_ms=500,
                trailing_silence_ms=0, on_event=event, on_audio=frame))
            for future in futures:
                future.result()
            row['history_after_interrupt'] = list(session.history)
            corrected = next((e.text for e in events if e.slice_type == 2 and
                ('没有社保' in e.text if name == 'social' else '40' in e.text or '四十' in e.text)), None)
            row.update(base_text=base.text, asr_events=[vars(e) for e in events],
                corrected_asr_text=corrected, decisions=decisions, cancellation=stopped[0],
                stop_latency_s=stopped[0]['last_scheduled_dac_wall'] - origin[0] - onset if stopped[0] else None,
                receipts=receipts.fragments, heard_text=receipts.heard,
                dropped_text=stopped[0]['history']['dropped'] if stopped[0] else None,
                stopped_before_natural_end=bool(stopped[0] and stopped[0]['dropped_frames'] > 0))
            player.close()
            player = None
            if corrected is None or not row['stopped_before_natural_end']:
                row['acceptance_error'] = 'No confirmed correction or no actual early stop'
            else:
                reply = session.turn(corrected)
                row['next_model_input_context'] = prompts[-1]
                reply_receipts = PlaybackReceipts(session, reply.playback_id)
                player = TrackedDacPlayer(reply_receipts)
                audio_out = []
                for text in _sentences(reply.text):
                    token = reply_receipts.enqueue(text)
                    audio = speech.synth(text, on_audio=player.feed)[2]
                    audio_out.append(audio)
                    player.sentence_end(token)
                player.drain()
                wav(REC / (name + '_reply.wav'), b''.join(audio_out))
                row.update(reply_text=reply.text, reply_timings=reply.timings,
                    final_profile=dict(session.profile), final_history=list(session.history),
                    next_model_has_unheard_full_reply=base.text in prompts[-1],
                    old_heard_only=all(text == receipts.heard for speaker, text in row['history_after_interrupt'] if speaker == '坐席'),
                    corrected_price_present=str(expected) in reply.text)
        except Exception as exc:
            row['error_type'] = type(exc).__name__
        finally:
            pool.shutdown(wait=True)
            if player is not None:
                player.close()
            tts.close()
        save(path, row)
        print(json.dumps({k: row.get(k) for k in ['id', 'heard_text', 'stop_latency_s', 'old_heard_only',
            'next_model_has_unheard_full_reply', 'corrected_price_present', 'error_type', 'acceptance_error']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
