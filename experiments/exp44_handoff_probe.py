"""Replay only the failed final voice turn from its exact saved heard prefix."""
import json
from copy import deepcopy
from functools import partial
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import exp38_complex_conversation as voice
import exp36_new_voice_acceptance as base
from agent import cloud
from agent.session import CallSession
from agent.evidence_pipeline import EvidencePipeline
from agent.fast_lane import CachedSpeech
from exp42_full_history import Recorder,evidence
from windows_https_transport import native_client


def main():
    out=ROOT/'experiments/results/exp44_handoff_probe'
    out.mkdir(exist_ok=True)
    path=out/'turn.json'
    if path.exists():raise ValueError('Probe already saved; do not overwrite')
    previous=json.loads((ROOT/'experiments/results/exp44_voice_direct/voice/06_turn.json').read_text(encoding='utf-8'))
    recorder=Recorder(inner=native_client())
    session=CallSession(EvidencePipeline(evidence(),client=recorder),history=[tuple(h) for h in previous['history_after']],
                        profile=deepcopy(previous['profile']),flags=deepcopy(previous['flags']),
                        consultation=deepcopy(previous['consultation']))
    voice.SESSION=session
    record=ROOT/'experiments/recordings/exp44_handoff_probe';record.mkdir(parents=True,exist_ok=True)
    voice.REC=record;base.REC=record/'user';base.REC.mkdir(exist_ok=True)
    cloud.tencent_asr_stream=partial(cloud.tencent_asr_stream,proxy=None)
    tts=cloud.RealtimeTTS(voice='Cherry',speech_rate=1.2)
    try:
        cache=CachedSpeech(tts);cache.warm();voice.CACHE.update(cache.audio)
    finally:tts.close()
    row=voice.voice_one({'id':'T07','question':voice.QUESTIONS[-1]},1,evidence())
    row['api_calls']=recorder.trace
    path.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:row.get(k) for k in ['id','session_text','safe_error_message','error_type','error_stack']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
