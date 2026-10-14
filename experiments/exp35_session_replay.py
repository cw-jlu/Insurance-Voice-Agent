"""Replay saved real model outputs through session execution; no new cloud/audio call."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent.evidence_answer import Evidence, EvidenceAnswerer
from agent.evidence_pipeline import EvidencePipeline
from agent.session import CallSession

def main():
    out=ROOT/'experiments/results'
    evidence=[Evidence(**e) for e in json.loads((out/'exp33_organized_sources.json').read_text(encoding='utf-8'))['evidence']]
    rows=[]
    for case in ['H15','H10','H13','H07']:
        record=json.loads((out/'exp35_records'/(case+'.json')).read_text(encoding='utf-8'))
        engine=EvidenceAnswerer(client=object())
        pipe=EvidencePipeline(evidence,client=object(),answerer=engine)
        pipe.understand=lambda *args:{'intent':'保障','slots':{},'items':[],'topics':[],'gaps':[]}
        session=CallSession(pipe,profile=record['known'],voice=False)
        spoken=[]
        with patch.object(engine,'call',return_value=record['calls'][0]) as call, \
             patch('agent.small_model.classify_act',return_value='提问'), \
             patch('agent.small_model.detect_gaps') as gaps:
            result=session.turn(record['question'],on_sentence=spoken.append)
        assert call.call_count==1
        gaps.assert_not_called()
        assert ''.join(spoken)==result.text
        if record['outcome']['action']=='handoff':
            assert session.ended and session.flags['handoff'] and not session.flags['link']
            assert session.handoff_request['spoken_text']==record['text']
            assert session.handoff_request['status']=='requested'
        if record['outcome']['action']=='clarify':
            assert not session.ended and not session.flags['handoff'] and not session.flags['link']
        rows.append({'id':case,'question':record['question'],'model_text':record['text'],
            'model_generation_s':record['total_s'],'outcome':record['outcome'],'spoken_chunks':spoken,
            'session_text':result.text,'events':result.events,'flags':session.flags,
            'ended':session.ended,'timings':result.timings})
    (out/'exp35_session_replay.json').write_text(json.dumps({'method':'Integration replay of saved exp35 real model outputs. Understanding and act classification stubbed; no new cloud generation, STT, TTS or actual phone transfer. H13 intentionally preserves unnecessary handoff to prove no insurance rule repair.','rows':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'id':r['id'],'events':r['events'],'ended':r['ended']} for r in rows],ensure_ascii=False))

if __name__=='__main__':main()
