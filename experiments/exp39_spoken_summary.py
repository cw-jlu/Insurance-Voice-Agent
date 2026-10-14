"""Fresh short-answer candidate on the saved complex conversation context."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agent import cloud
from agent.evidence_answer import Evidence, EvidenceAnswerer, handoff_system

OUT = ROOT / 'experiments/results'


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    evidence = [Evidence(**e) for e in json.loads((OUT / 'exp33_organized_sources.json').read_text(encoding='utf-8'))['evidence']]
    folder = OUT / 'exp39_spoken_summary_records'
    folder.mkdir(exist_ok=True)
    save(OUT / 'exp39_plan.json', {'model': cloud.SLOW_MODEL, 'prompt_sha256': hashlib.sha256(handoff_system('v3').encode()).hexdigest(),
        'cases': ['02', '03', '06'], 'method': 'Fresh DS v3 short spoken summaries in original exp38 ASR/history/profile context. Baseline is saved prior sequential run, not randomized contemporaneous control. Real TTS audio duration, no new ASR/DAC test or automated semantic judge.'})
    engine = EvidenceAnswerer()
    tts = cloud.RealtimeTTS(voice='Cherry', speech_rate=1.2)
    try:
        for number in ['02', '03', '06']:
            old = json.loads((OUT / 'exp38_conversation_records' / (number + '_turn.json')).read_text(encoding='utf-8'))
            history = old['history_before'] + [['用户', old['asr_text']]]
            known = {**old['profile'], '_history': '\n'.join(f'{s}：{t}' for s, t in history[-6:])}
            new = engine.answer(old['asr_text'], known, evidence, allow_handoff=True, handoff_version='v3')
            row = {'id': old['id'], 'question': old['asr_text'], 'known': known, 'baseline_text': old['timings']['answer_detail']['evidence_result']['text'],
                   'baseline_total_spoken_audio_s': old['audio_duration_s'], 'candidate': new}
            save(folder / (old['id'] + '.json'), row)
            if new['outcome_validation']['valid'] and new['provenance']['valid']:
                _, _, pcm = tts.synth(new['text'])
                row['candidate_audio_s'] = len(pcm) / 48000
                save(folder / (old['id'] + '.json'), row)
            print(json.dumps({'id': old['id'], 'text': new['text'], 'chars': len(new['text']), 'audio_s': row.get('candidate_audio_s'), 'generation_s': new['total_s']}, ensure_ascii=False), flush=True)
    finally:
        tts.close()
