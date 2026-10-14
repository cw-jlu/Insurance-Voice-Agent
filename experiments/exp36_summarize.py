"""Summarize fresh voice acceptance without model grading or silent exclusions."""
import hashlib
import json
import math
import statistics
from pathlib import Path

OUT=Path(__file__).resolve().parent/'results'

def read_folder(name):
    return [json.loads(f.read_text(encoding='utf-8')) for f in sorted((OUT/name).glob('*.json'))]

def distribution(values):
    values=sorted(v for v in values if v is not None)
    if not values:return None
    return {'n':len(values),'p50':statistics.median(values),'p90':values[math.ceil(.9*len(values))-1],
        'p95':values[math.ceil(.95*len(values))-1],'max':max(values)}

def score(n,total):
    return {'passed':n,'n':total,'percent':round(100*n/total,2) if total else None}

def main():
    plan=json.loads((OUT/'exp36_plan.json').read_text(encoding='utf-8'))
    policy=json.loads((OUT/'exp35_plan.json').read_text(encoding='utf-8'))
    required={p['id'] for p in policy['policies_not_sent_to_model'] if p['required_external']}
    labels=json.loads((OUT/'exp36_voice_manual_audit.json').read_text(encoding='utf-8'))['decisions']
    voice=read_folder('exp36_voice_records');reviewed=[]
    for r in voice:
        key=f"{r['id']}_r{r['round']}"
        c,t,mapped,reason=labels[key]
        evidence=r.get('timings',{}).get('answer_detail',{}).get('evidence_result')
        valid=not r.get('error_type') and not r.get('split_intended_turn') and r.get('first_business_dac_s',-1)>=0
        handoff=bool(r.get('flags',{}).get('handoff'))
        externally_required=r['id'] in required
        handled=valid and c and t and mapped and (handoff or not externally_required)
        reviewed.append({'key':key,'content_correct':c,'model_source_supported':t,
            'all_spoken_facts_mapped':mapped,'voice_turn_valid':valid,'ds_invoked':evidence is not None,
            'required_external':externally_required,'handoff_requested':handoff,
            'handled_correctly_and_traceable':handled,'reason':reason,
            'record_sha256':hashlib.sha256((OUT/'exp36_voice_records'/(key+'.json')).read_bytes()).hexdigest()})
    valid_voice=[r for r in voice if not r.get('error_type') and not r.get('split_intended_turn') and r.get('first_business_dac_s',-1)>=0]
    invoked=[r for r in reviewed if r['ds_invoked']]
    interrupted=read_folder('exp36_interrupt_records')
    it=[r for r in interrupted if r['kind']=='interrupt'];bc=[r for r in interrupted if r['kind']=='backchannel'];echo=[r for r in interrupted if r['kind']=='echo']
    actual_stop=lambda r:not r.get('error_type') and r.get('stopped') and (r.get('cancellation') or {}).get('dropped_frames',0)>0
    correct_stop=lambda r:actual_stop(r) and not r.get('wrong_stop_before_user') and r.get('stop_latency_s') is not None and r['stop_latency_s']>=0
    cancelled=read_folder('exp36_cancel_records');live=read_folder('exp36_live_correction_records')
    result={'date':'2026-10-12','plan_sha256':hashlib.sha256((OUT/'exp36_plan.json').read_bytes()).hexdigest(),
        'status':{'voice_complete':len(voice)==24,'lowest_echo_gate_complete':len(interrupted)==38,
            'generation_cancel_complete':len(cancelled)==4,'live_correction_complete':len(live)==2},
        'method':plan['method'],
        'voice':{'planned':24,'completed':len(voice),'service_errors':[f"{r['id']}_r{r['round']}" for r in voice if r.get('error_type')],
            'split_failures':[f"{r['id']}_r{r['round']}" for r in voice if r.get('split_intended_turn')],
            'valid_complete_turns':len(valid_voice),
            'first_business_dac_s':distribution([r['first_business_dac_s'] for r in valid_voice]),
            'complete_dac_playback_s':distribution([r['actual_dac_playback_end_s'] for r in valid_voice]),
            'first_within_3s_all_planned':score(sum(0<=r['first_business_dac_s']<=3 for r in valid_voice),24),
            'complete_within_10s_all_planned':score(sum(0<=r['actual_dac_playback_end_s']<=10 for r in valid_voice),24),
            'asr_endpoint_s':distribution([r['asr_endpoint_s'] for r in valid_voice]),
            'flow_understand_s':distribution([r['timings'].get('understand') for r in valid_voice]),
            'ds_generate_s':distribution([r['timings'].get('answer') for r in valid_voice]),
            'spoken_chars':distribution([len(r['session_text']) for r in valid_voice]),
            'audio_duration_s':distribution([r['audio_duration_s'] for r in valid_voice]),
            'content_correct':score(sum(r['content_correct'] for r in reviewed),len(reviewed)),
            'ds_source_semantics':score(sum(r['model_source_supported'] for r in invoked),len(invoked)),
            'all_spoken_facts_mapped_and_supported':score(sum(r['all_spoken_facts_mapped'] for r in reviewed),len(reviewed)),
            'handled_correctly_and_traceable':score(sum(r['handled_correctly_and_traceable'] for r in reviewed),len(reviewed)),
            'required_handoffs':score(sum(r['required_external'] and r['handoff_requested'] for r in reviewed),sum(r['required_external'] for r in reviewed)),
            'missed_handoffs':[r['key'] for r in reviewed if r['required_external'] and not r['handoff_requested']],
            'speaker_statuses':[{'key':f"{r['id']}_r{r['round']}",'statuses':r.get('playback_statuses')} for r in voice if r.get('playback_statuses')]},
        'interrupt':{'scope':'Lowest injected echo gain 3% acceptance gate; 20 actual interruptions, 10 backchannels, 8 echo-only. Runtime decisions during ASR and real DAC-buffer cancellation. Other two echo gains planned but not required once this gate fails.',
            'completed':len(interrupted),'errors':[r['key'] for r in interrupted if r.get('error_type')],
            'planned_interrupts':20,
            'correct_stop_or_continue_actions':score(sum(correct_stop(r) for r in it)+sum(not r.get('error_type') and not actual_stop(r) for r in bc+echo),len(interrupted)),
            'interrupted_before_natural_end':score(sum(correct_stop(r) for r in it),len(it)),
            'stopped_within_500ms':score(sum(correct_stop(r) and r['stop_latency_s']<=.5 for r in it),len(it)),
            'stop_latency_s':distribution([r['stop_latency_s'] for r in it if correct_stop(r)]),
            'backchannel_false_stops':score(sum(actual_stop(r) for r in bc),len(bc)),
            'echo_only_false_stops':score(sum(actual_stop(r) for r in echo),len(echo)),
            'late_stop_decisions_after_queue_drained':[r['key'] for r in interrupted if r.get('stopped') and not actual_stop(r)],
            'premature_stops':[r['key'] for r in it if r.get('wrong_stop_before_user')],
            'unexecuted_higher_echo_trials':76},
        'generation_cancel':{'n':len(cancelled),'in_flight_confirmed':sum(r.get('in_flight_at_cancel',False) for r in cancelled),
            'old_reply_released':[r['id'] for r in cancelled if r.get('initial_text_released')],
            'error_ids':[r['id'] for r in cancelled if r.get('error_type')]},
        'live_correction':{'n':len(live),'unheard_full_answer_in_history':[r['id'] for r in live if r.get('unheard_full_answer_retained_in_history')],
            'error_ids':[r['id'] for r in live if r.get('error_type')]},
        'limits':['Diagnostic fixed 12 questions repeated with two TTS voices; not random customer holdout. Do not compare accuracy with all143.',
            'DS semantic citations and complete spoken trace coverage are different: flow-added insurance facts are absent from model claims.',
            'Two ASR split failures retained; invalid negative timings are not counted as fast successful turns.',
            'Output timestamps are real WASAPI scheduled DAC times, not acoustic loopback or telephone measurements. Input and echo are synthetic.',
            'Input/answer WAVs preserve source PCM, not room microphone recordings.',
            'Four speculative in-flight cancellation tests and two same-session playback corrections do not certify full production interruption success.'],
        'voice_audit_rows':reviewed}
    (OUT/'exp36_acceptance_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['voice_audit_rows','limits','method']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
