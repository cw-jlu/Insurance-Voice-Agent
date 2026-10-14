"""Execute structured decisions without deciding insurance facts or repairing text."""
from copy import deepcopy


def validate_outcome(value, evidence):
    from agent.evidence_answer import source_registry
    if not isinstance(value, dict):
        raise ValueError('Missing model outcome')
    action=value.get('action')
    kind=value.get('reason_kind')
    if action not in ('answer','clarify','handoff'):
        raise ValueError('Invalid outcome action')
    allowed={'answer':{'none'},'clarify':{'missing_user_condition'},
        'handoff':{'missing_material','source_conflict','individual_review'}}
    if kind not in allowed[action]:
        raise ValueError('Outcome reason does not match action')
    for field in ('pending_questions','source_ids'):
        if not isinstance(value.get(field),list) or not all(isinstance(x,str) and x.strip() for x in value[field]):
            raise ValueError('Invalid outcome list')
    if not isinstance(value.get('reason'),str):
        raise ValueError('Invalid outcome reason')
    if action!='answer' and (not value['reason'].strip() or not value['pending_questions']):
        raise ValueError('Unresolved decision requires reason and questions')
    if action=='answer' and value['pending_questions']:
        raise ValueError('Resolved answer cannot carry pending questions')
    if action=='handoff' and not value['source_ids']:
        raise ValueError('Handoff requires boundary sources')
    registry=source_registry(evidence)
    if any(s not in registry for s in value['source_ids']):
        raise ValueError('Unknown outcome source')
    if kind=='source_conflict' and len(set(value['source_ids']))<2:
        raise ValueError('Conflict requires both source references')
    return deepcopy(value)


def handoff_request(question, known, result):
    outcome=result['outcome']
    if outcome['action']!='handoff':
        return None
    return {'status':'requested','origin':'post_answer','question':question,
        'known':deepcopy(known),'reason_kind':outcome['reason_kind'],'reason':outcome['reason'],
        'pending_questions':list(outcome['pending_questions']),
        'source_ids':list(outcome['source_ids']),'spoken_text':result['text'],
        'claims':deepcopy(result['structured']['claims']),
        'provenance':deepcopy(result['provenance'])}
