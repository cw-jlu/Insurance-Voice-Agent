"""Persist unfinished questions independently of the sales stages and transcript."""
from copy import deepcopy
import re

STATUSES = {'answered', 'needs_user', 'needs_review', 'cancelled'}
QUESTIONS = {
    'age':'被保人现在多少周岁', 'has_social':'被保人是否有基本医保',
    'insured_with_social':'投保时选择的是有医保身份还是无医保身份',
    'remaining_deductible':'本年度还剩多少免赔额',
    'social_settled':'本次是否已按医保身份就诊并结算',
    'family_size':'适用家庭单规则的累计家庭成员人数是多少',
    'first_insurance':'是否首次投保', 'plan':'希望了解哪个保障计划',
    'payment':'希望按年缴还是月缴了解价格',
    'occupation':'被保人的具体职业和作业情况是什么',
    'diagnosis_timing':'这项疾病是投保前已有，还是投保后首次确诊',
}
_UNKNOWN = re.compile(r'不知道|不清楚|不确定|没说|没告诉|未说明|不记得')
_EXAMPLE = re.compile(r'假设|假如|举例|举个例|比如|要是|如果')


def pending(state):
    return bool(state) and state['status'] in ('pending','error','awaiting_playback')


def start(question, identifier):
    return {'id':identifier, 'initial_question':question, 'status':'pending',
            'revision':0, 'tasks':[], 'facts':[], 'guard_faults':[]}


def reopen(state, changed_fields=()):
    state['status'] = 'pending'
    # A correction must not be answered using an obsolete confirmed condition.
    state['facts'] = [f for f in state.get('facts',[]) if f['name'] not in changed_fields]
    for task in state.get('tasks',[]):
        if task['status'] == 'answered' and (not changed_fields or
                set(task.get('requires',[])) & set(changed_fields)):
            task['status'] = 'pending'
            task.pop('quoted_amount',None)


def model_context(state):
    return deepcopy(state)


def question_plan(value, current_text, previous=None):
    """Accept new questions only with a literal quote from the current turn.

    History resolves references; it cannot supply the quote for a new task.
    Existing task IDs are hints from the planner, never invented IDs to insert.
    """
    if not isinstance(value,list):
        return []
    task_ids = {t['id'] for t in (previous or {}).get('tasks',[])}
    plan = []
    for entry in value:
        if not isinstance(entry,dict):
            continue
        question, quote = entry.get('question'), entry.get('user_quote')
        task_id = entry.get('task_id')
        if (not isinstance(question,str) or not question.strip() or
                not isinstance(quote,str) or not quote.strip() or quote not in current_text):
            continue
        if task_id not in (None,'') and (not isinstance(task_id,str) or task_id not in task_ids):
            continue
        item = {'question':question.strip(),'user_quote':quote,'task_id':task_id or None}
        if item not in plan:
            plan.append(item)
    return plan[:8]


def add_questions(state,questions,current_text=None):
    """Seed current questions from the independent small-model planner.

    Conditions-only replies contain no new questions. Seeded IDs become mandatory
    input tasks so the answer model cannot silently omit a compound question.
    """
    if current_text is not None:
        questions = question_plan(questions,current_text,state)
    existing={re.sub(r'[\s，,。！？?；;]','',t['question']):t for t in state['tasks']}
    used_ids={t.get('id') for t in state['tasks']}
    for i,question in enumerate(questions,1):
        entry = question if isinstance(question,dict) else None
        question = entry['question'] if entry else question
        key=re.sub(r'[\s，,。！？?；;]','',question)
        old = next((t for t in state['tasks'] if entry and t['id']==entry.get('task_id')),None)
        old = old if old is not None else existing.get(key)
        if not key:continue
        if old is not None:
            if entry:
                old['seeded'] = True
                old['user_quote'] = entry['user_quote']
                if old['status'] in ('answered','cancelled'):
                    old['status'] = 'needs_answer'
                    old.pop('quoted_amount',None)
            continue
        base_id=f"{state['id']}-u{state['revision']+1}-{i}"
        task_id=base_id
        suffix=2
        while task_id in used_ids:
            task_id=f"{base_id}-{suffix}"
            suffix+=1
        task = {'id':task_id,
            'question':question,'kind':'other','status':'needs_answer','requires':[],
            'is_example':False,'seeded':True}
        if entry:
            task['user_quote'] = entry['user_quote']
        state['tasks'].append(task)
        existing[key] = task
        used_ids.add(task_id)


def validate_progress(value, previous, user_texts, profile, current_text, condition_user_texts=None, answer_source_ids=()):
    """Validate task structure and user-owned condition provenance.

    Does not decide clinical/contract facts or invent a corrected business answer.
    Invalid assumed conditions become missing and are asked before releasing an
    individual calculation. Raw model output remains available in the trace.
    """
    if not isinstance(value,dict) or not isinstance(value.get('tasks'),list) or not value['tasks']:
        raise ValueError('Missing consultation tasks')
    if not isinstance(value.get('facts'),list):
        raise ValueError('Missing consultation facts')
    tasks, facts = deepcopy(value['tasks']), deepcopy(value['facts'])
    if not all(isinstance(t,dict) for t in tasks) or not all(isinstance(f,dict) for f in facts):
        raise ValueError('Invalid consultation task/fact object')
    if len({t.get('id') for t in tasks}) != len(tasks):
        raise ValueError('Duplicate consultation task id')
    old_ids = {t['id'] for t in previous.get('tasks',[])}
    if not old_ids.issubset({t.get('id') for t in tasks}):
        raise ValueError('Unfinished consultation tasks dropped')
    if len({f.get('name') for f in facts}) != len(facts):
        raise ValueError('Duplicate condition name')
    faults = []
    old_tasks = {t['id']:t for t in previous.get('tasks',[])}
    for task in tasks:
        old = old_tasks.get(task.get('id'),{})
        # Model output cannot erase the independent planner's release contract.
        for field in ('seeded','user_quote'):
            if field in old:
                task[field] = old[field]
    for f in facts:
        if not isinstance(f.get('name'),str) or not f['name'] or f.get('state') not in ('confirmed','missing','assumed'):
            raise ValueError('Invalid consultation fact')
        quote = f.get('user_quote','')
        if not isinstance(quote,str):
            raise ValueError('Invalid user quote')
        if f['state']=='confirmed':
            valid = bool(quote.strip()) and any(quote in text for text in user_texts) and not _UNKNOWN.search(quote) and f.get('value') is not None
            if condition_user_texts is not None and f['name'] in ('age','has_social','gender','insured_with_social','remaining_deductible'):
                valid = valid and any(quote in text for text in condition_user_texts)
            if f['name']=='insured_with_social':
                valid = valid and bool(re.search(r'投保|参保|保单',quote)) and type(f.get('value')) is bool
            if f['name']=='has_social':
                valid = valid and bool(re.search(r'医保|社保|公费医疗|医疗保险',quote)) and type(f.get('value')) is bool
            if f['name']=='social_settled':
                negative=bool(re.search(r'未.{0,8}(?:医保|社保)|(?:没走|没用|没按|没有用|没有按).{0,4}(?:医保|社保)|(?:医保|社保).{0,4}(?:未|没)',quote))
                valid = valid and bool(re.search(r'医保|社保|公费医疗',quote)) and bool(re.search(r'结算|报销|报了|已报|医保报|社保报|没走',quote)) and type(f.get('value')) is bool and f['value']!=negative
            if f['name']=='remaining_deductible':
                valid = valid and '免赔' in quote and bool(re.search(r'剩|还|已|抵|用',quote)) and isinstance(f.get('value'),(int,float)) and not isinstance(f.get('value'),bool)
            if f['name'] in ('age','has_social') and profile.get(f['name']) is not None:
                valid = valid and f.get('value') == profile[f['name']]
            if not valid:
                f['state'], f['value'] = 'missing', None
                faults.append('unconfirmed_condition:'+f['name'])
            elif _EXAMPLE.search(quote):
                f['state'] = 'assumed'
    by_name = {f['name']:f for f in facts}
    # Normalize a model's legacy field name only when its own user-owned quote
    # explicitly identifies the policy's enrollment identity.
    social = by_name.get('has_social')
    if 'insured_with_social' not in by_name and social and social['state']=='confirmed' and type(social['value']) is bool and re.search(r'投保|参保|保单',social['user_quote']):
        canonical = {**social,'name':'insured_with_social'}
        facts.append(canonical)
        by_name['insured_with_social'] = canonical
    example_authorized = bool(_EXAMPLE.search(previous['initial_question']) or _EXAMPLE.search(current_text))
    for task in tasks:
        if not isinstance(task.get('id'),str) or not task['id'] or not isinstance(task.get('question'),str) or not task['question'].strip():
            raise ValueError('Invalid consultation task')
        if task.get('status') not in STATUSES or not isinstance(task.get('kind'),str):
            raise ValueError('Invalid consultation task state')
        if not isinstance(task.get('requires'),list) or not all(isinstance(n,str) and n for n in task['requires']):
            raise ValueError('Invalid task dependencies')
        if task['status']=='cancelled':
            quote = task.get('cancel_user_quote','')
            if not quote or not any(quote in text for text in user_texts) or not re.search(r'不问|别问|不用|不需要|先不|取消|不买',quote):
                raise ValueError('Task cancellation lacks explicit user evidence')
        sources=task.get('source_ids',[])
        if not isinstance(sources,list) or not all(isinstance(i,str) for i in sources):
            raise ValueError('Invalid consultation task source ids')
        if sources and not set(sources).issubset(set(answer_source_ids)):
            raise ValueError('Task sources not cited in the actual answer')
        sources=sources or answer_source_ids
        responsibilities={i.split(':')[0] for i in sources if re.match(r'^R\d\d:',i)}
        if task['kind']=='claim_calculation':
            # Shared annual deductible applies to R01/R02, not daily allowances,
            # zero-deductible outpatient/addon benefits or pharmacy monthly caps.
            if not responsibilities or responsibilities & {'R01','R02','R09'}:
                required=['insured_with_social','remaining_deductible']
            elif responsibilities & {'R03','R04','R06','R11','R12','R13'}:
                required=['insured_with_social']
                task['requires']=[n for n in task['requires'] if n!='remaining_deductible']
            elif responsibilities.issubset({'R05','R07','R08','R10','R14','R15','R16'}):
                required=[]
                task['requires']=[n for n in task['requires'] if n not in ('has_social','insured_with_social','remaining_deductible')]
            else:
                required=['insured_with_social','remaining_deductible']
            task['requires'] = list(dict.fromkeys(task['requires']+required))
            settled=by_name.get('social_settled')
            if settled and settled['state']=='confirmed' and settled['value'] is True:
                # The 60% adjustment requires BOTH social enrollment AND no
                # social settlement. Confirmed settlement already excludes it.
                task['requires']=[n for n in task['requires'] if n not in ('insured_with_social','has_social')]
                task['requires']=list(dict.fromkeys(task['requires']+['social_settled']))
        if task['kind'] in ('price','price_difference'):
            unavailable = task['kind']=='price' and task['status']=='answered' and 'quoted_amount' in task and task['quoted_amount'] is None
            rates=[i for i in sources if i.startswith('rate:')]
            independent_addon=bool(rates) and all('_加油包:' in i for i in rates) and bool(re.search(
                r'重疾关爱|重大疾病扩展特需|住院医疗费用补偿|住院津贴|门急诊医疗加油包|药省保',task['question'])) and '家庭共享' not in task['question']
            if independent_addon:
                task['requires']=[n for n in task['requires'] if n!='has_social']
            task['requires'] = list(dict.fromkeys(task['requires']+(['age'] if unavailable or independent_addon else ['age','has_social'])))
        if task['kind']=='discount':
            task['requires'] = list(dict.fromkeys(task['requires']+['family_size']))
        missing = []
        for name in task['requires']:
            f = by_name.get(name)
            allowed = f and (f['state']=='confirmed' or (f['state']=='assumed' and task.get('is_example') is True and example_authorized))
            if not allowed:
                missing.append(name)
        task['missing_conditions'] = missing
        if missing and task['status'] not in ('needs_review','cancelled'):
            if task['status']=='answered':
                faults.append('answer_missing_conditions:'+task['id'])
            task['status']='needs_user'
    state = {**previous, 'revision':previous['revision']+1, 'tasks':tasks, 'facts':facts, 'guard_faults':faults}
    state['status'] = 'pending' if any(t['status'] in ('needs_user','needs_review') for t in tasks) else 'complete'
    return state


def missing_questions(state):
    names = list(dict.fromkeys(n for t in state['tasks'] if t['status']=='needs_user' for n in t['missing_conditions']))
    questions=[]
    for name in names:
        if name in QUESTIONS:questions.append(QUESTIONS[name])
        else:
            task=next(t for t in state['tasks'] if name in t['missing_conditions'])
            questions.append('关于“'+task['question']+'”，还需要补充哪些具体情况')
    return questions


def quote_scope_known(profile, user_texts):
    """Only allow the legacy sales quote in its explicit standard-price scope."""
    if type(profile.get('age')) is not int or not 1 <= profile['age'] <= 70 or type(profile.get('has_social')) is not bool:
        return False
    text = '\n'.join(user_texts)
    if re.search(r'续保|重投|重新投保|上一张|加油包|家庭|一家|优惠|折扣|一起买',text):
        return False
    return bool(re.search(r'首次|第一次',text) and re.search(r'必选|基础',text) and re.search(r'个人单|自己|本人|单人',text))
