"""Experimental source-backed answering; one pass or bounded blind deliberation.

No answer repair, semantic guard, canned answer or majority vote. Structural citation
checks report defects rather than changing what the model wrote.
"""
from __future__ import annotations

import hashlib
import csv
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agent import cloud
from agent.context import HistoryPrompt, with_history
from agent.knowledge import CLEAN, Chunk, clause_chunks, coverage_items
from agent.pipeline import item_chunks
from agent.source_locations import pdf_locations, excel_location

SYSTEM = '''你是尊享e生·百万医疗保险2026版的电话助理。仅依据本轮证据回答原问题。
先在内部确定用户本轮各个必须回答的事实，再写一个短答；不要输出思维过程。
责任名称、基础/加油包、适用对象、免赔、比例及医院条件必须属于同一责任，不能拼接不同责任的数字。
证据中的“必选责任”属于基础版；“加油包”需另购。资料中有相似责任时核对原文区分，不靠词相似猜。
只保留改变本轮结论的条件：不把条件保障说成保证赔付；不漏用户所问事项；不罗列未问的所有条款。
自称有病史的用户不能保证承保，核保与既往症赔付限制分别回答；纯假设疾病不能当成用户已有病史。
原文缺失的年龄、续保、核保个案等结论，说明本轮资料无法确认；引用边界证据不能冒充缺失条款。
问金额时给结果与简短可核查公式。总费用不等于可赔费用：若社保/其他补偿、费用合规性、年度免赔已使用情况未知，明确标注计算是假设示例，不作确定实际赔付。共享免赔按原文共用，不按费用项目重复扣除。
意外与疾病适用条件分别读取原句，不把疾病等待期扩大到意外。不添加原文没有的限制。
报价按费率表区间及社保身份取值，说明缴费单位；未知条件只追问该条件。
尽量用80至160字，复杂问题可适当超过，不为字数删必要条件，不重复、不添加销售追问。
只输出JSON：
{"segments":[{"text":"完整、直接对用户说的一个句子，不读引用编号", "citations":[{"id":"证据ID", "quote":"该证据中逐字原文，不加省略号"}], "derivation":"仅计算事实填简短公式与假设，否则空字符串"}], "unknowns":[{"question":"不能确认的被问事项", "reason":"缺少的证据或用户条件"}]}。
最终口播直接按顺序连接segments.text，不会另行改写。用2至5个自然句，不能输出条款摘抄代替回答。
所有事实、金额、适用条件和承诺都在segments内；每个segment至少一个直接支持它的原文引用，必要时多个。
引用存在不代表支持：引用必须支持整个句子的事实及其条件；派生金额必须引用费率/免赔/比例的原始证据。
unknowns只记录不确定事项，不用于漏掉已能回答的部分。不要输出原文之外的知识。'''

INDEPENDENT = '''你是独立保险证据分析员，仅按本轮原问题及原文形成自己的答案和引用。
你看不到任何其他分析员的答案；不要预测别人意见，不追求一致。
你的专长提示为：%s。但仍必须完整回答原问题，不能只回答专长部分。
''' + SYSTEM

CHALLENGE = '''你已经独立完成初稿，现在只能做一次证据质询。你的初稿和对方初稿不代表权威。
不得根据对方语气、模型名、票数或一致性改结论。仅记录有原文支持的具体争议或遗漏。
你可以坚持原结论，也可以明确承认原结论错误；只输出JSON：
{"issues":[{"claim":"争议事实", "position":"支持/反对/无法确认", "citations":[{"id":"证据ID","quote":"逐字原文"}], "reason":"简短事实说明"}]}。
不要重写答案，不输出推理过程。'''

ARBITER = '''给定两份匿名独立初稿和各自一次质询。独立初稿均已在互不可见时提交。
请重新以原文为依据作最终回答。禁止多数投票、因双方同意就采信、或偏好第一份。
每个争议逐一核对原始证据；两方都错时独立给正确结论；证据无法确认则明确留白。
草稿中的引用、金额和计算也可能错误，不能直接复制。''' + SYSTEM

# Second frozen candidate: minimal sufficient facts and canonical reference expansion.
MINIMAL_SYSTEM = '''你是本系统配置的AI保险电话助理，回答尊享e生百万医疗2026版的问题。
只凭证据，不输出思维过程，也不反思重写。输出JSON：
{"segments":[{"text":"直接对用户说的完整自然句", "source_ids":["直接支持这句话的证据ID"], "derivation":"计算才写简短公式与假设，否则空字符串"}],"unknowns":[{"question":"尚不能确认的问项","reason":"缺的资料或用户条件"}]}。
口播直接连接segments.text；每句的原文引用由程序按source_ids读取，禁止自己编造或改写引用。
先在内部选出本轮必须说的事实，再用2至4句写短答。只答用户问到的结论、金额以及改变该结论的必要条件。
不要加入用户未问的其他保额、其他责任、额外费率例子、销售追问；它们会引入无关错误。
一个句子含几个事实就引用支持这些事实的全部证据，不能仅有一个数字匹配就算有据。
基础与加油包、医疗报销与津贴、普通病与约定重疾、个人共享与家庭共享都按原文区分，不能移用其他责任条件。
条件不得缩掉一半：例如原文同时要求参保身份及就诊结算方式，必须保留两个前提；“有医保”不等于“已按有医保身份投保”。
如果本轮没有问比例，就不要主动补充未结算降低比例；如果主动说比例，必须带足改变比例的身份条件。
用户问费用时，按正确责任给简短公式。费用是否合规、医保/其他补偿、免赔已抵扣不明时，显式给“假设费用全符合、免赔未用、无其他补偿”的试算，不能说成确定赔付。
现有病史的承保资格与既往症免责分别回答；不能承诺承保、也不能靠“核保为准”漏答有原文的免责问题。
等待期只限制原文所指情形，不把疾病限制扩到意外；同样不能把原因免责扩大为某种身份的全面免责。
源中存在不同办理路径时区分路径；同一路径存在冲突就说明证据冲突需确认，不能挑一条当全部。
源不足只能明确说不足，仍回答其它有据问项；没有续保年龄、疾病定义、职业清单时不能用常识猜。
不要为了少字漏结论或必要条件，也不要为了显得全面抄所有证据。'''

BOUND_SYSTEM = MINIMAL_SYSTEM + '''
只生成用户本轮问到的责任、费用和结论，不扩写未问的其它责任与费率。
凡是回答附加保障的津贴、给付或赔付，明确“需另购”或“假设已购”，只说包名不够。
百分比句要完全保留适用身份条件；不要说“必须有社保才能按100%”或“否则60%”，原文并未把无社保身份投保一律归入60%。
不要把示例费用余额或免赔额算成负数给付；“最多能保几岁”不能仅用首次年龄回答，也不能用费率表最高年龄承诺续保。
价格证据已经按字段标注，年龄区间是闭区间，按用户给定年龄精确选中一行，不能取相邻区间。
若引用的疾病保障是约定重疾/特定疾病，必须说清需要满足合同定义；名称相似不能推断已满足定义。
对牙科、屈光不正等免责要保留原句意外例外，不能改成所有情形一概不赔。
同一证据句存在主条款限额与服务备注冲突时，明确有待核对；不能把任何一路径“不限额”冒充整个保险责任无上限。
自我身份与能力来自runtime证据，不来自保险合同，不能用保险条款证明是AI。'''

SELF_DESCRIPTION = '系统身份配置：本系统为AI语音保险助理，声音由语音合成生成，并非真人坐席。可根据所提供的保险资料回答问题，不能保证核保或理赔。'

SCOPED_SYSTEM = BOUND_SYSTEM + '''
本轮证据按整项责任打包。bundle中字段和单元格都是原始资料，字段不是额外承诺。
回答时以“责任—适用情形—条件—结论”为单位读取，不能只取百分比忽略同项限定。
用户描述普通疾病或肿瘤时，若相关责任存在合同定义分支，按是否满足定义给条件化答案，不默认为已满足或不满足。
用户已明确的病史是当前事实；直接回答当前事实下的核保边界与相关费用免责，不能用“如果不是既往症”绕开已有病史。
保险费用给付、服务办理和健康管理是不同路径；同项备注和主体条款的数字不一致时区分路径，并明确尚不能统一承诺。
没有证据列出完整集合时不能声称“只有”“所有”“一律”；不知道个案条件不等于不知道资料已经明确的责任。
把不同触发条件写成独立条件句，不用一个“否则”把所有不满足情形混为同一比例。
引用整项bundle可提供归属及上下文，但仍须给跨责任、免责、费率等其他所需来源；大量引用不能替代直接支持。
'''


def fewshot_system(version='v1') -> str:
    """Fixed synthetic demonstrations; never insert evaluation answers or evidence."""
    path = Path(__file__).with_name('prompts') / 'evidence_fewshot_v1.json'
    examples = json.loads(path.read_text(encoding='utf-8'))['examples']
    if version not in ('v1', 'v2', 'v3'):
        raise ValueError('Unsupported fewshot version')
    if version in ('v2', 'v3'):
        extras = path.with_name('evidence_fewshot_v2_additions.json')
        examples += json.loads(extras.read_text(encoding='utf-8'))
    prefix = '''【固定行为示例】
以下均为虚构产品，数字、疾病、年龄和DEMO/引用只属于示例，不能用于实际用户答案。
只迁移应用已知条件、计算、区分范围与短答的方法。实际回答只能依据用户消息里的本轮证据，source_ids只能取本轮证据ID。
示例有资料目录不表示本轮也有目录；没有目录时不要编造缺失证明，指出具体已读材料的局限。
'''
    result = prefix + json.dumps(examples, ensure_ascii=False, separators=(',', ':')) + '\n【实际任务规则】\n' + BOUND_SYSTEM
    if version == 'v3':
        result += '''
【本轮表达约束】
内部选事实后直接作答，不输出草稿、复核或反思。每个问项只给结论、必要分支和影响结论的条件；答完即停。
用户只问价格时只给该价格和适用口径；没有问是否承保，不追加职业、等待期、医院或续保说明。用户只问某原因会否排除另一无关费用时，只回答该因果边界，不追加完整报销条件清单。
每句只承载同一责任、同一办理路径的结论。多个责任分别用责任名或明确指代绑定条件；禁止在跨责任回答末尾用一条统一的免赔、医院、等待期或比例条件收尾。
把“默认给付规则＋某特定身份下的例外”保持为默认加例外，不能把例外的反面改成默认给付的必要前提。原文的“意外或等待期后疾病”保留或关系，不能写成所有情形均须等待。
有条件分支则只保留改变本题结论的分支；不通过增加无关责任、核保免责声明或反问来显得全面。资料缺失只就缺少的问项留白，引用本轮资料目录证明材料范围；引用目录不能证明真实合同没有该保障或限制。
用户现在已有某病、问未来购买后的该病费用，时间顺序已经给出；直接应用投保前病史规则。只有用户未说明的因果联系才需区分。
计算保留用户费用口径：总费用扣已补偿，医保后个人余额不重复扣医保，年度免赔仅扣尚未抵扣部分。计算句引用每个原始参数的证据。
费率行和费率适用说明共同决定报价；首次、未在指定期限重投、在指定期限重投是不同口径。不能把高年龄费率适用说明写成只有高年龄才能重投，也不能用未及时重投价格承诺及时重投价格。
'''
    return result


def handoff_system(version='v1') -> str:
    """One generation decides answer/clarify/handoff after reading the evidence."""
    if version not in ('v1','v2','v3','v4'):
        raise ValueError('Unsupported handoff prompt version')
    result = fewshot_system('v3') + '''
【后置转人工决策：本段补充实际输出格式】
读完证据后，在同一次生成中输出outcome字段，不进行另一轮自我反思。JSON结构增加：
"outcome":{"action":"answer或clarify或handoff","reason_kind":"none或missing_user_condition或missing_material或source_conflict或individual_review","reason":"简短说明","pending_questions":["需要追问或人工核实的本轮问项"],"source_ids":["支持缺失、冲突或个案审核边界的本轮证据ID"]}。
answer：本轮问项均能依据材料给正确结论或完整条件分支。已知既往症费用不赔、无关事故不因病史整体排除、意外无疾病等待期等有依据的问题，应直接回答，不能因提到疾病就转人工。
clarify：决定答案所缺的是用户可以补充的信息，例如年龄、医保投保身份、年度免赔已抵扣、实际职业或是否已购某包；只追问影响本题的未知条件。能用简短条件分支完整回答时可直接answer，不必追问。
handoff：用户所问结论依赖本轮未提供的关键附件、适用条款，或材料冲突无法统一，或需要个案核保/理赔裁定。不可用“可能”“核保为准”掩盖缺失后仍给确定承诺。合理转人工不算回答错误，但不能把所有问题都转交。
handoff必须指出具体待核实问项及原因，并引用支持边界的材料：缺附件引用materials:manifest，冲突引用双方原文，个案核保引用核保/告知条款。不能只用自评低置信度作为转交理由。缺用户条件优先clarify，不能冒充缺资料。
复合问题先答已被证据明确支持的部分，仅把无法确定的部分列为pending_questions。转交并不能免除已说事实的准确性：不得先给无依据金额/承诺再转人工。
segments仍为实际口播，必须与outcome一致。handoff用一句说明“需要转人工核实”，禁止说已接通、人工一定批准或可以获取本系统没有的材料。程序会记录转人工请求，不代表真实电话已接通。转交前保留问题、已知条件和原文出处给人工，不让用户重复描述。
资料缺失或冲突只在影响本轮结论时转人工。只问特药免赔时已有0免赔，不因药品完整清单缺失转交；只问能否承诺续保105岁可直接回答不能承诺，但问本人能否获准明年续保需转交核实。用户仅问本轮是否有附件时，可直接说明目录范围，不必转人工。
只输出JSON，source_ids均用本轮真实ID；不要输出审核评分、思维过程或多轮草稿。
'''
    if version in ('v2','v3','v4'):
        result += '''
【决策格式示例，全部虚构，只迁移格式】
示例A：用户问虚构责任免赔；证据DEMO/免赔写“无免赔额”。
{"segments":[{"text":"这项责任无免赔额。","source_ids":["DEMO/免赔"],"derivation":""}],"unknowns":[],"outcome":{"action":"answer","reason_kind":"none","reason":"本题已能回答","pending_questions":[],"source_ids":[]}}
示例B：用户询价但未提供年龄；缺的是用户条件。
{"segments":[{"text":"请问被保人几岁？","source_ids":["DEMO/年龄费率"],"derivation":""}],"unknowns":[{"question":"保费多少","reason":"缺年龄"}],"outcome":{"action":"clarify","reason_kind":"missing_user_condition","reason":"报价缺用户年龄","pending_questions":["被保人年龄"],"source_ids":[]}}
示例C：用户要求确认虚构药名能否报，资料目录DEMO/目录明确未提供完整药品附件。
{"segments":[{"text":"本轮没有完整药品附件，无法确认这个药名，需要转人工核实。","source_ids":["DEMO/目录"],"derivation":""}],"unknowns":[{"question":"指定药能否报","reason":"缺药品清单"}],"outcome":{"action":"handoff","reason_kind":"missing_material","reason":"缺本题必要的药品清单附件","pending_questions":["该药是否在约定清单内"],"source_ids":["DEMO/目录"]}}
实际任务只用本轮证据ID。outcome.action只选answer、clarify、handoff三者之一，不能写中文标签、多个动作或其他字符串。
answer的reason_kind固定none且pending_questions固定空列表：给了完整条件分支或明确否定不当承诺，就已回答，不把无关未知项列为本题待处理。
clarify的reason_kind固定missing_user_condition。handoff的reason_kind只选missing_material、source_conflict、individual_review；其reason、pending_questions、source_ids都不可为空。材料矛盾的source_ids必须列两处原始证据。
outcome.pending_questions只记录本题真正未完成的问项；不会因为某个未问附件缺失而新增待核实事项。unknowns与outcome不能相互矛盾。问句直接放在segments里，不能只写在元数据让用户听不到。
'''
    if version == 'v4':
        result += '''
【持续咨询待办：本段优先于前述单轮完成口径】
已知用户条件中的_consultation是程序保存的原咨询和待办。本轮用户可能只补年龄、医保或更正条件，必须继续原咨询的全部未完成问项；不能只回答最新短句或转去销售。更正后，原问中的旧年龄/医保不能覆盖最新条件，年/月价和差额等所有受影响问项都须更新。
原咨询尚未完成又插入新问题时保留旧待办并追加新项。用同一task id继续同一问项，不得删除旧id；用户明确取消才标cancelled并附cancel_user_quote。已经正确答完且未受更正影响的问项不必重复口播。
输入里seeded=true的任务由独立小模型拆出的当前问项形成，每一个都必须保留原id并实际处理；不可把三项合并成两项后丢掉第三个id。多个任务可共用答句，但不能以共用为由删任务。
每个segments增加task_ids数组，列这个口播段实际回答或追问的待办id。保险结论必须能对应任务，不把未确认的金额计算与已确认的报价/保障答句混在同一段。
原咨询仍缺条件时，也要先回答新加入的、条件齐全的问项；只追问真正依赖缺项的任务，不能让旧例子的未知免赔堵住无关的报价、差额、等待期或免责问题。outcome用clarify保留旧任务，并在segments实际说完已能回答的部分。不要重复追问用户已经明确说明的结算行为。
输入待办的needs_answer表示上次内容没有向用户发布，不是已经答完；即使条件都已齐全也必须实际口播该问项的结论，再标answered。程序可能为避免先说错误金额而暂缓其他答句，不能凭上次原始输出宣布这些问项完成。
额外输出consultation：{"facts":[{"name":"age","value":62,"state":"confirmed或missing或assumed","user_quote":"用户实际说过的连续原话"}],"tasks":[{"id":"q1","question":"用户原本问的年缴保费","kind":"price或price_difference或discount或eligibility或claim_calculation或coverage或other","status":"answered或needs_user或needs_review或cancelled","requires":["age","has_social"],"is_example":false}]}。
facts是各待办依赖的用户条件，不是保险条款：confirmed须用户明确提供，quote逐字来自用户原话，不能引用坐席旧回答、销售默认值或常识。missing的value用null；assumed只用于用户明确要求的假设示例，不能写回用户画像。同名条件保留最新值，不能把旧报价当新条件。
更换被保人后不得继承上一人的医保、免赔等个人条件；按最新对象判断年龄资格和报价。原对象已不买且咨询已完成时不重复原对象结论，只处理当前_consultation的新对象问项。
常用条件name：age、has_social（报价医保身份）、insured_with_social（理赔时保单医保投保身份）、social_settled（本次医保结算）、remaining_deductible、family_size、first_insurance、plan、payment；其它条件可按原问自定义。没走医保只描述就诊结算，不能证明insured_with_social；医保报销金额不能证明年度免赔用了多少。各计算任务的依赖按实际责任确定。
【按责任确定依赖】
一般医疗/基础门急诊的实际费用测算才需要医保投保身份、剩余年免赔；住院津贴/ICU按天给付、药省保月/年限额、重疾定额给付不依赖医保身份或一般医疗年免赔，不能追问这些无关条件。门急诊加油包、住院补偿包等0免赔责任也不能套用一般医疗的1万元免赔，只问改变该责任金额的条件。
每个任务的source_ids列对应口播已经引用的证据ID，不能借其它问项的来源判断本项。facts必须列出每个requires的条件，不能在口播已使用而后台漏记。用户说“年免赔没用/未使用”已能结合责任原文确认剩余额；已抵扣金额要换算剩额，不得重复追问已经说明的条件。一家三口等明确人数要保留为条件，不能因为没写阿拉伯数字就忽略。
用户已明确按医保结算或医保已报销时，记录social_settled=true，原文的“有医保身份参保但未按医保结算”例外已被排除，不再为这个比例追问保单身份。用户未走医保结算时才需确认医保投保身份。原问给了金额、剩余免赔等条件，必须在facts记录，不得明明用其计算却省略后台条件。
费率按实际列决定依赖：重疾关爱取年龄和性别；其它加油包表中“有无社保”的同一价格列不依赖医保身份；基础必选及家庭共享免赔费率才区分医保。用户仅问加油包价格时不要追加基础版或问无关医保。
confirmed原话须包含该条件的实际依据，不能拿没有提医保的整句就宣称有医保。依赖缺条件时给中文自然问句，不能把occupation等字段名读给用户。
实际报价的price和price_difference必须列age、has_social依赖，不能因为给了两种医保价格就把用户实际询价记为完成。已有依据明确不能投保时，价格任务quoted_amount用null，直接说明无法按该口径报价，不再收集无关医保、缴费条件。首次/重投、计划等费率口径未明时也须澄清，不能擅设为首次必选个人单。只问规则的任务使用other或coverage，不要误标为实际价格任务。discount代表具体适用优惠，必须确认family_size；只介绍完整优惠档位不能替代用户具体优惠。
只问年龄是否符合时，只核对年龄范围；年龄已经不符合，无须继续追问职业/医保。用户问整体承保时，年龄符合不代表核保通过。原问有能否投保问项时，口播必须明确回答年龄条件是否符合及必要审核边界，不能只报价或在任务字段标answered。
用户问自身能赔多少、实际价格多少时，缺少改变金额的条件须needs_user/clarify并问出来，不能只选一个假设后宣布解决；“测算”不等于授权猜条件。用户问公式/一般规则/明确假设时，可给完整分支或示例，is_example须匹配用户原话。问有无优惠可先说规则，但要给用户具体优惠须确认适用人数与口径。
金额条件缺失时直接追问缺项，不先用默认免赔额/医保身份给数值示例。已经明确的用户条件不得重复问。具体保费已报价的price任务增加quoted_amount数值字段；不能投保或无法给具体价格时用null，不可拿年龄、额度或免赔额填这个字段。
requires所需事实未确认的任务不能标answered。needs_user任务的缺项放进outcome.pending_questions，实际问句必须在segments中。需要审核/缺资料的任务保留needs_review；如果其它问项还缺用户条件，先clarify补齐可以自动回答的部分，再转交待核项。资料缺口和用户缺条件不能混为一谈。
只有所有问项确实已答或已按用户明确要求取消，才能outcome=answer。转交须包含全部待核问项和尚未完成的原问。kind=price代表具体保费已答，不能用于只有年龄资格或含糊价格介绍的任务。不要追加投保链接、重复报价或通用销售话术。
'''
    if version == 'v3':
        result += '''
【口语短答与后台溯源分离】
segments.text是给用户听的口语概括，可以改写、合并、省略与本题无关的原文细节，不要求复述引用原话。source_ids是后台出处索引；程序另存原文和定位，不把这些原文读给用户。
以本轮整条回答为溯源单位：每个实际说出的保险事实都要能由本轮所列原文及已知条件支持，不要求每个短句重复贴同一出处。引用中有额外内容，不意味着口播必须把那些内容都说出来。
先用一句回答用户最关心的结论，再用短分句回答用户明确问到的各项。单问尽量30至60字，复合问题尽量80至150字；这是表达目标，不能为了字数删掉会改变结论的前提。
每个责任只说本题需要的结论和边界；不主动列保额、床位限额、其它治疗项目、服务流程或法条解释。用户问医院时直接说该责任的医院/药店门槛，不绕成完整条款介绍。
用通俗词概括责任名，保留准确指代；避免反复说“按某某费用医疗保险金”“在保险期间内”“按合同约定”。共性条件只有在每项证据都支持同样条件时才合并说明。
不重复结论和同一条件，不用解释原文如何写来代替回答。需要个案核实就一句说明边界，不再重复整段免责分析。
条件的重要性由本题决定：删除后会让不符合条件的人误以为可赔、无免赔或拿到某个价格的条件必须保留；不会改变本题结论的细节留给追问。
例如院外特药能否报，药品必须在约定清单内是必要前提；只问该责任免赔额是多少时可以只答免赔额，不列完整购药手续。仅引用清单条文不能使缺失的清单前提自动成为用户听到的条件。
'''
    return result


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    source: str
    path: str
    sha256: str
    location: dict = field(default_factory=dict)


def from_chunk(c: Chunk) -> Evidence:
    path = 'data/clean/保险责任.json' if c.doc == '保障计划表' else (
        'data/clean/text/尊享E生百万医疗2026投保须知.md' if c.doc == '投保须知'
        else 'data/clean/text/尊享E生百万医疗2026责任免除.md')
    location = c.meta.get('location', {})
    if c.doc == '保障计划表':
        iid, cell = c.chunk_id.split(':',1)
        location = excel_location(iid, cell if cell!='head' else None)
    return Evidence(c.chunk_id, c.text, c.source, path, hashlib.sha256(c.text.encode()).hexdigest(), location)


def full_evidence(labelled=False) -> list[Evidence]:
    # Original table cells, including metadata needed to distinguish mandatory/add-on plans.
    chunks = []
    for item in coverage_items():
        chunks.extend(item_chunks(item['id']))
    chunks.extend(c for c in clause_chunks() if c.doc != '保障计划表')
    out = [from_chunk(c) for c in chunks]
    # All rate rows are available; no test answer or question-specific oracle enters the pack.
    for path in sorted((CLEAN / 'premium').glob('*.csv')):
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        for i, line in enumerate(lines[1:], 2):
            content = f'{path.stem}；列头：{lines[0]}；本行：{line}'
            if labelled:
                header = next(csv.reader([lines[0]]))
                values = next(csv.reader([line]))
                content = path.stem + '；' + '；'.join(f'{key}={value}' for key,value in zip(header,values))
            out.append(Evidence(f'rate:{path.stem}:{i}', content, f'{path.name} 第{i}行',
                                path.relative_to(CLEAN.parent.parent).as_posix(), hashlib.sha256(content.encode()).hexdigest()))
    # Fee table prose contains family discounts; CSV alone does not contain them.
    for path in sorted((CLEAN / 'text').glob('*费率.md')):
        locations = pdf_locations(path.name)
        for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if line.strip() and not line.startswith('<!--'):
                out.append(Evidence(f'rate-notice:{path.stem}:{i}', line, f'{path.stem} 第{i}行',
                                    path.relative_to(CLEAN.parent.parent).as_posix(), hashlib.sha256(line.encode()).hexdigest(),
                                    locations.get(i,{})))
    if labelled:
        out.append(Evidence('runtime:identity',SELF_DESCRIPTION,'系统身份配置',
                            'src/agent/evidence_answer.py',hashlib.sha256(SELF_DESCRIPTION.encode()).hexdigest()))
    return list({e.id: e for e in out}.values())


def grouped_evidence() -> list[Evidence]:
    """Lossless responsibility bundles; no inferred conditions or test-specific facts.

    Preserve parent metadata alongside all original D cells. Bundle members retain
    the old IDs so both experiments can be traced to the same raw table locations.
    """
    original = full_evidence(labelled=True)
    out = []
    for item in coverage_items():
        members = [e for e in original if e.id.startswith(item['id'] + ':')]
        text = json.dumps({'责任原始字段': item,
                           '原始片段': [{'id':e.id,'位置':e.source,'定位':model_location(e.location),'原文':e.text} for e in members]},
                          ensure_ascii=False, separators=(',', ':'))
        out.append(Evidence(item['id'] + ':bundle', text,
                            f"保障计划表 原始行{item['source_rows']}，含合并字段与全部责任描述",
                            'data/clean/保险责任.json', hashlib.sha256(text.encode()).hexdigest(), excel_location(item['id'])))
    out.extend(e for e in original if not e.id.startswith(tuple(i['id'] + ':' for i in coverage_items())))
    return out


def sparse_evidence(retrieved: list[Chunk], full: list[Evidence]) -> list[Evidence]:
    # Identical old direct candidate responsibility selection, expanded to full item cells.
    selected = list(dict.fromkeys(c.meta['item'] for c in retrieved if c.meta.get('item')))[:5]
    chunks = [c for iid in selected for c in item_chunks(iid)]
    chunks.extend(retrieved)
    # Keep all rate evidence common to both treatments, so sparse coverage is the manipulated variable.
    out = [from_chunk(c) for c in chunks] + [e for e in full if e.id.startswith('rate')]
    return list({e.id:e for e in out}.values())


def model_location(location: dict) -> dict:
    # File integrity hashes stay in the trace; the model only needs navigation fields.
    keys = ('raw_path','clean_path','page','page_text_line_start','page_text_line_end',
            'md_line','sheet','cell','row_start','row_end')
    return {key:location[key] for key in keys if key in location and location[key] is not None}


def evidence_text(evidence: list[Evidence]) -> str:
    return '\n'.join(f'[{e.id}] ({e.source})'
                     + (f" 定位={json.dumps(model_location(e.location),ensure_ascii=False,separators=(',',':'))}" if e.location else '')
                     + f' {e.text}' for e in evidence)


def source_registry(evidence: list[Evidence]) -> dict[str, Evidence]:
    registry = {e.id:e for e in evidence}
    for bundle in evidence:
        if not bundle.id.endswith(':bundle'):
            continue
        # Nested original cell IDs are declared in the evidence pack. Resolving
        # them must retain their parent responsibility, not lose its conditions.
        for member in json.loads(bundle.text)['原始片段']:
            sid = member['id']
            registry[sid] = Evidence(sid, bundle.text,
                                    member['位置'] + '；完整所属责任上下文',
                                    bundle.path, bundle.sha256, {**bundle.location,**member.get('定位',{})})
    return registry


def citation_check(answer: dict, evidence: list[Evidence], scope: str = 'segment') -> dict:
    """Mechanical provenance integrity, NEVER semantic entailment or correctness."""
    by_id = source_registry(evidence)
    faults, refs = [], 0
    for i, claim in enumerate(answer.get('claims', [])):
        if not claim.get('text') or claim['text'] not in answer.get('answer', ''):
            faults.append(f'claim[{i}] not an answer span')
        if scope == 'segment' and not claim.get('citations'):
            faults.append(f'claim[{i}] has no citations')
        for ref in claim.get('citations', []):
            refs += 1
            e = by_id.get(ref.get('id'))
            if e is None or not ref.get('quote') or ref['quote'] not in e.text:
                faults.append(f'claim[{i}] invalid reference or nonliteral quote')
    if not answer.get('claims'):
        faults.append('no claims')
    if scope == 'answer' and refs == 0:
        faults.append('answer has no citations')
    return {'valid': not faults, 'faults': faults, 'references': refs,
            'scope': scope, 'semantic_entailment_checked': False}


class EvidenceAnswerer:
    def __init__(self, client=None):
        self.client = client or cloud.llm_client()

    def call(self, system: str, user: str, model: str) -> dict:
        t0 = time.perf_counter()
        # The history's JSON string is the only replaceable span in this prompt.
        history = getattr(user, 'history', '')
        messages = with_history([{'role':'system','content':system}, {'role':'user','content':user}],
                                history, encoded=True, start=getattr(user, 'history_start', None))
        raw = cloud.chat(messages,
                         model=model, client=self.client, temperature=0,
                         response_format={'type':'json_object'}, max_tokens=2800)
        parsed = json.loads(raw)
        return {'model':model,'output':parsed,'seconds':time.perf_counter()-t0,
                'context_events':getattr(messages, 'context_events', [])}

    def answer(self, question: str, known: dict, evidence: list[Evidence], debate=False, minimal=False, bound=False, scoped=False, fewshot=False, fewshot_version='v1', allow_handoff=False, handoff_version='v1') -> dict:
        t0 = time.perf_counter()
        prefix = f'原问题：{question}\n已知用户条件：'
        known_json = json.dumps(known,ensure_ascii=False)
        history = known.get('_history', '')
        start = len(prefix)+known_json.index('"_history": ')+len('"_history": ') if history else None
        user = HistoryPrompt(prefix+known_json+'\n【全部本轮证据】\n'+evidence_text(evidence),history,start)
        calls = []
        if allow_handoff and debate:
            raise ValueError('Post-answer handoff uses one generation, not debate')
        if not debate:
            final = self.call(handoff_system(handoff_version) if allow_handoff else fewshot_system(fewshot_version) if fewshot else SCOPED_SYSTEM if scoped else BOUND_SYSTEM if bound else MINIMAL_SYSTEM if minimal else SYSTEM, user, cloud.SLOW_MODEL)
            calls.append(final)
        else:
            models = [cloud.SLOW_MODEL,cloud.QWEN_ANSWER_MODEL]
            roles = ['核对责任边界、基础与附加以及未知条件','核对金额、免赔合并、资格和改变结论的条件']
            # Barrier: both originals committed before either model sees another output.
            with ThreadPoolExecutor(2) as pool:
                jobs = [pool.submit(self.call,INDEPENDENT % role,user,model) for role,model in zip(roles,models)]
                originals = [f.result() for f in jobs]
            with ThreadPoolExecutor(2) as pool:
                jobs = [pool.submit(self.call,CHALLENGE,
                                    user+'\n自己的已提交初稿：'+json.dumps(originals[i]['output'],ensure_ascii=False)+
                                    '\n匿名对方初稿：'+json.dumps(originals[1-i]['output'],ensure_ascii=False), model)
                        for i,model in enumerate(models)]
                challenges = [f.result() for f in jobs]
            packets = [{'draft':o['output'],'challenge':c['output']} for o,c in zip(originals,challenges)]
            random.Random(hashlib.sha256(question.encode()).hexdigest()).shuffle(packets)
            final = self.call(ARBITER,user+'\n【匿名独立意见，无模型标识】\n'+json.dumps(packets,ensure_ascii=False),cloud.SLOW_MODEL)
            calls.extend(originals+challenges+[final])
        # Rendering only: the spoken facts and their cited segments are identical by construction.
        structured = {**final['output'], 'claims':final['output'].get('segments', [])}
        if minimal or bound or scoped or fewshot or allow_handoff:
            registry = source_registry(evidence)
            structured['claims'] = [{**s, 'citations':[
                {'id':sid,'quote':registry[sid].text if sid in registry else '',
                 'location':registry[sid].location if sid in registry else {}}
                for sid in s.get('source_ids', [])]} for s in structured['claims']]
        structured['answer'] = ''.join(s['text'] for s in structured['claims'])
        scope = 'answer' if allow_handoff and handoff_version == 'v3' else 'segment'
        structured['answer_source_ids'] = list(dict.fromkeys(
            ref['id'] for claim in structured['claims'] for ref in claim.get('citations', [])))
        result = {'text':structured['answer'], 'structured':structured,
                'provenance':citation_check(structured,evidence,scope=scope), 'calls':calls,
                'total_s':time.perf_counter()-t0, 'evidence_ids':[e.id for e in evidence]}
        if allow_handoff:
            from agent.handoff import validate_outcome, handoff_request
            result['outcome'] = final['output'].get('outcome')
            result['handoff_request'] = None
            try:
                result['outcome'] = validate_outcome(result['outcome'], evidence)
                result['outcome_validation'] = {'valid':True,'fault':None}
                result['handoff_request'] = handoff_request(question, known, result)
            except ValueError as exc:
                # Preserve the raw answer/decision for audit, never repair it or
                # silently count an invalid declaration as a human transfer.
                result['outcome_validation'] = {'valid':False,'fault':str(exc)}
        return result
