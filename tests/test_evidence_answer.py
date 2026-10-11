import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent.evidence_answer import Evidence, EvidenceAnswerer, citation_check, grouped_evidence, source_registry


class EvidenceAnswerTest(unittest.TestCase):
    def test_original_cell_reference_keeps_parent_responsibility_context(self):
        evidence = grouped_evidence()
        registry = source_registry(evidence)
        ref = registry['R15:D118']
        self.assertEqual(ref.text, registry['R15:bundle'].text)
        self.assertIn('尊享住院津贴加油包', ref.text)
        self.assertIn('次免赔天数为0天', ref.text)
        self.assertIn('D117', ref.text)
        answer = {'answer':'ICU津贴另购。', 'claims':[{
            'text':'ICU津贴另购。', 'citations':[{'id':ref.id,'quote':ref.text}]}]}
        self.assertTrue(citation_check(answer, evidence)['valid'])

    def test_fabricated_or_nonliteral_citation_is_reported_without_repair(self):
        evidence=[Evidence('R01:D8','扣除1万元年免赔额后，按100%赔付','D8','source.json','hash')]
        answer={'answer':'扣除免赔额后赔付。','claims':[{'text':'扣除免赔额后赔付。',
                'citations':[{'id':'R01:D8','quote':'没有免赔额'}]}]}
        check=citation_check(answer,evidence)
        self.assertFalse(check['valid'])
        self.assertFalse(check['semantic_entailment_checked'])
        self.assertEqual(answer['claims'][0]['citations'][0]['quote'],'没有免赔额')

    def test_both_blind_drafts_committed_before_challenge_and_arbitration_is_anonymous(self):
        answerer=EvidenceAnswerer(client=object())
        lock=threading.Lock()
        committed=[]
        final_inputs=[]
        def fake_call(system,user,model):
            if '你看不到任何其他分析员' in system:
                self.assertNotIn('对方初稿',user)
                with lock:
                    committed.append(model)
                output={'segments':[{'text':'需扣免赔额。','citations':[{'id':'d','quote':'扣免赔额'}]}]}
            elif '只能做一次证据质询' in system:
                self.assertEqual(len(committed),2)
                output={'issues':[]}
            else:
                self.assertEqual(len(committed),2)
                final_inputs.append(user)
                output={'segments':[{'text':'需扣免赔额。','citations':[{'id':'d','quote':'扣免赔额'}]}]}
            return {'model':model,'output':output,'seconds':0}
        with patch.object(answerer,'call',side_effect=fake_call):
            result=answerer.answer('如何报销',{},[Evidence('d','扣免赔额','s','p','h')],debate=True)
        self.assertEqual(len(result['calls']),5)
        self.assertTrue(result['provenance']['valid'])
        self.assertEqual(result['text'],'需扣免赔额。')
        anonymous=final_inputs[0].split('【匿名独立意见，无模型标识】')[1]
        for model in committed:
            self.assertNotIn(model,anonymous)


if __name__=='__main__':
    unittest.main()
