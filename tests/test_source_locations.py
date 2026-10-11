import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent import source_locations as loc
from agent.knowledge import _text_chunks
from agent.evidence_answer import EvidenceAnswerer, full_evidence, grouped_evidence, source_registry

class SourceLocationsTest(unittest.TestCase):
    def test_each_paragraph_resolves_to_original_page_and_extracted_lines(self):
        index = loc.build_index()
        self.assertEqual(len(index['documents']),4)
        count=0
        for filename,doc in index['documents'].items():
            md=(loc.ROOT/doc['clean_path']).read_text(encoding='utf-8').splitlines()
            raw=(loc.ROOT/doc['extracted_path']).read_text(encoding='utf-8').splitlines()
            for r in doc['records']:
                self.assertEqual(md[r['md_line']-1].strip(),r['text'])
                window=''.join(raw[r['extracted_file_line_start']-1:r['extracted_file_line_end']])
                compact=lambda s:''.join(c for c in s if not c.isspace())
                self.assertIn(compact(r['text']),compact(window))
                self.assertGreaterEqual(r['page_text_line_start'],1)
                count+=1
        self.assertEqual(count,248)

    def test_stale_original_is_rejected_instead_of_citing_old_page_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'raw.pdf').write_bytes(b'new version')
            index=root/'index.json'
            index.write_text(json.dumps({'documents':{'x.md':{
                'raw_path':'raw.pdf','raw_sha256':'old hash','records':[]}}}),encoding='utf-8')
            loc.pdf_locations.cache_clear()
            with patch.object(loc,'ROOT',root),patch.object(loc,'INDEX',index):
                with self.assertRaisesRegex(ValueError,'Stale'):
                    loc.pdf_locations('x.md')
            loc.pdf_locations.cache_clear()

    def test_existing_ids_and_final_citations_keep_locators(self):
        chunks=_text_chunks('尊享E生百万医疗2026投保须知.md','投保须知')
        self.assertEqual(chunks[5].chunk_id,'投保须知#5')
        self.assertIn('60%',chunks[5].text)
        self.assertEqual(chunks[5].meta['location']['page'],1)
        evidence=full_evidence(labelled=True)
        by_id={e.id:e for e in evidence}
        answerer=EvidenceAnswerer(client=object())
        output={'segments':[{'text':'按条款条件赔付。','source_ids':['投保须知#5'],'derivation':''}]}
        with patch.object(answerer,'call',return_value={'output':output}):
            answer=answerer.answer('如何赔付',{},evidence,bound=True)
        self.assertEqual(answer['structured']['claims'][0]['citations'][0]['location'],by_id['投保须知#5'].location)
        cell=source_registry(grouped_evidence())['R15:D118'].location
        self.assertEqual(cell['cell'],'D118')
        self.assertIn('raw_sha256',cell)
        self.assertTrue(cell['sheet'])

if __name__=='__main__':unittest.main()
