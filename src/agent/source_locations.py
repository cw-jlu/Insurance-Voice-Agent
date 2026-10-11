"""Lossless links from cleaned PDF paragraphs to extracted page lines."""
from __future__ import annotations
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / 'data/clean/locations/pdf_source_index.json'
PAGE = re.compile(r'<<<PAGE (\d+)>>>')
MD_PAGE = re.compile(r'<!--\s*page\s+(\d+)\s*-->')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build_index(root=ROOT):
    documents = {}
    for md in sorted((root/'data/clean/text').glob('*.md')):
        extracted = root/'data/extracted/pdf_text'/f'{md.stem}.pdfplumber.txt'
        if not extracted.exists():
            continue
        lines = md.read_text(encoding='utf-8').splitlines()
        header = re.match(r'<!-- source: (.*); sha256=([0-9a-f]+) -->', lines[0])
        if not header:
            raise ValueError(f'Missing original source header: {md}')
        raw_path, raw_hash = header.groups()
        if digest(root/raw_path) != raw_hash:
            raise ValueError(f'Original PDF changed: {raw_path}')
        pages = {}
        page, local_line = None, 0
        for global_line, text in enumerate(extracted.read_text(encoding='utf-8').splitlines(), 1):
            marker = PAGE.fullmatch(text)
            if marker:
                page, local_line = int(marker[1]), 0
                pages[page] = {'chars':[], 'local_lines':[], 'file_lines':[]}
                continue
            if page is None:
                raise ValueError('Text before page marker')
            local_line += 1
            for char in text:
                if not char.isspace():
                    pages[page]['chars'].append(char)
                    pages[page]['local_lines'].append(local_line)
                    pages[page]['file_lines'].append(global_line)
        offsets = {p:0 for p in pages}
        records, page = [], None
        for md_line, text in enumerate(lines, 1):
            marker = MD_PAGE.search(text)
            if marker:
                page = int(marker[1])
                continue
            if not text.strip() or text.startswith('<!--'):
                continue
            compact = ''.join(c for c in text if not c.isspace())
            start = offsets[page]
            end = start + len(compact)
            source = pages[page]
            if compact != ''.join(source['chars'][start:end]):
                raise ValueError(f'Content mismatch at {md.name}:{md_line}')
            records.append({'id':f'{md.stem}:p{page}:md{md_line}', 'text':text.strip(),
                'page':page, 'md_line':md_line,
                'page_text_line_start':source['local_lines'][start],
                'page_text_line_end':source['local_lines'][end-1],
                'extracted_file_line_start':source['file_lines'][start],
                'extracted_file_line_end':source['file_lines'][end-1],
                'non_whitespace_char_start':start,'non_whitespace_char_end_exclusive':end})
            offsets[page] = end
        if any(offsets[p] != len(v['chars']) for p,v in pages.items()):
            raise ValueError(f'Missing page content: {md}')
        documents[md.name] = {'raw_path':raw_path,'raw_sha256':raw_hash,
            'clean_path':md.relative_to(root).as_posix(),'clean_sha256':digest(md),
            'extracted_path':extracted.relative_to(root).as_posix(),
            'extracted_sha256':digest(extracted),'records':records}
    return {'schema_version':1, 'line_convention':'1-based lines of pdfplumber extracted page text, not printed PDF line numbers; no pixel coordinates',
            'documents':documents}

@lru_cache(maxsize=None)
def pdf_locations(filename):
    if not INDEX.exists():
        return {}
    doc = json.loads(INDEX.read_text(encoding='utf-8'))['documents'][filename]
    for path_key, hash_key in [('raw_path','raw_sha256'),('clean_path','clean_sha256'),('extracted_path','extracted_sha256')]:
        if digest(ROOT/doc[path_key]) != doc[hash_key]:
            raise ValueError(f'Stale source location index: {doc[path_key]}; rebuild index')
    common = {k:v for k,v in doc.items() if k!='records'}
    return {r['md_line']:{**common,**{k:v for k,v in r.items() if k!='text'}} for r in doc['records']}

@lru_cache(maxsize=1)
def excel_source():
    doc=json.loads((ROOT/'data/clean/保险责任.json').read_text(encoding='utf-8'))
    extracted=json.loads((ROOT/'data/extracted/xlsx_cells.json').read_text(encoding='utf-8'))
    if digest(ROOT/doc['source'])!=doc['sha256']:
        raise ValueError('Original responsibility workbook changed; re-extract before citing')
    return doc, extracted['sheet']

def excel_location(item_id, cell=None):
    doc,sheet=excel_source()
    item=next(i for i in doc['items'] if i['id']==item_id)
    return {'raw_path':doc['source'].replace('\\','/'),'raw_sha256':doc['sha256'],'sheet':sheet,
            'row_start':item['source_rows'][0],'row_end':item['source_rows'][1],
            'cell':cell, 'item_id':item_id}
