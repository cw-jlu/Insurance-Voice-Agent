"""Preserve source validation across Git line endings and Windows source paths.

The frozen index owns expected hashes. Only newline conversion may match an
alternative hash; changed words or changed binary sources still fail validation.
"""
import hashlib
import json
from pathlib import Path


def source_digest(path,expected_by_path):
    normalized=Path(str(path).replace('\\','/'))
    content=normalized.read_bytes()
    actual=hashlib.sha256(content).hexdigest()
    expected=expected_by_path.get(normalized.resolve().as_posix())
    if expected and actual!=expected and normalized.suffix.lower() in ('.md','.txt'):
        lf=content.replace(b'\r\n',b'\n')
        for candidate in (lf,lf.replace(b'\n',b'\r\n')):
            digest=hashlib.sha256(candidate).hexdigest()
            if digest==expected:return digest
    return actual


def install_source_compat(root):
    from agent import source_locations
    index=json.loads((root/'data/clean/locations/pdf_source_index.json').read_text(encoding='utf-8'))
    expected={}
    for doc in index['documents'].values():
        for pk,hk in (('raw_path','raw_sha256'),('clean_path','clean_sha256'),('extracted_path','extracted_sha256')):
            path=(root/doc[pk].replace('\\','/')).resolve().as_posix()
            expected[path]=doc[hk]
    source_locations.digest=lambda path:source_digest(path,expected)
    source_locations.pdf_locations.cache_clear()
    source_locations.excel_source.cache_clear()
