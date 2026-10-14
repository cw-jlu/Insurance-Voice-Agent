"""Verify normalized source and data bytes against the frozen core manifest."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def verify():
    manifest=json.loads((ROOT/'releases/core-2026-10-14.json').read_text(encoding='utf-8'))
    faults=[]
    for name,expected in manifest['files'].items():
        path=ROOT/name
        digest=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest() if path.exists() else None
        if digest!=expected:faults.append(name)
    return faults


if __name__=='__main__':
    faults=verify()
    print('Frozen core verified.' if not faults else 'Frozen files changed: '+', '.join(faults))
    sys.exit(bool(faults))
