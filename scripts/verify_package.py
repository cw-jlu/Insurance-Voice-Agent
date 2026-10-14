"""Verify the delivered archive after extraction, using Python's standard library."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    manifest_path=ROOT/'MANIFEST.json'
    if not manifest_path.exists():
        print('Run this script inside the extracted submission package.');sys.exit(1)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    faults=[]
    for name,expected in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():faults.append(name);continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:faults.append(name)
    print('Package verified: '+str(len(manifest['files']))+' files.' if not faults else 'Changed/missing files: '+', '.join(faults))
    sys.exit(bool(faults))
