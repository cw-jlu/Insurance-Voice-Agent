"""Build page/line provenance without changing source text or chunk IDs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent.source_locations import INDEX,build_index

if __name__=='__main__':
    index=build_index()
    INDEX.parent.mkdir(parents=True,exist_ok=True)
    INDEX.write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"{len(index['documents'])} PDFs; {sum(len(d['records'])for d in index['documents'].values())} located paragraphs; {INDEX}")
