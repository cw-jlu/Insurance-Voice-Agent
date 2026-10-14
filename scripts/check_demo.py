"""Print configuration presence and optional connectivity, never credential values."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true',help='Send a small text connectivity request')
    parser.add_argument('--native-https',action='store_true')
    args=parser.parse_args()
    if args.native_https:os.environ['DEMO_NATIVE_HTTPS']='1'
    from demo.app import capabilities
    from demo.service import DemoStore
    caps=capabilities()
    report={'capabilities':caps,'missing':[],'credentials_printed':False}
    for service in ('deepseek','asr','cloud_tts'):
        if not caps[service]:report['missing'].append(service)
    if args.live:
        if not caps['deepseek']:
            report['live_text']={'ok':False,'reason':'DeepSeek configuration missing'}
        else:
            try:
                import time
                store=DemoStore();session=store.create('live','deepseek')
                t=time.perf_counter()
                result=session.pipe.client.chat.completions.create(model=session.pipe.answerer.model,
                     messages=[{'role':'user','content':'仅回复：连接正常'}],max_tokens=32)
                report['live_text']={'ok':bool(result.choices[0].message.content),
                                     'model':session.pipe.answerer.model,'elapsed_s':round(time.perf_counter()-t,3)}
            except Exception as exc:
                report['live_text']={'ok':False,'error_type':type(exc).__name__}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    sys.exit(1 if args.live and not report.get('live_text',{}).get('ok') else 0)
