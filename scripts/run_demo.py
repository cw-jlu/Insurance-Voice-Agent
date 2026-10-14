"""One-process launcher. No installs or credential prompts at startup."""
import argparse
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Start the local insurance consultation demo')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--native-https',action='store_true')
    args=parser.parse_args()
    if args.native_https:os.environ['DEMO_NATIVE_HTTPS']='1'
    import uvicorn
    print(f'Open http://{args.host}:{args.port}  |  Ctrl+C to stop',flush=True)
    uvicorn.run('demo.app:app',host=args.host,port=args.port,workers=1,ws_max_size=256_000)
