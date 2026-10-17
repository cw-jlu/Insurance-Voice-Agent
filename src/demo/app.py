"""Self-hosted browser demo; credentials remain on the Python server."""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parents[2]
for name,value in dotenv_values(ROOT/'.env.demo').items():
    if value:
        os.environ.setdefault(name,value)

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from demo.service import CORE_TAG, DEMO_TAG, SCENARIOS, DemoStore, capabilities
from agent import cloud


class CreateSession(BaseModel):
    mode: str = Field(default='replay',pattern='^(live|replay)$')
    provider: str = Field(default='deepseek',pattern='^(deepseek|qwen)$')
    scenario: str = Field(default='consultation',pattern='^(consultation|outbound)$')


def create_app(store=None,prewarm=False):
    store=store or DemoStore()

    @asynccontextmanager
    async def lifespan(app):
        warm_task=(asyncio.create_task(asyncio.to_thread(store.warm_speech))
                   if prewarm and capabilities()['cloud_tts'] else None)
        yield
        if warm_task is not None:
            warm_task.cancel()
            await asyncio.gather(warm_task,return_exceptions=True)
        await asyncio.gather(*(s.close() for s in store.sessions.values()),return_exceptions=True)

    app=FastAPI(title='保险复杂咨询演示台',lifespan=lifespan)
    app.state.store=store

    @app.middleware('http')
    async def security_headers(request,call_next):
        # Mutations are same-origin. Loopback is the default deployment boundary.
        origin=request.headers.get('origin')
        if request.method not in ('GET','HEAD','OPTIONS') and origin:
            from urllib.parse import urlsplit
            if urlsplit(origin).netloc!=request.headers.get('host'):
                from fastapi.responses import JSONResponse
                return JSONResponse({'detail':'仅允许同源请求'},status_code=403)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob: data:; connect-src 'self' ws: wss:; frame-ancestors 'none'"
        return response

    @app.get('/')
    def index():return FileResponse(ROOT/'web/index.html')

    @app.get('/api/health')
    def health():return {'ok':True,'capabilities':capabilities(),'core_tag':CORE_TAG,'demo_tag':DEMO_TAG,
                         'speech_cache':{'status':store.speech_cache.status,'error':store.speech_cache.error}}

    @app.get('/api/scenarios')
    def scenarios():return SCENARIOS

    @app.post('/api/sessions',status_code=201)
    def create_session(body:CreateSession):
        try:return store.create(body.mode,body.provider,body.scenario).snapshot()
        except ValueError as exc:raise HTTPException(400,str(exc)) from None

    def get_session(sid):
        session=store.sessions.get(sid)
        if session is None:raise HTTPException(404,'会话不存在或已过期')
        return session

    @app.get('/api/sessions/{sid}')
    def state(sid:str):return get_session(sid).snapshot()

    @app.get('/api/sessions/{sid}/export')
    def export(sid:str):return get_session(sid).export()

    @app.get('/api/sessions/{sid}/sources/{source_id:path}')
    def source(sid:str,source_id:str):
        get_session(sid)
        from agent.evidence_answer import grouped_evidence, source_registry
        from demo.source_compat import install_source_compat
        install_source_compat(ROOT)
        if store.evidence is None:store.evidence=grouped_evidence()
        item=source_registry(store.evidence).get(source_id)
        if item is None:raise HTTPException(404,'来源不在证据索引中')
        return {'id':item.id,'text':item.text,'source':item.source,'path':item.path,
                'sha256':item.sha256,'location':item.location}

    @app.delete('/api/sessions/{sid}')
    async def delete(sid:str):
        session=get_session(sid)
        await session.close();store.sessions.pop(sid,None)
        return {'deleted':True}

    async def accept(websocket):
        from urllib.parse import urlsplit
        origin=websocket.headers.get('origin')
        if origin and urlsplit(origin).netloc!=websocket.headers.get('host'):
            await websocket.close(code=1008);return False
        await websocket.accept();return True

    @app.websocket('/api/sessions/{sid}/ws')
    async def conversation(websocket:WebSocket,sid:str):
        session=store.sessions.get(sid)
        if session is None:
            await websocket.close(code=1008);return
        if session.connected:
            await websocket.close(code=1008);return
        if not await accept(websocket):return
        session.connected=True
        send_lock=asyncio.Lock()
        async def send(value):
            async with send_lock:await websocket.send_json(value)
        await send({'type':'state','state':session.snapshot()})
        try:
            while True:
                data=await websocket.receive_json()
                if not isinstance(data,dict):
                    await send({'type':'error','message':'请求格式无效。'});continue
                kind=data.get('type')
                if kind in ('turn','start'):
                    opening=kind=='start'
                    text='' if opening else data.get('text')
                    if not opening and (not isinstance(text,str) or not text.strip() or len(text)>2000):
                        await send({'type':'error','message':'请输入1至2000字的问题。'});continue
                    if session.task is not None and not session.task.done():
                        await send({'type':'error','message':'请先停止当前回答。'});continue
                    if session.active_text and not session.cancelled.is_set() and any(not s.received for s in session.speech):
                        session.stop()
                    async def work(text=text,voice=bool(data.get('cloud_voice')),opening=opening):
                        try:await session.run(text,send,voice,opening=opening)
                        except ValueError as exc:await send({'type':'error','message':str(exc)})
                    session.task=asyncio.create_task(work())
                elif kind=='receipt':
                    accepted=session.receipt(data.get('turn'),data.get('index'))
                    await send({'type':'receipt','accepted':accepted,'state':session.snapshot()})
                elif kind=='browser_metrics':
                    metrics=data.get('metrics')
                    if isinstance(metrics,dict):session.browser_metric(data.get('turn'),metrics)
                    await send({'type':'state','state':session.snapshot()})
                elif kind=='voice_event':
                    metrics=data.get('metrics')
                    session.voice_event(data.get('turn'),data.get('kind'),data.get('text',''),metrics if isinstance(metrics,dict) else {})
                elif kind=='stop':
                    session.stop()
                    if session.task is not None and not session.task.done():
                        session.task.cancel();await asyncio.gather(session.task,return_exceptions=True)
                    await send({'type':'stopped','state':session.snapshot()})
                elif kind=='seen':
                    # Text-only reading is explicit and never described as acoustic playback.
                    if session._generation_committed:
                        for i in range(len(session.speech)):session.receipt(session.turn_id,i)
                    await send({'type':'state','state':session.snapshot()})
                else:await send({'type':'error','message':'不支持的会话操作。'})
        except (WebSocketDisconnect,RuntimeError):
            pass
        finally:
            await session.close();session.connected=False

    @app.websocket('/api/asr')
    async def asr(websocket:WebSocket):
        if not await accept(websocket):return
        if not capabilities()['asr']:
            await websocket.send_json({'type':'error','message':'未配置语音识别，可继续文字演示。'});await websocket.close();return
        import websockets
        total=0
        try:
            async with websockets.connect(cloud.tencent_asr_url(),proxy=None,max_size=1_000_000,open_timeout=10,close_timeout=2) as upstream:
                hello=json.loads(await upstream.recv())
                if hello.get('code')!=0:raise RuntimeError('ASR handshake failed')
                await websocket.send_json({'type':'ready'})
                async def sender():
                    nonlocal total
                    while True:
                        msg=await websocket.receive()
                        if msg['type']=='websocket.disconnect':raise WebSocketDisconnect()
                        pcm=msg.get('bytes')
                        if pcm is not None:
                            total+=len(pcm)
                            if len(pcm)%2 or len(pcm)>64000 or total>16000*2*60:raise ValueError('audio limit')
                            await upstream.send(pcm)
                        elif msg.get('text'):
                            if json.loads(msg['text']).get('type')=='end':
                                await upstream.send(b'\0\0'*24000)
                                await upstream.send(json.dumps({'type':'end'}));return
                async def receiver():
                    async for raw in upstream:
                        item=json.loads(raw)
                        if item.get('code')!=0:raise RuntimeError('ASR service failed')
                        result=item.get('result') or {}
                        if result.get('voice_text_str'):
                            await websocket.send_json({'type':'final' if result.get('slice_type')==2 else 'partial',
                                                       'text':result['voice_text_str'],'segment':result.get('index',0)})
                        if item.get('final')==1:
                            await websocket.send_json({'type':'done'});return
                await asyncio.wait_for(asyncio.gather(sender(),receiver()),75)
        except WebSocketDisconnect:pass
        except Exception:
            try:await websocket.send_json({'type':'error','message':'语音识别未完成，请检查服务配置或改用文字。'})
            except Exception:pass
        finally:
            try:await websocket.close()
            except Exception:pass

    app.mount('/assets',StaticFiles(directory=ROOT/'web'),name='assets')
    return app


app=create_app(prewarm=True)
