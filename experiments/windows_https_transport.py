"""Test-only curl/Windows HTTPS transport, retaining native certificate checks.

Credentials and JSON travel on stdin, never command-line args or temporary files.
This buffered transport is only for non-streaming text validation on this host.
"""
import json
import subprocess

import httpx


class WindowsHttpsTransport(httpx.BaseTransport):
    def __init__(self,total_timeout=None):
        self.total_timeout=total_timeout

    def handle_request(self, request):
        timeouts = request.extensions.get('timeout',{})
        connect = timeouts.get('connect') or 30
        read = timeouts.get('read') or 30
        write = timeouts.get('write') or 30
        config = [f'url = {json.dumps(str(request.url),ensure_ascii=False)}',
                  f'request = {json.dumps(request.method)}']
        for key,value in request.headers.items():
            config.append(f'header = {json.dumps(key+": "+value,ensure_ascii=False)}')
        body = request.read()
        if body:
            config.append(f'data-binary = {json.dumps(body.decode("utf-8"),ensure_ascii=False)}')
        total=connect+read+write
        if self.total_timeout is not None:total=min(total,self.total_timeout)
        process = subprocess.run(['curl.exe','--silent','--show-error','--include','--config','-',
                                  '--connect-timeout',str(min(connect,total)),'--max-time',str(total)],
                                 input=('\n'.join(config)+'\n').encode('utf-8'),
                                 capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
        if process.returncode:
            # Do not expose curl config or credential-bearing requests in errors.
            raise httpx.ConnectError(f'Windows HTTPS transport failed (curl exit {process.returncode})',request=request)
        raw = process.stdout
        while True:
            header,separator,content = raw.partition(b'\r\n\r\n')
            if not separator:
                raise httpx.ProtocolError('Missing HTTPS response headers',request=request)
            lines = header.split(b'\r\n')
            status = int(lines[0].split()[1])
            if status < 200:
                raw = content
                continue
            headers = [(line.split(b':',1)[0],line.split(b':',1)[1].strip()) for line in lines[1:] if b':' in line]
            # curl already removes chunked-transfer framing, not content encodings.
            headers = [(k,v) for k,v in headers if k.lower() != b'transfer-encoding']
            return httpx.Response(status,headers=headers,content=content,request=request)


def native_client():
    from agent import cloud
    from agent.model_router import AnswerModelRouter
    qwen = cloud.qwen_client().with_options(http_client=httpx.Client(transport=WindowsHttpsTransport()))
    if cloud.ANSWER_PROVIDER != 'deepseek':
        return qwen
    router = AnswerModelRouter(qwen,cloud.SLOW_MODEL)
    router.deepseek = router.deepseek.with_options(http_client=httpx.Client(transport=WindowsHttpsTransport()))
    return router
