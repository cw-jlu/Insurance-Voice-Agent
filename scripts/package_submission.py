"""Build the reproducibility bundle from selected tracked source and result files."""
import argparse
import hashlib
import html
import json
import subprocess
import zipfile
from pathlib import Path
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parents[1]
TAG='v0.2.0-defense-20261014'
PREFIX='ChenWei-InsuranceVoiceAgent-20261014'

README='''# 陈为｜保险复杂咨询语音 Agent

个人参赛；命题OPEN-JS-01。演示版本v0.2.0-defense-20261014，核心v0.1.0-core-20261014。

先看 docs/31-陈为个人参赛答辩与提交说明.md：3分钟总结、7分钟现场演示、5分钟答辩准备。
正式指标见 docs/24-当前指标与验收口径.md；新包装不改变未达标项。
materials/ 保留截图和真实云连通结果；Qwen对照和合成麦克风测试明确标注，非DeepSeek正式复验。

## 复现步骤

1. 解压整个包，不要只复制网页。安装Python（推荐3.12或更高版本）。
2. 在解压目录执行 python scripts/verify_package.py 和 python scripts/verify_freeze.py。
3. 执行 python -m pip install -r requirements-demo.txt（首次安装需要网络）。
4. 执行 python -m unittest discover -s tests -q，预期142项通过。
5. 执行 python scripts/run_demo.py，打开 http://127.0.0.1:8080。
6. 无密钥时可顺序播放7轮已保存实验。T06失败和T07人工通知按原记录保留。
7. 要现场问新问题，复制.env.demo.example为.env.demo并填写DeepSeek、Qwen、腾讯ASR配置。
   执行 python scripts/check_demo.py 检查配置，然后重启服务。默认DeepSeek，Qwen只是主动选择的对照。
   Windows若Python TLS异常，可使用 python scripts/run_demo.py --native-https。
8. 语音需localhost或HTTPS、麦克风授权。每轮点击开始说话；默认在ASR判定句段结束后自动发送。
   可关自动发送，先核对文字再提交。播放速度0.75–2倍；网页查看耗时、来源、已播与生成记录并导出。

## 复核实测数据

experiments/results/exp44_task_release/ 包含143题原始结果和manual_audit.json。
experiments/results/exp44_current_metrics.json 是当前汇总。
experiments/results/exp44_voice_current/voice/ 是7轮真实云服务语音实验的原始逐轮JSON。
其余历史文字结果与实验脚本保留，可按docs链接追溯；大体积录音和音频缓存不放入本包。
执行历史实验脚本可能产生付费云调用，请先读脚本说明；回放和单元测试无需调用模型。

## 当前边界

复杂题26/30=86.7%，未达90%；复杂整轮播完仍超10秒。附和/回声误停、来源支持和人工漏转未清零。
短信、真实人工接通、理赔进度接口未集成。浏览器回执不是DAC/声学验收，录音按钮为显式打断。
当前机器尚缺DeepSeek密钥，答辩版DeepSeek完整现场链路待补配置预演；不能用Qwen成绩替代。
Docker配置提供但未实际构建；直接依赖固定，不保证传递依赖或外部模型版本永久不变。
外部服务的实时新结果可能与已保存实验不同。关闭服务会丢失内存会话，结束前导出JSON。
资料包不包含真实服务密钥；运行环境配置由接收者自行提供。
'''


def selected_files():
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
    names+=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode('utf-8').split('\0')
    roots={'src','data','web','docs','tests','scripts','releases'}
    singles={'README.md','requirements-core.lock','requirements-demo.txt','requirements.txt',
             '.env.demo.example','start-demo.ps1','Dockerfile','compose.yaml','PRODUCT.md','DESIGN.md'}
    for name in sorted(set(names)):
        if not name:continue
        path=ROOT/name;parts=Path(name).parts
        if not path.is_file():continue
        if '__pycache__' in parts:continue
        allowed=parts[0] in roots or name in singles
        if parts[0]=='experiments':
            allowed=path.suffix.lower() in ('.py','.json','.md','.txt','.csv') and not any(p in parts for p in ('cache','recordings'))
        if allowed:yield name,path.read_bytes()


def demonstration_report():
    evidence_path=ROOT/'demo-runtime/defense-voice-check.json'
    if not evidence_path.exists():return None
    evidence=json.loads(evidence_path.read_text(encoding='utf-8'))
    turn=evidence['export']['turns'][0]
    metrics=turn['browser_metrics'];trace=turn['trace']
    labels={'asr_connect_s':'ASR连接','recording_s':'录音采集（含终点等待）','asr_finalize_s':'结束到识别确认',
            'first_sound_s':'首声（含承接）','first_business_sound_s':'业务正文首声','playback_end_s':'整轮播完'}
    rows=''.join('<tr><td>'+label+'</td><td>'+format(metrics[key],'.2f')+'秒</td></tr>'
                 for key,label in labels.items() if key in metrics)
    stages=''.join('<tr><td>'+html.escape(s['name']+(' · '+s['model'] if s.get('model') else ''))+
                   '</td><td>'+format(s.get('duration_s',0),'.2f')+'秒</td></tr>' for s in turn['stages'])
    sources=''.join('<details><summary>'+html.escape(c['id'])+'</summary><pre>'+html.escape(c.get('quote',''))+
                    '</pre><pre>'+html.escape(json.dumps(c.get('location',{}),ensure_ascii=False,indent=2))+'</pre></details>'
                    for c in trace.get('citations',[]))
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
      '<title>陈为｜现场结果演示材料</title><style>body{max-width:920px;margin:48px auto;padding:0 24px;font:16px/1.8 "Microsoft YaHei",system-ui;color:#292426}'
      'h1{font-size:30px}h2{margin-top:36px;font-size:21px}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}'
      'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.7 inherit}a{color:#b60043}details{margin:14px 0}small{color:#60595c}</style>'
      '<h1>保险复杂咨询语音 Agent</h1><p>陈为 · 个人参赛 · OPEN-JS-01 · 2026年10月14日</p>'
      '<p>答辩演示版 v0.2.0-defense-20261014。现场主模型选择DeepSeek，当前机器密钥待补；本材料展示已实测的Qwen对照连通结果，输入为合成麦克风。没有真人AEC或DeepSeek现场全链路达标声明。</p>'
      '<h2>语音新问题到完整回答</h2><p><strong>识别问题：</strong>'+html.escape(turn['question'])+'</p><p><strong>生成答复：</strong>'+html.escape(turn['response'])+
      '</p><p><strong>已播文本：</strong>'+html.escape(turn['heard'])+'</p><p>自动结束录音并提交；速度'+str(metrics['rate'])+'倍；逐句播放回执完成。页面脚本错误0，桌面与移动均无横向溢出。</p>'
      '<h2>本轮耗时</h2><table>'+rows+'</table><p>服务器生成与校验'+format(trace['text_ready_s'],'.2f')+'秒，全部音频就绪'+format(trace['audio_ready_s'],'.2f')+
      '秒。首声从已识别文本提交起算，不是用户声学句尾；浏览器播放事件不等于扬声器/DAC。阶段并行且嵌套，不能相加。</p><details><summary>各阶段与实际模型调用</summary><table>'+stages+'</table></details>'
      '<h2>答句来源</h2>'+sources+
      '<h2>正式指标与未完成项</h2><table><tr><th>验收项</th><th>exp44完整批次</th></tr>'
      '<tr><td>固定承接首声</td><td>0.68–0.83秒，7/7</td></tr><tr><td>复杂题正确处理</td><td>26/30＝86.7%，未达90%</td></tr>'
      '<tr><td>复杂完整回答</td><td>T02整轮45.59秒，未达10秒</td></tr><tr><td>来源语义支持</td><td>132/138，仍6条缺口</td></tr>'
      '<tr><td>安全人工路由</td><td>必转12/13；记录请求未实际接通</td></tr><tr><td>自然打断</td><td>停播20/20；附和/回声误停与真人AEC未完成</td></tr></table>'
      '<p>整体133/143＝93.0%不能替代复杂题分母。142项程序测试和冻结哈希通过，不能替代业务验收。</p>'
      '<h2>演示与复现入口</h2><p><a href="../docs/31-陈为个人参赛答辩与提交说明.md">3分钟讲稿、7分钟演示、5分钟答辩准备</a> · '
      '<a href="../SUBMISSION_README.md">复现说明</a> · <a href="synthetic-voice-qwen.json">语音实测完整JSON</a> · '
      '<a href="browser-live-quote-qwen.json">423元报价实测JSON</a></p><p>安装依赖后运行 python scripts/run_demo.py，打开 '
      '<a href="http://127.0.0.1:8080/">本地演示台</a>。配置DeepSeek后可提出新问题；无需密钥可回放exp44。</p></html>').encode('utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='demo-runtime/submission')
    args=parser.parse_args();out=(ROOT/args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    payload=dict(selected_files())
    payload['SUBMISSION_README.md']=README.encode('utf-8')
    optional={'defense-live.png':'screenshots/live.png','defense-mobile-final.png':'screenshots/mobile.png',
              'defense-browser-live.json':'browser-live-quote-qwen.json','defense-voice-check.json':'synthetic-voice-qwen.json'}
    for source,target in optional.items():
        path=ROOT/'demo-runtime'/source
        if path.exists():payload['materials/'+target]=path.read_bytes()
    report=demonstration_report()
    if report:payload['materials/现场结果演示.html']=report
    secrets=[]
    for env in ('.env','.env.demo','.env.deepseek'):
        for key,value in dotenv_values(ROOT/env).items():
            if value and len(value)>=8 and any(part in key.upper() for part in ('KEY','SECRET','TOKEN','PASSWORD')):
                secrets.append(value.encode('utf-8'))
    for name,data in payload.items():
        if any(value in data for value in secrets):raise RuntimeError('Credential found; bundle blocked: '+name)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest={'tag':TAG,'source_commit':commit,'core_tag':'v0.1.0-core-20261014',
              'files':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(payload.items())}}
    payload['MANIFEST.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    archive=out/(PREFIX+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(PREFIX+'/'+name,date_time=(2026,10,14,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,data)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (out/(archive.name+'.sha256')).write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps({'archive':str(archive),'files':len(payload),'bytes':archive.stat().st_size,
                      'sha256':digest,'known_local_credentials_found':False},ensure_ascii=False))


if __name__=='__main__':main()
